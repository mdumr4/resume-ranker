# Resume Ranking System — Production Optimization Plan

**Goal:** hybrid RRF retrieval (100k resumes) → BAAI/bge-reranker-v2-m3 rerank top-100 → Qwen3-1.7B 1-2 sentence rationale per candidate.
**Hard constraints:** ≤5GB total storage · CPU-only · zero internet during ranking · ranking phase ≤5 minutes.
**Current state:** 10+GB storage, 20+ minutes. Models are fixed (not being swapped) — everything below is quantization / runtime / architecture, not model choice.

---

## 0. One assumption, stated up front

"Run everything under 5 min" is interpreted as the **ranking (query-time) phase** — one job description in, ranked top-100 with rationales out. Embedding 100,000 resumes from scratch can't realistically happen in 5 minutes on CPU with a cross-encoder+LLM in the loop, and you already said embeddings are precomputed and reused. So the plan splits into:

- **Phase A — Indexing** (batch, offline, run once + incrementally when resumes are added). Loose time budget.
- **Phase B — Ranking** (per query, the thing that must hit ≤5 min / CPU-only / zero internet).

If you actually meant the full 100k index build must also complete in 5 minutes, that's not achievable with this architecture on CPU and the plan needs to change (e.g., a much smaller/faster indexing encoder) — flag it and I'll revise.

---

## 1. Why you're at 10GB+ / 20+ minutes today

These are the four things almost certainly responsible, in order of impact:

1. **fp32 weights, no quantization.** All four models stored at full precision add up fast:

   | Model                   | Params                               | fp32 size         |
   | ----------------------- | ------------------------------------ | ----------------- |
   | Qwen3-Embedding-0.6B    | 600M (confirmed)                     | ~2.4GB            |
   | naver/splade-v3         | 110M, BERT-base backbone (confirmed) | ~0.44GB           |
   | BAAI/bge-reranker-v2-m3 | 568M (confirmed, 2.27GB on disk)     | ~2.3GB            |
   | Qwen3-1.7B              | 1.7B dense                           | ~6.8GB            |
   | **Total**         |                                      | **~11.9GB** |

   This alone explains your 10GB+ — you're very likely storing every model at full fp32 precision. This is the single biggest lever.
2. **sentence-transformers pulls in full PyTorch.** On top of model weight size, `pip install torch` on a CPU box (without pointing at the CPU-only wheel index) commonly pulls several GB of unused NVIDIA CUDA shared libraries (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, etc.) even though you never touch a GPU. **Check this first** — `pip list | grep nvidia` or check `du -sh` on your venv's `site-packages/nvidia*`. This is a very common and very large silent contributor.
3. **Likely no persistent warm service.** If each ranking run re-imports torch/transformers and reloads ~10GB of fp32 weights from disk, that alone can burn 1-3+ minutes before any actual ranking work starts.
4. **Unbatched, unquantized LLM generation.** Generating 1-2 sentences for 100 candidates, one at a time, in fp32, via `transformers.generate()` on CPU, is almost certainly your biggest time sink — this single step can plausibly account for 10+ of your 20+ minutes.

None of these require changing your model choices. All four are runtime/packaging/engineering problems, which is good news — they're fixable without touching quality.

---

## 2. Target architecture

```
PHASE A — OFFLINE INDEXING (batch, run once + incrementally, loose time budget, internet OK)
  raw resumes ──► clean/parse text ──► dense embed (Qwen3-Embedding-0.6B, INT8/Q8) ──┐
                                    └► sparse embed (SPLADE-v3, INT8 ONNX) ──────────┼──► persisted to disk
                                    └► store raw text + metadata ────────────────────┘

PHASE B — RANKING (per query, ≤5 min, CPU-only, ZERO internet, long-lived warm process)
  job query ──► embed (dense+sparse)
             ──► FAISS dense top-K  ┐
             ──► sparse top-K       ┴──► RRF fuse ──► top-100
             ──► bge-reranker-v2-m3 (INT8 ONNX, batched) ──► re-ordered top-100
             ──► Qwen3-1.7B (GGUF, batched/parallel decode) ──► 1-2 sentence rationale each
             ──► return ranked list + rationales
```

Internet is needed **exactly once**: to download the four model checkpoints and convert/quantize them. After that artifact is built, both Phase A and Phase B run forever with zero network access — this is a packaging decision (see §8), not a per-run dependency.

---

## 3. Model-by-model optimization

### 3.1 Qwen3-Embedding-0.6B — dense retrieval

This is your **recall-critical, first-stage** model — an error here can't be fixed downstream, so keep it conservative on precision, and spend your compression budget on the *output dimension* instead:

- **Keep INT8 / Q8_0** (not Q4) — first-stage retrieval quality is worth the extra ~300-400MB.
- **Use native MRL to shrink the stored vector, not the weights.** This model was explicitly trained with Matryoshka Representation Learning and officially supports truncating output embeddings from 1024 down to as low as 32 dims, with graceful quality degradation. Truncate to **256 or 512 dims** for the *stored index* — same model, same weights, just take a prefix of the output vector. This is not "changing the encoder," it's using a built-in feature of the model you're already committed to. Test recall@100 at 256 vs 512 vs 1024 on a validation set of queries before locking it in.
- **Runtime:** Qwen publishes an official GGUF build (`Qwen/Qwen3-Embedding-0.6B-GGUF`), so you can run this through **llama.cpp** in embedding mode instead of sentence-transformers/PyTorch. This matters because it lets you use the *same* lightweight runtime (llama.cpp) for both this model and Qwen3-1.7B (see §7).
- **Size:** ~0.6GB.

### 3.2 naver/splade-v3 — sparse retrieval

Also first-stage/recall-critical — keep it conservative too.

- BERT-base backbone (110M params), outputs a 30522-dim sparse vector via an MLM head + ReLU + max-pool. This head is non-standard, so **export to ONNX** (via `optimum-cli export onnx`) rather than trying to force it through llama.cpp — ONNX preserves the exact computation graph including the MLM projection.
- Quantize the exported graph to **INT8** with `onnxruntime.quantization.quantize_dynamic`.
- **Size:** ~0.12GB.
- *(Optional, not required — you said the encoders aren't changing: `naver/splade-v3-distilbert` is a DistilBERT variant of the same checkpoint family, ~half the size/compute. Only worth considering if you revisit model choice later.)*

### 3.3 BAAI/bge-reranker-v2-m3 — cross-encoder rerank of top-100

Only ever touches 100 candidates per query, so it's a pure speed problem, not a recall problem — INT8 here is standard, well-established practice, not a risky call.

- 568M params, XLM-RoBERTa/bge-m3 backbone, 2.27GB fp32 (confirmed). Export to ONNX + INT8 dynamic quantization.
- **FastEmbed** (Qdrant's ONNX-based embedding/reranking library) explicitly lists `BAAI/bge-reranker-v2-m3` as a supported cross-encoder out of the box (`fastembed.rerank.cross_encoder.TextCrossEncoder`) — this can save you the manual export step if its packaged ONNX graph meets your needs; otherwise export yourself with `optimum`.
- **Cap `max_length` to 384-512 tokens.** The model architecturally supports up to 8192, but resumes rarely need it, and reranker compute scales roughly linearly with sequence length — this single flag change can be a 4-16x speed difference vs. leaving it at default/max.
- **Batch all 100 (query, resume) pairs into one or a handful of ONNX Runtime calls**, not 100 sequential `.run()` calls. This is the single biggest lever for this stage.
- **Size:** ~0.6GB.

### 3.4 Qwen3-1.7B — rationale generation

The most user-facing, quality-sensitive model (people will read this text), but also the most tolerant of numerical error (a slightly-different-but-still-fluent sentence is fine; a slightly-different ranking score is not) — and your biggest storage/time risk today.

- Dense decoder, official + community GGUF quantizations exist (e.g. `Qwen/Qwen3-1.7B-GGUF`, `bartowski/Qwen_Qwen3-1.7B-GGUF`). Extrapolating from the confirmed Qwen3-4B GGUF sizes (same family):

  | Quant            | Est. size for 1.7B   | Quality                             |
  | ---------------- | -------------------- | ----------------------------------- |
  | Q4_K_M           | ~1.0-1.1GB           | good, mainstream default            |
  | **Q5_K_M** | **~1.2-1.3GB** | **sweet spot — recommended** |
  | Q6_K             | ~1.35-1.45GB         | near-lossless vs fp16               |

  Since your storage budget has headroom (see §4), **Q5_K_M or Q6_K** is affordable and gives a real quality margin on the one output people actually read — no need to default straight to Q4_K_M here. *(These are extrapolated from the 4B model's confirmed GGUF sizes, not measured directly on 1.7B — verify exact file size once you pull the repo.)*
- **Runtime: llama.cpp** (via `llama-cpp-python` or a local `llama-server`), not `transformers.generate()`. This is the change that actually fixes your 20-minute problem — see §6 for why.
- **This is the step to actively engineer for speed**, via prompt design and batching, not just quantization — details in §6.

---

## 4. Storage budget

| Component                                                   | Size                  | Notes                                    |
| ----------------------------------------------------------- | --------------------- | ---------------------------------------- |
| Qwen3-Embedding-0.6B (Q8/INT8)                              | ~0.60GB               | conservative — recall-critical          |
| naver/splade-v3 (INT8 ONNX)                                 | ~0.12GB               | conservative — recall-critical          |
| bge-reranker-v2-m3 (INT8 ONNX)                              | ~0.60GB               | standard practice                        |
| Qwen3-1.7B (Q5_K_M GGUF)                                    | ~1.25GB               | quality margin, storage allows it        |
| **Model subtotal**                                    | **~2.57GB**     | vs. ~11.9GB fp32 today                   |
| onnxruntime + llama-cpp-python + faiss-cpu (installed libs) | ~0.30GB               | vs. multi-GB torch+CUDA-wheel stack      |
| Dense index, 100k × 256-512 dims, fp16                     | 0.05–0.10GB          | via MRL truncation                       |
| Sparse index, 100k docs, ~150-250 nonzero terms each        | 0.08–0.15GB          | CSR / (idx,val) pairs                    |
| Resume text + metadata (SQLite, compressed)                 | 0.10–0.25GB          | needed by reranker + LLM stages          |
| **Data subtotal**                                     | **~0.3–0.5GB** |                                          |
| **Grand total**                                       | **~3.2–3.4GB** | **~1.6–1.8GB headroom under 5GB** |

You have real headroom — enough to run the LLM at Q5_K_M/Q6_K instead of squeezing to Q4, and enough that dense-index PQ compression or further tricks aren't needed at this corpus size (100k). Revisit only if the corpus grows toward the millions.

---

## 5. Retrieval index & storage design

**Dense (FAISS, CPU):**

```python
import faiss
index = faiss.IndexFlatIP(dim)      # exact search; 100k vectors is fast enough that
                                     # approximate (IVF/HNSW) isn't needed yet
index.add(vectors_fp32)             # faiss wants fp32 in memory even if you store fp16 on disk
faiss.write_index(index, "resumes.dense.index")
```

`IndexFlatIP` over 100k × 256-1024 dims is sub-second on CPU — no need for approximate search at this scale. Store vectors as fp16 on disk, upcast to fp32 on load (FAISS's flat index wants fp32 in memory; the disk savings are what matter for your 5GB budget, not RAM).

**Sparse (SPLADE):** store as `scipy.sparse.csr_matrix`, saved with `scipy.sparse.save_npz`. Query time is one sparse matrix–vector product against a 100k×30522 CSR matrix — fast (well under a second) even unoptimized.

**Payload (resume text + metadata):** SQLite, single file, zero server, stdlib `sqlite3`. FAISS only stores vectors + integer IDs — it needs a companion keyed store for the actual text the reranker/LLM stages need, and SQLite is the simplest thing that won't fall over in production (unlike a raw pickle/JSON file you're rewriting on every update).

**RRF fusion** — trivial, keep it as plain Python, no library needed:

```python
def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores = {}
    for ranking in rankings:                 # one list per retriever (dense, sparse)
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores
```

**On Qdrant as an alternative:** Qdrant's embedded "local mode" (`QdrantClient(path=...)`) does support native dense+sparse+RRF in one call, which is appealing — but its own docs recommend it for **under ~20,000 points** and note it uses brute-force search (no HNSW) at that scale, single-process only. At 100k points you'd want a real self-hosted Qdrant **server** (via Docker, still fully offline/local, just not the embedded client mode) to get proper indexing — that's a reasonable production upgrade path later if you want built-in persistence/filtering/hybrid-query ergonomics, but it's an added service to operate. Given your tight storage/time constraints right now, FAISS + scipy + SQLite is the leaner, zero-extra-process choice and is proven at far larger scale than 100k.

---

## 6. Time budget for the 5-minute ranking phase

| Stage                                                          | Est. time                 | Key lever                                                                                                                                                                              |
| -------------------------------------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model load                                                     | **~0s** (amortized) | Long-lived warm process — load all 4 models once at startup, never per-request. This is likely where a large chunk of your current 20+ min is going if you're cold-starting each run. |
| Query embed (dense + sparse)                                   | <1s                       | single short string, negligible                                                                                                                                                        |
| FAISS dense search, 100k                                       | <0.5s                     | flat index, no tuning needed at this scale                                                                                                                                             |
| Sparse search, 100k                                            | <1s                       | scipy CSR matvec                                                                                                                                                                       |
| RRF fuse                                                       | <0.1s                     | pure Python dict ops                                                                                                                                                                   |
| Rerank top-100 (INT8 ONNX, batched, capped 384-512 tokens)     | ~15-45s                   | batch all 100 pairs together; cap max_length                                                                                                                                           |
| **Generate 100 rationales (Qwen3-1.7B GGUF, llama.cpp)** | **~45-150s**        | **dominant, most variable cost — see below**                                                                                                                                    |
| Orchestration/IO                                               | ~5-10s                    |                                                                                                                                                                                        |
| **Total (typical)**                                      | **~1-3.5 min**      | comfortable margin under 5 min                                                                                                                                                         |

**The generation step is the one to actively engineer, not just quantize:**

1. **Batch/parallelize across candidates.** Don't loop 100 sequential `generate()` calls. Run `llama-server` with `--parallel N` (continuous batching across N concurrent decode slots) or use `llama-cpp-python`'s batch API. CPU parallel decoding doesn't scale as cleanly as GPU, but even a conservative 2-4x aggregate speedup on a multi-core CPU turns a 2-4 minute sequential step into 30-90 seconds.
2. **Hard-cap `max_new_tokens`** to ~40-60 — you asked for 1-2 sentences, so enforce that at the token-budget level, not just the prompt level.
3. **Keep the prompt short and templated.** A fixed system-prompt prefix (job criteria + instructions) that's identical across all 100 candidates, with only the per-candidate resume summary varying, lets you reuse KV-cache/prompt-prefix caching in llama.cpp rather than reprocessing the full prompt 100 times.
4. **Add a timeout-based fallback.** Per-candidate generation time on CPU has real variance. Set a per-candidate timeout (e.g. 3-4s); if exceeded, fall back to a simple extractive template ("Matches on: X, Y, Z") instead of blocking the whole batch. This guarantees the 5-minute SLA can never be blown by one slow generation, at the cost of an occasional lower-quality rationale.
5. **Benchmark on your actual target CPU** with `llama-bench` before trusting any of the numbers above — CPU LLM inference is memory-bandwidth-bound, so core count *and* RAM channel/speed both matter, and vary a lot machine to machine.

---

## 7. Runtime/library replacement summary

| Today                                                                 | Replace with                                                                                                                                                                          | Why                                                                                                                                                                                                     |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sentence-transformers` (full PyTorch under the hood, all 4 models) | **ONNX Runtime** for SPLADE + reranker; **llama.cpp** (GGUF) for embedding model + LLM                                                                                    | Two lightweight C++ inference engines instead of one heavy Python+PyTorch stack. onnxruntime's CPU wheel is tens of MB; llama-cpp-python is similarly compact. No CUDA wheels get pulled in for either. |
| `transformers.generate()` for Qwen3-1.7B                            | `llama.cpp` / `llama-cpp-python` (GGUF)                                                                                                                                           | Purpose-built for quantized CPU generation: optimized kernels, proper KV-cache handling, batching support. This is the fix for your 20-minute problem, not just a nice-to-have.                         |
| `pip install torch` (default)                                       | Either eliminate torch entirely (recommended, given the above), or if you must keep it for any reason,`pip install torch --index-url https://download.pytorch.org/whl/cpu`          | Default installs commonly pull GBs of unused CUDA shared libraries on a CPU-only box.                                                                                                                   |
| Reranker/embedding inference                                          | `optimum` (HF→ONNX export + `onnxruntime.quantization.quantize_dynamic` for INT8) or `fastembed` (pre-packaged ONNX models, confirmed support for `BAAI/bge-reranker-v2-m3`) | Standard, well-established path to INT8 CPU inference for encoder-style transformer models.                                                                                                             |
| Vector search                                                         | `faiss-cpu`                                                                                                                                                                         | Purpose-built, proven from thousands to billions of vectors, small dependency footprint.                                                                                                                |

---

## 8. Deployment architecture

**Directory layout:**

```
resume-ranker/
├── models/
│   ├── qwen3-embedding-0.6b-q8.gguf
│   ├── splade-v3-int8.onnx
│   ├── bge-reranker-v2-m3-int8.onnx
│   └── qwen3-1.7b-q5_k_m.gguf
├── index/
│   ├── resumes.dense.index      # FAISS
│   ├── resumes.sparse.npz       # scipy CSR
│   └── resumes.db               # SQLite: id -> text, metadata
├── service/
│   └── rank_server.py           # long-lived process, loads all 4 models once
└── indexing/
    └── build_index.py           # Phase A batch job
```

**Enforce offline, in layers (don't rely on one mechanism):**

- Environment variables as the first layer: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` — needed even in an ONNX/GGUF setup if any tokenizer loading still goes through `transformers.AutoTokenizer`, since tokenizer files can trigger a Hub call if not fully cached.
- **Network-level enforcement as the real guarantee:** run the Phase B ranking service in a container with `docker run --network none` (or equivalent firewall rule), so it's physically incapable of reaching the internet regardless of what any library tries to do. Env vars are a courtesy; network isolation is the actual control.

**Build with internet, ship without it — two-stage Docker build:**

```dockerfile
# ---- build stage: has internet, downloads + converts + quantizes models ----
FROM python:3.11-slim AS build
RUN pip install --break-system-packages optimum[onnxruntime] huggingface_hub llama-cpp-python
# ... download, export to ONNX, quantize, produce GGUF files into /artifacts ...

# ---- runtime stage: zero internet needed, ever ----
FROM python:3.11-slim
COPY --from=build /artifacts /app/models
COPY service/ /app/service
RUN pip install --break-system-packages onnxruntime llama-cpp-python faiss-cpu
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CMD ["python", "/app/service/rank_server.py"]
# run this image with --network none in production
```

**Long-lived warm service, not a per-query script.** Load all four models once at process startup (this is where you pay the ~2.5GB-of-weights loading cost, one time), then serve ranking requests over a local API (FastAPI/similar) or as an in-process function if called from the same app. This single change removes model-loading time from the per-query 5-minute budget entirely.

**Incremental indexing.** New resumes shouldn't trigger a full 100k rebuild:

```python
# FAISS supports incremental adds directly — no rebuild needed
index.add(new_vectors)
faiss.write_index(index, "resumes.dense.index")
# append new rows to the sparse CSR matrix and the SQLite table similarly
```

Periodically re-save the index to disk after a batch of additions (not after every single one, to avoid excessive I/O).

**Concurrency.** This pipeline is CPU-bound end to end. If multiple ranking queries can arrive concurrently (e.g. several recruiters searching at once), don't let them run fully in parallel by default — CPU-bound work beyond your core count just causes contention and blows the 5-minute SLA for everyone. Use a small request queue (concurrency limit tied to physical core count) rather than unbounded parallelism.

---

## 9. Quality assurance before you ship the quantized versions

Quantization is generally safe at the levels recommended here, but "generally" isn't "verified" — run these once before trusting it in production, and again after any model/runtime upgrade:

- **Retrieval recall check:** on a held-out set of queries, compare top-100 candidate-ID overlap between the fp32 baseline and the INT8/MRL-truncated pipeline. Expect >95% overlap; investigate if lower.
- **Reranker score correlation:** Spearman correlation between fp32 and INT8 reranker scores on a sample of (query, resume) pairs — should be very high (>0.98 is a reasonable bar).
- **LLM rationale spot-check:** manually read 20-30 generated rationales at your chosen GGUF quant level (Q5_K_M/Q6_K) for coherence and factual grounding against the actual resume — this is the one place quantization quality is subjective, not just numeric, so a human read is warranted before shipping.

---

## 10. Migration checklist

1. Check `pip list | grep nvidia` / venv size — confirm or rule out the CUDA-wheel storage leak (§1.2).
2. Export SPLADE-v3 and bge-reranker-v2-m3 to ONNX via `optimum-cli export onnx`; quantize both to INT8 with `onnxruntime.quantization.quantize_dynamic`.
3. Pull `Qwen/Qwen3-Embedding-0.6B-GGUF` (Q8_0) and a Qwen3-1.7B GGUF (Q5_K_M) — verify exact file sizes against the storage budget in §4.
4. Rewrite the embedding pipeline to call llama.cpp (embedding mode) instead of sentence-transformers; wire up MRL truncation to 256/512 dims.
5. Rebuild the dense (FAISS) and sparse (scipy CSR) indices from the new pipeline; move resume text/metadata into SQLite.
6. Rewrite the reranking step to batch all top-100 pairs into onnxruntime calls, with `max_length` capped.
7. Rewrite the rationale-generation step on llama.cpp/GGUF with batched/parallel decode, short templated prompts, capped `max_new_tokens`, and a per-candidate timeout fallback.
8. Wrap everything in a long-lived warm service (no per-query model loading).
9. Run the QA checks in §9 against the fp32 baseline.
10. Build the two-stage Docker image; verify the runtime stage functions correctly with `--network none`.
11. Benchmark end-to-end ranking latency on the actual target CPU with `llama-bench`/timed runs; tune batch sizes and quant levels against the measured numbers, not the estimates in this document.
12. Load-test with the realistic concurrency you expect in production; set the request queue limit accordingly.

---

## 11. Risks & mitigations

| Risk                                                                                       | Mitigation                                                                                                                                          |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Quantization measurably hurts ranking quality                                              | QA harness in §9, run before every deploy; conservative (INT8, not lower) on the two retrieval-stage models specifically                           |
| LLM generation time is more variable than estimated on your specific CPU                   | Per-candidate timeout + extractive-template fallback (§6); benchmark early with`llama-bench` rather than trusting this doc's estimates           |
| Cold-start / model-load time eats the 5-minute budget                                      | Persistent warm service is non-negotiable — this is architectural, not a tuning knob                                                               |
| Corpus grows well past 100k over time                                                      | Current design (FAISS flat + scipy CSR) has headroom; if it later reaches the low millions, revisit FAISS IVF/PQ and/or a self-hosted Qdrant server |
| A library silently reaches out to the internet (tokenizer fetch, hub metadata check, etc.) | Network-level isolation (`--network none`) as the real guarantee, not just `HF_HUB_OFFLINE` env vars                                            |
