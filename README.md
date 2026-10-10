# Redrob Candidate Discovery & Resume Ranker

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FAISS](https://img.shields.io/badge/FAISS-IndexFlatIP-orange.svg)](https://github.com/facebookresearch/faiss)
[![SPLADE](https://img.shields.io/badge/SPLADE-v3--sparse-green.svg)](https://github.com/naver/splade)
[![CrossEncoder](https://img.shields.io/badge/CrossEncoder-MiniLM--L12-purple.svg)](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2)
[![HuggingFace Bucket](https://img.shields.io/badge/HF%20Bucket-mdumr4%2FRedrob-yellow.svg)](https://huggingface.co/buckets/mdumr4/Redrob)

A production-grade, CPU-bound candidate discovery and reranking pipeline developed for the **Redrob Talent Intelligence Hackathon** (India Runs Data and AI Challenge).

The system filters and ranks candidate profiles from an initial pool of **100,000 resumes**, dynamically computes candidate platform trust and career gap penalties, and generates natural language match rationales. The entire end-to-end workflow executes fully locally on commodity CPU hardware in **under 2.5 minutes**, strictly respecting the 5-minute offline competition budget.

---

## Architecture Flow

```mermaid
graph TD
    %% Offline Asset Preparation
    subgraph Offline ["Cloud Storage & Pre-Computation (HF Bucket)"]
        Raw["candidates.jsonl (100K profiles)"] --> Prep["CandidatePreprocessor"]
        Prep --> Feat["features_v3.parquet (Trust Scores & Text)"]
        Prep --> DenseIdx["FAISS Dense Index (Qwen Embeddings)"]
        Prep --> SparseIdx["SPLADE Sparse Matrix (CSR)"]
        JD["Job Description (JD)"] --> JDEmb["Precomputed JD Vectors (Dense & Sparse)"]
    end

    %% Online Real-Time Pipeline
    subgraph Online ["Online Inference Pipeline (Strict 5-Minute CPU Budget)"]
        JDEmb --> S1["Stage 1: Vectorized Hybrid Retrieval"]
        DenseIdx --> S1
        SparseIdx --> S1
        Feat --> S1
        S1 -->|FAISS + SPLADE + RRF (k=60) * Trust| TopK["Top 100 Candidates (< 2s)"]

        TopK --> S2["Stage 2: Two-Part Cross-Encoder Reranking"]
        Feat --> S2
        S2 -->|70% Tech Match + 30% Logistics Match| Reranked["Precision Reranked Top 100 (~14s)"]

        Reranked --> S3["Stage 3: High-Throughput LLM Reasoning"]
        S3 -->|llama-server Continuous Batching (Qwen2.5-0.5B)| Rationales["Candidate Match Rationales (~110s)"]

        Rationales --> S4["Stage 4: Compliance Validation & Assembly"]
        S4 --> Out["submission.csv (100 Rows, Monotonic Scores)"]
    end
```

---

## Multi-Stage Technical Methodology

### 1. Stage 0: Zero-RAM Static JD Representation
Standard inference pipelines load both dense bi-encoders (Qwen-1.5B) and sparse lexical models into RAM at runtime, consuming $\sim 1.5	ext{ GB}$ of memory and adding $13	ext{s}$ of startup latency.
* Because the target Job Description is fixed, its dense query vector (`jd_dense_query.npy`, $4.2	ext{ KB}$) and sparse SPLADE query representation (`jd_sparse_query.npy`, $244	ext{ KB}$) are precomputed.
* **Result**: Stage 0 asset loading drops from $13.2	ext{s}$ to **$0.2	ext{s}$**, eliminating runtime embedding overhead.

### 2. Stage 1: Vectorized Hybrid Retrieval & Dynamic Trust Engine
* **Dense Retrieval (FAISS)**: Performs approximate nearest neighbor search via `faiss.IndexFlatIP` across 100,000 vectors, retrieving the top 10,000 candidates and shifting cosine scores to $[0, 1]$.
* **Sparse Retrieval (SPLADE-v3)**: Evaluates exact keyword intersections via sparse CSR dot-products partitioned to the top 10,000 candidates.
* **Dynamic Trust & Gap Penalty Engine**:
  * Email/phone verification gate (unverified profiles capped at $0.10$).
  * Multi-factor scoring: profile completeness ($20\%$), open-to-work flag ($20\%$), recruiter response rate ($10\%$), search appearance ($10\%$), interview completion ($10\%$), and GitHub activity ($5\%$).
  * **Merged Timeline Gap Detection**: Merges contiguous/overlapping career and education intervals, penalizing cumulative multi-year gaps (up to $-0.45$).
  * Inactivity penalty: Penalizes accounts dormant $> 200$ days.
* **Reciprocal Rank Fusion (RRF)**:
  $$	ext{RRF}(c) = \left( rac{1}{60 + 	ext{rank}_{	ext{dense}}} + rac{1}{60 + 	ext{rank}_{	ext{sparse}}} ight) 	imes 	ext{trust\_score}(c)$$
  Extracts the unified top 100 candidates in **$1.11	ext{s}$**.

### 3. Stage 2: Two-Part Cross-Encoder Semantic Reranking
Standard single-pass rerankers lose resolution when concatenating lengthy job descriptions with complex resumes. We decouple evaluation into two targeted cross-encoder inferences using `ms-marco-MiniLM-L-12-v2`:
* **Part A (Tech Match - 70% weight)**: Cross-references technical requirements (ranking, retrieval, vector DBs, Python, evaluation metrics) against candidate career history and verified skills.
* **Part B (Logistics Match - 30% weight)**: Cross-references location (Pune/Noida), notice period ($\le 30$ days), and work mode preferences against candidate logistics.
* **Sigmoid Fusion**: Normalizes raw logits via Sigmoid ($\sigma$) and applies additive weighting:
  $$	ext{Final Score} = 0.70 \cdot \sigma(	ext{logit}_{	ext{tech}}) + 0.30 \cdot \sigma(	ext{logit}_{	ext{logistics}})$$
  Completes top 100 evaluation in **$13.8	ext{s}$** on CPU.

### 4. Stage 3: Continuous Batching LLM Reasoning
* Bypasses the Python Global Interpreter Lock (GIL) by launching a local C++ server (`llama-server.exe`) running `Qwen2.5-0.5B-Instruct` quantized to 4-bit (`Q4_K_M` GGUF).
* Dispatches 32 concurrent HTTP requests utilizing continuous batching and shared KV-cache prompt reuse.
* Enforces strict two-sentence format: (1) Reason for rank based on JD fit, (2) Targeted area for improvement.
* Includes deterministic fallback generation if requests exceed timeout thresholds.

---

## Benchmark Performance Profile (CPU-Only)

Executed on an 8-Core Intel CPU / 16GB RAM:

| Pipeline Stage | Operation | Compute Overhead | Latency |
|---|---|---|---|
| **Stage 0** | Load Precomputed Indices & Weights | Memory I/O | $1.8	ext{s}$ |
| **Stage 1** | FAISS Dense Search ($10	ext{k}$) + SPLADE Sparse ($10	ext{k}$) + RRF | Vectorized Math | $1.1	ext{s}$ |
| **Stage 2** | Two-Part Cross-Encoder Reranking ($100$ candidates) | PyTorch Batched Math | $13.8	ext{s}$ |
| **Stage 3** | Continuous Batching LLM Reasoning ($100$ rationales) | C++ llama-server | $\sim 110	ext{s}$ |
| **Stage 4** | Monotonicity Check & CSV Assembly | Disk I/O | $0.1	ext{s}$ |
| **Total** | **End-to-End Execution** | **Strictly CPU** | **$pprox 127	ext{s}$ ($2.1	ext{ min}$)** |

*(Note: Fast-mode execution with `--skip-llm` completes Stage 0-2 in **$16.9	ext{ seconds}$**).*

---

## Quickstart & Reproducibility

### 1. Installation
Clone the repository and install dependencies with [uv](https://github.com/astral-sh/uv) or pip:

```bash
git clone https://github.com/mdumr4/resume-ranker.git
cd resume-ranker
uv sync
```

### 2. Download Precomputed Cloud Assets
All golden assets ($2.95	ext{ GB}$) are hosted on the Hugging Face Bucket `hf://buckets/mdumr4/Redrob`. Synchronize them with one command:

```bash
# Authenticate with Hugging Face (first time only)
hf auth login

# Download indices, models, and candidate dataset
python sync_assets.py --all
```

### 3. Run Pipeline Inference
Execute the official reproduction command from the root of the repository:

```bash
# Standard reproduction with LLM reasoning
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

# Fast mode (retrieval + two-part reranking + deterministic rationales in ~17s)
python rank.py --skip-llm --out ./submission.csv
```

### 4. Run Diagnostic Test Suite
Run the automated verification suite covering trust calculation, date interval merging, RRF math, two-part cross-encoder weighting, and submission compliance:

```bash
uv run pytest tests/ -v
```

---

## Repository Structure

```
resume-ranker/
├── context/                           # Domain specifications & sample data
│   ├── job_description.docx           # Official target JD
│   ├── parsed_jd.json                 # Structured JD criteria & weights
│   ├── redrob_signals_doc.docx        # Behavioral signals spec
│   ├── submission_spec.docx           # Challenge rules & validation criteria
│   └── Sample/
│       ├── sample_candidates.json     # 50 sample candidates fixture
│       └── sample_submission.csv
├── data/                              # Local data directory (git-ignored)
│   └── candidates.jsonl               # Full 100K candidate dataset (487 MB)
├── index/                             # Vector indices (git-ignored)
│   ├── faiss_qwen_index.bin           # 409 MB dense FAISS index
│   ├── splade_combined_v3.npz         # 180 MB sparse CSR matrix
│   ├── features_v3.parquet            # 44 MB precomputed text & trust scores
│   ├── jd_dense_query.npy             # Precomputed dense JD query
│   └── jd_sparse_query.npy            # Precomputed sparse JD query
├── models/                            # Model weights (git-ignored)
│   ├── ms-marco-MiniLM-L-12-v2/       # Cross-Encoder model directory
│   ├── splade_onnx_int8/              # INT8 quantized SPLADE model
│   └── qwen2.5-0.5b-instruct-q4_k_m.gguf # Quantized reasoning LLM
├── bin/                               # Local C++ binaries (git-ignored)
│   └── llama/                         # llama-server executable and DLLs
├── src/                               # Modular Object-Oriented Core
│   ├── __init__.py
│   ├── config.py                      # Strongly-typed Dataclass configuration
│   ├── preprocessor.py                # Text serializer & Dynamic Trust Engine
│   ├── retriever.py                   # FAISS + SPLADE + RRF hybrid engine
│   ├── reranker.py                    # Two-Part Cross-Encoder reranker
│   ├── reasoner.py                    # Continuous batching LLM client
│   ├── validator.py                   # Monotonic compliance & schema checker
│   └── pipeline.py                    # Master CandidateRankingPipeline
├── tests/                             # Diagnostic test suite
│   ├── test_preprocessor.py           # Unit tests for Trust Score & timeline merging
│   ├── test_retriever_and_reranker.py # Unit tests for RRF & sigmoid weighting
│   ├── test_validator.py              # Unit tests for submission rules
│   └── test_pipeline_integration.py   # End-to-end integration test
├── rank.py                            # Production CLI entry point
├── sync_assets.py                     # One-click cloud bucket downloader
├── submission_metadata.yaml           # Official team & submission manifest
└── pyproject.toml                     # Modern package dependency specification
```

---

## Submission & Team Metadata

* **Competition**: Redrob Talent Intelligence Hackathon (India Runs Data and AI Challenge)
* **Team Name**: Aura Farmers
* **Team Members**:
  * Mohammed Umar F (`mdumr4@gmail.com`) — Team Lead, AI & Backend Engineer
  * Soujanya K Hegde (`shegde092@gmail.com`) — System Engineer, Backend Engineer
* **Official Repository**: `https://github.com/mdumr4/resume-ranker`
* **Cloud Bucket**: `hf://buckets/mdumr4/Redrob`
* **License**: MIT
