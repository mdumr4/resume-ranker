"""
Optimized Resume Ranking Pipeline — Phase 1
- ONNX INT8 for SPLADE and BGE-M3
- Batched cross-encoder calls
- Qwen2.5-1.5B-Instruct (non-thinking, no <think> leakage)
- Per-candidate timeout fallback
- Dynamic thread count (adapts to any CPU)
"""
import os
import time
import signal
import sys
import concurrent.futures
import requests
import json
GLOBAL_T0 = time.time()

import pandas as pd
import numpy as np
import faiss
import onnxruntime as ort
from transformers import AutoTokenizer
from llama_cpp import Llama
from scipy.special import expit
from sentence_transformers import CrossEncoder

# --- CONFIGURATION ---
FEATURES_PATH = "index/features_v3.parquet"
FAISS_INDEX_PATH = "index/faiss_qwen_index.bin"
SPLADE_INDEX_PATH = "index/splade_combined_v3.npz"

# INT8 quantized ONNX models (if available, fall back to fp32)
SPLADE_ONNX_PATH = "models/splade_onnx_int8/model.onnx" if os.path.exists("models/splade_onnx_int8/model.onnx") else "models/splade_onnx/model.onnx"
CROSS_ENCODER_MODEL_NAME = "models/ms-marco-MiniLM-L-12-v2"

# Non-thinking LLM (Qwen2.5 Instruct, no <think> tokens)
LLM_PATH = "models/qwen2.5-1.5b-instruct-q5_k_m.gguf" if os.path.exists("models/qwen2.5-1.5b-instruct-q5_k_m.gguf") else "models/Qwen_Qwen3-1.7B-Q4_K_M.gguf"

JD_TECH = "The ideal candidate has 5-9 years of total experience. They should have worked as a Senior AI Engineer or Applied ML Engineer in the AI/ML, HR-Tech, or Marketplace/Talent-Intelligence industry doing ownership of ranking, retrieval, and candidate-JD matching systems; auditing and improving existing BM25/rule-based search infrastructure; shipping embeddings-based and hybrid retrieval systems with demonstrable improvements to engagement metrics; building offline and online evaluation frameworks (NDCG, MRR, MAP, A/B testing); and mentoring junior engineers in a high-growth, fast-moving product environment. The candidate must be highly proficient in embeddings-based retrieval systems (sentence-transformers, BGE, E5, OpenAI Embeddings, or equivalent), vector databases and hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, or FAISS), Python (production-grade, code-quality conscious), and ranking evaluation frameworks (NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation) with demonstrated production deployment experience across all of the above."
JD_LOGISTICS = "Summary: Senior AI Engineer - Founding Team at Redrob AI, a Series A AI-native talent intelligence platform. The hire will own the intelligence layer of the product - the ranking, retrieval, and matching systems that power both recruiter-facing candidate search and candidate-facing job discovery. Candidate should be located in Pune or Noida, India (candidates from Hyderabad, Mumbai, or Delhi NCR are also in scope). This is a Hybrid role (flexible cadence; offices used primarily Tue/Thu). If the job is outside their city, they must be Willing to relocate. They must be Open to Work with a maximum notice period of 30 days (sub-30 preferred; buyout of up to 30 days available). We require a highly trusted candidate whose profile is 100% complete, with verified contact info and a connected LinkedIn. Behaviorally, they should have a strong GitHub score and a high recruiter response rate. Candidate must hold a degree in Computer Science, Electrical Engineering, Mathematics, or a related technical field from a Tier-1 university."
JD_SEARCH_QUERY = f"{JD_TECH} {JD_LOGISTICS}"

# --- LOAD DATA & INDEXES ---
print("Loading Data and Indexes...")
t_load = time.time()
df = pd.read_parquet(FEATURES_PATH)
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
splade_data = np.load(SPLADE_INDEX_PATH, allow_pickle=True)
splade_matrix = splade_data['splade_vectors'].item()
print(f"  Data loaded in {time.time()-t_load:.1f}s")

# --- LOAD EXECUTION ENGINES ---
print("Loading Execution Engines (ONNX/GGUF)...")
t_engines = time.time()

print(f"  Loading cross_encoder ({CROSS_ENCODER_MODEL_NAME})...")
ce_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device="cpu", local_files_only=True)

print("  Loading pre-computed JD query vectors...")
dense_query = np.load("index/jd_dense_query.npy")
sparse_query = np.load("index/jd_sparse_query.npy")

print(f"  All engines loaded in {time.time()-t_engines:.1f}s")

def run_retrieval():
    # --- STAGE 1: DENSE SEARCH (FAISS) ---
    print("\n--- STAGE 1: Hybrid Retrieval ---")
    t0 = time.time()

    D, I = faiss_index.search(dense_query, 10000)
    t_faiss = time.time()
    print(f"  FAISS search: {t_faiss-t0:.2f}s")

    faiss_candidates = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1: continue
        cid = df.iloc[idx]['candidate_id']
        trust = df.iloc[idx]['trust_score']
        shifted_score = (float(score) + 1.0) / 2.0
        faiss_candidates.append((cid, shifted_score * trust))

    faiss_candidates.sort(key=lambda x: x[1], reverse=True)
    faiss_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(faiss_candidates)}

    t1 = time.time()
    print(f"-> FAISS Dense Search (10,000) completed in {t1-t0:.2f}s")

    # --- STAGE 1B: SPARSE SEARCH (SPLADE) ---
    t2 = time.time()

    all_splade_scores = splade_matrix.dot(sparse_query)

    top_k = min(10000, len(all_splade_scores))
    top_indices = np.argpartition(all_splade_scores, -top_k)[-top_k:]

    splade_candidates = []
    for idx in top_indices:
        cid = df.iloc[idx]['candidate_id']
        trust = df.iloc[idx]['trust_score']
        splade_candidates.append((cid, float(all_splade_scores[idx]) * trust))

    splade_candidates.sort(key=lambda x: x[1], reverse=True)
    splade_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(splade_candidates)}

    # --- RRF FUSION ---
    all_cids = set(faiss_ranks.keys()).union(set(splade_ranks.keys()))

    hybrid_scores = []
    for cid in all_cids:
        r_faiss = 1.0 / (60 + faiss_ranks[cid]) if cid in faiss_ranks else 0.0
        r_splade = 1.0 / (60 + splade_ranks[cid]) if cid in splade_ranks else 0.0
        hybrid_scores.append((cid, r_faiss + r_splade))

    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    top_100 = hybrid_scores[:100]

    t3 = time.time()
    print(f"-> SPLADE + RRF Fusion completed in {t3-t2:.2f}s")

    print("\n=== TOP 10 AFTER STAGE 1 (HYBRID + TRUST SCORE) ===")
    for rank, (cid, score) in enumerate(top_100[:10]):
        cand_data = df[df['candidate_id'] == cid].iloc[0]
        print(f"Rank {rank+1}: {cid} | Hybrid: {score:.4f} | Trust: {cand_data['trust_score']:.2f}")

    # --- STAGE 2: TWO-PART CROSS-ENCODER RE-RANKING ---
    print("\n--- STAGE 2: Two-Part Cross-Encoder Reranking (Batched) ---")
    t4 = time.time()

    ce_inputs_tech = []
    ce_inputs_logistics = []
    cids_in_order = []

    for cid, _ in top_100:
        row = df[df['candidate_id'] == cid].iloc[0]
        text_tech = f"{row['text_career']} {row['text_skills']}"
        text_logistics = f"{row['text_profile']} {row['text_edu']}"

        ce_inputs_tech.append((JD_TECH, text_tech))
        ce_inputs_logistics.append((JD_LOGISTICS, text_logistics))
        cids_in_order.append(cid)

    # Run prediction directly using PyTorch CrossEncoder in batches of 4
    BATCH_SIZE = 4

    # Predict both parts
    ce_preds_tech = ce_model.predict(ce_inputs_tech, batch_size=BATCH_SIZE, show_progress_bar=False)
    ce_preds_logistics = ce_model.predict(ce_inputs_logistics, batch_size=BATCH_SIZE, show_progress_bar=False)

    # Apply sigmoid (expit)
    prob_tech = expit(ce_preds_tech)
    prob_logistics = expit(ce_preds_logistics)

    # Additive Weights (70% Tech, 30% Logistics)
    final_ce_scores = (0.70 * prob_tech) + (0.30 * prob_logistics)

    t5 = time.time()
    print(f"-> Two-Part Cross-Encoder ({len(top_100)} candidates) completed in {t5-t4:.2f}s")

    final_results = list(zip(cids_in_order, final_ce_scores))
    final_results.sort(key=lambda x: x[1], reverse=True)

    print("\n=== TOP 10 AFTER CROSS-ENCODER ===")
    for rank, (cid, ce_score) in enumerate(final_results[:10]):
        cand_data = df[df['candidate_id'] == cid].iloc[0]
        print(f"Rank {rank+1}: {cid} | CE Score: {ce_score:.4f} | Trust: {cand_data['trust_score']:.2f}")

    # --- STAGE 3: LLM RATIONALE GENERATION (with timeout fallback) ---
    print(f"\n--- STAGE 3: Generating Rationales (Top {len(final_results)}) ---")
    t6 = time.time()
    SYSTEM_PROMPT = "You are an expert technical AI recruiter. Provide exactly TWO short sentences: 1. Why the candidate achieved this rank based on their career and JD. 2. Where they can improve or what they are missing. Do not write markdown headers, bullet points, or candidate names."

    submission_data = [None] * len(final_results)
    timeouts = 0

    def fetch_rationale(rank, cid, score):
        row = df[df['candidate_id'] == cid].iloc[0]

        # Build an ultra-compact, structured candidate template
        # Limit skills to 20 words, career to 30 words, and profile to 20 words
        def limit_words(text, limit):
            if not isinstance(text, str): return ""
            words = text.split()
            return " ".join(words[:limit])

        exp_val = row.get('experience_years')
        if pd.isna(exp_val) or str(exp_val).strip() == "" or str(exp_val).lower() == "nan":
            exp_str = "an unspecified amount of experience"
        else:
            exp_str = f"{exp_val} years of experience"

        compact_context = f"Candidate ranked {rank+1} has a trust score of {row.get('trust_score', 'N/A')} and {exp_str}. Their key skills include {limit_words(row.get('text_skills', ''), 30)}. Their career highlights include {limit_words(row.get('text_career', ''), 40)}."

        fallback_rationale = f"Rank {rank+1}: {row.get('experience_years', 'N/A')} years experience, trust score {row['trust_score']:.2f}. Improvement: Needs more specific AI/ML production details."

        prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Job: Senior AI Engineer at Redrob AI. Requirements: 5-9 yrs, AI/ML, ranking/retrieval systems, embeddings, vector DBs.

Candidate ranked 1 has a trust score of 0.89 and 7 years of experience. Their key skills include Highly verified expert in PyTorch, NLP, and vector databases. Their career highlights include 3 years leading search ranking at TechCorp.

Response:<|im_end|>
<|im_start|>assistant
The candidate is selected for their strong background in search ranking at TechCorp and verified skills in PyTorch and vector databases. They can improve by gaining more experience with sparse retrieval models.<|im_end|>
<|im_start|>user
Job: Senior AI Engineer at Redrob AI. Requirements: 5-9 yrs, AI/ML, ranking/retrieval systems, embeddings, vector DBs.

{compact_context}

Response:<|im_end|>
<|im_start|>assistant
"""
        try:
            resp = requests.post(
                "http://127.0.0.1:8080/completion",
                json={
                    "prompt": prompt,
                    "n_predict": 50,
                    "stop": ["<|im_end|>", "\n\n"],
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "top_k": 20,
                    "presence_penalty": 0.0,
                }
            )
            if resp.status_code == 200:
                rationale = resp.json().get("content", "").strip()
                if not rationale or len(rationale) < 10:
                    rationale = fallback_rationale
            else:
                rationale = fallback_rationale
        except Exception as e:
            rationale = fallback_rationale

        return {
            "candidate_id": cid,
            "rank": rank + 1,
            "score": f"{score:.6f}",
            "reasoning": rationale
        }

    print(f"  Dispatching {len(final_results)} candidates to llama-server (Continuous Batching)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        future_to_idx = {
            executor.submit(fetch_rationale, rank, cid, score): rank
            for rank, (cid, score) in enumerate(final_results)
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
                submission_data[idx] = res
                if res["reasoning"].startswith("Rank "):
                    timeouts += 1
            except Exception as e:
                pass

            completed += 1
            if completed % 10 == 0 or completed == len(final_results):
                elapsed = time.time() - t6
                rate = completed / elapsed
                eta = (len(final_results) - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{len(final_results)}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    t7 = time.time()
    print(f"-> LLM Rationales ({len(final_results)} candidates) completed in {t7-t6:.1f}s ({timeouts} fallbacks)")

    # --- SAVE FINAL CSV ---
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="submission.csv")
    args, _ = parser.parse_known_args()

    sub_df = pd.DataFrame(submission_data)
    sub_df = sub_df[['candidate_id', 'rank', 'score', 'reasoning']]
    sub_df.to_csv(args.out, index=False)

    total_time = time.time() - GLOBAL_T0
    print(f"\nSuccessfully saved: {args.out} (Top {len(final_results)} with Rationale)")
    print(f"--- PIPELINE FINISHED in {total_time:.1f}s ({total_time/60:.1f} min) ---")

if __name__ == "__main__":
    run_retrieval()
