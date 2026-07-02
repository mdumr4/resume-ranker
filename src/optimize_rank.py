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

# --- CONFIGURATION ---
FEATURES_PATH = "index/features_v3.parquet"
FAISS_INDEX_PATH = "index/faiss_qwen_index.bin"
SPLADE_INDEX_PATH = "index/splade_combined_v3.npz"

# INT8 quantized ONNX models (if available, fall back to fp32)
SPLADE_ONNX_PATH = "models/splade_onnx_int8/model.onnx" if os.path.exists("models/splade_onnx_int8/model.onnx") else "models/splade_onnx/model.onnx"
BGE_ONNX_PATH = "models/bge_onnx_v2_int8/model.onnx" if os.path.exists("models/bge_onnx_v2_int8/model.onnx") else "models/bge_onnx_v2/model.onnx"

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

# ONNX session options — limit threads to prevent thrashing
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 8
sess_options.inter_op_num_threads = 2
providers = ['CPUExecutionProvider']

print(f"  Loading dense_model (Qwen Embedding GGUF)...")
dense_model = Llama(
    model_path="models/Qwen3-Embedding-0.6B-Q8_0.gguf",
    embedding=True,
    verbose=False
)

print(f"  Loading splade_session ({SPLADE_ONNX_PATH})...")
splade_tokenizer = AutoTokenizer.from_pretrained("models/splade_tokenizer", local_files_only=True)
splade_session = ort.InferenceSession(SPLADE_ONNX_PATH, sess_options, providers=providers)

print(f"  Loading bge_session ({BGE_ONNX_PATH})...")
bge_tokenizer = AutoTokenizer.from_pretrained("models/bge_tokenizer", local_files_only=True)
bge_session = ort.InferenceSession(BGE_ONNX_PATH, sess_options, providers=providers)

print(f"  All engines loaded in {time.time()-t_engines:.1f}s")

def run_retrieval():
    # --- STAGE 1: DENSE SEARCH (FAISS) ---
    print("\n--- STAGE 1: Hybrid Retrieval ---")
    t0 = time.time()
    
    dense_query = np.array(
        dense_model.create_embedding(JD_SEARCH_QUERY)["data"][0]["embedding"]
    ).astype(np.float32).reshape(1, -1)
    
    t_embed = time.time()
    print(f"  JD embedding: {t_embed-t0:.2f}s")
    
    D, I = faiss_index.search(dense_query, 10000) 
    t_faiss = time.time()
    print(f"  FAISS search: {t_faiss-t_embed:.2f}s")
    
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
    
    splade_inputs = splade_tokenizer([JD_SEARCH_QUERY], return_tensors="np", truncation=True, max_length=512)
    splade_logits = splade_session.run(None, {
        "input_ids": splade_inputs["input_ids"],
        "attention_mask": splade_inputs["attention_mask"]
    })[0]
    
    relu_logits = np.maximum(0, splade_logits)
    log_relu = np.log1p(relu_logits)
    att_mask = np.expand_dims(splade_inputs["attention_mask"], axis=-1)
    weighted = log_relu * att_mask
    sparse_query = np.max(weighted, axis=1).squeeze()
        
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
    
    # --- STAGE 2: BATCHED CROSS-ENCODER RE-RANKING ---
    print("\n--- STAGE 2: Cross-Encoder Reranking (Batched) ---")
    t4 = time.time()
    
    queries = []
    docs = []
    cids_in_order = []
    
    for cid, _ in top_100[:50]:
        row = df[df['candidate_id'] == cid].iloc[0]
        text_combined = f"{row['text_career']} {row['text_skills']} {row['text_profile']} {row['text_edu']}"
        queries.append(JD_SEARCH_QUERY)
        docs.append(text_combined)
        cids_in_order.append(cid)
        
    # Batch in chunks of 4 for memory efficiency on CPU (prevent RAM swapping)
    BATCH_SIZE = 4
    final_ce_scores = []
    
    for i in range(0, len(queries), BATCH_SIZE):
        b_queries = queries[i:i+BATCH_SIZE]
        b_docs = docs[i:i+BATCH_SIZE]
        
        bge_inputs = bge_tokenizer(
            b_queries, b_docs, 
            padding=True, truncation=True, max_length=512, return_tensors="np"
        )
        
        feed = {"input_ids": bge_inputs["input_ids"], "attention_mask": bge_inputs["attention_mask"]}
        # Some ONNX graphs also expect token_type_ids
        if "token_type_ids" in bge_inputs:
            feed["token_type_ids"] = bge_inputs["token_type_ids"]
        
        bge_logits = bge_session.run(None, feed)[0]
        final_ce_scores.extend(expit(bge_logits).flatten())
        
    t5 = time.time()
    print(f"-> Cross-Encoder ({len(queries)} candidates, batch={BATCH_SIZE}) completed in {t5-t4:.2f}s")
    
    final_results = list(zip(cids_in_order, final_ce_scores))
    final_results.sort(key=lambda x: x[1], reverse=True)
    
    # Append the remaining candidates from Hybrid search (rank 51-100)
    for cid, score in top_100[50:]:
        final_results.append((cid, score))

    print("\n=== TOP 10 AFTER CROSS-ENCODER ===")
    for rank, (cid, ce_score) in enumerate(final_results[:10]):
        cand_data = df[df['candidate_id'] == cid].iloc[0]
        print(f"Rank {rank+1}: {cid} | CE Score: {ce_score:.4f} | Trust: {cand_data['trust_score']:.2f}")

    # --- STAGE 3: LLM RATIONALE GENERATION (with timeout fallback) ---
    print(f"\n--- STAGE 3: Generating Rationales (Top {len(final_results)}) ---")
    t6 = time.time()
    
    SYSTEM_PROMPT = "You are an expert technical AI recruiter. Provide exactly TWO short sentences: 1. Why the candidate achieved this rank based on their skills. 2. Where they can improve or what they are missing. Keep the total response under 60 words."
    
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

{compact_context}

Response:<|im_end|>
<|im_start|>assistant
"""
        try:
            resp = requests.post(
                "http://127.0.0.1:8080/completion",
                json={
                    "prompt": prompt,
                    "n_predict": 75,
                    "stop": ["<|im_end|>", "\n\n"],
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 20,
                    "presence_penalty": 1.5,
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
            "score": f"{1.0 - (rank * 0.001):.4f}",
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
    sub_df = pd.DataFrame(submission_data)
    sub_df = sub_df[['candidate_id', 'rank', 'score', 'reasoning']]
    sub_df.to_csv("submission.csv", index=False)
    
    total_time = time.time() - GLOBAL_T0
    print(f"\nSuccessfully saved: submission.csv (Top {len(final_results)} with Rationale)")
    print(f"--- PIPELINE FINISHED in {total_time:.1f}s ({total_time/60:.1f} min) ---")

if __name__ == "__main__":
    run_retrieval()
