import os
import time
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# --- CONFIGURATION ---
FEATURES_PATH = "../../index/features_v3.parquet"
FAISS_INDEX_PATH = "../../index/faiss_index_v3.bin"
SPLADE_INDEX_PATH = "../../index/splade_combined_v3.npz"

# Example Output from LLM_PROMPT.md
# We split the query into two vectors: one for Career/Skills (FAISS/SPLADE) and one for Edu/Profile (FAISS/SPLADE).
# But since we have a single combined index, we combine the queries for search, and the Cross-Encoder gets the whole thing.
JD_SEARCH_QUERY = "5-9 years experience, building ranking models, search systems, machine learning pipelines. Python, Machine Learning, Recommendation Systems, XGBoost, PyTorch, FAISS. Fast-paced startup environment, autonomous, high ownership. Location: Pune, Noida. Mode: Hybrid. Budget: 15-40 LPA. Bachelors in CS or related field. Real-world experience matters more"

# Ensure we use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def load_data():
    print("Loading Parquet Data...")
    df = pd.read_parquet(FEATURES_PATH)
    
    print("Loading FAISS Index...")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    
    print("Loading SPLADE Index...")
    splade_data = np.load(SPLADE_INDEX_PATH, allow_pickle=True)
    splade_matrix = splade_data['splade_vectors'].item() 
    
    return df, faiss_index, splade_matrix

def run_retrieval():
    df, faiss_index, splade_matrix = load_data()
    
    print("Loading Retrieval Models...")
    dense_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    print("Loading SPLADE Model...")
    hf_token = "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"
    splade_model = AutoModelForMaskedLM.from_pretrained("naver/splade-v3", token=hf_token)
    splade_model.to(device)
    splade_model.eval()
    tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3", token=hf_token)
    
    # --- STAGE 1: DENSE SEARCH (FAISS) ---
    print("\nEncoding JD for Dense Search...")
    t0 = time.time()
    dense_query = dense_model.encode([JD_SEARCH_QUERY])
    
    # CRITICAL FIX: We retrieve a massive pool (K=10000) from FAISS independently
    D, I = faiss_index.search(dense_query, 10000) 
    
    faiss_candidates = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1: continue
        cid = df.iloc[idx]['candidate_id']
        trust = df.iloc[idx]['trust_score']
        # FAISS Cosine is [-1, 1]. Shift to strictly [0, 1] before applying trust!
        shifted_score = (float(score) + 1.0) / 2.0
        faiss_candidates.append((cid, shifted_score * trust))
        
    faiss_candidates.sort(key=lambda x: x[1], reverse=True)
    faiss_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(faiss_candidates)}
    
    t1 = time.time()
    print(f"-> FAISS Dense Search (10,000) completed in {t1-t0:.4f} seconds.")

    # --- STAGE 1B: SPARSE SEARCH (SPLADE) ---
    print("\nEncoding JD for Sparse Search...")
    t2 = time.time()
    with torch.no_grad():
        inputs = tokenizer([JD_SEARCH_QUERY], return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = splade_model(**inputs)
        sparse_query = torch.max(
            torch.log(1 + torch.relu(outputs.logits)) * inputs.attention_mask.unsqueeze(-1),
            dim=1
        )[0].cpu().numpy().squeeze()
        
    print("Executing Global SPLADE Dot Product...")
    # Calculate SPLADE dot product for all 100,000 candidates instantly via sparse matrix multiplication
    all_splade_scores = splade_matrix.dot(sparse_query)
    
    # Efficiently get top 10,000 SPLADE scores
    top_k = min(10000, len(all_splade_scores))
    top_indices = np.argpartition(all_splade_scores, -top_k)[-top_k:]
    
    splade_candidates = []
    for idx in top_indices:
        cid = df.iloc[idx]['candidate_id']
        trust = df.iloc[idx]['trust_score']
        # SPLADE is already non-negative, just apply trust
        splade_candidates.append((cid, float(all_splade_scores[idx]) * trust))
        
    splade_candidates.sort(key=lambda x: x[1], reverse=True)
    splade_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(splade_candidates)}
    
    # --- STAGE 1C: RECIPROCAL RANK FUSION (RRF) ---
    print("Fusing Dense and Sparse ranks via RRF...")
    all_cids = set(faiss_ranks.keys()).union(set(splade_ranks.keys()))
    
    hybrid_scores = []
    for cid in all_cids:
        # If candidate is missing from a stream's Top 10k, they contribute 0 to the sum
        r_faiss = 1.0 / (60 + faiss_ranks[cid]) if cid in faiss_ranks else 0.0
        r_splade = 1.0 / (60 + splade_ranks[cid]) if cid in splade_ranks else 0.0
        hybrid_scores.append((cid, r_faiss + r_splade))
        
    # Sort the final global pool by RRF Score and slice Top 500
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    top_500 = hybrid_scores[:500]
    
    t3 = time.time()
    print(f"-> SPLADE & RRF Fusion completed in {t3-t2:.4f} seconds.")
    
    print("\n=== TOP 10 AFTER STAGE 1 (HYBRID + TRUST SCORE) ===")
    for rank, (cid, score) in enumerate(top_500[:10]):
        cand_data = df[df['candidate_id'] == cid].iloc[0]
        print(f"Rank {rank+1}: {cid} | Hybrid Score: {score:.4f} | Trust: {cand_data['trust_score']:.2f}")

    # --- STAGE 2: CROSS-ENCODER RE-RANKING ---
    print("\nLoading Cross-Encoder...")
    ce_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
    
    ce_inputs = []
    cids_in_order = []
    
    for cid, _ in top_500:
        row = df[df['candidate_id'] == cid].iloc[0]
        # PERFECT ALIGNMENT: Concatenating exactly in the user's requested order
        # Career -> Profile -> Skills -> Edu
        text_combined = f"{row['text_career']} {row['text_profile']} {row['text_skills']} {row['text_edu']}"
        ce_inputs.append([JD_SEARCH_QUERY, text_combined])
        cids_in_order.append(cid)
        
    print(f"\nScoring Top {len(top_500)} with Cross-Encoder...")
    t4 = time.time()
    ce_predictions = ce_model.predict(ce_inputs)
    t5 = time.time()
    print(f"-> Cross-Encoder (500 candidates) completed in {t5-t4:.4f} seconds.")
    
    final_results = list(zip(cids_in_order, ce_predictions))
    final_results.sort(key=lambda x: x[1], reverse=True)

    # --- STAGE 3: GENERATE KAGGLE SUBMISSION ---
    print("\nGenerating Kaggle submission.csv...")
    submission_data = []
    
    # We only need the Top 100 for the submission
    for rank, (cid, score) in enumerate(final_results[:100]):
        # Grab original row for reasoning
        row = df[df['candidate_id'] == cid].iloc[0]
        
        # Build a reasoning string similar to the sample submission
        try:
            # We try to extract their latest job title from the career history JSON
            # But since we flattened it, we can just grab their trust score and YOE from tabular
            yoe = row['num_jobs'] * 2 # Rough estimation if real YOE column is missing, or we use Trust Score
            reasoning = f"Matched via Hybrid Cross-Encoder; Trust Score: {row['trust_score']:.2f}; Jobs on Profile: {row['num_jobs']}"
        except:
            reasoning = f"Matched via Hybrid Cross-Encoder; Trust Score: {row['trust_score']:.2f}"
            
        submission_data.append({
            "candidate_id": cid,
            "rank": rank + 1,
            "score": f"{score:.4f}",
            "reasoning": reasoning
        })
        
    sub_df = pd.DataFrame(submission_data)
    sub_df.to_csv("submission.csv", index=False)
    print("Successfully saved: submission.csv (Top 100 Candidates) inside Trail_6 folder")
    
    print("\n=== FINAL TOP 10 AFTER CROSS-ENCODER ===")
    for rank, (cid, ce_score) in enumerate(final_results[:10]):
        cand_data = df[df['candidate_id'] == cid].iloc[0]
        # Notice we do NOT subtract structural math penalties here, because
        # we already multiplied the Trust Score in Stage 1!
        print(f"Rank {rank+1}: {cid} | CE Score: {ce_score:.4f} | Trust: {cand_data['trust_score']:.2f}")
        print(f"   -> Career: {cand_data['text_career'][:80]}...")

if __name__ == "__main__":
    run_retrieval()
