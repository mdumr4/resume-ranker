import os
import time
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from splade.models.transformer_rep import Splade
import torch
from transformers import AutoTokenizer

# --- CONFIGURATION ---
FEATURES_PATH = "../../index/features_v3.parquet"
FAISS_INDEX_PATH = "../../index/faiss_index_v3.bin"
SPLADE_INDEX_PATH = "../../index/splade_combined_v3.npz"

# Example Output from LLM_PROMPT.md
# We split the query into two vectors: one for Career/Skills (FAISS/SPLADE) and one for Edu/Profile (FAISS/SPLADE).
# But since we have a single combined index, we combine the queries for search, and the Cross-Encoder gets the whole thing.
JD_SEARCH_QUERY = "Candidate has a total of 5 years of experience. Worked as a Backend Engineer in the Fintech industry. Highly verified expert in Python and Apache Kafka. Verified advanced in Kubernetes and AWS. Holds a B.E. in Computer Science. Achieved a grade from a tier_1 institution. Candidate expects 20-30 LPA, prefers hybrid work mode, notice period is 30 days, willing to relocate: True."

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
    
    # We use SPLADE v3 for Sparse Search
    splade_model = Splade("naver/splade-v3", agg="max")
    splade_model.to(device)
    splade_model.eval()
    tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3")
    
    # --- STAGE 1: DENSE SEARCH (FAISS) ---
    print("\nEncoding JD for Dense Search...")
    t0 = time.time()
    dense_query = dense_model.encode([JD_SEARCH_QUERY])
    
    # CRITICAL FIX: We must retrieve a massive pool (K=10000) from FAISS first!
    D, I = faiss_index.search(dense_query, 10000) 
    
    dense_scores = {}
    for score, idx in zip(D[0], I[0]):
        cid = df.iloc[idx]['candidate_id']
        dense_scores[cid] = float(score)
    t1 = time.time()
    print(f"-> FAISS Dense Search (10,000) completed in {t1-t0:.4f} seconds.")

    # --- STAGE 1: SPARSE SEARCH (SPLADE) ---
    print("\nEncoding JD for Sparse Search...")
    t2 = time.time()
    with torch.no_grad():
        inputs = tokenizer(JD_SEARCH_QUERY, return_tensors="pt", truncation=True, max_length=512).to(device)
        sparse_query = splade_model(**inputs).squeeze().cpu().numpy()
        
    print("Executing SPLADE Dot Product and applying Trust Math...")
    hybrid_scores = []
    
    # Iterate through the massive 10,000 candidate pool
    for cid in dense_scores.keys():
        idx = df.index[df['candidate_id'] == cid].tolist()[0]
        
        cand_sparse = splade_matrix.getrow(idx).toarray().squeeze()
        splade_score = np.dot(sparse_query, cand_sparse)
        
        # Add FAISS + SPLADE
        raw_total_score = dense_scores[cid] + float(splade_score)
        
        # APPY TRUST SCORE TO THE MASSIVE POOL
        trust = df.iloc[idx]['trust_score']
        final_stage1_score = raw_total_score * trust
        
        hybrid_scores.append((cid, final_stage1_score))
        
    # NOW we sort the 10,000 candidates by their Trust-adjusted scores
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    top_500 = hybrid_scores[:500]
    t3 = time.time()
    print(f"-> SPLADE & Trust Math completed in {t3-t2:.4f} seconds.")
    
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
    
    print("\n=== FINAL TOP 10 AFTER CROSS-ENCODER ===")
    for rank, (cid, ce_score) in enumerate(final_results[:10]):
        cand_data = df[df['candidate_id'] == cid].iloc[0]
        # Notice we do NOT subtract structural math penalties here, because
        # we already multiplied the Trust Score in Stage 1!
        print(f"Rank {rank+1}: {cid} | CE Score: {ce_score:.4f} | Trust: {cand_data['trust_score']:.2f}")
        print(f"   -> Career: {cand_data['text_career'][:80]}...")

if __name__ == "__main__":
    run_retrieval()
