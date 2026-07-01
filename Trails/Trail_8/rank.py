import os
import gc
import time
import json
import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForMaskedLM, AutoTokenizer
from llama_cpp import Llama, LlamaGrammar

GLOBAL_T0 = time.time()

# --- HF SPACES OFFLINE PATHS ---
# In Hugging Face Spaces, we download these into the space via git-lfs or dataset mounts.
FEATURES_PATH = "index/features_v3.parquet"
FAISS_INDEX_PATH = "index/faiss_qwen_index.bin"
SPLADE_INDEX_PATH = "index/splade_combined_v3.npz"

# Local model paths for HF Spaces (No Internet)
QWEN_EMBED_PATH = "models/Qwen3-Embedding-0.6B"
SPLADE_MODEL_PATH = "models/splade-v3"
BGE_RERANKER_PATH = "models/bge-reranker-v2-m3"
QWEN_REASONING_GGUF = "models/qwen2.5-3b-instruct-q4_k_m.gguf"

# (Mock extracted JD based on our new Grammatical Template extraction)
# In production, this is dynamically passed from the extraction LLM
JD_EXTRACTED = {
    "Career": "The ideal candidate has 5-9 years of total experience. They should have worked as a Senior AI Engineer or Applied ML Engineer in the AI/ML, HR-Tech, or Marketplace/Talent-Intelligence industry doing ownership of ranking, retrieval, and candidate-JD matching systems; auditing and improving existing BM25/rule-based search infrastructure; shipping embeddings-based and hybrid retrieval systems with demonstrable improvements to engagement metrics; building offline and online evaluation frameworks (NDCG, MRR, MAP, A/B testing); and mentoring junior engineers in a high-growth, fast-moving product environment.",
    "Skills": "The candidate must be highly proficient in embeddings-based retrieval systems (sentence-transformers, BGE, E5, OpenAI Embeddings, or equivalent), vector databases and hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, or FAISS), Python (production-grade, code-quality conscious), and ranking evaluation frameworks (NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation) with demonstrated production deployment experience across all of the above.",
    "Profile": "Summary: Senior AI Engineer - Founding Team at Redrob AI, a Series A AI-native talent intelligence platform. The hire will own the intelligence layer of the product - the ranking, retrieval, and matching systems that power both recruiter-facing candidate search and candidate-facing job discovery. Candidate should be located in Pune or Noida, India (candidates from Hyderabad, Mumbai, or Delhi NCR are also in scope). This is a Hybrid role (flexible cadence; offices used primarily Tue/Thu). If the job is outside their city, they must be Willing to relocate. They must be Open to Work with a maximum notice period of 30 days (sub-30 preferred; buyout of up to 30 days available). We require a highly trusted candidate whose profile is 100% complete, with verified contact info and a connected LinkedIn. Behaviorally, they should have a strong GitHub score and a high recruiter response rate.",
    "Education": "Candidate must hold a degree in Computer Science, Electrical Engineering, Mathematics, or a related technical field from a Tier-1 university. (Note: no explicit degree requirement or tier was stated in the JD; this row should be left as optional/unscored in the database.)"
}

# The flat string passed to Encoders
JD_SEARCH_QUERY = f"[Career] {JD_EXTRACTED['Career']} [Skills] {JD_EXTRACTED['Skills']} [Profile] {JD_EXTRACTED['Profile']} [Education] {JD_EXTRACTED['Education']}"

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_data():
    print("Loading Data...")
    df = pd.read_parquet(FEATURES_PATH)
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    splade_data = np.load(SPLADE_INDEX_PATH, allow_pickle=True)
    splade_matrix = splade_data['splade_vectors'].item()
    
    # Load raw JSON to extract raw trust formula components for the reasoning LLM
    print("Loading Raw Candidates JSON...")
    candidates_dict = {}
    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            cand = json.loads(line)
            candidates_dict[cand['candidate_id']] = cand
            
    return df, faiss_index, splade_matrix, candidates_dict

def run_pipeline():
    df, faiss_index, splade_matrix, candidates_dict = load_data()

    # --- STAGE 1: DENSE SEARCH ---
    t0 = time.time()
    print("Loading Dense Model...")
    dense_model = SentenceTransformer(QWEN_EMBED_PATH, device=device, trust_remote_code=True)
    dense_query = dense_model.encode([JD_SEARCH_QUERY], normalize_embeddings=True)
    
    # Garbage collect Dense model to free RAM
    del dense_model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    D, I = faiss_index.search(dense_query, 10000)
    
    faiss_candidates = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1: continue
        cid = df.iloc[idx]['candidate_id']
        trust = df.iloc[idx]['trust_score']
        shifted_score = (float(score) + 1.0) / 2.0
        faiss_candidates.append((cid, shifted_score * trust)) # Apply Trust before RRF
        
    faiss_candidates.sort(key=lambda x: x[1], reverse=True)
    faiss_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(faiss_candidates)}
    t1 = time.time()
    print(f"[Profiler] FAISS Dense Search (10,000): {t1-t0:.4f}s")

    # --- STAGE 1B: SPARSE SEARCH ---
    print("Loading Sparse Model...")
    splade_model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL_PATH)
    splade_model.to(device)
    splade_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL_PATH)
    
    tokens = tokenizer(JD_SEARCH_QUERY, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = splade_model(**tokens)
        sparse_query = torch.max(
            torch.log(1 + torch.relu(outputs.logits)) * tokens.attention_mask.unsqueeze(-1),
            dim=1
        )[0].cpu().numpy().squeeze()
        
    # Garbage collect Sparse model
    del splade_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    all_splade_scores = splade_matrix.dot(sparse_query)
    top_k = min(10000, len(all_splade_scores))
    top_indices = np.argpartition(all_splade_scores, -top_k)[-top_k:]
    
    splade_candidates = []
    for idx in top_indices:
        cid = df.iloc[idx]['candidate_id']
        trust = df.iloc[idx]['trust_score']
        splade_candidates.append((cid, float(all_splade_scores[idx]) * trust)) # Apply Trust before RRF
        
    splade_candidates.sort(key=lambda x: x[1], reverse=True)
    splade_ranks = {cid: rank + 1 for rank, (cid, _) in enumerate(splade_candidates)}
    t2 = time.time()
    print(f"[Profiler] SPLADE Sparse Search (10,000): {t2-t1:.4f}s")

    # --- RRF ---
    all_cids = set(faiss_ranks.keys()).union(set(splade_ranks.keys()))
    hybrid_scores = []
    for cid in all_cids:
        r_faiss = 1.0 / (60 + faiss_ranks[cid]) if cid in faiss_ranks else 0.0
        r_splade = 1.0 / (60 + splade_ranks[cid]) if cid in splade_ranks else 0.0
        hybrid_scores.append((cid, r_faiss + r_splade))
        
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)
    top_500 = hybrid_scores[:500]
    t3 = time.time()
    print(f"[Profiler] RRF & Trust Score Sorting: {t3-t2:.4f}s")

    # --- STAGE 2: CROSS-ENCODER ---
    print("Loading Cross-Encoder Model...")
    ce_model = CrossEncoder(BGE_RERANKER_PATH, device=device)
    ce_inputs = []
    cids_in_order = []
    
    for cid, _ in top_500:
        row = df[df['candidate_id'] == cid].iloc[0]
        text_combined = f"{row['text_career']} {row['text_skills']} {row['text_profile']} {row['text_edu']}"
        ce_inputs.append([JD_SEARCH_QUERY, text_combined])
        cids_in_order.append(cid)
        
    ce_preds = ce_model.predict(ce_inputs, batch_size=8)
    
    # Garbage collect Cross-Encoder
    del ce_model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    t4 = time.time()
    print(f"[Profiler] Cross-Encoder Scoring (500): {t4-t3:.4f}s")

    final_results = list(zip(cids_in_order, ce_preds))
    final_results.sort(key=lambda x: x[1], reverse=True)

    # --- STAGE 3: GENERATIVE REASONING (QWEN3-4B via LLAMA.CPP) ---
    submission_data = []
    
    print("Loading Qwen Reasoning LLM (llama.cpp)...")
    llm = Llama(model_path=QWEN_REASONING_GGUF, n_ctx=1024, n_threads=os.cpu_count() or 8, verbose=False)
    
    grammar = r'''
    root ::= "{" ws "\"score\":" ws [1-9] [0]? "," ws "\"reasoning\":" ws string "}"
    string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\""
    ws ::= [ \t\n]*
    '''
    json_grammar = LlamaGrammar.from_string(grammar)
    
    t5 = time.time()
    print(f"[Profiler] Llama.cpp Initialization: {t5-t4:.4f}s")

    for rank_idx, (cid, score) in enumerate(final_results[:100]):
        rank_pos = rank_idx + 1
        row = df[df['candidate_id'] == cid].iloc[0]
        
        # Get raw trust signals from the JSON dictionary
        raw_cand = candidates_dict.get(cid, {})
        sig = raw_cand.get("redrob_signals", {})
        
        # Format the precise trust formula variables
        trust_context = (
            f"- Profile Completeness: {sig.get('profile_completeness_score', 0)}%\n"
            f"- Verified Identity: Email ({sig.get('verified_email', False)}), Phone ({sig.get('verified_phone', False)}), LinkedIn ({sig.get('linkedin_connected', False)})\n"
            f"- Recruiter Response Rate: {sig.get('recruiter_response_rate', 0)*100}%\n"
            f"- Interview Completion Rate: {sig.get('interview_completion_rate', 0)*100}%\n"
            f"- GitHub Activity Score: {sig.get('github_activity_score', 0)}\n"
            f"- Date Gap Violations: {row.get('trust_career_gap_violation', False)} | Timeline Violations: {row.get('trust_date_violation', False)}\n"
        )
        
        # Rank-Aware Prompt Logic
        rank_context = f"This candidate is ranked #{rank_pos} out of 100,000. "
        if rank_pos <= 10:
            rank_context += "They are a phenomenal, top-tier match for the role. Focus on highlighting their absolute strengths and high trust."
        elif rank_pos <= 50:
            rank_context += "They are a very strong match for the role, though slightly outclassed by the top 10. Focus on their solid fit."
        else:
            rank_context += "They are a good baseline match, but are at the bottom of the Top 100. Acknowledge they meet requirements but lack the extreme depth of top candidates."
            
        prompt = f"""<|im_start|>system
You are an expert technical recruiter evaluating candidates. Output valid JSON containing a 'score' (1-10) and a concise, 1-sentence 'reasoning'.
CRITICAL: {rank_context}<|im_end|>
<|im_start|>user
[JOB REQUIREMENTS]
Career: {JD_EXTRACTED['Career']}
Skills: {JD_EXTRACTED['Skills']}
Education: {JD_EXTRACTED['Education']}
Logistics & Behavior: {JD_EXTRACTED['Profile']}

[CANDIDATE FACTS]
Career History: {row['text_career']}
Skills: {row['text_skills']}
Education: {row['text_edu']}
Logistics & Behavior: {row['text_profile']}

[RAW TRUST SCORE BREAKDOWN (Final Score: {row['trust_score']:.2f}/1.00)]
{trust_context}

Based on the candidate's exact facts and RAW trust formula signals compared to the job requirements, write the 1-sentence reasoning explaining their fit.<|im_end|>
<|im_start|>assistant
"""
        ts = time.time()
        output = llm(prompt, max_tokens=80, grammar=json_grammar, temperature=0.1)
        te = time.time()
        
        gen_text = output['choices'][0]['text']
        try:
            res_dict = json.loads(gen_text)
            reasoning_text = res_dict.get('reasoning', gen_text)
        except:
            reasoning_text = gen_text
            
        print(f"[Profiler] Rank {rank_pos} Generation: {te-ts:.4f}s -> {reasoning_text}")
        
        submission_data.append({
            "candidate_id": cid,
            "rank": rank_pos,
            "score": f"{score:.4f}",
            "reasoning": reasoning_text
        })
        
    sub_df = pd.DataFrame(submission_data)
    sub_df.to_csv("submission.csv", index=False)
    
    total_time = time.time() - GLOBAL_T0
    print(f"\n[Profiler] Total Pipeline Execution: {total_time:.4f}s")
    print(f"Successfully saved: submission.csv (Top 100 Candidates)")

if __name__ == "__main__":
    run_pipeline()
