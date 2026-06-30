import os
import json
import time
import psutil
import torch
import numpy as np
import pandas as pd
import faiss
import logging
from scipy.sparse import load_npz, csr_matrix
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sentence_transformers import CrossEncoder, SentenceTransformer

from jd_parser import JDParser

# Force HuggingFace to offline mode to prevent httpx crashes during model loading
os.environ["HF_HUB_OFFLINE"] = "1"

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

class Trail4Ranker:
    def __init__(self, top_k=5000, rrf_k=60):
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tabular_features = None
        self.parsed_jd = {}
        self.metrics = []

    def log_metric(self, step_name, start_time, start_mem):
        end_time = time.time()
        end_mem = get_memory_usage()
        duration = end_time - start_time
        mem_diff = end_mem - start_mem
        
        metric = f"[{step_name}] Time: {duration:.2f}s | Mem Change: {mem_diff:+.2f} MB | Peak Mem: {end_mem:.2f} MB"
        logging.info(metric)
        self.metrics.append(metric)

    def load_artifacts(self):
        start_time = time.time()
        start_mem = get_memory_usage()
        
        logging.info("Parsing Pre-processed JD...")
        parser = JDParser("context/parsed_jd.json")
        self.parsed_jd = parser.parse()
        
        logging.info("Loading pre-computed indices and models...")
        self.tabular_features = pd.read_parquet("index/features.parquet")
        
        self.splade_tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3", local_files_only=True)
        self.splade_model = AutoModelForMaskedLM.from_pretrained("naver/splade-v3", local_files_only=True).to(self.device)
        self.splade_model.eval()
        
        self.sparse_matrix = load_npz("index/splade_index.npz")
        
        # Retry loop for model loading to handle intermittent httpx crashes
        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.dense_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=self.device, local_files_only=True)
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=512, device=self.device, local_files_only=True)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logging.warning(f"Model load failed: {str(e)}. Retrying in 2 seconds...")
                time.sleep(2)
        
        self.faiss_index = faiss.read_index("index/faiss.index")
        
        self.log_metric("Artifact Loading", start_time, start_mem)

    def retrieve_hybrid(self):
        start_time = time.time()
        start_mem = get_memory_usage()
        
        logging.info("Executing Sparse Search (SPLADE)...")
        jd_summary_text = self.parsed_jd["summary"]
        inputs = self.splade_tokenizer(jd_summary_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.splade_model(**inputs)
            
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.logits.size())
        logits = outputs.logits * mask
        sparse_vec = torch.max(torch.relu(logits), dim=1)[0].cpu().numpy()
        jd_csr = csr_matrix(sparse_vec)
        
        sparse_scores = self.sparse_matrix.dot(jd_csr.T).toarray().flatten()
        sparse_indices = np.argsort(sparse_scores)[::-1]
        
        logging.info("Executing Dense Search (FAISS)...")
        dense_vec = self.dense_model.encode([jd_summary_text], normalize_embeddings=True)
        dense_scores, dense_indices = self.faiss_index.search(dense_vec, len(self.tabular_features))
        dense_indices = dense_indices[0]
        
        logging.info("Fusing via Reciprocal Rank Fusion (RRF)...")
        rrf_scores = {}
        
        for rank, cand_idx in enumerate(sparse_indices):
            rrf_scores[cand_idx] = rrf_scores.get(cand_idx, 0) + (1.0 / (self.rrf_k + rank + 1))
            
        for rank, cand_idx in enumerate(dense_indices):
            rrf_scores[cand_idx] = rrf_scores.get(cand_idx, 0) + (1.0 / (self.rrf_k + rank + 1))
            
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_indices = [x[0] for x in sorted_rrf[:self.top_k]]
        
        # Get the retrieved candidates DataFrame
        retrieved_df = self.tabular_features.iloc[top_k_indices].copy()
        
        # Add RRF score 
        rrf_list = [rrf_scores[idx] for idx in top_k_indices]
        retrieved_df['rrf_score'] = rrf_list
        
        # Normalize RRF Score (0 to 1) for the formula
        max_rrf = retrieved_df['rrf_score'].max()
        if max_rrf > 0:
            retrieved_df['rrf_score_norm'] = retrieved_df['rrf_score'] / max_rrf
        else:
            retrieved_df['rrf_score_norm'] = 0.0
            
        self.log_metric("Hybrid Retrieval (RRF)", start_time, start_mem)
        return retrieved_df

    def score_sections(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Sectional Listwise Cross-Encoder (Real Inference)")
        start_time = time.time()
        start_mem = get_memory_usage()
        
        logging.info("  -> Scoring Career...")
        career_pairs = [[self.parsed_jd["career"], str(text)] for text in df['text_career']]
        df['ce_career'] = self.cross_encoder.predict(career_pairs)
        
        logging.info("  -> Scoring Skills...")
        skills_pairs = [[self.parsed_jd["skills"], str(text)] for text in df['text_skills']]
        df['ce_skills'] = self.cross_encoder.predict(skills_pairs)
        
        logging.info("  -> Scoring Profile...")
        profile_pairs = [[self.parsed_jd["profile"], str(text)] for text in df['text_profile']]
        df['ce_profile'] = self.cross_encoder.predict(profile_pairs)
        
        logging.info("  -> Scoring Education...")
        edu_pairs = [[self.parsed_jd["education"], str(text)] for text in df['text_edu']]
        df['ce_edu'] = self.cross_encoder.predict(edu_pairs)
        
        # Normalize cross-encoder scores
        for col in ['ce_career', 'ce_skills', 'ce_profile', 'ce_edu']:
            df[col] = 1 / (1 + np.exp(-df[col]))
            
        self.log_metric("Cross-Encoder Inference (L-12-v2)", start_time, start_mem)
        return df

    def combine_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Final Math Formula (Combining Text + Structural Constraints)")
        start_time = time.time()
        start_mem = get_memory_usage()
        
        # 1. Structural Integrity Penalty
        trust_penalty = df[['trust_date_violation', 'trust_salary_violation']].max(axis=1)
        
        # 2. Extract Constraints
        c = self.parsed_jd["constraints"]
        weights = c["weights"]
        
        # Hard Structural Penalties based on JSON constraints
        exp_penalty = (df['years_of_experience'] < c.get("exp_min", 0)).astype(float) * 0.15
        notice_penalty = (df['notice_period'] > c.get("notice_max", 90)).astype(float) * 0.05
        
        total_structural_penalty = (0.10 * trust_penalty) + exp_penalty + notice_penalty
        
        # 3. Final Formula using Dynamic JSON Weights
        df['final_score'] = (
            (weights.get('career', 0.35) * df['ce_career']) + 
            (weights.get('skills', 0.25) * df['ce_skills']) + 
            (weights.get('profile', 0.20) * df['ce_profile']) + 
            (weights.get('education', 0.05) * df['ce_edu']) + 
            (0.15 * df['rrf_score_norm'])
        ) - total_structural_penalty
        
        df = df.sort_values(by=['final_score', 'candidate_id'], ascending=[False, True])
        
        self.log_metric("Math Scoring & Sorting", start_time, start_mem)
        return df

    def run(self, output_csv: str = "Trails/Trail_4/submission.csv"):
        total_start = time.time()
        
        self.load_artifacts()
        
        retrieved_df = self.retrieve_hybrid()
        scored_df = self.score_sections(retrieved_df)
        final_df = self.combine_scores(scored_df)
        
        logging.info("Reasoning Generation")
        reasonings = []
        for _, row in final_df.iterrows():
            score = row['final_score']
            trust = "Clear structural integrity." if row['trust_date_violation'] == 0 else "WARNING: Structural anomalies detected."
            r = f"Score: {score:.3f}. Career match: {row['ce_career']:.2f}, Skills match: {row['ce_skills']:.2f}. {trust}"
            reasonings.append(r)
            
        final_df['reasoning'] = reasonings
        
        submission = final_df[['candidate_id', 'final_score', 'reasoning', 'rrf_score']].copy()
        submission['rank'] = range(1, len(submission) + 1)
        submission = submission[['candidate_id', 'rank', 'final_score', 'reasoning', 'rrf_score']]
        submission = submission.rename(columns={'final_score': 'score'})
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        submission.to_csv(output_csv, index=False)
        
        with open("Trails/Trail_4/metrics_report.txt", "w") as f:
            f.write("=== Trail 4 Resource Profile (Hybrid + Sectional L-12-v2) ===\n")
            for m in self.metrics:
                f.write(m + "\n")
            f.write(f"\nTotal Pipeline Execution: {time.time() - total_start:.2f} seconds\n")
            
        logging.info(f"Done! Saved to {output_csv} in {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    ranker = Trail4Ranker(top_k=5000)
    ranker.run()
