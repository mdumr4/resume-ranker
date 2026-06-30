import os
import sys
import logging
import time
import psutil
import pandas as pd
import numpy as np
import torch
from scipy.sparse import csr_matrix, load_npz
from sentence_transformers import CrossEncoder
from transformers import AutoModelForMaskedLM, AutoTokenizer
from jd_parser import JDParser

# Set HF to offline
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

class RankerTest500:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.parsed_jd = {}
        
    def load_models(self):
        logging.info("Parsing Pre-processed JD...")
        parser = JDParser("context/parsed_jd.json")
        self.parsed_jd = parser.parse()
        
        logging.info("Loading tabular features...")
        self.tabular_features = pd.read_parquet("index/features.parquet")
        
        logging.info("Loading SPLADE (Offline)...")
        splade_path = "naver/splade-v3"
        self.splade_tokenizer = AutoTokenizer.from_pretrained(splade_path, local_files_only=True)
        self.splade_model = AutoModelForMaskedLM.from_pretrained(splade_path, local_files_only=True).to(self.device)
        self.splade_model.eval()
        
        logging.info("Loading SPLADE matrix...")
        self.sparse_matrix = load_npz("index/splade_index.npz")
        
        logging.info("Loading Cross-Encoder (Offline)...")
        ce_model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
        self.cross_encoder = CrossEncoder(ce_model_name, device=self.device, max_length=512, local_files_only=True)
        
    def retrieve_splade_only(self, top_k=500):
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
        
        top_indices = sparse_indices[:top_k]
        top_scores = sparse_scores[top_indices]
        
        df = self.tabular_features.iloc[top_indices].copy()
        
        # Normalize SPLADE score (Min-Max)
        if len(top_scores) > 1 and top_scores.max() != top_scores.min():
            df['splade_score_norm'] = (top_scores - top_scores.min()) / (top_scores.max() - top_scores.min())
        else:
            df['splade_score_norm'] = 1.0
            
        return df
        
    def score_sections(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Sectional Listwise Cross-Encoder (Real Inference on 500)")
        
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
        
        # Normalize cross-encoder scores (Sigmoid)
        for col in ['ce_career', 'ce_skills', 'ce_profile', 'ce_edu']:
            df[col] = 1 / (1 + np.exp(-df[col]))
            
        return df
        
    def combine_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Final Math Formula (Combining Text + Structural Constraints)")
        
        trust_penalty = df[['trust_date_violation', 'trust_salary_violation']].max(axis=1)
        
        c = self.parsed_jd["constraints"]
        weights = c["weights"]
        
        exp_penalty = (df['years_of_experience'] < c.get("exp_min", 0)).astype(float) * 0.15
        notice_penalty = (df['notice_period'] > c.get("notice_max", 90)).astype(float) * 0.05
        total_structural_penalty = (0.10 * trust_penalty) + exp_penalty + notice_penalty
        
        # Note: Using splade_score_norm instead of rrf_score_norm
        df['final_score'] = (
            (weights.get('career', 0.35) * df['ce_career']) + 
            (weights.get('skills', 0.25) * df['ce_skills']) + 
            (weights.get('profile', 0.20) * df['ce_profile']) + 
            (weights.get('education', 0.05) * df['ce_edu']) + 
            (0.15 * df['splade_score_norm'])
        ) - total_structural_penalty
        
        df = df.sort_values(by=['final_score', 'candidate_id'], ascending=[False, True])
        
        # Build Reasoning
        def build_reasoning(row):
            parts = [f"Score: {row['final_score']:.3f}"]
            parts.append(f"Career match: {row['ce_career']:.2f}")
            parts.append(f"Skills match: {row['ce_skills']:.2f}")
            
            if row['trust_date_violation'] or row['trust_salary_violation']:
                parts.append("Warning: structural inconsistencies detected")
            else:
                parts.append("Clear structural integrity")
            return ". ".join(parts) + "."
            
        df['reasoning'] = df.apply(build_reasoning, axis=1)
        return df

if __name__ == "__main__":
    ranker = RankerTest500()
    ranker.load_models()
    
    df_top_500 = ranker.retrieve_splade_only(top_k=500)
    df_scored = ranker.score_sections(df_top_500)
    df_final = ranker.combine_scores(df_scored)
    
    df_final['rank'] = range(1, len(df_final) + 1)
    
    output_path = "Trails/Trail_4/test_500_submission.csv"
    df_final[['candidate_id', 'rank', 'final_score', 'reasoning']].to_csv(output_path, index=False)
    logging.info(f"Test saved to {output_path}")
    
    # Compare with Trail 2
    t2 = pd.read_csv('Trails/Trail_2/submission.csv')
    t4 = pd.read_csv(output_path)
    
    t2_top100 = set(t2['candidate_id'].head(100))
    t4_top100 = set(t4['candidate_id'].head(100))
    
    overlap = len(t4_top100.intersection(t2_top100))
    print(f"\n--- COMPARISON ---")
    print(f"Test SPLADE+CE (Top 500) vs Trail 2 Top 100 Overlap: {overlap}")
    print(f"\nTop 10 from Test:")
    for i, row in t4.head(10).iterrows():
        print(f"{row['rank']}. ID: {row['candidate_id']} | Score: {row['final_score']:.3f} | {row['reasoning']}")
