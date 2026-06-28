import logging
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Suppress verbose warnings
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

try:
    import faiss
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    logging.warning("Missing dependencies. Run: uv add sentence-transformers faiss-cpu pandas")

class RankingPipeline:
    def __init__(self, index_dir="Trails/Trail_1/index"):
        self.index_dir = Path(index_dir)
        
        self.faiss_index = None
        self.tabular_features = None
        
        self.embedder = None
        self.cross_encoder = None
        
        # Parsed JD (Hardcoded for this challenge instead of calling LLM at query time)
        self.parsed_jd = {
            "career": "5-9 years experience, building ranking models, search systems, machine learning pipelines.",
            "skills": "Python, Machine Learning, Recommendation Systems, XGBoost, PyTorch, FAISS.",
            "profile": "Fast-paced startup environment, autonomous, high ownership. Location: Pune, Noida. Mode: Hybrid. Budget: 15-40 LPA.",
            "education": "Bachelors in CS or related field. Real-world experience matters more."
        }
        
    def load_artifacts(self):
        logging.info("Loading pre-computed indices and models...")
        start = time.time()
        
        # Load FAISS and Features
        faiss_path = self.index_dir / "faiss.index"
        feat_path = self.index_dir / "features.parquet"
        
        if not faiss_path.exists() or not feat_path.exists():
            raise FileNotFoundError("FAISS or features.parquet missing. Run index_builder.py first.")
            
        self.faiss_index = faiss.read_index(str(faiss_path))
        self.tabular_features = pd.read_parquet(feat_path)
        
        logging.info("Loading Dense Embedder (BAAI/bge-small)...")
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        logging.info("Loading Cross-Encoder (ms-marco-MiniLM)...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            
        logging.info(f"Loaded artifacts in {time.time() - start:.2f} seconds")

    def retrieve(self, top_k: int = 100) -> pd.DataFrame:
        """Dense Retrieval using FAISS."""
        logging.info("Step 1: Dense Retrieval")
        # Create a holistic string for the JD
        jd_dense_string = " ".join(self.parsed_jd.values())
        
        jd_emb = self.embedder.encode([jd_dense_string], normalize_embeddings=True)
        
        search_k = min(top_k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(jd_emb, search_k)
        
        # Get the retrieved candidates
        retrieved_df = self.tabular_features.iloc[indices[0]].copy()
        retrieved_df['dense_score'] = scores[0]
        
        return retrieved_df

    def score_sections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs the REAL cross-encoder for each section."""
        logging.info("Step 2: Sectional Listwise Cross-Encoder (Real Inference)")
        
        # 1. Career
        logging.info("  -> Scoring Career...")
        career_pairs = [[self.parsed_jd["career"], text] for text in df['text_career']]
        df['ce_career'] = self.cross_encoder.predict(career_pairs)
        
        # 2. Skills
        logging.info("  -> Scoring Skills...")
        skills_pairs = [[self.parsed_jd["skills"], text] for text in df['text_skills']]
        df['ce_skills'] = self.cross_encoder.predict(skills_pairs)
        
        # 3. Profile
        logging.info("  -> Scoring Profile...")
        profile_pairs = [[self.parsed_jd["profile"], text] for text in df['text_profile']]
        df['ce_profile'] = self.cross_encoder.predict(profile_pairs)
        
        # 4. Education
        logging.info("  -> Scoring Education...")
        edu_pairs = [[self.parsed_jd["education"], text] for text in df['text_edu']]
        df['ce_edu'] = self.cross_encoder.predict(edu_pairs)
        
        # Normalize cross-encoder scores (they are logits, usually between -10 and 10)
        # We apply sigmoid to convert to 0-1 probability
        for col in ['ce_career', 'ce_skills', 'ce_profile', 'ce_edu']:
            df[col] = 1 / (1 + np.exp(-df[col]))
            
        return df

    def combine_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the final weighted mathematical formula."""
        logging.info("Step 3: Final Math Formula (Combining Text + Structural)")
        
        # Trust Penalty
        # If date violation or salary violation, penalty = 1.0
        trust_penalty = df[['trust_date_violation', 'trust_salary_violation']].max(axis=1)
        
        # Final Formula
        df['final_score'] = (
            (0.35 * df['ce_career']) + 
            (0.25 * df['ce_skills']) + 
            (0.20 * df['ce_profile']) + 
            (0.05 * df['ce_edu']) + 
            (0.15 * df['dense_score'])
        ) - (0.10 * trust_penalty)
        
        # Sort by final score (desc) and candidate_id (asc)
        df = df.sort_values(by=['final_score', 'candidate_id'], ascending=[False, True])
        return df

    def run(self, output_csv: str = "Trails/Trail_1/submission.csv"):
        start = time.time()
        
        self.load_artifacts()
        
        # 1. Retrieval
        retrieved_df = self.retrieve(top_k=100) # Fetch top 100 for Cross-Encoder
        
        # 2. Sectional Scoring
        scored_df = self.score_sections(retrieved_df)
        
        # 3. Final Formula
        final_df = self.combine_scores(scored_df)
        
        # 4. Generate Reasoning
        logging.info("Step 4: Reasoning Generation")
        reasonings = []
        for _, row in final_df.iterrows():
            score = row['final_score']
            trust = "Clear structural integrity." if row['trust_date_violation'] == 0 else "WARNING: Structural anomalies detected."
            r = f"Score: {score:.3f}. Career match: {row['ce_career']:.2f}, Skills match: {row['ce_skills']:.2f}. {trust}"
            reasonings.append(r)
            
        final_df['reasoning'] = reasonings
        
        # 5. Format Submission
        submission = final_df[['candidate_id', 'final_score', 'reasoning']].copy()
        submission['rank'] = range(1, len(submission) + 1)
        submission = submission[['candidate_id', 'rank', 'final_score', 'reasoning']]
        submission = submission.rename(columns={'final_score': 'score'})
        
        submission.to_csv(output_csv, index=False)
        logging.info(f"Pipeline finished in {time.time() - start:.2f} seconds. Output saved to {output_csv}")

if __name__ == "__main__":
    pipeline = RankingPipeline()
    pipeline.run("submission.csv")
