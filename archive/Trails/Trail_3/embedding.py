import json
import logging
from typing import Dict, Any, List, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    # Note: For production SPLADE, we would use a library like splade or pyserini,
    # but for simplicity and constraints, we can use sentence-transformers if it supports the architecture,
    # or rely on transformers directly. For the hackathon context, we will mock SPLADE if missing 
    # or use a standard token expansion strategy if we can't install specific splade packages.
    # A standard alternative for sparse retrieval within sentence-transformers is not native,
    # so we'll structure this to accept the BGE dense model first.
except ImportError:
    SentenceTransformer = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CandidateEmbedder:
    """
    Handles parsing a candidate into a unified text representation 
    and generating embeddings (Dense + Sparse/SPLADE).
    """
    def __init__(self, dense_model_name: str = "BAAI/bge-small-en-v1.5", use_splade: bool = False):
        self.dense_model_name = dense_model_name
        self.use_splade = use_splade
        
        if SentenceTransformer:
            logging.info(f"Loading dense model: {dense_model_name}")
            self.dense_model = SentenceTransformer(dense_model_name)
            
            if self.use_splade:
                logging.warning("SPLADE initialization requires specific transformers setup. " 
                                "Ensure naver/splade-cocondenser-ensembledistil is available.")
                # self.splade_model = ... # Initialize SPLADE here
        else:
            logging.error("SentenceTransformers not installed. Cannot load models.")
            self.dense_model = None

    def candidate_to_text(self, candidate: Dict[str, Any]) -> str:
        """
        Converts the candidate JSON into a dense text representation designed 
        specifically to maximize retrieval signal across the 44 templates and skills.
        """
        profile = candidate.get('profile', {})
        headline = profile.get('headline', '')
        summary = profile.get('summary', '')
        
        history = candidate.get('career_history', [])
        # Extract the most recent/relevant career description
        career_desc = ""
        if history:
            career_desc = history[0].get('description', '')
            
        skills = candidate.get('skills', [])
        # Only take top skills (advanced/expert or high duration) to reduce noise
        top_skills = [s.get('name', '') for s in skills if s.get('proficiency') in ['advanced', 'expert']][:15]
        skills_text = ", ".join(top_skills)
        
        # We combine them in a way that gives context to the embedder.
        text = f"Headline: {headline}. Summary: {summary}. Recent Experience: {career_desc}. Key Skills: {skills_text}."
        
        return text

    def embed_dense(self, texts: List[str]) -> np.ndarray:
        """Generate dense embeddings for a list of texts."""
        if not self.dense_model:
            raise RuntimeError("Dense model not loaded.")
        
        # Normalize embeddings for cosine similarity via dot product later
        embeddings = self.dense_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        return embeddings
        
    def embed_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Generate sparse (SPLADE) representations.
        Returns a list of dicts {token: weight} for inverted index building.
        """
        if not self.use_splade:
            return [{}] * len(texts)
            
        # Placeholder for SPLADE logic
        # inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        # outputs = splade_model(**inputs)
        # # ... pooling and token mapping ...
        return [{}] * len(texts)

if __name__ == "__main__":
    import os
    sample_path = "../candidates.jsonl" if not os.path.exists("Sample/sample_candidates.json") else "Sample/sample_candidates.json"
    
    try:
        if sample_path.endswith(".jsonl"):
            with open(sample_path, 'r', encoding='utf-8') as f:
                c = json.loads(f.readline())
        else:
            with open(sample_path, 'r', encoding='utf-8') as f:
                c = json.load(f)[0]
                
        embedder = CandidateEmbedder()
        text_rep = embedder.candidate_to_text(c)
        print("--- Text Representation ---")
        print(text_rep)
        print("\nGenerating embedding...")
        emb = embedder.embed_dense([text_rep])
        print(f"Embedding shape: {emb.shape}")
    except Exception as e:
        print(f"Could not run test: {e}")
