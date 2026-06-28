import json
import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    logging.warning("Missing dependencies. Run: uv add sentence-transformers faiss-cpu pandas pyarrow")

from features import CandidateProcessor

class IndexBuilder:
    def __init__(self, output_dir: str = "index"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processor = CandidateProcessor()
        
        logging.info("Loading Dense Embedding Model...")
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
    def build_indices(self, jsonl_path: str, max_records: int = 500):
        """Builds both the tabular parquet file and the FAISS dense index."""
        logging.info(f"Starting index build from {jsonl_path} (Limit: {max_records})")
        
        features_list = []
        text_blocks_list = []
        dense_texts = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_records and i >= max_records:
                    break
                candidate = json.loads(line)
                
                # 1. Process structured and text features
                tab_feats, text_blocks = self.processor.process(candidate)
                features_list.append(tab_feats)
                text_blocks_list.append(text_blocks)
                
                # 2. Create the Dense String (concatenating all blocks for holistic search)
                dense_string = f"{text_blocks['career']} {text_blocks['skills']} {text_blocks['profile']} {text_blocks['education']}"
                dense_texts.append(dense_string)
                
        # --- Save Tabular Features ---
        logging.info("Saving Tabular features and raw text blocks to Parquet...")
        df = pd.DataFrame(features_list)
        
        # We also save the text blocks so we don't have to read the raw JSONL at query time!
        df['text_career'] = [t['career'] for t in text_blocks_list]
        df['text_skills'] = [t['skills'] for t in text_blocks_list]
        df['text_profile'] = [t['profile'] for t in text_blocks_list]
        df['text_edu'] = [t['education'] for t in text_blocks_list]
        
        df.to_parquet(self.output_dir / "features.parquet", index=False)
        logging.info(f"Saved {len(df)} records to {self.output_dir / 'features.parquet'}")
        
        # --- Build FAISS Dense Index ---
        logging.info("Generating Dense Embeddings (This may take a minute)...")
        embeddings = self.embedder.encode(dense_texts, normalize_embeddings=True, show_progress_bar=True)
        
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(np.array(embeddings, dtype=np.float32))
        
        faiss.write_index(faiss_index, str(self.output_dir / "faiss.index"))
        logging.info(f"Saved FAISS index to {self.output_dir / 'faiss.index'}")

if __name__ == "__main__":
    sample_path = "candidates.jsonl"
    
    if not os.path.exists(sample_path):
        logging.error(f"Cannot find {sample_path}")
    else:
        builder = IndexBuilder()
        builder.build_indices(sample_path, max_records=None)
