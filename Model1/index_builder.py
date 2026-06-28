import json
import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np

# We'll use absolute/relative imports properly in production, 
# but for standalone execution we can just import from the local files.
try:
    from features import CandidateFeatureExtractor
    from embedding import CandidateEmbedder
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from features import CandidateFeatureExtractor
    from embedding import CandidateEmbedder

try:
    import faiss
except ImportError:
    faiss = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IndexBuilder:
    """
    Builds the offline indices and feature matrices from the candidates.jsonl file.
    Output:
      - features.parquet (Tabular features for LambdaMART)
      - embeddings.npy (Raw dense embeddings)
      - faiss.index (Dense FAISS index for retrieval)
    """
    def __init__(self, output_dir: str = "index"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = CandidateFeatureExtractor()
        # Ensure we only try to load embedder if sentence-transformers is ready
        self.embedder = CandidateEmbedder()
        
    def build_indices(self, input_filepath: str, max_records: int = None):
        logging.info(f"Starting index build from {input_filepath}")
        
        features_list = []
        texts = []
        candidate_ids = []
        
        # 1. Parse JSONL and Extract Features/Text
        count = 0
        with open(input_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                if max_records and count >= max_records: break
                
                c = json.loads(line)
                candidate_ids.append(c.get('candidate_id'))
                
                # Tabular features
                features_list.append(self.feature_extractor.extract_features(c))
                
                # Text for embedding
                texts.append(self.embedder.candidate_to_text(c))
                
                count += 1
                if count % 10000 == 0:
                    logging.info(f"Parsed {count} candidates...")
                    
        # 2. Save Tabular Features
        logging.info("Saving tabular features to Parquet...")
        df_features = pd.DataFrame(features_list)
        df_features.to_parquet(self.output_dir / "features.parquet", index=False)
        logging.info(f"Saved {len(df_features)} records to features.parquet")
        
        # 3. Generate Dense Embeddings
        # In a real environment, we would batch this.
        logging.info("Generating dense embeddings (this may take a while)...")
        batch_size = 512
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_emb = self.embedder.embed_dense(batch_texts)
            all_embeddings.append(batch_emb)
            if i % (batch_size * 5) == 0:
                logging.info(f"Embedded {i}/{len(texts)} candidates...")
                
        embeddings_matrix = np.vstack(all_embeddings)
        
        # Save raw embeddings
        np.save(self.output_dir / "embeddings.npy", embeddings_matrix)
        
        # 4. Build FAISS Index
        if faiss:
            logging.info("Building FAISS index...")
            dim = embeddings_matrix.shape[1]
            # Inner Product (Cosine similarity since embeddings are normalized)
            index = faiss.IndexFlatIP(dim) 
            
            # Since Candidate IDs are string (CAND_XXXXXXX), FAISS IndexFlat doesn't natively map strings well.
            # We use an IndexIDMap and map string IDs to integer indices (e.g. 0 to 99,999)
            # The mapping is preserved by the order in candidates_ids.json
            
            index.add(embeddings_matrix)
            faiss.write_index(index, str(self.output_dir / "faiss.index"))
            logging.info("FAISS index saved.")
        else:
            logging.error("FAISS not installed, skipping index generation.")
            
        # Save the ID mapping
        with open(self.output_dir / "candidate_ids.json", "w") as f:
            json.dump(candidate_ids, f)
            
        logging.info("Index build complete!")

if __name__ == "__main__":
    builder = IndexBuilder()
    
    # Check for test run
    sample_path = "../candidates.jsonl" if not os.path.exists("candidates.jsonl") else "candidates.jsonl"
    
    if os.path.exists(sample_path):
        # Build index for first 500 for testing
        builder.build_indices(sample_path, max_records=500)
    else:
        print(f"Could not find {sample_path}")
