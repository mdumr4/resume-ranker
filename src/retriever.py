import time
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import faiss
from src.config import PipelineConfig, default_config

logger = logging.getLogger("retriever")


class HybridRetriever:
    """
    Executes vectorized hybrid retrieval combining dense semantic search (FAISS)
    and sparse lexical search (SPLADE), fused via Reciprocal Rank Fusion (RRF)
    and modulated by candidate dynamic trust scores.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or default_config
        self.df: Optional[pd.DataFrame] = None
        self.faiss_index = None
        self.splade_matrix = None
        self.dense_query = None
        self.sparse_query = None

    def load_index(self):
        """Loads features, FAISS index, SPLADE sparse matrix, and precomputed query vectors."""
        logger.info("Loading precomputed data and vector indexes...")
        t0 = time.time()

        logger.info(f"  Reading features: {self.config.features_path}")
        self.df = pd.read_parquet(self.config.features_path)

        logger.info(f"  Reading FAISS index: {self.config.faiss_index_path}")
        self.faiss_index = faiss.read_index(self.config.faiss_index_path)

        logger.info(f"  Reading SPLADE matrix: {self.config.splade_index_path}")
        splade_data = np.load(self.config.splade_index_path, allow_pickle=True)
        self.splade_matrix = splade_data["splade_vectors"].item()

        logger.info(f"  Reading precomputed query vectors: {self.config.dense_query_path}")
        self.dense_query = np.load(self.config.dense_query_path)
        self.sparse_query = np.load(self.config.sparse_query_path)

        logger.info(f"All retrieval assets loaded in {time.time() - t0:.2f}s (Candidates pool: {len(self.df):,})")

    def dense_search(self, top_k: int = 10000) -> Dict[str, int]:
        """Runs FAISS search, shifts inner product score to [0, 1], multiplies by trust score, returns ranks."""
        if self.faiss_index is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")

        D, I = self.faiss_index.search(self.dense_query, top_k)
        candidates = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            row = self.df.iloc[idx]
            cid = row["candidate_id"]
            trust = float(row.get("trust_score", 1.0))
            shifted_score = (float(score) + 1.0) / 2.0
            candidates.append((cid, shifted_score * trust))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return {cid: rank + 1 for rank, (cid, _) in enumerate(candidates)}

    def sparse_search(self, top_k: int = 10000) -> Dict[str, int]:
        """Runs SPLADE sparse dot-product against CSR matrix, applies trust score, returns ranks."""
        if self.splade_matrix is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")

        scores = self.splade_matrix.dot(self.sparse_query)
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]

        candidates = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            cid = row["candidate_id"]
            trust = float(row.get("trust_score", 1.0))
            candidates.append((cid, float(scores[idx]) * trust))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return {cid: rank + 1 for rank, (cid, _) in enumerate(candidates)}

    def retrieve(self, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Executes dense and sparse searches and fuses ranks using Reciprocal Rank Fusion:
        RRF(c) = 1 / (k + rank_dense) + 1 / (k + rank_sparse)
        """
        if self.df is None:
            self.load_index()

        n = top_n or self.config.hybrid_top_k
        t0 = time.time()

        faiss_ranks = self.dense_search(self.config.faiss_top_k)
        t_faiss = time.time()
        logger.info(f"  FAISS Dense Search ({len(faiss_ranks):,} candidates): {t_faiss - t0:.2f}s")

        splade_ranks = self.sparse_search(self.config.splade_top_k)
        t_splade = time.time()
        logger.info(f"  SPLADE Sparse Search ({len(splade_ranks):,} candidates): {t_splade - t_faiss:.2f}s")

        # RRF Fusion
        all_cids = set(faiss_ranks.keys()).union(set(splade_ranks.keys()))
        k = self.config.rrf_k
        hybrid_scores = []

        for cid in all_cids:
            r_dense = 1.0 / (k + faiss_ranks[cid]) if cid in faiss_ranks else 0.0
            r_sparse = 1.0 / (k + splade_ranks[cid]) if cid in splade_ranks else 0.0
            hybrid_scores.append((cid, r_dense + r_sparse))

        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = hybrid_scores[:n]

        logger.info(f"Stage 1 Hybrid Retrieval complete in {time.time() - t0:.2f}s (Selected Top {len(top_candidates)})")
        return top_candidates
