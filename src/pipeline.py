import time
import logging
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from src.config import PipelineConfig, default_config
from src.retriever import HybridRetriever
from src.reranker import TwoPartCrossEncoder
from src.reasoner import LLMReasoner
from src.validator import validate_submission_df

logger = logging.getLogger("pipeline")


class CandidateRankingPipeline:
    """Master orchestrator for the Production Candidate Discovery & Ranking Pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or default_config
        self.retriever = HybridRetriever(self.config)
        self.reranker = TwoPartCrossEncoder(self.config)
        self.reasoner = LLMReasoner(self.config)

    def run(
        self,
        output_csv: Union[str, Path] = "output/ranked_candidates.csv",
        skip_llm: bool = False,
        validate: bool = True,
    ) -> pd.DataFrame:
        """Executes the end-to-end ranking workflow."""
        global_t0 = time.time()
        logger.info("==================================================")
        logger.info("      PRODUCTION CANDIDATE RANKING PIPELINE       ")
        logger.info("==================================================")

        # Stage 0: Load Precomputed Indices & Metadata
        self.retriever.load_index()

        # Stage 1: Vectorized Hybrid Retrieval (FAISS + SPLADE + Trust-Weighted RRF)
        top_candidates = self.retriever.retrieve(top_n=self.config.hybrid_top_k)

        # Stage 2: Two-Part Cross-Encoder Semantic Reranking (70% Tech, 30% Logistics)
        reranked_candidates = self.reranker.rerank(top_candidates, self.retriever.df)

        # Stage 3: Explainable AI Rationales (Continuous Batching or Deterministic Fallback)
        ranked_records = self.reasoner.generate_rationales(
            reranked_candidates[:100],
            self.retriever.df,
            skip_llm=skip_llm,
        )

        # Stage 4: Assembly & Monotonic Schema Validation
        df_ranked = pd.DataFrame(ranked_records)
        df_ranked = df_ranked[["candidate_id", "rank", "score", "reasoning"]]

        # Enforce exact monotonic non-increasing scores if precision collisions occur
        for i in range(len(df_ranked) - 1):
            if df_ranked.loc[i, "score"] < df_ranked.loc[i + 1, "score"]:
                df_ranked.loc[i + 1, "score"] = df_ranked.loc[i, "score"]

        # Validation Gate
        if validate:
            errors = validate_submission_df(df_ranked)
            if errors:
                logger.error(f"Validation failed with {len(errors)} issues:")
                for err in errors:
                    logger.error(f"  - {err}")
                raise ValueError(f"Ranking validation failed: {errors[0]}")
            logger.info("Ranking passed all schema and monotonic checks flawlessly!")

        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_ranked.to_csv(out_path, index=False)

        total_time = time.time() - global_t0
        logger.info(f"Pipeline finished in {total_time:.1f}s ({total_time / 60:.2f} min).")
        logger.info(f"Successfully saved ranked results to {out_path.resolve()}")
        return df_ranked
