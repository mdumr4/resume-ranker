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
    """Master orchestrator for the Redrob 3-Stage Candidate Discovery & Ranking Pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or default_config
        self.retriever = HybridRetriever(self.config)
        self.reranker = TwoPartCrossEncoder(self.config)
        self.reasoner = LLMReasoner(self.config)

    def run(
        self,
        output_csv: Union[str, Path] = "submission.csv",
        skip_llm: bool = False,
        validate: bool = True,
    ) -> pd.DataFrame:
        """Executes the end-to-end ranking workflow."""
        global_t0 = time.time()
        logger.info("==================================================")
        logger.info("   REDROB PRODUCTION CANDIDATE RANKING PIPELINE   ")
        logger.info("==================================================")

        # Stage 0: Load Assets
        self.retriever.load_index()

        # Stage 1: Hybrid Retrieval
        top_candidates = self.retriever.retrieve(top_n=self.config.hybrid_top_k)

        # Stage 2: Two-Part Cross-Encoder Reranking
        reranked_candidates = self.reranker.rerank(top_candidates, self.retriever.df)

        # Stage 3: LLM Reasoning
        submission_records = self.reasoner.generate_rationales(
            reranked_candidates[:100],
            self.retriever.df,
            skip_llm=skip_llm,
        )

        # Stage 4: Assembly & Validation
        df_sub = pd.DataFrame(submission_records)
        df_sub = df_sub[["candidate_id", "rank", "score", "reasoning"]]

        # Enforce exact monotonic non-increasing scores if floats collided
        for i in range(len(df_sub) - 1):
            if df_sub.loc[i, "score"] < df_sub.loc[i + 1, "score"]:
                df_sub.loc[i + 1, "score"] = df_sub.loc[i, "score"]

        # Validate
        if validate:
            errors = validate_submission_df(df_sub)
            if errors:
                logger.error(f"Validation failed with {len(errors)} issues:")
                for err in errors:
                    logger.error(f"  - {err}")
                raise ValueError(f"Submission failed validation: {errors[0]}")
            logger.info("Submission passed all schema and monotonic checks flawlessly!")

        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_sub.to_csv(out_path, index=False)

        total_time = time.time() - global_t0
        logger.info(f"Pipeline finished in {total_time:.1f}s ({total_time / 60:.2f} min).")
        logger.info(f"Successfully saved final submission to {out_path.resolve()}")
        return df_sub
