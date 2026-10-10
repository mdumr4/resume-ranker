import time
import logging
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.special import expit
from sentence_transformers import CrossEncoder
from src.config import PipelineConfig, default_config

logger = logging.getLogger("reranker")


class TwoPartCrossEncoder:
    """
    Reranks candidates using a two-part cross-encoder evaluation:
    - Tech Match: Evaluates candidate career and skills against tech requirements.
    - Logistics Match: Evaluates location, notice period, work mode, and profile.
    Applies Sigmoid (expit) normalization and combines additively:
    Final Score = 0.70 * P(Tech) + 0.30 * P(Logistics)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or default_config
        self.model: Optional[CrossEncoder] = None

    def load_model(self):
        """Loads the cross-encoder model."""
        logger.info(f"Loading Cross-Encoder model: {self.config.cross_encoder_path}")
        t0 = time.time()
        self.model = CrossEncoder(
            self.config.cross_encoder_path,
            device="cpu",
            local_files_only=True
        )
        logger.info(f"Cross-Encoder loaded in {time.time() - t0:.2f}s")

    def rerank(
        self, candidates: List[Tuple[str, float]], df: pd.DataFrame
    ) -> List[Tuple[str, float]]:
        """Evaluates and reranks the candidate pool using two-part cross-encoder fusion."""
        if self.model is None:
            self.load_model()

        t0 = time.time()
        logger.info(f"Reranking {len(candidates)} candidates via Two-Part Cross-Encoder...")

        ce_inputs_tech = []
        ce_inputs_logistics = []
        cids_in_order = []

        # Index lookup map for fast retrieval
        df_map = df.set_index("candidate_id")

        for cid, _ in candidates:
            if cid not in df_map.index:
                continue
            row = df_map.loc[cid]
            text_tech = f"{row.get('text_career', '')} {row.get('text_skills', '')}"
            text_logistics = f"{row.get('text_profile', '')} {row.get('text_edu', '')}"

            ce_inputs_tech.append((self.config.jd_tech, text_tech))
            ce_inputs_logistics.append((self.config.jd_logistics, text_logistics))
            cids_in_order.append(cid)

        # Predict in configured batches
        batch_size = self.config.ce_batch_size
        preds_tech = self.model.predict(ce_inputs_tech, batch_size=batch_size, show_progress_bar=False)
        preds_logistics = self.model.predict(ce_inputs_logistics, batch_size=batch_size, show_progress_bar=False)

        # Sigmoid normalization
        prob_tech = expit(preds_tech)
        prob_logistics = expit(preds_logistics)

        # Weighted additive combination
        w_tech = self.config.tech_weight
        w_logistics = self.config.logistics_weight
        final_scores = (w_tech * prob_tech) + (w_logistics * prob_logistics)

        results = list(zip(cids_in_order, [float(s) for s in final_scores]))
        results.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Stage 2 Two-Part Cross-Encoder complete in {time.time() - t0:.2f}s")
        return results
