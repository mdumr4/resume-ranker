import os
from pathlib import Path
import pytest
import pandas as pd
from src.pipeline import CandidateRankingPipeline
from src.config import PipelineConfig
from src.validator import validate_submission_df


def test_pipeline_dry_run_fast():
    """
    Dry run integration test: executes Hybrid Retrieval + Two-Part Cross-Encoder
    + Deterministic Fast Rationales, validating output schema and monotonicity.
    """
    tmp_out = Path(__file__).resolve().parent / "tmp_test_submission.csv"
    if tmp_out.exists():
        tmp_out.unlink()

    try:
        pipeline = CandidateRankingPipeline()
        df = pipeline.run(output_csv=tmp_out, skip_llm=True, validate=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert list(df.columns) == ["candidate_id", "rank", "score", "reasoning"]
        assert df["score"].is_monotonic_decreasing

        # Check file on disk
        assert tmp_out.exists()
        errors = validate_submission_df(df)
        assert len(errors) == 0, f"Validation errors found: {errors}"

    finally:
        if tmp_out.exists():
            tmp_out.unlink()
