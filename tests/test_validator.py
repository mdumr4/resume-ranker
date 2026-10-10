import pandas as pd
import pytest
from src.validator import validate_submission_df


def test_valid_submission():
    rows = []
    for i in range(1, 101):
        cid = f"CAND_{i:07d}"
        score = 1.0 - (i * 0.005)
        rows.append({
            "candidate_id": cid,
            "rank": i,
            "score": score,
            "reasoning": "Selected for strong relevance to the JD. Can improve in sparse retrieval."
        })
    df = pd.DataFrame(rows)
    errors = validate_submission_df(df)
    assert len(errors) == 0, f"Expected 0 errors, got: {errors}"


def test_invalid_row_count():
    rows = [{"candidate_id": f"CAND_{i:07d}", "rank": i, "score": 0.5, "reasoning": "Good fit."} for i in range(1, 51)]
    df = pd.DataFrame(rows)
    errors = validate_submission_df(df)
    assert any("Expected exactly 100 data rows" in e for e in errors)


def test_invalid_candidate_id_format():
    rows = [{"candidate_id": f"INVALID_{i}", "rank": i, "score": 1.0 - i * 0.001, "reasoning": "Reason"} for i in range(1, 101)]
    df = pd.DataFrame(rows)
    errors = validate_submission_df(df)
    assert any("invalid format" in e for e in errors)


def test_non_monotonic_scores():
    rows = [{"candidate_id": f"CAND_{i:07d}", "rank": i, "score": 0.5, "reasoning": "Good fit."} for i in range(1, 101)]
    # Break monotonicity
    rows[10]["score"] = 0.9
    df = pd.DataFrame(rows)
    errors = validate_submission_df(df)
    assert any("Score must be non-increasing" in e for e in errors)
