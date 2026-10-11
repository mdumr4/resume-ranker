import json
import pytest
from pathlib import Path
from src.preprocessor import CandidatePreprocessor, parse_date


@pytest.fixture
def preprocessor():
    return CandidatePreprocessor()


@pytest.fixture
def sample_candidate():
    sample_file = Path(__file__).resolve().parent.parent / "data" / "sample_candidates.json"
    with open(sample_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[0]


def test_parse_date():
    assert parse_date("2024-03-08") is not None
    assert parse_date("2024-03-08").year == 2024
    assert parse_date(None) is None
    assert parse_date("invalid-date") is None


def test_serialize_candidate(preprocessor, sample_candidate):
    res = preprocessor.serialize_candidate(sample_candidate)
    assert "candidate_id" in res
    assert res["candidate_id"] == sample_candidate["candidate_id"]
    assert "text_career" in res
    assert "text_profile" in res
    assert "text_skills" in res
    assert "text_edu" in res
    assert "trust_score" in res
    assert 0.1 <= res["trust_score"] <= 1.0


def test_trust_score_unverified_gate(preprocessor):
    unverified_sig = {
        "verified_email": False,
        "verified_phone": False,
        "profile_completeness_score": 100,
        "open_to_work_flag": True,
    }
    score = preprocessor.calculate_trust_score(unverified_sig, [], [])
    assert score == 0.10


def test_trust_score_career_gap_merging(preprocessor):
    verified_sig = {
        "verified_email": True,
        "verified_phone": True,
        "profile_completeness_score": 90,
        "open_to_work_flag": True,
    }
    # Two overlapping stints from 2018 to 2021
    career = [
        {"start_date": "2018-01-01", "end_date": "2020-01-01", "is_current": False},
        {"start_date": "2019-01-01", "end_date": "2021-01-01", "is_current": False},
    ]
    score = preprocessor.calculate_trust_score(verified_sig, career, [])
    assert 0.1 <= score <= 1.0
