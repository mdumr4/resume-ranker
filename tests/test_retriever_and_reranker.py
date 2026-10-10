import numpy as np
import pytest
from scipy.special import expit


def test_rrf_formula():
    k = 60
    dense_rank = 1
    sparse_rank = 1
    rrf_top = (1.0 / (k + dense_rank)) + (1.0 / (k + sparse_rank))
    assert pytest.approx(rrf_top, 0.0001) == 2.0 / 61.0

    dense_rank_low = 100
    sparse_rank_low = 100
    rrf_low = (1.0 / (k + dense_rank_low)) + (1.0 / (k + sparse_rank_low))
    assert rrf_top > rrf_low


def test_two_part_sigmoid_weighting():
    # Model logits for Tech and Logistics
    logits_tech = np.array([2.5, -1.0, 0.0])
    logits_logistics = np.array([1.0, 1.5, 0.0])

    prob_tech = expit(logits_tech)
    prob_logistics = expit(logits_logistics)

    # 70% Tech, 30% Logistics
    final_scores = (0.70 * prob_tech) + (0.30 * prob_logistics)
    assert len(final_scores) == 3
    assert all(0.0 <= s <= 1.0 for s in final_scores)
    assert final_scores[0] > final_scores[1]
