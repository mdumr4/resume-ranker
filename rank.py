#!/usr/bin/env python3
"""
Production CLI Entry Point for Resume Ranker.
Ranks 100K+ candidate profiles against job requirements using Hybrid Retrieval
(FAISS + SPLADE + RRF), Two-Part Cross-Encoder semantic reranking, and
explainable rationale generation.
"""
import sys
import argparse
from pathlib import Path

# Add repo root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import CandidateRankingPipeline
from src.config import PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description="Resume Ranker - Production Multi-Stage Candidate Discovery & Ranking Engine"
    )
    parser.add_argument(
        "--candidates",
        default="./data/candidates.jsonl",
        help="Path to candidates dataset file (default: ./data/candidates.jsonl)",
    )
    parser.add_argument(
        "--out",
        default="./output/ranked_candidates.csv",
        help="Output CSV path for ranked candidates (default: ./output/ranked_candidates.csv)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM server and use deterministic fast rationales (runs in ~17s)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of CPU threads to allocate (default: auto-detected)",
    )
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.threads:
        cfg.num_threads = args.threads

    pipeline = CandidateRankingPipeline(cfg)
    pipeline.run(
        output_csv=args.out,
        skip_llm=args.skip_llm,
        validate=True,
    )


if __name__ == "__main__":
    main()
