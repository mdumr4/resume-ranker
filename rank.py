#!/usr/bin/env python3
"""
Redrob Candidate Discovery & Resume Ranker - Production CLI Entry Point.
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
        description="Redrob Candidate Discovery & Resume Ranker - Production Pipeline"
    )
    parser.add_argument(
        "--candidates",
        default="./data/candidates.jsonl",
        help="Path to raw candidates JSON or JSONL file (default: ./data/candidates.jsonl)",
    )
    parser.add_argument(
        "--out",
        default="./submission.csv",
        help="Output path for submission CSV (default: ./submission.csv)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM generation and use deterministic rationales (fast mode)",
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
