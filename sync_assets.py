#!/usr/bin/env python3
"""
Asset Synchronization Utility for Resume Ranker.
Downloads precomputed indices, model weights, and datasets directly from
the Hugging Face Bucket: hf://buckets/mdumr4/Resume-Ranker
"""

import sys
import os
import argparse
import subprocess
import shutil
from pathlib import Path

BUCKET_URI = "hf://buckets/mdumr4/Resume-Ranker"
REPO_ROOT = Path(__file__).resolve().parent


def run_cmd(cmd_list):
    """Executes a command and streams output."""
    cmd_str = " ".join(cmd_list)
    print(f"Running: {cmd_str}")
    res = subprocess.run(cmd_list, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        print(f"Command failed with exit code {res.returncode}")
        return False
    return True


def check_hf_cli():
    """Checks if hf CLI is installed."""
    if shutil.which("hf") is None:
        print("Error: `hf` CLI is not installed or not in PATH.")
        print("Install it via: pip install huggingface_hub[cli]")
        print("And login via: hf auth login")
        return False
    return True


def sync_indices():
    """Syncs index files from bucket."""
    print("\n=== Syncing Vector Indices (FAISS, SPLADE, Features) ===")
    os.makedirs(REPO_ROOT / "index", exist_ok=True)
    cmd = ["hf", "buckets", "cp", "-r", f"{BUCKET_URI}/index", str(REPO_ROOT / "index")]
    return run_cmd(cmd)


def sync_models():
    """Syncs model weights from bucket."""
    print("\n=== Syncing Model Weights (Cross-Encoder, GGUF, ONNX) ===")
    os.makedirs(REPO_ROOT / "models", exist_ok=True)
    cmd = ["hf", "buckets", "cp", "-r", f"{BUCKET_URI}/models", str(REPO_ROOT / "models")]
    return run_cmd(cmd)


def sync_data():
    """Syncs raw candidates.jsonl dataset from bucket."""
    print("\n=== Syncing Dataset (candidates.jsonl) ===")
    os.makedirs(REPO_ROOT / "data", exist_ok=True)
    cmd = ["hf", "buckets", "cp", f"{BUCKET_URI}/data/candidates.jsonl", str(REPO_ROOT / "data" / "candidates.jsonl")]
    return run_cmd(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Sync assets from Hugging Face Bucket (hf://buckets/mdumr4/Resume-Ranker)"
    )
    parser.add_argument("--all", action="store_true", help="Download all assets (index, models, and data)")
    parser.add_argument("--index", action="store_true", help="Download vector indices (FAISS, SPLADE, features)")
    parser.add_argument("--models", action="store_true", help="Download model weights (GGUF, ONNX, cross-encoder)")
    parser.add_argument("--data", action="store_true", help="Download raw candidates.jsonl dataset")
    args = parser.parse_args()

    if not check_hf_cli():
        sys.exit(1)

    if not (args.all or args.index or args.models or args.data):
        args.all = True

    success = True
    if args.all or args.index:
        success = sync_indices() and success

    if args.all or args.models:
        success = sync_models() and success

    if args.all or args.data:
        success = sync_data() and success

    if success:
        print("\nAll requested assets synced successfully!")
    else:
        print("\nSome assets failed to sync. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
