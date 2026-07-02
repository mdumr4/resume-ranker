# Antigravity AI - Redrob Challenge

This repository contains the final submission for the India Runs Data & AI Challenge.

## Pipeline Architecture
Our ranking pipeline operates strictly within the 5-minute CPU constraint while effectively scoring 100,000 candidates and generating deep semantic rationales.

1. **Stage 1 (Hybrid Retrieval):** We compute FAISS Dense Embeddings (Qwen-Embeddings) and SPLADE Sparse Vectors, fusing them using Reciprocal Rank Fusion (RRF) to filter 100,000 candidates to a Top 100 list in ~10 seconds.
2. **Stage 2 (Cross-Encoder Re-Ranking):** We pass the Top 50 candidates through a heavy BGE-M3 Cross-Encoder. We quantized the ONNX graph to `INT8` to allow it to execute in ~115 seconds on a CPU without memory swapping.
3. **Stage 3 (LLM Rationales):** We run the highly efficient `Qwen2.5-0.5B-Instruct` model locally using `llama-server.exe` with Continuous Batching. It generates human-readable reasoning for all 100 candidates simultaneously in ~145 seconds.

## Reproduction Instructions

### 1. Prerequisites
- **OS:** Windows (or a system capable of running `powershell`).
- **Dependencies:** Python 3.12, `uv`. 

Ensure all dependencies are installed via `uv`:
```bash
uv sync
```

### 2. Execution
Run the following single command to execute the pipeline end-to-end:
```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
```

This script will automatically boot the local LLM server in the background, run the Python ranking scripts, generate the final CSV, and gracefully terminate the server upon completion.

### 3. Output
The final ranked output will be automatically saved in the root directory as:
`submission.csv`

## Pre-computed Artifacts
The pre-computed `index/` directory is required for Stage 1 execution and is included in this repository.

## Sandbox Link
To test our ranking pipeline in an isolated, reproducible sandbox environment, please refer to the provided `colab_instructions.md` and `kaggle_instructions.md` (these contain direct links to our hosted interactive sandboxes).
