# Antigravity AI - Redrob Challenge

Welcome to the **Antigravity AI** submission for the **India Runs Data & AI Challenge**. 

This repository contains an ultra-optimized, strictly local, CPU-bound ranking pipeline capable of sorting 100,000 candidate profiles and generating semantic LLM rationales in under 5 minutes.

## 🚀 Pipeline Architecture
Our pipeline uses a robust 3-Stage approach to balance blazing speed with extreme semantic accuracy:

1. **Stage 1: Hybrid Retrieval (Filtering 100K -> 100)**
   - Computes dense embeddings using **Qwen-Embeddings** (via FAISS).
   - Computes sparse token weights using **SPLADE**.
   - Fuses both metrics using **Reciprocal Rank Fusion (RRF)** combined with a behavioral Trust Score.
   - *Time:* ~12 seconds.
2. **Stage 2: Cross-Encoder Re-Ranking (Scoring Top 50)**
   - Passes the Top 50 candidates through an **INT8 Quantized BGE-M3 Cross-Encoder**.
   - Runs purely natively on ONNX runtime, evaluating deep cross-attention between the JD and candidate profiles without memory swapping.
   - *Time:* ~58 seconds.
3. **Stage 3: Generative Rationales (Reasoning Top 100)**
   - Boots a local **llama.cpp** server using a 4-bit quantized **Qwen2.5-0.5B-Instruct** GGUF.
   - Uses Continuous Batching to synchronously generate 100 pristine, human-sentence rationales simultaneously.
   - *Time:* ~178 seconds.

**Total Execution Time:** ~4.5 minutes (Passes the 5-minute strict budget!)

---

## 📁 Repository Structure

```text
├── archive/                         # Archived experimental code, trails, and unquantized models
├── bin/
│   └── llama/                       # Contains the llama-server.exe for local LLM continuous batching
├── context/                         # Original hackathon context files (rules, JD, specs)
├── index/                           # Pre-computed vectors (FAISS index, SPLADE NPZ, Parquet features)
├── models/                          
│   ├── bge_onnx_v2_int8/            # INT8 Quantized BGE-M3 Cross-Encoder
│   ├── splade_onnx_int8/            # INT8 Quantized SPLADE Document Encoder
│   └── qwen2.5-0.5b...gguf          # Q4 Quantized Qwen2.5 LLM for rationale generation
├── scripts/                         # Helper scripts for data pre-computation and environment setup
├── src/                             
│   ├── optimize_rank.py             # THE CORE RANKING PIPELINE 
│   └── clean_csv.py                 # Post-processing utilities
├── README.md                        # You are here!
├── run_pipeline.ps1                 # Single execution script to run the entire system
├── submission_metadata.yaml         # Official Hackathon submission metadata
└── validate_submission.py           # Provided validation script
```

---

## 🛠️ Reproduction Instructions

### 1. Prerequisites
- **OS:** Windows (or a system capable of running `powershell`).
- **Hardware:** 16 GB RAM (Strictly CPU-only execution).
- **Dependencies:** Python 3.12, `uv`. 

Install dependencies rapidly via `uv`:
```bash
uv sync
```

### 2. Execution
Run the following single command to execute the pipeline end-to-end. This script will automatically boot the local LLM server in the background, run the Python ranking scripts, generate the final CSV, and gracefully terminate the server upon completion.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
```

### 3. Output
The final ranked output will be automatically saved in the root directory as:
`submission.csv`

---

## 🧪 Validation

You can verify the strict compliance of the generated CSV using the official validator:
```bash
uv run python validate_submission.py submission.csv
```

*Note: The pipeline automatically generates synthetic descending scores (`1.0 - rank/1000`) at the end of the ranking process to perfectly conform to the validation rules while explicitly preserving the model's authoritative ranking.*

---

## 🔗 Sandbox Link
To test our ranking pipeline in an isolated, reproducible sandbox environment, please refer to the provided `colab_instructions.md` and `kaggle_instructions.md` located in the repository (these contain direct links to our hosted interactive sandboxes).
