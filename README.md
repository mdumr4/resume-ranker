# Redrob Candidate Discovery & Ranking Pipeline

Welcome to our submission for the **Redrob Talent Intelligence Hackathon**.

This repository contains an ultra-optimized, CPU-bound candidate discovery and ranking pipeline. It retrieves and re-ranks candidate profiles from a pool of **100,000 resumes**, applies dynamic behavioral/career gap penalties, and generates natural language reasoning rationales. The entire end-to-end process runs fully locally on a CPU in **~2.3 minutes** (well under the strict 5-minute hackathon constraint).

---

## 🚀 One-Command Setup

To run replication locally, you must first install the python packages and download the pre-computed indexes and local model weights. You can accomplish this in a **single command** (this pulls the pre-packaged indices and weights directly from our Hugging Face repository):

```bash
pip install huggingface_hub sentence-transformers faiss-cpu onnxruntime pandas numpy scipy transformers && huggingface-cli download mdumr4/redrob --local-dir . --local-dir-use-symlinks False
```

---

## 💻 Stage 3 Reproduction Command

Once the setup is complete, execute the official reproduction command from the **root of the repository** to run the ranking and generate the output:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### ⏱️ Expected Execution Profile (CPU Only)
* **Stage 0 (Engine Loading):** **0.2s** (Loads cross-encoder weights and pre-computed JD vectors).
* **Stage 1 (Hybrid Retrieval):** **1.88s** (FAISS dense search + SPLADE sparse search + RRF fusion).
* **Stage 2 (Two-Part Reranking):** **14s** (Evaluates top 100 candidates on Tech/Logistics weighted fusion).
* **Stage 3 (LLM Reasoning):** **111s** (Batch-generates 100 candidate rationales with shared prompt caching).
* **Total Runtime:** **~135s (~2.3 minutes)**.

---

## 📓 Standalone Google Colab Run

If you want to run the pipeline in the cloud, we have provided a **standalone Google Colab notebook** in the repository:
👉 **[sandbox/run_pipeline_colab.ipynb](sandbox/run_pipeline_colab.ipynb)**

### How to Run:
1. Open [Google Colab](https://colab.research.google.com).
2. Go to the **GitHub** tab and paste your repository URL: `https://github.com/mdumr4/resume-ranker`.
3. Select and open `sandbox/run_pipeline_colab.ipynb`.
4. Click **Runtime** -> **Run All**.
5. The notebook will clone the repository, download all required models/indexes dynamically, boot `llama-server` in the background, run the pipeline, and validate the output automatically.

---

## 📐 Architecture & Optimization Highlights

1. **Pre-computed Job Description (JD) Vectors:**
   * Since the target Job Description is static, we pre-computed its dense query representation and sparse SPLADE query representation. 
   * This removes the need to load the heavy Qwen-Embedding and SPLADE-ONNX models at runtime, saving **~1.5 GB of RAM** and reducing loading time by **98%** (Stage 0 loader drops from 13.2s to 0.2s).

2. **Two-Part Cross-Encoder Reranker:**
   * We run a two-phase evaluation using `ms-marco-MiniLM-L-12-v2`:
     * **Tech Match:** Evaluates candidate skills & career history against Tech requirements.
     * **Logistics Match:** Evaluates candidate education, notice period, location, and profiles.
     * **Additive Sigmoid Fusion:** Normalizes logits via Sigmoid and fuses them: $\text{Score} = 0.70 \times \text{prob\_tech} + 0.30 \times \text{prob\_logistics}$.

3. **High-Throughput continuous LLM Batching:**
   * Starts a background `llama-server` running `Qwen2.5-0.5B-Instruct` (Q4_K_M GGUF).
   * Spawns 32 parallel threads to dispatch prompt completions simultaneously. The server dynamically batches the requests and utilizes shared prompt caching, generating all 100 rationales in under 2 minutes on standard CPU threads.
