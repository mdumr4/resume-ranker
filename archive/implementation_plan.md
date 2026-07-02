# Trails 2: Clean Architecture Plan

You are absolutely right. Trail 8 became a mess of hotfixes, skipped steps, and patched logic. Let's wipe the slate clean and build this perfectly from the ground up in the `trails2/` folder. 

This architecture will be fully deterministic, fully offline-capable, and mathematically sound without any "black box" truncations.

## Proposed Changes

We will create exactly three files in the `trails2/` directory:

### [NEW] [requirements.txt](file:///C:/Users/conqu/Desktop/Umar/Workspace/India%20Runs/India_runs_data_and_ai_challenge/trails2/requirements.txt)
A clean, minimal list of required pip dependencies.
- `faiss-cpu`, `sentence-transformers`, `transformers`, `llama-cpp-python`, `pandas`, `numpy`, `huggingface-hub`.

### [NEW] [download_models.py](file:///C:/Users/conqu/Desktop/Umar/Workspace/India%20Runs/India_runs_data_and_ai_challenge/trails2/download_models.py)
A bulletproof download script that pulls the correct 4 models into `trails2/models/`.
- **Dense Encoder:** `Qwen/Qwen3-Embedding-0.6B`
- **Sparse Encoder:** `naver/splade-v3`
- **Cross Encoder:** `BAAI/bge-reranker-v2-m3`
- **Reasoning LLM:** `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (Corrected to the official 1.5B model to prevent 404 errors).

### [NEW] [rank.py](file:///C:/Users/conqu/Desktop/Umar/Workspace/India%20Runs/India_runs_data_and_ai_challenge/trails2/rank.py)
The single, golden ranking script. It will be architected with these strict mathematical rules:
1. **Inputs:** It will natively point to the `index/` folder and `candidates.jsonl` located in your root directory.
2. **Stage 1 (FAISS) & 1B (SPLADE):** Both models evaluate all 100,000 candidates and retrieve their Top 10,000. We multiply these raw scores by the candidate's `trust_score` **before** performing the Reciprocal Rank Fusion (RRF) to extract the absolute best Top 500.
3. **Stage 2 (Cross-Encoder):** The Top 500 are evaluated by BGE-M3. We will enforce `max_length=8192` so the candidate text is never truncated. The raw output logits will be normalized via a Sigmoid function `[0,1]` and multiplied by the `trust_score` to determine the final Top 100.
4. **Stage 3 (Qwen 1.5B Reasoning):** The final Top 100 are fed into `llama.cpp` using the exact Grammatical Template constraints. It will output a 1-sentence reasoning incorporating their profile fit and trust signals.
5. **Output:** Directly outputs the final `submission.csv`.

## User Review Required

> [!CAUTION]
> Because the Cross-Encoder requires evaluating 500 candidates natively without 512-token truncation, it will be computationally heavy. On a standard CPU, Stage 2 will take ~1 hour to run. 
> Do you want to implement a "Debug Mode" flag in `rank.py` that caps the Cross-Encoder to only the Top 20 candidates so you can test the pipeline locally in just 5 minutes before deploying to Colab?

## Open Questions

> [!IMPORTANT]
> 1. In Stage 3, should I use the `Qwen2.5-1.5B-Instruct` model, or do you want to stick to the `Qwen3-4B` model that you successfully downloaded earlier to save network bandwidth?
> 2. Are you comfortable with me writing all 3 files now based on this exact spec?
