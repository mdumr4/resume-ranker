# Redrob Candidate Discovery & Ranking Pipeline

Welcome to our submission for the **Redrob Talent Intelligence Hackathon**.

This repository contains an ultra-optimized, strictly CPU-bound ranking pipeline designed to process 100,000 candidate profiles, dynamically calculate trust and timeline career gap penalties, re-rank them semantically, and generate natural language reasoning rationales—all running fully locally on a CPU in **~3.6 minutes** (safely within the strict 5-minute hackathon constraint).

---

## 🚀 Restructured Codebase Layout

To comply with the Stage 3 Sandboxed Reproduction and Stage 5 manual review guidelines, we have restructured all core execution modules, pre-computed indices, and models into a self-contained **`sandbox/`** directory.

```text
├── sandbox/                         # Core execution sandbox
│   ├── index/                       # Pre-computed indices (FAISS binary, SPLADE NPZ, Parquet features)
│   ├── models/                      # Local models (Qwen Embedding, SPLADE ONNX, Qwen2.5 GGUF)
│   ├── src/
│   │   └── optimize_rank.py         # The core hybrid retrieval & two-phase re-ranking script
│   ├── run_pipeline_colab.ipynb     # The Google Colab replication notebook
│   ├── explain.md                   # Technical Deep-Dive & Architecture Whitepaper
│   ├── submission_metadata.yaml     # Hackathon portal metadata
│   ├── pyproject.toml & uv.lock     # Sandbox package manager and exact dependency lock file
│   └── validate_submission.py       # Submission validator script
│
├── context/                         # Hackathon specification documents and specifications
└── Trails/                          # Experimental history and trails
```

---

## 💻 Colab Reproduction Instructions (Recommended)

Our sandbox runs end-to-end on **Google Colab** on standard CPU runtimes in under 4 minutes.

### Steps to Reproduce:

1. Upload the `sandbox/` folder to your Google Drive.
2. Open **`run_pipeline_colab.ipynb`** in Google Colab.
3. Execute the cells in order:
   * **Cell 1 & 2:** Mount Google Drive and navigate to `/content/drive/MyDrive/.../sandbox`.
   * **Cell 4:** Install required dependencies (`pip install sentence-transformers faiss-cpu onnxruntime pandas numpy scipy transformers llama-cpp-python`).
   * **Cell 5:** Fetches the correct Ubuntu Linux x64 build of `llama-server` and dependencies on-the-fly and marks it executable.
   * **Cell 6:** Starts `llama-server` in the background and runs the pipeline (`python src/optimize_rank.py`).
4. The final formatted output will be saved as **`sandbox/submission.csv`** containing exact ranks, real model scores, and two-sentence reasoning texts.

---

## 🧪 Submission Validation

Verify the formatting of the generated `submisC:\Users\conqu\Desktop\Umar\Workspace\India Runs\resume-ranker>python rank.py --candidates ./candidates.jsonl --out ./submission.csv
`

`Starting llama-server in the background...
`

`Error starting llama-server: [WinError 2] The system cannot find the file specified. Check path or permissions.
`

`
`

`C:\Users\conqu\Desktop\Umar\Workspace\India Runs\resume-ranker>python rank.py --candidates ./candidates.jsonl --out ./submission.csv
`

`Starting llama-server in the background...
`

`Waiting for llama-server to be ready...
`

`Server is ready! Running the pipeline...
`

`Traceback (most recent call last):
`

`  File "C:\Users\conqu\Desktop\Umar\Workspace\India Runs\resume-ranker\sandbox\src\optimize_rank.py", line 20, in <module>
`

`    import faiss
`

`ModuleNotFoundError: No module named 'faiss'
`

`Pipeline finished. Terminating llama-server...
`

`Done.
`

`Traceback (most recent call last):
`

`  File "C:\Users\conqu\Desktop\Umar\Workspace\India Runs\resume-ranker\rank.py", line 85, in <module>
`

`    main()
`

`    ~~~~^^
`

`  File "C:\Users\conqu\Desktop\Umar\Workspace\India Runs\resume-ranker\rank.py", line 75, in main
`

`    subprocess.run([sys.executable, "-u", "src/optimize_rank.py", "--out", out_path], check=True)
`

`    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`

`  File "C:\Users\conqu\AppData\Local\Programs\Python\Python313\Lib\subprocess.py", line 579, in run
`

`    raise CalledProcessError(retcode, process.args,
`

`                             output=stdout, stderr=stderr)
`

`subprocess.CalledProcessError: Command '['C:\\Users\\conqu\\AppData\\Local\\Programs\\Python\\Python313\\python.exe', '-u', 'src/optimize_rank.py', '--out', 'C:\\Users\\conqu\\Desktop\\Umar\\Workspace\\India Runs\\resume-ranker\\submission.csv']' returned non-zero exit status 1.
`

`
`

`C:\Users\conqu\Desktop\Umar\Workspace\India Runs\resume-ranker>sion.csv` inside the sandbox using:

```bash
python sandbox/validate_submission.py --submission sandbox/submission.csv
```

This utility ensures:

* Exactly 100 rows.
* Ranks start at 1.
* No duplicate candidate IDs.
* Scores are strictly monotonically decreasing with no ties.
* Output is formatted correctly.
