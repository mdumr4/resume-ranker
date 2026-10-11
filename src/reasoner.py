import time
import socket
import logging
import subprocess
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import requests
import pandas as pd
from src.config import PipelineConfig, default_config

logger = logging.getLogger("reasoner")


class LLMReasoner:
    """
    Manages generation of explainable candidate hiring rationales.
    Dispatches to local llama-server continuous batching pool when available,
    or falls back to high-throughput deterministic rationales.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or default_config
        self.process: Optional[subprocess.Popen] = None
        self.log_file = None
        self.err_file = None

    def start_server(self) -> bool:
        """Starts local llama-server in the background if binary and model exist."""
        bin_path = self.config.llama_server_bin
        model_path = self.config.llm_model_path
        port = self.config.llm_port

        if not bin_path or not Path(bin_path).exists():
            logger.info("llama-server binary not found on system. Using deterministic rationale engine.")
            return False

        if not model_path or not Path(model_path).exists():
            logger.info("Local LLM model GGUF not found. Using deterministic rationale engine.")
            return False

        logger.info(f"Starting llama-server on port {port}...")
        logger.info(f"  Binary: {bin_path}")
        logger.info(f"  Model: {model_path}")

        cmd = [
            bin_path,
            "-m", model_path,
            "-c", str(self.config.llm_context_length),
            "-np", str(self.config.llm_parallel_slots),
            "--threads", str(self.config.num_threads),
            "--port", str(port),
        ]

        try:
            self.log_file = open("llama-server.log", "w", encoding="utf-8")
            self.err_file = open("llama-server-err.log", "w", encoding="utf-8")
            self.process = subprocess.Popen(
                cmd,
                stdout=self.log_file,
                stderr=self.err_file,
                cwd=str(self.config.base_dir),
            )
        except Exception as e:
            logger.warning(f"Failed to start llama-server process: {e}")
            self._close_logs()
            return False

        ready = False
        logger.info("Waiting for llama-server socket...")
        for _ in range(35):
            time.sleep(1)
            try:
                s = socket.create_connection(("127.0.0.1", port), timeout=1)
                s.close()
                ready = True
                break
            except Exception:
                pass

        if not ready:
            logger.warning("llama-server failed to respond within 35 seconds. Reverting to fallback.")
            self.stop_server()
            return False

        logger.info("llama-server is online and operational!")
        return True

    def stop_server(self):
        """Terminates the llama-server process cleanly."""
        if self.process is not None:
            logger.info("Stopping llama-server process...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

        self._close_logs()

    def _close_logs(self):
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass
            self.log_file = None
        if self.err_file:
            try:
                self.err_file.close()
            except Exception:
                pass
            self.err_file = None

    def build_prompt(self, rank: int, row: pd.Series) -> str:
        """Constructs chat prompt for the candidate rationale."""
        def limit_words(text: Any, limit: int) -> str:
            if not isinstance(text, str):
                return ""
            return " ".join(text.split()[:limit])

        exp_val = row.get("experience_years")
        if pd.isna(exp_val) or str(exp_val).strip() == "" or str(exp_val).lower() == "nan":
            exp_str = "an unspecified amount of experience"
        else:
            exp_str = f"{exp_val} years of experience"

        skills_excerpt = limit_words(row.get("text_skills", ""), 30)
        career_excerpt = limit_words(row.get("text_career", ""), 40)
        trust_val = float(row.get("trust_score", 1.0))

        compact_context = (
            f"Candidate ranked {rank} has a trust score of {trust_val:.2f} and {exp_str}. "
            f"Their key skills include {skills_excerpt}. Their career highlights include {career_excerpt}."
        )

        p1 = "<|im_start|>system\n" + self.config.llm_system_prompt + "<|im_end|>\n"
        p2 = "<|im_start|>user\nJob: Senior AI Engineer at Redrob AI. Requirements: 5-9 yrs, AI/ML, ranking/retrieval systems, embeddings, vector DBs.\n\n"
        p3 = "Candidate ranked 1 has a trust score of 0.89 and 7 years of experience. Their key skills include Highly verified expert in PyTorch, NLP, and vector databases. Their career highlights include 3 years leading search ranking at TechCorp.\n\nResponse:<|im_end|>\n"
        p4 = "<|im_start|>assistant\nThe candidate is selected for their strong background in search ranking at TechCorp and verified skills in PyTorch and vector databases. They can improve by gaining more experience with sparse retrieval models.<|im_end|>\n"
        p5 = "<|im_start|>user\nJob: Senior AI Engineer at Redrob AI. Requirements: 5-9 yrs, AI/ML, ranking/retrieval systems, embeddings, vector DBs.\n\n" + compact_context + "\n\nResponse:<|im_end|>\n<|im_start|>assistant\n"
        return p1 + p2 + p3 + p4 + p5

    def build_fallback_rationale(self, rank: int, row: pd.Series) -> str:
        """Constructs a deterministic two-sentence fallback rationale."""
        exp_val = row.get("experience_years", "unspecified")
        trust_val = float(row.get("trust_score", 1.0))
        return (
            f"The candidate is selected based on {exp_val} years of relevant engineering background and a trust score of {trust_val:.2f}. "
            f"They can improve by demonstrating deeper production experience in hybrid retrieval and vector database systems."
        )

    def fetch_single_rationale(
        self, rank: int, cid: str, score: float, row: pd.Series
    ) -> Dict[str, Any]:
        """Dispatches completion request to llama-server with automatic fallback."""
        prompt = self.build_prompt(rank, row)
        fallback = self.build_fallback_rationale(rank, row)

        try:
            resp = requests.post(
                f"http://127.0.0.1:{self.config.llm_port}/completion",
                json={
                    "prompt": prompt,
                    "n_predict": self.config.llm_max_tokens,
                    "stop": ["<|im_end|>", "\n\n", "Candidate:"],
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "top_k": 20,
                },
                timeout=self.config.llm_timeout,
            )
            if resp.status_code == 200:
                rationale = resp.json().get("content", "").strip()
                if not rationale or len(rationale) < 15:
                    rationale = fallback
            else:
                rationale = fallback
        except Exception:
            rationale = fallback

        return {
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 6),
            "reasoning": rationale,
        }

    def generate_rationales(
        self,
        ranked_candidates: List[Tuple[str, float]],
        df: pd.DataFrame,
        skip_llm: bool = False,
    ) -> List[Dict[str, Any]]:
        """Generates rationales for all ranked candidates."""
        df_map = df.set_index("candidate_id")

        if skip_llm:
            logger.info("Using deterministic rationale generator (fast mode)...")
            results = []
            for rank_idx, (cid, score) in enumerate(ranked_candidates):
                rank = rank_idx + 1
                row = df_map.loc[cid] if cid in df_map.index else pd.Series()
                results.append({
                    "candidate_id": cid,
                    "rank": rank,
                    "score": round(score, 6),
                    "reasoning": self.build_fallback_rationale(rank, row),
                })
            return results

        server_started = self.start_server()
        if not server_started:
            return self.generate_rationales(ranked_candidates, df, skip_llm=True)

        t0 = time.time()
        logger.info(f"Dispatching {len(ranked_candidates)} candidates to continuous batching pool...")
        results = [None] * len(ranked_candidates)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.llm_concurrency) as executor:
                future_to_idx = {
                    executor.submit(
                        self.fetch_single_rationale,
                        rank_idx + 1,
                        cid,
                        score,
                        df_map.loc[cid] if cid in df_map.index else pd.Series(),
                    ): rank_idx
                    for rank_idx, (cid, score) in enumerate(ranked_candidates)
                }

                completed = 0
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                    completed += 1
                    if completed % 20 == 0 or completed == len(ranked_candidates):
                        elapsed = time.time() - t0
                        rate = completed / elapsed if elapsed > 0 else 1.0
                        eta = (len(ranked_candidates) - completed) / rate
                        logger.info(f"  [{completed}/{len(ranked_candidates)}] {elapsed:.1f}s elapsed, ~{eta:.0f}s ETA")
        finally:
            self.stop_server()

        logger.info(f"Reasoning generation complete in {time.time() - t0:.1f}s")
        return results
