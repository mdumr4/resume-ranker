import os
import sys
import shutil
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("config")


def get_optimal_thread_count() -> int:
    """Auto-detect available CPU threads with safe clamping."""
    try:
        if hasattr(os, "sched_getaffinity"):
            return max(1, min(len(os.sched_getaffinity(0)), 32))
        count = os.cpu_count() or 4
        return max(1, min(count, 32))
    except Exception:
        return 4


def find_first_existing(candidates: List[str], base_dir: Optional[Path] = None) -> str:
    """Searches candidate locations and returns the first existing path."""
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent

    for candidate in candidates:
        p = (base_dir / candidate).resolve()
        if p.exists():
            return str(p)
        p_lit = Path(candidate).resolve()
        if p_lit.exists():
            return str(p_lit)

    return str((base_dir / candidates[0]).resolve())


@dataclass
class PipelineConfig:
    """Production configuration parameters for Candidate Ranking Pipeline."""

    # Environment
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    num_threads: int = field(default_factory=get_optimal_thread_count)
    is_windows: bool = field(default_factory=lambda: sys.platform.startswith("win"))

    # File Paths
    features_path: str = field(default="")
    faiss_index_path: str = field(default="")
    splade_index_path: str = field(default="")
    dense_query_path: str = field(default="")
    sparse_query_path: str = field(default="")
    candidates_jsonl_path: str = field(default="")

    # Model & Binary Paths
    cross_encoder_path: str = field(default="")
    llm_model_path: str = field(default="")
    llama_server_bin: str = field(default="")

    # Retrieval & Reranking Hyperparameters
    faiss_top_k: int = 10000
    splade_top_k: int = 10000
    rrf_k: int = 60
    hybrid_top_k: int = 100
    ce_batch_size: int = 4
    tech_weight: float = 0.70
    logistics_weight: float = 0.30

    # LLM Settings
    llm_port: int = 8080
    llm_context_length: int = 16384
    llm_parallel_slots: int = 4
    llm_concurrency: int = 32
    llm_max_tokens: int = 50
    llm_timeout: float = 12.0

    # Static Job Description Strings
    jd_tech: str = (
        "The ideal candidate has 5-9 years of total experience. They should have worked as a "
        "Senior AI Engineer or Applied ML Engineer in the AI/ML, HR-Tech, or Marketplace/Talent-Intelligence "
        "industry doing ownership of ranking, retrieval, and candidate-JD matching systems; auditing and "
        "improving existing BM25/rule-based search infrastructure; shipping embeddings-based and hybrid "
        "retrieval systems with demonstrable improvements to engagement metrics; building offline and "
        "online evaluation frameworks (NDCG, MRR, MAP, A/B testing); and mentoring junior engineers in a "
        "high-growth, fast-moving product environment. The candidate must be highly proficient in "
        "embeddings-based retrieval systems (sentence-transformers, BGE, E5, OpenAI Embeddings, or equivalent), "
        "vector databases and hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, "
        "Elasticsearch, or FAISS), Python (production-grade, code-quality conscious), and ranking evaluation "
        "frameworks (NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation) with demonstrated "
        "production deployment experience across all of the above."
    )

    jd_logistics: str = (
        "Summary: Senior AI Engineer - Founding Team at Redrob AI, a Series A AI-native talent intelligence "
        "platform. The hire will own the intelligence layer of the product - the ranking, retrieval, and "
        "matching systems that power both recruiter-facing candidate search and candidate-facing job discovery. "
        "Candidate should be located in Pune or Noida, India (candidates from Hyderabad, Mumbai, or Delhi NCR "
        "are also in scope). This is a Hybrid role (flexible cadence; offices used primarily Tue/Thu). "
        "If the job is outside their city, they must be Willing to relocate. They must be Open to Work with "
        "a maximum notice period of 30 days (sub-30 preferred; buyout of up to 30 days available). We require "
        "a highly trusted candidate whose profile is 100% complete, with verified contact info and a connected "
        "LinkedIn. Behaviorally, they should have a strong GitHub score and a high recruiter response rate. "
        "Candidate must hold a degree in Computer Science, Electrical Engineering, Mathematics, or a related "
        "technical field from a Tier-1 university."
    )

    llm_system_prompt: str = (
        "You are an expert technical AI recruiter. Provide exactly TWO short sentences: "
        "1. Why the candidate achieved this rank based on their career and JD. "
        "2. Where they can improve or what they are missing. "
        "Do not write markdown headers, bullet points, or candidate names."
    )

    def __post_init__(self):
        # Resolve Data Paths
        if not self.features_path:
            self.features_path = find_first_existing([
                "index/features_v3.parquet",
            ], self.base_dir)

        if not self.faiss_index_path:
            self.faiss_index_path = find_first_existing([
                "index/faiss_qwen_index.bin",
            ], self.base_dir)

        if not self.splade_index_path:
            self.splade_index_path = find_first_existing([
                "index/splade_combined_v3.npz",
            ], self.base_dir)

        if not self.dense_query_path:
            self.dense_query_path = find_first_existing([
                "index/jd_dense_query.npy",
            ], self.base_dir)

        if not self.sparse_query_path:
            self.sparse_query_path = find_first_existing([
                "index/jd_sparse_query.npy",
            ], self.base_dir)

        if not self.candidates_jsonl_path:
            self.candidates_jsonl_path = find_first_existing([
                "data/candidates.jsonl",
                "data/sample_candidates.json",
            ], self.base_dir)

        # Resolve Model Paths (use local weights if present, fallback to HF Hub identifier)
        if not self.cross_encoder_path:
            local_ce = (self.base_dir / "models" / "ms-marco-MiniLM-L-12-v2").resolve()
            if local_ce.exists():
                self.cross_encoder_path = str(local_ce)
            else:
                self.cross_encoder_path = "cross-encoder/ms-marco-MiniLM-L-12-v2"

        if not self.llm_model_path:
            local_llm = (self.base_dir / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf").resolve()
            if local_llm.exists():
                self.llm_model_path = str(local_llm)
            else:
                self.llm_model_path = ""

        # Resolve Llama Server Binary (check env var, PATH, or local directory)
        if not self.llama_server_bin:
            env_bin = os.environ.get("LLAMA_SERVER_BIN", "")
            if env_bin and Path(env_bin).exists():
                self.llama_server_bin = env_bin
            else:
                which_bin = shutil.which("llama-server") or shutil.which("llama-server.exe")
                self.llama_server_bin = which_bin if which_bin else ""

    @property
    def jd_search_query(self) -> str:
        return f"{self.jd_tech} {self.jd_logistics}"


default_config = PipelineConfig()
