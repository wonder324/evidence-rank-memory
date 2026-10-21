from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYSTEM_NAME = "evidence-rank-memory"
SYSTEM_VERSION = "0.4.0"


def load_env_file(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env. Existing process environment wins."""
    env_path = path or (ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_env_file()

DEFAULT_WEIGHTS = {
    "bm25": 1.15,
    "dense": 0.45,
    "overlap": 0.85,
    "entity": 0.40,
    "literal": 0.90,
    "temporal": 0.70,
    "kind": 0.55,
    "hop": 0.35,
    "option": 0.18,
    "profile": 0.15,
}


def _load_weights(path: Path) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("weights", payload)
        for key, value in raw.items():
            if key in weights:
                weights[key] = float(value)
    return weights


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("DATA_DIR", ROOT / "data")))
    api_key: str = field(default_factory=lambda: os.environ.get("MEMORY_API_KEY", "").strip())
    embedder: str = field(default_factory=lambda: os.environ.get("EMBEDDER", "auto").strip().lower())
    dashscope_api_key: str = field(default_factory=lambda: os.environ.get("DASHSCOPE_API_KEY", "").strip())
    embedding_model: str = field(default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "text-embedding-v4"))
    llm_extract: bool = field(default_factory=lambda: os.environ.get("LLM_EXTRACT", "0").strip() in {"1", "true", "yes"})
    llm_extract_strict: bool = field(default_factory=lambda: os.environ.get("LLM_EXTRACT_STRICT", "0").strip() in {"1", "true", "yes"})
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", "").strip())
    openai_base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"))
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    weights_path: Path = field(default_factory=lambda: Path(os.environ.get("FUSION_WEIGHTS", ROOT / "models" / "fusion_weights.json")))
    host: str = field(default_factory=lambda: os.environ.get("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("PORT", "8080")))
    mmr_lambda: float = field(default_factory=lambda: float(os.environ.get("MMR_LAMBDA", "0.78")))
    full_scan_limit: int = field(default_factory=lambda: int(os.environ.get("FULL_SCAN_LIMIT", "2500")))
    candidate_limit: int = field(default_factory=lambda: int(os.environ.get("CANDIDATE_LIMIT", "800")))
    scheme_pref: bool = field(default_factory=lambda: os.environ.get("SCHEME_PREF", "1").strip() in {"1", "true", "yes"})
    scheme_hop: bool = field(default_factory=lambda: os.environ.get("SCHEME_HOP", "1").strip() in {"1", "true", "yes"})
    hop_residence_bonus: float = field(default_factory=lambda: float(os.environ.get("HOP_RESIDENCE_BONUS", "0.01")))
    reranker: bool = field(default_factory=lambda: os.environ.get("RERANKER", "1").strip() in {"1", "true", "yes"})
    reranker_model: str = field(default_factory=lambda: os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base"))
    reranker_top_n: int = field(default_factory=lambda: int(os.environ.get("RERANKER_TOP_N", "20")))
    reranker_context: bool = field(default_factory=lambda: os.environ.get("RERANKER_CONTEXT", "0").strip() in {"1", "true", "yes"})
    reranker_blend: float = field(default_factory=lambda: float(os.environ.get("RERANKER_BLEND", "0.45")))
    chronos: bool = field(default_factory=lambda: os.environ.get("CHRONOS", "0").strip() in {"1", "true", "yes"})

    def resolved_embedder(self) -> str:
        if self.embedder == "auto":
            return "dashscope" if self.dashscope_api_key else "hash"
        return self.embedder

    def weights(self) -> dict[str, float]:
        return _load_weights(self.weights_path)


def get_settings() -> Settings:
    return Settings()
