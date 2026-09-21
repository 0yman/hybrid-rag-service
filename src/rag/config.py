"""Runtime configuration, loaded from environment or a .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

EmbeddingBackend = Literal["gemini", "local", "hash"]
LLMBackend = Literal["gemini", "mock"]


class Settings(BaseSettings):
    """All knobs in one place so experiments stay reproducible."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="RAG_", extra="ignore", protected_namespaces=()
    )

    # --- providers -------------------------------------------------------
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    embedding_backend: EmbeddingBackend = "local"
    llm_backend: LLMBackend = "gemini"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_chat_model: str = "gemini-3.6-flash"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Gemini's free tier allows ~15 requests/minute; batch and pace accordingly.
    embed_batch_size: int = 32
    max_retries: int = 5
    retry_base_delay: float = 2.0
    # Per-request HTTP timeout in milliseconds, passed to the SDK.
    request_timeout_ms: int = 60_000

    # --- chunking --------------------------------------------------------
    chunk_size: int = 220          # words, not characters
    chunk_overlap: int = 40
    min_chunk_words: int = 25

    # --- retrieval -------------------------------------------------------
    dense_top_k: int = 20
    lexical_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    min_score_to_answer: float = 0.0

    # --- generation ------------------------------------------------------
    temperature: float = 0.0
    max_output_tokens: int = 1024

    # --- storage ---------------------------------------------------------
    index_dir: Path = REPO_ROOT / "data" / "index"
    corpus_dir: Path = REPO_ROOT / "data" / "corpus"

    def require_api_key(self) -> str:
        if not self.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and put it in .env, "
                "or set RAG_LLM_BACKEND=mock to run offline."
            )
        return self.google_api_key


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
