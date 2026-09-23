"""Runtime configuration, loaded from environment or a .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

EmbeddingBackend = Literal["gemini", "openai", "local", "hash"]
# "auto" picks the best answer engine available: a hosted model if a key is
# configured, otherwise extractive answers that need no key and no network.
LLMBackend = Literal["auto", "gemini", "openai", "extractive"]


class Settings(BaseSettings):
    """All knobs in one place so experiments stay reproducible."""

    model_config = SettingsConfigDict(
        # Absolute, so the app finds its .env whatever directory it is
        # launched from - a relative path silently loads nothing otherwise.
        env_file=REPO_ROOT / ".env",
        env_prefix="RAG_",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- providers -------------------------------------------------------
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    embedding_backend: EmbeddingBackend = "local"
    llm_backend: LLMBackend = "auto"
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_chat_model: str = "gemini-3.1-flash-lite"
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Which runtime loads the local model. auto = fastembed if installed.
    # The two give different vectors for the same model, so switching means
    # rebuilding the index - the app refuses to mix them.
    local_embedding_engine: Literal["auto", "fastembed", "sentence-transformers"] = "auto"

    # Any OpenAI-format endpoint: blank for OpenAI itself, or point at Groq,
    # Together, OpenRouter, a local Ollama or vLLM. Same wire format, same
    # adapter. (Note that not every such host serves an embeddings route.)
    openai_base_url: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536

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
    # Extractive answers: the least cosine similarity a sentence needs to the
    # question to be quoted at all. Below it, the app says the answer is not
    # in the documents. 0.45 sat in the gap between the most similar
    # off-topic question (0.40) and the least similar real one (0.54) on the
    # benchmark - a thin margin chosen on that same data, so treat it as a
    # starting point for other document collections, not a law.
    extractive_min_similarity: float = 0.45

    # --- storage ---------------------------------------------------------
    # Your documents: uploaded files and the index built from them.
    index_dir: Path = DATA_DIR / "index"
    uploads_dir: Path = DATA_DIR / "uploads"
    max_upload_mb: int = 25
    # The evaluation benchmark: a fixed corpus with hand-written questions,
    # kept in its own index so evaluating never touches your documents.
    benchmark_dir: Path = DATA_DIR / "benchmark"
    benchmark_index_dir: Path = DATA_DIR / "benchmark_index"

    def resolved_llm_backend(self) -> str:
        """The answer engine `auto` resolves to, given the keys present."""
        if self.llm_backend != "auto":
            return self.llm_backend
        if self.google_api_key:
            return "gemini"
        if self.openai_api_key:
            return "openai"
        return "extractive"

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Put it in .env, or use "
                "RAG_LLM_BACKEND=auto to fall back to extractive answers."
            )
        return self.openai_api_key

    def require_api_key(self) -> str:
        if not self.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and put it in .env, "
                "or use RAG_LLM_BACKEND=auto to fall back to extractive answers."
            )
        return self.google_api_key


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)
