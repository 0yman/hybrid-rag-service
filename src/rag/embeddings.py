"""Embedding backends behind one interface.

Three implementations, picked by config:

* ``gemini`` - Google's hosted ``gemini-embedding-001``. Free tier, but rate
  limited, so calls are batched and retried with exponential backoff.
* ``local``  - a sentence-transformers model. No network, no quota.
* ``hash``   - a deterministic hashing vectoriser. No model download at all,
  which is what keeps CI fast and offline.

Every backend returns L2-normalised float32 vectors, so a dot product is
already the cosine similarity and the vector store needs no extra maths.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
import time
from abc import ABC, abstractmethod

import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving all-zero rows untouched."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder(ABC):
    """Common interface. Query and document embeddings are separate calls
    because hosted models often want a different task type for each."""

    dim: int
    name: str

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...


class HashEmbedder(Embedder):
    """Hashing vectoriser over word unigrams and bigrams.

    Deterministic and dependency-free. It has no semantic understanding, so it
    is for tests and CI - never for a real index.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vector(self, text: str) -> np.ndarray:
        tokens = _TOKEN.findall(text.lower())
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        vec = np.zeros(self.dim, dtype=np.float32)
        for gram in grams:
            digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            # Sign from a separate byte keeps unrelated collisions from
            # always reinforcing each other.
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._vector(text))[0]


class LocalEmbedder(Embedder):
    """sentence-transformers, loaded lazily so importing this module is cheap."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        # Renamed in sentence-transformers 5.x; keep working on older versions.
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self.name = model_name

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, batch_size=32, show_progress_bar=False)
        return l2_normalize(np.asarray(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(np.asarray(self._model.encode([text])))[0]


class GeminiEmbedder(Embedder):
    """Hosted embeddings with batching and 429-aware retries."""

    def __init__(self, settings: Settings, dim: int = 768) -> None:
        from google import genai
        from google.genai import types

        self._settings = settings
        self._client = genai.Client(
            api_key=settings.require_api_key(),
            # One retry layer only. The SDK retries internally by default,
            # which would compose with the backoff below into 25 attempts
            # and a request that looks like a hang.
            http_options=types.HttpOptions(
                timeout=settings.request_timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self._model = settings.gemini_embedding_model
        self.dim = dim
        self.name = f"{self._model}-{dim}"

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        from google.genai import types

        vectors: list[list[float]] = []
        batch = self._settings.embed_batch_size
        for start in range(0, len(texts), batch):
            window = texts[start : start + batch]
            response = self._call_with_retry(
                window,
                types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self.dim
                ),
            )
            vectors.extend(e.values for e in response.embeddings)
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def _call_with_retry(self, texts: list[str], config):
        last_error: Exception | None = None
        for attempt in range(self._settings.max_retries):
            try:
                return self._client.models.embed_content(
                    model=self._model, contents=texts, config=config
                )
            except Exception as exc:  # the SDK raises several error shapes
                if not _is_retryable(exc) or attempt == self._settings.max_retries - 1:
                    raise
                last_error = exc
                delay = self._settings.retry_base_delay * (2**attempt)
                delay += random.uniform(0, delay * 0.1)  # jitter, avoids lockstep retries
                logger.warning(
                    "Embedding call failed (%s); retrying in %.1fs", exc, delay
                )
                time.sleep(delay)
        raise RuntimeError("Unreachable retry state") from last_error

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


_RETRYABLE_MARKERS = ("429", "rate limit", "resource_exhausted", "503", "500", "unavailable", "deadline")


def _is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _RETRYABLE_MARKERS)


def get_embedder(settings: Settings) -> Embedder:
    backend = settings.embedding_backend
    if backend == "gemini":
        return GeminiEmbedder(settings)
    if backend == "local":
        return LocalEmbedder(settings.local_embedding_model)
    if backend == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend!r}")
