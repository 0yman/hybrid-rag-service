"""Embedding backends behind one interface.

Picked by config:

* ``local``  - a small model run on this machine (the default). No key, and
  no network after the first download.
* ``gemini`` - Google's hosted ``gemini-embedding-001``. Free tier, but rate
  limited, so calls are batched and retried with exponential backoff.
* ``openai`` - any OpenAI-format embeddings endpoint.
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
from .providers import import_genai, import_openai

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def _installed(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


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
    #: Whether similarity between these vectors tracks meaning. False only for
    #: the hashing stub, whose cosine is closer to word overlap - anything that
    #: reasons about "how close in meaning" has to know the difference.
    semantic: bool = True

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...


class HashEmbedder(Embedder):
    """Hashing vectoriser over word unigrams and bigrams.

    Deterministic and dependency-free. It has no semantic understanding, so it
    is for tests and CI - never for a real index.
    """

    semantic = False

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
    """A small embedding model run on this machine. No key, no network after
    the first download.

    Two runtimes can load it:

    * ``fastembed`` (ONNX Runtime) - the default. About 15 MB installed.
    * ``sentence-transformers`` (PyTorch) - used only if fastembed is missing
      or asked for. On Linux, a plain ``pip install`` of PyTorch pulls the
      CUDA build, which is gigabytes - enough to lose someone who just wanted
      to try the app, which is why it is not a default dependency.

    **They do not produce the same vectors**, despite loading a model with the
    same name. On real document chunks the two agree to a cosine similarity of
    about 0.88 on average and as little as 0.69 on some chunks - short, clean
    sentences agree almost perfectly, which is exactly what makes the
    difference easy to miss. So the runtime is part of the embedder's `name`:
    an index built by one is refused by the other and has to be rebuilt,
    rather than searched with vectors from a different space.
    """

    def __init__(self, model_name: str, engine: str = "auto") -> None:
        if engine == "auto":
            engine = "fastembed" if _installed("fastembed") else "sentence-transformers"
        if engine == "fastembed":
            self._init_fastembed(model_name)
        elif engine == "sentence-transformers":
            self._init_sentence_transformers(model_name)
        else:
            raise ValueError(f"Unknown local embedding engine: {engine!r}")
        self.name = f"{self._engine}:{model_name}"

    def _init_fastembed(self, model_name: str) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("fastembed is not installed: `pip install fastembed`.") from exc
        self._engine = "fastembed"
        self._model = TextEmbedding(model_name)
        self.dim = int(len(next(iter(self._model.embed(["dimension probe"])))))

    def _init_sentence_transformers(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "No local embedding engine is installed. Run "
                "`pip install fastembed` (small), or set "
                "RAG_EMBEDDING_BACKEND=gemini / openai to use a hosted one."
            ) from exc
        self._engine = "sentence-transformers"
        self._model = SentenceTransformer(model_name)
        # Renamed in sentence-transformers 5.x; keep working on older versions.
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._engine == "fastembed":
            return np.asarray(list(self._model.embed(texts, batch_size=32)))
        return np.asarray(self._model.encode(texts, batch_size=32, show_progress_bar=False))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return l2_normalize(self._encode(texts))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._encode([text]))[0]


class GeminiEmbedder(Embedder):
    """Hosted embeddings with batching and 429-aware retries."""

    def __init__(self, settings: Settings, dim: int = 768) -> None:
        # The key first: someone without one needs to hear about the key, not
        # about a package they would only need once they had one.
        api_key = settings.require_api_key()
        genai, types = import_genai()

        self._settings = settings
        self._client = genai.Client(
            api_key=api_key,
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


class OpenAIEmbedder(Embedder):
    """OpenAI-format embeddings, batched and retried.

    `text-embedding-3-*` supports `dimensions`, which truncates the vector
    server-side. Smaller vectors mean a smaller index and faster search for a
    small accuracy cost - worth exposing rather than hard-coding.
    """

    def __init__(self, settings: Settings) -> None:
        api_key = settings.require_openai_key()  # before the import; see GeminiLLM
        OpenAI = import_openai()  # noqa: N806 - it is a class

        self._settings = settings
        self._client = OpenAI(
            api_key=api_key,
            base_url=settings.openai_base_url or None,
            timeout=settings.request_timeout_ms / 1000,
            max_retries=0,
        )
        self._model = settings.openai_embedding_model
        self.dim = settings.openai_embedding_dim
        self.name = f"{self._model}-{self.dim}"

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        batch = self._settings.embed_batch_size
        for start in range(0, len(texts), batch):
            response = self._call_with_retry(texts[start : start + batch])
            # The API does not guarantee response order, so sort by index.
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def _call_with_retry(self, texts: list[str]):
        settings = self._settings
        for attempt in range(settings.max_retries):
            try:
                return self._client.embeddings.create(
                    model=self._model, input=texts, dimensions=self.dim
                )
            except Exception as exc:
                if not _is_retryable(exc) or attempt == settings.max_retries - 1:
                    raise
                delay = settings.retry_base_delay * (2**attempt)
                delay += random.uniform(0, delay * 0.1)
                logger.warning("Embedding call failed (%s); retrying in %.1fs", exc, delay)
                time.sleep(delay)
        raise RuntimeError("Unreachable retry state")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]


_RETRYABLE_MARKERS = ("429", "rate limit", "resource_exhausted", "503", "500", "unavailable", "deadline")


def _is_retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _RETRYABLE_MARKERS)


def get_embedder(settings: Settings) -> Embedder:
    backend = settings.embedding_backend
    if backend == "gemini":
        return GeminiEmbedder(settings)
    if backend == "openai":
        return OpenAIEmbedder(settings)
    if backend == "local":
        return LocalEmbedder(settings.local_embedding_model, settings.local_embedding_engine)
    if backend == "hash":
        return HashEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend!r}")
