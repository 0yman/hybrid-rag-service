"""End-to-end orchestration: ingest -> index -> retrieve -> generate."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .chunking import chunk_document
from .config import Settings, get_settings
from .embeddings import Embedder, get_embedder
from .generator import Generator
from .lexical import BM25Index
from .llm import ExtractiveLLM, get_llm
from .loaders import load_directory, load_file
from .models import Answer, Chunk, Document, ScoredChunk
from .retriever import CrossEncoderReranker, HybridRetriever
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


def _describe_failure(exc: Exception) -> str:
    """A short, human reason for a provider failure - not a stack trace."""
    message = str(exc).lower()
    if "503" in message or "unavailable" in message or "high demand" in message:
        return "it is overloaded right now"
    if "429" in message or "resource_exhausted" in message or "rate limit" in message:
        return "the free-tier rate limit was reached"
    if "api key" in message or "api_key" in message or "401" in message or "403" in message:
        return "the API key was rejected"
    if "timeout" in message or "deadline" in message:
        return "it timed out"
    return "it returned an error"


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        use_reranker: bool = False,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        reranker = CrossEncoderReranker() if use_reranker else None
        self.retriever = HybridRetriever(
            vector_store, bm25_index, embedder, settings, reranker
        )
        self._generator: Generator | None = None

    @property
    def generator(self) -> Generator:
        """Built on first use.

        Ingestion needs no LLM at all, so constructing one eagerly would make
        `scripts/ingest.py` fail without a chat API key for no reason.
        """
        if self._generator is None:
            if self.settings.resolved_llm_backend() == "extractive":
                self._generator = Generator(self._extractive_engine())
            else:
                self._generator = Generator(get_llm(self.settings))
        return self._generator

    def _extractive_engine(self) -> ExtractiveLLM:
        """Quote by meaning when the index has a real embedding model behind
        it; fall back to shared words when it is the hashing stub."""
        return ExtractiveLLM(
            embedder=self.embedder if self.embedder.semantic else None,
            min_similarity=self.settings.extractive_min_similarity,
        )

    # --- construction ----------------------------------------------------

    @classmethod
    def create(cls, settings: Settings | None = None, use_reranker: bool = False) -> RAGPipeline:
        """A pipeline with an empty index, ready to ingest."""
        settings = settings or get_settings()
        embedder = get_embedder(settings)
        return cls(
            settings=settings,
            embedder=embedder,
            vector_store=VectorStore(embedder.dim, embedder.name),
            bm25_index=BM25Index(),
            use_reranker=use_reranker,
        )

    @classmethod
    def load(cls, settings: Settings | None = None, use_reranker: bool = False) -> RAGPipeline:
        """A pipeline restored from a previously saved index."""
        settings = settings or get_settings()
        embedder = get_embedder(settings)
        vector_store = VectorStore.load(settings.index_dir)
        if vector_store.embedder_name != embedder.name:
            raise ValueError(
                f"Index was built with {vector_store.embedder_name!r} but the "
                f"current config uses {embedder.name!r}. Vectors from different "
                "models are not comparable - re-ingest the corpus."
            )
        return cls(
            settings=settings,
            embedder=embedder,
            vector_store=vector_store,
            bm25_index=BM25Index.load(settings.index_dir),
            use_reranker=use_reranker,
        )

    @classmethod
    def open(cls, settings: Settings | None = None) -> RAGPipeline:
        """Load the saved index if there is one, otherwise start empty.

        What the app wants on startup: a first run has no index yet, and that
        is a normal state to begin in, not an error to report.
        """
        settings = settings or get_settings()
        try:
            return cls.load(settings)
        except FileNotFoundError:
            return cls.create(settings)

    # --- ingestion -------------------------------------------------------

    def ingest_documents(self, documents: list[Document]) -> int:
        settings = self.settings
        # Ingesting a document that is already indexed replaces it, so
        # re-uploading an edited file updates it rather than duplicating
        # every one of its chunks.
        indexed = self.indexed_doc_ids()
        for document in documents:
            if document.doc_id in indexed:
                self.remove_document(document.doc_id)

        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(
                chunk_document(
                    document,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                    min_chunk_words=settings.min_chunk_words,
                )
            )
        if not chunks:
            return 0

        logger.info("Embedding %d chunks from %d documents", len(chunks), len(documents))
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.vector_store.add(chunks, vectors)
        self.bm25_index.add(chunks)
        return len(chunks)

    def ingest_path(self, path: Path) -> tuple[int, int]:
        """Ingest a file or a directory. Returns (documents, chunks)."""
        path = Path(path)
        if path.is_dir():
            documents = list(load_directory(path))
        else:
            document = load_file(path)
            documents = [document] if document else []
        if not documents:
            return 0, 0
        return len(documents), self.ingest_documents(documents)

    def save(self) -> None:
        self.vector_store.save(self.settings.index_dir)
        self.bm25_index.save(self.settings.index_dir)

    # --- document management ---------------------------------------------

    def indexed_doc_ids(self) -> set[str]:
        return {chunk.doc_id for chunk in self.vector_store.chunks}

    def list_documents(self) -> list[dict[str, Any]]:
        """One entry per indexed document, in the order they were added."""
        documents: dict[str, dict[str, Any]] = {}
        for chunk in self.vector_store.chunks:
            entry = documents.setdefault(
                chunk.doc_id,
                {
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "chunks": 0,
                    "words": 0,
                },
            )
            entry["chunks"] += 1
            entry["words"] += len(chunk.text.split())
        return list(documents.values())

    def remove_document(self, doc_id: str) -> int:
        removed = self.vector_store.remove_document(doc_id)
        self.bm25_index.remove_document(doc_id)
        return removed

    def clear(self) -> None:
        self.vector_store.clear()
        self.bm25_index.clear()

    # --- querying --------------------------------------------------------

    def retrieve(self, question: str, top_k: int | None = None) -> list[ScoredChunk]:
        contexts, _ = self.retriever.retrieve(question, top_k)
        return contexts

    def query(self, question: str, top_k: int | None = None) -> Answer:
        contexts, debug = self.retriever.retrieve(question, top_k)
        try:
            answer = self.generator.generate(question, contexts)
        except Exception as exc:
            # A hosted model can be down, rate limited or mis-keyed. The
            # retrieval already succeeded, so rather than turn that into an
            # error page, answer from the same passages by quoting them - and
            # say plainly that this is what happened.
            if self.settings.resolved_llm_backend() == "extractive":
                raise
            logger.warning("Answer engine failed, falling back to extractive: %s", exc)
            answer = Generator(self._extractive_engine()).generate(question, contexts)
            answer.notice = (
                f"The AI model could not be reached ({_describe_failure(exc)}), "
                "so this answer quotes your documents directly instead. "
                "Try again in a minute for a written answer."
            )
        logger.debug(
            "q=%r dense=%d lexical=%d fused=%d cited=%s",
            question, debug.dense_hits, debug.lexical_hits,
            debug.fused_hits, answer.cited_ordinals,
        )
        return answer

    def stats(self) -> dict[str, Any]:
        return {
            "chunks": len(self.vector_store),
            "documents": len({c.doc_id for c in self.vector_store.chunks}),
            "embedder": self.embedder.name,
            "embedding_dim": self.embedder.dim,
            # Report the backend `auto` resolves to rather than touching
            # `generator`, which would construct a client just to answer a
            # health check.
            "llm_backend": self.settings.resolved_llm_backend(),
            "reranker": getattr(self.retriever.reranker, "name", None),
        }
