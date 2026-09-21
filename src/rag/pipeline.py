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
from .llm import get_llm
from .loaders import load_directory, load_file
from .models import Answer, Chunk, Document, ScoredChunk
from .retriever import CrossEncoderReranker, HybridRetriever
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


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
            self._generator = Generator(get_llm(self.settings))
        return self._generator

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

    # --- ingestion -------------------------------------------------------

    def ingest_documents(self, documents: list[Document]) -> int:
        settings = self.settings
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

    # --- querying --------------------------------------------------------

    def retrieve(self, question: str, top_k: int | None = None) -> list[ScoredChunk]:
        contexts, _ = self.retriever.retrieve(question, top_k)
        return contexts

    def query(self, question: str, top_k: int | None = None) -> Answer:
        contexts, debug = self.retriever.retrieve(question, top_k)
        answer = self.generator.generate(question, contexts)
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
            # Report the configured backend rather than touching `generator`,
            # which would construct a client (and demand a key) just to answer
            # a health check.
            "llm_backend": self.settings.llm_backend,
            "reranker": getattr(self.retriever.reranker, "name", None),
        }
