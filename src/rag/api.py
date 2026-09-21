"""FastAPI service wrapping the pipeline.

The index is loaded once at startup rather than per request - embedding models
and FAISS indexes are expensive to build and cheap to reuse.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import get_settings
from .pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_state: dict[str, Any] = {"pipeline": None}


def get_pipeline() -> RAGPipeline:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Index is not loaded. POST /ingest or run scripts/ingest.py.",
        )
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        _state["pipeline"] = RAGPipeline.load(settings)
        logger.info("Loaded index: %s", _state["pipeline"].stats())
    except (FileNotFoundError, ValueError) as exc:
        # Starting without an index is legitimate - /ingest can build one.
        # Failing startup here would make the container crash-loop instead.
        logger.warning("Starting with no index: %s", exc)
        _state["pipeline"] = None
    yield
    _state.clear()


app = FastAPI(
    title="Hybrid RAG Service",
    version="1.0.0",
    description=(
        "Retrieval-augmented question answering over a document corpus, using "
        "dense + BM25 retrieval fused with Reciprocal Rank Fusion. Answers "
        "carry citations and abstain when the corpus does not support them."
    ),
    lifespan=lifespan,
)

try:  # optional: /metrics for Prometheus scraping
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, include_in_schema=False)
except ImportError:  # pragma: no cover
    logger.info("prometheus-fastapi-instrumentator not installed; /metrics disabled")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    marker: int
    doc_id: str
    title: str
    source: str


class ContextOut(BaseModel):
    marker: int
    chunk_id: str
    title: str
    source: str
    score: float
    component_scores: dict[str, float]
    preview: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    abstained: bool
    citations: list[Citation]
    contexts: list[ContextOut]
    latency_ms: float
    usage: dict[str, int]


class IngestRequest(BaseModel):
    path: str = Field(description="File or directory path readable by the service")


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    total_chunks: int


@app.get("/health")
def health() -> dict[str, Any]:
    pipeline = _state.get("pipeline")
    return {
        "status": "ok",
        "index_loaded": pipeline is not None,
        "stats": pipeline.stats() if pipeline else None,
    }


@app.get("/stats")
def stats() -> dict[str, Any]:
    return get_pipeline().stats()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    pipeline = get_pipeline()
    answer = pipeline.query(request.question, request.top_k)
    return QueryResponse(
        question=answer.question,
        answer=answer.text,
        abstained=answer.abstained,
        citations=[Citation(**c) for c in answer.citations],
        contexts=[
            ContextOut(
                marker=i,
                chunk_id=scored.chunk.chunk_id,
                title=scored.chunk.title,
                source=scored.chunk.source,
                score=round(scored.score, 6),
                component_scores={
                    k: round(v, 6) for k, v in scored.component_scores.items()
                },
                preview=scored.chunk.text[:300],
            )
            for i, scored in enumerate(answer.contexts, start=1)
        ],
        latency_ms=round(answer.latency_ms, 2),
        usage=answer.usage,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    path = Path(request.path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {path}")

    settings = get_settings()
    pipeline = _state.get("pipeline")
    if pipeline is None:
        pipeline = RAGPipeline.create(settings)
        _state["pipeline"] = pipeline

    documents, chunks = pipeline.ingest_path(path)
    if chunks == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No supported documents found at {path} (.txt, .md, .pdf).",
        )
    pipeline.save()
    return IngestResponse(
        documents=documents, chunks=chunks, total_chunks=len(pipeline.vector_store)
    )
