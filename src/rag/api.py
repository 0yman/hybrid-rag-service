"""FastAPI service: the web app at `/`, and the JSON API behind it.

The index is loaded once at startup rather than per request - embedding models
and FAISS indexes are expensive to build and cheap to reuse. An empty index is
a normal starting state, not an error: a first run opens straight onto the
upload screen.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .loaders import SUPPORTED_SUFFIXES, load_file
from .pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

_state: dict[str, Any] = {"pipeline": None, "error": None}
# One user, one index. FastAPI runs sync endpoints on a thread pool, so an
# upload and a question could otherwise race on the same FAISS index.
_lock = threading.Lock()

ENGINE_LABELS = {
    "gemini": "Google Gemini",
    "openai": "OpenAI-compatible model",
    "extractive": "Extractive (no AI model)",
}


def get_pipeline() -> RAGPipeline:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"The search index could not start: {_state.get('error') or 'unknown error'}",
        )
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        _state["pipeline"] = RAGPipeline.open(settings)
        _state["error"] = None
        logger.info("Ready: %s", _state["pipeline"].stats())
    except Exception as exc:
        # Most likely the embedding model could not load. Keep the server up
        # so the page can say so, instead of a connection-refused error.
        logger.exception("Could not open the index")
        _state["pipeline"], _state["error"] = None, str(exc)
    yield
    _state["pipeline"] = None


app = FastAPI(
    title="Hybrid RAG Service",
    version="2.0.0",
    description=(
        "Ask questions about your own documents. Dense + BM25 retrieval fused "
        "with Reciprocal Rank Fusion; answers carry citations and abstain when "
        "the documents do not support them."
    ),
    lifespan=lifespan,
)

try:  # optional: /metrics for Prometheus scraping
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, include_in_schema=False)
except ImportError:  # pragma: no cover
    pass


# --- models -----------------------------------------------------------------


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
    engine: str
    notice: str | None = None


class IngestRequest(BaseModel):
    path: str = Field(description="File or directory path readable by the service")


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    total_chunks: int


class UploadResult(BaseModel):
    filename: str
    status: str            # "added", "updated" or "skipped"
    chunks: int = 0
    reason: str | None = None


# --- helpers ----------------------------------------------------------------

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]+")
_MARKDOWN_HEADING = re.compile(r"(^|\s)#{1,6}\s+")


def _readable(text: str) -> str:
    """Passage text for display: heading markers off, whitespace collapsed.
    The stored chunk keeps them - they help retrieval - but a reader should
    not see '## Eligibility' in a quoted source."""
    return " ".join(_MARKDOWN_HEADING.sub(r"\1", text).split())


def safe_filename(name: str) -> str:
    """Reduce an uploaded filename to something safe to write to disk.

    `Path(name).name` drops any directory part, so `../../etc/passwd` cannot
    escape the uploads folder; the character filter then removes anything a
    filesystem might interpret.
    """
    base = Path(name or "").name
    stem, suffix = Path(base).stem, Path(base).suffix.lower()
    stem = _UNSAFE_CHARS.sub("_", stem).strip(" ._") or "document"
    return f"{stem[:120]}{suffix}"


def _save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> None:
    written = 0
    with destination.open("wb") as handle:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                handle.close()
                destination.unlink(missing_ok=True)
                raise ValueError(f"larger than the {max_bytes // (1024 * 1024)} MB limit")
            handle.write(chunk)


def _engine_status(pipeline: RAGPipeline) -> dict[str, Any]:
    backend = pipeline.settings.resolved_llm_backend()
    return {
        "engine": backend,
        "engine_label": ENGINE_LABELS.get(backend, backend),
        "has_model": backend != "extractive",
    }


# --- pages ------------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# --- status -----------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    pipeline = _state.get("pipeline")
    return {
        "status": "ok" if pipeline else "down",
        "index_loaded": pipeline is not None,
        "stats": pipeline.stats() if pipeline else None,
        "error": _state.get("error"),
    }


@app.get("/stats")
def stats() -> dict[str, Any]:
    return get_pipeline().stats()


@app.get("/status")
def status() -> dict[str, Any]:
    """Everything the web page needs to draw itself."""
    pipeline = get_pipeline()
    settings = pipeline.settings
    documents = pipeline.list_documents()
    return {
        **_engine_status(pipeline),
        "documents": len(documents),
        "chunks": len(pipeline.vector_store),
        "sample_available": settings.benchmark_dir.is_dir()
        and any(settings.benchmark_dir.iterdir()),
        "max_upload_mb": settings.max_upload_mb,
        "accepted_types": sorted(SUPPORTED_SUFFIXES),
    }


# --- documents --------------------------------------------------------------


@app.get("/documents")
def list_documents() -> list[dict[str, Any]]:
    return get_pipeline().list_documents()


@app.post("/documents", response_model=list[UploadResult])
def upload_documents(files: list[UploadFile] = File(...)) -> list[UploadResult]:
    pipeline = get_pipeline()
    settings = pipeline.settings
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_mb * 1024 * 1024

    results: list[UploadResult] = []
    to_ingest = []
    with _lock:
        already = pipeline.indexed_doc_ids()
        for upload in files:
            name = safe_filename(upload.filename or "")
            if Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
                results.append(UploadResult(
                    filename=upload.filename or name, status="skipped",
                    reason=f"unsupported type - use {', '.join(sorted(SUPPORTED_SUFFIXES))}",
                ))
                continue

            destination = settings.uploads_dir / name
            try:
                _save_upload(upload, destination, max_bytes)
                document = load_file(destination)
            except ValueError as exc:
                results.append(UploadResult(filename=name, status="skipped", reason=str(exc)))
                continue
            except Exception as exc:  # a corrupt or encrypted PDF, say
                destination.unlink(missing_ok=True)
                logger.warning("Could not read %s: %s", name, exc)
                results.append(UploadResult(
                    filename=name, status="skipped", reason="the file could not be read",
                ))
                continue

            if document is None:
                destination.unlink(missing_ok=True)
                results.append(UploadResult(
                    filename=name, status="skipped",
                    reason="no readable text found (a scanned PDF needs OCR first)",
                ))
                continue

            status_ = "updated" if document.doc_id in already else "added"
            to_ingest.append((document, name, status_))

        if to_ingest:
            pipeline.ingest_documents([doc for doc, _, _ in to_ingest])
            pipeline.save()
            chunk_counts = {d["doc_id"]: d["chunks"] for d in pipeline.list_documents()}
            for document, name, status_ in to_ingest:
                results.append(UploadResult(
                    filename=name, status=status_, chunks=chunk_counts.get(document.doc_id, 0),
                ))
    return results


@app.post("/documents/sample", response_model=IngestResponse)
def load_sample_documents() -> IngestResponse:
    pipeline = get_pipeline()
    sample_dir = pipeline.settings.benchmark_dir
    if not sample_dir.is_dir():
        raise HTTPException(status_code=404, detail="The sample documents are not installed.")
    with _lock:
        documents, chunks = pipeline.ingest_path(sample_dir)
        pipeline.save()
        return IngestResponse(documents=documents, chunks=chunks, total_chunks=len(pipeline.vector_store))


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, Any]:
    pipeline = get_pipeline()
    with _lock:
        match = next((d for d in pipeline.list_documents() if d["doc_id"] == doc_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail="No such document.")
        removed = pipeline.remove_document(doc_id)
        pipeline.save()
        # Only ever delete the copy this app made, never a file elsewhere.
        uploaded = pipeline.settings.uploads_dir / match["source"]
        if uploaded.is_file():
            uploaded.unlink()
    return {"removed_chunks": removed, "doc_id": doc_id}


@app.delete("/documents")
def clear_documents() -> dict[str, Any]:
    pipeline = get_pipeline()
    with _lock:
        count = len(pipeline.list_documents())
        pipeline.clear()
        pipeline.save()
        uploads = pipeline.settings.uploads_dir
        if uploads.is_dir():
            shutil.rmtree(uploads)
    return {"removed_documents": count}


# --- questions --------------------------------------------------------------


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    pipeline = get_pipeline()
    if len(pipeline.vector_store) == 0:
        raise HTTPException(
            status_code=409,
            detail="There are no documents yet. Upload some, or load the sample documents.",
        )
    with _lock:
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
                component_scores={k: round(v, 6) for k, v in scored.component_scores.items()},
                preview=_readable(scored.chunk.text)[:400],
            )
            for i, scored in enumerate(answer.contexts, start=1)
        ],
        latency_ms=round(answer.latency_ms, 2),
        usage=answer.usage,
        engine=answer.engine,
        notice=answer.notice,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Index a file or folder by server path. For scripts; the web page uploads."""
    path = Path(request.path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {path}")

    pipeline = get_pipeline()
    with _lock:
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
