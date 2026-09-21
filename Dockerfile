# Slim rather than alpine: faiss and torch ship manylinux wheels, and alpine's
# musl libc would force a source build of both.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes,
# so ordinary code edits rebuild in seconds instead of re-downloading torch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY eval/ ./eval/
COPY scripts/ ./scripts/
COPY data/corpus/ ./data/corpus/

# Build the index at image build time so the container starts ready to serve.
# Uses the local embedder, so no API key is needed to produce a working image.
RUN python scripts/ingest.py --embedding-backend local

# Run as a non-root user; the index directory has to be writable for /ingest.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health', timeout=4).json()['status']=='ok' else 1)"

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
