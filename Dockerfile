# Slim rather than alpine: faiss and onnxruntime ship manylinux wheels, and
# alpine's musl libc would force a source build of both.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    # Inside /app, not fastembed's default of /tmp, so it is owned by the
    # non-root user below along with everything else.
    FASTEMBED_CACHE_PATH=/app/models

WORKDIR /app

# Dependencies first: this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/benchmark/ ./data/benchmark/

# Download the embedding model at build time, so the container works with no
# network access and the first upload is not a surprise minute-long wait.
RUN python -c "import sys; sys.path.insert(0, 'src'); \
from rag.config import get_settings; from rag.embeddings import get_embedder; \
get_embedder(get_settings())"

# The data directories must exist in the image: a named volume mounted over a
# missing path comes up owned by root, and the app user could not save uploads.
RUN mkdir -p data/index data/uploads \
    && useradd --create-home --uid 1000 app \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
