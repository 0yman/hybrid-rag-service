"""Build the search index from a corpus directory.

    python scripts/ingest.py                      # defaults to data/corpus
    python scripts/ingest.py --path ~/my-docs
    python scripts/ingest.py --embedding-backend gemini
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag.config import get_settings  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=None, help="Corpus file or directory")
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument(
        "--embedding-backend", choices=["gemini", "local", "hash"], default=None
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    overrides = {
        key: value
        for key, value in {
            "index_dir": args.index_dir,
            "embedding_backend": args.embedding_backend,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
        }.items()
        if value is not None
    }
    settings = get_settings(**overrides)
    corpus = args.path or settings.corpus_dir

    if not Path(corpus).exists():
        print(
            f"Corpus not found at {corpus}.\n"
            "Run `python scripts/fetch_corpus.py` first, or pass --path.",
            file=sys.stderr,
        )
        return 1

    print(f"Corpus:    {corpus}")
    print(f"Embedder:  {settings.embedding_backend}")
    print(f"Chunking:  {settings.chunk_size} words, {settings.chunk_overlap} overlap")

    started = time.perf_counter()
    pipeline = RAGPipeline.create(settings)
    documents, chunks = pipeline.ingest_path(Path(corpus))
    if chunks == 0:
        print("No supported documents found (.txt, .md, .pdf).", file=sys.stderr)
        return 1
    pipeline.save()
    elapsed = time.perf_counter() - started

    print(
        f"\nIndexed {documents} documents into {chunks} chunks "
        f"in {elapsed:.1f}s -> {settings.index_dir}"
    )
    for key, value in pipeline.stats().items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
