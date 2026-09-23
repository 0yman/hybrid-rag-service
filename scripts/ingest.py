"""Add documents to the app from the command line - the web page's upload
button, for folders too large to drag in.

    python scripts/ingest.py ~/Documents/contracts       # add a folder
    python scripts/ingest.py report.pdf                  # add one file
    python scripts/ingest.py ~/notes --fresh             # replace everything

Documents already in the index are updated in place, not duplicated.
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="A file or a folder of .txt, .md or .pdf files")
    parser.add_argument("--fresh", action="store_true", help="Remove everything already indexed first")
    parser.add_argument("--index-dir", type=Path, default=None, help="Index location (default: the app's)")
    parser.add_argument("--embedding-backend", choices=["gemini", "openai", "local", "hash"], default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.path.exists():
        print(f"Nothing at {args.path}.", file=sys.stderr)
        return 1

    overrides = {
        key: value
        for key, value in {
            "index_dir": args.index_dir,
            "embedding_backend": args.embedding_backend,
        }.items()
        if value is not None
    }
    settings = get_settings(**overrides)

    started = time.perf_counter()
    pipeline = RAGPipeline.create(settings) if args.fresh else RAGPipeline.open(settings)
    before = len(pipeline.list_documents())
    documents, chunks = pipeline.ingest_path(args.path)
    if chunks == 0:
        print("No readable .txt, .md or .pdf files found there.", file=sys.stderr)
        return 1
    pipeline.save()

    total = len(pipeline.list_documents())
    print(
        f"Indexed {documents} document(s) into {chunks} passages in "
        f"{time.perf_counter() - started:.1f}s. The app now has {total} document(s)"
        + ("" if args.fresh else f" ({total - before:+d})")
        + "."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
