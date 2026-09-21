"""Turn files on disk into `Document` objects.

Supports .txt, .md and .pdf. Unknown extensions are skipped rather than
raising, so pointing the loader at a messy folder still works.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from .models import Document

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


def _doc_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]


def _title_from(path: Path, text: str) -> str:
    # A leading markdown H1 is a better title than the filename.
    match = re.match(r"\s*#\s+(.+)", text)
    if match:
        return match.group(1).strip()
    return path.stem.replace("_", " ").replace("-", " ").strip()


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Reading PDFs requires `pip install pypdf`") from exc

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)


def load_file(path: Path) -> Document | None:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return None

    text = _read_pdf(path) if suffix == ".pdf" else path.read_text(encoding="utf-8", errors="replace")
    text = normalize_whitespace(text)
    if not text.strip():
        return None

    return Document(
        doc_id=_doc_id(path),
        title=_title_from(path, text),
        text=text,
        source=path.name,
        metadata={"path": str(path), "chars": len(text)},
    )


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ blank lines into a paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_directory(directory: Path, patterns: Iterable[str] = ("**/*",)) -> Iterator[Document]:
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            if not path.is_file() or str(path) in seen:
                continue
            seen.add(str(path))
            doc = load_file(path)
            if doc is not None:
                yield doc
