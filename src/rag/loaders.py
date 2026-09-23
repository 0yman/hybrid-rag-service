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


def _doc_id(key: str) -> str:
    """A stable id from a *location-independent* key.

    This used to hash the absolute path, which made results depend on where
    the project was cloned: chunk ids break ties in rank fusion, so the same
    benchmark scored 0.933 in one folder and 0.867 in another. The key is now
    the path relative to the folder being indexed (or just the file name).
    """
    return hashlib.sha1(key.encode()).hexdigest()[:12]


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
    return "\n\n".join(reflow_pdf_text(page) for page in pages)


_SENTENCE_END = (".", "!", "?", ":", ";")


def reflow_pdf_text(text: str) -> str:
    """Rebuild paragraphs from a PDF's visual lines.

    PDF text comes out one printed line at a time: every line ends in a
    newline whether or not the sentence did, and headings are just short lines
    with nothing to mark them. Left like that, an answer quotes
    "Eligibility Employees become eligible after a three month probation
    period" - a heading glued onto a sentence broken where the page wrapped.

    Wrapped lines are joined with a space. A short line with no closing
    punctuation that starts a new block is taken to be a heading and marked
    as markdown, so the rest of the pipeline treats it exactly like a heading
    in a .md file - including the first one becoming the document's title.
    """
    lines = [line.strip() for line in text.split("\n")]
    out: list[str] = []
    paragraph: list[str] = []
    first_heading = True

    def flush() -> None:
        if paragraph:
            out.append(" ".join(paragraph))
            paragraph.clear()

    for i, line in enumerate(lines):
        if not line:
            flush()
            continue
        previous = next((lines[j] for j in range(i - 1, -1, -1) if lines[j]), "")
        following = next((lines[j] for j in range(i + 1, len(lines)) if lines[j]), "")
        is_heading = (
            len(line.split()) <= 8
            and not line.endswith((*_SENTENCE_END, ","))
            and (not previous or previous.endswith(_SENTENCE_END))
            and following[:1].isupper()
        )
        if is_heading:
            flush()
            # A single newline keeps the heading in the same block as the
            # paragraph it introduces, which is how the answer engine knows to
            # strip it rather than quote it.
            paragraph.append(("# " if first_heading else "## ") + line + "\n")
            first_heading = False
        elif paragraph and paragraph[-1].endswith("-"):
            paragraph[-1] = paragraph[-1] + line  # a word split across lines
        else:
            paragraph.append(line)
    flush()
    return "\n\n".join(block.replace("\n ", "\n") for block in out)


def load_file(path: Path, root: Path | None = None) -> Document | None:
    """Read one file. `root` is the folder it was found under, if any; its
    path relative to that folder identifies it, so the same file keeps the
    same id wherever the folder lives - and re-adding it replaces it."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return None

    text = _read_pdf(path) if suffix == ".pdf" else path.read_text(encoding="utf-8", errors="replace")
    text = normalize_whitespace(text)
    if not text.strip():
        return None

    key = path.relative_to(root).as_posix() if root is not None else path.name
    return Document(
        doc_id=_doc_id(key),
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
            doc = load_file(path, root=directory)
            if doc is not None:
                yield doc
