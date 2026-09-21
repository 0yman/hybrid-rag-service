"""Sentence-aware chunking with overlap.

Splitting on a fixed character count is the usual shortcut and it routinely
cuts sentences — and therefore facts — in half. This packs whole sentences up
to a word budget instead, and carries a configurable tail of sentences into
the next chunk so a fact that straddles a boundary survives in both.
"""

from __future__ import annotations

import re

from .models import Chunk, Document

# A sentence boundary is punctuation + whitespace, but plenty of periods are
# not boundaries at all. Rather than an unreadable lookbehind chain, candidate
# boundaries are found first and then rejected if the token before them is a
# known abbreviation or a single initial ("J. Smith").
_ABBREVIATIONS = {
    "no", "vs", "etc", "fig", "figs", "approx", "dr", "mr", "mrs", "ms", "st",
    "jr", "sr", "prof", "inc", "ltd", "co", "corp", "dept", "est", "cf", "al",
    "e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "vol", "eq", "ref", "sec",
}
_BOUNDARY = re.compile(r"""(?<=[.!?])["')\]]*\s+""")
_TRAILING_TOKEN = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")


def split_sentences(text: str) -> list[str]:
    """Split a block of text into sentences, preserving their content."""
    text = text.strip()
    if not text:
        return []

    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        # `head` stops at the terminator so the abbreviation check sees the
        # period; the emitted sentence runs to `match.end()` so a closing
        # quote or bracket stays attached instead of being dropped.
        head = text[start : match.start()]
        token = _TRAILING_TOKEN.search(head)
        if token:
            word = token.group(1).lower().rstrip(".")
            # Abbreviations and initials end in a period without ending a sentence.
            if word in _ABBREVIATIONS or len(word) == 1:
                continue
        piece = text[start : match.end()].strip()
        if piece:
            sentences.append(piece)
        start = match.end()

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def _overlap_tail(sentences: list[str], overlap_words: int) -> list[str]:
    """The last few sentences of a chunk, totalling at most `overlap_words`."""
    if overlap_words <= 0:
        return []
    tail: list[str] = []
    budget = overlap_words
    for sentence in reversed(sentences):
        words = _word_count(sentence)
        if words > budget and tail:
            break
        tail.insert(0, sentence)
        budget -= words
        if budget <= 0:
            break
    return tail


def chunk_text(
    text: str,
    chunk_size: int = 220,
    chunk_overlap: int = 40,
    min_chunk_words: int = 25,
) -> list[str]:
    """Pack sentences into ~`chunk_size`-word chunks with sentence overlap."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if not current:
            return
        chunks.append(" ".join(current))
        tail = _overlap_tail(current, chunk_overlap)
        current = list(tail)
        current_words = sum(_word_count(s) for s in current)

    for paragraph in split_paragraphs(text):
        for sentence in split_sentences(paragraph):
            words = _word_count(sentence)
            # A single sentence longer than the budget becomes its own chunk
            # rather than being silently truncated.
            if words >= chunk_size:
                flush()
                if current:
                    chunks.append(" ".join(current))
                    current, current_words = [], 0
                chunks.append(sentence)
                continue
            if current_words + words > chunk_size:
                flush()
            current.append(sentence)
            current_words += words

    if current:
        chunks.append(" ".join(current))

    # Fold a too-small trailing chunk back into its predecessor; on its own it
    # carries too little context to ever retrieve well. Only do so when the
    # merged chunk still fits the budget - otherwise a stray short tail would
    # silently produce an oversized chunk, which is worse than a small one.
    if len(chunks) > 1 and _word_count(chunks[-1]) < min_chunk_words:
        merged = f"{chunks[-2]} {chunks[-1]}"
        if _word_count(merged) <= chunk_size:
            chunks[-2] = merged
            chunks.pop()

    return [c for c in chunks if c.strip()]


def chunk_document(
    document: Document,
    chunk_size: int = 220,
    chunk_overlap: int = 40,
    min_chunk_words: int = 25,
) -> list[Chunk]:
    texts = chunk_text(document.text, chunk_size, chunk_overlap, min_chunk_words)
    return [
        Chunk(
            chunk_id=f"{document.doc_id}::{i}",
            doc_id=document.doc_id,
            title=document.title,
            text=text,
            ordinal=i,
            source=document.source,
            metadata=dict(document.metadata),
        )
        for i, text in enumerate(texts)
    ]
