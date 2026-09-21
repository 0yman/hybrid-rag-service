"""BM25 keyword retrieval.

Dense retrieval is strong on paraphrase and weak on rare exact tokens - a
berth code, an IMO number, a product SKU. BM25 is the opposite. Keeping both
and fusing them is why this pipeline beats either one alone; see fusion.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .models import Chunk, ScoredChunk

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "how", "in", "into", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "there", "these", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "will", "with",
}
_STATE_FILE = "lexical.jsonl"


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords.

    Numbers are kept deliberately - in a technical question they are often the
    single most discriminative token.
    """
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks.append(chunk)
            # Indexing the title too lets a question that names a document find
            # it even when the body never repeats that name.
            self._tokens.append(tokenize(f"{chunk.title} {chunk.text}"))
        self._rebuild()

    def _rebuild(self) -> None:
        # BM25Okapi computes corpus statistics up front, so it has to be
        # rebuilt whenever documents are added. Fine for batch ingestion.
        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if self._bm25 is None or not self._chunks:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        top_k = min(top_k, len(self._chunks))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            ScoredChunk(
                chunk=self._chunks[i],
                score=float(scores[i]),
                component_scores={"lexical": float(scores[i])},
            )
            for i in best
            if scores[i] > 0
        ]

    # --- persistence -----------------------------------------------------

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / _STATE_FILE).open("w", encoding="utf-8") as handle:
            for chunk in self._chunks:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        path = Path(directory) / _STATE_FILE
        index = cls()
        if not path.exists():
            return index
        with path.open(encoding="utf-8") as handle:
            chunks = [
                Chunk.from_dict(json.loads(line)) for line in handle if line.strip()
            ]
        index.add(chunks)
        return index
