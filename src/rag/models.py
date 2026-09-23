"""Core data structures shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    ordinal: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Chunk:
        return cls(**raw)


@dataclass(slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    # Where the chunk came from, e.g. {"dense": 0.81, "lexical": 12.4}.
    component_scores: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class Answer:
    question: str
    text: str
    contexts: list[ScoredChunk]
    cited_ordinals: list[int]
    abstained: bool = False
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    # Which engine actually wrote the answer - it can differ from the one
    # configured when a hosted model is unavailable and the pipeline falls back.
    engine: str = ""
    # Set when something the reader should know about happened, such as that
    # fallback. None on an ordinary answer.
    notice: str | None = None

    @property
    def citations(self) -> list[dict[str, Any]]:
        """The subset of retrieved contexts the answer actually cited."""
        out = []
        for i in self.cited_ordinals:
            if 1 <= i <= len(self.contexts):
                c = self.contexts[i - 1].chunk
                out.append(
                    {"marker": i, "doc_id": c.doc_id, "title": c.title, "source": c.source}
                )
        return out
