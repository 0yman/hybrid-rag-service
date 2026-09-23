"""Grounded answer generation.

Two properties matter more than fluency here:

1. **Citations.** Every claim carries a ``[n]`` marker pointing at the passage
   it came from, so an answer can be audited instead of trusted.
2. **Abstention.** When the retrieved passages do not contain the answer, the
   model is required to say so. A RAG system that always answers is a RAG
   system that hallucinates on out-of-corpus questions.
"""

from __future__ import annotations

import re
import time

from .llm import LLMClient
from .models import Answer, ScoredChunk

ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"

SYSTEM_PROMPT = f"""You are a precise question-answering assistant. You answer \
ONLY from the numbered context passages you are given.

Rules:
1. Every factual sentence must end with a citation marker naming the passage \
it came from, like [1] or [2][3].
2. Never use knowledge that is not in the passages, even if you are confident \
it is correct.
3. If the passages do not contain enough information to answer, reply with \
exactly {ABSTAIN_TOKEN} and nothing else. Do not guess, and do not partially \
answer from outside knowledge.
4. Be concise: two to four sentences unless the question demands more.
5. If passages disagree, say so and cite both."""

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")


def build_prompt(question: str, contexts: list[ScoredChunk]) -> str:
    """Assemble the user turn: question first, then numbered passages.

    Markers are 1-based and positional, so ``[2]`` always means
    ``contexts[1]`` - the model never sees internal chunk ids.
    """
    lines = [f"Question: {question}", "", "Context passages:"]
    for i, scored in enumerate(contexts, start=1):
        chunk = scored.chunk
        lines.append(f"[{i}] {chunk.title} - {chunk.source}")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines).strip()


def parse_citations(text: str, max_marker: int) -> list[int]:
    """Extract valid citation markers, in order of first appearance.

    Markers outside the passage range are dropped: a model that invents [9]
    when it was given five passages has cited nothing, and silently keeping
    the number would make the citation look checkable when it is not.
    """
    seen: list[int] = []
    for match in _CITATION_RE.finditer(text):
        marker = int(match.group(1))
        if 1 <= marker <= max_marker and marker not in seen:
            seen.append(marker)
    return seen


def strip_invalid_citations(text: str, max_marker: int) -> str:
    def replace(match: re.Match[str]) -> str:
        marker = int(match.group(1))
        return match.group(0) if 1 <= marker <= max_marker else ""

    return _CITATION_RE.sub(replace, text)


class Generator:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(self, question: str, contexts: list[ScoredChunk]) -> Answer:
        started = time.perf_counter()

        if not contexts:
            return Answer(
                question=question,
                text="I could not find anything relevant in the indexed documents.",
                contexts=[],
                cited_ordinals=[],
                abstained=True,
                latency_ms=(time.perf_counter() - started) * 1000,
                engine=self.llm.name,
            )

        prompt = build_prompt(question, contexts)
        response = self.llm.generate(SYSTEM_PROMPT, prompt)
        text = response.text.strip()

        abstained = ABSTAIN_TOKEN in text
        if abstained:
            text = (
                "The indexed documents do not contain enough information to "
                "answer this question."
            )
            citations: list[int] = []
        else:
            citations = parse_citations(text, len(contexts))
            text = strip_invalid_citations(text, len(contexts))

        return Answer(
            question=question,
            text=text,
            contexts=contexts,
            cited_ordinals=citations,
            abstained=abstained,
            usage=response.usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            engine=self.llm.name,
        )
