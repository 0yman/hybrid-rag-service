"""The extractive answer engine - the app's default when no API key is set.

It is what a first-time user meets, so its two failure modes matter most:
declining a question whose answer is sitting in the documents, and quoting
something unrelated as though it answered. A fake embedder with hand-set
similarities makes both testable exactly, with no model download.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag.generator import SYSTEM_PROMPT, build_prompt
from rag.llm import ExtractiveLLM
from rag.models import Chunk, ScoredChunk

WARRANTY = "The warranty covers manufacturing defects for twenty four months from purchase."
RECEIPT = "Claims must be made in writing and include the original receipt."
SHIPPING = "Orders ship within three working days of payment."


class FakeEmbedder:
    """Vectors chosen so the cosine between two texts is whatever the test
    says it is.

    Unlisted questions and unlisted sentences default to two *different*
    axes. Giving them the same default vector would make every unknown
    sentence a perfect match for every unknown question - cosine 1.0, the
    opposite of unrelated.
    """

    semantic = True
    dim = 4
    _UNKNOWN_QUERY = np.array([0, 0, 1, 0], dtype=np.float32)
    _UNKNOWN_DOC = np.array([0, 0, 0, 1], dtype=np.float32)

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = {key: np.asarray(v, dtype=np.float32) / np.linalg.norm(v) for key, v in table.items()}

    def embed_query(self, text: str) -> np.ndarray:
        return self.table.get(text, self._UNKNOWN_QUERY)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.table.get(t, self._UNKNOWN_DOC) for t in texts])


def prompt_for(question: str, *sentences: str) -> str:
    chunk = Chunk(
        chunk_id="doc::0", doc_id="doc", title="Warranty terms",
        text=" ".join(sentences), ordinal=0, source="terms.txt",
    )
    return build_prompt(question, [ScoredChunk(chunk=chunk, score=1.0)])


def cosine_pair(similarity: float) -> tuple[list[float], list[float]]:
    """Two vectors whose cosine is exactly `similarity`."""
    return [1.0, 0.0, 0.0, 0.0], [similarity, float(np.sqrt(1 - similarity**2)), 0.0, 0.0]


QUESTION = "How long does the warranty last?"


class TestParaphrase:
    def test_word_matching_misses_a_paraphrase(self):
        """The failure that motivated matching by meaning: one shared word."""
        engine = ExtractiveLLM(embedder=None)
        result = engine.generate(SYSTEM_PROMPT, prompt_for(QUESTION, WARRANTY, RECEIPT))
        assert result.text == "INSUFFICIENT_CONTEXT"

    def test_meaning_matching_finds_it(self):
        q, s = cosine_pair(0.62)
        engine = ExtractiveLLM(embedder=FakeEmbedder({QUESTION: q, WARRANTY: s}))
        result = engine.generate(SYSTEM_PROMPT, prompt_for(QUESTION, WARRANTY, RECEIPT))
        assert "twenty four months" in result.text
        assert result.text.endswith("[1]")


class TestThreshold:
    # Either side of the floor rather than on it: float32 rounding makes an
    # exact 0.45 land on whichever side it likes.
    @pytest.mark.parametrize("similarity,answers", [(0.44, False), (0.46, True), (0.70, True)])
    def test_the_similarity_floor_decides(self, similarity, answers):
        q, s = cosine_pair(similarity)
        engine = ExtractiveLLM(embedder=FakeEmbedder({QUESTION: q, WARRANTY: s}), min_similarity=0.45)
        result = engine.generate(SYSTEM_PROMPT, prompt_for(QUESTION, WARRANTY))
        assert (result.text != "INSUFFICIENT_CONTEXT") is answers

    def test_an_unrelated_question_is_declined(self):
        """Nothing in the passage is close in meaning: say so, quote nothing."""
        engine = ExtractiveLLM(embedder=FakeEmbedder({}))
        result = engine.generate(SYSTEM_PROMPT, prompt_for("Who painted the Mona Lisa?", WARRANTY, SHIPPING))
        assert result.text == "INSUFFICIENT_CONTEXT"

    def test_weak_matches_are_not_padded_in(self):
        """One strong match should not be diluted with two weak ones just
        because the engine is allowed up to three sentences."""
        strong_q, strong_s = cosine_pair(0.80)
        _, weak = cosine_pair(0.50)  # above the floor, but far below the best
        engine = ExtractiveLLM(
            embedder=FakeEmbedder({QUESTION: strong_q, WARRANTY: strong_s, SHIPPING: weak})
        )
        result = engine.generate(SYSTEM_PROMPT, prompt_for(QUESTION, WARRANTY, SHIPPING))
        assert "twenty four months" in result.text
        assert "three working days" not in result.text


class TestQuoting:
    def test_sentences_repeated_by_chunk_overlap_are_quoted_once(self):
        q, s = cosine_pair(0.9)
        engine = ExtractiveLLM(embedder=FakeEmbedder({QUESTION: q, WARRANTY: s}))
        chunks = [
            ScoredChunk(
                chunk=Chunk(chunk_id=f"d::{i}", doc_id="d", title="T", text=WARRANTY,
                            ordinal=i, source="t.txt"),
                score=1.0,
            )
            for i in range(2)
        ]
        result = engine.generate(SYSTEM_PROMPT, build_prompt(QUESTION, chunks))
        assert result.text.count("twenty four months") == 1

    def test_headings_are_never_quoted(self):
        engine = ExtractiveLLM(embedder=None)
        chunk = Chunk(
            chunk_id="d::0", doc_id="d", title="Warranty terms",
            text="# Warranty terms\n\n## Coverage\nThe warranty covers defects for twenty four months.",
            ordinal=0, source="t.md",
        )
        result = engine.generate(
            SYSTEM_PROMPT,
            build_prompt("How many months does the warranty cover defects?", [ScoredChunk(chunk=chunk, score=1.0)]),
        )
        assert "#" not in result.text
        assert "Coverage" not in result.text

    def test_no_passages_means_no_answer(self):
        engine = ExtractiveLLM(embedder=None)
        assert engine.generate(SYSTEM_PROMPT, "Question: anything?").text == "INSUFFICIENT_CONTEXT"


class TestPipelineWiring:
    def test_a_real_embedder_gets_meaning_matching(self, pipeline):
        pipeline.embedder.semantic = True  # pretend the hashing stub understands meaning
        try:
            engine = pipeline._extractive_engine()
            assert engine.embedder is pipeline.embedder
        finally:
            pipeline.embedder.semantic = False

    def test_the_hashing_stub_gets_word_matching(self, pipeline):
        assert pipeline._extractive_engine().embedder is None

    def test_the_threshold_comes_from_settings(self, pipeline):
        pipeline.settings = pipeline.settings.model_copy(update={"extractive_min_similarity": 0.6})
        assert pipeline._extractive_engine().min_similarity == 0.6
