from __future__ import annotations

import json

import numpy as np
import pytest

from rag.embeddings import HashEmbedder, l2_normalize
from rag.generator import build_prompt, parse_citations, strip_invalid_citations
from rag.lexical import BM25Index, tokenize
from rag.models import ScoredChunk
from rag.pipeline import RAGPipeline
from rag.vectorstore import VectorStore


class TestEmbeddings:
    def test_vectors_are_unit_length(self):
        embedder = HashEmbedder()
        vectors = embedder.embed_documents(["one two three", "four five six"])
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    def test_embedding_is_deterministic(self):
        a = HashEmbedder().embed_query("container dwell time")
        b = HashEmbedder().embed_query("container dwell time")
        assert np.allclose(a, b)

    def test_similar_text_scores_above_unrelated_text(self):
        embedder = HashEmbedder()
        docs = embedder.embed_documents(["port container terminal", "chocolate cake recipe"])
        query = embedder.embed_query("container port")
        assert (docs @ query)[0] > (docs @ query)[1]

    def test_normalising_a_zero_vector_does_not_divide_by_zero(self):
        result = l2_normalize(np.zeros((1, 4), dtype=np.float32))
        assert np.all(np.isfinite(result))


class TestVectorStore:
    def test_rejects_mismatched_counts(self, sample_chunks):
        store = VectorStore(dim=4)
        with pytest.raises(ValueError, match="line up"):
            store.add(sample_chunks, np.zeros((2, 4), dtype=np.float32))

    def test_rejects_wrong_dimension(self, sample_chunks):
        store = VectorStore(dim=4)
        with pytest.raises(ValueError, match="dim"):
            store.add(sample_chunks, np.zeros((3, 8), dtype=np.float32))

    def test_search_on_an_empty_index_returns_nothing(self):
        assert VectorStore(dim=4).search(np.zeros(4, dtype=np.float32), 5) == []

    def test_asking_for_more_than_exists_is_clamped(self, sample_chunks):
        embedder = HashEmbedder()
        store = VectorStore(embedder.dim)
        store.add(sample_chunks, embedder.embed_documents([c.text for c in sample_chunks]))
        assert len(store.search(embedder.embed_query("alpha"), 100)) == 3

    def test_save_and_load_roundtrip(self, tmp_path, sample_chunks):
        embedder = HashEmbedder()
        store = VectorStore(embedder.dim, embedder.name)
        store.add(sample_chunks, embedder.embed_documents([c.text for c in sample_chunks]))
        store.save(tmp_path)

        restored = VectorStore.load(tmp_path)
        assert len(restored) == 3
        assert restored.embedder_name == embedder.name
        assert [c.chunk_id for c in restored.chunks] == [c.chunk_id for c in sample_chunks]

    def test_loading_a_missing_index_is_explicit(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ingest"):
            VectorStore.load(tmp_path / "nothing")


class TestLexical:
    def test_stopwords_removed_numbers_kept(self):
        assert tokenize("The berth is at No. 4") == ["berth", "no", "4"]

    def test_exact_rare_token_is_found(self, sample_chunks):
        index = BM25Index()
        index.add(sample_chunks)
        results = index.search("epsilon", 3)
        assert results and results[0].chunk.text == "delta epsilon zeta"

    def test_query_of_only_stopwords_returns_nothing(self, sample_chunks):
        index = BM25Index()
        index.add(sample_chunks)
        assert index.search("the and of", 3) == []

    def test_empty_index_returns_nothing(self):
        assert BM25Index().search("anything", 3) == []


class TestPromptAndCitations:
    def test_markers_are_positional_and_one_based(self, sample_chunks):
        contexts = [ScoredChunk(chunk=c, score=1.0) for c in sample_chunks]
        prompt = build_prompt("What is alpha?", contexts)
        assert "[1] Doc - doc.md" in prompt
        assert "[3] Doc - doc.md" in prompt
        assert "Question: What is alpha?" in prompt

    def test_valid_markers_kept_in_order_of_appearance(self):
        assert parse_citations("Claim [2]. Another [1]. Repeat [2].", 3) == [2, 1]

    def test_out_of_range_markers_are_dropped(self):
        assert parse_citations("Invented source [9].", 3) == []

    def test_out_of_range_markers_are_stripped_from_the_text(self):
        assert strip_invalid_citations("A [1] and B [9].", 3) == "A [1] and B ."


class TestPipeline:
    def test_ingest_produces_chunks_in_both_indexes(self, pipeline):
        stats = pipeline.stats()
        assert stats["documents"] == 3
        assert stats["chunks"] > 0
        assert len(pipeline.bm25_index) == len(pipeline.vector_store)

    def test_answers_an_in_corpus_question_with_a_citation(self, pipeline):
        answer = pipeline.query("What was the average container dwell time?")
        assert not answer.abstained
        assert answer.cited_ordinals
        assert "6.2" in answer.text

    def test_abstains_on_an_out_of_corpus_question(self, pipeline):
        answer = pipeline.query("Who won the 1998 football World Cup?")
        assert answer.abstained
        assert answer.citations == []

    def test_every_citation_points_at_a_retrieved_context(self, pipeline):
        answer = pipeline.query("How many moves per hour did berth 4 average?")
        for citation in answer.citations:
            assert 1 <= citation["marker"] <= len(answer.contexts)

    def test_keyword_query_reaches_the_right_document(self, pipeline):
        """The case dense retrieval alone handles worst: a bare rare token."""
        contexts = pipeline.retrieve("demurrage free time")
        assert "customs.md" in {c.chunk.source for c in contexts}

    def test_save_and_load_preserves_answers(self, settings, corpus_dir):
        original = RAGPipeline.create(settings)
        original.ingest_path(corpus_dir)
        original.save()

        restored = RAGPipeline.load(settings)
        question = "What was the average container dwell time?"
        assert restored.query(question).text == original.query(question).text

    def test_loading_with_a_different_embedder_is_refused(self, settings, corpus_dir):
        """Vectors from two different models are not comparable; failing loudly
        beats returning quietly meaningless results.

        The saved index metadata is edited rather than switching to a real
        second backend, so the test stays offline and fast.
        """
        pipe = RAGPipeline.create(settings)
        pipe.ingest_path(corpus_dir)
        pipe.save()

        info_path = settings.index_dir / "index_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["embedder"] = "some-other-model"
        info_path.write_text(json.dumps(info), encoding="utf-8")

        with pytest.raises(ValueError, match="re-ingest"):
            RAGPipeline.load(settings)

    def test_empty_directory_ingests_nothing(self, settings, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert RAGPipeline.create(settings).ingest_path(empty) == (0, 0)
