"""Backend selection and the OpenAI-compatible adapters.

The pipeline is written against `Embedder` and `LLMClient`, so swapping
providers should be a config change. These tests hold that line: they check
the factories route correctly, that a missing key fails with a message that
says what to do, and that the OpenAI response mapping produces the shapes the
rest of the pipeline expects - all without a network call or an API key.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rag.config import get_settings
from rag.embeddings import HashEmbedder, get_embedder
from rag.llm import ExtractiveLLM, get_llm


class TestBackendSelection:
    def test_hash_embedder_is_selected(self, settings):
        assert isinstance(get_embedder(settings), HashEmbedder)

    def test_extractive_engine_is_selected(self, settings):
        assert isinstance(get_llm(settings), ExtractiveLLM)

    def test_auto_with_no_keys_answers_extractively(self, settings):
        """The first-run experience: no .env at all still answers."""
        auto = settings.model_copy(
            update={"llm_backend": "auto", "google_api_key": None, "openai_api_key": None}
        )
        assert auto.resolved_llm_backend() == "extractive"
        assert isinstance(get_llm(auto), ExtractiveLLM)

    def test_auto_prefers_gemini_then_openai(self, settings):
        both = settings.model_copy(
            update={"llm_backend": "auto", "google_api_key": "g", "openai_api_key": "o"}
        )
        assert both.resolved_llm_backend() == "gemini"
        openai_only = both.model_copy(update={"google_api_key": None})
        assert openai_only.resolved_llm_backend() == "openai"

    def test_an_explicit_choice_beats_auto(self, settings):
        forced = settings.model_copy(update={"llm_backend": "extractive", "google_api_key": "g"})
        assert forced.resolved_llm_backend() == "extractive"

    def test_unknown_embedding_backend_is_rejected(self, settings):
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            get_embedder(settings.model_copy(update={"embedding_backend": "word2vec"}))

    def test_unknown_llm_backend_is_rejected(self, settings):
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            get_llm(settings.model_copy(update={"llm_backend": "llama"}))


class TestMissingCredentials:
    """A missing key should say which one and how to proceed, not raise a
    bare KeyError three frames deep inside an SDK."""

    def test_openai_llm_without_a_key(self, settings):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            get_llm(settings.model_copy(update={"llm_backend": "openai", "openai_api_key": None}))

    def test_openai_embedder_without_a_key(self, settings):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            get_embedder(
                settings.model_copy(
                    update={"embedding_backend": "openai", "openai_api_key": None}
                )
            )

    def test_gemini_without_a_key(self, settings):
        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            get_llm(settings.model_copy(update={"llm_backend": "gemini", "google_api_key": None}))

    def test_the_message_names_a_way_forward(self, settings):
        with pytest.raises(RuntimeError) as info:
            get_llm(settings.model_copy(update={"llm_backend": "gemini", "google_api_key": None}))
        assert "aistudio.google.com" in str(info.value)


class TestOpenAIResponseMapping:
    """Drive the adapters with fake SDK objects, so the mapping is verified
    without a key, a network call, or a bill."""

    def test_chat_response_is_mapped_to_llm_response(self, settings, monkeypatch):
        from rag import llm as llm_module

        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="  Answer [1].  "))],
                    usage=SimpleNamespace(prompt_tokens=42, completion_tokens=7),
                )

        class FakeClient:
            def __init__(self, **kwargs):
                self.init_kwargs = kwargs
                self.chat = SimpleNamespace(completions=FakeCompletions())

        monkeypatch.setattr("openai.OpenAI", FakeClient)
        client = llm_module.OpenAICompatibleLLM(
            settings.model_copy(update={"openai_api_key": "sk-test"})
        )
        result = client.generate("system rules", "Question: what?")

        assert result.text == "Answer [1]."  # whitespace stripped
        assert result.usage == {"prompt_tokens": 42, "output_tokens": 7}
        # System and user turns must arrive as separate messages.
        assert [m["role"] for m in captured["messages"]] == ["system", "user"]
        assert captured["messages"][0]["content"] == "system rules"

    def test_embeddings_are_reordered_and_normalised(self, settings, monkeypatch):
        from rag import embeddings as embeddings_module

        class FakeEmbeddings:
            def create(self, **kwargs):
                # Deliberately out of order: the API does not promise order,
                # so the adapter must sort by index or vectors get attached
                # to the wrong chunks.
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(index=1, embedding=[0.0, 3.0, 0.0, 0.0]),
                        SimpleNamespace(index=0, embedding=[4.0, 0.0, 0.0, 0.0]),
                    ]
                )

        class FakeClient:
            def __init__(self, **kwargs):
                self.embeddings = FakeEmbeddings()

        monkeypatch.setattr("openai.OpenAI", FakeClient)
        embedder = embeddings_module.OpenAIEmbedder(
            settings.model_copy(update={"openai_api_key": "sk-test", "openai_embedding_dim": 4})
        )
        vectors = embedder.embed_documents(["first", "second"])

        assert vectors.shape == (2, 4)
        assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
        # Row 0 must be the vector whose index was 0, not the first one sent back.
        assert vectors[0].argmax() == 0
        assert vectors[1].argmax() == 1

    def test_embedding_dimension_is_reported_before_any_call(self, settings, monkeypatch):
        from rag import embeddings as embeddings_module

        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: SimpleNamespace(embeddings=None))
        embedder = embeddings_module.OpenAIEmbedder(
            settings.model_copy(update={"openai_api_key": "sk-test", "openai_embedding_dim": 256})
        )
        assert embedder.dim == 256
        assert "256" in embedder.name


class TestIndexCompatibility:
    def test_index_records_which_embedder_built_it(self, settings, corpus_dir):
        """Vectors from two providers are not comparable, so the index has to
        carry the name that produced it."""
        from rag.pipeline import RAGPipeline

        pipeline = RAGPipeline.create(settings)
        pipeline.ingest_path(corpus_dir)
        pipeline.save()

        info = (settings.index_dir / "index_info.json").read_text(encoding="utf-8")
        assert "hash-512" in info


def test_settings_accept_an_alternative_base_url():
    """Groq, Together, OpenRouter and a local vLLM all speak this format;
    only the base URL changes."""
    settings = get_settings(
        llm_backend="openai",
        openai_base_url="https://api.groq.com/openai/v1",
        openai_chat_model="llama-3.3-70b-versatile",
    )
    assert settings.openai_base_url.endswith("/openai/v1")
    assert settings.openai_chat_model == "llama-3.3-70b-versatile"
