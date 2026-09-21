from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag import api
from rag.pipeline import RAGPipeline


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, settings):
    """Point the app at the test settings.

    `lifespan` and `/ingest` both call `get_settings()` themselves, which would
    otherwise read the real `data/index` - loading a sentence-transformers
    model and making every test depend on whatever the developer last
    ingested. Redirecting it keeps the suite offline, fast and hermetic.
    """
    monkeypatch.setattr(api, "get_settings", lambda **kwargs: settings)


@pytest.fixture
def client(pipeline: RAGPipeline):
    with TestClient(api.app) as test_client:
        api._state["pipeline"] = pipeline  # lifespan runs on enter and clears it
        yield test_client
    api._state["pipeline"] = None


@pytest.fixture
def empty_client():
    with TestClient(api.app) as test_client:
        api._state["pipeline"] = None
        yield test_client


class TestHealth:
    def test_reports_loaded_index(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["index_loaded"] is True
        assert body["stats"]["chunks"] > 0

    def test_health_works_without_an_index(self, empty_client):
        body = empty_client.get("/health").json()
        assert body["status"] == "ok"
        assert body["index_loaded"] is False


class TestQuery:
    def test_returns_answer_with_citations_and_contexts(self, client):
        response = client.post(
            "/query", json={"question": "What was the average container dwell time?"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["abstained"] is False
        assert body["citations"]
        assert body["contexts"]
        assert all(c["marker"] == i for i, c in enumerate(body["contexts"], start=1))

    def test_abstains_out_of_corpus(self, client):
        body = client.post("/query", json={"question": "Who won the 1998 World Cup?"}).json()
        assert body["abstained"] is True
        assert body["citations"] == []

    def test_top_k_is_honoured(self, client):
        body = client.post(
            "/query", json={"question": "container dwell time", "top_k": 1}
        ).json()
        assert len(body["contexts"]) == 1

    def test_component_scores_are_exposed_for_debugging(self, client):
        body = client.post("/query", json={"question": "berth productivity"}).json()
        assert body["contexts"][0]["component_scores"]

    @pytest.mark.parametrize(
        "payload",
        [{}, {"question": ""}, {"question": "ok", "top_k": 0}, {"question": "ok", "top_k": 99}],
    )
    def test_invalid_payloads_are_rejected(self, client, payload):
        assert client.post("/query", json=payload).status_code == 422

    def test_query_without_an_index_is_service_unavailable(self, empty_client):
        response = empty_client.post("/query", json={"question": "anything"})
        assert response.status_code == 503
        assert "ingest" in response.json()["detail"].lower()


class TestIngest:
    def test_ingesting_a_directory_grows_the_index(self, client, corpus_dir):
        before = client.get("/stats").json()["chunks"]
        body = client.post("/ingest", json={"path": str(corpus_dir)}).json()
        assert body["documents"] == 3
        assert body["total_chunks"] == before + body["chunks"]

    def test_missing_path_is_a_client_error(self, client):
        response = client.post("/ingest", json={"path": "/definitely/not/here"})
        assert response.status_code == 400

    def test_directory_with_no_supported_files(self, client, tmp_path):
        empty = tmp_path / "nothing"
        empty.mkdir()
        (empty / "notes.xlsx").write_text("binary-ish", encoding="utf-8")
        response = client.post("/ingest", json={"path": str(empty)})
        assert response.status_code == 400
        assert ".md" in response.json()["detail"]
