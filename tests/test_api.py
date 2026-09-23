"""The web app's API, exercised the way the page uses it.

Three starting states matter: a library with documents in it, an empty one (a
first run, which must be a normal state rather than an error), and a server
whose index failed to open at all.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from rag import api
from rag.pipeline import RAGPipeline


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, settings):
    """Point the app at the test settings, so no test reads the real data/ dir."""
    monkeypatch.setattr(api, "get_settings", lambda **kwargs: settings)


def _client_with(pipeline, error=None):
    test_client = TestClient(api.app)
    test_client.__enter__()
    api._state["pipeline"] = pipeline  # lifespan runs on enter and resets it
    api._state["error"] = error
    return test_client


@pytest.fixture
def client(pipeline: RAGPipeline):
    test_client = _client_with(pipeline)
    yield test_client
    test_client.__exit__(None, None, None)


@pytest.fixture
def empty_client(settings):
    test_client = _client_with(RAGPipeline.create(settings))
    yield test_client
    test_client.__exit__(None, None, None)


@pytest.fixture
def broken_client():
    test_client = _client_with(None, error="the embedding model could not load")
    yield test_client
    test_client.__exit__(None, None, None)


def upload(client, *files):
    """files: (name, bytes) pairs, sent the way a browser form sends them."""
    return client.post(
        "/documents",
        files=[("files", (name, io.BytesIO(data), "application/octet-stream")) for name, data in files],
    )


LONG_TEXT = (
    b"The warranty covers manufacturing defects for twenty four months from the date of "
    b"purchase. Claims must be made in writing and include the original receipt. Damage "
    b"caused by misuse, accidents or unauthorised repairs is not covered by this warranty. "
    b"Replacement parts are guaranteed for the remainder of the original warranty period."
)


class TestPage:
    def test_the_web_app_is_served_at_the_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "Ask your documents" in response.text


class TestHealthAndStatus:
    def test_health_reports_a_loaded_index(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["stats"]["chunks"] > 0

    def test_an_empty_library_is_healthy(self, empty_client):
        """A first run has nothing indexed yet. That is a normal state."""
        body = empty_client.get("/health").json()
        assert body["status"] == "ok"
        assert body["stats"]["chunks"] == 0

    def test_a_failed_start_is_reported_with_its_reason(self, broken_client):
        body = broken_client.get("/health").json()
        assert body["status"] == "down"
        assert "embedding model" in body["error"]

    def test_status_describes_the_answer_engine(self, client):
        body = client.get("/status").json()
        assert body["engine"] == "extractive"
        assert body["has_model"] is False
        assert "Extractive" in body["engine_label"]
        assert ".pdf" in body["accepted_types"]

    def test_status_counts_documents(self, client):
        assert client.get("/status").json()["documents"] == 3

    def test_status_of_an_empty_library(self, empty_client):
        assert empty_client.get("/status").json()["documents"] == 0

    def test_status_on_a_failed_start_explains_itself(self, broken_client):
        response = broken_client.get("/status")
        assert response.status_code == 503
        assert "could not start" in response.json()["detail"]


class TestUpload:
    def test_a_text_file_is_added_and_searchable(self, empty_client):
        results = upload(empty_client, ("warranty.txt", LONG_TEXT)).json()
        assert results == [
            {"filename": "warranty.txt", "status": "added", "chunks": 1, "reason": None}
        ]
        # The suite's hashing embedder has no sense of meaning, so this asks in
        # the document's own words; paraphrase is covered with a real model in
        # test_extractive.py.
        answer = empty_client.post(
            "/query", json={"question": "How many months does the warranty cover?"}
        ).json()
        assert "twenty four months" in answer["answer"]
        assert answer["citations"][0]["source"] == "warranty.txt"

    def test_several_files_in_one_request(self, empty_client):
        results = upload(
            empty_client, ("a.txt", LONG_TEXT), ("b.md", b"# Notes\n\n" + LONG_TEXT)
        ).json()
        assert {r["status"] for r in results} == {"added"}
        assert empty_client.get("/status").json()["documents"] == 2

    def test_re_uploading_a_file_updates_it_rather_than_duplicating(self, empty_client):
        upload(empty_client, ("warranty.txt", LONG_TEXT))
        results = upload(empty_client, ("warranty.txt", LONG_TEXT + b" Extended cover is sold separately.")).json()
        assert results[0]["status"] == "updated"
        documents = empty_client.get("/documents").json()
        assert len(documents) == 1

    def test_unsupported_types_are_skipped_with_a_reason(self, empty_client):
        results = upload(empty_client, ("sheet.xlsx", b"PK\x03\x04"), ("ok.txt", LONG_TEXT)).json()
        by_name = {r["filename"]: r for r in results}
        assert by_name["sheet.xlsx"]["status"] == "skipped"
        assert "unsupported" in by_name["sheet.xlsx"]["reason"]
        assert by_name["ok.txt"]["status"] == "added"

    def test_an_empty_file_is_skipped_not_indexed(self, empty_client):
        result = upload(empty_client, ("blank.txt", b"   \n  ")).json()[0]
        assert result["status"] == "skipped"
        assert "no readable text" in result["reason"]

    def test_a_corrupt_pdf_is_skipped_not_fatal(self, empty_client):
        result = upload(empty_client, ("broken.pdf", b"%PDF-1.4 this is not really a pdf")).json()[0]
        assert result["status"] == "skipped"
        assert empty_client.get("/health").json()["status"] == "ok"

    def test_oversized_files_are_refused(self, empty_client, settings):
        settings.max_upload_mb = 1
        big = b"word " * (1024 * 1024 // 5 + 10)
        result = upload(empty_client, ("big.txt", big)).json()[0]
        assert result["status"] == "skipped"
        assert "MB limit" in result["reason"]
        assert not (settings.uploads_dir / "big.txt").exists()

    def test_a_path_in_the_filename_cannot_escape_the_uploads_folder(self, empty_client, settings):
        """A browser sends a bare name, but a crafted request can send anything."""
        upload(empty_client, ("../../outside.txt", LONG_TEXT))
        assert (settings.uploads_dir / "outside.txt").exists()
        assert not (settings.uploads_dir.parent.parent / "outside.txt").exists()

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("report.PDF", "report.pdf"),
            ("../../etc/passwd.txt", "passwd.txt"),
            ("C:\\Windows\\evil.md", "evil.md"),
            ("my notes (final).txt", "my notes _final.txt"),
            ("(((.txt", "document.txt"),  # nothing usable left of the name
        ],
    )
    def test_filenames_are_sanitised(self, raw, expected):
        # Backslashes only separate paths on Windows; normalise for the check.
        assert api.safe_filename(raw.replace("\\", "/")) == expected


class TestDocumentManagement:
    def test_list_reports_each_document_once(self, client):
        documents = client.get("/documents").json()
        assert sorted(d["source"] for d in documents) == ["baking.md", "customs.md", "port_ops.md"]
        assert all(d["chunks"] >= 1 for d in documents)

    def test_removing_a_document_takes_it_out_of_search(self, client):
        customs = next(d for d in client.get("/documents").json() if d["source"] == "customs.md")
        assert client.delete(f"/documents/{customs['doc_id']}").status_code == 200

        remaining = {d["source"] for d in client.get("/documents").json()}
        assert "customs.md" not in remaining
        answer = client.post("/query", json={"question": "When does demurrage start to accrue?"}).json()
        assert all(c["source"] != "customs.md" for c in answer["contexts"])

    def test_removing_an_unknown_document_is_a_404(self, client):
        assert client.delete("/documents/not-a-real-id").status_code == 404

    def test_removing_an_upload_deletes_the_app_copy_only(self, empty_client, settings):
        upload(empty_client, ("warranty.txt", LONG_TEXT))
        doc_id = empty_client.get("/documents").json()[0]["doc_id"]
        empty_client.delete(f"/documents/{doc_id}")
        assert not (settings.uploads_dir / "warranty.txt").exists()

    def test_clear_all_empties_the_library(self, client):
        assert client.delete("/documents").json()["removed_documents"] == 3
        assert client.get("/documents").json() == []

    def test_sample_documents_load(self, empty_client, corpus_dir):
        body = empty_client.post("/documents/sample").json()
        assert body["documents"] == 3
        assert empty_client.get("/status").json()["documents"] == 3

    def test_changes_survive_a_restart(self, empty_client, settings):
        upload(empty_client, ("warranty.txt", LONG_TEXT))
        reopened = RAGPipeline.open(settings)
        assert [d["source"] for d in reopened.list_documents()] == ["warranty.txt"]


class TestQuery:
    def test_returns_answer_with_citations_and_contexts(self, client):
        response = client.post("/query", json={"question": "What was the average container dwell time?"})
        assert response.status_code == 200
        body = response.json()
        assert body["abstained"] is False
        assert body["citations"]
        assert body["engine"] == "extractive"
        assert all(c["marker"] == i for i, c in enumerate(body["contexts"], start=1))

    def test_abstains_out_of_corpus(self, client):
        body = client.post("/query", json={"question": "Who won the 1998 World Cup?"}).json()
        assert body["abstained"] is True
        assert body["citations"] == []

    def test_asking_before_adding_documents_says_what_to_do(self, empty_client):
        response = empty_client.post("/query", json={"question": "anything"})
        assert response.status_code == 409
        assert "Upload" in response.json()["detail"]

    def test_top_k_is_honoured(self, client):
        body = client.post("/query", json={"question": "container dwell time", "top_k": 1}).json()
        assert len(body["contexts"]) == 1

    def test_component_scores_are_exposed(self, client):
        body = client.post("/query", json={"question": "berth productivity"}).json()
        assert body["contexts"][0]["component_scores"]

    @pytest.mark.parametrize(
        "payload",
        [{}, {"question": ""}, {"question": "ok", "top_k": 0}, {"question": "ok", "top_k": 99}],
    )
    def test_invalid_payloads_are_rejected(self, client, payload):
        assert client.post("/query", json=payload).status_code == 422


class TestFallback:
    def test_a_failing_model_falls_back_to_quoting(self, client, pipeline, monkeypatch):
        """Free-tier models return 503 under load. The retrieval already
        worked, so the user gets a quoted answer and a plain explanation
        rather than an error page."""

        class Overloaded:
            name = "gemini-test"

            def generate(self, system, prompt):
                raise RuntimeError("503 UNAVAILABLE: model is experiencing high demand")

        from rag.generator import Generator

        pipeline.settings = pipeline.settings.model_copy(
            update={"llm_backend": "gemini", "google_api_key": "k"}
        )
        pipeline._generator = Generator(Overloaded())

        body = client.post("/query", json={"question": "What was the average container dwell time?"}).json()
        assert body["engine"] == "extractive"
        assert "6.2" in body["answer"]
        assert "overloaded" in body["notice"]

    def test_the_extractive_engine_does_not_mask_its_own_errors(self, pipeline, monkeypatch):
        class Broken:
            name = "extractive"

            def generate(self, system, prompt):
                raise RuntimeError("a genuine bug")

        from rag.generator import Generator

        pipeline._generator = Generator(Broken())
        with pytest.raises(RuntimeError, match="genuine bug"):
            pipeline.query("What was the average container dwell time?")


class TestIngest:
    def test_ingesting_a_directory_grows_the_index(self, empty_client, corpus_dir):
        body = empty_client.post("/ingest", json={"path": str(corpus_dir)}).json()
        assert body["documents"] == 3
        assert body["total_chunks"] == body["chunks"]

    def test_missing_path_is_a_client_error(self, client):
        assert client.post("/ingest", json={"path": "/definitely/not/here"}).status_code == 400
