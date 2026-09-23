"""Shared fixtures.

Every test runs on the `hash` embedder and the `extractive` engine: no model download,
no network, no API key. The suite therefore behaves identically on a laptop
and in CI, which is the only way a failure means something.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from rag.config import Settings, get_settings  # noqa: E402
from rag.models import Chunk, Document  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402

DOCS = {
    "port_ops.md": (
        "# Port Operations\n\n"
        "Container dwell time is the number of days a container stays in a terminal "
        "before it leaves by truck or rail. At the Port of Alexandria the average "
        "dwell time was 6.2 days during 2024, down from 7.1 days the year before.\n\n"
        "Berth productivity is measured in container moves per hour. Berth 4 averaged "
        "28 moves per hour in March, the highest of any berth in the west harbour.\n\n"
        "The terminal operating system assigns a yard block to every inbound "
        "container using its discharge sequence and its planned pickup window.\n"
    ),
    "customs.md": (
        "# Customs Clearance\n\n"
        "A bill of lading must be presented before cargo is released. Clearance "
        "usually takes two working days once the declaration is filed.\n\n"
        "Demurrage charges begin to accrue after the free time granted in the "
        "contract expires. The standard free period is five calendar days.\n"
    ),
    "baking.md": (
        "# Sourdough Baking\n\n"
        "A starter is a culture of flour and water. Hydration above 75 percent "
        "produces an open crumb structure.\n"
    ),
}


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "corpus"
    directory.mkdir()
    for name, text in DOCS.items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return get_settings(
        embedding_backend="hash",
        llm_backend="extractive",
        index_dir=tmp_path / "index",
        uploads_dir=tmp_path / "uploads",
        benchmark_dir=tmp_path / "corpus",
        benchmark_index_dir=tmp_path / "benchmark_index",
        final_top_k=3,
    )


@pytest.fixture
def pipeline(settings: Settings, corpus_dir: Path) -> RAGPipeline:
    pipe = RAGPipeline.create(settings)
    pipe.ingest_path(corpus_dir)
    return pipe


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"doc::{i}",
            doc_id="doc",
            title="Doc",
            text=text,
            ordinal=i,
            source="doc.md",
        )
        for i, text in enumerate(
            ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]
        )
    ]


@pytest.fixture
def sample_document() -> Document:
    return Document(
        doc_id="d1",
        title="Doc",
        text=DOCS["port_ops.md"],
        source="port_ops.md",
    )
