"""Results must not depend on where the project happens to live on disk.

Chunk ids break ties in rank fusion. When they were hashes of absolute paths,
the same benchmark scored differently in CI (/home/runner/...) than on a
laptop (C:\\Users\\...). These tests pin the fix.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from rag.loaders import load_directory, load_file
from rag.pipeline import RAGPipeline


def _write(folder: Path) -> Path:
    (folder / "sub").mkdir(parents=True)
    (folder / "a.md").write_text("# Alpha\n\nAlpha facts about berths.", encoding="utf-8")
    (folder / "sub" / "b.txt").write_text("Beta facts about cranes.", encoding="utf-8")
    return folder


def test_ids_are_the_same_wherever_the_folder_lives(tmp_path):
    one = {d.source: d.doc_id for d in load_directory(_write(tmp_path / "x" / "corpus"))}
    two = {d.source: d.doc_id for d in load_directory(_write(tmp_path / "somewhere" / "else"))}
    assert one == two


def test_same_name_in_different_subfolders_gets_different_ids(tmp_path):
    root = tmp_path / "c"
    (root / "one").mkdir(parents=True)
    (root / "two").mkdir()
    (root / "one" / "notes.txt").write_text("First notes.", encoding="utf-8")
    (root / "two" / "notes.txt").write_text("Second notes.", encoding="utf-8")
    ids = {d.doc_id for d in load_directory(root)}
    assert len(ids) == 2


def test_a_single_file_is_identified_by_its_name(tmp_path):
    first = tmp_path / "x" / "report.txt"
    second = tmp_path / "y" / "report.txt"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.write_text("Quarterly numbers.", encoding="utf-8")
    assert load_file(first).doc_id == load_file(second).doc_id


def test_retrieval_is_identical_after_moving_the_corpus(tmp_path, settings, corpus_dir):
    """The regression that caught this: identical question, identical
    documents, different folder - the ranking must not change."""
    moved = tmp_path / "moved" / "deeper" / "corpus"
    shutil.copytree(corpus_dir, moved)

    def ranking(folder: Path, index: str) -> list[str]:
        pipeline = RAGPipeline.create(settings.model_copy(update={"index_dir": tmp_path / index}))
        pipeline.ingest_path(folder)
        return [c.chunk.chunk_id for c in pipeline.retrieve("container dwell time demurrage")]

    assert ranking(corpus_dir, "i1") == ranking(moved, "i2")
