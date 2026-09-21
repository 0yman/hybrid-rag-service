from __future__ import annotations

import pytest

from rag.chunking import chunk_document, chunk_text, split_paragraphs, split_sentences


class TestSplitSentences:
    def test_splits_on_terminal_punctuation(self):
        assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    @pytest.mark.parametrize(
        "text",
        [
            "Dr. Smith arrived.",
            "The berth held approx. 400 containers.",
            "See No. 4 berth.",
            "J. R. Smith signed it.",
        ],
    )
    def test_abbreviations_and_initials_do_not_split(self, text):
        assert split_sentences(text) == [text]

    def test_closing_quote_stays_with_its_sentence(self):
        assert split_sentences('He said "go." She left.') == ['He said "go."', "She left."]

    def test_empty_input(self):
        assert split_sentences("   ") == []


class TestChunkText:
    def test_respects_word_budget(self):
        text = " ".join(["Dwell time rose sharply."] * 60)
        chunks = chunk_text(text, chunk_size=40, chunk_overlap=8)
        assert all(len(c.split()) <= 40 for c in chunks)
        assert len(chunks) > 1

    def test_consecutive_chunks_overlap(self):
        text = " ".join(f"Sentence number {i} of the document." for i in range(40))
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=12)
        for first, second in zip(chunks, chunks[1:], strict=False):
            assert set(first.split()) & set(second.split()), "overlap was lost"

    def test_no_content_is_dropped(self):
        text = "Alpha one. Bravo two. Charlie three. Delta four. Echo five."
        chunks = chunk_text(text, chunk_size=6, chunk_overlap=2)
        joined = " ".join(chunks)
        for word in ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]:
            assert word in joined

    def test_sentence_longer_than_budget_survives_whole(self):
        long_sentence = " ".join(["word"] * 80) + "."
        chunks = chunk_text(long_sentence, chunk_size=20, chunk_overlap=5)
        assert any(len(c.split()) >= 80 for c in chunks), "long sentence was truncated"

    def test_tiny_trailing_chunk_is_merged(self):
        text = " ".join(["Filler text here."] * 20) + " Tail."
        chunks = chunk_text(text, chunk_size=25, chunk_overlap=5, min_chunk_words=10)
        assert len(chunks[-1].split()) >= 10

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with pytest.raises(ValueError, match="smaller"):
            chunk_text("text", chunk_size=10, chunk_overlap=10)

    def test_empty_text_yields_no_chunks(self):
        assert chunk_text("") == []


class TestSplitParagraphs:
    def test_blank_line_separates(self):
        assert split_paragraphs("one\n\ntwo\n\n\nthree") == ["one", "two", "three"]


class TestChunkDocument:
    def test_ids_are_unique_and_ordered(self, sample_document):
        chunks = chunk_document(sample_document, chunk_size=25, chunk_overlap=5)
        assert len(chunks) > 1
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_metadata_is_carried_through(self, sample_document):
        sample_document.metadata["path"] = "/tmp/port_ops.md"
        chunks = chunk_document(sample_document, chunk_size=25, chunk_overlap=5)
        assert all(c.source == "port_ops.md" for c in chunks)
        assert all(c.metadata["path"] == "/tmp/port_ops.md" for c in chunks)
