"""Reading documents: especially PDFs, whose text arrives as printed lines.

Everything downstream - chunking, retrieval, the quoted answer - inherits
whatever the loader produces, so a heading glued to a sentence here turns up
as "Eligibility Employees become eligible..." in front of a user.
"""

from __future__ import annotations

from pathlib import Path

from rag.loaders import load_file, normalize_whitespace, reflow_pdf_text

# What pypdf actually returns for a simple policy document: one line per
# printed line, no blank lines, headings indistinguishable from text.
PDF_LINES = (
    "Northwind Remote Work Policy\n"
    "This policy applies to all employees from 1 March. It\n"
    "explains who may work remotely.\n"
    "Eligibility\n"
    "Employees become eligible after a three month probation\n"
    "period. Remote work is limited to three days per week in the\n"
    "Lisbon office.\n"
    "Equipment and expenses\n"
    "A one-time home office allow-\n"
    "ance of 400 euros may be claimed.\n"
)


class TestReflow:
    def test_wrapped_lines_become_one_sentence(self):
        text = reflow_pdf_text(PDF_LINES)
        assert "three month probation period." in text
        assert "probation\nperiod" not in text

    def test_a_wrap_before_a_capitalised_word_is_still_a_wrap(self):
        """'in the / Lisbon office' - the capital is a proper noun, not a heading."""
        assert "in the Lisbon office." in reflow_pdf_text(PDF_LINES)

    def test_headings_are_marked_and_kept_with_their_paragraph(self):
        text = reflow_pdf_text(PDF_LINES)
        assert "## Eligibility\nEmployees become eligible" in text
        assert "## Equipment and expenses\n" in text

    def test_the_first_heading_becomes_the_title(self):
        assert reflow_pdf_text(PDF_LINES).startswith("# Northwind Remote Work Policy\n")

    def test_a_word_hyphenated_across_lines_is_rejoined(self):
        assert "allow-ance of 400 euros" in reflow_pdf_text(PDF_LINES)

    def test_a_short_sentence_is_not_mistaken_for_a_heading(self):
        """It ends in a full stop, so it is text however short it is."""
        text = reflow_pdf_text("Intro line.\nNo exceptions.\nThe rest follows here.\n")
        assert "#" not in text

    def test_blank_lines_still_separate_paragraphs(self):
        text = reflow_pdf_text("First paragraph here.\n\nSecond paragraph here.\n")
        assert text == "First paragraph here.\n\nSecond paragraph here."

    def test_empty_page(self):
        assert reflow_pdf_text("") == ""


class TestLoadFile:
    def test_markdown_title_comes_from_its_heading(self, tmp_path: Path):
        path = tmp_path / "notes.md"
        path.write_text("# Quarterly Review\n\nSales rose.", encoding="utf-8")
        assert load_file(path).title == "Quarterly Review"

    def test_plain_text_title_comes_from_the_filename(self, tmp_path: Path):
        path = tmp_path / "meeting_notes-march.txt"
        path.write_text("We agreed to ship on Friday.", encoding="utf-8")
        assert load_file(path).title == "meeting notes march"

    def test_unsupported_type_is_ignored(self, tmp_path: Path):
        path = tmp_path / "data.xlsx"
        path.write_bytes(b"PK")
        assert load_file(path) is None

    def test_whitespace_only_file_is_ignored(self, tmp_path: Path):
        path = tmp_path / "blank.txt"
        path.write_text("  \n\n  ", encoding="utf-8")
        assert load_file(path) is None

    def test_same_path_gives_same_id(self, tmp_path: Path):
        """Re-uploading a file replaces it because its id is stable."""
        path = tmp_path / "a.txt"
        path.write_text("Some content.", encoding="utf-8")
        assert load_file(path).doc_id == load_file(path).doc_id


def test_normalize_collapses_runs_of_blank_lines():
    assert normalize_whitespace("a\r\n\r\n\r\n\r\nb") == "a\n\nb"
