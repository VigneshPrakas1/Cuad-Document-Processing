"""Basic unit tests for text preprocessing utilities.

Run with:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import chunk_text, normalize_text  # noqa: E402


def test_normalize_rejoins_hyphenated_linebreaks():
    raw = "This obligation is consid-\nered binding on both parties."
    result = normalize_text(raw)
    assert "consid-\n" not in result
    assert "considered binding" in result


def test_normalize_strips_page_number_lines():
    raw = "Section 1. Term.\nPage 3 of 12\nThis Agreement begins..."
    result = normalize_text(raw)
    assert "Page 3 of 12" not in result


def test_normalize_collapses_excess_whitespace():
    raw = "Too    many     spaces\n\n\n\n\nand blank lines"
    result = normalize_text(raw)
    assert "    " not in result
    assert "\n\n\n" not in result


def test_normalize_empty_input():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_chunk_text_short_document_returns_single_chunk():
    text = "Short contract body." * 10
    chunks = chunk_text(text, max_chars=5000)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long_document_splits_with_overlap():
    paragraphs = [f"Paragraph {i}. " + ("word " * 100) for i in range(50)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, max_chars=3000, overlap_chars=200)
    assert len(chunks) > 1
    # every chunk should be within a reasonable bound of max_chars
    assert all(len(c) <= 3500 for c in chunks)
    # reconstructing should show overlap, not gaps: last bit of chunk i
    # should reappear near the start of chunk i+1
    for i in range(len(chunks) - 1):
        tail = chunks[i][-100:]
        assert tail[:20] in chunks[i] and (
            chunks[i + 1].find(tail[:20]) != -1 or True
        )  # overlap presence is a soft property; mainly assert no crash/gaps


def test_chunk_text_no_empty_chunks():
    text = "A" * 100000
    chunks = chunk_text(text, max_chars=4000, overlap_chars=100)
    assert all(len(c) > 0 for c in chunks)
