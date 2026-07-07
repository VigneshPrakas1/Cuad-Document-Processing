"""
preprocessing.py
-----------------
Text normalization for raw contract text pulled from PDFs / txt files.

PDF-extracted legal text tends to have:
  - hyphenated words split across a line break ("consid-\nered")
  - inconsistent whitespace / stray page headers and footers
  - repeated form-feed / page-number artifacts
  - non-breaking spaces and other unicode oddities

None of this is fatal for an LLM, but cleaning it up (a) reduces token
count/cost and (b) makes chunking boundaries land on sentence breaks instead
of mid-word.
"""

from __future__ import annotations

import re
import unicodedata

_PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*)?\d+\s*(of\s*\d+)?\s*$", re.IGNORECASE)
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_LINES_RE = re.compile(r"(\n\s*){3,}")


def normalize_text(text: str) -> str:
    """Apply a pipeline of cleanup steps to raw extracted contract text."""
    if not text:
        return ""

    # 1. Unicode normalize (curly quotes, non-breaking spaces, etc.)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\ufeff", "")

    # 2. Re-join words that were hyphenated across a line break.
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    # 3. Drop lines that are just page numbers / "Page X of Y" footers.
    lines = [ln for ln in text.split("\n") if not _PAGE_NUMBER_RE.match(ln)]
    text = "\n".join(lines)

    # 4. Collapse repeated whitespace/newlines.
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_LINES_RE.sub("\n\n", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()


def chunk_text(text: str, max_chars: int = 12_000, overlap_chars: int = 500) -> list[str]:
    """
    Split long contract text into overlapping chunks so it fits comfortably
    within an LLM context window (character-based approximation of tokens;
    ~4 chars/token means 12,000 chars is roughly 3,000 tokens).

    Chunk boundaries are snapped to the nearest paragraph break where
    possible so a clause isn't split mid-sentence.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # try to snap to the last paragraph break in this window
            snap = text.rfind("\n\n", start, end)
            if snap > start + max_chars // 2:
                end = snap
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap_chars, 0)
    return [c for c in chunks if c]
