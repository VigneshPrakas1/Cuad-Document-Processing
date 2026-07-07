"""
data_loader.py
--------------
Utilities for loading a subset of the CUAD (Contract Understanding Atticus
Dataset) contracts from disk.

CUAD is distributed (https://www.atticusprojectai.org/cuad, mirrored on
GitHub at TheAtticusProject/cuad) in a few common layouts:

  1. full_contract_pdf/<category>/<contract_name>.pdf   - original PDFs
  2. full_contract_txt/<contract_name>.txt              - pre-extracted text
  3. CUAD_v1_master_clauses.csv                          - annotated clauses
     (ground truth, one row per contract, one column per clause type)

This loader is deliberately layout-agnostic: point it at a directory and it
will pick up every .pdf and .txt file it finds (PDFs take priority if a
matching .txt does not exist), so it works whether you downloaded the raw
CUAD release or just dropped a handful of contracts in a folder.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_TEXT_EXT = {".txt"}
SUPPORTED_PDF_EXT = {".pdf"}


@dataclass
class Contract:
    contract_id: str
    filename: str
    raw_text: str
    source_path: str


def _extract_pdf_text(path: Path) -> str:
    """Extract raw text from a PDF using pdfplumber, page by page."""
    try:
        import pdfplumber
    except ImportError as e:
        raise ImportError(
            "pdfplumber is required to read PDF contracts. "
            "Install it with `pip install pdfplumber`."
        ) from e

    text_parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def _load_single_file(path: Path) -> str:
    if path.suffix.lower() in SUPPORTED_TEXT_EXT:
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in SUPPORTED_PDF_EXT:
        return _extract_pdf_text(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def discover_contract_files(data_dir: str) -> list[Path]:
    """
    Walk `data_dir` recursively and return one path per contract, preferring
    .txt over .pdf when both exist for the same base filename (text
    extraction from the CUAD-provided .txt files is cleaner/faster than
    re-parsing the PDF).
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    txt_files = {p.stem: p for p in data_dir.rglob("*.txt")}
    pdf_files = {p.stem: p for p in data_dir.rglob("*.pdf")}

    all_stems = set(txt_files) | set(pdf_files)
    chosen = []
    for stem in sorted(all_stems):
        chosen.append(txt_files.get(stem, pdf_files.get(stem)))
    return chosen


def load_contracts(data_dir: str, n: Optional[int] = 50) -> list[Contract]:
    """
    Load up to `n` contracts from `data_dir`.

    Parameters
    ----------
    data_dir : str
        Folder containing CUAD contract files (.pdf and/or .txt).
    n : Optional[int]
        Maximum number of contracts to load. None loads everything found.

    Returns
    -------
    list[Contract]
    """
    files = discover_contract_files(data_dir)
    if n is not None:
        files = files[:n]

    contracts: list[Contract] = []
    for path in files:
        try:
            text = _load_single_file(path)
        except Exception as exc:  # noqa: BLE001 - log and skip bad files
            logger.warning("Skipping %s: %s", path, exc)
            continue

        if not text.strip():
            logger.warning("No extractable text in %s, skipping.", path)
            continue

        contracts.append(
            Contract(
                contract_id=path.stem,
                filename=path.name,
                raw_text=text,
                source_path=str(path),
            )
        )

    logger.info("Loaded %d contracts from %s", len(contracts), data_dir)
    return contracts


def load_master_clauses_csv(csv_path: str) -> pd.DataFrame:
    """
    Load CUAD's ground-truth `CUAD_v1_master_clauses.csv` (or the master
    clauses file bundled with the v1 release), if available. This is not
    required to run the pipeline but is useful for evaluating extraction
    quality against human annotations.
    """
    df = pd.read_csv(csv_path)
    # CUAD's master CSV uses "Filename" as the join key to the contract text
    if "Filename" in df.columns:
        df["contract_id"] = df["Filename"].str.replace(
            r"\.(pdf|txt)$", "", regex=True
        )
    return df
