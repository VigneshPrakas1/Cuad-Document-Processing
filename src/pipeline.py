"""
pipeline.py
-----------
End-to-end orchestration: load contracts -> normalize text -> run LLM
extraction + summarization -> write results to CSV/JSON.

See main.py for the CLI entry point.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .data_loader import load_contracts
from .llm_extractor import extract_clauses, summarize_contract
from .preprocessing import normalize_text

logger = logging.getLogger(__name__)


def run_pipeline(
    data_dir: str,
    output_dir: str,
    n_contracts: int = 50,
    formats: tuple[str, ...] = ("json", "csv"),
) -> list[dict]:
    """
    Run the full pipeline over up to `n_contracts` contracts in `data_dir`
    and write results to `output_dir`.

    Returns the list of per-contract result dicts (also written to disk).
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    contracts = load_contracts(data_dir, n=n_contracts)
    if not contracts:
        logger.warning("No contracts found in %s -- nothing to do.", data_dir)
        return []

    results = []
    for contract in tqdm(contracts, desc="Processing contracts"):
        clean_text = normalize_text(contract.raw_text)

        try:
            clauses = extract_clauses(clean_text)
        except Exception:  # noqa: BLE001
            logger.exception("Clause extraction failed for %s", contract.contract_id)
            clauses = {
                k: {"found": False, "excerpts": [], "notes": "ERROR"}
                for k in ("termination_clause", "confidentiality_clause", "liability_clause")
            }

        try:
            summary = summarize_contract(clean_text)
        except Exception:  # noqa: BLE001
            logger.exception("Summarization failed for %s", contract.contract_id)
            summary = "ERROR: summarization failed"

        results.append(
            {
                "contract_id": contract.contract_id,
                "filename": contract.filename,
                "summary": summary,
                "termination_clause": clauses["termination_clause"],
                "confidentiality_clause": clauses["confidentiality_clause"],
                "liability_clause": clauses["liability_clause"],
            }
        )

    if "json" in formats:
        json_path = output_dir_path / "results.json"
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Wrote %s", json_path)

    if "csv" in formats:
        csv_path = output_dir_path / "results.csv"
        _write_csv(results, csv_path)
        logger.info("Wrote %s", csv_path)

    return results


def _write_csv(results: list[dict], csv_path: Path) -> None:
    """
    Flatten the nested clause dicts into simple string columns so the CSV
    stays spreadsheet-friendly, per the requested
    [contract_id, summary, termination_clause, confidentiality_clause,
    liability_clause] schema.
    """
    rows = []
    for r in results:
        rows.append(
            {
                "contract_id": r["contract_id"],
                "filename": r["filename"],
                "summary": r["summary"],
                "termination_clause": _flatten_clause(r["termination_clause"]),
                "confidentiality_clause": _flatten_clause(r["confidentiality_clause"]),
                "liability_clause": _flatten_clause(r["liability_clause"]),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def _flatten_clause(clause: dict) -> str:
    if not clause.get("found"):
        return "NOT FOUND"
    excerpts = " | ".join(clause.get("excerpts", []))
    notes = clause.get("notes", "")
    return f"{notes}\n\nExcerpts: {excerpts}" if excerpts else notes


def process_uploaded_contract(file_path: str):
    """
    Process a single uploaded PDF/TXT contract and return a result dictionary.
    """

    contracts = load_contracts(
        str(Path(file_path).parent),
        n=1
    )

    if not contracts:
        raise ValueError("No contract found.")

    contract = contracts[0]

    clean_text = normalize_text(contract.raw_text)

    clauses = extract_clauses(clean_text)

    summary = summarize_contract(clean_text)

    return {
        "contract_id": contract.contract_id,
        "filename": contract.filename,
        "summary": summary,
        "termination_clause": clauses["termination_clause"],
        "confidentiality_clause": clauses["confidentiality_clause"],
        "liability_clause": clauses["liability_clause"],
    }
