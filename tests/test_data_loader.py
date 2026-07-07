"""Basic unit tests for contract loading."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_contracts  # noqa: E402

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data"


def test_load_contracts_finds_sample_files():
    contracts = load_contracts(str(SAMPLE_DATA_DIR), n=50)
    assert len(contracts) == 3
    ids = {c.contract_id for c in contracts}
    assert "sample_msa_001" in ids
    assert "sample_nda_002" in ids
    assert "sample_license_003" in ids


def test_load_contracts_respects_n_limit():
    contracts = load_contracts(str(SAMPLE_DATA_DIR), n=2)
    assert len(contracts) == 2


def test_load_contracts_raises_on_missing_dir():
    import pytest

    with pytest.raises(FileNotFoundError):
        load_contracts("/nonexistent/path/xyz", n=10)


def test_loaded_contract_has_nonempty_text():
    contracts = load_contracts(str(SAMPLE_DATA_DIR), n=1)
    assert contracts[0].raw_text.strip() != ""
