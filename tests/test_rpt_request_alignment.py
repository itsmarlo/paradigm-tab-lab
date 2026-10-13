"""The legacy SAP request must never be scored against revised labels."""

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_rpt16_invoices import run_experiment, validate_request_source


def test_saved_request_matches_archived_source_but_rejects_working_labels():
    data = ROOT / "data/synthetic_invoices"
    original = ROOT / "archive/synthetic_invoice_experiments/published_baseline/payment_behavior.csv"

    assert len(validate_request_source(data, original)) == 1152
    with pytest.raises(ValueError, match="stale is_fraud context labels"):
        validate_request_source(data)
    with pytest.raises(ValueError, match="stale is_fraud context labels"):
        run_experiment(response_path=ROOT / "nonexistent-response.json", data=data)
