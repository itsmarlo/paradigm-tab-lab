import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_rpt_context_mix import parse_positive_scores, select_rows, split_data


def test_context_mix_is_deterministic_and_keeps_query_out():
    pool, validation, _ = split_data()
    first = select_rows(pool, 128, 0.35, 11)
    repeated = select_rows(pool, 128, 0.35, 11)
    assert first.invoice_id.tolist() == repeated.invoice_id.tolist()
    assert len(first) == first.invoice_id.nunique() == 128
    assert int(first.is_fraud.sum()) == 45
    assert set(first.invoice_id).isdisjoint(validation.invoice_id)


def test_parse_scores_aligns_by_invoice_id_and_requires_both_classes():
    query = pd.DataFrame({"invoice_id": ["A", "B"], "is_fraud": [1, 0]})
    response = {"predictions": [
        {"invoice_id": "B", "is_fraud": [
            {"prediction": "0", "confidence": 0.8},
            {"prediction": "1", "confidence": 0.2}]},
        {"invoice_id": "A", "is_fraud": [
            {"prediction": "1", "confidence": 0.7},
            {"prediction": "0", "confidence": 0.3}]},
    ]}
    scored = parse_positive_scores(response, query)
    assert scored.invoice_id.tolist() == ["A", "B"]
    assert scored.score_positive.tolist() == [0.7, 0.2]
    assert scored.top_label.tolist() == [1, 0]
    response["predictions"][0]["is_fraud"].pop()
    with pytest.raises(ValueError, match="both fraud labels"):
        parse_positive_scores(response, query)
