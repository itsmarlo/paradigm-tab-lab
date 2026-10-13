import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from context_engineering import assert_no_leakage, select_context
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_context_benchmark import request_for
from run_semantic_ablation import request_payload, context_indices, verify_pairing
from run_feature_selection_ablation import request_for as feature_request


@pytest.fixture
def frames():
    source = pd.read_csv(Path(__file__).resolve().parents[1] / "data/synthetic_invoices/payment_behavior.csv")
    return source.iloc[:512], source.iloc[1024:1025]


@pytest.mark.parametrize("strategy", ["random", "stratified", "balanced", "similarity", "diversity", "hybrid"])
def test_selector_preserves_budget_and_split(frames, strategy):
    pool, query = frames
    chosen, _ = select_context(pool, "is_fraud", query, strategy, 32, 11)
    assert len(chosen) == chosen.invoice_id.nunique() == 32
    assert_no_leakage(pool, query, chosen)
    repeated, _ = select_context(pool, "is_fraud", query, strategy, 32, 11)
    assert chosen.invoice_id.tolist() == repeated.invoice_id.tolist()


def test_leakage_rejected(frames):
    pool, query = frames
    with pytest.raises(AssertionError):
        assert_no_leakage(pool, pool.iloc[:1])


def test_query_labels_masked_and_other_target_removed(frames):
    pool, query = frames
    context = pool.iloc[:32]
    request = request_for(context, query, "is_fraud")
    assert request["rows"][-1]["is_fraud"] == "[PREDICT]"
    assert all("paid_late" not in row for row in request["rows"])
    assert all(row["invoice_id"] != query.invoice_id.iloc[0] for row in request["rows"][:-1])


def test_semantic_conditions_keep_all_values_and_samples():
    source = pd.read_csv(Path(__file__).resolve().parents[1] / "data/semantic_ablation/source_snapshot.csv")
    indices = context_indices(11)
    verify_pairing(source, indices)
    assert [row["invoice_id"] for row in request_payload(source, indices, "S0")["rows"]] == [row["invoice_id"] for row in request_payload(source, indices, "S3")["rows"]]


def test_feature_removal_changes_only_one_predictor():
    source = pd.read_csv(Path(__file__).resolve().parents[1] / "data/semantic_ablation/source_snapshot.csv")
    indices = context_indices(11)
    baseline = feature_request(source, indices, "is_fraud", False)
    removed = feature_request(source, indices, "is_fraud", True)
    assert [r["invoice_id"] for r in baseline["rows"]] == [r["invoice_id"] for r in removed["rows"]]
    for left, right in zip(baseline["rows"], removed["rows"]):
        assert {key: value for key, value in left.items() if key != "region"} == right
    assert all("paid_late" not in row for row in removed["rows"])
    assert all(row["is_fraud"] == "[PREDICT]" for row in removed["rows"][-128:])
