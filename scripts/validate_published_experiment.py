"""Check that the published TechEd data and saved predictions belong together."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

from generate_synthetic_invoices import CONTEXT_ROWS, PREDICTION_ROWS, TARGETS, make_rows


DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic_invoices"
source = pd.read_csv(DATA / "payment_behavior.csv")
assert len(source) == CONTEXT_ROWS + PREDICTION_ROWS == 1152
assert source.invoice_id.is_unique and not source.isna().any().any()
assert source[list(TARGETS)].isin([0, 1]).all().all()
assert source.iloc[:CONTEXT_ROWS].is_fraud.sum() == 187
assert source.iloc[CONTEXT_ROWS:].is_fraud.sum() == 22

generated = pd.DataFrame(make_rows())
pd.testing.assert_frame_equal(source, generated, check_dtype=False)

upload = pd.read_csv(DATA / "exports" / "rpt_upload.csv", dtype={target: str for target in TARGETS})
assert len(upload) == len(source)
pd.testing.assert_frame_equal(upload.drop(columns=list(TARGETS)), source.drop(columns=list(TARGETS)))
for target in TARGETS:
    assert upload[target].iloc[:CONTEXT_ROWS].astype(int).tolist() == source[target].iloc[:CONTEXT_ROWS].tolist()
    assert upload[target].iloc[CONTEXT_ROWS:].eq("[PREDICT]").all()

request_path = DATA / "rpt16_request.json"
request = json.loads(request_path.read_text(encoding="utf-8"))
assert request["index_column"] == "invoice_id"
assert {item["name"] for item in request["prediction_config"]["target_columns"]} == set(TARGETS)
request_rows = pd.DataFrame(request["rows"])
pd.testing.assert_frame_equal(request_rows.drop(columns=list(TARGETS)), upload.drop(columns=list(TARGETS)))
for target in TARGETS:
    assert request_rows[target].iloc[:CONTEXT_ROWS].astype(int).tolist() == source[target].iloc[:CONTEXT_ROWS].tolist()
    assert request_rows[target].iloc[CONTEXT_ROWS:].eq("[PREDICT]").all()

metadata = json.loads((DATA / "rpt16_run_metadata.json").read_text(encoding="utf-8"))
result_path = DATA / "rpt16_results.csv"
assert metadata["source"] == "btp_deployment"
assert metadata["test_rows"] == PREDICTION_ROWS
assert metadata["request_sha256"] == hashlib.sha256(request_path.read_bytes()).hexdigest()
assert metadata["results_sha256"] == hashlib.sha256(result_path.read_bytes()).hexdigest()
assert not ({"deployment_id", "deployment_host", "resource_group", "deployment_url_sha256"} & metadata.keys())

test = source.iloc[CONTEXT_ROWS:].reset_index(drop=True)
for name, path in (("XGBoost", DATA / "xgboost_results.csv"), ("RPT", result_path)):
    result = pd.read_csv(path)
    assert len(result) == PREDICTION_ROWS and result.invoice_id.is_unique
    assert result.invoice_id.tolist() == test.invoice_id.tolist()
    for target in TARGETS:
        assert result[f"actual_{target}"].tolist() == test[target].tolist()
        assert result[f"predicted_{target}"].isin([0, 1]).all()
    tn, fp, fn, tp = confusion_matrix(test.is_fraud, result.predicted_is_fraud, labels=[0, 1]).ravel()
    print(f"{name} fraud: {tp} found, {fn} missed, {fp} false alarms, {tn} true negatives")

print("Published dataset, masked requests, saved predictions, and BTP provenance are consistent.")
