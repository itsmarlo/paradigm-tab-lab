"""Prepare and score a leakage-safe SAP RPT fraud threshold experiment.

The original 128 test rows are never used to select the threshold. This script
does not call SAP; run the generated requests with the configured deployment.
"""

import argparse
import csv
import json
import random
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/synthetic_invoices"
CONTEXT_COUNT = 1024
TEST_COUNT = 128
VALIDATION_COUNT = 128


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def make_request(rows, context_ids, query_ids, original):
    selected = []
    for row in rows:
        if row["invoice_id"] in context_ids:
            selected.append({**row, "is_fraud": int(row["is_fraud"])})
        elif row["invoice_id"] in query_ids:
            selected.append({**row, "is_fraud": "[PREDICT]"})
    payload = {**original, "rows": selected}
    payload["prediction_config"] = {"target_columns": [{
        "name": "is_fraud", "prediction_placeholder": "[PREDICT]",
        "task_type": "classification", "top_k": 2,
    }]}
    # The other outcome is not available when a new invoice is scored.
    for row in payload["rows"]:
        row.pop("paid_late", None)
    payload["data_schema"] = {
        key: value for key, value in original["data_schema"].items()
        if key != "paid_late"
    }
    return payload


def prepare(data, output):
    rows = read_rows(data / "payment_behavior.csv")
    if len(rows) != CONTEXT_COUNT + TEST_COUNT:
        raise ValueError("Expected 1,024 context and 128 test rows")
    original = json.loads((data / "rpt16_request.json").read_text(encoding="utf-8"))
    rng = random.Random(20261004)
    context = rows[:CONTEXT_COUNT]
    validation = []
    for label in ("0", "1"):
        group = [row for row in context if row["is_fraud"] == label]
        count = round(VALIDATION_COUNT * len(group) / CONTEXT_COUNT)
        validation.extend(rng.sample(group, count))
    # Rounding happens to yield 128 for this dataset; fail if that changes.
    if len(validation) != VALIDATION_COUNT:
        raise ValueError("Validation split did not contain 128 rows")
    validation_ids = {row["invoice_id"] for row in validation}
    context_ids = {row["invoice_id"] for row in context} - validation_ids
    test_ids = {row["invoice_id"] for row in rows[CONTEXT_COUNT:]}
    output.mkdir(parents=True, exist_ok=True)
    requests = {
        "validation_request.json": make_request(rows, context_ids, validation_ids, original),
        "test_request.json": make_request(rows, context_ids | validation_ids, test_ids, original),
    }
    for name, payload in requests.items():
        (output / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Validation: {len(validation_ids)} rows, {sum(r['is_fraud'] == '1' for r in validation)} fraud cases")
    print(f"Requests saved in {output}; no SAP call was made")


def positive_scores(response_path, expected_ids):
    response = json.loads(response_path.read_text(encoding="utf-8"))
    if response.get("status", {}).get("code") != 0:
        raise ValueError(f"SAP response status is not OK: {response.get('status')}")
    predictions = response.get("predictions", [])
    if len(predictions) != len(expected_ids):
        raise ValueError("Response has the wrong number of predictions")
    scores = {}
    for row in predictions:
        invoice_id = row.get("invoice_id")
        if invoice_id in scores or invoice_id not in expected_ids:
            raise ValueError("Duplicate or unexpected invoice ID in response")
        choices = row.get("is_fraud", [])
        by_class = {str(item["prediction"]): item["confidence"] for item in choices}
        if set(by_class) != {"0", "1"} or any(v is None for v in by_class.values()):
            raise ValueError("Response must include confidence for both fraud classes")
        scores[invoice_id] = float(by_class["1"])
    if set(scores) != expected_ids:
        raise ValueError("Response does not cover the expected rows")
    return scores


def counts(truth, scores, threshold):
    tp = fp = fn = tn = 0
    for invoice_id, score in scores.items():
        actual = int(truth[invoice_id]["is_fraud"])
        predicted = int(score >= threshold)
        if actual and predicted:
            tp += 1
        elif actual:
            fn += 1
        elif predicted:
            fp += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def f2(tp, fp, fn):
    return 5 * tp / (5 * tp + 4 * fn + fp) if tp else 0.0


def score(data, output, validation_response, test_response):
    rows = read_rows(data / "payment_behavior.csv")
    truth = {row["invoice_id"]: row for row in rows}
    validation_request = json.loads((output / "validation_request.json").read_text(encoding="utf-8"))
    validation_ids = {row["invoice_id"] for row in validation_request["rows"] if row["is_fraud"] == "[PREDICT]"}
    test_ids = {row["invoice_id"] for row in rows[CONTEXT_COUNT:]}
    validation_scores = positive_scores(validation_response, validation_ids)
    # Candidate thresholds include the default 0.5 and every observed validation score.
    candidates = sorted({0.5, 1.0, *validation_scores.values()})
    threshold = max(candidates, key=lambda t: (f2(*counts(truth, validation_scores, t)[:3]), -t))
    test_scores = positive_scores(test_response, test_ids)
    baseline = counts(truth, test_scores, 0.5)
    print(f"Test at default 0.5: found={baseline[0]}/{baseline[0] + baseline[2]}, "
          f"false_alarms={baseline[1]}, F2={f2(*baseline[:3]):.3f}")
    for name, scores in (("Validation", validation_scores), ("Test", test_scores)):
        tp, fp, fn, tn = counts(truth, scores, threshold)
        print(f"{name}: threshold={threshold:.4f}, found={tp}/{tp + fn}, "
              f"false_alarms={fp}, precision={tp / (tp + fp) if tp + fp else 0:.3f}, "
              f"recall={tp / (tp + fn) if tp + fn else 0:.3f}, F2={f2(tp, fp, fn):.3f}")
    print("Threshold selected on validation only; test scores were not used to tune it")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "tfm-rpt-threshold")
    parser.add_argument("--validation-response", type=Path)
    parser.add_argument("--test-response", type=Path)
    args = parser.parse_args()
    if bool(args.validation_response) != bool(args.test_response):
        parser.error("Provide both response paths to score the experiment")
    if args.validation_response:
        score(args.data_dir, args.output_dir, args.validation_response, args.test_response)
    else:
        prepare(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
