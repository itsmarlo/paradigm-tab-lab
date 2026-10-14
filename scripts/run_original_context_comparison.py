"""Re-run both models on the original 22-case source with matched 128-row contexts.

The saved 1,024-row predictions remain the presentation's first benchmark. This
script adds a random 128-row control and balanced 128-row engineered condition.
SAP calls happen only with --live; completed SAP results are reused on rerun.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from context_engineering import assert_no_leakage, select_context
from run_context_benchmark import call_rpt, metric_row, request_for, score, xgb_predict
from run_rpt16_invoices import connection_settings

SOURCE = ROOT / "archive/synthetic_invoice_experiments/published_baseline/payment_behavior.csv"
OUT = ROOT / "data/original_context_comparison"
TARGETS = ("paid_late", "is_fraud")
STRATEGIES = ("random", "balanced")
SEED = 11
XGB_SEED = 20261002
BUDGET = 128


def run(live=False):
    source = pd.read_csv(SOURCE)
    assert len(source) == 1152 and source.invoice_id.is_unique and not source.isna().any().any()
    pool, holdout = source.iloc[:1024].copy(), source.iloc[1024:].copy()
    assert len(holdout) == 128 and int(holdout.is_fraud.sum()) == 22
    assert int(holdout.paid_late.sum()) == 51
    assert_no_leakage(pool, holdout)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    protocol = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "source": str(SOURCE.relative_to(ROOT)), "pool_rows": 1024,
                "holdout_rows": 128, "holdout_fraud_positives": 22,
                "budget": BUDGET, "selection_seed": SEED, "xgboost_seed": XGB_SEED,
                "strategies": list(STRATEGIES),
                "targets": list(TARGETS),
                "note": "128-row random control and balanced treatment; original saved baseline used 1024 rows"}
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        assert previous == protocol, "Source or experiment settings changed since saved runs"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    result_path, prediction_path = OUT / "results.csv", OUT / "predictions.csv"
    results = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    predictions = pd.read_csv(prediction_path).to_dict("records") if prediction_path.exists() else []
    completed = {row["run_id"] for row in results}
    needed_sap = {f"original22|{strategy}|{target}|{SEED}|SAP-RPT"
                  for strategy in STRATEGIES for target in TARGETS}
    connection = connection_settings() if live and not needed_sap <= completed else None
    for strategy in STRATEGIES:
        for target in TARGETS:
            context, _ = select_context(pool, target, holdout.iloc[:1], strategy, BUDGET, SEED)
            assert_no_leakage(pool, holdout, context)
            for model in ("XGBoost", "SAP-RPT"):
                run_id = f"original22|{strategy}|{target}|{SEED}|{model}"
                if run_id in completed or (model == "SAP-RPT" and not live):
                    continue
                if model == "XGBoost":
                    predicted, _, _, _ = xgb_predict(context, holdout, target, XGB_SEED)
                else:
                    response, _ = call_rpt(request_for(context, holdout, target), connection)
                    predicted = score(response, holdout, target)
                actual = holdout[target].to_numpy()
                results.append({"run_id": run_id, "model": model, "target": target,
                                "strategy": strategy, "seed": SEED, "context_rows": BUDGET,
                                "context_positives": int(context[target].sum()),
                                **metric_row(actual, predicted)})
                predictions.extend({"run_id": run_id, "invoice_id": invoice_id,
                                    "actual": int(truth), "predicted": int(guess)}
                                   for invoice_id, truth, guess in
                                   zip(holdout.invoice_id, actual, predicted))
                pd.DataFrame(results).to_csv(result_path, index=False)
                pd.DataFrame(predictions).to_csv(prediction_path, index=False)
                completed.add(run_id)
                print(run_id, flush=True)
    print(f"Saved {result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call the configured SAP deployment for missing runs")
    run(parser.parse_args().live)
