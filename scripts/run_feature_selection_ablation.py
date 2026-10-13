"""Single-target SAP RPT ablation of features absent from each synthetic label formula."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from run_rpt16_invoices import connection_settings
from run_semantic_ablation import (OUT, SEEDS, TARGETS, call_rpt, canonical,
    context_indices, metrics, request_payload)

SOURCE = OUT / "source_snapshot.csv"
RESULTS = OUT / "feature_selection_results.csv"
PREDICTIONS = OUT / "feature_selection_predictions.csv"
DROP = {"paid_late": "weekend_submission", "is_fraud": "region"}


def request_for(source, indices, target, remove):
    payload = request_payload(source, indices, "S0")
    other = next(item for item in TARGETS if item != target)
    payload["prediction_config"]["target_columns"] = [
        item for item in payload["prediction_config"]["target_columns"] if item["name"] == target]
    payload["data_schema"].pop(other)
    if remove:
        payload["data_schema"].pop(DROP[target])
    for row in payload["rows"]:
        row.pop(other)
        if remove:
            row.pop(DROP[target])
    return payload


def main(live=False):
    source = pd.read_csv(SOURCE)
    rows = pd.read_csv(RESULTS).to_dict("records") if RESULTS.exists() else []
    predictions = pd.read_csv(PREDICTIONS).to_dict("records") if PREDICTIONS.exists() else []
    done = {(int(row["seed"]), row["target"], row["condition"]) for row in rows}
    connection = connection_settings() if live else None
    protocol = {"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "seeds": SEEDS, "context_size": 512, "dropped_features": DROP,
        "sap_model_configured": connection["model_configured"] if connection else None,
        "sap_deployment_configuration_name": connection["deployment_configuration_name"] if connection else None}
    path = OUT / "feature_selection_protocol.json"
    if path.exists() and not live:
        previous = json.loads(path.read_text())
        assert previous["source_sha256"] == protocol["source_sha256"]
        protocol = previous
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    test = source.iloc[1024:]
    for seed in SEEDS:
        indices = context_indices(seed)
        for target in TARGETS:
            for condition, remove in (("single_target_baseline", False), ("feature_removed", True)):
                if (seed, target, condition) in done or not live:
                    continue
                payload = request_for(source, indices, target, remove)
                response, elapsed = call_rpt(payload, connection)
                raw = response.get("predictions")
                assert isinstance(raw, list) and len(raw) == len(test)
                by_id = {str(item["invoice_id"]): item for item in raw}
                assert len(by_id) == len(test) and set(by_id) == set(test.invoice_id)
                scored = pd.DataFrame({"invoice_id": test.invoice_id.to_numpy(),
                    f"actual_{target}": test[target].to_numpy(),
                    f"predicted_{target}": [int(by_id[invoice_id][target][0]["prediction"])
                        for invoice_id in test.invoice_id]})
                rows.append({"seed": seed, "target": target, "condition": condition,
                    "removed_feature": DROP[target] if remove else "",
                    "request_sha256": hashlib.sha256(canonical(payload)).hexdigest(),
                    "inference_seconds": elapsed,
                    **metrics(scored[f"actual_{target}"], scored[f"predicted_{target}"])})
                for _, item in scored.iterrows():
                    predictions.append({"seed": seed, "target": target, "condition": condition,
                        "invoice_id": item.invoice_id, "actual": int(item[f"actual_{target}"]),
                        "predicted": int(item[f"predicted_{target}"])})
                pd.DataFrame(rows).to_csv(RESULTS, index=False)
                pd.DataFrame(predictions).to_csv(PREDICTIONS, index=False)
                print(seed, target, condition, flush=True)
    print(RESULTS)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    main(args.live)
