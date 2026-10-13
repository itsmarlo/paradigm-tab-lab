"""Select semantic representation on validation rows, without consulting holdout labels."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_rpt16_invoices import connection_settings
from run_semantic_ablation import (CONTEXT_SIZE, OUT, SEEDS, TARGETS, call_rpt,
    canonical, metrics, request_payload, score_rpt, verify_pairing)

SOURCE = OUT / "source_snapshot.csv"
RESULTS = OUT / "validation_results.csv"
PREDICTIONS = OUT / "validation_predictions.csv"


def main(live=False):
    source = pd.read_csv(SOURCE)
    assert len(source) == 1152 and source.invoice_id.is_unique
    # Reuse request builder: its 1024-row prefix is only an indexable pool;
    # sampled indices are restricted to 0:896. The final 128 positions hold validation.
    validation_data = pd.concat([source.iloc[:1024], source.iloc[896:1024]], ignore_index=True)
    test = validation_data.iloc[1024:]
    rows = pd.read_csv(RESULTS).to_dict("records") if RESULTS.exists() else []
    predictions = pd.read_csv(PREDICTIONS).to_dict("records") if PREDICTIONS.exists() else []
    done = {(int(row["seed"]), row["condition"]) for row in rows}
    connection = connection_settings() if live else None
    protocol = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "selection_pool": [source.invoice_id.iloc[0], source.invoice_id.iloc[895]],
        "validation": [test.invoice_id.iloc[0], test.invoice_id.iloc[-1]],
        "context_size": CONTEXT_SIZE, "seeds": SEEDS,
        "sap_model_configured": connection["model_configured"] if connection else None,
        "sap_deployment_configuration_name": connection["deployment_configuration_name"] if connection else None,
    }
    protocol_path = OUT / "validation_protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        assert previous["source_sha256"] == protocol["source_sha256"]
        if not live and previous.get("sap_model_configured"):
            protocol = previous
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    for seed in SEEDS:
        indices = np.random.default_rng(seed).choice(896, CONTEXT_SIZE, replace=False).tolist()
        verify_pairing(validation_data, indices)
        assert not set(validation_data.iloc[indices].invoice_id) & set(test.invoice_id)
        for condition in ("S0", "S1", "S2", "S3"):
            if (seed, condition) in done or not live:
                continue
            payload = request_payload(validation_data, indices, condition)
            response, elapsed = call_rpt(payload, connection)
            scored = score_rpt(response, test)
            for target in TARGETS:
                rows.append({"seed": seed, "condition": condition, "target": target,
                    "inference_seconds": elapsed,
                    "request_sha256": hashlib.sha256(canonical(payload)).hexdigest(),
                    **metrics(scored[f"actual_{target}"], scored[f"predicted_{target}"])})
                for _, item in scored.iterrows():
                    predictions.append({"seed": seed, "condition": condition, "target": target,
                        "invoice_id": item.invoice_id, "actual": int(item[f"actual_{target}"]),
                        "predicted": int(item[f"predicted_{target}"])})
            pd.DataFrame(rows).to_csv(RESULTS, index=False)
            pd.DataFrame(predictions).to_csv(PREDICTIONS, index=False)
            print(seed, condition, elapsed, flush=True)
    print(RESULTS)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    main(args.live)
