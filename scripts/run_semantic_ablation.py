"""Paired SAP RPT / XGBoost semantic ablation on the current invoice CSV.

The output directory contains a frozen input copy, request/response hashes,
row-level predictions, and per-run metrics. No raw SAP response or secret is saved.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from generate_synthetic_invoices import CONTEXT_ROWS, FEATURES, TARGETS
from run_rpt16_invoices import connection_settings


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/synthetic_invoices/payment_behavior.csv"
OUT = ROOT / "data/semantic_ablation"
SEEDS = (11, 23, 37)
CONTEXT_SIZE = 512

# Verified against generate_synthetic_invoices.py. No label implies a new fact.
NAMES = {
    "invoice_amount_eur": "invoice_amount_in_euros",
    "payment_terms_days": "invoice_payment_terms_in_days",
    "customer_tenure_months": "customer_relationship_duration_in_months",
    "prior_late_payment_rate": "customer_historical_late_payment_fraction",
    "open_invoice_count": "customer_current_open_invoice_count",
    "customer_segment": "customer_size_segment",
    "region": "customer_region",
    "billing_address_mismatch": "billing_address_mismatch_indicator",
    "bank_account_changed_recently": "recent_bank_account_change_indicator",
    "weekend_submission": "invoice_submitted_on_weekend_indicator",
}
CATEGORIES = {
    "customer_segment": {
        "small": "small customer segment",
        "midmarket": "midmarket customer segment",
        "enterprise": "enterprise customer segment",
    },
    "region": {
        "DACH": "DACH region",
        "Nordics": "Nordic region",
        "UK": "United Kingdom region",
        "Southern Europe": "Southern European region",
    },
}
CONDITIONS = {
    "S0": (False, False, False),
    "S1": (True, False, False),
    "S2": (False, True, False),
    "S3": (True, True, False),
    "S4": (False, False, True),
    "S5": (True, True, True),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def load_and_audit(source):
    data = pd.read_csv(source)
    assert list(data.columns) == ["invoice_id", *FEATURES, *TARGETS]
    assert len(data) == 1152 and data.invoice_id.is_unique
    assert not data.isna().any().any()
    assert all(set(data[t]) == {0, 1} for t in TARGETS)
    for col, mapping in CATEGORIES.items():
        assert set(data[col]) == set(mapping), (col, set(data[col]))
    # Every value is already valid and complete; S4/S5 keep the same values.
    assert (data.select_dtypes(include="number").ge(0)).all().all()
    return data


def context_indices(seed):
    return np.random.default_rng(seed).choice(CONTEXT_ROWS, CONTEXT_SIZE, replace=False).tolist()


def semantic_frame(frame, names, categories):
    result = frame.copy()
    if categories:
        for col, mapping in CATEGORIES.items():
            result[col] = result[col].map(mapping)
            assert result[col].notna().all()
    if names:
        result = result.rename(columns=NAMES)
    return result


def request_payload(data, indices, condition):
    names, categories, _ = CONDITIONS[condition]
    selected = pd.concat([data.iloc[indices], data.iloc[CONTEXT_ROWS:]], ignore_index=True)
    represented = semantic_frame(selected, names, categories)
    rows = represented.to_dict(orient="records")
    for row in rows[CONTEXT_SIZE:]:
        for target in TARGETS:
            row[target] = "[PREDICT]"
    schema = {key: {"dtype": "string" if key in ("invoice_id", "customer_segment", "region", NAMES["customer_segment"], NAMES["region"]) else "numeric"} for key in represented.columns}
    return {
        "prediction_config": {"target_columns": [
            {"name": target, "prediction_placeholder": "[PREDICT]", "task_type": "classification"}
            for target in TARGETS
        ]},
        "index_column": "invoice_id",
        "rows": rows,
        "data_schema": schema,
    }


def verify_pairing(data, indices):
    base = request_payload(data, indices, "S0")
    original = pd.DataFrame(base["rows"])
    for condition in CONDITIONS:
        payload = request_payload(data, indices, condition)
        names, categories, _ = CONDITIONS[condition]
        frame = pd.DataFrame(payload["rows"])
        if names:
            frame = frame.rename(columns={value: key for key, value in NAMES.items()})
        if categories:
            for col, mapping in CATEGORIES.items():
                frame[col] = frame[col].replace({value: key for key, value in mapping.items()})
        pd.testing.assert_frame_equal(frame[original.columns], original)
        schema = payload["data_schema"]
        if names:
            schema = {({value: key for key, value in NAMES.items()}.get(key, key)): value
                for key, value in schema.items()}
        assert schema == base["data_schema"]
        assert payload["prediction_config"] == base["prediction_config"]
        assert len(payload["rows"]) == CONTEXT_SIZE + 128


def encoded_matrix(train, test):
    # Semantic strings never reach XGBoost. A single matrix is reused for S0-S5.
    columns = [column for column in train if column not in ("invoice_id", *TARGETS)]
    encoded_train = pd.get_dummies(train[columns], dtype=int)
    encoded_test = pd.get_dummies(test[columns], dtype=int).reindex(
        columns=encoded_train.columns, fill_value=0)
    return encoded_train, encoded_test


def metrics(y, prediction):
    return {
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "precision": precision_score(y, prediction, zero_division=0),
        "recall": recall_score(y, prediction, zero_division=0),
        "f1": f1_score(y, prediction, zero_division=0),
    }


def score_rpt(response, test):
    predictions = response.get("predictions")
    assert isinstance(predictions, list) and len(predictions) == len(test)
    by_id = {str(row["invoice_id"]): row for row in predictions}
    assert len(by_id) == len(test) and set(by_id) == set(test.invoice_id)
    scored = []
    for _, source in test.iterrows():
        item = {"invoice_id": source.invoice_id}
        for target in TARGETS:
            choices = by_id[source.invoice_id][target]
            assert isinstance(choices, list) and choices
            prediction = int(choices[0]["prediction"])
            assert prediction in (0, 1)
            item[f"actual_{target}"] = int(source[target])
            item[f"predicted_{target}"] = prediction
        scored.append(item)
    return pd.DataFrame(scored)


def call_rpt(payload, connection):
    request = Request(
        connection["deployment_url"] + "/predict",
        data=canonical(payload),
        headers={
            "Authorization": f"Bearer {connection['token']}",
            "AI-Resource-Group": connection["resource_group"],
            "Content-Type": "application/json",
        },
        method="POST",
    )
    start = time.perf_counter()
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    return result, time.perf_counter() - start


def run(source=SOURCE, out=OUT, live=False):
    data = load_and_audit(source)
    out.mkdir(parents=True, exist_ok=True)
    frozen = out / "source_snapshot.csv"
    if frozen.exists():
        assert digest(frozen.read_bytes()) == digest(source.read_bytes()), "Input changed since the experiment began"
    else:
        frozen.write_bytes(source.read_bytes())
    source_hash = digest(frozen.read_bytes())
    result_path = out / "results.csv"
    prediction_path = out / "predictions.csv"
    existing = pd.read_csv(result_path).to_dict("records") if result_path.exists() else []
    predictions = pd.read_csv(prediction_path).to_dict("records") if prediction_path.exists() else []
    completed = {(r["model"], r["condition"], int(r["seed"]), r["target"]) for r in existing}
    connection = connection_settings() if live else None
    metadata = {
        "source_sha256": source_hash,
        "source_path": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "context_size": CONTEXT_SIZE,
        "test_size": len(data) - CONTEXT_ROWS,
        "seeds": list(SEEDS),
        "conditions": CONDITIONS,
        "names": NAMES,
        "categories": CATEGORIES,
        "cleaning": "identity: no missing values, duplicates, invalid category codes or out-of-range numeric values",
        "sap_model_configured": connection["model_configured"] if connection else None,
        "sap_deployment_configuration_name": connection["deployment_configuration_name"] if connection else None,
        "xgboost_version": __import__("xgboost").__version__,
    }
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        assert previous["source_sha256"] == source_hash, "Source changed since saved runs"
        if not live and previous.get("sap_model_configured"):
            metadata = previous
            metadata["source_path"] = (
                str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source)
            )
    protocol_path.write_text(json.dumps(metadata, indent=2) + "\n")
    for seed in SEEDS:
        indices = context_indices(seed)
        verify_pairing(data, indices)
        train = data.iloc[indices]
        test = data.iloc[CONTEXT_ROWS:]
        x_train, x_test = encoded_matrix(train, test)
        for target in TARGETS:
            if ("XGBoost", "S0", seed, target) not in completed:
                positives = int(train[target].sum())
                model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9, objective="binary:logistic",
                    eval_metric="logloss", random_state=seed, n_jobs=1,
                    scale_pos_weight=(len(train) - positives) / positives if target == "is_fraud" else 1)
                model.fit(x_train, train[target])
                start = time.perf_counter()
                probability = model.predict_proba(x_test)[:, 1]
                prediction = (probability >= 0.5).astype(int)
                elapsed = time.perf_counter() - start
                for condition in CONDITIONS:
                    values = {"model": "XGBoost", "condition": condition, "seed": seed, "target": target,
                        "inference_seconds": elapsed, "source_sha256": source_hash,
                        "roc_auc": roc_auc_score(test[target], probability),
                        "average_precision": average_precision_score(test[target], probability),
                        **metrics(test[target], prediction)}
                    existing.append(values)
                    for invoice_id, actual, pred, prob in zip(test.invoice_id, test[target], prediction, probability):
                        predictions.append({"model": "XGBoost", "condition": condition, "seed": seed,
                            "target": target, "invoice_id": invoice_id, "actual": int(actual),
                            "predicted": int(pred), "probability": float(prob)})
                pd.DataFrame(existing).to_csv(result_path, index=False)
                pd.DataFrame(predictions).to_csv(prediction_path, index=False)
        if not live:
            continue
        for condition in CONDITIONS:
            if all(("SAP-RPT", condition, seed, target) in completed for target in TARGETS):
                continue
            payload = request_payload(data, indices, condition)
            response, elapsed = call_rpt(payload, connection)
            scored = score_rpt(response, test)
            request_hash, response_hash = digest(canonical(payload)), digest(canonical(response))
            for target in TARGETS:
                existing.append({"model": "SAP-RPT", "condition": condition, "seed": seed,
                    "target": target, "inference_seconds": elapsed, "source_sha256": source_hash,
                    "request_sha256": request_hash, "response_sha256": response_hash,
                    **metrics(scored[f"actual_{target}"], scored[f"predicted_{target}"])})
                for _, row in scored.iterrows():
                    predictions.append({"model": "SAP-RPT", "condition": condition, "seed": seed,
                        "target": target, "invoice_id": row.invoice_id,
                        "actual": int(row[f"actual_{target}"]), "predicted": int(row[f"predicted_{target}"])})
            pd.DataFrame(existing).to_csv(result_path, index=False)
            pd.DataFrame(predictions).to_csv(prediction_path, index=False)
            print(f"seed={seed} {condition} SAP-RPT {elapsed:.1f}s", flush=True)
    print(f"Results: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--live", action="store_true", help="Send paired requests to configured SAP deployment")
    args = parser.parse_args()
    run(args.source, args.output_dir, args.live)
