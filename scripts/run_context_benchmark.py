"""Run resumable SAP RPT context-size, selection, and label-noise experiments.

Default is a dry run that validates constructed requests and leakage checks.
Use --live to call the configured SAP deployment. Results are new files only.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from context_engineering import assert_no_leakage, select_context
from run_rpt16_invoices import connection_settings

SOURCE = ROOT / "data/semantic_ablation/source_snapshot.csv"
OUT = ROOT / "data/context_benchmark"
SEEDS = (11, 23, 37, 41, 53)
TARGETS = ("paid_late", "is_fraud")
SIZES = (16, 32, 64, 128, 256, 512, 1024)
STRATEGIES = ("random", "stratified", "balanced", "diversity", "similarity", "hybrid")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def request_for(context, query, target):
    assert_no_leakage(context, query)
    rows = pd.concat([context, query], ignore_index=True).to_dict("records")
    for row in rows[len(context):]:
        row[target] = "[PREDICT]"
    # The other target is unavailable at prediction time and excluded throughout.
    other = next(t for t in TARGETS if t != target)
    for row in rows:
        row.pop(other)
    columns = [c for c in context if c != other]
    schema = {c: {"dtype": "string" if c in ("invoice_id", "customer_segment", "region") else "numeric"} for c in columns}
    assert all(row[target] == "[PREDICT]" for row in rows[len(context):])
    return {"prediction_config": {"target_columns": [{"name": target,
        "prediction_placeholder": "[PREDICT]", "task_type": "classification"}]},
        "index_column": "invoice_id", "rows": rows, "data_schema": schema}


def call_rpt(payload, connection):
    request = Request(connection["deployment_url"] + "/predict", data=canonical(payload),
        headers={"Authorization": f"Bearer {connection['token']}",
            "AI-Resource-Group": connection["resource_group"], "Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    last = None
    for attempt in range(5):
        try:
            with urlopen(request, timeout=300) as response:
                result = json.load(response)
            time.sleep(1.5)  # controlled serial rate for query-specific requests
            return result, time.perf_counter() - start
        except HTTPError as exc:
            last = exc
            if exc.code == 429:
                time.sleep(max(30, int(exc.headers.get("Retry-After", "30"))))
            elif 500 <= exc.code < 600 and attempt < 4:
                time.sleep(2 ** attempt)
            else:
                raise
        except Exception as exc:
            last = exc
            if attempt < 4:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"SAP request failed after 5 attempts: {last}")


def score(response, query, target):
    predictions = response.get("predictions")
    assert isinstance(predictions, list) and len(predictions) == len(query)
    by_id = {str(row["invoice_id"]): row for row in predictions}
    assert len(by_id) == len(query) and set(by_id) == set(query.invoice_id)
    result = []
    for _, row in query.iterrows():
        value = int(by_id[row.invoice_id][target][0]["prediction"])
        assert value in (0, 1)
        result.append(value)
    return np.asarray(result)


def metric_row(actual, predicted):
    return {"accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "false_positives": int(((np.asarray(actual) == 0) & (predicted == 1)).sum()),
        "true_positives": int(((np.asarray(actual) == 1) & (predicted == 1)).sum())}


def xgb_predict(context, query, target, seed):
    columns = [c for c in context if c not in ("invoice_id", *TARGETS)]
    x_train = pd.get_dummies(context[columns], dtype=int)
    x_test = pd.get_dummies(query[columns], dtype=int).reindex(columns=x_train.columns, fill_value=0)
    labels = context[target]
    start = time.perf_counter()
    if labels.nunique() == 1:
        prediction = np.full(len(query), int(labels.iloc[0]))
        return prediction, time.perf_counter() - start, 0.0, prediction.astype(float)
    positives = int(labels.sum())
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=.05,
        subsample=.9, colsample_bytree=.9, objective="binary:logistic",
        eval_metric="logloss", random_state=seed, n_jobs=1,
        scale_pos_weight=(len(context) - positives) / positives if target == "is_fraud" else 1)
    model.fit(x_train, labels)
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= .5).astype(int)
    return prediction, time.perf_counter() - start, training_seconds, probability


def run(mode, live=False, source=SOURCE, out=OUT):
    data = pd.read_csv(source)
    assert len(data) == 1152 and data.invoice_id.is_unique and not data.isna().any().any()
    selection_pool = data.iloc[:896].copy()
    validation = data.iloc[896:1024].copy()
    holdout = data.iloc[1024:].copy()
    assert_no_leakage(selection_pool, validation)
    assert_no_leakage(data.iloc[:1024], holdout)
    out.mkdir(parents=True, exist_ok=True)
    output = out / f"{mode}_results.csv"
    prediction_file = out / f"{mode}_predictions.csv"
    partial_file = out / f"{mode}_query_partial.csv"
    rows = pd.read_csv(output).to_dict("records") if output.exists() else []
    predictions = pd.read_csv(prediction_file).to_dict("records") if prediction_file.exists() else []
    partial = pd.read_csv(partial_file).to_dict("records") if partial_file.exists() else []
    completed = {r["run_id"] for r in rows}
    connection = connection_settings() if live else None
    protocol = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selection_pool": [selection_pool.invoice_id.iloc[0], selection_pool.invoice_id.iloc[-1]],
        "validation": [validation.invoice_id.iloc[0], validation.invoice_id.iloc[-1]],
        "holdout": [holdout.invoice_id.iloc[0], holdout.invoice_id.iloc[-1]],
        "seeds": SEEDS, "mode": mode,
        "sap_model_configured": connection["model_configured"] if connection else None,
        "sap_deployment_configuration_name": connection["deployment_configuration_name"] if connection else None}
    protocol_path = out / f"{mode}_protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        assert previous["source_sha256"] == protocol["source_sha256"], "Source changed since saved runs"
        if not live and previous.get("sap_model_configured"):
            protocol = previous
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    jobs = []
    if mode == "size":
        for size in SIZES:
            for seed in SEEDS:
                pool = data.iloc[:1024] if size == 1024 else selection_pool
                for target in TARGETS:
                    jobs.append(("holdout", pool, holdout, "random", size, seed, target, 0))
    elif mode == "selection_validation":
        for strategy in ("random", "stratified", "balanced", "diversity"):
            for seed in SEEDS:
                for target in TARGETS:
                    jobs.append(("validation", selection_pool, validation, strategy, 128, seed, target, 0))
    elif mode == "selection_holdout":
        for strategy in STRATEGIES:
            for seed in SEEDS if strategy not in ("similarity", "hybrid") else (11,):
                for target in TARGETS:
                    jobs.append(("holdout", data.iloc[:1024], holdout, strategy, 128, seed, target, 0))
    elif mode == "noise":
        for level in (0, 5, 10, 20):
            for seed in SEEDS:
                for target in TARGETS:
                    jobs.append(("holdout", data.iloc[:1024], holdout, "random", 128, seed, target, level))
    elif mode == "quality":
        for strategy in ("duplicate", "region_shift"):
            for seed in SEEDS:
                for target in TARGETS:
                    jobs.append(("holdout", data.iloc[:1024], holdout, strategy, 128, seed, target, 0))
    else:
        raise ValueError(mode)
    for split, pool, query, strategy, size, seed, target, noise in jobs:
        eligible = pool[pool.region == "DACH"] if strategy == "region_shift" else pool
        context, build_seconds = select_context(eligible, target, query.iloc[:1], "random" if mode in ("noise", "quality") else strategy,
            size, seed) if strategy not in ("similarity", "hybrid") else (None, 0)
        if strategy == "duplicate":
            # Replace 20% of distinct examples with copies; unique row keys remain valid.
            context = context.copy().reset_index(drop=True)
            count = round(size * .2)
            for offset in range(count):
                destination = size - count + offset
                copied = context.iloc[offset].copy()
                copied["invoice_id"] = f"DUP-{seed}-{target}-{destination}"
                context.iloc[destination] = copied
        if context is not None and noise:
            context = context.copy()
            n_flip = round(len(context) * noise / 100)
            indices = np.random.default_rng(seed + 1000).choice(len(context), n_flip, replace=False)
            context.iloc[indices, context.columns.get_loc(target)] = 1 - context.iloc[indices][target].to_numpy()
        models = ("XGBoost", "SAP-RPT") if mode == "size" else ("SAP-RPT",)
        for model_name in models:
            run_id = f"{mode}|{split}|{strategy}|{size}|{seed}|{target}|{noise}|{model_name}"
            if run_id in completed or (model_name == "SAP-RPT" and not live):
                continue
            if model_name == "XGBoost":
                prediction, inference_seconds, training_seconds, probability = xgb_predict(context, query, target, seed)
            elif context is not None:
                payload = request_for(context, query, target)
                response, inference_seconds = call_rpt(payload, connection)
                prediction = score(response, query, target)
                training_seconds = 0
            else:
                all_prediction = []
                inference_seconds = build_seconds = 0
                for _, one in query.iterrows():
                    one = one.to_frame().T
                    prior = next((r for r in partial if r["run_id"] == run_id and r["invoice_id"] == one.invoice_id.iloc[0]), None)
                    if prior is not None:
                        all_prediction.append(int(prior["predicted"]))
                        inference_seconds += float(prior["inference_seconds"])
                        build_seconds += float(prior["context_build_seconds"])
                        continue
                    selected, build = select_context(pool, target, one, strategy, size, seed)
                    payload = request_for(selected, one, target)
                    response, latency = call_rpt(payload, connection)
                    value = int(score(response, one, target)[0])
                    all_prediction.append(value)
                    inference_seconds += latency
                    build_seconds += build
                    partial.append({"run_id": run_id, "invoice_id": one.invoice_id.iloc[0],
                        "predicted": value, "inference_seconds": latency, "context_build_seconds": build})
                    pd.DataFrame(partial).to_csv(partial_file, index=False)
                prediction = np.asarray(all_prediction)
                training_seconds = 0
            result = {"run_id": run_id, "split": split, "strategy": strategy, "size": size,
                "seed": seed, "target": target, "noise_percent": noise, "model": model_name,
                "context_positives": int(context[target].sum()) if context is not None else -1,
                "context_build_seconds": build_seconds, "inference_seconds": inference_seconds,
                "training_seconds": training_seconds, **metric_row(query[target], prediction)}
            if model_name == "XGBoost":
                result["roc_auc"] = roc_auc_score(query[target], probability)
                result["average_precision"] = average_precision_score(query[target], probability)
            rows.append(result)
            for position, (invoice_id, actual, predicted) in enumerate(zip(query.invoice_id, query[target], prediction)):
                item = {"run_id": run_id, "invoice_id": invoice_id,
                    "actual": int(actual), "predicted": int(predicted)}
                if model_name == "XGBoost":
                    item["probability"] = float(probability[position])
                predictions.append(item)
            pd.DataFrame(rows).to_csv(output, index=False)
            pd.DataFrame(predictions).to_csv(prediction_file, index=False)
            print(run_id, flush=True)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["size", "selection_validation", "selection_holdout", "noise", "quality"])
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    run(args.mode, args.live)
