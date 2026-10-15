"""Test fraud-context composition and class scores for SAP RPT.

The first 896 original invoices supply labeled context. The next 128 select
the context size, fraud fraction, and score threshold. The final 128 are scored
only after selection. Their results remain exploratory because earlier work
already examined that holdout.

Dry run: python scripts/run_rpt_context_mix.py
Live validation: python scripts/run_rpt_context_mix.py --live
Live holdout: python scripts/run_rpt_context_mix.py --live --split holdout
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from context_engineering import assert_no_leakage
from run_context_benchmark import call_rpt, metric_row, request_for
from run_rpt16_invoices import connection_settings

SOURCE = ROOT / "archive/synthetic_invoice_experiments/published_baseline/payment_behavior.csv"
OUT = ROOT / "data/rpt_context_mix"
SEEDS = (11, 23, 37, 41, 53)
SIZES = (128, 256)
POSITIVE_FRACTIONS = (0.20, 0.35, 0.50)
THRESHOLDS = tuple(round(x, 2) for x in np.arange(0.05, 0.951, 0.05))
FPR_CAP = 9 / 106  # XGBoost's reference false-alarm rate on the 128-invoice test set.


def split_data():
    data = pd.read_csv(SOURCE)
    assert len(data) == 1152 and data.invoice_id.is_unique and not data.isna().any().any()
    context_pool, validation, holdout = data.iloc[:896], data.iloc[896:1024], data.iloc[1024:]
    assert len(validation) == len(holdout) == 128
    assert_no_leakage(context_pool, validation)
    assert_no_leakage(data.iloc[:1024], holdout)
    return context_pool, validation, holdout


def select_rows(pool, size, positive_fraction, seed):
    """Sample a fixed number of positive and negative examples without replacement."""
    n_positive = round(size * positive_fraction)
    positive = pool.index[pool.is_fraud == 1].to_numpy()
    negative = pool.index[pool.is_fraud == 0].to_numpy()
    if not 0 < n_positive < size or n_positive > len(positive) or size - n_positive > len(negative):
        raise ValueError("Requested context composition is unavailable in the pool")
    rng = np.random.default_rng(seed)
    chosen = np.r_[rng.choice(positive, n_positive, replace=False),
                   rng.choice(negative, size - n_positive, replace=False)]
    rng.shuffle(chosen)
    selected = pool.loc[chosen].copy()
    assert len(selected) == size and int(selected.is_fraud.sum()) == n_positive
    return selected


def parse_positive_scores(response, query):
    """Require both binary candidates with finite confidence values for every row."""
    rows = response.get("predictions")
    if not isinstance(rows, list) or len(rows) != len(query):
        raise ValueError("SAP RPT did not return one prediction row per query invoice")
    by_id = {str(row["invoice_id"]): row for row in rows}
    if len(by_id) != len(query) or set(by_id) != set(query.invoice_id.astype(str)):
        raise ValueError("SAP RPT prediction IDs do not match the query invoices")
    result = []
    for invoice_id, actual in zip(query.invoice_id.astype(str), query.is_fraud):
        choices = by_id[invoice_id].get("is_fraud")
        if not isinstance(choices, list) or len(choices) != 2:
            raise ValueError("top_k=2 did not return both fraud labels")
        scores = {}
        for choice in choices:
            label = int(choice["prediction"])
            confidence = float(choice["confidence"])
            if label not in (0, 1) or label in scores or not np.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("SAP RPT returned invalid binary candidate scores")
            scores[label] = confidence
        if set(scores) != {0, 1}:
            raise ValueError("SAP RPT did not return both fraud labels")
        result.append({"invoice_id": invoice_id, "actual": int(actual),
                       "score_negative": scores[0], "score_positive": scores[1],
                       "top_label": int(choices[0]["prediction"])})
    return pd.DataFrame(result)


def run_plan(split, selected=None):
    if split == "validation":
        return [(size, fraction, seed) for size in SIZES
                for fraction in POSITIVE_FRACTIONS for seed in SEEDS]
    if selected is None:
        raise ValueError("A validation-selected configuration is required for holdout")
    return [(int(selected["context_rows"]), float(selected["positive_fraction"]), seed)
            for seed in SEEDS]


def score_grid(predictions, split):
    rows = []
    for (size, fraction), subset in predictions.groupby(["context_rows", "positive_fraction"]):
        for threshold in THRESHOLDS:
            seed_rows = []
            for seed, group in subset.groupby("seed"):
                predicted = (group.score_positive.to_numpy() >= threshold).astype(int)
                metrics = metric_row(group.actual.to_numpy(), predicted)
                negatives = int((group.actual == 0).sum())
                seed_rows.append({"seed": seed, **metrics,
                                  "false_positive_rate": metrics["false_positives"] / negatives})
            per_seed = pd.DataFrame(seed_rows)
            rows.append({"split": split, "context_rows": int(size),
                         "positive_fraction": float(fraction), "threshold": threshold,
                         "seeds": len(per_seed), "mean_f1": per_seed.f1.mean(),
                         "mean_recall": per_seed.recall.mean(),
                         "mean_precision": per_seed.precision.mean(),
                         "mean_true_positives": per_seed.true_positives.mean(),
                         "mean_false_positives": per_seed.false_positives.mean(),
                         "mean_false_positive_rate": per_seed.false_positive_rate.mean()})
    return pd.DataFrame(rows)


def save_csv(frame, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def main(live=False, split="validation", max_runs=None):
    context_pool, validation, holdout = split_data()
    selected_path = OUT / "selected.json"
    if split == "holdout":
        if not selected_path.exists():
            raise FileNotFoundError("Finish and select the validation experiment first")
        selected = json.loads(selected_path.read_text())
        pool, query = pd.concat([context_pool, validation]), holdout
    else:
        selected = None
        pool, query = context_pool, validation
    plan = run_plan(split, selected)
    for size, fraction, seed in plan:
        context = select_rows(pool, size, fraction, seed)
        assert_no_leakage(pool, query, context)
        payload = request_for(context, query, "is_fraud")
        payload["prediction_config"]["target_columns"][0]["top_k"] = 2
        assert len(payload["rows"]) == size + len(query)
        assert all(row["is_fraud"] == "[PREDICT]" for row in payload["rows"][size:])
    print(f"Checked {len(plan)} {split} requests with top_k=2 and masked query labels")
    if not live:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path = OUT / "protocol.json"
    protocol = {"source": str(SOURCE.relative_to(ROOT)),
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "validation_rows": len(validation), "holdout_rows": len(holdout),
                "sizes": SIZES, "positive_fractions": POSITIVE_FRACTIONS,
                "seeds": SEEDS, "top_k": 2, "thresholds": THRESHOLDS,
                "fpr_cap": FPR_CAP}
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != json.loads(json.dumps(protocol)):
            raise ValueError("Saved RPT experiment protocol differs from this run")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    prediction_path = OUT / f"{split}_predictions.csv"
    predictions = pd.read_csv(prediction_path) if prediction_path.exists() else pd.DataFrame()
    complete = set(predictions.run_id.unique()) if not predictions.empty else set()
    missing = [(size, fraction, seed) for size, fraction, seed in plan
               if f"{split}|{size}|{fraction:.2f}|{seed}" not in complete]
    if max_runs is not None:
        missing = missing[:max_runs]
    if missing:
        connection = connection_settings()
        for size, fraction, seed in missing:
            context = select_rows(pool, size, fraction, seed)
            payload = request_for(context, query, "is_fraud")
            payload["prediction_config"]["target_columns"][0]["top_k"] = 2
            response, seconds = call_rpt(payload, connection)
            scored = parse_positive_scores(response, query)
            run_id = f"{split}|{size}|{fraction:.2f}|{seed}"
            scored.insert(0, "run_id", run_id)
            scored.insert(1, "context_rows", size)
            scored.insert(2, "positive_fraction", fraction)
            scored.insert(3, "context_positives", int(context.is_fraud.sum()))
            scored.insert(4, "seed", seed)
            predictions = pd.concat([predictions, scored], ignore_index=True)
            save_csv(predictions, prediction_path)
            print(f"Saved {run_id} ({seconds:.1f}s)", flush=True)
    expected_ids = {f"{split}|{size}|{fraction:.2f}|{seed}" for size, fraction, seed in plan}
    if set(predictions.run_id.unique()) != expected_ids:
        print(f"Saved {len(set(predictions.run_id.unique()))}/{len(expected_ids)} runs; selection awaits completion")
        return
    grid = score_grid(predictions, split)
    save_csv(grid, OUT / f"{split}_threshold_grid.csv")
    if split == "validation":
        eligible = grid[(grid.seeds == len(SEEDS)) &
                        (grid.mean_false_positive_rate <= FPR_CAP)]
        if eligible.empty:
            print("No candidate met the false-alarm cap; no holdout configuration selected")
            return
        winner = eligible.sort_values(["mean_f1", "mean_false_positive_rate"],
                                      ascending=[False, True]).iloc[0]
        selection = {"context_rows": int(winner.context_rows),
                     "positive_fraction": float(winner.positive_fraction),
                     "threshold": float(winner.threshold),
                     "validation_mean_f1": float(winner.mean_f1),
                     "validation_mean_false_positives": float(winner.mean_false_positives),
                     "validation_mean_true_positives": float(winner.mean_true_positives),
                     "rule": "Maximize mean F1 over five seeds subject to FPR <= 9/106"}
        selected_path.write_text(json.dumps(selection, indent=2) + "\n")
        print("Selected", selection)
    else:
        condition = grid[(grid.context_rows == selected["context_rows"]) &
                         (grid.positive_fraction == selected["positive_fraction"]) &
                         (grid.threshold == selected["threshold"])].iloc[0]
        print("Holdout at the selected threshold:", condition.to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Call SAP RPT for missing conditions")
    parser.add_argument("--split", choices=("validation", "holdout"), default="validation")
    parser.add_argument("--max-runs", type=int, help="Limit new live calls; saved results are reusable")
    args = parser.parse_args()
    main(args.live, args.split, args.max_runs)
