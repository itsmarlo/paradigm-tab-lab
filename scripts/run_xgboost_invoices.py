"""Train an XGBoost baseline on the synthetic invoice split."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from generate_synthetic_invoices import CONTEXT_ROWS, PREDICTION_ROWS, TARGETS


DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic_invoices"


def load_split(data=DATA):
    invoices = pd.read_csv(data / "payment_behavior.csv")
    if len(invoices) != CONTEXT_ROWS + PREDICTION_ROWS:
        raise ValueError("The invoice dataset has an unexpected number of rows")
    train = invoices.iloc[:CONTEXT_ROWS]
    test = invoices.iloc[CONTEXT_ROWS:]
    return train, test


def encode_features(frame, columns=None):
    excluded = ["invoice_id", *TARGETS]
    encoded = pd.get_dummies(frame.drop(columns=excluded, errors="ignore"), dtype=int)
    return encoded if columns is None else encoded.reindex(columns=columns, fill_value=0)


def train_models(train):
    x_train = encode_features(train)
    models = {}
    for target in TARGETS:
        positives = int(train[target].sum())
        negatives = len(train) - positives
        model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=20261002,
            n_jobs=1,
            scale_pos_weight=negatives / positives if target == "is_fraud" else 1,
        )
        model.fit(x_train, train[target])
        models[target] = model
    return models, x_train.columns


def predict_frame(models, columns, frame):
    x_test = encode_features(frame, columns)
    results = frame.drop(columns=list(TARGETS), errors="ignore").copy()
    for target in TARGETS:
        probabilities = models[target].predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        if target in frame:
            results[f"actual_{target}"] = frame[target].to_numpy()
        results[f"predicted_{target}"] = predictions
        results[f"probability_{target}"] = probabilities
    return results


def main(data=DATA):
    train, test = load_split(data)
    models, columns = train_models(train)
    results = predict_frame(models, columns, test)
    for target in TARGETS:
        predictions = results[f"predicted_{target}"]
        probabilities = results[f"probability_{target}"]
        print(
            f"{target}: accuracy={accuracy_score(test[target], predictions):.3f}, "
            f"ROC AUC={roc_auc_score(test[target], probabilities):.3f}, "
            f"precision={precision_score(test[target], predictions, zero_division=0):.3f}, "
            f"recall={recall_score(test[target], predictions, zero_division=0):.3f}"
        )
    output = data / "xgboost_results.csv"
    results.to_csv(output, index=False)
    print(f"Predictions: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()
    main(args.data_dir)
