"""Train an XGBoost baseline on the synthetic invoice split."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from generate_synthetic_invoices import CONTEXT_ROWS, PREDICTION_ROWS, TARGETS


DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic_invoices"


def main():
    invoices = pd.read_csv(DATA / "payment_behavior.csv")
    if len(invoices) != CONTEXT_ROWS + PREDICTION_ROWS:
        raise ValueError("The invoice dataset has an unexpected number of rows")
    train = invoices.iloc[:CONTEXT_ROWS]
    test = invoices.iloc[CONTEXT_ROWS:]
    excluded = ["invoice_id", *TARGETS]
    x_train = pd.get_dummies(train.drop(columns=excluded), dtype=int)
    x_test = pd.get_dummies(test.drop(columns=excluded), dtype=int).reindex(
        columns=x_train.columns, fill_value=0
    )
    results = test.drop(columns=list(TARGETS)).copy()
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
        probabilities = model.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        results[f"actual_{target}"] = test[target]
        results[f"predicted_{target}"] = predictions
        results[f"probability_{target}"] = probabilities
        print(
            f"{target}: accuracy={accuracy_score(test[target], predictions):.3f}, "
            f"ROC AUC={roc_auc_score(test[target], probabilities):.3f}, "
            f"precision={precision_score(test[target], predictions, zero_division=0):.3f}, "
            f"recall={recall_score(test[target], predictions, zero_division=0):.3f}"
        )
    output = DATA / "xgboost_results.csv"
    results.to_csv(output, index=False)
    print(f"Predictions: {output}")


if __name__ == "__main__":
    main()
