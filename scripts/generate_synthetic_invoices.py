"""Create repeatable synthetic invoice data for XGBoost and SAP-RPT-1.6."""

import csv
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "synthetic_invoices"
SEED = 20261002
CONTEXT_ROWS = 1024
PREDICTION_ROWS = 128
TARGETS = ("paid_late", "is_fraud")
FEATURES = [
    "invoice_amount_eur",
    "payment_terms_days",
    "customer_tenure_months",
    "prior_late_payment_rate",
    "open_invoice_count",
    "customer_segment",
    "region",
    "billing_address_mismatch",
    "bank_account_changed_recently",
    "weekend_submission",
]
COLUMNS = ["invoice_id", *FEATURES, *TARGETS]


def make_rows():
    rng = random.Random(SEED)
    rows = []
    for index in range(CONTEXT_ROWS + PREDICTION_ROWS):
        segment = rng.choices(["small", "midmarket", "enterprise"], [0.45, 0.35, 0.20])[0]
        region = rng.choice(["DACH", "Nordics", "UK", "Southern Europe"])
        amount = round(math.exp(rng.gauss(7.5, 0.85)), 2)
        terms = rng.choice([14, 30, 45, 60])
        tenure = rng.randint(1, 120)
        prior_late = round(min(0.95, max(0.0, rng.betavariate(1.5, 4.5))), 3)
        open_count = rng.randrange(0, 11)
        address_mismatch = int(rng.random() < 0.12)
        bank_change = int(rng.random() < 0.08)
        weekend = int(rng.random() < 0.25)
        # A learnable, noisy outcome built only from information available at invoice issue.
        score = (
            -2.0
            + 3.3 * prior_late
            + 0.13 * open_count
            + 0.00011 * amount
            + (0.35 if terms >= 45 else 0)
            - 0.008 * tenure
            + (0.35 if segment == "small" else -0.2 if segment == "enterprise" else 0)
            + (0.15 if region == "Southern Europe" else 0)
        )
        probability = 1 / (1 + math.exp(-score))
        fraud_score = (
            -4.2
            + 2.2 * address_mismatch
            + 2.5 * bank_change
            + 0.6 * weekend
            + 0.00008 * amount
            + 0.6 * prior_late
            + (0.35 if tenure < 6 else 0)
        )
        fraud_probability = 1 / (1 + math.exp(-fraud_score))
        rows.append({
            "invoice_id": f"SYN-{index + 1:05d}",
            "invoice_amount_eur": amount,
            "payment_terms_days": terms,
            "customer_tenure_months": tenure,
            "prior_late_payment_rate": prior_late,
            "open_invoice_count": open_count,
            "customer_segment": segment,
            "region": region,
            "billing_address_mismatch": address_mismatch,
            "bank_account_changed_recently": bank_change,
            "weekend_submission": weekend,
            "paid_late": int(rng.random() < probability),
            "is_fraud": int(rng.random() < fraud_probability),
        })
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = make_rows()
    context = rows[:CONTEXT_ROWS]
    holdout = rows[CONTEXT_ROWS:]
    prompt_rows = context + [
        {**row, **{target: "[PREDICT]" for target in TARGETS}} for row in holdout
    ]
    assert len({row["invoice_id"] for row in rows}) == len(rows)
    for target in TARGETS:
        assert {row[target] for row in context} == {0, 1}
        assert {row[target] for row in holdout} == {0, 1}
    write_csv(OUT / "payment_behavior.csv", rows)
    write_csv(OUT / "rpt16_prompt.csv", prompt_rows)
    payload = {
        "prediction_config": {"target_columns": [
            {"name": target, "prediction_placeholder": "[PREDICT]", "task_type": "classification"}
            for target in TARGETS
        ]},
        "index_column": "invoice_id",
        "rows": prompt_rows,
        "data_schema": {
            "invoice_id": {"dtype": "string"},
            "invoice_amount_eur": {"dtype": "numeric"},
            "payment_terms_days": {"dtype": "numeric"},
            "customer_tenure_months": {"dtype": "numeric"},
            "prior_late_payment_rate": {"dtype": "numeric"},
            "open_invoice_count": {"dtype": "numeric"},
            "customer_segment": {"dtype": "string"},
            "region": {"dtype": "string"},
            "billing_address_mismatch": {"dtype": "numeric"},
            "bank_account_changed_recently": {"dtype": "numeric"},
            "weekend_submission": {"dtype": "numeric"},
            **{target: {"dtype": "numeric"} for target in TARGETS},
        },
    }
    (OUT / "rpt16_request.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(context)} context rows and {len(holdout)} held-out rows in {OUT}")


if __name__ == "__main__":
    main()
