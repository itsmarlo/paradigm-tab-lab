# Archived eight-case synthetic invoice dataset

This was the original eight-case experiment: 53 of 1,024 context invoices and 8 of 128 test invoices have the fraud label. The labels are synthetic and the eight fraud cases are too few for a strong performance claim.

Use `exports/rpt_upload.csv` for a new Playground run; its 128 test targets are masked as `[PREDICT]`. `rpt16_request.json` is the matching BTP request. The saved XGBoost and RPT results belong only to this eight-case test. `exports/rpt_playground_export.csv` is a supplied Playground result export; its exact model configuration is unverified. `model_tradeoff_with_playground.png` visualizes the three saved results.

To reproduce the inputs and local XGBoost result from the repository root:

```sh
python3 scripts/generate_synthetic_invoices.py --output-dir archive/synthetic_invoice_experiments/rare_fraud_baseline --fraud-intercept -4.2
python3 scripts/run_xgboost_invoices.py --data-dir archive/synthetic_invoice_experiments/rare_fraud_baseline
python3 scripts/plot_rare_fraud_baseline.py
```

The saved RPT result came from a direct BTP run. Rerunning `scripts/run_rpt16_invoices.py --data-dir archive/synthetic_invoice_experiments/rare_fraud_baseline` requires a configured deployment and makes a live inference call. Do not mix these results with the [active 22-case experiment](../../../data/synthetic_invoices/README.md).
