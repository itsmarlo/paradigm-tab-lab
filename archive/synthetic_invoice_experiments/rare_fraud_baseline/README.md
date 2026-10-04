# Archived rare-fraud baseline

This is the original synthetic dataset: 53 of 1,024 context invoices and 8 of 128 test invoices have the fraud label. It is retained as a secondary sensitivity comparison. The [main TechEd dataset](../../../data/synthetic_invoices/README.md) has 187 context fraud labels and 22 test fraud labels.

The saved XGBoost and RPT results belong only to this eight-case test. `exports/rpt_playground_export.csv` is a supplied Playground table export for these rows; its exact Playground model configuration is unverified. It must not be compared with the main 22-case dataset. `model_tradeoff_with_playground.png` visualizes these archived results.

To reproduce the inputs and local XGBoost result from the repository root:

```sh
python3 scripts/generate_synthetic_invoices.py --output-dir archive/synthetic_invoice_experiments/rare_fraud_baseline --fraud-intercept -4.2
python3 scripts/run_xgboost_invoices.py --data-dir archive/synthetic_invoice_experiments/rare_fraud_baseline
python3 scripts/plot_rare_fraud_baseline.py
```

The saved RPT result came from a direct BTP run. Rerunning `scripts/run_rpt16_invoices.py --data-dir archive/synthetic_invoice_experiments/rare_fraud_baseline` requires a configured deployment and makes a live inference call. These synthetic results are illustrative, and eight fraud cases are too few for a strong general claim.
