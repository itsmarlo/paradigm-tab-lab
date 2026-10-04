# Fraud context experiment

This secondary sensitivity check changes only the synthetic fraud labels in the first 1,024 context rows. The fraud label generator uses an intercept of `-2.8` instead of `-4.2`, producing 187 fraud labels instead of 53. All invoice predictors, all late-payment labels, and the final 128 test rows and labels match the archived rare-fraud baseline. The RPT request masks both targets for those test rows. The published main dataset uses the `-2.8` generator for both context and test labels.

Both models were rerun on this scenario. `xgboost_results.csv` is the local XGBoost result. `rpt16_results.csv` is the direct SAP BTP result from the configured `sap-rpt-1.6-large` deployment; `rpt16_run_metadata.json` records its source and checksums. The original rare-fraud files are in `../rare_fraud_baseline/`.
The [before-and-after chart](fraud_context_enrichment_for_ppt.png) is a secondary sensitivity figure, not the main TechEd comparison slide.

| Model | Fraud found of 8 | False alarms of 120 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| XGBoost, original context | 2 | 6 | 25.0% | 25.0% |
| XGBoost, more fraud labels | 5 | 19 | 20.8% | 62.5% |
| RPT BTP, original context | 0 | 2 | 0.0% | 0.0% |
| RPT BTP, more fraud labels | 3 | 12 | 20.0% | 37.5% |

To regenerate the scenario and rerun the models from the repository root:

```sh
python3 scripts/generate_synthetic_invoices.py --output-dir archive/synthetic_invoice_experiments/context_only --fraud-intercept -2.8 --test-labels-from archive/synthetic_invoice_experiments/rare_fraud_baseline/payment_behavior.csv
python3 scripts/run_xgboost_invoices.py --data-dir archive/synthetic_invoice_experiments/context_only
python3 scripts/run_rpt16_invoices.py --data-dir archive/synthetic_invoice_experiments/context_only
python3 scripts/plot_fraud_enrichment.py
```

The RPT command makes a live BTP inference call. This experiment tests a synthetic change in label prevalence; it does not establish that either model will perform this way on real fraud. Only eight test invoices are fraudulent, and the enriched context has a different fraud prevalence than the fixed test set.
