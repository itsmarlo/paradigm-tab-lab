# Working synthetic invoice dataset and historical model outputs

The working `payment_behavior.csv` has revised fraud labels and matches the frozen [context engineering source](../semantic_ablation/source_snapshot.csv). It has 1,152 synthetic invoices: the first 1,024 can supply training or context rows, and the final 128 form the test set. The current holdout has **8 fraud cases**. `invoice_id` is a row key, not a predictor. Both targets use `1` for yes.

The saved `xgboost_results.csv`, `rpt16_results.csv`, masked request and export in this directory belong to the **historical 22-case version**, preserved as an [archived source](../../archive/synthetic_invoice_experiments/published_baseline/payment_behavior.csv). Those saved predictions do not match the working CSV's fraud labels. Use the [eight-case pilot](../../notebooks/eight_case_pilot/README.md) for the revised-label comparison. The main [before/after sequence](../../notebooks/README.md) uses the original 22-case source.

The historical dataset used seed `20261002` and fraud intercept `-2.8`. It had 187 fraud labels in context and 22 in test. Its saved XGBoost run found **15 of 22 fraud cases** with **9 false alarms**; the saved direct RPT run found **10 of 22** with **5 false alarms**. These labels are generated from noisy formulas; they do not represent real invoices or verified fraud.

| Columns | Meaning |
| --- | --- |
| `invoice_id` | Unique row identifier; excluded from XGBoost features and used as the RPT index |
| `invoice_amount_eur`, `payment_terms_days`, `customer_tenure_months`, `prior_late_payment_rate`, `open_invoice_count` | Numeric invoice and customer history predictors |
| `customer_segment`, `region` | Categorical predictors |
| `billing_address_mismatch`, `bank_account_changed_recently`, `weekend_submission` | Binary predictors known when the invoice is issued |
| `paid_late`, `is_fraud` | Synthetic binary targets (`1` = yes, `0` = no) |

| File | Role |
| --- | --- |
| `payment_behavior.csv` | Current working source with eight holdout fraud cases; not the source of the historical saved predictions |
| [archived payment_behavior.csv](../../archive/synthetic_invoice_experiments/published_baseline/payment_behavior.csv) | Historical source for the saved XGBoost and RPT outputs below |
| `exports/rpt_upload.csv` | Playground upload: same 1,152 rows, with both test targets replaced by `[PREDICT]` |
| `rpt16_request.json` | Equivalent masked JSON request for SAP AI Core |
| `xgboost_results.csv` | XGBoost predictions and class-1 probabilities on the final 128 rows |
| `rpt16_results.csv` | Saved direct BTP top-1 RPT predictions and selected-class confidence on the final 128 rows |
| `rpt16_run_metadata.json` | Public run time, model label, and request/result checksums; no credentials or deployment identifiers |

Raw SAP responses are kept outside the repository; repository-root `.env` is ignored by Git. The scored RPT result can be inspected without access to SAP BTP. A new RPT run requires your own deployment and may incur charges.
The metadata's `model_configured` field is the local configuration label; confirm it against the SAP deployment configuration when describing the model version publicly.
XGBoost fits a separate classifier for each target and weights the fraud class using the context class ratio; its saved class labels use a 0.5 probability threshold. RPT receives labeled context rows and returns top-1 class predictions without a matching class-weight setting. Both see the same predictors and hidden test labels, but their modeling procedures differ.

## Reproduce the historical 22-case experiment

From the repository root, install `requirements-experiment.txt`, then run:

```sh
python3 scripts/generate_synthetic_invoices.py
python3 scripts/run_xgboost_invoices.py
```

The generator overwrites the working eight-case CSV with the historical 22-case labels, then creates the masked Playground CSV and RPT JSON request. Running `scripts/run_rpt16_invoices.py` sends that request to a configured SAP BTP deployment only when its source labels match the request. To review the historical saved results without a new request, open [compare_saved_results.ipynb](../../notebooks/before_context_engineering/original_128/compare_saved_results.ipynb); it uses the archived labels and checks both models' test IDs, actual labels, and saved BTP hashes.

The [other archived eight-case dataset](../../archive/synthetic_invoice_experiments/rare_fraud_baseline/README.md) is a separate experiment. Do not combine its metrics with this run or the current context engineering benchmark.

## Optional: evaluate a historical RPT fraud alert threshold

The historical RPT request returned only the most likely class, so its saved result cannot be used to choose a different fraud alert threshold. The script defaults to the archived 22-case labels that match that request. Generate two new requests with both class scores:

```sh
python3 scripts/evaluate_rpt_fraud_threshold.py
```

The script writes `validation_request.json` and `test_request.json` to a temporary directory outside the repository (printed by the command). The validation request holds out 128 of the original context rows, including 23 fraud cases. The test request retains all 1,024 context rows and the original 128 test invoices. Both requests predict only `is_fraud`, request `top_k: 2`, and omit `paid_late`, which would not be known for a new invoice. No SAP request is made by the preparation command.

Send each JSON file to the same SAP deployment's `/predict` endpoint and save the full responses as `validation_response.json` and `test_response.json` in that directory. Then run, replacing `/tmp/tfm-rpt-threshold` with the printed directory if needed:

```sh
python3 scripts/evaluate_rpt_fraud_threshold.py \
  --validation-response /tmp/tfm-rpt-threshold/validation_response.json \
  --test-response /tmp/tfm-rpt-threshold/test_response.json
```

The scorer selects a threshold for fraud class confidence by maximizing F2 on validation rows, then reports the result on the untouched test rows alongside the default 0.5 threshold. It rejects incomplete or mismatched responses. This tests whether a different decision threshold helps; it does not retrain RPT or guarantee better results. With 23 validation fraud cases and 22 test fraud cases, treat any gain as exploratory.
