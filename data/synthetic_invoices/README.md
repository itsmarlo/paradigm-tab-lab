# Active synthetic invoice dataset

`payment_behavior.csv` is the active labeled dataset for the XGBoost and SAP RPT 1.6 comparison. It has 1,152 synthetic invoices: the first 1,024 are XGBoost training rows and RPT context rows; the final 128 are the shared test rows. `invoice_id` is a row key, not a predictor. Both models predict `paid_late` and `is_fraud`, where `1` means yes.

The dataset uses seed `20261002` and fraud intercept `-2.8`. It contains 187 fraud labels in context and 22 in test. Late-payment labels are unchanged from the earlier rare-fraud scenario: 450 in context and 51 in test. These labels are generated from noisy formulas; they do not represent real invoices or verified fraud.

The saved direct RPT run found **10 of 22 fraud cases** with **5 false alarms**. This is the strongest saved fraud-detection result for RPT in this repository, but the archived studies use different context or test labels, so their scores are not a controlled model-improvement comparison. The threshold experiment below is the next test for improving recall on this fixed dataset.

| Columns | Meaning |
| --- | --- |
| `invoice_id` | Unique row identifier; excluded from XGBoost features and used as the RPT index |
| `invoice_amount_eur`, `payment_terms_days`, `customer_tenure_months`, `prior_late_payment_rate`, `open_invoice_count` | Numeric invoice and customer history predictors |
| `customer_segment`, `region` | Categorical predictors |
| `billing_address_mismatch`, `bank_account_changed_recently`, `weekend_submission` | Binary predictors known when the invoice is issued |
| `paid_late`, `is_fraud` | Synthetic binary targets (`1` = yes, `0` = no) |

| File | Role |
| --- | --- |
| `payment_behavior.csv` | One labeled source dataset for both models |
| `exports/rpt_upload.csv` | Playground upload: same 1,152 rows, with both test targets replaced by `[PREDICT]` |
| `rpt16_request.json` | Equivalent masked JSON request for SAP AI Core |
| `xgboost_results.csv` | XGBoost predictions and class-1 probabilities on the final 128 rows |
| `rpt16_results.csv` | Saved direct BTP top-1 RPT predictions and selected-class confidence on the final 128 rows |
| `rpt16_run_metadata.json` | Public run time, model label, and request/result checksums; no credentials or deployment identifiers |

Raw SAP responses are kept outside the repository; repository-root `.env` is ignored by Git. The scored RPT result can be inspected without access to SAP BTP. A new RPT run requires your own deployment and may incur charges.
The metadata's `model_configured` field is the local configuration label; confirm it against the SAP deployment configuration when describing the model version publicly.
XGBoost fits a separate classifier for each target and weights the fraud class using the context class ratio; its saved class labels use a 0.5 probability threshold. RPT receives labeled context rows and returns top-1 class predictions without a matching class-weight setting. Both see the same predictors and hidden test labels, but their modeling procedures differ.

## Reproduce the active experiment

From the repository root, install `requirements-experiment.txt`, then run:

```sh
python3 scripts/generate_synthetic_invoices.py
python3 scripts/run_xgboost_invoices.py
```

The generator creates the labeled CSV, masked Playground CSV, and masked RPT JSON request. Running `scripts/run_rpt16_invoices.py` sends that request to a configured SAP BTP deployment. To review the saved results without making a request, open the three notebooks in the order documented in the root README. The comparison notebook checks that both models' test IDs and actual labels match this dataset and that the saved BTP request/result hashes match its metadata.

The [archived eight-case dataset](../../archive/synthetic_invoice_experiments/rare_fraud_baseline/README.md) is a separate experiment with different test labels. Do not combine its metrics with this run.

## Evaluate a fraud alert threshold

The original RPT request returned only the most likely class, so its saved result cannot be used to choose a different fraud alert threshold. Generate two new requests with both class scores:

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
