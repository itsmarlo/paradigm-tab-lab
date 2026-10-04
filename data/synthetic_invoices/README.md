# TechEd synthetic invoice dataset

`payment_behavior.csv` is the **main labeled dataset** for the XGBoost and SAP RPT 1.6 comparison. It has 1,152 synthetic invoices: the first 1,024 are XGBoost training rows and RPT context rows; the final 128 are the shared test rows. `invoice_id` is a row key, not a predictor. Both models predict `paid_late` and `is_fraud`, where `1` means yes.

The dataset uses seed `20261002` and fraud intercept `-2.8`. It contains 187 fraud labels in context and 22 in test. Late-payment labels are unchanged from the earlier rare-fraud scenario: 450 in context and 51 in test. These labels are generated from noisy formulas; they do not represent real invoices or verified fraud.

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

The raw `.local/rpt16_response.json` and repository-root `.env` are local-only and ignored by Git. The scored RPT result can be inspected without access to SAP BTP. A new RPT run requires your own deployment and may incur charges.
The metadata's `model_configured` field is the local configuration label; confirm it against the SAP deployment configuration when describing the model version publicly.
XGBoost fits a separate classifier for each target and weights the fraud class using the context class ratio; its saved class labels use a 0.5 probability threshold. RPT receives labeled context rows and returns top-1 class predictions without a matching class-weight setting. Both see the same predictors and hidden test labels, but their modeling procedures differ.

## Reproduce the main experiment

From the repository root, install `requirements-experiment.txt`, then run:

```sh
python3 scripts/generate_synthetic_invoices.py
python3 scripts/run_xgboost_invoices.py
```

The generator creates the labeled CSV, masked Playground CSV, and masked RPT JSON request. Running `scripts/run_rpt16_invoices.py` sends that request to a configured SAP BTP deployment. To review the saved results without making a request, open the three notebooks in the order documented in the root README. The comparison notebook checks that both models' test IDs and actual labels match this dataset and that the saved BTP request/result hashes match its metadata.

The [archive](../../archive/synthetic_invoice_experiments/README.md) holds secondary studies: `rare_fraud_baseline/` is the original 53-context/8-test scenario, and `context_only/` changes only context fraud labels while retaining the baseline's eight test fraud cases. Their results must not be mixed with the main dataset's 22-case test.
