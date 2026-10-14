# Synthetic Invoice Benchmark

## SAP TechEd 2026 experiment sequence

The [notebook index](notebooks/README.md) follows one main sequence: [original benchmark](notebooks/before_context_engineering/README.md), [context engineering walkthrough](notebooks/context_engineering/00_original_22_case_context_workflow.ipynb), and [after benchmark](notebooks/after_context_engineering/README.md). All use the same [original dataset](archive/synthetic_invoice_experiments/published_baseline/payment_behavior.csv) and final 128 test invoices with **22 fraud cases**.

The original saved benchmark used all 1,024 eligible training/context rows. The new context experiment compares 128 randomly selected rows with 128 class-balanced rows on those same test invoices. This matched pair isolates the row-selection change; the original-to-new difference also includes a context-budget and SAP request-schema change. Reproduce the saved XGBoost results with `python scripts/run_original_context_comparison.py`. Add `--live` if SAP RPT conditions are missing from the saved outputs.

The [eight-case pilot](notebooks/eight_case_pilot/README.md) and its [assessment](reports/experiment_assessment.md) used revised fraud labels. Its results are kept for exploratory reference and are not used as the after scores for this original-data sequence.

The [original-data results and interpretation](reports/original_22_case_context_comparison.md) show the before benchmark and the matched random versus balanced context comparison, including fraud detections and false alarms for both models.

## Original model comparison before context engineering

The saved XGBoost and SAP RPT predictions use the [original labels](archive/synthetic_invoice_experiments/published_baseline/payment_behavior.csv). Both models predict late payment and fraud on the same 128 held-out invoice IDs.

## Experiment design

| Item | Description |
| --- | --- |
| Dataset | 1,152 generated invoices with the archived original labels |
| Split | First 1,024 rows for XGBoost training and RPT context; final 128 rows for testing |
| Targets | `paid_late` and `is_fraud`, with `1` meaning yes |
| Test positives | 51 late payments and 22 fraud cases |
| Comparison | Both models use the same test invoice IDs and true labels |

XGBoost fits a separate classifier for each target and weights the fraud class. RPT uses labeled context rows to predict masked test targets. Their modeling procedures differ, so these scores describe the outcome of this experiment rather than an equal-training comparison. The [dataset guide](data/synthetic_invoices/README.md) explains the columns, split, and optional fraud-threshold study.

## Saved results

| Target | XGBoost | SAP RPT 1.6 |
| --- | --- | --- |
| Paid late | 32 of 51 found (62.7% recall); 65.6% accuracy | 34 of 51 found (66.7% recall); 71.1% accuracy |
| Fraud | 15 of 22 found; 9 false alarms | 10 of 22 found; 5 false alarms |

RPT scored higher on late-payment accuracy and recall in this test split. XGBoost found more fraud cases; RPT raised fewer false alarms. XGBoost ROC AUC can be calculated from its saved class-1 probabilities. RPT ROC AUC is omitted because the saved top-1 BTP response lacks a positive-class score for every row. The data and labels are synthetic, and one split does not establish production performance or a general model ranking.

![Saved comparison on the shared synthetic test set](notebooks/saved_results_comparison.png)

## Check the published results

From the repository root, install the Python dependencies and validate the archived source, masked requests, saved predictions, and RPT provenance:

```sh
python3 -m pip install -r requirements-experiment.txt
python3 scripts/validate_published_experiment.py
```

Open [compare_saved_results.ipynb](notebooks/before_context_engineering/original_128/compare_saved_results.ipynb) to recompute the original metrics and chart. It checks the shared test IDs and true labels, as well as the saved RPT request/result checksums. No SAP request is made.

To regenerate the original dataset and rerun XGBoost locally, use [xgboost_run.ipynb](notebooks/before_context_engineering/original_128/xgboost_run.ipynb) for a non-overwriting notebook run, or run:

```sh
python3 scripts/generate_synthetic_invoices.py
python3 scripts/run_xgboost_invoices.py
```

These commands overwrite the working dataset and XGBoost results with the original 22-case version; run them only when you intend to replace the current 8-case source. The generator uses a fixed seed. [rpt_btp_run.ipynb](notebooks/before_context_engineering/original_128/rpt_btp_run.ipynb) reviews the saved direct BTP result with live inference off by default (`RUN_LIVE_RPT = False`). To make a new request, copy [.env.example](.env.example) to `.env`, configure your SAP AI Core deployment, and explicitly enable live inference in the notebook. A live run can incur SAP charges. The [combined walkthrough](notebooks/before_context_engineering/original_128/payment_behavior_experiment.ipynb) is optional.

## Data files

| File | Purpose |
| --- | --- |
| [archived payment_behavior.csv](archive/synthetic_invoice_experiments/published_baseline/payment_behavior.csv) | Original labels used by the before and after model comparisons |
| [exports/rpt_upload.csv](data/synthetic_invoices/exports/rpt_upload.csv) | Playground input with both targets masked as `[PREDICT]` in test rows |
| [rpt16_request.json](data/synthetic_invoices/rpt16_request.json) | Equivalent masked BTP request |
| [xgboost_results.csv](data/synthetic_invoices/xgboost_results.csv) | Saved XGBoost predictions and class-1 probabilities |
| [rpt16_results.csv](data/synthetic_invoices/rpt16_results.csv) | Saved direct BTP top-1 predictions and selected-class confidence |
| [rpt16_run_metadata.json](data/synthetic_invoices/rpt16_run_metadata.json) | Public model label and request/result checksums |

The Playground CSV is an input, **not a scored result**. A Playground summary cannot establish accuracy or recall; its row-level predictions would need to be matched to the held-out labels. Earlier [sensitivity studies](archive/synthetic_invoice_experiments/README.md) use different labels and are separate from this benchmark. Raw BTP responses and credentials are kept outside the repository.

## Repository layout

- `data/synthetic_invoices/`: working dataset and original saved model outputs; read its guide before mixing them
- `notebooks/`: original before, context walkthrough, and after groups, plus the eight-case pilot
- `scripts/`: generation, inference, validation, and analysis helpers
- `archive/`: separate earlier experiments
- `api.py`: optional local HTTP interface for invoice predictions and experiment metrics
- `.github/workflows/benchmark.yml`: validates the published data and comparison notebook on pushes and pull requests

To use the optional API, run `python3 -m uvicorn api:app --host 127.0.0.1 --port 8000` after installing the Python requirements, then open <http://127.0.0.1:8000/docs>. Live RPT inference through the API requires your own SAP deployment and credentials.
