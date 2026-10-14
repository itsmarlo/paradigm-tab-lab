# Eight-case exploratory pilot: context engineering and model comparison

This report analyzes the revised-label eight-case pilot. The main presentation sequence uses the [original 22-case benchmark](../notebooks/before_context_engineering/README.md), [original-data context walkthrough](../notebooks/context_engineering/00_original_22_case_context_workflow.ipynb), and [after benchmark](../notebooks/after_context_engineering/01_original_22_case_after_benchmark.ipynb). Pilot scores below are not after scores for that sequence.

## What context engineering means here

SAP RPT receives labeled invoice examples in each prediction request. The experiment changes **which examples are supplied**, while keeping the SAP model weights fixed. XGBoost trains new weights on those same selected examples for the direct model comparison. The four investigated choices are:

1. **Size:** 16–1,024 labeled examples with random selection.
2. **Selection:** random, stratified, class-balanced, diversity, and exploratory query-specific methods at 128 examples.
3. **Quality:** flipped labels, duplicate examples, and a regional restriction.
4. **Semantics:** names and category text, tested separately with fixed 512-row contexts.

The first 896 rows supply strategy-selection examples, the next 128 form validation, and the last 128 form the final holdout. The validation-selected strategy is class-balanced for both late payment and fraud. For the final runs, the eligible training/context pool contains the first 1,024 rows. The balanced fraud context contains all 53 positive examples plus 75 negatives; random context contains 3–8 positives across five seeds. This changes the effective positive-class fraction from roughly 4% to 41% at the same 128-row budget.

## Current matched model comparison

The [before/after notebook](../notebooks/context_engineering/07_before_after_context_engineering.ipynb) evaluates both models on the same 128 held-out invoices and the same selected example IDs for each target and seed. XGBoost's matched before/after results are reproduced locally; SAP RPT's are saved run results. Values are means over five context seeds.

| Target | Model | Random F1 | Balanced F1 | Balanced recall | Balanced false positives / 128 |
| --- | --- | ---: | ---: | ---: | ---: |
| Late payment | XGBoost | 0.565 | 0.550 | 0.627 | 33.4 |
| Late payment | SAP RPT | 0.544 | 0.558 | 0.616 | 30.0 |
| Fraud | XGBoost | 0.027 | 0.325 | 0.725 | 22.2 |
| Fraud | SAP RPT | 0.000 | 0.388 | 0.750 | 17.0 |

SAP RPT has the higher mean F1 after balancing for both targets, but the late-payment difference is 0.008 and the fraud holdout contains only eight positives. Treat the ranking as descriptive. For fraud, the balanced SAP condition finds six of eight cases on average and produces 17 false positives among 120 nonfraud invoices. A user-facing alert policy would need to judge whether that cost is acceptable.

## What the experiment supports

- **Useful finding:** Selection of labeled examples materially changes fraud detection for both models. The increase is closely tied to exposing far more positive examples, not evidence that SAP RPT acquired new weights or that feature names became more meaningful.
- **Good design controls:** The final holdout is separate from the validation slice; IDs and labels are checked against the frozen source; row-level predictions and protocol hashes are saved. The matched model notebook uses the same selected rows for both models.
- **Negative result:** More descriptive names and category text did not improve the measured SAP result at a 512-row budget. Validation favored S3 for late payment, but holdout F1 fell from 0.592 (S0) to 0.570 (S3). Cleaning made no change because the frozen source has no recorded missing or invalid values.
- **Limited inference:** The data and labels are synthetic. Validation contains nine fraud cases and the final holdout eight. Reusing the same holdout across seeds does not create five independent test sets. Query-specific methods have one seed, and the semantic ablation has three. The configured SAP model label is not independent proof of the deployed version.

## Repository and notes hygiene

The historical 22-fraud-case outputs under `data/synthetic_invoices/` use the [archived original source](../archive/synthetic_invoice_experiments/published_baseline/payment_behavior.csv). The working CSV and current frozen snapshot have eight holdout fraud cases. Their metrics must remain separate. The root README and data guide now state this, and the legacy validator checks the archived source.

The notebook directory still has overlapping semantic notebooks (`04_semantic_context_engineering.ipynb` and `06_semantic_enrichment_ablation.ipynb`) and duplicate sequence numbers (`05_*` and `06_*`). Their generated content is retained for now because they may be referenced by existing notes. A later documentation pass can choose one canonical semantic notebook and renumber the series after checking links.

Next research step: evaluate on a larger external dataset with more fraud cases and a prespecified alert-cost target. Repeat strategy selection without using final-holdout outcomes, then report uncertainty across independent datasets or folds. Class-probability outputs would allow threshold and calibration comparisons where supported.
