# Original 22-case invoice experiment

The main [notebook sequence](../notebooks/README.md) uses one original synthetic dataset throughout. Its first 1,024 invoices supply labeled training/context examples; the final 128 are held out, with 51 late payments and 22 fraud cases. Both models predict the same test IDs and labels. The saved first benchmark used all 1,024 eligible rows. The new comparison uses seed 11 and a fixed 128-row budget to compare random selection with a class-balanced selection.

## Fraud predictions on the same 128 invoices

| Condition | XGBoost found / 22 | XGBoost false alarms | XGBoost F1 | SAP RPT found / 22 | SAP RPT false alarms | SAP RPT F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| First benchmark: 1,024 rows | 15 | 9 | 0.652 | 10 | 5 | 0.541 |
| Random control: 128 rows | 10 | 20 | 0.385 | 10 | 3 | 0.571 |
| Balanced context: 128 rows | 16 | 32 | 0.457 | 16 | 12 | 0.640 |

On the original first benchmark, XGBoost wins fraud recall and F1, while SAP RPT gives fewer false alarms. On the new balanced context, both detect 16 fraud cases; SAP RPT has higher fraud F1 and fewer false alarms. Balancing raises recall for both models relative to the matched 128-row random control, but also increases false alarms. Late-payment F1 in the balanced condition is 0.571 for XGBoost and 0.505 for SAP RPT; the first benchmark was 0.593 and 0.648, respectively.

## SAP RPT after enhancing the in-context examples

At the same 128-row context budget, balancing the labeled examples improves SAP RPT's **fraud recall from 45.5% to 72.7%** and **fraud F1 from 0.571 to 0.640**. It detects six additional fraud cases (10 to 16 of 22). Precision falls from 76.9% to 57.1%, because false alarms increase from 3 to 12. The [after notebook](../notebooks/03_after_context_engineering.ipynb) shows this matched comparison in a chart and a table. On late payment, SAP RPT F1 changes from 0.511 to 0.505, so the improvement claim applies to fraud detection only.

## What context engineering changed

The [walkthrough](../notebooks/02_in_context_changes.ipynb) checks the source, keeps the test set separate, selects labeled examples, verifies class composition and unique IDs, masks test targets in the SAP request, and excludes the other target from each request. SAP RPT weights stay fixed. XGBoost is refit on the corresponding selected examples. The balanced fraud context has 64 positive and 64 negative examples; the random 128-row control has 27 positive and 101 negative examples. No source fraud labels were edited for this comparison.

The change from the first benchmark to either new condition includes a **1,024-to-128 context-budget change**. The saved original SAP request also carried both target columns, while each new request includes only its prediction target. Compare the two 128-row rows to assess selection with the budget and request schema held fixed. XGBoost keeps the original model seed, 20261002; the context selection seed is 11. The earlier [eight-case pilot](experiment_assessment.md) suggested examining class balance, but its changed fraud labels prevent its saved scores from measuring this experiment. The findings are descriptive: the invoices are synthetic, the holdout is one fixed set, and this new comparison uses one selection seed.

Reproduce the metrics with `python scripts/validate_published_experiment.py`, `python scripts/run_original_context_comparison.py`, and the [after notebook](../notebooks/03_after_context_engineering.ipynb). Existing SAP predictions are reused by the runner; `--live` is needed only for missing SAP conditions.
