# After context engineering

The [after benchmark](01_balanced_context_benchmark.ipynb) compares XGBoost and SAP RPT after validation selected **class-balanced context**. It uses the same frozen source, targets, 128-row budget, five seeds, and final 128-invoice holdout as the [before benchmark](../before_context_engineering/01_random_context_baseline.ipynb). The final holdout has eight fraud cases. SAP predictions come from saved runs; XGBoost is recomputed locally on matching selected rows. No live SAP call is required.

| Read order | Notebook | Purpose |
| --- | --- | --- |
| 1 | [Balanced-context benchmark](01_balanced_context_benchmark.ipynb) | Compare both models after validation selected class-balanced examples. |
| 2 | [Combined before/after comparison](../context_engineering/07_before_after_context_engineering.ipynb) | See the two stages together, including the paired prediction tradeoff. |

The [context engineering notebooks](../context_engineering/README.md) explain how context size, selection, quality, and semantic representation were tested. The [older 22-case experiment](../historical_22_case_experiment/README.md) used different fraud labels and is separate from this before/after comparison.
