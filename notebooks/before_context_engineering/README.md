# Before context engineering

The [baseline notebook](01_random_context_baseline.ipynb) benchmarks XGBoost and SAP RPT using **128 randomly selected labeled examples** from `data/semantic_ablation/source_snapshot.csv`. Both models predict the same final 128 invoices, including eight fraud cases. Five seeds vary the selected examples. SAP results are saved predictions; XGBoost is recomputed locally on the matching selected rows.

The [after benchmark](../after_context_engineering/01_balanced_context_benchmark.ipynb) uses the **same source, targets, holdout, context budget, seeds, and model procedures**. It changes the selection strategy to class-balanced examples chosen on validation. The [context engineering notebooks](../context_engineering/README.md) show how that strategy was evaluated.

The [older 22-case experiment](../historical_22_case_experiment/README.md) uses different fraud labels and is kept only as historical evidence.
