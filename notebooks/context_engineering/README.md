# Context engineering notebooks

Start with [the original 22-case walkthrough](00_original_22_case_context_workflow.ipynb). It is the main context-engineering notebook for the before/after sequence and uses the same original dataset as the presentation slides.

## Eight-case exploratory pilot

Use the frozen `data/semantic_ablation/source_snapshot.csv` with these notebooks. Its final 128 invoices contain eight fraud cases. The following older method notebooks use a revised eight-case label version and are retained as an exploratory pilot. Their saved scores do not measure the after stage of the original 22-case experiment.

| Read order | Notebook | Question |
| --- | --- | --- |
| 1 | [Fundamentals](01_context_engineering_fundamentals.ipynb) | What is the data split and how is context used? |
| 2 | [Context size](02_context_size_benchmark.ipynb) | How does the 16–1,024 example budget change scores? |
| 3 | [Selection strategies](03_context_selection_strategies.ipynb) | Which 128-row strategy wins validation? |
| 4 | [Quality and robustness](05_context_quality_robustness.ipynb) | What happens with label noise, duplicates, or regional restriction? |
| 5 | [Semantic ablation](06_semantic_enrichment_ablation.ipynb) | Do clearer feature names, category text, or cleaning help? |
| 6 | [Final selection benchmark](06_final_context_optimization_benchmark.ipynb) | What transfers to the untouched holdout? |
| 7 | [XGBoost versus SAP RPT](07_before_after_context_engineering.ipynb) | How do both models' predictions compare before and after balanced selection? |

`04_semantic_context_engineering.ipynb` contains the core semantic cells also present in the fuller `06_semantic_enrichment_ablation.ipynb`. `05_research_summary.ipynb` is a concise synthesis. Both are retained for existing links; the table above is the canonical reading path until the notebook names can be consolidated. Run `scripts/build_context_notebooks.py` only when you intend to rebuild its generated notebooks; doing so replaces their saved outputs until they are executed again. The before/after notebook has its own builder, `scripts/build_before_after_context_notebook.py`.
