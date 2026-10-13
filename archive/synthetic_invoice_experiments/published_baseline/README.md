# Historical published baseline source

`payment_behavior.csv` preserves the original 1,152-row synthetic dataset used by the saved XGBoost and direct SAP RPT predictions in `data/synthetic_invoices/`. It has 187 fraud cases in the first 1,024 rows and 22 in the final 128. Its SHA-256 is `dde242d3776c1e5024fd7e3d4cc26be90f6328fbe94c5ea0125d98eeed835b92`.

The working `data/synthetic_invoices/payment_behavior.csv` now has revised fraud labels: 53 in the first 1,024 rows and eight in the final 128. It matches `data/semantic_ablation/source_snapshot.csv` and belongs to the current context engineering benchmark. Historical scores must use this archived source; current scores must use the frozen snapshot.

Run `python scripts/validate_published_experiment.py` from the repository root to verify the historical source, request, outputs, and metadata together.
