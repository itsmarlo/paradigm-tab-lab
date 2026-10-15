# Original-data context comparison outputs

`protocol.json` fixes the original 22-case source, final 128-row holdout, 128-row context budget, and seeds. `results.csv` holds summary scores; `predictions.csv` holds one row per held-out invoice, target, model, and context condition. Both random and balanced contexts use the same original data. The original 1,024-row saved predictions remain in `data/synthetic_invoices/`.

Run `python scripts/run_original_context_comparison.py` to reproduce missing XGBoost conditions locally. Add `--live` only when a SAP RPT condition is missing and a deployment is configured. The runner reuses completed conditions and checks the source hash. The [after notebook](../../notebooks/03_after_context_engineering.ipynb) verifies prediction IDs and labels against the source before plotting.
