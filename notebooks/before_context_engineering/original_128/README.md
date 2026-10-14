# Original benchmark before context engineering

These are the original XGBoost versus SAP RPT notebooks behind the presentation slides. They use the [original dataset](../../../archive/synthetic_invoice_experiments/published_baseline/payment_behavior.csv): 1,024 labeled invoices and the same final 128 test invoices, including **22 fraud cases**. Saved SAP requests and predictions are under `data/synthetic_invoices/`.

| Read order | Notebook | Purpose |
| --- | --- | --- |
| 1 | [XGBoost run](xgboost_run.ipynb) | Recompute XGBoost predictions from the original labels. |
| 2 | [SAP RPT run](rpt_btp_run.ipynb) | Check the saved masked request and direct BTP predictions. |
| 3 | [Saved results comparison](compare_saved_results.ipynb) | Compare both models on the identical 128-invoice test set. |
| Optional | [Combined walkthrough](payment_behavior_experiment.ipynb) | Review the data, XGBoost training, and SAP results together. |

Continue with the [context engineering workflow](../../context_engineering/00_original_22_case_context_workflow.ipynb) and [after benchmark](../../after_context_engineering/01_original_22_case_after_benchmark.ipynb).
