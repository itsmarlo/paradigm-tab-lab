# Historical 22-case experiment

This is the older saved XGBoost versus SAP RPT experiment. It trained or supplied context from **1,024 synthetic invoices** and scored the same **128 held-out invoices**. The source is the [archived original dataset](../../archive/synthetic_invoice_experiments/published_baseline/payment_behavior.csv), with **22 fraud cases in the holdout**. The saved SAP request and predictions are under `data/synthetic_invoices/` and match that archived source.

| Read order | Notebook | Purpose |
| --- | --- | --- |
| 1 | [XGBoost run](xgboost_run.ipynb) | Recompute XGBoost predictions from the archived labels. |
| 2 | [SAP RPT run](rpt_btp_run.ipynb) | Check the saved masked request and direct BTP predictions; live inference is off by default. |
| 3 | [Saved results comparison](compare_saved_results.ipynb) | Compare both saved models on the identical 128-invoice test set. |
| Optional | [Combined walkthrough](payment_behavior_experiment.ipynb) | Follow data checks, XGBoost training, and saved SAP review in one notebook. |

This experiment is separate from the [before](../before_context_engineering/README.md) and [after](../after_context_engineering/README.md) context engineering benchmarks. Those use revised labels with **eight fraud cases** in the same final 128. Differences between this historical run and those benchmarks cannot be attributed to context engineering because the fraud labels also changed.
