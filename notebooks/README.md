# Three-notebook experiment

Read these notebooks in order. All three use the archived original dataset and the same final 128 invoices, including 22 fraud cases. They read saved predictions and make no live SAP request.

| Step | Notebook | What it answers |
| --- | --- | --- |
| 1 | [Before context engineering](01_before_context_engineering.ipynb) | How did the original XGBoost and SAP RPT runs perform? Includes XGBoost ROC AUC and average precision. |
| 2 | [In-context changes](02_in_context_changes.ipynb) | Which labeled examples changed, and how were held-out labels masked? |
| 3 | [After context engineering](03_after_context_engineering.ipynb) | What changed in SAP RPT predictions at a matched 128-row budget, and how does the result compare descriptively with XGBoost? Includes fraud ROC AUC where class scores exist. |

The matched random-versus-balanced SAP RPT pair isolates example selection. The original SAP RPT run used 1,024 rows and a different request schema, so its comparison with the 128-row conditions is historical context. The separate [eight-case pilot report](../reports/experiment_assessment.md) and [saved pilot data](../data/context_benchmark/) remain available without adding another notebook to this sequence.
