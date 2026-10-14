# Notebook reading order

1. [Before context engineering](before_context_engineering/README.md): the XGBoost and SAP RPT comparison on 128 held-out invoices with 22 fraud cases.
2. [Context engineering](context_engineering/00_original_22_case_context_workflow.ipynb): source checks, context selection, label masking, and matched model setup on that same original data.
3. [After context engineering](after_context_engineering/README.md): SAP RPT with random versus balanced context on the same 22-case holdout.

The [eight-case pilot](eight_case_pilot/README.md) used revised fraud labels. Its saved results remain available for exploratory analysis, but do not supply the after scores in this sequence.
