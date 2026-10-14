# Before context engineering: original benchmark

Start with the [original XGBoost and SAP RPT notebooks](original_128/README.md). Their saved predictions use the original 1,152-row source: 1,024 labeled rows for training/context and the final 128 for testing, including **22 fraud cases**. XGBoost found 15 fraud cases with nine false alarms; SAP RPT found 10 with five false alarms. Both models predicted the same invoice IDs.

Next, read the [context engineering walkthrough](../context_engineering/00_original_22_case_context_workflow.ipynb) and [after benchmark](../after_context_engineering/01_original_22_case_after_benchmark.ipynb). They keep the same source and test invoices. A matched random 128-row condition accompanies the balanced 128-row condition so the effect of selection can be separated from the change in context budget.
