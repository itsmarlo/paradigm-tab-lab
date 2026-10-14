# After context engineering: original 22-case data

The [after benchmark](01_original_22_case_after_benchmark.ipynb) puts the original 1,024-row saved results beside a 128-row random control and 128-row class-balanced context condition. All conditions use the original labels and final 128 test invoices with **22 fraud cases**. The matched 128-row pair isolates the example-selection change. XGBoost results are reproducible locally; SAP RPT conditions require live requests to the configured deployment.

For SAP RPT, balanced examples improve fraud recall from **45.5% to 72.7%** and fraud F1 from **0.571 to 0.640** versus random examples at the same context budget. False alarms rise from **3 to 12**. SAP RPT has higher fraud F1 than XGBoost after balancing; its late-payment F1 does not improve.

The [context engineering walkthrough](../context_engineering/00_original_22_case_context_workflow.ipynb) shows the data checks, selection, request masking, and result checks. The [eight-case pilot](../eight_case_pilot/README.md) is separate and supplies no scores here.
