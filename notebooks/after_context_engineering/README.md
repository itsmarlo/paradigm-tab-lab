# After context engineering: SAP RPT on the original 22-case data

The [after benchmark](01_original_22_case_after_benchmark.ipynb) focuses on SAP RPT. It puts the original 1,024-row saved result beside a 128-row random control and 128-row class-balanced context condition. All conditions use the original labels and final 128 test invoices with **22 fraud cases**. The matched 128-row pair isolates the example-selection change; SAP RPT's weights stay fixed. The notebook reuses saved predictions. Run `python scripts/run_original_context_comparison.py --live` only if a SAP RPT condition is missing.

Balanced examples improve SAP RPT fraud recall from **45.5% to 72.7%** and fraud F1 from **0.571 to 0.640** versus random examples at the same context budget. False alarms rise from **3 to 12**. Late-payment F1 does not improve.

The notebook also shows the **unchanged XGBoost result** as a reference: it found 15 of 22 fraud cases with 9 false alarms and F1 0.652, versus 16, 12, and 0.640 for balanced SAP RPT. XGBoost trained on 1,024 rows; SAP RPT used 128 context rows. This is a descriptive comparison on the same test invoices, not an equal-budget model ranking. The matched SAP RPT random-versus-balanced comparison measures the context-selection effect.

The [comparison graph](../../reports/figures/xgboost_vs_context_enhanced_rpt.png) shows both targets' F1 scores and the fraud detection and false-alarm counts. The after notebook regenerates this PNG from the saved predictions.

The [context engineering walkthrough](../context_engineering/00_original_22_case_context_workflow.ipynb) shows the data checks, selection, request masking, and result checks. The [original before notebooks](../before_context_engineering/original_128/README.md) retain the XGBoost and SAP RPT baseline comparison. The [eight-case pilot](../eight_case_pilot/README.md) is separate and supplies no scores here.
