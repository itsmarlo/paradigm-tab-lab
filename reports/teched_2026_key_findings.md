# SAP TechEd 2026: key findings

- **Research question:** Can context composition improve SAP RPT predictions without retraining?
- **Baseline:** At 128 random examples, mean SAP RPT F1 was 0.544 for late payment and 0.000 for fraud.
- **Intervention:** Validation selected class-balanced context for both tasks. The fraud context used all 53 eligible positives in the final 1,024-row pool, versus 3–8 positives under random selection.
- **Observed result:** Final-holdout F1 was 0.558 for late payment and 0.388 for fraud; fraud recall was 0.750 with 17.0 false positives per 128 invoices.
- **Interpretation:** Context composition changed the sensitivity/false-alarm tradeoff. Semantic enrichment gave S3−S0 late-payment F1 -0.022; cleaning was an identity transformation.
- **Limitation:** Synthetic data, eight holdout fraud positives, three semantic seeds, and one seed for query-specific selection.
- **Slide takeaway:** Better context can recover missed fraud cases in this setup, but the added alerts matter; descriptive feature names did not improve measured performance.
