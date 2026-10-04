# Archived synthetic invoice experiments

These two studies document how the original rare-fraud setup and a context-only label change affected the models. They are kept for audit and presentation history. The [main TechEd dataset](../../data/synthetic_invoices/README.md) and the three primary notebooks use the 187-context/22-test fraud scenario.

| Study | Context fraud labels | Test fraud labels | Purpose |
| --- | ---: | ---: | --- |
| [Rare-fraud baseline](rare_fraud_baseline/README.md) | 53 | 8 | Original XGBoost, BTP, and Playground comparison |
| [Context-only change](context_only/README.md) | 187 | 8 | Holds the original test labels fixed to study added context fraud labels |

Do not combine their saved predictions with the main 22-case test results. Raw SAP responses remain in ignored `.local/` folders; the archived scored results and reduced provenance are safe to inspect without credentials.
