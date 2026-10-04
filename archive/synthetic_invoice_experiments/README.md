# Archived synthetic invoice experiments

The [active 22-case dataset](../../data/synthetic_invoices/README.md) is the only dataset under `data/`. The earlier eight-case studies remain here for audit history.

| Study | Context fraud labels | Test fraud labels | Purpose |
| --- | ---: | ---: | --- |
| [Rare-fraud baseline](rare_fraud_baseline/README.md) | 53 | 8 | Original XGBoost, BTP, and Playground comparison |
| [Context-only change](context_only/README.md) | 187 | 8 | Holds the original test labels fixed to study added context fraud labels |

Do not combine these saved predictions with the active 22-case test results. Raw SAP responses are kept outside the repository; scored results and reduced provenance are safe to inspect without credentials.
