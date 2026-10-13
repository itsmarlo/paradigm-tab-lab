# Archived synthetic invoice experiments

These archives preserve label versions used by earlier saved predictions. The [current context engineering study](../../reports/context_engineering_benchmark.md) uses a separate frozen eight-case holdout. The working CSV and historical saved outputs under `data/synthetic_invoices/` do not share the same fraud labels.

| Study | Context fraud labels | Test fraud labels | Purpose |
| --- | ---: | ---: | --- |
| [Published baseline source](published_baseline/README.md) | 187 | 22 | Original labels for saved XGBoost and direct BTP predictions under `data/synthetic_invoices/` |
| [Rare-fraud baseline](rare_fraud_baseline/README.md) | 53 | 8 | Original XGBoost, BTP, and Playground comparison |
| [Context-only change](context_only/README.md) | 187 | 8 | Holds the original test labels fixed to study added context fraud labels |

Do not combine scores across these studies. Raw SAP responses are kept outside the repository; scored results and reduced provenance are safe to inspect without credentials.
