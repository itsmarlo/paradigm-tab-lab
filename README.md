# Paradigm Lab

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/itsmarlo/paradigm-tab-lab?quickstart=1)

Paradigm Lab is a reproducible **benchmarking project** with an interactive guide to three approaches to prediction: classical machine learning, large language models, and tabular foundation models. The measured experiment compares **XGBoost and SAP RPT 1.6** on the same synthetic invoice test set. The React website illustrates how the approaches work; its simulations are separate from the measured benchmark.

## Benchmark at a glance

| | Setup |
| --- | --- |
| Data | 1,152 synthetic invoices: 1,024 labeled context/training rows and 128 held-out test rows |
| Targets | `paid_late` and `is_fraud` (`1` = yes) |
| Models | XGBoost trained locally; SAP RPT 1.6 queried through SAP BTP with labeled context rows |
| Evaluation | Same test invoice IDs and true labels for both models; accuracy and recall for late payment, detected cases and false alarms for fraud |

The test set contains **51 late payments and 22 fraud cases**. All labels are synthetic; these results describe this fixed experiment, not real-world fraud detection.

### Results on the 128 test invoices

| Target | XGBoost | SAP RPT 1.6 |
| --- | --- | --- |
| Paid late | 32 of 51 found (62.7% recall); 65.6% accuracy | 34 of 51 found (66.7% recall); 71.1% accuracy |
| Fraud | 15 of 22 found; 9 false alarms | 10 of 22 found; 5 false alarms |

RPT did better on late-payment accuracy and recall in this split. XGBoost detected more fraud cases, while RPT raised fewer false alarms. The models use different procedures: XGBoost fits a classifier for each target and weights the fraud class; RPT predicts from labeled context rows without a matching class-weight setting. One synthetic split is too small to establish a general ranking or production performance. See the [comparison notebook](notebooks/compare_saved_results.ipynb) for the calculations and the [dataset guide](data/synthetic_invoices/README.md) for the split and feature definitions.

![Comparison of XGBoost and SAP RPT 1.6 on the shared synthetic test set](notebooks/saved_results_comparison.png)

## Reproduce and inspect

You can validate the published data and inspect both saved result sets **without SAP credentials**:

```sh
python3 -m pip install -r requirements-experiment.txt
python3 scripts/validate_published_experiment.py
```

Then open [compare_saved_results.ipynb](notebooks/compare_saved_results.ipynb). It checks shared test IDs, labels, and saved RPT request/result checksums, then recomputes the metrics and comparison chart. XGBoost ROC AUC is available from its saved class-1 probabilities; RPT ROC AUC is omitted because the saved top-1 BTP response does not provide a positive-class score for every row.

To rerun the local part of the experiment, use [xgboost_run.ipynb](notebooks/xgboost_run.ipynb) or run the scripts in order:

```sh
python3 scripts/generate_synthetic_invoices.py
python3 scripts/run_xgboost_invoices.py
```

The generator uses a fixed seed and writes the labeled dataset and masked inputs for RPT. These commands overwrite the corresponding saved dataset and XGBoost result files in `data/`.

To review the saved RPT run or make a **new live** request, open [rpt_btp_run.ipynb](notebooks/rpt_btp_run.ipynb). Live inference is off by default (`RUN_LIVE_RPT = False`). For a new run, copy [.env.example](.env.example) to `.env`, provide your SAP AI Core settings, and explicitly enable live inference in the notebook. A live request can incur SAP charges. The [combined walkthrough](notebooks/payment_behavior_experiment.ipynb) is optional. The notebooks use a Python 3 kernel and include executed outputs.

### Data and saved outputs

| File | Purpose |
| --- | --- |
| [payment_behavior.csv](data/synthetic_invoices/payment_behavior.csv) | Labeled source for both models |
| [exports/rpt_upload.csv](data/synthetic_invoices/exports/rpt_upload.csv) | Playground upload with both targets masked as `[PREDICT]` in the final 128 rows |
| [rpt16_request.json](data/synthetic_invoices/rpt16_request.json) | Equivalent masked BTP request |
| [xgboost_results.csv](data/synthetic_invoices/xgboost_results.csv) | XGBoost test predictions and class-1 probabilities |
| [rpt16_results.csv](data/synthetic_invoices/rpt16_results.csv) | Saved direct BTP test predictions and selected-class confidence |
| [rpt16_run_metadata.json](data/synthetic_invoices/rpt16_run_metadata.json) | Public model label and request/result checksums |

The SAP RPT Playground upload is an **input file, not a scored result**. Its AI summary cannot establish accuracy or recall; a row-level prediction export must be matched with the held-out labels first. Archived [sensitivity studies](archive/synthetic_invoice_experiments/README.md) use different labels and are not part of this benchmark. Raw BTP responses and local credentials are kept outside the repository.

## Interactive guide

The website is a browser-based presentation built with React, TypeScript, and Vite. It needs no backend or API key. **Auto Demo** walks through the three approaches and their comparison; you can also edit example inputs in each section. The classical, LLM, and tabular FM behaviors are educational simulations, not runs of the benchmark models.

### Run locally

Requires Node.js 22.6+ and npm:

```sh
npm ci
npm run dev
```

Open <http://localhost:5173>. Run `npm test` for the website's model behavior checks or `npm run build` for a production build.

### Open in GitHub Codespaces

Use the badge above or select **Code → Codespaces → Create codespace on main** in [the repository](https://github.com/itsmarlo/paradigm-tab-lab). Dependencies install automatically and Vite starts on port **5173**. Open **Paradigm Lab — interactive preview** from the **Ports** panel if it does not open automatically. Keep the forwarded port **Private** for your own presentation. If the server is not running, enter `npm run dev` in the terminal. Existing Codespaces may need **Codespaces: Rebuild Container** after pulling changes.

### GitHub Pages

The [publish workflow](.github/workflows/publish.yml) validates the dataset, runs the saved comparison notebook, tests and builds the website, and deploys it when GitHub Pages is enabled with **GitHub Actions** as its source. After a successful deployment, the site is at <https://itsmarlo.github.io/paradigm-tab-lab/>. Check **Actions → Publish Paradigm Lab** for status.

## Scope and repository layout

The website's classical algorithms use simplified scoring; its LLM tokens and attention are illustrative; and its tabular predictions use similarity-weighted context labels, not pretrained transformer weights. The benchmark measures only XGBoost and the saved direct BTP RPT run. The website does not send invoice data to a server or model.

- `data/synthetic_invoices/`: active dataset, masked RPT inputs, and saved predictions
- `notebooks/`: experiment runs and comparison
- `scripts/`: data generation, model runs, and published-data validation
- `src/`: interactive guide
- `archive/`: separate earlier experiments

For a local HTTP interface, install the Python requirements and run `python3 -m uvicorn api:app --host 127.0.0.1 --port 8000`. Its `/docs` page documents the API. A live RPT request through the API requires your own SAP deployment and credentials; the website does not use this API.
