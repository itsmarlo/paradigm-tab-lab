# Paradigm Lab

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/itsmarlo/paradigm-tab-lab?quickstart=1)

Paradigm Lab combines a reproducible **synthetic invoice benchmark** with an interactive guide to classical machine learning, large language models, and tabular foundation models. The benchmark compares XGBoost and SAP RPT 1.6 on one fixed test split; the website explains the three learning approaches with educational simulations. It uses React, TypeScript, and Vite without a backend or API keys.

## Start here

| Goal | Start with |
| --- | --- |
| Explore the invoice benchmark | Open the [comparison notebook](notebooks/compare_saved_results.ipynb) or read [the results](#read-the-results) |
| Explore the interactive presentation | [Open a GitHub Codespace](https://codespaces.new/itsmarlo/paradigm-tab-lab?quickstart=1) and open the port 5173 preview |
| Run the presentation locally | Follow [Run locally](#run-locally) |
| Try the dataset in SAP RPT Playground | Upload the [masked invoice CSV](data/synthetic_invoices/exports/rpt_upload.csv) |

The interactive website is an educational simulation. The Python notebooks contain the separate invoice experiment: XGBoost runs locally, and the saved SAP RPT predictions came from a direct BTP inference request. **No SAP credentials are needed** to browse the site or inspect the saved results.

## GitHub Pages

The [Publish Paradigm Lab workflow](.github/workflows/publish.yml) is configured to test, build, and publish the website after GitHub Pages is enabled with **GitHub Actions** as its source. Once a deployment succeeds, the site will be available at **https://itsmarlo.github.io/paradigm-tab-lab/**. Until then, use Codespaces or run the app locally.

Deployment status appears under **Actions → Publish Paradigm Lab**. If a published change needs to be rolled back, revert the source commit and push to `main`.

## Develop in GitHub Codespaces

1. Open [itsmarlo/paradigm-tab-lab](https://github.com/itsmarlo/paradigm-tab-lab), or use the **Open in GitHub Codespaces** button above.
2. Select **Code → Codespaces → Create codespace on main**.
3. Wait for the container setup to finish. Dependencies install automatically and the Vite server starts on port **5173**.
4. Open **Paradigm Lab — interactive preview** from the **Ports** panel if the browser does not open automatically.

Your preview URL is `https://CODESPACE-NAME-5173.app.github.dev`. Keep port visibility **Private** for your own presentation. The server restarts automatically when the Codespace starts again. If it fails to start, run `npm run dev` in the terminal; automatic startup logs are in `/tmp/paradigm-lab-vite.log`.

If you already have a Codespace created from an older version of this project, pull the latest changes and run **Codespaces: Rebuild Container** from the command palette.

The Vite configuration permits only the current Codespace's forwarded hostname and uses a fixed port, so Codespaces' port forwarding stays aligned with the server.

[GitHub port forwarding documentation](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace)

## Run locally

Requires Node.js 22.6+ and npm.

```sh
npm ci
npm run dev
```

Open http://localhost:5173. For production:

```sh
npm run build
npm run preview
```

Run model behavior checks with `npm test`.

## Presentation

Use the navigation to explore each experiment. **Auto Demo** walks through classical training, LLM token processing, tabular pretraining and inference, and comparison. Selecting any navigation item stops the demo. **Next Step** highlights the comparison pipelines. The dark presentation theme uses blue for classical ML, amber for LLMs, and teal for tabular foundation models. Use browser fullscreen for projection; the layout also adapts to tablets and phones. Reduced-motion preferences are respected.

- Classical ML: edit labeled training values and targets, choose an algorithm, train, then adjust the separate inference row and predict. Training edits invalidate the trained model.
- LLM: enter a sentence, explore the sequence, choose a candidate, and repeatedly generate tokens.
- Tabular FM: switch between pretraining and inference. Edit context values and labels, add/remove rows, adjust the query, and predict without a training step.

## Synthetic invoice benchmark

This benchmark compares **XGBoost** and **SAP RPT 1.6** on the same fixed split and two targets: `paid_late` and `is_fraud`. The [active invoice dataset](data/synthetic_invoices/README.md) has **1,152 synthetic invoices**: 1,024 labeled rows for XGBoost training and RPT context, followed by 128 held-out test rows. There are **187 fraud labels in context and 22 in test**. The labels are generated, not real payment or fraud determinations.

| File | Purpose |
| --- | --- |
| [payment_behavior.csv](data/synthetic_invoices/payment_behavior.csv) | Single labeled source for both models |
| [exports/rpt_upload.csv](data/synthetic_invoices/exports/rpt_upload.csv) | Same rows for Playground, with both test targets masked as `[PREDICT]` |
| [rpt16_request.json](data/synthetic_invoices/rpt16_request.json) | Equivalent masked BTP request |
| [xgboost_results.csv](data/synthetic_invoices/xgboost_results.csv) | Saved XGBoost test predictions and class-1 probabilities |
| [rpt16_results.csv](data/synthetic_invoices/rpt16_results.csv) | Saved direct BTP test predictions and selected-class confidence |
| [rpt16_run_metadata.json](data/synthetic_invoices/rpt16_run_metadata.json) | Public model label and request/result checksums |

The saved results are included so the comparison works without SAP credentials. Raw BTP responses are kept outside the repository, and local `.env` is ignored. Earlier [sensitivity studies](archive/synthetic_invoice_experiments/README.md) remain outside `data/`; their results must not be mixed with this 22-case test.

The Playground upload is an **input**, not a scored result: the final 128 rows have `[PREDICT]` in both target columns. The Playground's AI summary alone does not show how many predictions were correct. A row-level export must be matched to the held-out labels before adding Playground metrics to the benchmark.

### Run the experiment notebooks

Install Python dependencies from the repository root, then open these notebooks in order:

```sh
python3 -m pip install -r requirements-experiment.txt
```

1. [XGBoost run](notebooks/xgboost_run.ipynb) trains on the main context rows and saves fresh XGBoost predictions.
2. [RPT BTP run](notebooks/rpt_btp_run.ipynb) validates the masked request and reviews the saved direct BTP result. `RUN_LIVE_RPT = False` by default. To rerun inference, copy [.env.example](.env.example) to `.env`, enter your SAP AI Core settings, and set `RUN_LIVE_RPT = True` in that notebook. Each live run can incur SAP charges.
3. [Compare saved results](notebooks/compare_saved_results.ipynb) verifies shared test IDs, true labels, and BTP checksums; it calculates the metrics and saves the [presentation comparison chart](notebooks/saved_results_comparison.png). It never calls BTP.

The [combined walkthrough](notebooks/payment_behavior_experiment.ipynb) is optional. All four notebooks use a portable Python 3 kernel and include their executed outputs. If using the command line, run `python3 scripts/generate_synthetic_invoices.py`, `python3 scripts/run_xgboost_invoices.py`, then optionally `python3 scripts/run_rpt16_invoices.py` with a configured BTP deployment.
The GitHub workflow validates the published dataset and executes the saved comparison notebook before deploying the website. It never makes a BTP request.

To check the published data without rerunning SAP inference:

```sh
python3 scripts/validate_published_experiment.py
```

### Read the results

| Target | XGBoost | SAP RPT 1.6 BTP |
| --- | --- | --- |
| Paid late | 65.6% accuracy; 62.7% recall | 71.1% accuracy; 66.7% recall |
| Fraud | 15 of 22 found; 9 false alarms | 10 of 22 found; 5 false alarms |

The results show a trade-off: RPT scores higher on late-payment accuracy and recall; XGBoost finds more synthetic fraud cases, while RPT raises fewer false alarms. XGBoost ROC AUC uses saved class-1 probabilities. RPT ROC AUC is omitted because this BTP top-1 response lacks a positive-class score for every row. One fixed synthetic split does not establish production performance or a general ranking between the models.

![XGBoost and SAP RPT 1.6 results on the shared synthetic test set](notebooks/saved_results_comparison.png)

For a Playground demo, upload the [masked CSV](data/synthetic_invoices/exports/rpt_upload.csv) and show its prediction explanation, uncertainty, and relevant context rows. The archived Playground export belongs to the earlier eight-case dataset and must not be mixed with this benchmark. Use the direct BTP results above for the quantitative slide.

The [dataset guide](data/synthetic_invoices/README.md) documents the split, feature columns, and an optional fraud-threshold experiment.

### Optional local API

Install the same Python requirements and run `python3 -m uvicorn api:app --host 127.0.0.1 --port 8000`. Open `http://127.0.0.1:8000/docs`. The API trains XGBoost locally and can trigger a live RPT request through `POST /api/experiment/rpt` when SAP credentials are configured. The presentation website does not call this API.

## Educational scope

The site teaches data flow, not actual model benchmarking. Classical algorithms share a simplified centroid-based scoring function with different scales. LLM tokens, embeddings, attention, and probabilities are illustrative. Tabular predictions use normalized, similarity-weighted context labels, not actual pretrained transformer weights. Real tabular foundation model capabilities and preprocessing requirements vary; this visualization focuses on in-context inference models. Income is in generic units of thousands. All examples are generic.

## Structure

- `src/main.tsx`: shared diagrams, tables, controls, and experiment components
- `src/model.ts`: deterministic educational scoring functions
- `src/styles.css`: responsive design, shared tokens, motion
- `.devcontainer/devcontainer.json`: Codespaces environment

Fonts use Google Fonts with local system fallbacks. The website sends no invoice data to a model or server; the optional Python RPT runner sends the masked synthetic request to the configured BTP deployment.
