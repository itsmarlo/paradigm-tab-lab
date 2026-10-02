# Paradigm Lab

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/itsmarlo/paradigm-tab-lab?quickstart=1)

An interactive conference presentation comparing classical machine learning, large language models, and tabular foundation models. Built with React, TypeScript, Vite, lightweight CSS, and Lucide icons. No backend or API keys.

## Use the published app

Open **https://itsmarlo.github.io/paradigm-tab-lab/** to use the app directly. No GitHub account, Codespace, installation, or terminal is needed.

The app is hosted on GitHub Pages. Pushing changes to `main` automatically runs the behavior checks, builds the app, and publishes the updated website. The workflow sets the production asset path to `/paradigm-tab-lab/`, while Codespaces and local development continue to use `/`.

Deployment progress appears under the repository's **Actions → Publish Paradigm Lab**. If a deployment fails, the previously published version remains available. To roll back, revert the source commit and push to `main`.

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

## Synthetic XGBoost and SAP-RPT-1.6 experiment

`data/synthetic_invoices/payment_behavior.csv` is the **single labeled input dataset** for both models. It contains 1,152 synthetic invoices in a fixed order. Rows 1–1,024 are the shared training/context set; rows 1,025–1,152 are the shared 128-row test set. The two targets are `paid_late` and `is_fraud` (1 means yes, 0 means no). Each row represents one invoice; all ten predictors are known when the invoice is issued. The generated outcomes contain random noise, so neither model should be expected to predict perfectly. The fraud label is a synthetic scenario for comparing models, not a real fraud determination.

| File | Use |
| --- | --- |
| `payment_behavior.csv` | The one labeled source dataset for both targets |
| `rpt16_prompt.csv` | The same rows in the same order, with both targets in the final 128 rows replaced by `[PREDICT]`; upload to the SAP-RPT Playground |
| `rpt16_request.json` | The same RPT input as a JSON payload for an AI Core deployment |
| `xgboost_results.csv` | XGBoost output on the final 128 rows: actual labels, predictions, and probabilities for both targets. Created by the runner, not a second input dataset |

Regenerate the files with `python3 scripts/generate_synthetic_invoices.py`. The seed and row counts are fixed. `invoice_id` is only a row identifier: exclude it from XGBoost features. The RPT request uses it as `index_column`. Compare both models' predictions with the last 128 labeled rows of `payment_behavior.csv` by `invoice_id`. The test labels must remain hidden from each model during prediction; the RPT files already mask them.

To run the XGBoost baseline in a Python environment:

```sh
python3 -m pip install pandas scikit-learn xgboost
python3 scripts/run_xgboost_invoices.py
```

The script prints accuracy, ROC AUC, precision, and recall for each target and writes `xgboost_results.csv` in the data directory. This is a **results file**, not a separate dataset. To run RPT-1.6 on the same rows, use an authorized SAP AI Core deployment:

```sh
export SAP_RPT_DEPLOYMENT_URL="https://YOUR-DEPLOYMENT-URL"
export SAP_RPT_AUTH_TOKEN="YOUR-TEMPORARY-TOKEN"
export SAP_RPT_RESOURCE_GROUP="default"
python3 scripts/run_rpt16_invoices.py
```

Set those variables locally; do not commit credentials. The script sends `rpt16_request.json`, writes the raw `rpt16_response.json` and a scored `rpt16_results.csv`, and checks that SAP returned exactly the shared 128 test invoice IDs. If you used the [SAP-RPT Playground](https://rpt.cloud.sap/) instead, run `python3 scripts/run_rpt16_invoices.py --response path/to/exported_response.json` to score its JSON response. The request has 1,024 context rows and 128 prediction rows with two prediction columns, within SAP's documented limits for the standard model. RPT calls require your own access. The browser presentation still uses its separate educational scoring functions, so its displayed predictions are not benchmark results.

## Educational scope

The site teaches data flow, not actual model benchmarking. Classical algorithms share a simplified centroid-based scoring function with different scales. LLM tokens, embeddings, attention, and probabilities are illustrative. Tabular predictions use normalized, similarity-weighted context labels, not actual pretrained transformer weights. Real tabular foundation model capabilities and preprocessing requirements vary; this visualization focuses on in-context inference models. Income is in generic units of thousands. All examples are generic.

## Structure

- `src/main.tsx`: shared diagrams, tables, controls, and experiment components
- `src/model.ts`: deterministic educational scoring functions
- `src/styles.css`: responsive design, shared tokens, motion
- `.devcontainer/devcontainer.json`: Codespaces environment

Fonts use Google Fonts with local system fallbacks. No data is sent to a model or server.
