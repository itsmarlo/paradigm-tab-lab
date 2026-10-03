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
| `exports/rpt_upload.csv` | Upload format derived from the same rows, with both targets in the final 128 rows replaced by `[PREDICT]`; upload to the SAP-RPT Playground |
| `rpt16_request.json` | The same RPT input as a JSON payload for an AI Core deployment |
| `xgboost_results.csv` | XGBoost output on the final 128 rows: actual labels, predictions, and probabilities for both targets. Created by the runner, not a second input dataset |
| `exports/rpt_playground_export.csv` | Copy of the supplied SAP-RPT Playground table export; its final 128 target values are treated as RPT class predictions in the notebook |

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

Set those variables locally; do not commit credentials. The script sends `rpt16_request.json`, writes the raw `rpt16_response.json`, a scored `rpt16_results.csv`, and `rpt16_run_metadata.json`, and checks that SAP returned exactly the shared 128 test invoice IDs. The metadata records the request and result checksums plus a hash of the deployment URL; it does not store the token. If you used the [SAP-RPT Playground](https://rpt.cloud.sap/) instead, run `python3 scripts/run_rpt16_invoices.py --response path/to/exported_response.json` to score its JSON response; the metadata then marks it as an exported response rather than a direct BTP run. The request has 1,024 context rows and 128 prediction rows with two prediction columns, within SAP's documented limits for the standard model. RPT calls require your own access. The browser presentation still uses its separate educational scoring functions, so its displayed predictions are not benchmark results.

The runner can also load a repository-root `.env` file with SAP AI Core service-key settings: `SAP_AI_CORE_API_URL`, `SAP_AI_CORE_AUTH_URL`, `SAP_AI_CORE_CLIENT_ID`, `SAP_AI_CORE_CLIENT_SECRET`, and `SAP_RPT_DEPLOYMENT_ID`. It obtains an OAuth token, fetches the deployment URL, and then makes the prediction request. `SAP_AI_CORE_RESOURCE_GROUP` and `SAP_RPT_MODEL_NAME` are optional. The configured model name is recorded as a label, so verify that it matches the actual deployment before presenting a model-version comparison. `RPT_DRY_RUN=true` prevents a live inference call.

### Local API and Jupyter notebook

From the repository root, install the experiment dependencies and start the API:

```sh
python3 -m pip install -r requirements-experiment.txt
python3 -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/docs** for interactive API testing. `GET /api/experiment` returns the fixed 128-row test metrics and a few XGBoost examples. `POST /api/predictions/xgboost` accepts one new invoice and predicts both targets. `POST /api/experiment/rpt` sends the same fixed test split to your configured SAP-RPT-1.6 deployment, saves its results, and can incur SAP inference charges each time it is called. Without a deployment URL and token, it returns HTTP 503 with `RPT_NOT_CONFIGURED`. The API is local only; it is not deployed or connected to the presentation website.

An example request for a new invoice:

```sh
curl -X POST http://127.0.0.1:8000/api/predictions/xgboost \
  -H 'Content-Type: application/json' \
  -d '{"invoice_amount_eur":2400,"payment_terms_days":30,"customer_tenure_months":24,"prior_late_payment_rate":0.2,"open_invoice_count":2,"customer_segment":"midmarket","region":"DACH","billing_address_mismatch":0,"bank_account_changed_recently":0,"weekend_submission":0}'
```

Open [the experiment notebook](notebooks/payment_behavior_experiment.ipynb) in Jupyter from the repository root. For a presentation screenshot, use the clearly labeled **“Screenshot this cell: XGBoost prediction results”** section; the same figure is saved as [xgboost_prediction_results.png](notebooks/xgboost_prediction_results.png). The next section runs the fraud classifier directly and lists all invoices it flagged. The notebook validates the supplied Playground table export against the shared dataset and plots [the XGBoost/RPT comparison](notebooks/model_comparison.png) on the same 128 test invoices. The export contains class labels but no model metadata or probabilities, so its RPT provenance is based on the supplied file and RPT ROC AUC cannot be calculated. The local API still expects `rpt16_results.csv` from a deployment response. The notebook and API use the same dataset and split as the command-line scripts.

For separate, reproducible runs, open these notebooks in order:

1. [XGBoost run](notebooks/xgboost_run.ipynb) trains on the fixed split and saves `xgboost_results.csv`.
2. [RPT BTP run](notebooks/rpt_btp_run.ipynb) validates the masked request and loads `.env` automatically. Change `RUN_LIVE_RPT` to `True` in its run cell to call your deployment and save its response, results, and metadata. The live call may incur charges.
3. [Compare saved results](notebooks/compare_saved_results.ipynb) checks matching invoice IDs and labels, calculates metrics, and saves [the comparison chart](notebooks/saved_results_comparison.png). It uses a validated direct BTP run when present; otherwise it explicitly labels the supplied Playground export. It never calls BTP.

The older combined notebook remains a presentation walkthrough. The three notebooks above keep model execution separate from comparison, so rerunning a chart does not trigger another BTP request.

## Educational scope

The site teaches data flow, not actual model benchmarking. Classical algorithms share a simplified centroid-based scoring function with different scales. LLM tokens, embeddings, attention, and probabilities are illustrative. Tabular predictions use normalized, similarity-weighted context labels, not actual pretrained transformer weights. Real tabular foundation model capabilities and preprocessing requirements vary; this visualization focuses on in-context inference models. Income is in generic units of thousands. All examples are generic.

## Structure

- `src/main.tsx`: shared diagrams, tables, controls, and experiment components
- `src/model.ts`: deterministic educational scoring functions
- `src/styles.css`: responsive design, shared tokens, motion
- `.devcontainer/devcontainer.json`: Codespaces environment

Fonts use Google Fonts with local system fallbacks. No data is sent to a model or server.
