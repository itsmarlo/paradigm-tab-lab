# Paradigm Lab

An interactive conference presentation comparing classical machine learning, large language models, and tabular foundation models. Built with React, TypeScript, Vite, lightweight CSS, and Lucide icons. No backend or API keys.

## Run locally

Requires Node.js 22+ and npm.

```sh
npm ci
npm run dev
```

Open http://localhost:5173. For production:

```sh
npm run build
npm run preview
```

## GitHub Codespaces

Push this project to a GitHub repository, then select **Code → Codespaces → Create codespace**. The included devcontainer installs dependencies and forwards port 5173. Run `npm run dev` in the Codespaces terminal and open the forwarded port.

## Presentation

Use the navigation to explore each experiment. **Auto Demo** walks through classical training, LLM token processing, tabular pretraining and inference, and comparison. Selecting any navigation item stops the demo. **Next Step** highlights the comparison pipelines. Use browser fullscreen for projection; the layout also adapts to tablets and phones. Reduced-motion preferences are respected.

- Classical ML: edit labeled training values and targets, choose an algorithm, train, then adjust the separate inference row and predict. Training edits invalidate the trained model.
- LLM: enter a sentence, explore the sequence, choose a candidate, and repeatedly generate tokens.
- Tabular FM: switch between pretraining and inference. Edit context values and labels, add/remove rows, adjust the query, and predict without a training step.

## Educational scope

The site teaches data flow, not actual model benchmarking. Classical algorithms share a simplified centroid-based scoring function with different scales. LLM tokens, embeddings, attention, and probabilities are illustrative. Tabular predictions use normalized, similarity-weighted context labels, not actual pretrained transformer weights. Real tabular foundation model capabilities and preprocessing requirements vary; this visualization focuses on in-context inference models. Income is in generic units of thousands. All examples are generic.

## Structure

- `src/main.tsx`: shared diagrams, tables, controls, and experiment components
- `src/model.ts`: deterministic educational scoring functions
- `src/styles.css`: responsive design, shared tokens, motion
- `.devcontainer/devcontainer.json`: Codespaces environment

Fonts use Google Fonts with local system fallbacks. No data is sent to a model or server.
