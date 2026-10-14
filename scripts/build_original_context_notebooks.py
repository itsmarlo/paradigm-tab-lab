"""Build the main context walkthrough and post-engineering comparison notebooks."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


SETUP = """from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'archive/synthetic_invoice_experiments/published_baseline/payment_behavior.csv').exists())
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from context_engineering import assert_no_leakage, select_context
from run_context_benchmark import request_for, metric_row
source_path = ROOT / 'archive/synthetic_invoice_experiments/published_baseline/payment_behavior.csv'
source = pd.read_csv(source_path)
pool, holdout = source.iloc[:1024].copy(), source.iloc[1024:].copy()
targets = ('paid_late', 'is_fraud')
seed, budget = 11, 128
assert len(source) == 1152 and len(pool) == 1024 and len(holdout) == 128
assert int(holdout.is_fraud.sum()) == 22 and int(holdout.paid_late.sum()) == 51
assert source.invoice_id.is_unique and not source.isna().any().any()
assert_no_leakage(pool, holdout)
"""


def write(relative, cells):
    path = ROOT / 'notebooks' / relative
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    )
    nbf.validate(notebook)
    nbf.write(notebook, path)
    print(path)


write('context_engineering/00_original_22_case_context_workflow.ipynb', [
    md("# Context engineering on the original 22-case experiment\n\n## Goal\nShow each change between the saved first benchmark and the new model comparison. The original data and final 128 invoice IDs stay fixed. This workflow selects labeled examples; it does not retrain SAP RPT's weights."),
    md("## Setup\nThe first 1,024 rows are the eligible training/context pool. The final 128 invoices, including 22 fraud cases, are the untouched test set. The saved first benchmark used all 1,024 pool rows and its SAP request included both targets. The new comparison uses a 128-row random context and a 128-row class-balanced context, with one target per SAP request. Budget and request schema change between the first benchmark and new results; random versus balanced at 128 rows isolates selection within this dataset."),
    code(SETUP),
    md("## Step 1: check the data and split\nThe older eight-case context files are a separate pilot and are not used here. Check IDs, missing values, and target counts before looking at model results."),
    code("""audit = pd.DataFrame({'slice': ['eligible pool', 'final test'],
                      'rows': [len(pool), len(holdout)],
                      'late_payment_cases': [int(pool.paid_late.sum()), int(holdout.paid_late.sum())],
                      'fraud_cases': [int(pool.is_fraud.sum()), int(holdout.is_fraud.sum())]})
display(audit)
assert set(pool.invoice_id).isdisjoint(holdout.invoice_id)
"""),
    md("## Step 2: select the context rows\nThe **random** condition samples 128 unique labeled rows. The **balanced** condition samples as close to half positive and half negative as the eligible pool permits, separately for each target. Both use the same seed and 128-row budget. This is an explicit selection intervention, not a change to the fraud labels. The earlier eight-case pilot suggested this strategy; its scores are not reused here. This 22-case run is exploratory because strategy choice was informed by that related pilot."),
    code("""contexts = {}
composition = []
for target in targets:
    for strategy in ('random', 'balanced'):
        selected, _ = select_context(pool, target, holdout.iloc[:1], strategy, budget, seed)
        assert_no_leakage(pool, holdout, selected)
        contexts[(target, strategy)] = selected
        composition.append({'target': target, 'selection': strategy,
                            'rows': len(selected), 'positive_examples': int(selected[target].sum()),
                            'negative_examples': int((selected[target] == 0).sum())})
display(pd.DataFrame(composition))
"""),
    md("## Step 3: build a masked SAP prediction request\nOnly the selected labeled examples supply the target. All final-test target cells are `[PREDICT]`. The other outcome is excluded because it would be unavailable when predicting a new invoice. The invoice ID is an index, not a predictor."),
    code("""sample = request_for(contexts[('is_fraud', 'balanced')], holdout, 'is_fraud')
assert sample['index_column'] == 'invoice_id'
assert len(sample['rows']) == budget + len(holdout)
assert all(row['is_fraud'] == '[PREDICT]' for row in sample['rows'][budget:])
assert all('paid_late' not in row for row in sample['rows'])
display(pd.DataFrame([{'context_rows': budget, 'prediction_rows': len(holdout),
                       'target': sample['prediction_config']['target_columns'][0]['name'],
                       'query_targets_masked': True, 'other_target_excluded': True}]))
"""),
    md("## Step 4: train and score the models\nXGBoost is fit on the selected rows with the same configuration used by the context benchmark. SAP RPT receives those rows in its request, with weights fixed. `scripts/run_original_context_comparison.py` saves row-level predictions and summary metrics; use `--live` only to fill missing SAP conditions. No call to SAP is made by this notebook."),
    code("""results_path = ROOT / 'data/original_context_comparison/results.csv'
results = pd.read_csv(results_path)
expected = {(strategy, target, model) for strategy in ('random', 'balanced')
            for target in targets for model in ('XGBoost', 'SAP-RPT')}
available = set(zip(results.strategy, results.target, results.model))
display(results[['strategy', 'target', 'model', 'context_positives', 'true_positives',
                 'false_positives', 'precision', 'recall', 'f1']].round(3))
print('Conditions complete:', len(available & expected), 'of', len(expected))
"""),
    md("## Checks and interpretation\nThe first benchmark remains the saved 1,024-row run in `notebooks/before_context_engineering/original_128/`. The 128-row random control and balanced treatment use the same source, target labels, test IDs, context selection seed, and model procedures. XGBoost keeps the original model seed (20261002). In this one-seed run, balancing fraud context raises XGBoost detections from 10 to 16 and false alarms from 20 to 32; SAP RPT detections rise from 10 to 16 and false alarms from 3 to 12. The data are synthetic and the test set is fixed, so these changes are descriptive."),
])

write('after_context_engineering/01_original_22_case_after_benchmark.ipynb', [
    md("# After context engineering: XGBoost and SAP RPT on the original data\n\n## tl;dr\nSAP RPT performs better **on fraud F1 and recall** with balanced in-context examples than with random examples at the same 128-row budget: F1 rises from **0.571 to 0.640**, and it finds **16 rather than 10 of 22** fraud cases. False alarms rise from **3 to 12**. In the balanced condition SAP RPT has higher fraud F1 than XGBoost (**0.640 versus 0.457**). Late-payment F1 does not improve. The saved first benchmark and both new conditions use the same original 128 test invoices."),
    md("## Context & methods\nThe source, labels and final test IDs are identical across these rows. The published first benchmark used 1,024 examples and carried both targets in its SAP request. The controlled context comparison uses 128 examples in each condition, selection seed 11, and one target per SAP request. XGBoost trains on each selected set with the original model seed, 20261002. SAP RPT receives it as labeled request context. See the [context workflow](../context_engineering/00_original_22_case_context_workflow.ipynb) for request masking and selection checks."),
    code(SETUP),
    md("## Results\nRecompute counts from saved predictions rather than trusting display text. A missing SAP condition stays missing until a live request succeeds."),
    code("""saved = {'XGBoost': pd.read_csv(ROOT / 'data/synthetic_invoices/xgboost_results.csv'),
         'SAP-RPT': pd.read_csv(ROOT / 'data/synthetic_invoices/rpt16_results.csv')}
new = pd.read_csv(ROOT / 'data/original_context_comparison/results.csv')
predictions = pd.read_csv(ROOT / 'data/original_context_comparison/predictions.csv')
rows = []
for target in targets:
    for model, frame in saved.items():
        assert frame.invoice_id.tolist() == holdout.invoice_id.tolist()
        assert frame[f'actual_{target}'].tolist() == holdout[target].tolist()
        rows.append({'stage': 'first benchmark (1024 rows)', 'target': target, 'model': model,
                     **metric_row(holdout[target].to_numpy(), frame[f'predicted_{target}'].to_numpy())})
for result in new.itertuples():
    sample = predictions[predictions.run_id == result.run_id].set_index('invoice_id').loc[holdout.invoice_id]
    assert sample.actual.tolist() == holdout[result.target].tolist()
    checked = metric_row(sample.actual.to_numpy(), sample.predicted.to_numpy())
    assert checked['true_positives'] == result.true_positives
    assert checked['false_positives'] == result.false_positives
    rows.append({'stage': f'{result.strategy} (128 rows)', 'target': result.target,
                 'model': result.model, **checked})
comparison = pd.DataFrame(rows)
display(comparison[['stage', 'target', 'model', 'true_positives', 'false_positives',
                    'precision', 'recall', 'f1', 'accuracy']].round(3))
"""),
    md("### What improved for SAP RPT?\nThis is the controlled in-context learning comparison: only example selection changes between random and balanced contexts. The table reports percentage-point changes for recall, precision, and F1, plus invoice counts. Higher recall comes with more false alarms."),
    code("""rpt_fraud = (comparison[(comparison.model == 'SAP-RPT') &
                         (comparison.target == 'is_fraud')]
             .set_index('stage'))
before = rpt_fraud.loc['random (128 rows)']
after = rpt_fraud.loc['balanced (128 rows)']
assert (int(before.true_positives), int(before.false_positives)) == (10, 3)
assert (int(after.true_positives), int(after.false_positives)) == (16, 12)
improvement = pd.DataFrame({
    'Measure': ['Fraud cases found / 22', 'False alarms / 106', 'Fraud recall',
                'Fraud precision', 'Fraud F1'],
    'Random 128': [before.true_positives, before.false_positives,
                   before.recall, before.precision, before.f1],
    'Balanced 128': [after.true_positives, after.false_positives,
                     after.recall, after.precision, after.f1],
})
improvement['Change'] = improvement['Balanced 128'] - improvement['Random 128']
display(improvement.set_index('Measure').round(3))
fig, ax = plt.subplots(figsize=(7, 4.2))
positions = np.arange(3)
metrics = ['recall', 'precision', 'f1']
for offset, condition, row, color in [(-.18, 'Random 128', before, '#8C9AA9'),
                                       (.18, 'Balanced 128', after, '#0E807A')]:
    values = [float(row[metric]) for metric in metrics]
    bars = ax.bar(positions + offset, values, width=.35, color=color, label=condition)
    ax.bar_label(bars, labels=[f'{value:.3f}' for value in values], padding=3, fontsize=9)
ax.set_xticks(positions, ['Recall', 'Precision', 'F1'])
ax.set_ylim(0, 0.9)
ax.set_ylabel('Score on 128-invoice holdout')
ax.set_title('SAP RPT fraud prediction: random vs balanced context')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=.2)
ax.set_axisbelow(True)
fig.tight_layout()
plt.show()
"""),
    md("### Fraud detections and false alarms\nBars show counts on the same 22-positive holdout. Blank model conditions are not estimated from the eight-case pilot."),
    code("""fraud = comparison[comparison.target == 'is_fraud']
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
stages = ['first benchmark (1024 rows)', 'random (128 rows)', 'balanced (128 rows)']
models = ['XGBoost', 'SAP-RPT']
colors = {'XGBoost': '#245EAA', 'SAP-RPT': '#0E807A'}
for ax, column, title in zip(axes, ['true_positives', 'false_positives'],
                             ['Fraud cases found (of 22)', 'False fraud alarms (of 106)']):
    for offset, model in [(-0.18, 'XGBoost'), (0.18, 'SAP-RPT')]:
        subset = fraud[fraud.model == model].set_index('stage')
        for index, stage in enumerate(stages):
            if stage in subset.index:
                value = int(subset.loc[stage, column])
                ax.bar(index + offset, value, width=0.34, color=colors[model],
                       label=model if index == 0 else None)
                ax.text(index + offset, value + 0.3, str(value), ha='center', fontsize=9)
    ax.set_xticks(range(3), ['Original\\n1024', 'Random\\n128', 'Balanced\\n128'])
    ax.set_title(title)
    ax.set_ylabel('Invoices')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False)
fig.tight_layout()
plt.show()
"""),
    md("## Takeaways\nOn the original 1,024-row benchmark, XGBoost found 15 of 22 fraud cases with nine false alarms; SAP RPT found 10 with five. At a matched 128-row budget, balancing raises fraud detections from 10 to 16 for both models. False alarms also rise, from 20 to 32 for XGBoost and from 3 to 12 for SAP RPT. Balanced-context fraud F1 is 0.457 for XGBoost and 0.640 for SAP RPT; SAP RPT wins that metric in this after condition. XGBoost has higher late-payment F1 (0.571 versus 0.505). Budget and SAP request schema changed from the first benchmark, so the 128-row random control is the proper comparison for selection alone. These rankings are descriptive for one synthetic split and one selection seed."),
])
