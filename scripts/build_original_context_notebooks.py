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
FIONEER_BLUE = '#0F37FF'
FIONEER_NAVY = '#353C54'
FIONEER_LIGHT_BLUE = '#CFDEFF'
FIONEER_YELLOW = '#FFD53C'
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
    md("# After context engineering: SAP RPT on the original data\n\n## tl;dr\nAt the same 128-row context budget, balanced examples raise SAP RPT fraud recall from **45.5% to 72.7%** and F1 from **0.571 to 0.640**. The model finds **16 rather than 10 of 22** fraud cases, while false alarms rise from **3 to 12**. Late-payment F1 changes from **0.511 to 0.505**. Against the unchanged XGBoost benchmark, balanced SAP RPT finds **16 versus 15** fraud cases, with **12 versus 9** false alarms and F1 **0.640 versus 0.652**. This last comparison is descriptive because XGBoost trained on 1,024 rows while SAP RPT received 128 context rows."),
    md("## Context & methods\nThe source, labels and final 128 test invoices are identical in both conditions. The first SAP RPT benchmark used 1,024 labeled examples and carried both targets in its request. The controlled comparison uses 128 examples in each condition, selection seed 11, and one target per request. SAP RPT weights stay fixed; only the labeled context rows change between random and balanced selection. See the [context workflow](../context_engineering/00_original_22_case_context_workflow.ipynb) for request masking and selection checks. The earlier 1,024-row SAP RPT result is historical context because its budget and request schema differ. The separate XGBoost reference uses its saved original 1,024-row training result; no context selection or retraining was applied to XGBoost for that reference."),
    code(SETUP),
    md("## Results\nRecompute SAP RPT counts from saved predictions. A missing condition stays missing until a live request succeeds."),
    code("""saved = pd.read_csv(ROOT / 'data/synthetic_invoices/rpt16_results.csv')
new = pd.read_csv(ROOT / 'data/original_context_comparison/results.csv')
new = new[new.model == 'SAP-RPT'].copy()
predictions = pd.read_csv(ROOT / 'data/original_context_comparison/predictions.csv')
assert set(zip(new.strategy, new.target)) == {
    (strategy, target) for strategy in ('random', 'balanced') for target in targets
}
rows = []
for target in targets:
    assert saved.invoice_id.tolist() == holdout.invoice_id.tolist()
    assert saved[f'actual_{target}'].tolist() == holdout[target].tolist()
    rows.append({'stage': 'first benchmark (1024 rows)', 'target': target,
                 **metric_row(holdout[target].to_numpy(), saved[f'predicted_{target}'].to_numpy())})
for result in new.itertuples():
    sample = predictions[predictions.run_id == result.run_id].set_index('invoice_id').loc[holdout.invoice_id]
    assert sample.actual.tolist() == holdout[result.target].tolist()
    checked = metric_row(sample.actual.to_numpy(), sample.predicted.to_numpy())
    assert checked['true_positives'] == result.true_positives
    assert checked['false_positives'] == result.false_positives
    rows.append({'stage': f'{result.strategy} (128 rows)', 'target': result.target,
                 **checked})
comparison = pd.DataFrame(rows)
display(comparison[['stage', 'target', 'true_positives', 'false_positives',
                    'precision', 'recall', 'f1', 'accuracy']].round(3))
"""),
    md("### What improved for SAP RPT?\nThis is the controlled in-context learning comparison: only example selection changes between random and balanced contexts. The table reports percentage-point changes for recall, precision, and F1, plus invoice counts. Higher recall comes with more false alarms."),
    code("""rpt_fraud = comparison[comparison.target == 'is_fraud'].set_index('stage')
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
for offset, condition, row, color in [(-.18, 'Random 128', before, FIONEER_YELLOW),
                                       (.18, 'Balanced 128', after, FIONEER_BLUE)]:
    values = [float(row[metric]) for metric in metrics]
    bars = ax.bar(positions + offset, values, width=.35, color=color,
                  edgecolor=FIONEER_NAVY, linewidth=.6, label=condition)
    ax.bar_label(bars, labels=[f'{value:.3f}' for value in values], padding=3, fontsize=9)
ax.set_xticks(positions, ['Recall', 'Precision', 'F1'])
ax.set_ylim(0, 0.9)
ax.set_ylabel('Score on 128-invoice holdout')
ax.set_title('SAP RPT fraud prediction: random vs balanced context')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=.2)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(ROOT / 'reports/figures/original22_rpt_fraud_random_vs_balanced.png',
            dpi=200, bbox_inches='tight')
plt.show()
"""),
    md("### Fraud detections and false alarms\nThe first 1,024-row SAP RPT benchmark provides historical context. Compare the two 128-row conditions to assess selection while holding the context budget fixed. All bars use the same test set with 22 fraud and 106 nonfraud invoices."),
    code("""fraud = comparison[comparison.target == 'is_fraud'].set_index('stage')
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
stages = ['first benchmark (1024 rows)', 'random (128 rows)', 'balanced (128 rows)']
for ax, column, title in zip(axes, ['true_positives', 'false_positives'],
                             ['Fraud cases found (of 22)', 'False fraud alarms (of 106)']):
    values = [int(fraud.loc[stage, column]) for stage in stages]
    bars = ax.bar(range(3), values, width=0.56,
                  color=[FIONEER_NAVY, FIONEER_YELLOW, FIONEER_BLUE],
                  edgecolor=FIONEER_NAVY, linewidth=.6)
    ax.bar_label(bars, labels=[str(value) for value in values], padding=3, fontsize=9)
    ax.set_xticks(range(3), ['Original\\n1024', 'Random\\n128', 'Balanced\\n128'])
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_title(title)
    ax.set_ylabel('Invoices')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
fig.suptitle('SAP RPT fraud predictions on the same 128 invoices')
fig.tight_layout()
fig.savefig(ROOT / 'reports/figures/original22_rpt_fraud_detections_false_alarms.png',
            dpi=200, bbox_inches='tight')
plt.show()
"""),
    md("### Reference: balanced SAP RPT and unchanged XGBoost\nBoth predictions use the same original 128 test invoice IDs and labels. XGBoost trained on all 1,024 eligible rows in the first benchmark; SAP RPT used 128 balanced in-context examples. This table locates the after-context SAP RPT result against the XGBoost result. Different example budgets and modeling procedures prevent a controlled model ranking or an estimate of the context-selection effect from this cross-model comparison."),
    code("""xgb_saved = pd.read_csv(ROOT / 'data/synthetic_invoices/xgboost_results.csv')
reference_rows = []
for target in targets:
    assert xgb_saved.invoice_id.tolist() == holdout.invoice_id.tolist()
    assert xgb_saved[f'actual_{target}'].tolist() == holdout[target].tolist()
    xgb = metric_row(holdout[target].to_numpy(), xgb_saved[f'predicted_{target}'].to_numpy())
    rpt = comparison[(comparison.target == target) &
                     (comparison.stage == 'balanced (128 rows)')].iloc[0]
    for model, labeled_rows, metrics in [
        ('XGBoost', 1024, xgb),
        ('SAP RPT balanced', 128, rpt),
    ]:
        reference_rows.append({
            'Target': target, 'Condition': model, 'Labeled rows': labeled_rows,
            'Cases found': f"{int(metrics['true_positives'])}/{int(holdout[target].sum())}",
            'False alarms': int(metrics['false_positives']),
            'Precision': float(metrics['precision']),
            'Recall': float(metrics['recall']),
            'F1': float(metrics['f1']),
        })
reference = pd.DataFrame(reference_rows)
display(reference.round(3))
"""),
    md("### Visual reference: XGBoost versus balanced-context SAP RPT\nThe same 128 holdout invoices are scored in both conditions. The F1 panel covers both targets; the fraud panel shows the detection and false-alarm tradeoff. XGBoost trained on 1,024 rows, while SAP RPT used 128 selected context rows, so these bars are a descriptive reference rather than a controlled model ranking."),
    code("""labels = ['XGBoost', 'SAP RPT balanced']
colors = [FIONEER_YELLOW, FIONEER_BLUE]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1.05, 1]})
positions = np.arange(2)
width = 0.34
for index, (condition, color) in enumerate(zip(labels, colors)):
    subset = reference.set_index(['Target', 'Condition'])
    scores = [float(subset.loc[(target, condition), 'F1']) for target in ('paid_late', 'is_fraud')]
    bars = axes[0].bar(positions + (index - .5) * width, scores, width,
                       color=color, edgecolor=FIONEER_NAVY, linewidth=.6, label=condition)
    axes[0].bar_label(bars, labels=[f'{score:.3f}' for score in scores], padding=3, fontsize=9)
    fraud = subset.loc[('is_fraud', condition)]
    counts = [int(fraud['Cases found'].split('/')[0]), int(fraud['False alarms'])]
    bars = axes[1].bar(positions + (index - .5) * width, counts, width,
                       color=color, edgecolor=FIONEER_NAVY, linewidth=.6)
    axes[1].bar_label(bars, labels=[str(count) for count in counts], padding=3, fontsize=9)
axes[0].set_xticks(positions, ['Late payment', 'Fraud'])
axes[0].set_ylim(0, .8)
axes[0].set_ylabel('F1 score')
axes[0].set_title('Prediction quality by target')
axes[1].set_xticks(positions, ['Cases found\\n(of 22)', 'False alarms\\n(of 106)'])
axes[1].set_ylim(0, 22)
axes[1].set_ylabel('Invoices')
axes[1].set_title('Fraud decisions')
for ax in axes:
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
           bbox_to_anchor=(.5, .91), ncol=2, frameon=False)
fig.suptitle('XGBoost vs balanced-context SAP RPT | same 128 test invoices',
             fontsize=13, y=.995)
fig.text(.5, .025,
         'XGBoost: 1,024 training rows  •  SAP RPT: 128 context rows  •  Descriptive reference; budgets differ',
         ha='center', fontsize=9, color=FIONEER_NAVY)
fig.subplots_adjust(top=.77, bottom=.23, left=.08, right=.98, wspace=.28)
figure_path = ROOT / 'reports/figures/xgboost_vs_context_enhanced_rpt.png'
figure_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_path, dpi=200, bbox_inches='tight')
plt.show()
print(f'Saved {figure_path}')
"""),
    md("## Takeaways\nAt the matched 128-row budget, SAP RPT finds six more fraud cases with balanced context than with random context: **16 versus 10 of 22**. False alarms rise from **3 to 12**, and precision falls from **76.9% to 57.1%** even as fraud F1 rises from **0.571 to 0.640**. Late-payment F1 changes from **0.511 to 0.505**. The unchanged XGBoost result finds **15 of 22** fraud cases with **9** false alarms and F1 **0.652**, compared with **16**, **12**, and **0.640** for balanced SAP RPT. XGBoost trained on 1,024 rows, while SAP RPT used 128 context rows, so the cross-model figures are a reference rather than an equal-budget model comparison. These findings are descriptive for one synthetic split and one selection seed."),
])
