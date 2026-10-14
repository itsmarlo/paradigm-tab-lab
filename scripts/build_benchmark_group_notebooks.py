"""Build the revised-label eight-case pilot benchmark notebooks."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(source):
    return nbf.v4.new_markdown_cell(source)


def code(source):
    return nbf.v4.new_code_cell(source)


SETUP = """from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents]
            if (path / 'data/semantic_ablation/source_snapshot.csv').exists())
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from context_engineering import select_context, assert_no_leakage
from run_context_benchmark import xgb_predict, metric_row

source = pd.read_csv(ROOT / 'data/semantic_ablation/source_snapshot.csv')
pool, holdout = source.iloc[:1024].copy(), source.iloc[1024:].copy()
sap_results = pd.read_csv(ROOT / 'data/context_benchmark/selection_holdout_results.csv')
sap_predictions = pd.read_csv(ROOT / 'data/context_benchmark/selection_holdout_predictions.csv')
targets = {'paid_late': 'Late payment', 'is_fraud': 'Fraud'}
seeds = (11, 23, 37, 41, 53)
assert len(source) == 1152 and len(holdout) == 128 and holdout.is_fraud.sum() == 8
assert_no_leakage(pool, holdout)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 120})
"""


EVALUATE = """xgb_rows = []
xgb_predictions = []
for target in targets:
    for seed in seeds:
        context, _ = select_context(pool, target, holdout.iloc[:1], strategy, 128, seed)
        assert_no_leakage(context, holdout)
        predicted, _, _, _ = xgb_predict(context, holdout, target, seed)
        xgb_rows.append({'model': 'XGBoost', 'target': target, 'strategy': strategy,
                         'seed': seed, 'context_positives': int(context[target].sum()),
                         **metric_row(holdout[target].to_numpy(), predicted)})
        xgb_predictions.extend({'model': 'XGBoost', 'target': target, 'seed': seed,
                                'invoice_id': invoice_id, 'actual': int(actual),
                                'predicted': int(prediction)}
                               for invoice_id, actual, prediction in
                               zip(holdout.invoice_id, holdout[target], predicted))

xgb = pd.DataFrame(xgb_rows)
xgb_predictions = pd.DataFrame(xgb_predictions)
sap = sap_results[(sap_results.model == 'SAP-RPT') &
                  (sap_results.strategy == strategy)].copy()
assert len(xgb) == len(sap) == 10
matched = xgb.merge(sap[['target', 'seed', 'context_positives']],
                    on=['target', 'seed'], suffixes=('_xgb', '_sap'))
assert len(matched) == 10
assert matched.context_positives_xgb.equals(matched.context_positives_sap)

saved_predictions = sap_predictions[sap_predictions.run_id.isin(sap.run_id)]
assert len(saved_predictions) == len(sap) * len(holdout)
for result in sap.itertuples():
    rows = saved_predictions[saved_predictions.run_id == result.run_id]
    assert set(rows.invoice_id) == set(holdout.invoice_id)
    checked = metric_row(rows.actual.to_numpy(), rows.predicted.to_numpy())
    assert np.isclose(checked['f1'], result.f1)
    assert checked['true_positives'] == result.true_positives
    assert checked['false_positives'] == result.false_positives

results = pd.concat([xgb, sap], ignore_index=True)
results['Target'] = results.target.map(targets)
summary = (results.groupby(['Target', 'model'])
           .agg(F1=('f1', 'mean'), F1_seed_SD=('f1', 'std'),
                Precision=('precision', 'mean'), Recall=('recall', 'mean'),
                True_positives=('true_positives', 'mean'),
                False_positives=('false_positives', 'mean'),
                Positive_context_rows=('context_positives', 'mean')))
display(summary.round(3))
"""


CHART = """fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
colors = {'XGBoost': '#FFD53C', 'SAP-RPT': '#0F37FF'}
for index, (target, label) in enumerate(targets.items()):
    for model, offset in [('XGBoost', -0.19), ('SAP-RPT', 0.19)]:
        values = results[(results.target == target) & (results.model == model)].f1
        x = index + offset
        axes[0].bar(x, values.mean(), width=0.35, yerr=values.std(),
                    capsize=3, color=colors[model], label=model)
        axes[0].text(x, values.mean() + 0.035, f'{values.mean():.3f}',
                     ha='center', fontsize=9)
axes[0].set_xticks([0, 1], list(targets.values()))
axes[0].set_ylabel('Holdout F1 (mean ± seed SD)')
axes[0].set_ylim(0, 0.8)
axes[0].set_title('Prediction F1 by target')
axes[0].grid(axis='y', alpha=0.2)
axes[0].set_axisbelow(True)
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles[:2], labels[:2], frameon=False)

fraud = results[results.target == 'is_fraud'].groupby('model')
for index, model in enumerate(['XGBoost', 'SAP-RPT']):
    values = fraud.get_group(model)
    axes[1].bar(index - 0.19, values.true_positives.mean(), width=0.35,
                color=colors[model], label='True positives' if index == 0 else None)
    axes[1].bar(index + 0.19, values.false_positives.mean(), width=0.35,
                color=colors[model], alpha=0.45, hatch='//',
                label='False positives' if index == 0 else None)
    axes[1].text(index - 0.19, values.true_positives.mean() + 0.5,
                 f'{values.true_positives.mean():.1f}', ha='center', fontsize=9)
    axes[1].text(index + 0.19, values.false_positives.mean() + 0.5,
                 f'{values.false_positives.mean():.1f}', ha='center', fontsize=9)
axes[1].set_xticks([0, 1], ['XGBoost', 'SAP RPT'])
axes[1].set_ylabel('Invoices per 128-row holdout')
axes[1].set_title('Fraud detections and false alarms')
axes[1].legend(frameon=False)
axes[1].grid(axis='y', alpha=0.2)
axes[1].set_axisbelow(True)
fig.tight_layout()
plt.show()
"""


def build(name, strategy, title, summary, method, takeaway):
    assert strategy in ("random", "balanced")
    cells = [
        md(f"# {title}\n\n## tl;dr\n{summary}"),
        md("## Context & methods\n" + method + "\n\nBoth models use the same 128-row selected training/context IDs per target and seed, and predict the same frozen 128-invoice holdout. Five seeds vary the selected examples; the holdout itself is fixed. SAP RPT results are saved predictions; XGBoost is reproduced locally. No live SAP call is made."),
        code(SETUP),
    ]
    if strategy == "balanced":
        cells += [
            md("## Validation choice\nOnly global strategies evaluated with five seeds are eligible for the final selection. The selection is computed without final-holdout labels."),
            code("""validation = pd.read_csv(ROOT / 'data/context_benchmark/selection_validation_results.csv')
eligible = validation[validation.strategy.isin(['random', 'stratified', 'balanced', 'diversity'])]
chosen = (eligible.groupby(['target', 'strategy']).f1.mean()
          .groupby(level=0).idxmax().map(lambda item: item[1]).to_dict())
assert chosen == {'paid_late': 'balanced', 'is_fraud': 'balanced'}
display(pd.DataFrame({'Target': list(targets.values()),
                      'Validation-selected strategy': [chosen[target] for target in targets]}))
"""),
        ]
    cells += [
        md("## Results\n\n### Select rows and verify model predictions"),
        code(f"strategy = '{strategy}'\n" + EVALUATE),
        md("### Visual comparison\nF1 error bars show standard deviation across the five context seeds. Fraud detection and false-alarm bars show means per 128-invoice holdout."),
        code(CHART),
        md("## Takeaways\n" + takeaway),
    ]
    path = NOTEBOOKS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )
    nbf.validate(notebook)
    nbf.write(notebook, path)
    print(path)


build(
    "eight_case_pilot/01_random_context_baseline.ipynb",
    "random",
    "Eight-case pilot: random-context benchmark",
    "This exploratory pilot samples 128 labeled examples at random from the revised-label source and compares XGBoost with SAP RPT on the same 128 held-out invoices as the balanced pilot condition.",
    "Source: `data/semantic_ablation/source_snapshot.csv`. Random rows are drawn from the first 1,024 invoices; the final 128 are held out. Saved SAP runs come from `data/context_benchmark/selection_holdout_*`. This is an eight-case exploratory pilot, separate from the [original 22-case benchmark](../before_context_engineering/README.md).",
    "Random context contains few fraud positives, so both models detect few fraud cases. Read precision, recall, F1 and false positives together. The [balanced pilot](02_balanced_context_benchmark.ipynb) changes only the selected examples on this same frozen dataset.",
)

build(
    "eight_case_pilot/02_balanced_context_benchmark.ipynb",
    "balanced",
    "Eight-case pilot: balanced-context benchmark",
    "Validation selected a class-balanced 128-row context for both targets. This exploratory pilot compares XGBoost and SAP RPT on the revised-label holdout and matched context IDs.",
    "Source: `data/semantic_ablation/source_snapshot.csv`. The strategy was chosen from SAP RPT mean F1 on `data/context_benchmark/selection_validation_results.csv`, using rows 897–1,024 as validation. Final context/training examples can come from rows 1–1,024; the final 128 remain untouched. The fraud context uses all 53 available positive examples and 75 negative examples. This is a selection change, not a feature-name change or SAP weight update.",
    "The balanced strategy raises fraud recall for both models and also produces more false alarms. The small eight-fraud-case holdout makes the model ranking descriptive. Compare with the [random pilot](01_random_context_baseline.ipynb); these results do not provide after scores for the original 22-case benchmark.",
)
