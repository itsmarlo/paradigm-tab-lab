"""Build eight executable, saved-result research notebooks without live API calls."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "notebooks/context_engineering"
DEST.mkdir(parents=True, exist_ok=True)


def md(value):
    return nbf.v4.new_markdown_cell(value)


def code(value):
    return nbf.v4.new_code_cell(value)


SETUP = """from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'data/semantic_ablation/source_snapshot.csv').exists())
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
plt.rcParams.update({'figure.figsize': (9, 4), 'axes.spines.top': False, 'axes.spines.right': False})
"""


def notebook(name, cells):
    nb = nbf.v4.new_notebook(cells=[code(SETUP), *cells], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nbf.write(nb, DEST / name)


notebook("01_context_engineering_fundamentals.ipynb", [
    md("# 01 · Context engineering fundamentals\n\n## tl;dr\nSAP RPT receives labeled invoice rows inside each inference request. XGBoost fits model weights on the same rows. This notebook shows the frozen data and checks the split used by later experiments."),
    md("## Context & methods\nThe source is `data/semantic_ablation/source_snapshot.csv`. Rows 1–896 form the selection pool, 897–1024 validation, and 1025–1152 the final holdout. Context examples must come from eligible earlier rows. Global selection chooses one subset for all queries; query-specific selection chooses a subset per invoice without reading its target."),
    code("data = pd.read_csv(ROOT / 'data/semantic_ablation/source_snapshot.csv')\nselection, validation, holdout = data.iloc[:896], data.iloc[896:1024], data.iloc[1024:]\nassert data.invoice_id.is_unique and not data.isna().any().any()\nassert not set(selection.invoice_id) & set(holdout.invoice_id)\ndisplay(pd.DataFrame({'split': ['selection', 'validation', 'holdout'], 'rows': [len(selection), len(validation), len(holdout)], 'late_positives': [selection.paid_late.sum(), validation.paid_late.sum(), holdout.paid_late.sum()], 'fraud_positives': [selection.is_fraud.sum(), validation.is_fraud.sum(), holdout.is_fraud.sum()]}))"),
    code("composition = pd.DataFrame({'late payment': [selection.paid_late.mean(), validation.paid_late.mean(), holdout.paid_late.mean()], 'fraud': [selection.is_fraud.mean(), validation.is_fraud.mean(), holdout.is_fraud.mean()]}, index=['selection','validation','holdout'])\nax = composition.plot.bar(rot=0, color=['#0F37FF','#FFD53C']); ax.set_ylabel('Positive-label fraction'); ax.set_title('Target composition by split'); plt.tight_layout(); plt.show()"),
    md("## Takeaways\nFraud is rare in the frozen snapshot. Precision, recall, F1 and false positives are essential to interpret changes; accuracy alone would hide missed fraud.")
])

notebook("02_context_size_benchmark.ipynb", [
    md("# 02 · Context size benchmark\n\n## tl;dr\nSeven labeled-data budgets, five seeds, identical selected rows for SAP RPT and XGBoost at each budget and seed. The 1,024-row context has only one distinct SAP input; repeated calls do not add independent samples."),
    md("## Context & methods\nThe fixed 128-invoice holdout is never sampled into context. For sizes below 1,024, subsets come from rows 1–896. At 1,024 all non-holdout records are used. Saved rows are actual model predictions, not templates. RPT returns class labels without comparable probability scores."),
    code("result = pd.read_csv(ROOT / 'data/context_benchmark/size_results.csv')\nassert set(result['size']) == {16,32,64,128,256,512,1024}\ndisplay(result.groupby(['target','model','size'])[['f1','recall','balanced_accuracy','inference_seconds','context_positives']].agg(['mean','std']).round(3))\ndisplay(result[result.model=='XGBoost'].groupby(['target','size'])[['roc_auc','average_precision']].agg(['mean','std']).round(3))"),
    code("fig, axes = plt.subplots(1, 2, figsize=(11,4), sharey=True)\nfor ax, target in zip(axes, ['paid_late','is_fraud']):\n    for model, color in [('SAP-RPT','#0F37FF'),('XGBoost','#FFD53C')]:\n        sub = result[(result.target==target)&(result.model==model)].groupby('size').f1.agg(['mean','std'])\n        ax.errorbar(sub.index, sub['mean'], yerr=sub['std'].fillna(0), marker='o', capsize=2, label=model, color=color)\n    ax.set_xscale('log',base=2); ax.set_title(target); ax.set_xlabel('Labeled examples'); ax.set_ylim(bottom=0); ax.grid(alpha=.2)\naxes[0].set_ylabel('F1 (mean ± seed SD)'); axes[1].legend(); plt.tight_layout(); plt.show()"),
    md("## Takeaways\nRead the per-seed table with the chart. Context size changes both the number of labeled examples and, at small budgets, the minority-class count. SAP request latency and XGBoost prediction latency measure different execution paths; XGBoost fitting time is reported separately.")
])

notebook("03_context_selection_strategies.ipynb", [
    md("# 03 · Context selection strategies\n\n## tl;dr\nCompare random, stratified, class-balanced, diversity, similarity and hybrid selection at a fixed 128-row budget. Validation results, not final-holdout results, choose the final global strategy."),
    md("## Methods\nNumeric features are standardized and categorical fields one-hot encoded using eligible context data only. Similarity and hybrid each completed the full 128-invoice holdout with one seed after API rate limiting; global strategies have five seeds. Balanced sampling changes the effective class prior, so inspect false positives."),
    code("validation = pd.read_csv(ROOT / 'data/context_benchmark/selection_validation_results.csv')\nresult = pd.read_csv(ROOT / 'data/context_benchmark/selection_holdout_results.csv')\nchosen = validation.groupby(['target','strategy']).f1.mean().groupby(level=0).idxmax()\ndisplay(pd.DataFrame({'validation_selected_strategy': {k:v[1] for k,v in chosen.items()}}))\ndisplay(result.groupby(['target','strategy'])[['f1','recall','false_positives','context_build_seconds','inference_seconds']].agg(['mean','std']).round(3))"),
    code("fig, axes = plt.subplots(1, 2, figsize=(11,4), sharey=True)\nfor ax, target in zip(axes,['paid_late','is_fraud']):\n    sub=result[result.target==target].groupby('strategy').f1.agg(['mean','std'])\n    ax.bar(sub.index,sub['mean'],yerr=sub['std'].fillna(0),capsize=2,color='#0F37FF'); ax.tick_params(axis='x',rotation=45); ax.set_title(target); ax.grid(axis='y',alpha=.2)\naxes[0].set_ylabel('F1 (SD where n=5)'); fig.suptitle('Global strategies: 5 seeds; query-specific: 1 seed'); plt.tight_layout(); plt.show()"),
    md("## Takeaways\nThe final strategy must be named from validation before reading the final holdout. Similarity and hybrid are exploratory one-seed measurements with one SAP request per invoice; neither was eligible for the validation-selected final strategy. Their cumulative runtime includes API pacing and retry delays.")
])

SEMANTIC_CELLS = [
    md("# Semantic context engineering\n\n## tl;dr\nS0–S3 isolate names and category labels on identical rows and values. S4–S5 test conventional cleaning; the audited source needs no numeric or category repair, so these are identity controls."),
    md("## Context & methods\nThree paired 512-row contexts and one 128-row holdout were used. Targets and invoice IDs retain their original names. XGBoost receives one invariant encoded matrix across all semantic conditions. The original names are already fairly descriptive and there are no free-text fields."),
    code("import json\nprotocol=json.loads((ROOT/'data/semantic_ablation/protocol.json').read_text())\nname_map=pd.DataFrame(protocol['names'].items(),columns=['original feature','enhanced feature'])\ndisplay(name_map)\nfor feature,mapping in protocol['categories'].items():\n    print(feature); display(pd.DataFrame(mapping.items(),columns=['original value','enhanced value']))"),
    code("results=pd.read_csv(ROOT/'data/semantic_ablation/results.csv')\ndisplay(results.groupby(['model','condition','target'])[['f1','recall','balanced_accuracy','inference_seconds']].agg(['mean','std']).round(3))\ndisplay(results[results.model=='XGBoost'].groupby(['condition','target'])[['roc_auc','average_precision']].mean().round(3))"),
    code("validation_results=pd.read_csv(ROOT/'data/semantic_ablation/validation_results.csv')\ndisplay(validation_results.groupby(['target','condition'])[['f1','recall']].agg(['mean','std']).round(3))\nprint('Validation-selected:',{target: sub.groupby('condition').f1.mean().idxmax() for target,sub in validation_results.groupby('target')})"),
    md("### Feature selection control\nThe generator's late-payment formula does not use `weekend_submission`; its fraud formula does not use `region`. A separate single-target request compares each feature removal with a matched single-target baseline. This is not mixed into S0–S5 because changing the target configuration would confound that ablation."),
    code("feature_results=pd.read_csv(ROOT/'data/semantic_ablation/feature_selection_results.csv')\ndisplay(feature_results.groupby(['target','condition'])[['f1','recall','accuracy','inference_seconds']].agg(['mean','std']).round(3))\nfeature_paired=feature_results.pivot_table(index=['seed','target'],columns='condition',values='f1')\nprint('Feature-removed minus matched baseline F1:')\ndisplay((feature_paired['feature_removed']-feature_paired['single_target_baseline']).round(3))"),
    code("fig, axes=plt.subplots(1,2,figsize=(10,4),sharey=True)\nfor ax,target in zip(axes,['paid_late','is_fraud']):\n    sub=results[(results.model=='SAP-RPT')&(results.target==target)].groupby('condition').f1.agg(['mean','std'])\n    ax.bar(sub.index,sub['mean'],yerr=sub['std'].fillna(0),capsize=2,color='#0F37FF'); ax.set_title(target); ax.grid(axis='y',alpha=.2)\naxes[0].set_ylabel('F1 (mean ± seed SD)'); plt.tight_layout(); plt.show()"),
    code("paired=results[results.model=='SAP-RPT'].pivot_table(index=['seed','target'],columns='condition',values='f1')\nfor target in ['paid_late','is_fraud']:\n    q=paired.xs(target,level='target')\n    delta=q['S3']-q['S0']\n    print(target,'S3-S0 paired F1:',delta.round(4).to_dict(),'mean',round(delta.mean(),4),'SD',round(delta.std(),4))\nassert (paired['S0']==paired['S4']).all() and (paired['S3']==paired['S5']).all()\nxgb=results[results.model=='XGBoost']; assert xgb.groupby(['seed','target']).f1.nunique().eq(1).all()"),
    code("for target in ['paid_late','is_fraud']:\n    q=paired.xs(target,level='target')\n    interaction=(q['S5']-q['S4'])-(q['S3']-q['S0'])\n    print(target,'context-sample SD: S0',round(q['S0'].std(),4),'S3',round(q['S3'].std(),4),'cleaning × semantics interaction',interaction.round(4).tolist())"),
    code("from itertools import product\nlate=(paired.xs('paid_late',level='target')['S3']-paired.xs('paid_late',level='target')['S0']).to_numpy()\nnull_means=[np.mean(late*np.array(signs)) for signs in product([-1,1],repeat=len(late))]\np_value=np.mean(np.abs(null_means)>=abs(np.mean(late))-1e-12)\nprint('Exact paired sign-flip p value for late-payment S3-S0:',p_value,'(n=3; exploratory)')"),
    md("## Statistical interpretation\nThree paired seeds give a descriptive effect and SD, not a reliable significance claim. Fraud has only eight holdout positives. S0=S4 and S3=S5 by design because cleaning changed no source values. Validation favored S3 for late payment, but its gain reversed on holdout. Any semantic effect is distinct from context selection because every seed uses identical context IDs and size across S0–S5."),
]
notebook("04_semantic_context_engineering.ipynb", SEMANTIC_CELLS)

notebook("05_context_quality_robustness.ipynb", [
    md("# 05 · Context quality and robustness\n\n## tl;dr\nControlled label noise changes context labels only. Duplicate examples replace 20% of a fixed 128-row context; a DACH-only context tests a synthetic regional distribution shift. All use the same final holdout."),
    md("## Methods\nNoise levels are 0%, 5%, 10%, and 20%; flip positions are deterministic by seed. Symmetric flips change the effective fraud class prior because fraud is rare, so the test combines mislabeling and prior shift. Duplicate copies receive unique request IDs while retaining feature/label content. The regional intervention restricts context to DACH; it is a synthetic shift, not a claim about production regions."),
    code("noise=pd.read_csv(ROOT/'data/context_benchmark/noise_results.csv')\nquality=pd.read_csv(ROOT/'data/context_benchmark/quality_results.csv')\ndisplay(noise.groupby(['target','noise_percent'])[['f1','recall','false_positives']].agg(['mean','std']).round(3))\ndisplay(quality.groupby(['target','strategy'])[['f1','recall','false_positives']].agg(['mean','std']).round(3))"),
    code("fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)\nfor ax,target in zip(axes,['paid_late','is_fraud']):\n    sub=noise[noise.target==target].groupby('noise_percent').f1.agg(['mean','std'])\n    ax.errorbar(sub.index,sub['mean'],yerr=sub['std'].fillna(0),marker='o',capsize=2,color='#FFD53C'); ax.set_title(target); ax.set_xlabel('Context labels flipped (%)'); ax.grid(alpha=.2)\naxes[0].set_ylabel('F1 (mean ± seed SD)'); plt.tight_layout(); plt.show()"),
    md("## Takeaways\nThese interventions change different aspects of context and should be interpreted separately. In particular, regional restriction changes the context population; it does not test semantic understanding.")
])

notebook("06_final_context_optimization_benchmark.ipynb", [
    md("# 06 · Final context optimization benchmark\n\n## tl;dr\nUse validation mean F1 to select the global 128-row strategy and the 512-row semantic representation separately. Compare each winner with its matched baseline on the untouched holdout. Do not merge the two interventions into a new claim without a combined validation experiment."),
    md("## Methods\nFive seeds for global selection and three for semantic validation. The final holdout has 128 rows and eight fraud positives. Selection validation uses rows 897–1024; final holdout labels never choose a winner. Relative improvement is undefined when baseline F1 is zero, so report absolute change. Semantic comparisons use a different 512-row budget and are shown separately."),
    code("validation=pd.read_csv(ROOT/'data/context_benchmark/selection_validation_results.csv')\nselection=pd.read_csv(ROOT/'data/context_benchmark/selection_holdout_results.csv')\nsize=pd.read_csv(ROOT/'data/context_benchmark/size_results.csv')\nchosen={k:v[1] for k,v in validation.groupby(['target','strategy']).f1.mean().groupby(level=0).idxmax().items()}\nrows=[]\nfor target in ['paid_late','is_fraud']:\n    for label,frame in [('XGBoost, 128',size[(size.model=='XGBoost')&(size['size']==128)]),('SAP RPT, random',selection[selection.strategy=='random']),('SAP RPT, validated',selection[selection.strategy==chosen[target]])]:\n        sub=frame[frame.target==target]; rows.append({'target':target,'configuration':label,'F1 mean':sub.f1.mean(),'F1 SD':sub.f1.std(),'recall mean':sub.recall.mean(),'false positives mean':sub.false_positives.mean(),'inference seconds mean':sub.inference_seconds.mean()})\nsummary=pd.DataFrame(rows); display(summary.round(3))"),
    code("fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)\nfor ax,target in zip(axes,['paid_late','is_fraud']):\n    sub=summary[summary.target==target]; ax.bar(sub.configuration,sub['F1 mean'],yerr=sub['F1 SD'],capsize=2,color=['#FFD53C','#353C54','#0F37FF']); ax.tick_params(axis='x',rotation=35); ax.set_title(target); ax.grid(axis='y',alpha=.2)\naxes[0].set_ylabel('Holdout F1 (mean ± seed SD)'); plt.tight_layout(); plt.show()"),
    code("sem_val=pd.read_csv(ROOT/'data/semantic_ablation/validation_results.csv')\nsem_final=pd.read_csv(ROOT/'data/semantic_ablation/results.csv')\nsemantic_choice={target: sub.groupby('condition').f1.mean().idxmax() for target,sub in sem_val.groupby('target')}\nprint('Validation-selected semantic representation:',semantic_choice)\nsemantic_summary=sem_final[(sem_final.model=='SAP-RPT') & (sem_final.condition.isin(['S0',*semantic_choice.values()]))].groupby(['target','condition'])[['f1','recall']].agg(['mean','std'])\ndisplay(semantic_summary.round(3))"),
    code("pred=pd.read_csv(ROOT/'data/context_benchmark/selection_holdout_predictions.csv')\nfor target in ['paid_late','is_fraud']:\n    for strategy in ['random',chosen[target]]:\n        run=f'selection_holdout|holdout|{strategy}|128|11|{target}|0|SAP-RPT'\n        sub=pred[pred.run_id==run]\n        errors=sub[sub.actual!=sub.predicted]\n        print(target,strategy,'seed 11 errors',len(errors),'false negatives',((errors.actual==1)&(errors.predicted==0)).sum(),'false positives',((errors.actual==0)&(errors.predicted==1)).sum())"),
    md("## Takeaways\nClass-balanced context was selected on validation for both targets, with more fraud recall and many more false alarms on holdout. Validation selected S3 semantics for late payment, but its paired holdout F1 was lower than S0; all semantic variants tied for fraud on validation and S0 was retained. A combined balanced-plus-semantic condition was not validated, so this notebook makes no interaction claim for that combination. This is one synthetic dataset and one deployment, not a general model ranking.")
])

notebook("05_research_summary.ipynb", [
    md("# Research summary · SAP RPT context and semantics\n\n## Does Semantic Context Improve SAP-RPT Predictions?\nAcross three paired 512-row contexts, enhanced names and labels lowered late-payment F1 by about 0.022; fraud F1 stayed at zero. S4 and S5 are identity controls because the source needs no documented value repair. Validation favored S3 for late payment, but that gain reversed on final holdout. A separate validation-selected class-balanced context had a much larger effect on fraud recall, while adding false alarms."),
    code("semantic=pd.read_csv(ROOT/'data/semantic_ablation/results.csv')\nselection=pd.read_csv(ROOT/'data/context_benchmark/selection_holdout_results.csv')\nvalidation=pd.read_csv(ROOT/'data/context_benchmark/selection_validation_results.csv')\npaired=semantic[semantic.model=='SAP-RPT'].pivot_table(index=['seed','target'],columns='condition',values='f1')\ndisplay((paired[['S1','S2','S3','S4','S5']].sub(paired['S0'],axis=0)).round(3))\ndisplay(selection.groupby(['target','strategy'])[['f1','recall','false_positives']].mean().round(3))"),
    code("fig,axes=plt.subplots(1,2,figsize=(10,4))\nfor ax,target in zip(axes,['paid_late','is_fraud']):\n    sub=semantic[(semantic.model=='SAP-RPT')&(semantic.target==target)].groupby('condition').f1.mean()\n    ax.bar(sub.index,sub,color='#0F37FF'); ax.set_title(target); ax.set_ylabel('Mean holdout F1'); ax.grid(axis='y',alpha=.2)\nplt.tight_layout();plt.show()"),
    md("## Interpretation\nThe original names are already descriptive and no text features exist. S4 and S5 are identity controls because there are no documented data-quality defects to repair. The semantic comparisons use three paired contexts, so interpret their differences as exploratory. Context selection is a separate intervention; use the validation-selected strategy only for final claims. See the Markdown report for exact results and limitations.")
])

notebook("06_semantic_enrichment_ablation.ipynb", SEMANTIC_CELLS + [
    md("## Does Semantic Context Improve SAP-RPT Predictions?\nThe paired table and plot above are the actual SAP deployment measurements. Descriptive names and category labels did not produce a robust positive effect here. The cleaning controls are identical to raw data, so they cannot quantify a cleaning gain. Context selection was evaluated independently in notebooks 02–03. **SAP TechEd 2026 conclusion:** the available evidence favors careful context composition over a claim that renaming these already descriptive invoice fields improves prediction quality.")
])

print("Created eight notebook artifacts")
