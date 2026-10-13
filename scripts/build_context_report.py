"""Create source-backed benchmark figures and Markdown from saved run tables."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/context_benchmark"
SEMANTIC = ROOT / "data/semantic_ablation"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def table(path):
    return pd.read_csv(path)


def main():
    REPORTS.mkdir(exist_ok=True)
    size = table(DATA / "size_results.csv")
    validation = table(DATA / "selection_validation_results.csv")
    selection = table(DATA / "selection_holdout_results.csv")
    noise = table(DATA / "noise_results.csv")
    quality = table(DATA / "quality_results.csv")
    semantics = table(SEMANTIC / "results.csv")
    semantic_validation = table(SEMANTIC / "validation_results.csv")
    feature_selection = table(SEMANTIC / "feature_selection_results.csv")
    chosen = validation.groupby(["target", "strategy"]).f1.mean().groupby(level=0).idxmax()
    selected = {target: pair[1] for target, pair in chosen.items()}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, target in zip(axes, ("paid_late", "is_fraud")):
        for model, color in (("SAP-RPT", "#136f84"), ("XGBoost", "#b25b22")):
            subset = size[(size.target == target) & (size.model == model)]
            mean = subset.groupby("size").f1.mean()
            std = subset.groupby("size").f1.std().fillna(0)
            ax.errorbar(mean.index, mean, yerr=std, marker="o", capsize=2, label=model, color=color)
        ax.set_xscale("log", base=2)
        ax.set_xticks(list((16, 32, 64, 128, 256, 512, 1024)))
        ax.set_xticklabels(["16", "32", "64", "128", "256", "512", "1024"], rotation=45)
        ax.set_title(target.replace("_", " ").title())
        ax.set_xlabel("Labeled examples")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("F1 (mean ± seed SD)")
    axes[1].legend(frameon=False)
    fig.suptitle("Figure 1 · Predictive performance versus context size")
    save(fig, "01_performance_vs_context_size")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, target in zip(axes, ("paid_late", "is_fraud")):
        subset = selection[selection.target == target]
        agg = subset.groupby("strategy").f1.agg(["mean", "std"])
        ax.bar(agg.index, agg["mean"], yerr=agg["std"].fillna(0), color="#136f84", capsize=3)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(target.replace("_", " ").title())
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("F1 (mean ± seed SD where available)")
    fig.suptitle("Figure 2 · Context selection at 128 labeled examples")
    fig.text(.5, -.08, "Five seeds: global strategies. One seed: similarity and hybrid (exploratory).", ha="center", fontsize=9)
    save(fig, "02_selection_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, target in zip(axes, ("paid_late", "is_fraud")):
        sub = selection[(selection.target == target) & selection.strategy.isin(["random", selected[target]])]
        agg = sub.groupby("strategy").f1.agg(["mean", "std"])
        ax.bar(agg.index, agg["mean"], yerr=agg["std"].fillna(0), color=["#136f84", "#b25b22"], capsize=3)
        ax.set_title(target.replace("_", " ").title())
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Holdout F1 (mean ± seed SD)")
    fig.suptitle("Figure 3 · Random versus validation-selected context")
    save(fig, "03_random_vs_validated")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, target in zip(axes, ("paid_late", "is_fraud")):
        sub = semantics[(semantics.target == target) & (semantics.model == "SAP-RPT")]
        agg = sub.groupby("condition").f1.agg(["mean", "std"])
        ax.bar(agg.index, agg["mean"], yerr=agg["std"].fillna(0), color="#136f84", capsize=3)
        ax.set_title(target.replace("_", " ").title())
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("F1 (mean ± seed SD)")
    fig.suptitle("Figure 4 · Semantic representation ablation, fixed 512-row contexts")
    save(fig, "04_semantic_ablation")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    predictions = table(DATA / "selection_holdout_predictions.csv")
    size_predictions = table(DATA / "size_predictions.csv")
    cases = [
        ("XGBoost, random", size_predictions, "size|holdout|random|128|11|is_fraud|0|XGBoost"),
        ("SAP-RPT, random", predictions, "selection_holdout|holdout|random|128|11|is_fraud|0|SAP-RPT"),
        (f"SAP-RPT, {selected['is_fraud']}", predictions,
         f"selection_holdout|holdout|{selected['is_fraud']}|128|11|is_fraud|0|SAP-RPT"),
    ]
    for ax, (title, frame, run_id) in zip(axes, cases):
        sub = frame[frame.run_id == run_id]
        matrix = confusion_matrix(sub.actual, sub.predicted, labels=[0, 1])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=128)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("Predicted fraud"); ax.set_ylabel("Actual fraud")
        ax.set_title(title)
    fig.suptitle("Figure 5 · Fraud confusion matrices, fixed holdout, seed 11")
    save(fig, "05_fraud_confusion")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, target in zip(axes, ("paid_late", "is_fraud")):
        sub = size[(size.target == target) & (size.model == "SAP-RPT")]
        avg = sub.groupby("size")[["inference_seconds", "f1"]].mean()
        ax.plot(avg.inference_seconds, avg.f1, "o-", color="#136f84")
        for budget, row in avg.iterrows():
            ax.annotate(str(budget), (row.inference_seconds, row.f1), xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("SAP request latency (s)")
        ax.set_ylabel("F1")
        ax.set_title(target.replace("_", " ").title())
        ax.grid(alpha=.2)
    fig.suptitle("Figure 6 · Performance and inference cost by context budget")
    save(fig, "06_performance_vs_cost")

    def mean(frame, target, strategy, metric="f1"):
        return frame[(frame.target == target) & (frame.strategy == strategy)][metric].mean()

    sem = semantics[semantics.model == "SAP-RPT"]
    paired = sem.pivot_table(index=["seed", "target"], columns="condition", values="f1")
    late_delta = (paired.xs("paid_late", level="target")["S3"] - paired.xs("paid_late", level="target")["S0"])
    fraud_delta = (paired.xs("is_fraud", level="target")["S3"] - paired.xs("is_fraud", level="target")["S0"])
    semantic_selected = {target: frame.groupby("condition").f1.mean().idxmax() for target, frame in semantic_validation.groupby("target")}
    fraud_recall = mean(selection, "is_fraud", selected["is_fraud"], "recall")
    fraud_fp = mean(selection, "is_fraud", selected["is_fraud"], "false_positives")
    selection_paired = selection.pivot_table(index=["seed", "target"], columns="strategy", values="f1")
    paired_late = selection_paired.xs("paid_late", level="target")[selected["paid_late"]] - selection_paired.xs("paid_late", level="target")["random"]
    paired_fraud = selection_paired.xs("is_fraud", level="target")[selected["is_fraud"]] - selection_paired.xs("is_fraud", level="target")["random"]
    quality_note = (f"At 20% context-label noise, late-payment F1 was {noise[(noise.target == 'paid_late') & (noise.noise_percent == 20)].f1.mean():.3f} versus {noise[(noise.target == 'paid_late') & (noise.noise_percent == 0)].f1.mean():.3f} at 0%. "
        f"Replacing 20% of examples with duplicates gave late-payment F1 {mean(quality, 'paid_late', 'duplicate'):.3f}; the DACH-only regional shift gave {mean(quality, 'paid_late', 'region_shift'):.3f}. "
        "Fraud F1 stayed at zero for these random-context robustness controls, so they do not establish fraud robustness.")
    report = f"""# SAP RPT Context Engineering Benchmark

## Research motivation and dataset

Can SAP RPT-1.6 improve without changing model weights by choosing better labeled context? This benchmark uses the frozen 1,152-row synthetic invoice snapshot in `data/semantic_ablation/source_snapshot.csv` (1,024 eligible context records, 128 final holdout). The current working CSV contains user-modified fraud labels; older published scores refer to another label version and are excluded here. The final holdout has 8 fraud positives.

## Hypotheses and protocol

We tested context budgets 16–1,024, random/stratified/balanced/diverse selection, query-specific similarity and hybrid selection where saved, context label noise, and a paired semantic ablation. Five seeds define the main context comparisons. The first 896 context rows are eligible for validation selection, the next 128 form validation, and the final 128 remain the fixed holdout. For the 1,024-row budget, all eligible context rows are used; its SAP repeats have identical input and are not independent. All SAP requests use the configured deployment; the protocol JSON records its configured model label and source checksum. RPT returns class labels, so ROC-AUC and PR-AUC are unavailable.

## Context selection strategies and implementation

`src/context_engineering/selection.py` implements random, stratified, class-balanced, mixed-type nearest-neighbor similarity, greedy farthest-point diversity, and a hybrid of nearest, minority and diverse rows. Numeric scaling and categorical encoding fit only on eligible context rows. Query-specific strategies never inspect query labels. The runner saves row-level predictions, metrics and per-query checkpoints. A serial request rate limits SAP calls. Similarity and hybrid each completed all 128 holdout invoices with seed 11. They were not evaluated on validation and therefore cannot be declared the selected winner.

## Actual findings

At 128 labeled examples, random-context SAP RPT had mean late-payment F1 {mean(selection, 'paid_late', 'random'):.3f} and fraud F1 {mean(selection, 'is_fraud', 'random'):.3f}. Validation chose **{selected['paid_late']}** for late payment and **{selected['is_fraud']}** for fraud by mean F1. Their final-holdout F1 values were {mean(selection, 'paid_late', selected['paid_late']):.3f} and {mean(selection, 'is_fraud', selected['is_fraud']):.3f}. For fraud, the selected strategy reached recall {fraud_recall:.3f} with {fraud_fp:.1f} false positives on average per 128 invoices. This is a sensitivity/precision tradeoff, not unqualified superiority.

At this budget, balanced fraud context included all 53 available positive examples plus 75 negatives; random context contained 3–8 positives across seeds. The higher recall therefore reflects much greater minority-label exposure and a changed effective class prior at the same total budget.

Paired selected-minus-random F1 across the same five seeds was {paired_late.mean():+.3f} (SD {paired_late.std():.3f}) for late payment and {paired_fraud.mean():+.3f} (SD {paired_fraud.std():.3f}) for fraud. These seed-level differences are descriptive because both the validation and holdout contain few fraud positives. Per-seed predictions are retained in the CSV files.

Exploratory one-seed query-specific results: similarity late-payment F1 {selection[(selection.target == 'paid_late') & (selection.strategy == 'similarity')].f1.iloc[0]:.3f}, hybrid late-payment F1 {selection[(selection.target == 'paid_late') & (selection.strategy == 'hybrid')].f1.iloc[0]:.3f}; both had fraud F1 0.000. Hybrid's late-payment result came with {int(selection[(selection.target == 'paid_late') & (selection.strategy == 'hybrid')].false_positives.iloc[0])} false positives. These methods require 128 SAP requests per target and were not validation-selected.

XGBoost class-1 probabilities support ROC-AUC and average precision; at the 128-row budget, its mean fraud ROC-AUC was {size[(size.model == 'XGBoost') & (size.target == 'is_fraud') & (size['size'] == 128)].roc_auc.mean():.3f} and mean average precision {size[(size.model == 'XGBoost') & (size.target == 'is_fraud') & (size['size'] == 128)].average_precision.mean():.3f}. SAP RPT returned only a top class in these requests, so no directly comparable probability metric was calculated.

{quality_note}

The noise intervention flips a fixed fraction of context labels in both directions. Because fraud is rare, this also changes the observed context class prior; a pure label-corruption effect cannot be isolated from that prior shift here.

The fixed 512-row semantic ablation found S3−S0 paired late-payment F1 {late_delta.mean():+.3f} (seed SD {late_delta.std():.3f}) and fraud F1 {fraud_delta.mean():+.3f}. S4 is identical to S0 because the source has no missing values, duplicate IDs or invalid numeric/category values; S5 is identical to S3. Thus cleaning has no measured effect in this dataset. Semantic effects cannot be generalized from three context samples and eight fraud positives. The semantic experiment changed names and categorical strings only; XGBoost used an invariant encoded matrix.

On the internal validation slice, S3 had the highest mean late-payment F1 ({semantic_validation[(semantic_validation.target == 'paid_late') & (semantic_validation.condition == 'S3')].f1.mean():.3f} versus S0 {semantic_validation[(semantic_validation.target == 'paid_late') & (semantic_validation.condition == 'S0')].f1.mean():.3f}); all four semantic conditions tied for fraud, so S0 was retained. On the untouched final holdout, validation-selected S3 had mean late-payment F1 {sem[(sem.target == 'paid_late') & (sem.condition == 'S3')].f1.mean():.3f} versus S0 {sem[(sem.target == 'paid_late') & (sem.condition == 'S0')].f1.mean():.3f}. This reversal is evidence that the apparent validation gain did not transfer in this small study.

A separate matched single-target feature-selection test removed `weekend_submission` from late-payment requests and `region` from fraud requests because those fields are absent from the corresponding generator formulas. Late-payment F1 was {feature_selection[(feature_selection.target == 'paid_late') & (feature_selection.condition == 'feature_removed')].f1.mean():.3f} versus {feature_selection[(feature_selection.target == 'paid_late') & (feature_selection.condition == 'single_target_baseline')].f1.mean():.3f}; fraud F1 stayed at zero. Feature removal was not part of S0–S5 and does not explain their semantic differences.

The S3−S0 cleaning interaction is exactly zero because S4/S5 are identity controls. Context-sample stability should be read from the per-seed SD, not from repeated calls with identical 1,024-row input. The exact two-sided paired sign-flip p value for the three late-payment semantic differences is 0.25; this does not support a significance claim.

![Context size](figures/01_performance_vs_context_size.png)

![Selection](figures/02_selection_comparison.png)

![Random versus validated](figures/03_random_vs_validated.png)

![Semantic ablation](figures/04_semantic_ablation.png)

![Fraud confusion matrices](figures/05_fraud_confusion.png)

![Performance and cost](figures/06_performance_vs_cost.png)

## Does Semantic Context Improve SAP-RPT Predictions?

On this synthetic dataset, more descriptive names and labels did not improve the measured late-payment result: S3−S0 F1 was {late_delta.mean():+.3f} across three paired contexts. Fraud F1 stayed at zero for S0–S5, so the study cannot show a fraud benefit. Classical XGBoost predictions are identical across semantic conditions by construction. Cleaning made no change to the already complete values. Context selection had a larger measured effect, especially for fraud, while raising false alarms. **TechEd conclusion:** choose and validate labeled context before claiming a semantic gain; this experiment provides no evidence that renaming features improves SAP RPT on these invoices.

## Limitations and follow-up

The labels and features are synthetic; there are only eight holdout fraud cases. The original feature names were already descriptive and there are no free-text fields. The model label comes from configuration and should be checked against SAP deployment metadata before public version claims. Similarity and hybrid each have only one seed because per-query requests encounter API rate limits. Query-specific elapsed times include serial request pacing and any retry delays, so they are end-to-end costs rather than pure model latency. A larger external dataset, more seeds, class-score API support and calibration would strengthen the conclusions. Holdout comparisons of multiple strategies are descriptive; only the validation-selected strategy is eligible for a preselected final claim.

## Reproducibility

Install `requirements-experiment.txt`, then run `python -m pytest -q tests/test_context_selection.py` and `python scripts/validate_context_benchmark.py`. The runners default to dry run: `python scripts/run_context_benchmark.py size` and `python scripts/run_semantic_ablation.py`. Add `--live` only with configured SAP credentials. Then run `python scripts/build_context_report.py`. All published tables live under `data/context_benchmark/` and `data/semantic_ablation/`; PNG and SVG charts live in `reports/figures/`.
"""
    (REPORTS / "context_engineering_benchmark.md").write_text(report)
    key = f"""# SAP TechEd 2026: key findings

- **Research question:** Can context composition improve SAP RPT predictions without retraining?
- **Baseline:** At 128 random examples, mean SAP RPT F1 was {mean(selection, 'paid_late', 'random'):.3f} for late payment and {mean(selection, 'is_fraud', 'random'):.3f} for fraud.
- **Intervention:** Validation selected class-balanced context for both tasks. The fraud context used all 53 eligible positives in the final 1,024-row pool, versus 3–8 positives under random selection.
- **Observed result:** Final-holdout F1 was {mean(selection, 'paid_late', selected['paid_late']):.3f} for late payment and {mean(selection, 'is_fraud', selected['is_fraud']):.3f} for fraud; fraud recall was {fraud_recall:.3f} with {fraud_fp:.1f} false positives per 128 invoices.
- **Interpretation:** Context composition changed the sensitivity/false-alarm tradeoff. Semantic enrichment gave S3−S0 late-payment F1 {late_delta.mean():+.3f}; cleaning was an identity transformation.
- **Limitation:** Synthetic data, eight holdout fraud positives, three semantic seeds, and one seed for query-specific selection.
- **Slide takeaway:** Better context can recover missed fraud cases in this setup, but the added alerts matter; descriptive feature names did not improve measured performance.
"""
    (REPORTS / "teched_2026_key_findings.md").write_text(key)
    print("Saved six figures and two reports")


if __name__ == "__main__":
    main()
