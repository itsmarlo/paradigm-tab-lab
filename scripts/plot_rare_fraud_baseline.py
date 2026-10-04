"""Plot the archived rare-fraud experiment, including its Playground export."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "archive" / "synthetic_invoice_experiments" / "rare_fraud_baseline"
OUTPUT = DATA / "model_tradeoff_with_playground.png"

xgb = pd.read_csv(DATA / "xgboost_results.csv").sort_values("invoice_id").reset_index(drop=True)
rpt = pd.read_csv(DATA / "rpt16_results.csv").sort_values("invoice_id").reset_index(drop=True)
playground = pd.read_csv(DATA / "exports" / "rpt_playground_export.csv")
playground = playground.tail(128).sort_values("invoice_id").reset_index(drop=True)
playground = playground.rename(columns={
    "paid_late": "predicted_paid_late", "is_fraud": "predicted_is_fraud"
})
assert len(xgb) == len(rpt) == 128
assert xgb.invoice_id.is_unique and rpt.invoice_id.is_unique
assert xgb.invoice_id.equals(rpt.invoice_id)
assert xgb.invoice_id.equals(playground.invoice_id)
for target in ("paid_late", "is_fraud"):
    assert xgb[f"actual_{target}"].equals(rpt[f"actual_{target}"])
    playground[f"actual_{target}"] = xgb[f"actual_{target}"]

models = {"XGBoost": xgb, "RPT BTP 1.6": rpt, "Playground export": playground}
colors = {"XGBoost": "#1769AA", "RPT BTP 1.6": "#D47928", "Playground export": "#008D83"}

fig, (late_ax, fraud_ax) = plt.subplots(
    1, 2, figsize=(13.6, 6.8), gridspec_kw={"width_ratios": [1.08, 1]}
)
fig.patch.set_facecolor("white")
fig.suptitle("Rare-fraud baseline: XGBoost, RPT and Playground", x=0.055, y=0.955, ha="left", fontsize=21, weight="bold")
fig.text(0.055, 0.892, "Same 128 synthetic test invoices, two prediction tasks", fontsize=12, color="#526174")

metric_functions = {
    "Accuracy": accuracy_score,
    "Precision": lambda y, p: precision_score(y, p, zero_division=0),
    "Recall": lambda y, p: recall_score(y, p, zero_division=0),
}
positions = np.arange(3)
late_ax.set_title("Paid late: both RPT runs score higher", loc="left", pad=19, fontsize=16, weight="bold")
for offset, (name, data) in zip((-0.26, 0, 0.26), models.items()):
    scores = [fn(data.actual_paid_late, data.predicted_paid_late) for fn in metric_functions.values()]
    bars = late_ax.barh(positions + offset, scores, height=0.24, color=colors[name], label=name)
    for bar, score in zip(bars, scores):
        late_ax.text(score + 0.018, bar.get_y() + bar.get_height() / 2,
                     f"{score:.1%}", va="center", fontsize=11, weight="bold", color="#243247")
late_ax.set_yticks(positions, metric_functions.keys(), fontsize=12)
late_ax.invert_yaxis()
late_ax.set_xlim(0, 0.91)
late_ax.set_xticks(np.arange(0, 0.81, 0.2), ["0%", "20%", "40%", "60%", "80%"])
late_ax.tick_params(axis="x", colors="#697789", labelsize=10, length=0)
late_ax.tick_params(axis="y", length=0, pad=9)
late_ax.xaxis.grid(True, color="#E5E9EF")
late_ax.set_axisbelow(True)
late_ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False, fontsize=10)

fraud_ax.set_title("Fraud: XGBoost found 2 of 8 cases", loc="left", pad=19, fontsize=16, weight="bold")
for row, (name, data) in enumerate(models.items()):
    actual = data.actual_is_fraud.eq(1)
    found = int((actual & data.predicted_is_fraud.eq(1)).sum())
    missed = int((actual & data.predicted_is_fraud.eq(0)).sum())
    false_alarms = int((~actual & data.predicted_is_fraud.eq(1)).sum())
    fraud_ax.barh(row, found, height=0.43, color=colors[name])
    fraud_ax.barh(row, missed, left=found, height=0.43, color="#DDE3EA")
    if found:
        fraud_ax.text(found / 2, row, f"{found} found", va="center", ha="center",
                      color="white", fontsize=11, weight="bold")
    else:
        fraud_ax.text(0.13, row - 0.32, "0 found", va="center", ha="left",
                      color=colors[name], fontsize=11, weight="bold")
    fraud_ax.text(found + missed / 2, row, f"{missed} missed", va="center", ha="center",
                  color="#344256", fontsize=11, weight="bold")
    fraud_ax.text(8.27, row, f"{false_alarms} false alarms", va="center",
                  fontsize=11, color="#344256")
fraud_ax.set_yticks(range(len(models)), models.keys(), fontsize=11)
fraud_ax.invert_yaxis()
fraud_ax.set_xlim(0, 11.5)
fraud_ax.set_xticks(range(0, 9, 2))
fraud_ax.tick_params(axis="x", colors="#697789", labelsize=10, length=0)
fraud_ax.tick_params(axis="y", length=0, pad=9)
fraud_ax.set_xlabel("Actual fraud cases (8 total)", fontsize=11, color="#526174", labelpad=10)
fraud_ax.axvline(8, color="#AEB9C6", lw=1)

for ax in (late_ax, fraud_ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.text(0.055, 0.065,
         "Test set: 51 late payments and 8 fraud cases. Playground is a supplied export; its model configuration is unverified.",
         fontsize=10.5, color="#526174")
fig.subplots_adjust(left=0.10, right=0.975, top=0.76, bottom=0.25, wspace=0.34)
fig.savefig(OUTPUT, dpi=220, facecolor="white")
print(OUTPUT)
