"""Plot the fixed-test fraud experiment for a presentation slide."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "synthetic_invoices"
ARCHIVE = ROOT / "archive" / "synthetic_invoice_experiments"
ORIGINAL = ARCHIVE / "rare_fraud_baseline"
ENRICHED = ARCHIVE / "context_only"
OUTPUT = ENRICHED / "fraud_context_enrichment_for_ppt.png"

original_data = pd.read_csv(ORIGINAL / "payment_behavior.csv")
enriched_data = pd.read_csv(ENRICHED / "payment_behavior.csv")
assert len(original_data) == len(enriched_data) == 1152
assert original_data.tail(128).equals(enriched_data.tail(128))
assert original_data.head(1024).drop(columns="is_fraud").equals(
    enriched_data.head(1024).drop(columns="is_fraud")
)

rows = []
for model, filename, color in (
    ("XGBoost", "xgboost_results.csv", "#1769AA"),
    ("SAP RPT 1.6", "rpt16_results.csv", "#D47928"),
):
    for scenario, directory in (("Original", ORIGINAL), ("More fraud labels", ENRICHED)):
        result = pd.read_csv(directory / filename)
        assert result.invoice_id.tolist() == original_data.invoice_id.tail(128).tolist()
        assert result.actual_is_fraud.tolist() == original_data.is_fraud.tail(128).tolist()
        actual = result.actual_is_fraud.eq(1)
        predicted = result.predicted_is_fraud.eq(1)
        rows.append({
            "label": f"{model} · {scenario}",
            "color": color,
            "found": int((actual & predicted).sum()),
            "missed": int((actual & ~predicted).sum()),
            "false_alarms": int((~actual & predicted).sum()),
            "scenario": scenario,
        })

fig, ax = plt.subplots(figsize=(13, 6.1))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
fig.suptitle("More synthetic fraud labels increased detection", x=0.055, y=0.955,
             ha="left", fontsize=23, weight="bold")
fig.text(0.055, 0.882, "Both models tested on the same 8 fraud and 120 non-fraud invoices",
         fontsize=12, color="#526174")

for position, row in enumerate(rows):
    alpha = 0.55 if row["scenario"] == "Original" else 1
    ax.barh(position, row["found"], height=0.53, color=row["color"], alpha=alpha)
    ax.barh(position, row["missed"], left=row["found"], height=0.53, color="#DDE3EA")
    if row["found"]:
        ax.text(row["found"] / 2, position, f"{row['found']} found",
                va="center", ha="center", color="white", fontsize=12, weight="bold")
    else:
        ax.text(0.1, position - 0.32, "0 found", va="center", ha="left",
                color=row["color"], fontsize=11, weight="bold")
    ax.text(row["found"] + row["missed"] / 2, position, f"{row['missed']} missed",
            va="center", ha="center", color="#344256", fontsize=11, weight="bold")
    ax.text(8.4, position, f"{row['false_alarms']} false alarms",
            va="center", ha="left", color="#344256", fontsize=12)

ax.set_yticks(range(len(rows)), [row["label"] for row in rows], fontsize=12)
ax.invert_yaxis()
ax.set_xlim(0, 11.6)
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_xlabel("Actual fraud cases (8 total)", color="#526174", fontsize=11, labelpad=8)
ax.tick_params(axis="x", colors="#697789", labelsize=10, length=0)
ax.tick_params(axis="y", length=0, pad=12)
ax.axvline(8, color="#AEB9C6", lw=1)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.text(0.055, 0.055,
         "Context fraud labels: 53 → 187 of 1,024. Test invoices and labels stayed fixed. More detections came with more false alarms.",
         fontsize=10.5, color="#526174")
fig.subplots_adjust(left=0.26, right=0.965, top=0.77, bottom=0.2)
fig.savefig(OUTPUT, dpi=220, facecolor="white")
print(OUTPUT)
