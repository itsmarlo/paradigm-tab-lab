"""Validate saved context and semantic benchmark results against the frozen input."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "data/context_benchmark"
SEMANTIC = ROOT / "data/semantic_ablation"


def main():
    source_path = SEMANTIC / "source_snapshot.csv"
    source = pd.read_csv(source_path)
    assert len(source) == 1152 and source.invoice_id.is_unique
    import hashlib
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    for path in [*CONTEXT.glob("*_protocol.json"), *SEMANTIC.glob("*protocol.json")]:
        protocol = json.loads(path.read_text())
        assert protocol["source_sha256"] == source_hash, path
        assert protocol.get("sap_model_configured") == "sap-rpt-1.6-large", path
    final = source.iloc[1024:].set_index("invoice_id")
    validation = source.iloc[896:1024].set_index("invoice_id")
    for name in ("size", "selection_validation", "selection_holdout", "noise", "quality"):
        results = pd.read_csv(CONTEXT / f"{name}_results.csv")
        predictions = pd.read_csv(CONTEXT / f"{name}_predictions.csv")
        assert results.run_id.is_unique
        assert set(results.run_id) == set(predictions.run_id)
        expected = validation if name == "selection_validation" else final
        for run_id, frame in predictions.groupby("run_id"):
            target = run_id.split("|")[5]
            assert len(frame) == 128 and set(frame.invoice_id) == set(expected.index), run_id
            assert frame.set_index("invoice_id").actual.eq(expected.loc[frame.invoice_id, target].to_numpy()).all(), run_id
    semantic = pd.read_csv(SEMANTIC / "predictions.csv")
    for (_, _, _, target), frame in semantic.groupby(["model", "condition", "seed", "target"]):
        assert len(frame) == 128 and set(frame.invoice_id) == set(final.index)
        assert frame.set_index("invoice_id").actual.eq(final.loc[frame.invoice_id, target].to_numpy()).all()
    scored = pd.read_csv(SEMANTIC / "results.csv")
    pivot = scored[scored.model == "SAP-RPT"].pivot_table(index=["seed", "target"], columns="condition", values="f1")
    assert pivot["S0"].equals(pivot["S4"]) and pivot["S3"].equals(pivot["S5"])
    xgb = scored[scored.model == "XGBoost"]
    assert xgb.groupby(["seed", "target"]).f1.nunique().eq(1).all()
    for name in ("validation_predictions.csv", "feature_selection_predictions.csv"):
        frame = pd.read_csv(SEMANTIC / name)
        expected = validation if name.startswith("validation") else final
        for target, subset in frame.groupby("target"):
            assert subset.groupby(["seed", "condition"]).invoice_id.nunique().eq(128).all()
            assert subset.set_index("invoice_id").actual.eq(expected.loc[subset.invoice_id, target].to_numpy()).all()
    print("Saved results, holdout IDs, labels, source hashes, and semantic controls verified")


if __name__ == "__main__":
    main()
