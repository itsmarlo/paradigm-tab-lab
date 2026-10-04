"""Local HTTP API for the shared invoice prediction experiment."""

import asyncio
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from generate_synthetic_invoices import CONTEXT_ROWS, PREDICTION_ROWS, TARGETS  # noqa: E402
from run_rpt16_invoices import has_rpt_configuration, run_experiment  # noqa: E402
from run_xgboost_invoices import DATA, load_split, predict_frame, train_models  # noqa: E402


app = FastAPI(
    title="Payment behavior experiment API",
    description="Local demonstration API using one synthetic invoice dataset. Fraud flags are illustrative.",
    version="1.0.0",
)


class InvoiceFeatures(BaseModel):
    invoice_amount_eur: float = Field(gt=0)
    payment_terms_days: int = Field(ge=1, le=365)
    customer_tenure_months: int = Field(ge=0, le=600)
    prior_late_payment_rate: float = Field(ge=0, le=1)
    open_invoice_count: int = Field(ge=0, le=100)
    customer_segment: Literal["small", "midmarket", "enterprise"]
    region: Literal["DACH", "Nordics", "UK", "Southern Europe"]
    billing_address_mismatch: int = Field(ge=0, le=1)
    bank_account_changed_recently: int = Field(ge=0, le=1)
    weekend_submission: int = Field(ge=0, le=1)


class TargetPrediction(BaseModel):
    predicted: int
    probability: float


class PredictionResponse(BaseModel):
    model: Literal["xgboost"]
    paid_late: TargetPrediction
    is_fraud: TargetPrediction


class TargetMetrics(BaseModel):
    testRows: int
    positiveCases: int
    accuracy: float
    precision: float
    recall: float
    rocAuc: float | None = None


class ExperimentResponse(BaseModel):
    dataset: str
    trainingRows: int
    testRows: int
    targets: list[str]
    xgboost: dict[str, TargetMetrics]
    rpt16: dict[str, TargetMetrics] | None
    rptStatus: Literal["completed", "not_run"]
    samplePredictions: list[dict]


@lru_cache(maxsize=1)
def _fitted_models(_dataset_modified_ns):
    train, _ = load_split()
    return train_models(train)


def fitted_models():
    return _fitted_models((DATA / "payment_behavior.csv").stat().st_mtime_ns)


def model_metrics(frame, include_auc):
    metrics = {}
    for target in TARGETS:
        actual = frame[f"actual_{target}"]
        predicted = frame[f"predicted_{target}"]
        result = {
            "testRows": len(frame),
            "positiveCases": int(actual.sum()),
            "accuracy": round(accuracy_score(actual, predicted), 3),
            "precision": round(precision_score(actual, predicted, zero_division=0), 3),
            "recall": round(recall_score(actual, predicted, zero_division=0), 3),
        }
        if include_auc:
            result["rocAuc"] = round(roc_auc_score(actual, frame[f"probability_{target}"]), 3)
        metrics[target] = result
    return metrics


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "rptConfigured": has_rpt_configuration(),
    }


@app.get("/api/experiment", response_model=ExperimentResponse)
def experiment():
    train, test = load_split()
    models, columns = fitted_models()
    xgboost_results = predict_frame(models, columns, test)
    rpt_path = DATA / "rpt16_results.csv"
    metadata_path = DATA / "rpt16_run_metadata.json"
    request_path = DATA / "rpt16_request.json"
    rpt_metrics = None
    if rpt_path.exists() and metadata_path.exists() and request_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rpt_results = pd.read_csv(rpt_path)
        if (
            metadata.get("source") == "btp_deployment"
            and metadata.get("request_sha256") == hashlib.sha256(request_path.read_bytes()).hexdigest()
            and metadata.get("results_sha256") == hashlib.sha256(rpt_path.read_bytes()).hexdigest()
            and rpt_results["invoice_id"].tolist() == test["invoice_id"].tolist()
            and all(
                rpt_results[f"actual_{target}"].tolist() == test[target].tolist()
                for target in TARGETS
            )
        ):
            rpt_metrics = model_metrics(rpt_results, include_auc=False)
    return {
        "dataset": "data/synthetic_invoices/payment_behavior.csv",
        "trainingRows": len(train),
        "testRows": len(test),
        "targets": list(TARGETS),
        "xgboost": model_metrics(xgboost_results, include_auc=True),
        "rpt16": rpt_metrics,
        "rptStatus": "completed" if rpt_metrics else "not_run",
        "samplePredictions": xgboost_results[
            ["invoice_id", "predicted_paid_late", "probability_paid_late", "predicted_is_fraud", "probability_is_fraud"]
        ].head(5).to_dict(orient="records"),
    }


@app.post("/api/predictions/xgboost", response_model=PredictionResponse)
def predict_xgboost(invoice: InvoiceFeatures):
    models, columns = fitted_models()
    results = predict_frame(models, columns, pd.DataFrame([invoice.model_dump()]))
    row = results.iloc[0]
    return PredictionResponse(
        model="xgboost",
        **{
            target: TargetPrediction(
                predicted=int(row[f"predicted_{target}"]),
                probability=float(row[f"probability_{target}"]),
            )
            for target in TARGETS
        },
    )


@app.post("/api/experiment/rpt", status_code=200)
async def run_rpt_experiment():
    """Run SAP-RPT on the fixed test split; each call makes a paid SAP inference request."""
    if not has_rpt_configuration():
        raise HTTPException(
            status_code=503,
            detail={"code": "RPT_NOT_CONFIGURED", "message": "Configure direct RPT credentials or SAP AI Core service-key settings."},
        )
    try:
        await asyncio.to_thread(run_experiment)
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(
            status_code=502,
            detail={"code": "INVALID_RPT_RESPONSE", "message": str(error)},
        ) from error
    except OSError as error:
        raise HTTPException(
            status_code=502,
            detail={"code": "RPT_REQUEST_FAILED", "message": str(error)},
        ) from error
    result = pd.read_csv(DATA / "rpt16_results.csv")
    return {"rpt16": model_metrics(result, include_auc=False), "testRows": PREDICTION_ROWS}
