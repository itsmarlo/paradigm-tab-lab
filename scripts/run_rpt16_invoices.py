"""Submit the shared invoice data to SAP-RPT-1.6 and score its response."""

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score

from generate_synthetic_invoices import CONTEXT_ROWS, TARGETS


DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic_invoices"
ROOT = DATA.parents[1]


def load_local_env():
    """Load local settings without replacing variables already set by the caller."""
    load_dotenv(ROOT / ".env", override=False)


def _https_url(value, name):
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{name} must be an HTTPS URL without embedded credentials")
    return value.rstrip("/")


def has_rpt_configuration():
    load_local_env()
    direct = bool(os.environ.get("SAP_RPT_DEPLOYMENT_URL") and os.environ.get("SAP_RPT_AUTH_TOKEN"))
    service_key = all(os.environ.get(name) for name in (
        "SAP_AI_CORE_API_URL", "SAP_AI_CORE_AUTH_URL", "SAP_AI_CORE_CLIENT_ID",
        "SAP_AI_CORE_CLIENT_SECRET", "SAP_RPT_DEPLOYMENT_ID",
    ))
    return direct or service_key


def connection_settings():
    """Resolve either a direct bearer token or SAP AI Core service-key settings."""
    load_local_env()
    deployment_url = os.environ.get("SAP_RPT_DEPLOYMENT_URL")
    token = os.environ.get("SAP_RPT_AUTH_TOKEN")
    resource_group = os.environ.get("SAP_RPT_RESOURCE_GROUP") or os.environ.get("SAP_AI_CORE_RESOURCE_GROUP", "default")
    model_name = os.environ.get("SAP_RPT_MODEL_NAME")
    if deployment_url and token:
        return {
            "deployment_url": _https_url(deployment_url, "SAP_RPT_DEPLOYMENT_URL"),
            "token": token,
            "resource_group": resource_group,
            "deployment_id": os.environ.get("SAP_RPT_DEPLOYMENT_ID"),
            "deployment_configuration_name": None,
            "model_configured": model_name,
        }

    required = (
        "SAP_AI_CORE_API_URL", "SAP_AI_CORE_AUTH_URL", "SAP_AI_CORE_CLIENT_ID",
        "SAP_AI_CORE_CLIENT_SECRET", "SAP_RPT_DEPLOYMENT_ID",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit("Missing SAP AI Core settings: " + ", ".join(missing))

    auth_url = _https_url(os.environ["SAP_AI_CORE_AUTH_URL"], "SAP_AI_CORE_AUTH_URL")
    auth_body = urlencode({
        "grant_type": "client_credentials",
        "client_id": os.environ["SAP_AI_CORE_CLIENT_ID"],
        "client_secret": os.environ["SAP_AI_CORE_CLIENT_SECRET"],
    }).encode()
    auth_request = Request(
        auth_url + "/oauth/token", data=auth_body,
        headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST",
    )
    with urlopen(auth_request, timeout=30) as response:
        token = json.load(response)["access_token"]

    api_url = _https_url(os.environ["SAP_AI_CORE_API_URL"], "SAP_AI_CORE_API_URL")
    deployment_id = os.environ["SAP_RPT_DEPLOYMENT_ID"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", deployment_id):
        raise ValueError("SAP_RPT_DEPLOYMENT_ID has invalid characters")
    details_request = Request(
        f"{api_url}/v2/lm/deployments/{deployment_id}",
        headers={"Authorization": f"Bearer {token}", "AI-Resource-Group": resource_group},
    )
    with urlopen(details_request, timeout=30) as response:
        details = json.load(response)
    if str(details.get("status", "")).upper() != "RUNNING":
        raise ValueError("The configured SAP AI Core deployment is not RUNNING")
    if str(details.get("id")) != deployment_id:
        raise ValueError("SAP AI Core returned a different deployment ID")
    return {
        "deployment_url": _https_url(details["deploymentUrl"], "deploymentUrl"),
        "token": token,
        "resource_group": resource_group,
        "deployment_id": deployment_id,
        "deployment_configuration_name": details.get("configurationName"),
        "model_configured": model_name,
    }


def load_response(path, connection=None):
    if path:
        return json.loads(path.read_text(encoding="utf-8"))
    connection = connection or connection_settings()
    request = Request(
        connection["deployment_url"] + "/predict",
        data=(DATA / "rpt16_request.json").read_bytes(),
        headers={
            "Authorization": f"Bearer {connection['token']}",
            "AI-Resource-Group": connection["resource_group"],
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    (DATA / "rpt16_response.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def score_response(response):
    predictions = response.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("RPT response has no predictions array")
    by_id = {str(row["invoice_id"]): row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("RPT response contains duplicate invoice IDs")
    with (DATA / "payment_behavior.csv").open(newline="", encoding="utf-8") as file:
        truth = list(csv.DictReader(file))[CONTEXT_ROWS:]
    expected_ids = {row["invoice_id"] for row in truth}
    if set(by_id) != expected_ids:
        raise ValueError("RPT response does not match the held-out invoice IDs")
    results = []
    for source in truth:
        invoice_id = source["invoice_id"]
        output = {"invoice_id": invoice_id}
        for target in TARGETS:
            choices = by_id[invoice_id][target]
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"Missing RPT prediction for {invoice_id} / {target}")
            predicted = int(choices[0]["prediction"])
            if predicted not in (0, 1):
                raise ValueError(f"Invalid RPT class for {invoice_id} / {target}")
            output[f"actual_{target}"] = int(source[target])
            output[f"predicted_{target}"] = predicted
            output[f"confidence_{target}"] = choices[0].get("confidence")
        results.append(output)
    path = DATA / "rpt16_results.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for target in TARGETS:
        actual = [row[f"actual_{target}"] for row in results]
        predicted = [row[f"predicted_{target}"] for row in results]
        print(
            f"{target}: accuracy={accuracy_score(actual, predicted):.3f}, "
            f"precision={precision_score(actual, predicted, zero_division=0):.3f}, "
            f"recall={recall_score(actual, predicted, zero_division=0):.3f}"
        )
    print(f"Predictions: {path}")
    return path


def run_experiment(response_path=None):
    """Score an RPT response and record enough provenance to identify the run."""
    load_local_env()
    if response_path is None and os.environ.get("RPT_DRY_RUN", "").lower() in ("1", "true", "yes", "on"):
        raise RuntimeError("RPT_DRY_RUN is enabled; no BTP inference request was sent")
    connection = connection_settings() if response_path is None else None
    response = load_response(response_path, connection)
    results_path = score_response(response)
    deployment_url = connection["deployment_url"] if connection else ""
    metadata = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "btp_deployment" if response_path is None else "exported_response",
        "model_configured": connection["model_configured"] if connection else None,
        "deployment_id": connection["deployment_id"] if connection else None,
        "deployment_configuration_name": connection["deployment_configuration_name"] if connection else None,
        "deployment_host": urlsplit(deployment_url).hostname if deployment_url else None,
        "deployment_url_sha256": hashlib.sha256(deployment_url.encode()).hexdigest() if deployment_url else None,
        "resource_group": connection["resource_group"] if connection else None,
        "request_sha256": hashlib.sha256((DATA / "rpt16_request.json").read_bytes()).hexdigest(),
        "response_sha256": hashlib.sha256(
            json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "response_id": response.get("id"),
        "test_rows": len(response["predictions"]),
    }
    (DATA / "rpt16_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return results_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, help="Use an exported RPT JSON response")
    args = parser.parse_args()
    run_experiment(args.response)


if __name__ == "__main__":
    main()
