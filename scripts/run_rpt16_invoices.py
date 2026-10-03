"""Submit the shared invoice data to SAP-RPT-1.6 and score its response."""

import argparse
import csv
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from sklearn.metrics import accuracy_score, precision_score, recall_score

from generate_synthetic_invoices import CONTEXT_ROWS, TARGETS


DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic_invoices"


def load_response(path):
    if path:
        return json.loads(path.read_text(encoding="utf-8"))
    url = os.environ.get("SAP_RPT_DEPLOYMENT_URL")
    token = os.environ.get("SAP_RPT_AUTH_TOKEN")
    if not url or not token:
        raise SystemExit(
            "Set SAP_RPT_DEPLOYMENT_URL and SAP_RPT_AUTH_TOKEN, "
            "or pass --response with an exported RPT JSON response."
        )
    request = Request(
        url.rstrip("/") + "/predict",
        data=(DATA / "rpt16_request.json").read_bytes(),
        headers={
            "Authorization": f"Bearer {token}",
            "AI-Resource-Group": os.environ.get("SAP_RPT_RESOURCE_GROUP", "default"),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, help="Use an exported RPT JSON response")
    args = parser.parse_args()
    score_response(load_response(args.response))


if __name__ == "__main__":
    main()
