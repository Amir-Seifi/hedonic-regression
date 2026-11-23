"""Check the published split, CV selection and test metrics against the source CSV."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), f"Mismatch: {label}")


def verify(source, results):
    info = json.loads((results / "metrics.json").read_text(encoding="utf-8"))
    raw = None
    if source is not None:
        require(
            hashlib.sha256(source.read_bytes()).hexdigest() == info["source_sha256"], "CSV digest"
        )
        raw = read_rows(source)
        require(len(raw) == info["cleaning"]["rows_raw"], "Raw row count")
    manifest = read_rows(results / "split.csv")
    predictions = read_rows(results / "test_predictions.csv")
    test_ids = set()
    seen = set()
    folds = set(range(1, info["cv_folds"] + 1))
    used_folds = set()
    for row in manifest:
        i = int(row["source_row"])
        require(
            0 <= i < info["cleaning"]["rows_raw"] and i not in seen,
            "Invalid or duplicate source row",
        )
        seen.add(i)
        if row["partition"] == "test":
            require(row["validation_fold"] == "", "Test row assigned to a validation fold")
            test_ids.add(i)
        else:
            require(row["partition"] == "train", "Unknown partition")
            used_folds.add(int(row["validation_fold"]))
    require(used_folds == folds, "Invalid training fold IDs")
    require(len(seen) == info["cleaning"]["rows_clean"], "Clean row count")
    require(len(seen - test_ids) == info["train_rows"], "Training row count")
    require(len(predictions) == len(test_ids) == info["test_rows"], "Test row count")
    require({int(r["source_row"]) for r in predictions} == test_ids, "Test prediction IDs")

    actual, errors = [], []
    for row in predictions:
        price = float(row["actual_bn"])
        if raw is not None:
            close(price, float(raw[int(row["source_row"])]["Price"]) / 1e9, "Raw test price")
        actual.append(price)
        errors.append(float(row["predicted_bn"]) - price)
    negative = sum(float(row["predicted_bn"]) < 0 for row in predictions)
    require(negative == info["test"]["negative_predictions"], "Negative prediction count")
    mse = mean(e * e for e in errors)
    close(mean(abs(e) for e in errors), info["test"]["mae"], "Test MAE")
    close(math.sqrt(mse), info["test"]["rmse"], "Test RMSE")
    mean_price = mean(actual)
    total = sum((y - mean_price) ** 2 for y in actual)
    squared_error = sum(e * e for e in errors)
    r2 = 1 - squared_error / total if total else float(squared_error == 0)
    close(r2, info["test"]["r2"], "Test R²")

    candidates = defaultdict(dict)
    for row in read_rows(results / "cv_scores.csv"):
        key = (row["model"], float(row["alpha"]) if row["alpha"] else None)
        fold = int(row["fold"])
        require(fold not in candidates[key], "Repeated candidate fold")
        candidates[key][fold] = row
    require(all(set(scores) == folds for scores in candidates.values()), "Missing CV fold")
    means = {
        key: {
            metric: mean(float(row[metric]) for row in scores.values())
            for metric in ("mae", "rmse", "r2")
        }
        for key, scores in candidates.items()
    }
    require({k[0] for k in means} == {r["name"] for r in info["models"]}, "Model names")
    for model in info["models"]:
        key = (model["name"], model["alpha"])
        family_min = min(m["mae"] for k, m in means.items() if k[0] == key[0])
        close(means[key]["mae"], family_min, "Selected alpha")
        for metric, value in means[key].items():
            close(value, model["cv_" + metric], f"{key[0]} CV {metric}")
    winner = min(means, key=lambda key: means[key]["mae"])
    require(winner == (info["selected_model"], info["selected_alpha"]), "Selected model")
    print(
        "Source CSV and labels verified."
        if raw is not None
        else "Using recorded labels. Supply --data-path to verify them against the source CSV."
    )
    print(
        f"Verified {len(seen)} split IDs, {len(candidates)} CV settings and "
        f"{len(predictions)} held-out predictions."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=Path, help="Original CSV for checking the digest and labels"
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    verify(args.data_path, args.results)
