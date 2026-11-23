"""Command-line entry point: clean the data, plot it, train and report."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from hedonic import __version__, config, data, modeling, plots

logger = logging.getLogger(__name__)

EXAMPLE_AREA = 100.0


def _fraction(value: str) -> float:
    number = float(value)
    if not 0.0 < number < 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hedonic",
        description="Fit and compare hedonic price models on property listings.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"hedonic {__version__}",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Local CSV to use instead of downloading the dataset.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download the dataset even if a cached copy exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.REPORTS_DIR,
        help=f"Where figures and metrics are written (default: {config.REPORTS_DIR}).",
    )
    parser.add_argument(
        "--test-size",
        type=_fraction,
        default=config.TEST_SIZE,
        help=f"Fraction of rows held out for testing (default: {config.TEST_SIZE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_STATE,
        help=f"Random seed for the train/test split (default: {config.RANDOM_STATE}).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures in a window as well as saving them.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress logs.",
    )
    return parser


def _section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def _split_manifest(frame: pd.DataFrame, split: modeling.Split, seed: int) -> pd.DataFrame:
    rows = frame[[config.SOURCE_ROW]].copy()
    rows["partition"] = "test"
    rows.loc[split.X_train.index, "partition"] = "train"
    rows["validation_fold"] = pd.Series(pd.NA, index=rows.index, dtype="Int64")
    for fold, (_, validation) in enumerate(modeling.cv_folds(len(split.X_train), seed), 1):
        rows.loc[split.X_train.index[validation], "validation_fold"] = fold
    return rows.sort_values(config.SOURCE_ROW)


def run(args: argparse.Namespace) -> int:
    frame, report = data.load_clean(args.data_path, refresh=args.refresh)
    _section("Cleaning")
    for line in report.as_lines():
        print(line)

    results, baseline, split = modeling.train_all(
        frame, test_size=args.test_size, random_state=args.seed
    )
    selected = min(results, key=lambda result: result.mae)
    _section("Five-fold cross-validation on training listings")
    print(f"{'model':<27} {'R²':>7} {'MAE':>8} {'RMSE':>8}")
    for result in results:
        print(result.as_row())

    predictions, test = modeling.evaluate_selected(selected, split)
    predictions.insert(0, config.SOURCE_ROW, frame.loc[predictions.index, config.SOURCE_ROW])
    _section(f"Held-out test | {selected.name}")
    print(f"MAE {test['mae']:.3f} and RMSE {test['rmse']:.3f} billion Toman; R² {test['r2']:.3f}.")

    train = frame.loc[split.X_train.index]
    example = {
        config.AREA: EXAMPLE_AREA,
        config.ROOM: 2,
        **dict.fromkeys(config.BOOLEAN_FEATURES, 1),
    }
    examples = []
    _section("Matched listing predictions")
    for address in train[config.ADDRESS].value_counts().head(2).index:
        listing = {**example, config.ADDRESS: address}
        price = modeling.predict_price(selected, listing)
        examples.append({**listing, "predicted_bn": price})
        print(f"{address}: {price:.2f} billion Toman")

    scores = pd.concat([result.fold_scores for result in results], ignore_index=True)
    source = args.data_path or config.CACHED_DATA_FILE
    metrics = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "versions": {
            pkg: version(pkg) for pkg in ["numpy", "pandas", "scikit-learn", "matplotlib"]
        },
        "seed": args.seed,
        "test_size": args.test_size,
        "cleaning": asdict(report),
        "train_rows": len(train),
        "test_rows": len(predictions),
        "cv_folds": config.CV_FOLDS,
        "selection": "Lowest mean training-fold MAE (billion Toman)",
        "models": [
            {"name": r.name, "alpha": r.alpha, "cv_mae": r.mae, "cv_rmse": r.rmse, "cv_r2": r.r2}
            for r in results
        ],
        "selected_model": selected.name,
        "selected_alpha": selected.alpha,
        "test": test,
        "matched_listings": examples,
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _split_manifest(frame, split, args.seed).to_csv(output / "split.csv", index=False)
    predictions.sort_values(config.SOURCE_ROW).to_csv(output / "test_predictions.csv", index=False)
    scores.to_csv(output / "cv_scores.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    if not args.no_plots:
        figures = output / "figures"
        plots.generate_all(train, results[0], baseline, figures, show=args.show)
        plots.neighbourhood_ablation(scores, figures, show=args.show)
    print(f"Results written to {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not args.show:
        plots.use_headless_backend()

    try:
        return run(args)
    except data.DatasetError as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
