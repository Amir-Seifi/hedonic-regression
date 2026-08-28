"""Command-line entry point: clean the data, plot it, train and report."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from hedonic import __version__, config, data, modeling, plots

logger = logging.getLogger(__name__)

EXAMPLE_AREA = 100.0
TOP_COEFFICIENTS = 10


def _fraction(value: str) -> float:
    """An argparse type for a strictly-between-0-and-1 float."""
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


def _example_listing(df: pd.DataFrame) -> dict[str, object]:
    """A typical listing used to demonstrate a prediction."""
    return {
        config.AREA: EXAMPLE_AREA,
        config.ROOM: 2,
        config.PARKING: 1,
        config.WAREHOUSE: 1,
        config.ELEVATOR: 1,
        config.ADDRESS: df[config.ADDRESS].mode().iat[0],
    }


def run(args: argparse.Namespace) -> int:
    frame, report = data.load_clean(args.data_path, refresh=args.refresh)

    _section("Cleaning")
    for line in report.as_lines():
        print(line)

    _section("Correlation with price")
    print(data.correlations(frame).round(3).to_string())

    results, baseline_split, _ = modeling.train_all(
        frame, test_size=args.test_size, random_state=args.seed
    )
    by_name = {result.name: result for result in results}

    _section("Model scores (test set)")
    print(f"{'model':<22} {'R²':>7} {'MAE':>8} {'RMSE':>8}")
    for result in results:
        print(result.as_row())
    print("\nMAE and RMSE are in billion Toman.")

    best = max(results, key=lambda result: result.r2)
    _section(f"Standardised coefficients — {best.name}")
    coefficients = modeling.coefficients(best)
    print("Largest positive effect on price:")
    print(coefficients.head(TOP_COEFFICIENTS).round(3).to_string())
    print("\nLargest negative effect on price:")
    print(coefficients.tail(TOP_COEFFICIENTS).sort_values().round(3).to_string())

    _section("Example prediction")
    listing = _example_listing(frame)
    baseline_price = modeling.predict_price(
        by_name["Area only"], {config.AREA: EXAMPLE_AREA}
    )
    best_price = modeling.predict_price(best, listing)
    print(f"{EXAMPLE_AREA:.0f} m² apartment in {listing[config.ADDRESS]}:")
    print(f"  area-only model  {baseline_price:.2f} billion Toman")
    print(f"  {best.name:<16} {best_price:.2f} billion Toman")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "rows_used": report.rows_clean,
                "test_size": args.test_size,
                "seed": args.seed,
                "models": [
                    {
                        "name": result.name,
                        "r2": round(result.r2, 4),
                        "mae_billion_toman": round(result.mae, 4),
                        "rmse_billion_toman": round(result.rmse, 4),
                    }
                    for result in results
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    written = []
    if not args.no_plots:
        written = plots.generate_all(
            frame,
            by_name["Area only"],
            baseline_split,
            args.output_dir / "figures",
            show=args.show,
        )

    _section("Output")
    print(f"metrics: {metrics_path}")
    for path in written:
        print(f"figure:  {path}")
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
