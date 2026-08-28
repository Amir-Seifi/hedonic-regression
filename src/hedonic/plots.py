"""Figures for the exploratory analysis and the fitted baseline model."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from hedonic import config
from hedonic.modeling import ModelResult, Split

logger = logging.getLogger(__name__)


def use_headless_backend() -> None:
    """Select a non-interactive backend so plotting works without a display.

    Must be called before ``matplotlib.pyplot`` is imported for the first time,
    which is why every function below imports pyplot lazily.
    """
    matplotlib.use("Agg")


def _save(fig, output_dir: Path, filename: str, *, show: bool) -> Path:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    logger.info("Wrote %s", path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def price_distribution(df: pd.DataFrame, output_dir: Path, *, show: bool = False) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        df[config.PRICE_BN],
        bins=config.HISTOGRAM_BINS,
        color="steelblue",
        edgecolor="white",
    )
    ax.set_xlabel("Price (billion Toman)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of prices")
    return _save(fig, output_dir, "price_distribution.png", show=show)


def area_distribution(df: pd.DataFrame, output_dir: Path, *, show: bool = False) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(
        df[config.AREA],
        bins=config.HISTOGRAM_BINS,
        color="seagreen",
        edgecolor="white",
    )
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of area")
    return _save(fig, output_dir, "area_distribution.png", show=show)


def price_vs_area(df: pd.DataFrame, output_dir: Path, *, show: bool = False) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(df[config.AREA], df[config.PRICE_BN], alpha=0.3, color="darkorange")
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Price (billion Toman)")
    ax.set_title("Price vs area")
    return _save(fig, output_dir, "price_vs_area.png", show=show)


def expensive_neighbourhoods(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    min_listings: int = config.MIN_LISTINGS_PER_NEIGHBOURHOOD,
    top_n: int = config.TOP_NEIGHBOURHOODS,
    show: bool = False,
) -> Path | None:
    """Neighbourhoods below ``min_listings`` are excluded: a single listing
    would otherwise put an outlier at the top of the chart.
    """
    import matplotlib.pyplot as plt

    counts = df[config.ADDRESS].value_counts()
    popular = counts[counts >= min_listings].index
    if len(popular) == 0:
        logger.warning(
            "No neighbourhood has at least %d listings; skipping the bar chart.",
            min_listings,
        )
        return None

    average = (
        df[df[config.ADDRESS].isin(popular)]
        .groupby(config.ADDRESS, observed=True)[config.PRICE_BN]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values()  # smallest at the bottom of a horizontal bar chart
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(average.index.astype(str), average.to_numpy(), color="mediumpurple")
    ax.set_xlabel("Average price (billion Toman)")
    ax.set_title(f"Most expensive neighbourhoods (≥ {min_listings} listings)")
    return _save(fig, output_dir, "expensive_neighbourhoods.png", show=show)


def regression_line(
    result: ModelResult,
    split: Split,
    output_dir: Path,
    *,
    show: bool = False,
) -> Path:
    """Test-set listings against the fitted area-only regression line."""
    import matplotlib.pyplot as plt

    areas = split.X_train[config.AREA]
    line = pd.DataFrame(
        {config.AREA: np.linspace(areas.min(), areas.max(), 100)}
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        split.X_test[config.AREA],
        split.y_test,
        alpha=0.3,
        color="darkorange",
        label="Actual (test set)",
    )
    ax.plot(
        line[config.AREA],
        result.pipeline.predict(line),
        color="black",
        linewidth=2,
        label=f"Model (R² = {result.r2:.2f})",
    )
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Price (billion Toman)")
    ax.set_title("Linear regression on area")
    ax.legend()
    return _save(fig, output_dir, "regression_line.png", show=show)


def generate_all(
    df: pd.DataFrame,
    baseline: ModelResult,
    baseline_split: Split,
    output_dir: Path,
    *,
    show: bool = False,
) -> list[Path]:
    paths = [
        price_distribution(df, output_dir, show=show),
        area_distribution(df, output_dir, show=show),
        price_vs_area(df, output_dir, show=show),
        expensive_neighbourhoods(df, output_dir, show=show),
        regression_line(baseline, baseline_split, output_dir, show=show),
    ]
    return [p for p in paths if p is not None]
