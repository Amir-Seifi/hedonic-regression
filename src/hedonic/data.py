"""Loading and cleaning of the property-listings dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from hedonic import config

logger = logging.getLogger(__name__)


class DatasetError(RuntimeError):
    """Raised when the dataset cannot be loaded or is missing columns."""


@dataclass(frozen=True)
class CleaningReport:
    """How many rows each cleaning step removed."""

    rows_raw: int
    rows_clean: int
    dropped_duplicates: int
    dropped_missing: int
    dropped_area_range: int
    dropped_price_outliers: int

    def as_lines(self) -> list[str]:
        return [
            f"raw rows                 {self.rows_raw:>6}",
            f"- duplicates             {self.dropped_duplicates:>6}",
            f"- missing values         {self.dropped_missing:>6}",
            f"- area outside range     {self.dropped_area_range:>6}",
            f"- price outliers         {self.dropped_price_outliers:>6}",
            f"clean rows               {self.rows_clean:>6}",
        ]


def load_raw(
    source: str | Path | None = None,
    *,
    cache_path: Path = config.CACHED_DATA_FILE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the raw dataset as a DataFrame.

    A local CSV path is read directly. Otherwise the dataset is downloaded
    once and cached under ``data/raw`` so later runs work offline; pass
    ``refresh=True`` to force a fresh download.
    """
    if source is not None and Path(source).exists():
        logger.info("Reading dataset from %s", source)
        return _read_csv(Path(source))

    if source is not None and not str(source).startswith(("http://", "https://")):
        raise DatasetError(f"Dataset source not found: {source}")

    url = str(source) if source is not None else config.DATA_URL

    if cache_path.exists() and not refresh:
        logger.info("Reading cached dataset from %s", cache_path)
        return _read_csv(cache_path)

    logger.info("Downloading dataset from %s", url)
    try:
        df = pd.read_csv(url)
    except Exception as exc:  # pragma: no cover - depends on network
        raise DatasetError(
            f"Could not download the dataset from {url}. "
            "Check your connection or pass a local CSV with --data-path."
        ) from exc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Cached dataset at %s", cache_path)
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise DatasetError(f"Could not read {path}: {exc}") from exc


def clean(
    df: pd.DataFrame,
    *,
    min_area: float = config.MIN_AREA_M2,
    max_area: float = config.MAX_AREA_M2,
    price_upper_quantile: float = config.PRICE_UPPER_QUANTILE,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean the raw dataset and report what was removed.

    Steps: validate columns, parse ``Area`` (stored as a string with thousands
    separators), drop duplicates and missing values, keep areas inside a
    plausible range, drop the extreme price tail, and add ``Price_Bn``.
    """
    missing_columns = [c for c in config.REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing_columns:
        raise DatasetError(f"Dataset is missing required columns: {missing_columns}")

    rows_raw = len(df)
    df = df.loc[:, config.REQUIRED_RAW_COLUMNS].copy()

    df = df.drop_duplicates()
    after_duplicates = len(df)

    # "1,250" -> 1250.0; unparseable values become NaN and are dropped below.
    df[config.AREA] = pd.to_numeric(
        df[config.AREA].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    df[config.PRICE] = pd.to_numeric(df[config.PRICE], errors="coerce")
    df[config.ROOM] = pd.to_numeric(df[config.ROOM], errors="coerce")
    for column in config.BOOLEAN_FEATURES:
        df[column] = df[column].astype(bool).astype(int)
    df[config.ADDRESS] = df[config.ADDRESS].astype("string").str.strip()

    df = df.dropna(subset=config.REQUIRED_RAW_COLUMNS)
    after_missing = len(df)

    df = df[df[config.AREA].between(min_area, max_area)]
    after_area = len(df)

    df = df[df[config.PRICE] > 0]
    # "higher" keeps the cap on an observed price, so the single most
    # expensive listing is never dropped just by interpolation.
    price_cap = df[config.PRICE].quantile(price_upper_quantile, interpolation="higher")
    df = df[df[config.PRICE] <= price_cap]
    after_price = len(df)

    if df.empty:
        raise DatasetError("No rows left after cleaning; check the cleaning thresholds.")

    df[config.PRICE_BN] = df[config.PRICE] / config.TOMAN_PER_BILLION
    df = df.reset_index(drop=True)

    report = CleaningReport(
        rows_raw=rows_raw,
        rows_clean=len(df),
        dropped_duplicates=rows_raw - after_duplicates,
        dropped_missing=after_duplicates - after_missing,
        dropped_area_range=after_missing - after_area,
        dropped_price_outliers=after_area - after_price,
    )
    return df, report


def load_clean(
    source: str | Path | None = None,
    *,
    refresh: bool = False,
    **clean_kwargs: float,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Convenience wrapper: load the raw dataset and clean it."""
    return clean(load_raw(source, refresh=refresh), **clean_kwargs)


def correlations(df: pd.DataFrame) -> pd.Series:
    """Correlation of every numeric feature with the price, strongest first."""
    columns = [*config.NUMERIC_FEATURES, config.PRICE_BN]
    return (
        df[columns]
        .astype(float)
        .corr()[config.PRICE_BN]
        .drop(config.PRICE_BN)
        .sort_values(ascending=False)
    )
