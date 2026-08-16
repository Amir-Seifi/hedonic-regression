"""Model building, training and evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_prices import config

logger = logging.getLogger(__name__)

# Suffix scikit-learn gives the column that pools rare categories.
INFREQUENT_SUFFIX = "_infrequent_sklearn"


@dataclass(frozen=True)
class Split:
    """A single train/test split shared by every model, so scores compare."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class ModelResult:
    """Test-set scores for one fitted pipeline."""

    name: str
    r2: float
    mae: float
    rmse: float
    pipeline: Pipeline

    def as_row(self) -> str:
        return f"{self.name:<22} {self.r2:>7.3f} {self.mae:>8.3f} {self.rmse:>8.3f}"


def make_split(
    df: pd.DataFrame,
    features: list[str],
    *,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Split:
    """Split the data into train and test sets for the given feature columns."""
    X = df[features]
    y = df[config.PRICE_BN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return Split(X_train, X_test, y_train, y_test)


def build_pipeline(
    features: list[str],
    regressor: LinearRegression | Ridge | Lasso,
    *,
    min_address_frequency: int = config.MIN_ADDRESS_FREQUENCY,
) -> Pipeline:
    """Build a preprocessing + regression pipeline for the given features.

    Preprocessing is part of the pipeline so it is fitted on the training fold
    only. ``handle_unknown="ignore"`` keeps prediction working for
    neighbourhoods that appear only in the test set, and ``min_frequency``
    pools rare neighbourhoods into one column instead of giving each a
    coefficient estimated from a couple of listings.
    """
    numeric = [c for c in features if c in config.NUMERIC_FEATURES]
    categorical = [c for c in features if c in config.CATEGORICAL_FEATURES]

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=min_address_frequency,
                    sparse_output=False,
                ),
                categorical,
            )
        )

    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers)),
            ("regressor", regressor),
        ]
    )


def fit_and_score(name: str, pipeline: Pipeline, split: Split) -> ModelResult:
    """Fit a pipeline on the training set and score it on the test set."""
    pipeline.fit(split.X_train, split.y_train)
    predictions = pipeline.predict(split.X_test)
    return ModelResult(
        name=name,
        r2=r2_score(split.y_test, predictions),
        mae=mean_absolute_error(split.y_test, predictions),
        rmse=float(mean_squared_error(split.y_test, predictions) ** 0.5),
        pipeline=pipeline,
    )


def train_all(
    df: pd.DataFrame,
    *,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    min_address_frequency: int = config.MIN_ADDRESS_FREQUENCY,
) -> tuple[list[ModelResult], Split, Split]:
    """Train the baseline and the multi-feature models.

    Returns the results plus both splits: the area-only split (used for the
    regression-line plot) and the full-feature split.
    """
    baseline_features = [config.AREA]
    full_features = [*config.NUMERIC_FEATURES, *config.CATEGORICAL_FEATURES]

    baseline_split = make_split(
        df, baseline_features, test_size=test_size, random_state=random_state
    )
    full_split = make_split(
        df, full_features, test_size=test_size, random_state=random_state
    )

    candidates: list[tuple[str, list[str], Split, LinearRegression | Ridge | Lasso]] = [
        ("Area only", baseline_features, baseline_split, LinearRegression()),
        ("All features", full_features, full_split, LinearRegression()),
        ("All + Ridge", full_features, full_split, Ridge(alpha=config.RIDGE_ALPHA)),
        ("All + Lasso", full_features, full_split, Lasso(alpha=config.LASSO_ALPHA)),
    ]

    results = []
    for name, features, split, regressor in candidates:
        pipeline = build_pipeline(
            features, regressor, min_address_frequency=min_address_frequency
        )
        logger.info("Training %s", name)
        results.append(fit_and_score(name, pipeline, split))
    return results, baseline_split, full_split


def coefficients(result: ModelResult) -> pd.Series:
    """Coefficients of a fitted pipeline, indexed by feature name.

    Features are standardised inside the pipeline, so the values are directly
    comparable: each one is the price change (in billion Toman) per one
    standard deviation of that feature.
    """
    preprocess: ColumnTransformer = result.pipeline.named_steps["preprocess"]
    regressor = result.pipeline.named_steps["regressor"]
    names = [
        # "numeric__Area" -> "Area"; the pooled rare-category column is named
        # "<feature>_infrequent_sklearn", which is not worth showing verbatim.
        name.split("__", 1)[-1].replace(INFREQUENT_SUFFIX, "_other (rare)")
        for name in preprocess.get_feature_names_out()
    ]
    return pd.Series(regressor.coef_, index=names).sort_values(ascending=False)


def predict_price(result: ModelResult, listing: dict[str, object]) -> float:
    """Predict the price (in billion Toman) of a single listing."""
    columns = list(result.pipeline.feature_names_in_)
    missing = [c for c in columns if c not in listing]
    if missing:
        raise ValueError(f"Listing is missing required features: {missing}")
    frame = pd.DataFrame([{c: listing[c] for c in columns}])
    return float(result.pipeline.predict(frame)[0])
