"""Model building, training and evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from hedonic import config
from hedonic.data import DatasetError

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
    """Training-fold mean scores and the fitted pipeline for the selected setting."""

    name: str
    r2: float
    mae: float
    rmse: float
    pipeline: Pipeline
    fold_scores: pd.DataFrame
    alpha: float | None

    def as_row(self) -> str:
        return f"{self.name:<27} {self.r2:>7.3f} {self.mae:>8.3f} {self.rmse:>8.3f}"


def make_split(
    df: pd.DataFrame,
    features: list[str],
    *,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> Split:
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
    log_price_and_area: bool = False,
) -> Pipeline:
    """Preprocessing sits inside the pipeline so it is fitted on the training
    fold only. Rare neighbourhoods are pooled rather than each getting a
    coefficient estimated from two or three listings.
    """
    numeric = [c for c in features if c in config.NUMERIC_FEATURES]
    categorical = [c for c in features if c in config.CATEGORICAL_FEATURES]

    transformers: list[tuple[str, object, list[str]]] = []
    if log_price_and_area:
        numeric.remove(config.AREA)
        log_area = Pipeline(
            [
                ("log", FunctionTransformer(np.log, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("log_area", log_area, [config.AREA]))
        regressor = TransformedTargetRegressor(
            regressor=regressor, func=np.log, inverse_func=np.exp
        )
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    drop="first",
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


def cv_folds(size: int, seed: int):
    if size < 2 * config.CV_FOLDS:
        raise DatasetError("Five-fold R² requires at least ten training listings.")
    return list(KFold(config.CV_FOLDS, shuffle=True, random_state=seed).split(np.arange(size)))


def train_all(
    df: pd.DataFrame,
    *,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    min_address_frequency: int = config.MIN_ADDRESS_FREQUENCY,
) -> tuple[list[ModelResult], Split, Split]:
    """Select each family's setting by mean training-fold MAE. Hold test labels in reserve."""
    test_count = int(np.ceil(len(df) * test_size))
    if test_count < 2 or len(df) - test_count < 2 * config.CV_FOLDS:
        raise DatasetError("The split requires at least ten training and two test listings.")
    full = make_split(
        df,
        [*config.NUMERIC_FEATURES, config.ADDRESS],
        test_size=test_size,
        random_state=random_state,
    )
    baseline = Split(
        full.X_train[[config.AREA]], full.X_test[[config.AREA]], full.y_train, full.y_test
    )
    folds = cv_folds(len(full.X_train), random_state)
    numeric = config.NUMERIC_FEATURES
    all_features = [*numeric, config.ADDRESS]
    families = [
        ("Area only", [config.AREA], [LinearRegression()], False),
        ("Property attributes", numeric, [LinearRegression()], False),
        ("Attributes + neighbourhood", all_features, [LinearRegression()], False),
        ("Ridge", all_features, [Ridge(alpha=a) for a in config.RIDGE_ALPHAS], False),
        (
            "Lasso",
            all_features,
            [Lasso(alpha=a, max_iter=10000) for a in config.LASSO_ALPHAS],
            False,
        ),
        ("Log-log Ridge", all_features, [Ridge(alpha=a) for a in config.RIDGE_ALPHAS], True),
    ]
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    results = []
    for name, features, regressors, log_price_and_area in families:
        rows = []
        candidates = []
        x = full.X_train[features]
        for regressor in regressors:
            model = build_pipeline(
                features,
                regressor,
                min_address_frequency=min_address_frequency,
                log_price_and_area=log_price_and_area,
            )
            scores = cross_validate(
                model, x, full.y_train, cv=folds, scoring=scoring, error_score="raise"
            )
            alpha = getattr(regressor, "alpha", None)
            mae, rmse, r2 = -scores["test_mae"], -scores["test_rmse"], scores["test_r2"]
            rows.extend(
                {"model": name, "alpha": alpha, "fold": i, "mae": m, "rmse": e, "r2": r}
                for i, (m, e, r) in enumerate(zip(mae, rmse, r2, strict=True), 1)
            )
            candidates.append(
                (float(mae.mean()), float(rmse.mean()), float(r2.mean()), alpha, model)
            )
        mae, rmse, r2, alpha, model = min(candidates, key=lambda item: item[0])
        model.fit(x, full.y_train)
        results.append(
            ModelResult(
                name, r2, mae, rmse, model, pd.DataFrame(rows).astype({"alpha": "float64"}), alpha
            )
        )
    return results, baseline, full


def evaluate_selected(result: ModelResult, split: Split) -> tuple[pd.DataFrame, dict]:
    columns = list(result.pipeline.feature_names_in_)
    predictions = result.pipeline.predict(split.X_test[columns])
    rows = pd.DataFrame({"actual_bn": split.y_test, "predicted_bn": predictions})
    actual = rows["actual_bn"]
    return rows, {
        "negative_predictions": int((predictions < 0).sum()),
        "r2": float(r2_score(actual, predictions)),
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
    }


def coefficients(result: ModelResult) -> pd.Series:
    """Return named slopes in model coordinates; only numeric columns are scaled."""
    preprocess: ColumnTransformer = result.pipeline.named_steps["preprocess"]
    regressor = result.pipeline.named_steps["regressor"]
    if isinstance(regressor, TransformedTargetRegressor):
        regressor = regressor.regressor_
    # "numeric__Area" -> "Area", and the pooled column sklearn calls
    # "<feature>_infrequent_sklearn" gets a readable label.
    names = [
        (
            "log(Area)"
            if name == "log_area__Area"
            else name.split("__", 1)[-1].replace(INFREQUENT_SUFFIX, "_other (rare)")
        )
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
