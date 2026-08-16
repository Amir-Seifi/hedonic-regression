from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from house_prices import config, modeling


def test_split_sizes_match_the_requested_fraction(clean_frame):
    split = modeling.make_split(clean_frame, [config.AREA], test_size=0.25)
    assert len(split.X_test) == round(0.25 * len(clean_frame))
    assert len(split.X_train) + len(split.X_test) == len(clean_frame)


def test_split_is_reproducible(clean_frame):
    first = modeling.make_split(clean_frame, [config.AREA], random_state=7)
    second = modeling.make_split(clean_frame, [config.AREA], random_state=7)
    pd.testing.assert_frame_equal(first.X_test, second.X_test)


def test_pipeline_handles_a_neighbourhood_unseen_in_training(clean_frame):
    features = [*config.NUMERIC_FEATURES, *config.CATEGORICAL_FEATURES]
    pipeline = modeling.build_pipeline(features, LinearRegression())
    train = clean_frame[clean_frame[config.ADDRESS] != "Rare"]
    pipeline.fit(train[features], train[config.PRICE_BN])

    unseen = pd.DataFrame(
        [
            {
                config.AREA: 100.0,
                config.ROOM: 2,
                config.PARKING: 1,
                config.WAREHOUSE: 1,
                config.ELEVATOR: 1,
                config.ADDRESS: "Somewhere New",
            }
        ]
    )
    assert pipeline.predict(unseen).shape == (1,)


def test_train_all_returns_every_model_with_sane_scores(clean_frame):
    results, baseline_split, full_split = modeling.train_all(clean_frame)

    assert [r.name for r in results] == [
        "Area only",
        "All features",
        "All + Ridge",
        "All + Lasso",
    ]
    for result in results:
        assert 0.0 < result.r2 <= 1.0
        assert result.mae > 0
        assert result.rmse >= result.mae

    # The synthetic prices depend on area and neighbourhood, so the full model
    # must beat the area-only baseline.
    by_name = {r.name: r for r in results}
    assert by_name["All features"].r2 > by_name["Area only"].r2
    assert list(baseline_split.X_train.columns) == [config.AREA]
    assert config.ADDRESS in full_split.X_train.columns


def test_coefficients_are_named_and_include_neighbourhoods(clean_frame):
    results, _, _ = modeling.train_all(clean_frame)
    coefficients = modeling.coefficients(results[1])

    assert config.AREA in coefficients.index
    assert any(name.startswith(f"{config.ADDRESS}_") for name in coefficients.index)
    assert coefficients.is_monotonic_decreasing
    assert coefficients[config.AREA] > 0
    # "Rare" is below the frequency threshold, so it is pooled and relabelled.
    assert not any(modeling.INFREQUENT_SUFFIX in name for name in coefficients.index)
    assert f"{config.ADDRESS}_other (rare)" in coefficients.index


def test_predict_price_returns_a_plausible_number(clean_frame):
    results, _, _ = modeling.train_all(clean_frame)
    baseline = results[0]
    price = modeling.predict_price(baseline, {config.AREA: 100.0})

    assert isinstance(price, float)
    assert 0 < price < 100


def test_predict_price_reports_missing_features(clean_frame):
    results, _, _ = modeling.train_all(clean_frame)
    with pytest.raises(ValueError, match="missing required features"):
        modeling.predict_price(results[1], {config.AREA: 100.0})
