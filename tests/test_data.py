from __future__ import annotations

import pandas as pd
import pytest

from hedonic import config, data


def test_clean_parses_area_with_thousands_separator(raw_frame):
    frame, _ = data.clean(raw_frame, max_area=2000)
    assert pd.api.types.is_numeric_dtype(frame[config.AREA])
    assert 1200.0 in set(frame[config.AREA])


def test_clean_drops_duplicates_missing_and_out_of_range(raw_frame):
    frame, report = data.clean(raw_frame)

    assert report.dropped_duplicates == 1  # the repeated Punak listing
    assert report.dropped_missing == 1  # unparseable area / missing address
    assert report.dropped_area_range == 2  # 20 m² and 1,200 m²
    assert report.rows_clean == len(frame)
    assert frame[config.AREA].between(config.MIN_AREA_M2, config.MAX_AREA_M2).all()


def test_clean_adds_price_in_billions(raw_frame):
    frame, _ = data.clean(raw_frame)
    expected = frame[config.PRICE] / config.TOMAN_PER_BILLION
    pd.testing.assert_series_equal(frame[config.PRICE_BN], expected, check_names=False)


def test_clean_casts_booleans_to_integers(raw_frame):
    frame, _ = data.clean(raw_frame)
    for column in config.BOOLEAN_FEATURES:
        assert pd.api.types.is_integer_dtype(frame[column])
        assert set(frame[column]) <= {0, 1}


def test_clean_resets_the_index(raw_frame):
    frame, _ = data.clean(raw_frame)
    assert list(frame.index) == list(range(len(frame)))


def test_clean_rejects_missing_columns(raw_frame):
    with pytest.raises(data.DatasetError, match="missing required columns"):
        data.clean(raw_frame.drop(columns=[config.PRICE]))


def test_clean_rejects_a_frame_that_empties_out(raw_frame):
    with pytest.raises(data.DatasetError, match="No rows left"):
        data.clean(raw_frame, min_area=10_000, max_area=20_000)


def test_load_raw_reads_a_local_file(tmp_path, raw_frame):
    path = tmp_path / "houses.csv"
    raw_frame.to_csv(path, index=False)
    assert len(data.load_raw(path)) == len(raw_frame)


def test_load_raw_reports_a_missing_local_file(tmp_path):
    with pytest.raises(data.DatasetError, match="not found"):
        data.load_raw(tmp_path / "nope.csv")


def test_load_raw_prefers_the_cache_over_the_network(tmp_path, raw_frame):
    cache = tmp_path / "cached.csv"
    raw_frame.to_csv(cache, index=False)
    # An unreachable URL proves no download happens when the cache is warm.
    frame = data.load_raw("https://invalid.invalid/x.csv", cache_path=cache)
    assert len(frame) == len(raw_frame)


def test_correlations_rank_area_first(clean_frame):
    result = data.correlations(clean_frame)
    assert result.index[0] == config.AREA
    assert config.PRICE_BN not in result.index
