"""Shared fixtures. Tests use synthetic data and never touch the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hedonic import config


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A small raw-shaped frame with the quirks of the real CSV."""
    return pd.DataFrame(
        {
            config.AREA: ["63", "1,200", "80", "80", "20", "abc", "150"],
            config.ROOM: [1, 4, 2, 2, 1, 2, 3],
            config.PARKING: [True, True, False, False, True, True, True],
            config.WAREHOUSE: [True, False, True, True, False, True, True],
            config.ELEVATOR: [False, True, True, True, False, True, True],
            config.ADDRESS: [
                "Shahran",
                "Pardis",
                "Punak",
                "Punak",
                "Pardis",
                None,
                "Shahran",
            ],
            config.PRICE: [1.85e9, 9.0e9, 2.5e9, 2.5e9, 4.0e8, 1.0e9, 6.0e9],
        }
    )


@pytest.fixture
def clean_frame() -> pd.DataFrame:
    """A larger cleaned frame, big enough to fit and score a model."""
    rng = np.random.default_rng(0)
    size = 300
    area = rng.uniform(40, 300, size)
    address = rng.choice(["Punak", "Pardis", "Shahran", "Rare"], size=size,
                         p=[0.4, 0.3, 0.28, 0.02])
    premium = np.where(address == "Shahran", 2.0, 0.0)
    frame = pd.DataFrame(
        {
            config.AREA: area,
            config.ROOM: rng.integers(1, 5, size),
            config.PARKING: rng.integers(0, 2, size),
            config.WAREHOUSE: rng.integers(0, 2, size),
            config.ELEVATOR: rng.integers(0, 2, size),
            config.ADDRESS: address,
        }
    )
    frame[config.PRICE_BN] = 0.05 * area + premium + rng.normal(0, 0.5, size)
    frame[config.PRICE] = frame[config.PRICE_BN] * config.TOMAN_PER_BILLION
    return frame
