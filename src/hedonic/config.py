"""Project-wide constants and default settings."""

from __future__ import annotations

from pathlib import Path

# Relative to the working directory, so an installed copy of the package never
# writes back into site-packages. Run the CLI from the project root, or pass
# --output-dir / --data-path.
RAW_DATA_DIR = Path("data") / "raw"
REPORTS_DIR = Path("reports")

DATA_URL = (
    "https://raw.githubusercontent.com/SharifiZarchi/IntroAI/main/"
    "Session_04/TehranHouses/housePrice.csv"
)
CACHED_DATA_FILE = RAW_DATA_DIR / "housePrice.csv"

# Column names as they appear in the raw CSV.
AREA = "Area"
ROOM = "Room"
PARKING = "Parking"
WAREHOUSE = "Warehouse"
ELEVATOR = "Elevator"
ADDRESS = "Address"
PRICE = "Price"

# Derived column: price in billions of Toman, which keeps plots and
# coefficients readable instead of scientific notation.
PRICE_BN = "Price_Bn"
TOMAN_PER_BILLION = 1_000_000_000

BOOLEAN_FEATURES = [PARKING, WAREHOUSE, ELEVATOR]
NUMERIC_FEATURES = [AREA, ROOM, *BOOLEAN_FEATURES]
CATEGORICAL_FEATURES = [ADDRESS]
REQUIRED_RAW_COLUMNS = [*NUMERIC_FEATURES, ADDRESS, PRICE]

# Cleaning defaults. Listings outside this range are data-entry errors
# rather than real apartments (the raw file contains areas above 10,000 m2).
MIN_AREA_M2 = 30.0
MAX_AREA_M2 = 500.0
# Drop the extreme right tail of prices, which is dominated by a handful of
# luxury outliers that a linear model cannot represent anyway.
PRICE_UPPER_QUANTILE = 0.995

# Modelling defaults.
TEST_SIZE = 0.2
RANDOM_STATE = 42
# Neighbourhoods with fewer listings than this are pooled into a single
# "infrequent" category so their coefficients stay meaningful.
MIN_ADDRESS_FREQUENCY = 10
RIDGE_ALPHA = 1.0
LASSO_ALPHA = 0.01

# Plotting defaults.
MIN_LISTINGS_PER_NEIGHBOURHOOD = 30
TOP_NEIGHBOURHOODS = 12
HISTOGRAM_BINS = 50
