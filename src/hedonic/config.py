"""Project-wide constants and default settings."""

from __future__ import annotations

from pathlib import Path

# Relative to the working directory, so an installed copy of the package never
# writes back into site-packages. Run the CLI from the project root, or pass
# --output-dir / --data-path.
RAW_DATA_DIR = Path("data") / "raw"
REPORTS_DIR = Path("reports")

DATA_URL = (
    "https://raw.githubusercontent.com/SharifiZarchi/IntroAI/"
    "f04daf55f28392bee7a0c125a6ee1d93cae30427/"
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

# Raw CSV row (zero-based) and derived price in billion Toman.
SOURCE_ROW = "source_row"
PRICE_BN = "Price_Bn"
TOMAN_PER_BILLION = 1_000_000_000

BOOLEAN_FEATURES = [PARKING, WAREHOUSE, ELEVATOR]
NUMERIC_FEATURES = [AREA, ROOM, *BOOLEAN_FEATURES]
CATEGORICAL_FEATURES = [ADDRESS]
REQUIRED_RAW_COLUMNS = [*NUMERIC_FEATURES, ADDRESS, PRICE]

# Fixed floor-area bounds define the population evaluated.
MIN_AREA_M2 = 30.0
MAX_AREA_M2 = 500.0
TEST_SIZE = 0.2
RANDOM_STATE = 42
# Neighbourhoods with fewer listings than this are pooled into a single
# "infrequent" category so their coefficients stay meaningful.
MIN_ADDRESS_FREQUENCY = 10
CV_FOLDS = 5
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
LASSO_ALPHAS = (0.001, 0.01, 0.1)

MIN_LISTINGS_PER_NEIGHBOURHOOD = 30
TOP_NEIGHBOURHOODS = 12
HISTOGRAM_BINS = 50
