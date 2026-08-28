# Hedonic Price Regression

[![CI](https://github.com/Amir-Seifi/hedonic-regression/actions/workflows/ci.yml/badge.svg)](https://github.com/Amir-Seifi/hedonic-regression/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A hedonic price model estimates the value of a good from its attributes. This
project applies that to residential property listings: area, room count,
parking / storage / elevator flags and neighbourhood, regressed on sale price.

One command cleans the data, writes the figures, trains four models and prints a
comparison. Preprocessing is fitted inside the cross-validation boundary, the
whole pipeline is unit-tested, and every run is reproducible from a seed.

## Results

Trained on 3,209 listings after cleaning, scored on a held-out 20% test set.
Prices are in billion Toman.

| Model         |    R² |   MAE |  RMSE |
| ------------- | ----: | ----: | ----: |
| Area only     | 0.528 | 2.300 | 4.238 |
| All features  | 0.738 | 1.687 | 3.160 |
| All + Ridge   | 0.736 | 1.689 | 3.172 |
| All + Lasso   | 0.717 | 1.756 | 3.281 |

Area is the strongest single predictor (correlation 0.75), but it explains only
about half the variance on its own. Adding the neighbourhood is what closes the
gap: location moves the predicted price further than any other feature.
Regularisation buys nothing here, which is the expected result at this ratio of
rows to features and is worth reporting as such.

![Linear regression on area](reports/figures/regression_line.png)
![Most expensive neighbourhoods](reports/figures/expensive_neighbourhoods.png)

## Install

Requires Python 3.10 or newer.

```bash
git clone https://github.com/Amir-Seifi/hedonic-regression.git
cd hedonic-regression
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
hedonic
```

The dataset is downloaded once and cached in `data/raw/`, so later runs work
offline. Figures and `metrics.json` are written to `reports/`. Both paths are
relative to the working directory, so run the command from the project root
(or point `--data-path` and `--output-dir` elsewhere).

Useful flags:

| Flag                | Meaning                                             |
| ------------------- | --------------------------------------------------- |
| `--data-path PATH`  | Use a local CSV instead of downloading               |
| `--refresh`         | Re-download even if a cached copy exists             |
| `--output-dir PATH` | Where figures and metrics go (default `reports/`)    |
| `--test-size FLOAT` | Held-out fraction (default `0.2`)                    |
| `--seed INT`        | Split seed (default `42`)                            |
| `--no-plots`        | Skip figures                                         |
| `--show`            | Open figures in a window as well as saving them      |
| `-v`                | Print progress logs                                  |
| `--version`         | Print the version and exit                           |

## Data

A public dataset of 3,479 residential listings in Tehran, with area, number of
rooms, parking / storage / elevator flags, neighbourhood and price. Source:
[`housePrice.csv`](https://raw.githubusercontent.com/SharifiZarchi/IntroAI/main/Session_04/TehranHouses/housePrice.csv).

Cleaning drops 270 rows in total:

| Step                              | Rows |
| --------------------------------- | ---: |
| Duplicate listings                |  208 |
| Missing values                    |   23 |
| Area outside 30–500 m²            |   23 |
| Price above the 99.5th percentile |   16 |

`Area` arrives as text with thousands separators (`"1,250"`), so it is parsed to
a number first. The area bounds remove data-entry errors: the raw file contains
"apartments" of over 10,000 m². The price cap trims the extreme right tail, so
the metrics above describe the trimmed distribution rather than the full market;
the 16 excluded listings are luxury properties whose prices are driven by
attributes this dataset does not record.

## Project layout

```
src/hedonic/
  config.py     constants and defaults
  data.py       download, cache, clean, correlations
  modeling.py   split, pipelines, training, coefficients
  plots.py      the five figures
  cli.py        argument parsing and the report
tests/          unit tests (synthetic data, no network)
```

Preprocessing lives inside the scikit-learn pipelines, so scaling and one-hot
encoding are fitted on the training fold only. Unseen neighbourhoods are handled
with `handle_unknown="ignore"`, and neighbourhoods with fewer than 10 listings are
pooled into one category so their coefficients stay meaningful.

Because every feature is standardised, the reported coefficients are directly
comparable: each is the price change in billion Toman per one standard deviation
of that feature.

## Scope and next steps

The model class is deliberately linear: the goal is an interpretable coefficient
per attribute, not the lowest possible error. The natural extensions, in order of
expected value:

- A gradient-boosted baseline, to quantify what the linear form costs in accuracy.
- A log-transformed target, since prices are right-skewed and the residuals show it.
- Cross-validated metrics with a spread, instead of a single split.
- A `predict` subcommand that scores new listings from a saved model.

## Development

```bash
pytest
ruff check .
```

## License

MIT. See [LICENSE](LICENSE).
