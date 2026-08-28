from __future__ import annotations

import json

import pytest

from hedonic import __version__, cli, config


def test_end_to_end_run_writes_metrics_and_figures(tmp_path, clean_frame, capsys):
    csv = tmp_path / "houses.csv"
    clean_frame.drop(columns=[config.PRICE_BN]).to_csv(csv, index=False)
    output = tmp_path / "reports"

    exit_code = cli.main(
        ["--data-path", str(csv), "--output-dir", str(output), "--seed", "1"]
    )
    assert exit_code == 0

    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["seed"] == 1
    assert len(metrics["models"]) == 4

    figures = sorted(p.name for p in (output / "figures").glob("*.png"))
    assert figures == [
        "area_distribution.png",
        "expensive_neighbourhoods.png",
        "price_distribution.png",
        "price_vs_area.png",
        "regression_line.png",
    ]

    stdout = capsys.readouterr().out
    assert "Model scores (test set)" in stdout
    assert "Example prediction" in stdout


def test_run_reports_a_missing_data_file(tmp_path, capsys):
    assert cli.main(["--data-path", str(tmp_path / "missing.csv")]) == 1


@pytest.mark.parametrize("value", ["0", "1", "1.5", "-0.2"])
def test_test_size_outside_the_unit_interval_is_rejected(value):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--test-size", value])


def test_version_flag_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_plots_skips_figures(tmp_path, clean_frame):
    csv = tmp_path / "houses.csv"
    clean_frame.drop(columns=[config.PRICE_BN]).to_csv(csv, index=False)
    output = tmp_path / "reports"

    assert cli.main(
        ["--data-path", str(csv), "--output-dir", str(output), "--no-plots"]
    ) == 0
    assert not (output / "figures").exists()
