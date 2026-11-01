"""Contract tests for ``meituan_eta.config`` (feature lists, params, smoke, argparse)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import meituan_eta.config as config

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_REDUCED = [
    "dist_rest_client_manhattan_km",
    "order_hour",
    "day_of_week",
    "is_weekend",
    "sin_order_hour",
    "cos_order_hour",
    "est_meal_prep_min",
    "da_id",
    "poi_target_enc",
]


def test_features_reduced_has_exactly_nine_names_in_order() -> None:
    assert config.FEATURES_REDUCED == EXPECTED_REDUCED
    assert len(config.FEATURES_REDUCED) == 9


def test_features_complete_is_nineteen_with_reduced_prefix() -> None:
    assert len(config.FEATURES_COMPLETE) == 19
    assert config.FEATURES_COMPLETE[:9] == EXPECTED_REDUCED
    # The ten additional columns are unique and none duplicate the reduced prefix.
    assert len(set(config.FEATURES_COMPLETE)) == 19


def test_load_params_round_trips_seed_and_max_samples() -> None:
    params = config.load_params(str(REPO_ROOT / "params.yaml"))
    assert params["seed"] == 42
    assert params["split"]["max_samples"] == 500000


def test_apply_smoke_deep_copies_and_applies_overrides() -> None:
    original = config.load_params(str(REPO_ROOT / "params.yaml"))
    smoked = config.apply_smoke(original)

    # Overrides landed on the copy.
    assert smoked["split"]["max_samples"] == 5000
    assert smoked["models"]["random_forest"]["n_estimators"] == 20
    assert smoked["models"]["lightgbm"]["n_estimators"] == {"reduced": 20, "complete": 20}
    assert smoked["models"]["hist_gradient_boosting"]["max_iter"] == 20

    # apply_smoke never mutates its input.
    assert original["split"]["max_samples"] == 500000
    assert original["models"]["random_forest"]["n_estimators"] == 100
    assert original["models"]["lightgbm"]["n_estimators"] == {"reduced": 300, "complete": 350}
    assert original["models"]["hist_gradient_boosting"]["max_iter"] == 500
    assert original is not smoked


def test_common_argparser_parses_params_and_smoke() -> None:
    parser = config.common_argparser("probe")
    args = parser.parse_args(["--params", "custom.yaml", "--smoke"])
    assert args.params == "custom.yaml"
    assert args.smoke is True

    defaults = parser.parse_args([])
    assert defaults.params == "params.yaml"
    assert defaults.smoke is False


def test_config_import_does_not_pull_pandas() -> None:
    """Importing ``meituan_eta.config`` must stay dependency-light (no pandas)."""
    code = "import sys; import meituan_eta.config; print('pandas' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
