"""Contract tests for the six model configurations in ``meituan_eta.models``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from meituan_eta.config import FEATURES_COMPLETE, FEATURES_REDUCED, load_params
from meituan_eta.models import MODEL_NAMES, STAGES, build_model, feature_list

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    return load_params(str(REPO_ROOT / "params.yaml"))


def test_all_six_combinations_build(params: dict[str, Any]) -> None:
    built = [build_model(model, stage, params) for model in MODEL_NAMES for stage in STAGES]
    assert len(built) == 6
    assert all(flavor in {"sklearn", "lightgbm"} for _, flavor in built)


def test_random_forest_hyperparameters(params: dict[str, Any]) -> None:
    reduced, flavor = build_model("random_forest", "reduced", params)
    reduced_params = reduced.get_params()
    assert flavor == "sklearn"
    assert reduced_params["n_estimators"] == 100
    assert reduced_params["max_depth"] == 16
    assert reduced_params["min_samples_split"] == 10
    assert reduced_params["random_state"] == 42

    complete, _ = build_model("random_forest", "complete", params)
    complete_params = complete.get_params()
    assert complete_params["n_estimators"] == 100
    assert complete_params["max_depth"] == 18
    assert complete_params["min_samples_split"] == 8
    assert complete_params["random_state"] == 42


def test_hist_gradient_boosting_hyperparameters(params: dict[str, Any]) -> None:
    estimator, flavor = build_model("hist_gradient_boosting", "complete", params)
    values = estimator.get_params()

    assert flavor == "sklearn"
    assert values["max_iter"] == 500
    assert values["learning_rate"] == 0.08
    assert values["max_leaf_nodes"] == 31
    assert values["early_stopping"] is True
    assert values["validation_fraction"] == 0.15
    assert values["n_iter_no_change"] == 20
    assert values["random_state"] == 42


def test_lightgbm_hyperparameters(params: dict[str, Any]) -> None:
    reduced, flavor = build_model("lightgbm", "reduced", params)
    reduced_params = reduced.get_params()
    assert flavor == "lightgbm"
    assert reduced_params["n_estimators"] == 300
    assert reduced_params["learning_rate"] == 0.05
    assert reduced_params["num_leaves"] == 31
    assert reduced_params["random_state"] == 42

    complete, _ = build_model("lightgbm", "complete", params)
    assert complete.get_params()["n_estimators"] == 350


def test_feature_list_lengths_and_contents() -> None:
    reduced = feature_list("reduced")
    complete = feature_list("complete")

    assert len(reduced) == 9
    assert len(complete) == 19
    assert reduced == FEATURES_REDUCED
    assert complete == FEATURES_COMPLETE
    assert complete[: len(reduced)] == reduced


def test_unknown_model_raises_value_error(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_model("xgboost", "reduced", params)


def test_unknown_stage_raises_value_error(params: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        build_model("random_forest", "tiny", params)
    with pytest.raises(ValueError, match="unknown stage"):
        feature_list("tiny")


def test_models_import_leaves_lightgbm_lazy() -> None:
    code = "import sys; import meituan_eta.models; print('lightgbm' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
