"""Model factories for the six benchmark configurations (3 models x 2 feature stages).

Mirrors the constructors of ``notebooks/meituan_delivery_prediction_v3.md`` (cells 571-596).
Every hyperparameter that lives in ``params.yaml`` is read from the params dict, never
hardcoded. ``lightgbm`` is imported lazily inside its branch so sklearn-only environments
can import this module without the optional dependency installed.
"""

from __future__ import annotations

from typing import Any, Literal, TypeGuard, assert_never

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from meituan_eta.config import FEATURES_COMPLETE, FEATURES_REDUCED

ModelName = Literal["random_forest", "hist_gradient_boosting", "lightgbm"]
Stage = Literal["reduced", "complete"]
Flavor = Literal["sklearn", "lightgbm"]

MODEL_NAMES: tuple[ModelName, ...] = (
    "random_forest",
    "hist_gradient_boosting",
    "lightgbm",
)
STAGES: tuple[Stage, ...] = ("reduced", "complete")


def _is_model_name(value: str) -> TypeGuard[ModelName]:
    return value in MODEL_NAMES


def _is_stage(value: str) -> TypeGuard[Stage]:
    return value in STAGES


def _random_forest(params: dict[str, Any], stage: Stage) -> tuple[Any, Flavor]:
    cfg = params["models"]["random_forest"]
    estimator = RandomForestRegressor(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg[stage]["max_depth"],
        min_samples_split=cfg[stage]["min_samples_split"],
        random_state=params["seed"],
        n_jobs=-1,
    )
    return estimator, "sklearn"


def _hist_gradient_boosting(params: dict[str, Any], stage: Stage) -> tuple[Any, Flavor]:
    cfg = params["models"]["hist_gradient_boosting"]
    estimator = HistGradientBoostingRegressor(
        max_iter=cfg["max_iter"],
        learning_rate=cfg["learning_rate"],
        max_leaf_nodes=cfg["max_leaf_nodes"],
        early_stopping=cfg["early_stopping"],
        validation_fraction=cfg["validation_fraction"],
        n_iter_no_change=cfg["n_iter_no_change"],
        random_state=params["seed"],
    )
    return estimator, "sklearn"


def _lightgbm(params: dict[str, Any], stage: Stage) -> tuple[Any, Flavor]:
    from lightgbm import LGBMRegressor  # lazy: keep lightgbm optional for sklearn-only envs

    cfg = params["models"]["lightgbm"]
    estimator = LGBMRegressor(
        n_estimators=cfg["n_estimators"][stage],
        learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"],
        random_state=params["seed"],
        n_jobs=-1,
        verbose=-1,
    )
    return estimator, "lightgbm"


def build_model(name: str, stage: str, params: dict[str, Any]) -> tuple[Any, Flavor]:
    """Return ``(estimator, flavor)`` for one of the six benchmark configurations.

    ``name`` is one of ``MODEL_NAMES`` and ``stage`` one of ``STAGES``; anything else
    raises ``ValueError`` naming the supported values.
    """
    if not _is_model_name(name):
        raise ValueError(
            f"unknown model {name!r}; supported models: {', '.join(MODEL_NAMES)}"
        )
    if not _is_stage(stage):
        raise ValueError(f"unknown stage {stage!r}; supported stages: {', '.join(STAGES)}")

    match name:
        case "random_forest":
            return _random_forest(params, stage)
        case "hist_gradient_boosting":
            return _hist_gradient_boosting(params, stage)
        case "lightgbm":
            return _lightgbm(params, stage)
        case _:
            assert_never(name)


def feature_list(stage: str) -> list[str]:
    """Return the ordered feature list for ``stage`` (``'reduced'`` or ``'complete'``)."""
    if not _is_stage(stage):
        raise ValueError(f"unknown stage {stage!r}; supported stages: {', '.join(STAGES)}")

    match stage:
        case "reduced":
            return FEATURES_REDUCED
        case "complete":
            return FEATURES_COMPLETE
        case _:
            assert_never(stage)
