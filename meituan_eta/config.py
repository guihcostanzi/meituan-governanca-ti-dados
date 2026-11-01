"""Shared constants, feature lists and params helpers for the Meituan ETA pipeline.

Deliberately dependency-light: importing this module must never pull pandas/numpy.
Later stage modules call ``common_argparser()`` and ``apply_smoke(load_params(...))``.
"""

from __future__ import annotations

import argparse
import copy
from typing import Any

import yaml

TZ_OFFSET_HOURS: int = 8
COORD_SCALE: float = 1e6

TARGET_COL: str = "target_total_time_min"
BASELINE_COL: str = "meituan_baseline_min"

FEATURES_REDUCED: list[str] = [
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

FEATURES_COMPLETE: list[str] = FEATURES_REDUCED + [
    "dist_driver_rest_manhattan_km",
    "dist_total_driver_trip_manhattan_km",
    "tempo_ate_aceite_min",
    "queue_wait_min",
    "rem_prep_at_dispatch_min",
    "prior_rejections",
    "carga_entregador",
    "da_demanda_20min",
    "wave_em_andamento",
    "pos_na_wave",
]

RAW_WAYBILL: str = "data/raw/all_waybill_info_meituan_0322.csv"
RAW_WAVES: str = "data/raw/courier_wave_info_meituan.csv"


def load_params(path: str = "params.yaml") -> dict[str, Any]:
    """Load the pipeline params from YAML (FileNotFoundError propagates naturally)."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_smoke(params: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``params`` with the ``smoke`` overrides applied.

    Overrides (only when a ``smoke:`` block is present):
      * ``split.max_samples`` -> ``smoke.max_samples``
      * every ``models.<model>`` key named ``n_estimators*`` -> ``smoke.n_estimators``
        (dict values get every sub-key overwritten, flat numbers are replaced)
      * ``models.hist_gradient_boosting.max_iter`` -> ``smoke.max_iter``

    The input dict is never mutated.
    """
    result = copy.deepcopy(params)
    smoke = result.get("smoke")
    if not smoke:
        return result

    result["split"]["max_samples"] = smoke["max_samples"]

    for model in result.get("models", {}).values():
        for key in model:
            if not key.startswith("n_estimators"):
                continue
            if isinstance(model[key], dict):
                for stage in model[key]:
                    model[key][stage] = smoke["n_estimators"]
            else:
                model[key] = smoke["n_estimators"]

    hist_gb = result.get("models", {}).get("hist_gradient_boosting")
    if hist_gb is not None and "max_iter" in hist_gb:
        hist_gb["max_iter"] = smoke["max_iter"]

    return result


def common_argparser(description: str) -> argparse.ArgumentParser:
    """Build an ArgumentParser carrying the shared ``--params`` / ``--smoke`` flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--params", default="params.yaml", help="path to params.yaml")
    parser.add_argument("--smoke", action="store_true", help="apply smoke-test overrides")
    return parser
