"""Training stage: fit the six benchmark configurations and record them in MLflow.

The stage reproduces the notebook fit block (``notebooks/meituan_delivery_prediction_v3.md``
lines 560-596) once per ``(model, stage)`` combination from
``meituan_eta.models``: build the estimator, select the stage feature list, log the
input dataset, fit, score on train and val, log metrics and the model artefact, and
pickle the fitted estimator to ``models/``. The test split is never read here —
``meituan_eta.evaluate`` owns the test metrics.

Each combination owns exactly one ``mlflow.start_run`` block; the run lifecycle is
never closed manually. After the six runs the ``--runs`` JSON is written as the
contract consumed by the evaluation stage (``model``, ``stage``, ``run_id``,
``model_path`` plus the train/val MAE).

Re-execution is idempotent: a combination whose run name already exists in the
experiment reuses that run instead of appending a duplicate, so ``dvc repro`` keeps
the experiment record at exactly six runs (plus the evaluation stage's baseline run).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from math import sqrt
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from meituan_eta.config import TARGET_COL, apply_smoke, common_argparser, load_params
from meituan_eta.models import MODEL_NAMES, STAGES, build_model, feature_list
from meituan_eta.tracking import init_tracking, log_dataset

RUN_PARAM_NAMES: tuple[str, ...] = (
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "max_iter",
    "learning_rate",
    "num_leaves",
    "max_leaf_nodes",
    "early_stopping",
    "validation_fraction",
    "n_iter_no_change",
    "random_state",
    "n_jobs",
)


def _score(y_true: pd.Series, y_pred: Any) -> dict[str, float]:
    """Return MAE, MSE, RMSE and R² as plain floats."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": sqrt(mse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _run_params(
    model: str,
    stage: str,
    params: dict[str, Any],
    estimator: Any,
    n_train: int,
    n_val: int,
    n_features: int,
) -> dict[str, Any]:
    """Flatten the run's primitive params, taking estimator values from ``get_params()``."""
    flat: dict[str, Any] = {
        "model": model,
        "stage": stage,
        "seed": int(params["seed"]),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_features": int(n_features),
    }
    estimator_params = estimator.get_params()
    for name in RUN_PARAM_NAMES:
        if name in estimator_params:
            flat[name] = estimator_params[name]
    return flat


def _log_model(estimator: Any, flavor: str) -> None:
    """Log ``estimator`` as the run's ``model`` artefact using the MLflow 3 signature.

    ``cloudpickle`` is requested explicitly: the skops default refuses tree models
    (``sklearn.tree._tree.Tree``) unless every internal type is declared trusted.
    """
    if flavor == "lightgbm":
        mlflow.lightgbm.log_model(estimator, name="model", serialization_format="cloudpickle")
    else:
        mlflow.sklearn.log_model(estimator, name="model", serialization_format="cloudpickle")


def _existing_run_id(experiment_name: str, run_name: str) -> str | None:
    """Return the id of the most recent run named ``run_name``, or ``None``.

    Reusing the run keeps re-execution idempotent: a stage rerun updates its own
    record instead of appending a duplicate to the experiment.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    found = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
    )
    return None if found.empty else str(found.iloc[0]["run_id"])


def train_one(
    model: str,
    stage: str,
    experiment_name: str,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    models_dir: Path,
    train_path: Path,
) -> dict[str, Any]:
    """Fit, score, log and pickle a single ``(model, stage)`` combination."""
    estimator, flavor = build_model(model, stage, params)
    features = feature_list(stage)

    run_name = f"{model}-{stage}"
    existing_id = _existing_run_id(experiment_name, run_name)
    started_run = (
        mlflow.start_run(run_id=existing_id)
        if existing_id is not None
        else mlflow.start_run(run_name=run_name)
    )

    with started_run as run:
        mlflow.log_params(
            _run_params(model, stage, params, estimator, len(train_df), len(val_df), len(features))
        )
        mlflow.log_param("features", ",".join(features))
        log_dataset(train_df, train_path, f"train-{stage}", TARGET_COL, "training")

        started = time.perf_counter()
        estimator.fit(train_df[features], train_df[TARGET_COL])
        fit_seconds = time.perf_counter() - started

        train_scores = _score(train_df[TARGET_COL], estimator.predict(train_df[features]))
        val_scores = _score(val_df[TARGET_COL], estimator.predict(val_df[features]))
        mlflow.log_metrics(
            {
                f"train_{name}": value for name, value in train_scores.items()
            }
            | {f"val_{name}": value for name, value in val_scores.items()}
        )

        log_started = time.perf_counter()
        _log_model(estimator, flavor)
        run_id = run.info.run_id
        pkl_path = models_dir / f"{model}-{stage}.pkl"
        with pkl_path.open("wb") as handle:
            pickle.dump(estimator, handle)
        log_seconds = time.perf_counter() - log_started

    print(
        f"[train] {model}/{stage} "
        f"train_mae={train_scores['mae']:.4f} val_mae={val_scores['mae']:.4f} "
        f"train_r2={train_scores['r2']:.4f} val_r2={val_scores['r2']:.4f} "
        f"fit={fit_seconds:.1f}s log={log_seconds:.1f}s run_id={run_id}"
    )
    return {
        "model": model,
        "stage": stage,
        "run_id": run_id,
        "model_path": str(pkl_path),
        "train_mae": train_scores["mae"],
        "val_mae": val_scores["mae"],
    }


def main() -> int:
    parser: argparse.ArgumentParser = common_argparser(
        "Train the six Meituan ETA benchmark configurations and record them in MLflow."
    )
    parser.add_argument("--train", default="data/processed/train.parquet", help="train parquet")
    parser.add_argument("--val", default="data/processed/val.parquet", help="validation parquet")
    parser.add_argument("--models-dir", default="models", help="directory for pickled models")
    parser.add_argument("--runs", default="reports/runs.json", help="JSON listing the six runs")
    args = parser.parse_args()

    started_at = time.perf_counter()
    train_path = Path(args.train)
    val_path = Path(args.val)
    for path in (train_path, val_path):
        if not path.is_file():
            print(f"ERROR: input file not found: {path}", file=sys.stderr)
            return 1

    params = load_params(args.params)
    if args.smoke:
        params = apply_smoke(params)
    print(f"[train] params loaded; smoke={args.smoke} seed={int(params['seed'])}")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    print(f"[train] loaded train {train_df.shape} | val {val_df.shape}")

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    runs_path = Path(args.runs)
    runs_path.parent.mkdir(parents=True, exist_ok=True)

    init_tracking(params)
    experiment_name = str(params["mlflow"]["experiment_name"])

    runs: list[dict[str, Any]] = []
    for model in MODEL_NAMES:
        for stage in STAGES:
            runs.append(
                train_one(
                    model,
                    stage,
                    experiment_name,
                    params,
                    train_df,
                    val_df,
                    models_dir,
                    train_path,
                )
            )

    with runs_path.open("w", encoding="utf-8") as handle:
        json.dump(runs, handle, indent=2)
        handle.write("\n")
    print(f"[OK] wrote {runs_path} ({len(runs)} runs)")

    elapsed = time.perf_counter() - started_at
    print(f"[train] {len(runs)} runs in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
