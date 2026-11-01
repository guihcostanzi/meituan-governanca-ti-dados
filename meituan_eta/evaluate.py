"""Evaluation stage: test metrics, the ``baseline-meituan`` reference run and the registry.

The stage consumes the six pickled estimators listed in ``reports/runs.json`` and the
held-out test split. For every ``(model, stage)`` it predicts, computes MAE/MSE/RMSE/R²,
resumes the *same* MLflow run recorded during training, logs the ``test_*`` metrics and
two artefacts (a residual hexbin plus feature importances), and then refreshes the
single ``baseline-meituan`` reference run computed from the official Meituan estimate.

The baseline run is looked up before it is created: ``dvc repro`` re-executes this stage
and the experiment must stay at exactly seven runs, so a second execution resumes the
existing baseline instead of creating an eighth. Nothing is trained or refit here; the
pickled estimators are only read.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import tempfile
import time
import warnings
from math import sqrt
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from meituan_eta.config import (
    BASELINE_COL,
    TARGET_COL,
    apply_smoke,
    common_argparser,
    load_params,
)
from meituan_eta.models import MODEL_NAMES, STAGES, feature_list
from meituan_eta.tracking import DEFAULT_EXPERIMENT_NAME, init_tracking, log_dataset

BASELINE_RUN_NAME = "baseline-meituan"
BASELINE_MODEL = "baseline_meituan"
BASELINE_STAGE = "reference"

REGISTRY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "data_version",
    "model",
    "stage",
    "parameters",
    "mae",
    "mse",
    "rmse",
    "r2",
    "execution_date",
)
HYPERPARAMETER_NAMES: tuple[str, ...] = (
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


def _plot_residuals(y_true: pd.Series, y_pred: Any, path: Path) -> None:
    """Save a predicted-vs-residual hexbin to ``path`` and close the figure."""
    residuals = y_true.to_numpy() - y_pred
    fig, ax = plt.subplots(figsize=(7, 5))
    hexbin = ax.hexbin(y_pred, residuals, gridsize=45, cmap="viridis", mincnt=1)
    ax.axhline(0.0, color="crimson", linewidth=1.0)
    ax.set_xlabel("Predicted total time (min)")
    ax.set_ylabel("Residual: actual - predicted (min)")
    ax.set_title("Test residuals")
    fig.colorbar(hexbin, ax=ax, label="orders")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_importance(importances: dict[str, float], path: Path, title: str) -> None:
    """Save a horizontal bar chart of feature importances to ``path``."""
    height = max(3.0, 0.3 * len(importances) + 1.0)
    fig, ax = plt.subplots(figsize=(8, height))
    if importances:
        names = list(importances)
        ax.barh(names, [importances[name] for name in names], color="steelblue")
        ax.set_xlabel("Importance")
    else:
        ax.text(
            0.5,
            0.5,
            "feature_importances_ unavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _importance_payload(estimator: Any, features: list[str]) -> dict[str, Any]:
    """Return the ``feature_importance.json`` payload for ``estimator``.

    Estimators without ``feature_importances_`` (e.g. histogram gradient boosting)
    yield an empty mapping plus an explanatory note.
    """
    raw = getattr(estimator, "feature_importances_", None)
    if raw is None:
        return {
            "importances": {},
            "note": "estimator exposes no feature_importances_ attribute",
        }
    return {
        "importances": {
            name: float(value) for name, value in zip(features, raw, strict=True)
        },
        "note": "from estimator.feature_importances_",
    }


def _log_artefacts(
    estimator: Any,
    features: list[str],
    y_true: pd.Series,
    y_pred: Any,
) -> None:
    """Render the residual and importance artefacts into a temp dir and log them."""
    payload = _importance_payload(estimator, features)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        residuals_png = tmpdir / "residuals.png"
        importance_png = tmpdir / "feature_importance.png"
        importance_json = tmpdir / "feature_importance.json"
        _plot_residuals(y_true, y_pred, residuals_png)
        _plot_importance(payload["importances"], importance_png, "Feature importance")
        with importance_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        for path in (residuals_png, importance_png, importance_json):
            mlflow.log_artifact(str(path), artifact_path="plots")


def _predict(estimator: Any, frame: pd.DataFrame) -> Any:
    """Predict deterministically.

    Tree ensembles average their leaf values in a thread-dependent order when
    ``n_jobs > 1``, which perturbs the last float bit between executions and would
    make ``metrics.json`` non-reproducible. Inference is cheap, so force one thread;
    the estimator's training hyperparameters (already logged) are untouched.
    """
    params = estimator.get_params()
    if params.get("n_jobs") not in (None, 1):
        estimator.set_params(n_jobs=1)
    return estimator.predict(frame)


def _evaluate_model(
    entry: dict[str, Any], model_path: Path, test_df: pd.DataFrame
) -> dict[str, float]:
    """Resume ``entry['run_id']``, log its test metrics and artefacts, return the scores."""
    with model_path.open("rb") as handle:
        estimator = pickle.load(handle)

    features = feature_list(entry["stage"])
    y_true = test_df[TARGET_COL]
    predictions = _predict(estimator, test_df[features])
    scores = _score(y_true, predictions)

    with mlflow.start_run(run_id=entry["run_id"]):
        mlflow.log_metrics({f"test_{name}": value for name, value in scores.items()})
        _log_artefacts(estimator, features, y_true, predictions)

    return {**scores, "n_test": int(len(test_df))}


def _find_baseline_run_id(experiment_id: str) -> str | None:
    """Return the existing ``baseline-meituan`` run id, or ``None`` when absent."""
    frame = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{BASELINE_RUN_NAME}'",
        max_results=1,
    )
    if frame.empty:
        return None
    return str(frame.iloc[0]["run_id"])


def _log_baseline(test_df: pd.DataFrame, test_path: Path) -> dict[str, float]:
    """Log the test dataset and the four metrics of the official Meituan estimate."""
    scores = _score(test_df[TARGET_COL], test_df[BASELINE_COL])
    # MLflow's local source resolver warns that a relative path is ambiguous; the
    # warning is harmless (task-7) but must not fail a ``-W error::Warning`` run.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The specified dataset source can be interpreted in multiple ways",
        )
        warnings.filterwarnings(
            "ignore",
            message="Hint: Inferred schema contains integer column",
        )
        warnings.filterwarnings(
            "ignore",
            message="Starting with pandas version 4.0 all arguments of all will be keyword-only",
        )
        log_dataset(test_df, str(test_path), "test", TARGET_COL, "testing")
    mlflow.log_metrics({f"test_{name}": value for name, value in scores.items()})
    mlflow.log_params(
        {
            "model": BASELINE_MODEL,
            "stage": BASELINE_STAGE,
            "n_test": int(len(test_df)),
        }
    )
    return {**scores, "n_test": int(len(test_df))}


def _ensure_baseline(
    test_df: pd.DataFrame, experiment_id: str, test_path: Path
) -> tuple[str, dict[str, float]]:
    """Create the baseline run once, then always reuse it on later executions."""
    run_id = _find_baseline_run_id(experiment_id)
    if run_id is None:
        with mlflow.start_run(run_name=BASELINE_RUN_NAME) as run:
            scores = _log_baseline(test_df, test_path)
            run_id = run.info.run_id
    else:
        with mlflow.start_run(run_id=run_id):
            scores = _log_baseline(test_df, test_path)
    return run_id, scores


def _coerce_param(value: str) -> Any:
    """Best-effort numeric/bool coercion of an MLflow string param."""
    if value == "None":
        return None
    if value in {"True", "False"}:
        return value == "True"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parameters(row: dict[str, Any]) -> str:
    """Return a compact JSON string of the run's model hyperparameters."""
    payload: dict[str, Any] = {}
    for name in HYPERPARAMETER_NAMES:
        value = row.get(f"params.{name}")
        if value is None or pd.isna(value):
            continue
        payload[name] = _coerce_param(str(value))
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _text(row: dict[str, Any], key: str) -> str:
    """Return ``row[key]`` as text, or ``''`` when missing."""
    value = row.get(key)
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _execution_date(value: Any) -> str:
    """Format an MLflow ``info.start_time`` timestamp as ISO-8601 UTC with milliseconds."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{stamp.microsecond // 1000:03d}Z"


def _registry_row(row: dict[str, Any]) -> dict[str, str]:
    """Project one ``search_runs`` row onto the registry columns."""
    return {
        "run_id": _text(row, "run_id"),
        "data_version": _text(row, "tags.data_version"),
        "model": _text(row, "params.model"),
        "stage": _text(row, "params.stage"),
        "parameters": _parameters(row),
        "mae": f"{float(row['metrics.test_mae']):.6f}",
        "mse": f"{float(row['metrics.test_mse']):.6f}",
        "rmse": f"{float(row['metrics.test_rmse']):.6f}",
        "r2": f"{float(row['metrics.test_r2']):.6f}",
        "execution_date": _execution_date(row["start_time"]),
    }


def _registry_rows(
    experiment_name: str,
    runs: list[dict[str, Any]],
    baseline_run_id: str,
) -> list[dict[str, str]]:
    """Build the seven registry rows from one ``search_runs`` query."""
    frame = mlflow.search_runs(experiment_names=[experiment_name])
    by_run = {
        str(run_id): row
        for run_id, row in zip(frame["run_id"], frame.to_dict("records"), strict=True)
    }
    lookup = {(entry["model"], entry["stage"]): entry for entry in runs}
    rows = [
        _registry_row(by_run[lookup[(model, stage)]["run_id"]])
        for model in MODEL_NAMES
        for stage in STAGES
        if (model, stage) in lookup
    ]
    rows.append(_registry_row(by_run[baseline_run_id]))
    return rows


def _write_metrics(path: Path, metrics: dict[str, dict[str, float]]) -> None:
    """Write ``metrics.json`` with ``indent=2`` and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the machine-readable registry with the fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=REGISTRY_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the human-readable registry table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(REGISTRY_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in REGISTRY_COLUMNS) + " |",
    ]
    for row in rows:
        cells = [row[column].replace("|", "\\|") for column in REGISTRY_COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser: argparse.ArgumentParser = common_argparser(
        "Evaluate the six Meituan ETA models on the test split and render the registry."
    )
    parser.add_argument("--test", default="data/processed/test.parquet", help="test parquet")
    parser.add_argument("--models-dir", default="models", help="directory holding the pickles")
    parser.add_argument("--runs", default="reports/runs.json", help="JSON listing the six runs")
    parser.add_argument("--metrics", default="reports/metrics.json", help="metrics JSON output")
    parser.add_argument(
        "--registry", default="reports/experiment_registry.md", help="markdown registry"
    )
    parser.add_argument(
        "--registry-csv", default="reports/experiment_registry.csv", help="CSV registry"
    )
    args = parser.parse_args()

    # MLflow's SQLAlchemy store raises a third-party ``noload`` deprecation the first
    # time the store is opened. It is outside this module's control and must not fail
    # a ``-W error::Warning`` run, so silence that single message process-wide.
    warnings.filterwarnings(
        "ignore",
        message="The ``noload`` loader strategy is deprecated",
        category=DeprecationWarning,
    )

    started_at = time.perf_counter()
    test_path = Path(args.test)
    runs_path = Path(args.runs)
    for path in (test_path, runs_path):
        if not path.is_file():
            print(f"ERROR: input file not found: {path}", file=sys.stderr)
            return 1

    with runs_path.open(encoding="utf-8") as handle:
        runs: list[dict[str, Any]] = json.load(handle)

    models_dir = Path(args.models_dir)
    for entry in runs:
        model_path = models_dir / Path(entry["model_path"]).name
        if not model_path.is_file():
            print(f"ERROR: model file not found: {model_path}", file=sys.stderr)
            return 1

    params = load_params(args.params)
    if args.smoke:
        params = apply_smoke(params)

    test_df = pd.read_parquet(test_path)
    print(f"[evaluate] loaded test {test_df.shape} from {test_path}")

    init_tracking(params)
    experiment_name = params.get("mlflow", {}).get(
        "experiment_name", DEFAULT_EXPERIMENT_NAME
    )
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"ERROR: MLflow experiment not found: {experiment_name}", file=sys.stderr)
        return 1

    metrics: dict[str, dict[str, float]] = {}
    for entry in runs:
        model_path = models_dir / Path(entry["model_path"]).name
        scores = _evaluate_model(entry, model_path, test_df)
        metrics[f"{entry['model']}/{entry['stage']}"] = scores
        print(
            f"[evaluate] {entry['model']}/{entry['stage']} "
            f"test_mae={scores['mae']:.4f} test_r2={scores['r2']:.4f}"
        )

    baseline_run_id, baseline_scores = _ensure_baseline(
        test_df, experiment.experiment_id, test_path
    )
    metrics["baseline/meituan"] = baseline_scores
    print(
        f"[evaluate] baseline/meituan test_mae={baseline_scores['mae']:.4f} "
        f"test_r2={baseline_scores['r2']:.4f} run_id={baseline_run_id}"
    )

    _write_metrics(Path(args.metrics), metrics)
    rows = _registry_rows(experiment_name, runs, baseline_run_id)
    _write_csv(Path(args.registry_csv), rows)
    _write_markdown(Path(args.registry), rows)

    print(f"[OK] wrote {args.metrics} ({len(metrics)} entries)")
    print(f"[OK] wrote {args.registry_csv} ({len(rows)} rows)")
    print(f"[OK] wrote {args.registry} ({len(rows)} rows)")
    elapsed = time.perf_counter() - started_at
    print(f"[evaluate] {len(rows)} rows in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
