"""MLflow experiment-tracking helpers for the Meituan ETA pipeline.

Design contract:

* Nothing happens at import time. The tracking URI / experiment are only touched
  when the caller explicitly invokes :func:`init_tracking`.
* These helpers never open or close a run; the caller owns the
  ``mlflow.start_run()`` / ``mlflow.end_run()`` boundary. :func:`log_dataset`
  therefore must be called *inside* an active run.
* Data is identified by a content hash of its source file, never by dumping the
  frame into params or tags.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import mlflow.data
import yaml

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps pandas out of import
    import pandas as pd

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "meituan-eta"
KNOWN_LIMITATION = "poi_target_enc_oof_pre_split"

_HASH_CHUNK_BYTES = 1024 * 1024
_VERSION_TAG_CHARS = 12


def init_tracking(params: dict[str, Any]) -> None:
    """Point MLflow at the configured tracking store and experiment.

    ``params['mlflow']`` may be omitted entirely; the defaults mirror
    ``params.yaml``. Callers invoke this once before opening runs.
    """
    mlflow_cfg = params.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", DEFAULT_TRACKING_URI)
    experiment_name = mlflow_cfg.get("experiment_name", DEFAULT_EXPERIMENT_NAME)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def sha256_file(path: str | Path) -> str:
    """Return the lowercase hex SHA-256 digest of a file, streamed in 1 MiB chunks.

    A missing path propagates :class:`FileNotFoundError` from :func:`open`.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_dataset(
    df: pd.DataFrame,
    source_path: str | Path,
    name: str,
    targets: str | list[str],
    context: str,
) -> None:
    """Log a DataFrame as an MLflow dataset plus its content-hash version.

    Requires an active run (the caller owns the run lifecycle). The frame
    contents themselves are never serialised into params or tags.
    """
    dataset = mlflow.data.from_pandas(
        df, source=str(source_path), name=name, targets=targets
    )
    mlflow.log_input(dataset, context=context)
    digest = sha256_file(source_path)
    mlflow.log_param("data_sha256", digest)
    mlflow.set_tag("data_version", digest[:_VERSION_TAG_CHARS])
    mlflow.set_tag("known_limitation", KNOWN_LIMITATION)


def _lock_outs(lock: dict[str, Any]) -> list[dict[str, Any]]:
    """Yield every ``outs[]`` entry of a dvc.lock, in file order."""
    outs = list(lock.get("outs") or [])
    for stage in (lock.get("stages") or {}).values():
        if isinstance(stage, dict):
            outs.extend(stage.get("outs") or [])
    return outs


def read_dvc_md5(
    path: str | Path = "dvc.lock",
    data_path: str | Path | None = None,
) -> str | None:
    """Return a dataset ``md5`` recorded in ``dvc.lock``.

    With ``data_path`` supplied, only the ``outs[]`` entry whose ``path``
    matches it is considered; otherwise the first ``outs[].md5`` wins.
    Returns ``None`` when the lock file is absent, has no ``outs`` entries,
    or the matching entry carries no ``md5``.
    """
    lock_path = Path(path)
    if not lock_path.is_file():
        return None

    with lock_path.open(encoding="utf-8") as handle:
        lock = yaml.safe_load(handle)
    if not isinstance(lock, dict):
        return None

    outs = _lock_outs(lock)
    if data_path is not None:
        wanted = Path(data_path)
        outs = [out for out in outs if Path(str(out.get("path", ""))) == wanted]

    for out in outs:
        md5 = out.get("md5")
        if md5 is not None:
            return str(md5)
    return None
