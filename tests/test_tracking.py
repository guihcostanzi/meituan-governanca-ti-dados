"""Contract tests for ``meituan_eta.tracking`` against a throwaway sqlite store."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mlflow
import pandas as pd
import pytest
import yaml

from meituan_eta.tracking import (
    KNOWN_LIMITATION,
    init_tracking,
    log_dataset,
    read_dvc_md5,
    sha256_file,
)


def _init_tmp_tracking(tmp_path: Path) -> str:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    init_tracking(
        {"mlflow": {"tracking_uri": uri, "experiment_name": "tracking-unit-test"}}
    )
    return uri


def test_init_tracking_sets_uri_and_experiment(tmp_path: Path) -> None:
    uri = _init_tmp_tracking(tmp_path)

    assert mlflow.get_tracking_uri() == uri
    assert mlflow.get_experiment_by_name("tracking-unit-test") is not None


def test_sha256_file_matches_hashlib_reference(tmp_path: Path) -> None:
    payload = b"meituan-eta deterministic payload"
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "does-not-exist.bin")


def test_log_dataset_writes_hash_version_tags_and_one_input(tmp_path: Path) -> None:
    _init_tmp_tracking(tmp_path)
    source = tmp_path / "train.parquet"
    frame = pd.DataFrame({"feature": [1.0, 2.0], "target": [3.0, 4.0]})
    frame.to_parquet(source)
    expected_digest = hashlib.sha256(source.read_bytes()).hexdigest()

    with mlflow.start_run(run_name="dataset-unit") as run:
        log_dataset(frame, source, "train", "target", "training")
        run_id = run.info.run_id

    logged = mlflow.get_run(run_id)
    assert logged.data.params["data_sha256"] == expected_digest
    assert logged.data.tags["data_version"] == expected_digest[:12]
    assert len(logged.data.tags["data_version"]) == 12
    assert logged.data.tags["known_limitation"] == KNOWN_LIMITATION
    assert len(logged.inputs.dataset_inputs) == 1


def test_read_dvc_md5_returns_none_for_missing_lock(tmp_path: Path) -> None:
    assert read_dvc_md5(tmp_path / "dvc.lock") is None


def test_read_dvc_md5_reads_stage_output_md5(tmp_path: Path) -> None:
    lock = tmp_path / "dvc.lock"
    lock.write_text(
        yaml.safe_dump(
            {
                "schema": "2.0",
                "stages": {
                    "features": {
                        "outs": [
                            {
                                "path": "data/processed/features.parquet",
                                "md5": "deadbeefcafef00d",
                            }
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert read_dvc_md5(lock) == "deadbeefcafef00d"
    assert (
        read_dvc_md5(lock, data_path="data/processed/features.parquet")
        == "deadbeefcafef00d"
    )
    assert read_dvc_md5(lock, data_path="data/processed/other.parquet") is None
