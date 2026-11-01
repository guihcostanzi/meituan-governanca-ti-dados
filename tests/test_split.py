"""Contract tests for the deterministic 70/20/10 split in ``meituan_eta.data.split``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meituan_eta.config import BASELINE_COL, FEATURES_COMPLETE, TARGET_COL
from meituan_eta.data.split import SPLIT_COLUMNS, split_frame


def make_split_frame(n_rows: int = 100) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {
        name: np.arange(n_rows, dtype="float64") + offset
        for offset, name in enumerate(FEATURES_COMPLETE)
    }
    data[TARGET_COL] = np.linspace(10.0, 20.0, n_rows)
    data[BASELINE_COL] = np.linspace(11.0, 19.0, n_rows)
    frame = pd.DataFrame(data)
    assert list(frame.columns) == SPLIT_COLUMNS
    return frame


def _split(frame: pd.DataFrame):
    return split_frame(
        frame,
        seed=42,
        max_samples=100,
        test_size=0.30,
        val_fraction_of_rest=1.0 / 3.0,
    )


def test_split_is_deterministic_70_20_10() -> None:
    frame = make_split_frame(100)
    (train, val, test), report = _split(frame)

    assert (len(train), len(val), len(test)) == (70, 20, 10)
    assert report["n_input"] == 100
    assert report["n_sample"] == 100
    assert (report["n_train"], report["n_val"], report["n_test"]) == (70, 20, 10)

    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)
    assert len(set(train.index) | set(val.index) | set(test.index)) == 100


def test_second_split_call_yields_identical_indices() -> None:
    frame = make_split_frame(100)
    (train_a, val_a, test_a), _ = _split(frame)
    (train_b, val_b, test_b), _ = _split(frame)

    assert train_a.index.equals(train_b.index)
    assert val_a.index.equals(val_b.index)
    assert test_a.index.equals(test_b.index)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(val_a, val_b)
    pd.testing.assert_frame_equal(test_a, test_b)
