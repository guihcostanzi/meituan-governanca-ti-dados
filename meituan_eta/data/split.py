"""Deterministic 70/20/10 split of the Meituan feature frame.

Reproduces the notebook's partition block verbatim
(``notebooks/meituan_delivery_prediction_v3.md`` lines 533-545): draw a faithful
``max_samples`` sample with ``random_state=seed`` when the frame is larger, then
call ``train_test_split`` twice with ``test_size=split.test_size`` (0.30) and
``test_size=split.val_fraction_of_rest`` (1/3), never stratifying and never
overriding ``shuffle``. Each split is written as a parquet holding only
``FEATURES_COMPLETE + [TARGET_COL, BASELINE_COL]``; column selection for the
reduced/complete model stages belongs to the training stage, not here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from meituan_eta.config import (
    BASELINE_COL,
    FEATURES_COMPLETE,
    TARGET_COL,
    apply_smoke,
    common_argparser,
    load_params,
)

SPLIT_COLUMNS: list[str] = FEATURES_COMPLETE + [TARGET_COL, BASELINE_COL]


def split_frame(
    frame: pd.DataFrame, seed: int, max_samples: int, test_size: float, val_fraction_of_rest: float
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], dict[str, Any]]:
    """Return ``((train, val, test), report)`` following the notebook exactly."""
    n_input = int(len(frame))
    if n_input > max_samples:
        frame_sample = frame.sample(n=max_samples, random_state=seed).copy()
    else:
        frame_sample = frame.copy()

    train_df, rest_df = train_test_split(frame_sample, test_size=test_size, random_state=seed)
    val_df, test_df = train_test_split(rest_df, test_size=val_fraction_of_rest, random_state=seed)

    n_sample = int(len(frame_sample))
    report: dict[str, Any] = {
        "n_input": n_input,
        "n_sample": n_sample,
        "seed": int(seed),
        "test_size": float(test_size),
        "val_fraction_of_rest": float(val_fraction_of_rest),
        "max_samples": int(max_samples),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "n_features": len(FEATURES_COMPLETE),
        "columns": list(SPLIT_COLUMNS),
        "train_fraction": len(train_df) / n_sample if n_sample else 0.0,
        "val_fraction": len(val_df) / n_sample if n_sample else 0.0,
        "test_fraction": len(test_df) / n_sample if n_sample else 0.0,
        "target": TARGET_COL,
        "baseline": BASELINE_COL,
    }
    return (train_df[SPLIT_COLUMNS], val_df[SPLIT_COLUMNS], test_df[SPLIT_COLUMNS]), report


def main() -> int:
    parser: argparse.ArgumentParser = common_argparser(
        "Split the Meituan feature frame into deterministic train/val/test parquets."
    )
    parser.add_argument("--input", default="data/processed/features.parquet", help="features parquet")
    parser.add_argument("--out-dir", default="data/processed", help="directory for train/val/test parquets")
    parser.add_argument("--report", default="reports/split.json", help="split JSON report")
    args = parser.parse_args()

    started = time.perf_counter()
    params = load_params(args.params)
    if args.smoke:
        params = apply_smoke(params)
    split_params = params["split"]
    seed = int(params["seed"])
    print(
        f"[split] params loaded; smoke={args.smoke} seed={seed} max_samples={split_params['max_samples']:,}"
    )

    frame = pd.read_parquet(args.input)
    print(f"[split] loaded {len(frame):,} rows x {frame.shape[1]} cols from {args.input}")

    (train_df, val_df, test_df), report = split_frame(
        frame,
        seed=seed,
        max_samples=int(split_params["max_samples"]),
        test_size=float(split_params["test_size"]),
        val_fraction_of_rest=float(split_params["val_fraction_of_rest"]),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # index=True keeps the original frame index so train/val/test remain provably disjoint.
    for name, frame_split in (("train", train_df), ("val", val_df), ("test", test_df)):
        path = out_dir / f"{name}.parquet"
        frame_split.to_parquet(path, index=True)
        print(f"[OK] wrote {path}")

    report["elapsed_seconds"] = time.perf_counter() - started
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(f"[split] train {train_df.shape} | val {val_df.shape} | test {test_df.shape}")
    print(f"[split] target range [{train_df[TARGET_COL].min():.3f}, {train_df[TARGET_COL].max():.3f}] min")
    print(f"[split] done in {report['elapsed_seconds']:.2f}s")
    print(f"[OK] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
