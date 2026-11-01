"""Clean the validated Meituan waybill table and publish the modelling frame.

Reproduces the notebook's sanitation block step for step and in the same order
(``notebooks/meituan_delivery_prediction_v3.md`` lines 211-255): drop refused
waybills, drop pre-booked orders, drop masked-zero sentinels in timestamps and
GPS coordinates, enforce chronology and finally bound the delivery duration to
``[min_total_time_min, max_total_time_min]``. ``prior_rejections`` is computed
on the raw frame *before* any filter, exactly like the notebook. The cleaned
frame keeps every raw column plus ``prior_rejections``, ``temp_total_time_min``,
``target_total_time_min`` and ``meituan_baseline_min``, and a JSON report records
the row count after each step so later stages can audit the funnel.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from meituan_eta.config import BASELINE_COL, TARGET_COL, apply_smoke, common_argparser, load_params

TIME_COLUMNS: tuple[str, ...] = (
    "platform_order_time",
    "dispatch_time",
    "estimate_meal_prepare_time",
    "estimate_arrived_time",
    "arrive_time",
    "grab_time",
)
COORD_COLUMNS: tuple[str, ...] = (
    "sender_lat",
    "sender_lng",
    "recipient_lat",
    "recipient_lng",
    "grab_lat",
    "grab_lng",
)
DERIVED_COLUMNS: tuple[str, ...] = ("prior_rejections", "temp_total_time_min", TARGET_COL, BASELINE_COL)


def _pct(count: int, total: int) -> float:
    return count / total * 100.0 if total else 0.0


def _print_step(label: str, count: int, total: int) -> None:
    print(f"[prepare] {label:<34} {count:>9,} rows ({_pct(count, total):5.1f}% of total)")


def _filter_positive(
    frame: pd.DataFrame, columns: tuple[str, ...], total: int, label: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep rows whose value is strictly positive for every column, in order."""
    drops: dict[str, int] = {}
    for column in columns:
        before = len(frame)
        frame = frame[frame[column] > 0]
        drops[column] = before - len(frame)
        if drops[column]:
            print(f"[prepare]   {label} {column}: -{drops[column]:,}")
    return frame, drops


def prepare(frame_raw: pd.DataFrame, min_minutes: float, max_minutes: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the notebook sanitation funnel and return ``(clean_frame, report)``."""
    # prior_rejections BEFORE filtering, over the whole raw frame (notebook lines 166-170).
    frame = frame_raw.sort_values(["order_id", "dispatch_time"]).reset_index(drop=True)
    frame["prior_rejections"] = frame.groupby("order_id").cumcount()
    n_total = int(len(frame))
    print(f"[prepare] raw rows: {n_total:,} rows x {frame.shape[1]} cols")
    print(f"[prepare] prior_rejections > 0: {int((frame['prior_rejections'] > 0).sum()):,} waybills")

    frame = frame[frame["is_courier_grabbed"] == 1].copy()
    n_grabbed = int(len(frame))
    _print_step("grabbed (is_courier_grabbed==1)", n_grabbed, n_total)

    frame = frame[frame["is_prebook"] == 0].copy()
    n_prebook = int(len(frame))
    _print_step("immediate (is_prebook==0)", n_prebook, n_total)

    frame, time_drops = _filter_positive(frame, TIME_COLUMNS, n_total, "time-sentinel")
    n_time_sentinel = int(len(frame))
    _print_step("time sentinels (>0)", n_time_sentinel, n_total)

    frame, coord_drops = _filter_positive(frame, COORD_COLUMNS, n_total, "coord-sentinel")
    n_coord_sentinel = int(len(frame))
    _print_step("coord sentinels (>0)", n_coord_sentinel, n_total)

    frame = frame[frame["dispatch_time"] >= frame["platform_order_time"]].copy()
    n_dispatch_order = int(len(frame))
    _print_step("dispatch_time >= platform_order_time", n_dispatch_order, n_total)

    frame = frame[frame["arrive_time"] > frame["dispatch_time"]].copy()
    n_chronology = int(len(frame))
    _print_step("arrive_time > dispatch_time", n_chronology, n_total)

    frame["temp_total_time_min"] = (frame["arrive_time"] - frame["platform_order_time"]) / 60.0
    frame = frame[
        (frame["temp_total_time_min"] >= min_minutes) & (frame["temp_total_time_min"] <= max_minutes)
    ].copy()
    n_clean = int(len(frame))
    _print_step(f"duration in [{min_minutes:g}, {max_minutes:g}] min", n_clean, n_total)

    # Target/baseline after the funnel; temp_total_time_min is kept side by side.
    frame[TARGET_COL] = (frame["arrive_time"] - frame["platform_order_time"]) / 60.0
    frame[BASELINE_COL] = (frame["estimate_arrived_time"] - frame["platform_order_time"]) / 60.0

    report: dict[str, Any] = {
        "n_total": n_total,
        "n_grabbed": n_grabbed,
        "n_prebook": n_prebook,
        "n_time_sentinel": n_time_sentinel,
        "n_coord_sentinel": n_coord_sentinel,
        "n_dispatch_order": n_dispatch_order,
        "n_chronology": n_chronology,
        "n_clean": n_clean,
        "dropped": {
            "grabbed": n_total - n_grabbed,
            "prebook": n_grabbed - n_prebook,
            "time_sentinel": n_prebook - n_time_sentinel,
            "coord_sentinel": n_time_sentinel - n_coord_sentinel,
            "dispatch_order": n_coord_sentinel - n_dispatch_order,
            "chronology": n_dispatch_order - n_chronology,
            "range": n_chronology - n_clean,
        },
        "sentinel_drops": {"time": time_drops, "coord": coord_drops},
        "filter_order": [
            "is_courier_grabbed == 1",
            "is_prebook == 0",
            "time columns > 0 (platform_order_time, dispatch_time, estimate_meal_prepare_time, "
            "estimate_arrived_time, arrive_time, grab_time)",
            "coordinate columns > 0 (sender_lat, sender_lng, recipient_lat, recipient_lng, grab_lat, grab_lng)",
            "dispatch_time >= platform_order_time",
            "arrive_time > dispatch_time",
            f"{min_minutes:g} <= total_time_min <= {max_minutes:g}",
        ],
        "min_total_time_min": float(min_minutes),
        "max_total_time_min": float(max_minutes),
        "target": TARGET_COL,
        "baseline": BASELINE_COL,
    }
    return frame, report


def main() -> int:
    parser: argparse.ArgumentParser = common_argparser(
        "Remove sentinel zeros and incoherent rows from the validated Meituan table."
    )
    parser.add_argument("--input", default="data/interim/validated.parquet", help="validated parquet")
    parser.add_argument("--out", default="data/interim/clean.parquet", help="clean parquet")
    parser.add_argument("--report", default="reports/prepare.json", help="prepare JSON report")
    args = parser.parse_args()

    started = time.perf_counter()
    params = load_params(args.params)
    if args.smoke:
        params = apply_smoke(params)
    clean_params = params["clean"]
    print(f"[prepare] params loaded; smoke={args.smoke}")

    frame_raw = pd.read_parquet(args.input)
    frame, report = prepare(
        frame_raw,
        min_minutes=float(clean_params["min_total_time_min"]),
        max_minutes=float(clean_params["max_total_time_min"]),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    target = frame[TARGET_COL]
    baseline = frame[BASELINE_COL]
    elapsed = time.perf_counter() - started
    print(f"[prepare] target  mean={target.mean():.3f} median={target.median():.3f} min={target.min():.3f} max={target.max():.3f}")
    print(f"[prepare] baseline mean={baseline.mean():.3f} median={baseline.median():.3f} min={baseline.min():.3f} max={baseline.max():.3f}")
    print(f"[prepare] kept {len(frame):,} rows x {frame.shape[1]} cols in {elapsed:.2f}s")
    print(f"[OK] wrote {out_path}")
    print(f"[OK] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
