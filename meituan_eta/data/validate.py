"""Validate the raw Meituan waybill table and publish a faithful parquet copy.

This stage is deliberately non-destructive: it asserts the 23 expected columns
are present and that ``waybill_id`` is unique, writes the table to parquet with
the source index dropped, and emits a JSON report mirroring the completeness /
validity checks of ``quality/VERIFICATION_SPEC.md`` (V-02, V-05, V-07). No rows
are filtered, no column is derived and no value is imputed — that is the prepare
stage's job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from meituan_eta.config import RAW_WAVES, RAW_WAYBILL, apply_smoke, common_argparser, load_params

EXPECTED_COLUMNS: tuple[str, ...] = (
    "dt",
    "order_id",
    "waybill_id",
    "courier_id",
    "da_id",
    "is_courier_grabbed",
    "is_weekend",
    "estimate_arrived_time",
    "is_prebook",
    "poi_id",
    "sender_lng",
    "sender_lat",
    "recipient_lng",
    "recipient_lat",
    "grab_lng",
    "grab_lat",
    "dispatch_time",
    "grab_time",
    "fetch_time",
    "arrive_time",
    "estimate_meal_prepare_time",
    "order_push_time",
    "platform_order_time",
)
BOOLEAN_COLUMNS: tuple[str, ...] = ("is_courier_grabbed", "is_prebook", "is_weekend")
TIMESTAMP_CHAIN: tuple[str, ...] = (
    "platform_order_time",
    "order_push_time",
    "dispatch_time",
    "grab_time",
    "fetch_time",
    "arrive_time",
)
LAT_COLUMNS: tuple[str, ...] = ("sender_lat", "recipient_lat", "grab_lat")
LNG_COLUMNS: tuple[str, ...] = ("sender_lng", "recipient_lng", "grab_lng")
COORDINATE_COLUMNS: tuple[str, ...] = ("sender_lng", "sender_lat", "recipient_lng", "recipient_lat", "grab_lng", "grab_lat")


def _fail(message: str) -> int:
    print(f"ERROR: {message}", file=sys.stderr)
    return 1


def _check_columns(frame: pd.DataFrame) -> int | None:
    missing = [name for name in EXPECTED_COLUMNS if name not in frame.columns]
    if missing:
        return _fail(f"missing expected column(s): {', '.join(missing)}")
    unexpected = [name for name in frame.columns if name not in EXPECTED_COLUMNS]
    if unexpected:
        return _fail(f"unexpected extra column(s): {', '.join(unexpected)}")
    return None


def _chronology_violations(frame: pd.DataFrame) -> dict[str, int]:
    violations: dict[str, int] = {}
    for earlier, later in zip(TIMESTAMP_CHAIN, TIMESTAMP_CHAIN[1:]):
        mask = (frame[earlier] > 0) & (frame[later] > 0) & (frame[earlier] > frame[later])
        count = int(mask.sum())
        if count > 0:
            violations[f"{earlier}>{later}"] = count
    return violations


def _coordinate_bbox(frame: pd.DataFrame) -> dict[str, Any]:
    per_column = {
        name: {"min": float(frame[name].min()), "max": float(frame[name].max())}
        for name in COORDINATE_COLUMNS
    }
    return {
        "method": "aggregate minima/maxima over the six coordinate columns; per_column nested below",
        "lat_min": min(per_column[name]["min"] for name in LAT_COLUMNS),
        "lat_max": max(per_column[name]["max"] for name in LAT_COLUMNS),
        "lng_min": min(per_column[name]["min"] for name in LNG_COLUMNS),
        "lng_max": max(per_column[name]["max"] for name in LNG_COLUMNS),
        "per_column": per_column,
    }


def build_report(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(frame.shape[0]),
        "n_cols": int(frame.shape[1]),
        "columns": list(EXPECTED_COLUMNS),
        "sentinel_zero_counts": {name: int((frame[name] == 0).sum()) for name in EXPECTED_COLUMNS},
        "boolean_domain_violations": {
            name: int((~frame[name].isin([0, 1])).sum()) for name in BOOLEAN_COLUMNS
        },
        "duplicate_waybill_ids": int(frame["waybill_id"].duplicated().sum()),
        "chronology_violations": _chronology_violations(frame),
        "coordinate_bbox": _coordinate_bbox(frame),
    }


def main() -> int:
    parser: argparse.ArgumentParser = common_argparser(
        "Validate the raw Meituan waybill table and write data/interim/validated.parquet."
    )
    parser.add_argument("--waybill", default=RAW_WAYBILL, help="raw waybill CSV")
    parser.add_argument("--waves", default=RAW_WAVES, help="raw courier wave CSV")
    parser.add_argument("--out", default="data/interim/validated.parquet", help="validated parquet")
    parser.add_argument("--report", default="reports/validation.json", help="validation JSON report")
    args = parser.parse_args()

    config = load_params(args.params)
    if args.smoke:
        config = apply_smoke(config)
    print(f"[validate] params loaded ({len(config)} top-level keys); smoke={args.smoke}")

    # index_col=0 drops the leading unnamed index column, leaving the 23 data columns.
    frame = pd.read_csv(args.waybill, index_col=0)

    column_error = _check_columns(frame)
    if column_error is not None:
        return column_error

    duplicate_count = int(frame["waybill_id"].duplicated().sum())
    if duplicate_count:
        return _fail(f"{duplicate_count} duplicate waybill_id value(s) found")

    report = build_report(frame)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    zero_hits = sum(1 for count in report["sentinel_zero_counts"].values() if count > 0)
    total_chronology = sum(report["chronology_violations"].values())
    print(f"[validate] rows={report['n_rows']:,} cols={report['n_cols']} duplicate_waybill_ids=0")
    print(f"[validate] sentinel-zero columns with hits: {zero_hits}/{report['n_cols']}")
    print(f"[validate] boolean domain violations: {report['boolean_domain_violations']}")
    print(f"[validate] chronology violations: total={total_chronology} {report['chronology_violations']}")
    print(
        f"[validate] bbox lat=[{report['coordinate_bbox']['lat_min']:.6f}, "
        f"{report['coordinate_bbox']['lat_max']:.6f}] "
        f"lng=[{report['coordinate_bbox']['lng_min']:.6f}, {report['coordinate_bbox']['lng_max']:.6f}]"
    )
    print(f"[validate] wrote {out_path}")
    print(f"[validate] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
