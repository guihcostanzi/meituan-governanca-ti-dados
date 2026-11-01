"""Contract tests for ``meituan_eta.data.validate``.

Uses a synthetic 23-column frame written to a temp CSV (with the leading index
column the real file carries) and drives the module's ``main()`` through a
patched ``sys.argv`` - no shelling out, no raw data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from meituan_eta.data import validate

REPO_ROOT = Path(__file__).resolve().parents[1]

REPORT_KEYS = {
    "n_rows",
    "n_cols",
    "columns",
    "sentinel_zero_counts",
    "boolean_domain_violations",
    "duplicate_waybill_ids",
    "chronology_violations",
    "coordinate_bbox",
}


def make_waybill_frame(n_rows: int = 3) -> pd.DataFrame:
    """Build a coherent synthetic frame with exactly the 23 expected columns."""
    rows = []
    for i in range(n_rows):
        platform = 1000 + i * 10000
        rows.append(
            {
                "dt": "2024-01-01",
                "order_id": 100 + i,
                "waybill_id": f"waybill-{i}",
                "courier_id": 10 + i,
                "da_id": 1,
                "is_courier_grabbed": 1,
                "is_weekend": 0,
                "estimate_arrived_time": platform + 2400,
                "is_prebook": 0,
                "poi_id": 5,
                "sender_lng": 11600000 + i * 100,
                "sender_lat": 39000000 + i * 100,
                "recipient_lng": 11601000 + i * 100,
                "recipient_lat": 39001000 + i * 100,
                "grab_lng": 11599000 + i * 100,
                "grab_lat": 38999000 + i * 100,
                "dispatch_time": platform + 60,
                "grab_time": platform + 120,
                "fetch_time": platform + 300,
                "arrive_time": platform + 1800,
                "estimate_meal_prepare_time": platform + 600,
                "order_push_time": platform + 30,
                "platform_order_time": platform,
            }
        )
    return pd.DataFrame(rows, columns=list(validate.EXPECTED_COLUMNS))


def test_main_writes_report_and_23_column_parquet(tmp_path, monkeypatch) -> None:
    frame = make_waybill_frame()
    csv_path = tmp_path / "waybill.csv"
    out_path = tmp_path / "validated.parquet"
    report_path = tmp_path / "validation.json"
    # Default to_csv() writes the leading unnamed index column the real CSV has.
    frame.to_csv(csv_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            "--params",
            str(REPO_ROOT / "params.yaml"),
            "--waybill",
            str(csv_path),
            "--out",
            str(out_path),
            "--report",
            str(report_path),
        ],
    )

    assert validate.main() == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert REPORT_KEYS <= set(report)
    assert report["n_rows"] == 3
    assert report["n_cols"] == 23
    assert report["columns"] == list(validate.EXPECTED_COLUMNS)
    assert report["duplicate_waybill_ids"] == 0
    assert set(report["sentinel_zero_counts"]) == set(validate.EXPECTED_COLUMNS)
    assert set(report["boolean_domain_violations"]) == {"is_courier_grabbed", "is_prebook", "is_weekend"}
    assert report["chronology_violations"] == {}
    assert report["coordinate_bbox"]["lat_min"] == 38999000.0
    assert report["coordinate_bbox"]["lat_max"] == 39001200.0

    written = pd.read_parquet(out_path)
    assert len(written.columns) == 23
    assert list(written.columns) == list(validate.EXPECTED_COLUMNS)


def test_missing_column_header_exits_nonzero(tmp_path, monkeypatch) -> None:
    frame = make_waybill_frame().drop(columns=["platform_order_time"])
    csv_path = tmp_path / "truncated.csv"
    out_path = tmp_path / "validated.parquet"
    report_path = tmp_path / "validation.json"
    frame.to_csv(csv_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            "--params",
            str(REPO_ROOT / "params.yaml"),
            "--waybill",
            str(csv_path),
            "--out",
            str(out_path),
            "--report",
            str(report_path),
        ],
    )

    assert validate.main() != 0
    assert not out_path.exists()
    assert not report_path.exists()


def test_duplicate_waybill_ids_exit_nonzero(tmp_path, monkeypatch) -> None:
    frame = make_waybill_frame()
    frame.loc[1, "waybill_id"] = frame.loc[0, "waybill_id"]
    csv_path = tmp_path / "dupes.csv"
    frame.to_csv(csv_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            "--params",
            str(REPO_ROOT / "params.yaml"),
            "--waybill",
            str(csv_path),
            "--out",
            str(tmp_path / "validated.parquet"),
            "--report",
            str(tmp_path / "validation.json"),
        ],
    )

    assert validate.main() != 0
