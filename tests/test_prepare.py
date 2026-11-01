"""Contract tests for ``meituan_eta.data.prepare`` sanitation funnel."""

from __future__ import annotations

import pandas as pd

from meituan_eta.config import BASELINE_COL, TARGET_COL
from meituan_eta.data.prepare import prepare


def make_prepare_frame() -> pd.DataFrame:
    """Five hand-built rows that exercise every funnel stage exactly once.

    Row 0: refused (is_courier_grabbed == 0)
    Row 1: pre-booked (is_prebook == 1)
    Row 2: masked-zero time sentinel (estimate_meal_prepare_time == 0)
    Row 3: coherent but 120 minutes (dropped by the [5, 90] range bound)
    Row 4: coherent 30-minute delivery (the only survivor)
    Rows 3 and 4 share order_id 200 so the survivor's ``prior_rejections`` proves
    the count is computed on the raw frame *before* filtering.
    """
    rows = [
        {  # 0: refused
            "order_id": 100,
            "dispatch_time": 1100,
            "is_courier_grabbed": 0,
            "is_prebook": 0,
            "platform_order_time": 1000,
            "estimate_meal_prepare_time": 1300,
            "estimate_arrived_time": 3400,
            "arrive_time": 2800,
            "grab_time": 1150,
        },
        {  # 1: pre-booked
            "order_id": 100,
            "dispatch_time": 1200,
            "is_courier_grabbed": 1,
            "is_prebook": 1,
            "platform_order_time": 1000,
            "estimate_meal_prepare_time": 1300,
            "estimate_arrived_time": 3400,
            "arrive_time": 2800,
            "grab_time": 1250,
        },
        {  # 2: time sentinel
            "order_id": 300,
            "dispatch_time": 3100,
            "is_courier_grabbed": 1,
            "is_prebook": 0,
            "platform_order_time": 3000,
            "estimate_meal_prepare_time": 0,
            "estimate_arrived_time": 5200,
            "arrive_time": 5000,
            "grab_time": 3150,
        },
        {  # 3: 120-minute row (same order as row 4, dispatched first)
            "order_id": 200,
            "dispatch_time": 2100,
            "is_courier_grabbed": 1,
            "is_prebook": 0,
            "platform_order_time": 2000,
            "estimate_meal_prepare_time": 2300,
            "estimate_arrived_time": 9800,
            "arrive_time": 9200,
            "grab_time": 2150,
        },
        {  # 4: valid 30-minute row (same order as row 3, dispatched second)
            "order_id": 200,
            "dispatch_time": 3100,
            "is_courier_grabbed": 1,
            "is_prebook": 0,
            "platform_order_time": 3000,
            "estimate_meal_prepare_time": 3300,
            "estimate_arrived_time": 5000,
            "arrive_time": 4800,
            "grab_time": 3150,
        },
    ]
    frame = pd.DataFrame(rows)
    for column in (
        "sender_lat",
        "sender_lng",
        "recipient_lat",
        "recipient_lng",
        "grab_lat",
        "grab_lng",
    ):
        frame[column] = 100.0
    return frame


def test_funnel_counts_match_the_five_hand_built_rows() -> None:
    clean, report = prepare(make_prepare_frame(), min_minutes=5.0, max_minutes=90.0)

    assert report["n_total"] == 5
    assert report["n_grabbed"] == 4
    assert report["n_prebook"] == 3
    assert report["n_time_sentinel"] == 2
    assert report["n_clean"] == 1
    assert len(clean) == 1
    assert report["dropped"]["grabbed"] == 1
    assert report["dropped"]["prebook"] == 1
    assert report["dropped"]["time_sentinel"] == 1
    assert report["dropped"]["range"] == 1
    assert report["dropped"]["coord_sentinel"] == 0
    assert report["dropped"]["dispatch_order"] == 0
    assert report["dropped"]["chronology"] == 0


def test_prior_rejections_is_computed_before_filtering() -> None:
    clean, _ = prepare(make_prepare_frame(), min_minutes=5.0, max_minutes=90.0)

    # The survivor is the second waybill of order 200. If prior_rejections were
    # counted after the 120-minute row was dropped it would be 0, not 1.
    assert int(clean["prior_rejections"].iloc[0]) == 1
    # And the filtered-out first waybill still consumed a rejection slot.
    assert int(clean["prior_rejections"].max()) == 1


def test_derived_columns_survive_with_expected_formulas() -> None:
    clean, _ = prepare(make_prepare_frame(), min_minutes=5.0, max_minutes=90.0)
    row = clean.iloc[0]

    assert "temp_total_time_min" in clean.columns
    assert row["temp_total_time_min"] == 30.0
    assert row[TARGET_COL] == (4800 - 3000) / 60.0 == 30.0
    assert row[BASELINE_COL] == (5000 - 3000) / 60.0
    assert abs(row[BASELINE_COL] - (2000 / 60.0)) < 1e-12
