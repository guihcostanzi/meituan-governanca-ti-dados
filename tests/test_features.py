"""Contract tests for the feature builders in ``meituan_eta.features.build``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from meituan_eta.config import TARGET_COL
from meituan_eta.features.build import (
    add_courier_load,
    add_district_demand,
    add_manhattan_features,
    add_poi_target_encoding,
    add_temporal_features,
    calc_manhattan_dist_km,
)


def test_identical_coordinates_give_exactly_zero_distance() -> None:
    lat = pd.Series([39000000.0, 39000100.0])
    lng = pd.Series([11600000.0, 11600200.0])
    distance = calc_manhattan_dist_km(lat, lng, lat, lng, np.radians(39.0))
    assert (distance == 0.0).all()
    assert distance.tolist() == [0.0, 0.0]


def test_local_midnight_has_zero_sine_and_unit_cosine() -> None:
    # 2024-01-01 00:00 at UTC+8 is 2023-12-31 16:00 UTC.
    local_midnight = int(pd.Timestamp("2024-01-01 00:00:00", tz="Asia/Shanghai").timestamp())
    frame = pd.DataFrame({"platform_order_time": [local_midnight]})

    add_temporal_features(frame)

    assert frame["order_hour"].iloc[0] == 0
    assert abs(frame["sin_order_hour"].iloc[0] - 0.0) <= 1e-12
    assert abs(frame["cos_order_hour"].iloc[0] - 1.0) <= 1e-12


def test_total_distance_is_the_sum_of_both_legs() -> None:
    frame = pd.DataFrame(
        {
            "sender_lat": [39000000.0, 39000100.0, 39000200.0],
            "sender_lng": [11600000.0, 11600150.0, 11600300.0],
            "recipient_lat": [39001000.0, 39001100.0, 39001200.0],
            "recipient_lng": [11601000.0, 11601150.0, 11601300.0],
            "grab_lat": [38999000.0, 38999100.0, 38999200.0],
            "grab_lng": [11599000.0, 11599150.0, 11599300.0],
        }
    )

    add_manhattan_features(frame)

    expected = (
        frame["dist_driver_rest_manhattan_km"] + frame["dist_rest_client_manhattan_km"]
    )
    assert np.array_equal(
        frame["dist_total_driver_trip_manhattan_km"].to_numpy(), expected.to_numpy()
    )


def test_courier_load_counts_concurrent_bags() -> None:
    # Courier 7: A grabs at 100 (arrives 300), B at 200 (arrives 400), C at 500 (arrives 600).
    # A: nothing in progress yet -> 0
    # B: A still travelling -> 1
    # C: A and B both finished by 500 -> 0
    frame = pd.DataFrame(
        {
            "courier_id": [7, 7, 7],
            "grab_time": [100, 200, 500],
            "arrive_time": [300, 400, 600],
        }
    )

    add_courier_load(frame)

    assert frame["carga_entregador"].tolist() == [0, 1, 0]


def test_district_demand_counts_pushes_in_trailing_window() -> None:
    # Same district, pushes at 90, 150, 400; grabs at 100, 200, 500 with a 1200 s window.
    # Every push is within the trailing window of each grab -> running counts 1, 2, 3.
    frame = pd.DataFrame(
        {
            "da_id": [9, 9, 9],
            "order_push_time": [90, 150, 400],
            "grab_time": [100, 200, 500],
        }
    )

    add_district_demand(frame, window_seconds=1200)

    assert frame["da_demanda_20min"].tolist() == [1, 2, 3]


def test_oof_poi_encoding_has_no_nan_and_rare_poi_falls_back_to_global_mean() -> None:
    # 36 rows for poi 1 (target 10), 3 rows for poi 2 (target 30) and a single
    # row for poi 3 (target 30). Global mean is 12.0.
    frame = pd.DataFrame(
        {
            "poi_id": [1] * 36 + [2] * 3 + [3],
            TARGET_COL: [10.0] * 36 + [30.0] * 4,
        }
    )

    report = add_poi_target_encoding(
        frame, target=TARGET_COL, smoothing=10.0, folds=5, seed=42
    )

    encoded = frame["poi_target_enc"]
    assert not encoded.isna().any()
    assert report["n_missing_poi_target_enc"] == 0
    global_mean = report["global_mean_target"]

    # A poi present in every training fold gets a smoothed value distinct from the mean.
    common = frame["poi_id"] == 1
    assert (np.abs(encoded[common] - global_mean) > 1e-6).all()

    # The single-row poi is never in any training fold, so it falls back to the
    # global mean rather than NaN.
    rare = frame["poi_id"] == 3
    assert np.allclose(encoded[rare].to_numpy(), global_mean, atol=1e-12)
