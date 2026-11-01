"""Build the modelling feature frame from the cleaned Meituan waybill table.

Reproduces the notebook's feature block (``notebooks/meituan_delivery_prediction_v3.md``
lines 287-410) in the same order: Manhattan (L1) distances, local-time cyclic encoding,
operational latencies, concurrent courier load, district demand in the last 20 minutes,
wave merge and out-of-fold POI target encoding.

The POI target encoding is computed on the FULL cleaned frame, *before* any train/test
split. That is a known leakage limitation inherited from the notebook; it is reproduced
verbatim (never silently fixed) and tagged ``known_limitation: poi_target_enc_oof_pre_split``
in the JSON report so later stages can audit it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from meituan_eta.config import COORD_SCALE, TARGET_COL, TZ_OFFSET_HOURS, common_argparser, load_params

KM_PER_DEGREE = 111.32
MINUTES_PER_DAY = 1440.0
KNOWN_LIMITATION = "poi_target_enc_oof_pre_split"


def calc_manhattan_dist_km(
    lat1: pd.Series,
    lng1: pd.Series,
    lat2: pd.Series,
    lng2: pd.Series,
    mean_lat_rad: float,
) -> pd.Series:
    """Manhattan (L1) distance in km between two scaled GPS points."""
    dlat = np.abs((lat2 - lat1) / COORD_SCALE * KM_PER_DEGREE)
    dlng = np.abs((lng2 - lng1) / COORD_SCALE * KM_PER_DEGREE * np.cos(mean_lat_rad))
    return dlat + dlng


def add_manhattan_features(frame: pd.DataFrame) -> None:
    mean_lat_rad = float(np.radians(frame["sender_lat"].mean() / COORD_SCALE))
    frame["dist_rest_client_manhattan_km"] = calc_manhattan_dist_km(
        frame["sender_lat"], frame["sender_lng"],
        frame["recipient_lat"], frame["recipient_lng"], mean_lat_rad,
    )
    frame["dist_driver_rest_manhattan_km"] = calc_manhattan_dist_km(
        frame["grab_lat"], frame["grab_lng"],
        frame["sender_lat"], frame["sender_lng"], mean_lat_rad,
    )
    frame["dist_total_driver_trip_manhattan_km"] = (
        frame["dist_driver_rest_manhattan_km"] + frame["dist_rest_client_manhattan_km"]
    )
    print("[features] manhattan: rest->client, driver->rest, total (L1, 111.32 km/deg)")


def add_temporal_features(frame: pd.DataFrame) -> None:
    order_dt_local = pd.to_datetime(frame["platform_order_time"], unit="s") + pd.Timedelta(hours=TZ_OFFSET_HOURS)
    order_hour = order_dt_local.dt.hour
    frame["order_hour"] = order_hour
    frame["day_of_week"] = order_dt_local.dt.day_of_week
    frame["is_friday"] = (order_dt_local.dt.day_of_week == 4).astype(int)
    minute_of_day = order_hour * 60 + order_dt_local.dt.minute
    frame["sin_order_hour"] = np.sin(2 * np.pi * minute_of_day / MINUTES_PER_DAY)
    frame["cos_order_hour"] = np.cos(2 * np.pi * minute_of_day / MINUTES_PER_DAY)
    print(f"[features] temporal: UTC+{TZ_OFFSET_HOURS} local hour/day_of_week + sin/cos")


def add_latency_features(frame: pd.DataFrame) -> None:
    frame["est_meal_prep_min"] = (frame["estimate_meal_prepare_time"] - frame["platform_order_time"]) / 60.0
    frame["queue_wait_min"] = (frame["dispatch_time"] - frame["platform_order_time"]) / 60.0
    frame["rem_prep_at_dispatch_min"] = (frame["estimate_meal_prepare_time"] - frame["dispatch_time"]) / 60.0
    frame["tempo_ate_aceite_min"] = (frame["grab_time"] - frame["platform_order_time"]) / 60.0
    print("[features] latencies: meal prep, queue wait, remaining prep, time-to-accept")


def add_courier_load(frame: pd.DataFrame) -> None:
    """Concurrent orders in a courier's bag at the instant of acceptance."""
    carga = np.zeros(len(frame), dtype=np.int32)
    for pos in frame.groupby("courier_id").indices.values():
        group = frame.iloc[pos]
        started = np.sort(group["grab_time"].to_numpy())
        finished = np.sort(group["arrive_time"].to_numpy())
        accepted = group["grab_time"].to_numpy()
        carga[pos] = np.searchsorted(started, accepted, "left") - np.searchsorted(finished, accepted, "right")
    frame["carga_entregador"] = carga
    print(f"[features] courier load: {int(frame['courier_id'].nunique())} couriers, max={int(carga.max())}")


def add_district_demand(frame: pd.DataFrame, window_seconds: int) -> None:
    """Orders pushed in the same district within the trailing demand window."""
    demanda = np.zeros(len(frame), dtype=np.int32)
    for pos in frame.groupby("da_id").indices.values():
        group = frame.iloc[pos]
        pushed = np.sort(group["order_push_time"].to_numpy())
        accepted = group["grab_time"].to_numpy()
        demanda[pos] = np.searchsorted(pushed, accepted, "right") - np.searchsorted(
            pushed, accepted - window_seconds, "left"
        )
    frame["da_demanda_20min"] = demanda
    print(f"[features] district demand: window={window_seconds}s, max={int(demanda.max())}")


def merge_waves(frame: pd.DataFrame, waves_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not waves_path.exists():
        print(f"[features] waves: {waves_path} missing -> wave columns set to 0")
        frame["wave_em_andamento"] = 0
        frame["pos_na_wave"] = 0
        return frame, {
            "waves_file": str(waves_path),
            "waves_missing": True,
            "wave_rows_matched": 0,
            "wave_coverage_pct": 0.0,
        }

    waves = pd.read_csv(waves_path)
    waves["order_list"] = waves["order_ids"].str.strip("[]").str.split(", ")
    long = waves[["dt", "courier_id", "wave_id", "wave_start_time", "order_list"]].explode("order_list")
    long = long[long["order_list"].str.len() > 0].copy()
    long["order_id"] = long["order_list"].astype("int64")
    long = long.drop(columns="order_list").drop_duplicates(["order_id", "courier_id"])

    frame = frame.merge(
        long[["order_id", "courier_id", "wave_id", "wave_start_time"]],
        on=["order_id", "courier_id"],
        how="left",
    )
    frame["wave_em_andamento"] = (frame["grab_time"] > frame["wave_start_time"]).fillna(0).astype(int)
    wave_position = frame.sort_values("grab_time").groupby(["dt", "courier_id", "wave_id"]).cumcount()
    frame["pos_na_wave"] = wave_position.reindex(frame.index).fillna(0).astype(int)

    matched = int(frame["wave_id"].notna().sum())
    coverage = matched / len(frame) * 100.0
    print(f"[features] waves: {len(waves):,} waves, {matched:,} rows matched ({coverage:.1f}%)")
    return frame, {
        "waves_file": str(waves_path),
        "waves_missing": False,
        "wave_rows_matched": matched,
        "wave_coverage_pct": coverage,
    }


def add_poi_target_encoding(
    frame: pd.DataFrame, target: str, smoothing: float, folds: int, seed: int
) -> dict[str, float]:
    """Out-of-fold, smoothed target encoding of ``poi_id`` (notebook K-Fold block)."""
    global_mean = float(frame[target].mean())
    enc = pd.Series(index=frame.index, dtype="float64")
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in kfold.split(frame):
        stats = frame.iloc[train_idx].groupby("poi_id")[target].agg(["count", "mean"])
        smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        mapped = frame.iloc[val_idx]["poi_id"].map(smoothed)
        enc.iloc[val_idx] = mapped.to_numpy()
    frame["poi_target_enc"] = enc.fillna(global_mean)
    print(f"[features] poi target encoding: {folds} folds, smoothing={smoothing:g}, global_mean={global_mean:.3f}")
    return {
        "n_poi": int(frame["poi_id"].nunique()),
        "global_mean_target": global_mean,
        "poi_target_enc_mean": float(frame["poi_target_enc"].mean()),
        "poi_target_enc_std": float(frame["poi_target_enc"].std()),
        "n_missing_poi_target_enc": int(frame["poi_target_enc"].isna().sum()),
    }


def build_features(frame: pd.DataFrame, params: dict[str, Any], waves_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_params = params["features"]
    seed = int(params["seed"])

    frame = frame.sort_values("grab_time").reset_index(drop=True)
    add_manhattan_features(frame)
    add_temporal_features(frame)
    add_latency_features(frame)
    add_courier_load(frame)
    add_district_demand(frame, int(feature_params["demand_window_seconds"]))
    frame, wave_report = merge_waves(frame, waves_path)
    encoding_report = add_poi_target_encoding(
        frame,
        target=TARGET_COL,
        smoothing=float(feature_params["target_encoding_smoothing"]),
        folds=int(feature_params["target_encoding_folds"]),
        seed=seed,
    )

    report: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "n_cols_out": int(frame.shape[1]),
        "dist_rest_client_mean_km": float(frame["dist_rest_client_manhattan_km"].mean()),
        "dist_driver_rest_mean_km": float(frame["dist_driver_rest_manhattan_km"].mean()),
        "dist_total_mean_km": float(frame["dist_total_driver_trip_manhattan_km"].mean()),
        "dist_rest_client_max_km": float(frame["dist_rest_client_manhattan_km"].max()),
        "dist_driver_rest_max_km": float(frame["dist_driver_rest_manhattan_km"].max()),
        "dist_total_max_km": float(frame["dist_total_driver_trip_manhattan_km"].max()),
        "carga_entregador_max": int(frame["carga_entregador"].max()),
        "da_demanda_20min_max": int(frame["da_demanda_20min"].max()),
        **wave_report,
        **encoding_report,
        "known_limitation": KNOWN_LIMITATION,
    }
    print(
        f"[features] manhattan means: rest->client={report['dist_rest_client_mean_km']:.2f} km | "
        f"driver->rest={report['dist_driver_rest_mean_km']:.2f} km | total={report['dist_total_mean_km']:.2f} km"
    )
    return frame, report


def main() -> int:
    parser = common_argparser("Build the Meituan ETA feature frame (Manhattan, temporal, load, waves, POI TE).")
    parser.add_argument("--input", default="data/interim/clean.parquet", help="clean parquet")
    parser.add_argument("--waves", default="data/courier_wave_info_meituan.csv", help="courier wave CSV")
    parser.add_argument("--out", default="data/processed/features.parquet", help="features parquet")
    parser.add_argument("--report", default="reports/features.json", help="features JSON report")
    args = parser.parse_args()

    started = time.perf_counter()
    params = load_params(args.params)
    print(f"[features] params loaded; input={args.input}")

    frame_in = pd.read_parquet(args.input)
    print(f"[features] loaded {len(frame_in):,} rows x {frame_in.shape[1]} cols")
    frame, report = build_features(frame_in, params, Path(args.waves))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)

    report["elapsed_seconds"] = time.perf_counter() - started
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(f"[features] kept {len(frame):,} rows x {frame.shape[1]} cols in {report['elapsed_seconds']:.2f}s")
    print(f"[OK] wrote {out_path}")
    print(f"[OK] wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
