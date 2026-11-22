"""
Unified Meituan ETA dataloader (INFORMS TSL 2024) for all project models.

Consolidates data cleaning, causal feature engineering, English column naming,
and leak-free post-split feature generation from notebook 4
(``4_meituan_delivery_prediction_v4.ipynb``) into a single shared interface for
Random Forest, Gradient Boosting, MLP/DCN, FT-Transformer, and TabPFN.

Key design principles:
1. Always returns pre-split DataFrames: ``(train, val, test)`` or ``(train, test)``
   when ``include_val=False``, preventing accidental data leakage across splits.
2. Removes raw future/internal timestamps (``arrive_time``, ``fetch_time``,
   ``estimate_arrived_time``, etc.) from the returned 29-column DataFrames.
3. Computes leak-sensitive transformations strictly AFTER splitting:
   - ``restaurant_target_enc`` via 5-fold out-of-fold K-Fold smoothing on ``train``.
   - Manhattan reference latitude (``ref_lat_rad``) from ``train`` only.

Typical usage (from ``notebooks_analysis/``):

    import sys
    sys.path.append("../data")
    from dataloader import load_meituan, FEATURES_REDUCED, FEATURES_COMPLETE, TARGET

    train_df, val_df, test_df = load_meituan(split="stratified_district")
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "cache"

WAYBILL_ZIP = "all_waybill_info_meituan_0322.csv.zip"
WAYBILL_CSV = "all_waybill_info_meituan_0322.csv"
WAVE_CSV = "courier_wave_info_meituan.csv"

TZ = pd.Timedelta(hours=8)      # Local operation timezone (China, UTC+8)
COORD_SCALE = 1e6               # Raw coordinates are stored as degrees * 1e6
EARTH_RADIUS_KM = 6371.0088
DEMAND_WINDOW_SEC = 20 * 60
ETA_WINDOW_SEC = 30 * 60
PREP_WINDOW_SEC = 60 * 60

TARGET = "target_total_time_min"
BASELINE = "meituan_baseline_min"

# Backward-compatible Portuguese alias
ALVO = TARGET

RAW_TIME_COLS = [
    "platform_order_time",
    "dispatch_time",
    "estimate_meal_prepare_time",
    "estimate_arrived_time",
    "arrive_time",
    "grab_time",
]
RAW_COORD_COLS = [
    "sender_lat",
    "sender_lng",
    "recipient_lat",
    "recipient_lng",
    "grab_lat",
    "grab_lng",
]

RAW_RENAME_MAP = {
    "poi_id": "restaurant_id",
    "da_id": "district_id",
    "sender_lat": "restaurant_lat",
    "sender_lng": "restaurant_lng",
    "recipient_lat": "customer_lat",
    "recipient_lng": "customer_lng",
    "grab_lat": "courier_lat",
    "grab_lng": "courier_lng",
}

ID_COLS = ["restaurant_id", "district_id", "courier_id"]
GPS_COLS = [
    "restaurant_lat",
    "restaurant_lng",
    "customer_lat",
    "customer_lng",
    "courier_lat",
    "courier_lng",
]
DISTANCE_COLS = [
    "dist_rest_client_km",
    "dist_driver_rest_km",
    "dist_total_driver_trip_km",
]
TEMPORAL_COLS = [
    "order_hour",
    "day_of_week",
    "is_weekend",
    "sin_order_hour",
    "cos_order_hour",
]
LATENCY_COLS = [
    "est_meal_prep_min",
    "time_to_grab_min",
    "queue_wait_min",
    "rem_prep_at_dispatch_min",
]
DYNAMICS_COLS = [
    "prior_rejections",
    "courier_active_orders",
    "district_demand_20min",
    "is_in_wave",
    "wave_position",
]
EXTRA_PRESPLIT_COLS = [
    "restaurant_active_orders",
    "district_recent_eta_30min",
    "district_recent_delay_30min",
    "district_demand_order_20min",
    "district_active_couriers_order_20min",
    "supply_demand_ratio_order",
    "restaurant_recent_prep_min_60min",
    "district_active_couriers_20min",
    "supply_demand_ratio",
    "district_recent_eta_dispatch_30min",
    "district_recent_delay_dispatch_30min",
]
EXTRA_POSTSPLIT_COLS = [
    "courier_target_enc",
    "courier_speed_enc",
]

PRESPLIT_COLS = (
    [TARGET, BASELINE]
    + ID_COLS
    + GPS_COLS
    + TEMPORAL_COLS
    + LATENCY_COLS
    + DYNAMICS_COLS
    + EXTRA_PRESPLIT_COLS
)

FINAL_COLS = (
    [TARGET, BASELINE]
    + ID_COLS
    + GPS_COLS
    + DISTANCE_COLS
    + TEMPORAL_COLS
    + LATENCY_COLS
    + DYNAMICS_COLS
    + ["restaurant_target_enc"]
    + EXTRA_PRESPLIT_COLS
    + EXTRA_POSTSPLIT_COLS
)

DISTANCES = ("haversine", "manhattan")
SPLITS = ("70_30", "80_20", "stratified_district", "district")

SPLIT_ALIASES = {
    "70_30": "70_30",
    "70/30": "70_30",
    "70_20_10": "70_30",
    "random": "70_30",
    "aleatorio": "70_30",
    "80_20": "80_20",
    "80/20": "80_20",
    "80_10_10": "80_20",
    "stratified_district": "stratified_district",
    "estratificado_distrito": "stratified_district",
    "stratified": "stratified_district",
    "estratificado": "stratified_district",
    "district": "district",
    "distrito": "district",
    "spatial": "district",
}

# Standard feature sets (English) — two operational stages from Notebook 4
FEATURES_REDUCED = [
    "dist_rest_client_km",
    "restaurant_lat",
    "restaurant_lng",
    "customer_lat",
    "customer_lng",
    "order_hour",
    "day_of_week",
    "is_weekend",
    "sin_order_hour",
    "cos_order_hour",
    "est_meal_prep_min",
    "district_id",
    "restaurant_target_enc",
]

FEATURES_COMPLETE = FEATURES_REDUCED + [
    "courier_lat",
    "courier_lng",
    "dist_driver_rest_km",
    "dist_total_driver_trip_km",
    "time_to_grab_min",
    "queue_wait_min",
    "rem_prep_at_dispatch_min",
    "prior_rejections",
    "courier_active_orders",
    "district_demand_20min",
    "is_in_wave",
    "wave_position",
]

# Extra dynamic & courier feature sets (Degrau 1: numerical dynamics, Degrau 2: + high-cardinality IDs)
EXTRA_REDUCED_COLS = [
    "restaurant_active_orders",
    "district_recent_eta_30min",
    "district_recent_delay_30min",
    "district_demand_order_20min",
    "district_active_couriers_order_20min",
    "supply_demand_ratio_order",
    "restaurant_recent_prep_min_60min",
]

EXTRA_DISPATCH_COLS = [
    "district_active_couriers_20min",
    "supply_demand_ratio",
    "district_recent_eta_dispatch_30min",
    "district_recent_delay_dispatch_30min",
    "courier_target_enc",
    "courier_speed_enc",
]

FEATURES_REDUCED_EXTRA = FEATURES_REDUCED + EXTRA_REDUCED_COLS
FEATURES_COMPLETE_EXTRA = FEATURES_COMPLETE + EXTRA_REDUCED_COLS + EXTRA_DISPATCH_COLS

# Degrau 2 adds raw high-cardinality categorical IDs causally available at each stage
FEATURES_REDUCED_EXTRA_CAT = FEATURES_REDUCED_EXTRA + ["restaurant_id"]
FEATURES_COMPLETE_EXTRA_CAT = FEATURES_COMPLETE_EXTRA + ["restaurant_id", "courier_id"]

CAT_COLS_BASE = ["district_id"]
CAT_COLS_REDUCED_EXTRA = ["district_id", "restaurant_id"]
CAT_COLS_COMPLETE_EXTRA = ["district_id", "restaurant_id", "courier_id"]

# Backward-compatible aliases
DISTANCIAS = DISTANCES
FEATURES_REDUZIDO = FEATURES_REDUCED
FEATURES_COMPLETO = FEATURES_COMPLETE


def _haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    """Great-circle distance between micro-degree coordinates, in km."""
    p1 = np.radians(lat1 / COORD_SCALE)
    p2 = np.radians(lat2 / COORD_SCALE)
    dphi = p2 - p1
    dlam = np.radians((lng2 - lng1) / COORD_SCALE)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _manhattan_km(lat1, lng1, lat2, lng2, ref_lat_rad: float) -> np.ndarray:
    """Orthogonal grid (L1) distance using training-set reference latitude, in km."""
    dlat = np.abs((lat2 - lat1) / COORD_SCALE * 111.32)
    dlng = np.abs((lng2 - lng1) / COORD_SCALE * 111.32 * np.cos(ref_lat_rad))
    return dlat + dlng


def _read_raw_files(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Read raw waybill archive and optional courier wave CSV."""
    with zipfile.ZipFile(data_dir / WAYBILL_ZIP) as zf:
        # Open explicitly by CSV name because the archive includes __MACOSX metadata
        waybill = pd.read_csv(zf.open(WAYBILL_CSV), index_col=0)
    wave_path = data_dir / WAVE_CSV
    wave = pd.read_csv(wave_path) if wave_path.exists() else None
    return waybill, wave


def _clean_waybills(
    df: pd.DataFrame,
    target_min: float,
    target_max: float,
    verbose: bool,
) -> pd.DataFrame:
    """Apply the 6-step data quality and target range filters."""
    steps = [("raw", len(df))]

    df = df[df["is_courier_grabbed"] == 1]
    steps.append(("accepted waybills only", len(df)))

    df = df[df["is_prebook"] == 0]
    steps.append(("non-prebook orders", len(df)))

    df = df[(df[RAW_TIME_COLS] > 0).all(axis=1)]
    steps.append(("valid non-zero timestamps", len(df)))

    df = df[(df[RAW_COORD_COLS] > 0).all(axis=1)]
    steps.append(("valid non-zero GPS coordinates", len(df)))

    df = df[
        (df["dispatch_time"] >= df["platform_order_time"])
        & (df["arrive_time"] > df["dispatch_time"])
    ]
    steps.append(("consistent event chronology", len(df)))

    total_min = (df["arrive_time"] - df["platform_order_time"]) / 60.0
    df = df[(total_min >= target_min) & (total_min <= target_max)].copy()
    steps.append((f"target in [{target_min:g}, {target_max:g}] min", len(df)))

    if verbose:
        n0 = steps[0][1]
        print("Cleaning summary:")
        for name, count in steps:
            print(f"  {name:<34} {count:>9,}  ({count / n0:6.1%})")
    return df


def _compute_courier_active_orders(df: pd.DataFrame) -> np.ndarray:
    """Count orders already grabbed and not yet delivered by the courier at grab_time."""
    out = np.zeros(len(df), dtype=np.int32)
    for _, pos in df.groupby("courier_id").indices.items():
        g = df.iloc[pos]
        grabbed = np.sort(g["grab_time"].values)
        arrived = np.sort(g["arrive_time"].values)
        t = g["grab_time"].values
        out[pos] = np.searchsorted(grabbed, t, "left") - np.searchsorted(arrived, t, "right")
    return out


def _compute_district_demand(
    df: pd.DataFrame,
    window_sec: int,
    query_col: str = "grab_time",
    event_col: str = "order_push_time",
) -> np.ndarray:
    """Count orders pushed/created in the same district during [query_time - window, query_time]."""
    out = np.zeros(len(df), dtype=np.int32)
    for _, pos in df.groupby("district_id").indices.items():
        g = df.iloc[pos]
        pushed = np.sort(g[event_col].values)
        t = g[query_col].values
        out[pos] = (
            np.searchsorted(pushed, t, "right")
            - np.searchsorted(pushed, t - window_sec, "left")
        )
    return out


def _compute_restaurant_active_orders(df: pd.DataFrame) -> np.ndarray:
    """
    Count orders at the same restaurant created strictly before platform_order_time
    and not yet fetched at platform_order_time (platform_order_time_prev < t < fetch_time_prev).
    """
    out = np.zeros(len(df), dtype=np.int32)
    order_all = df["platform_order_time"].to_numpy(dtype=np.int64)
    fetch_all = df["fetch_time"].to_numpy(dtype=np.int64)
    valid = (fetch_all > 0) & (fetch_all >= order_all)
    for _, pos in df.groupby("restaurant_id").indices.items():
        pos_v = pos[valid[pos]]
        if len(pos_v) == 0:
            continue
        started = np.sort(order_all[pos_v])
        fetched = np.sort(fetch_all[pos_v])
        t = order_all[pos]
        n_started = np.searchsorted(started, t, "left")
        n_fetched = np.searchsorted(fetched, t, "right")
        out[pos] = np.maximum(0, n_started - n_fetched)
    return out


def _compute_window_mean_by_group(
    df: pd.DataFrame,
    group_col: str,
    event_col: str,
    query_col: str,
    values: np.ndarray,
    window_sec: int,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Causal rolling mean of `values` for past events in [query_time - window_sec, query_time).
    Returns NaN when no completed events exist in the window (imputed post-split with train stats).
    """
    out = np.full(len(df), np.nan, dtype=np.float64)
    ev_all = df[event_col].to_numpy(dtype=np.int64)
    q_all = df[query_col].to_numpy(dtype=np.int64)
    for _, pos in df.groupby(group_col).indices.items():
        pos_ev = pos if valid_mask is None else pos[valid_mask[pos]]
        if len(pos_ev) == 0:
            continue
        ev_t = ev_all[pos_ev]
        order = np.argsort(ev_t, kind="mergesort")
        ev_sorted = ev_t[order]
        val_sorted = values[pos_ev][order]
        csum = np.concatenate(([0.0], np.cumsum(val_sorted, dtype=np.float64)))
        q_t = q_all[pos]
        hi = np.searchsorted(ev_sorted, q_t, "left")
        lo = np.searchsorted(ev_sorted, q_t - window_sec, "left")
        cnt = hi - lo
        m = cnt > 0
        if np.any(m):
            out[pos[m]] = (csum[hi[m]] - csum[lo[m]]) / cnt[m]
    return out


def _compute_district_active_couriers(
    df: pd.DataFrame,
    window_sec: int,
    query_col: str = "grab_time",
) -> np.ndarray:
    """
    Count distinct active couriers who grabbed an order in the same district
    during [query_time - window_sec, query_time) using an O(N) two-pointer sweep.
    """
    out = np.zeros(len(df), dtype=np.int32)
    grab_all = df["grab_time"].to_numpy(dtype=np.int64)
    cour_all = df["courier_id"].to_numpy(dtype=np.int64)
    q_all = df[query_col].to_numpy(dtype=np.int64)
    for _, pos in df.groupby("district_id").indices.items():
        ev_order = np.argsort(grab_all[pos], kind="mergesort")
        ev_t = grab_all[pos][ev_order]
        ev_c = cour_all[pos][ev_order]

        q_order = np.argsort(q_all[pos], kind="mergesort")
        q_t = q_all[pos][q_order]

        counts: dict[int, int] = {}
        distinct = 0
        lo = 0
        hi = 0
        n_ev = len(ev_t)
        res = np.zeros(len(pos), dtype=np.int32)

        for idx, t in enumerate(q_t):
            while hi < n_ev and ev_t[hi] < t:
                c = int(ev_c[hi])
                prev = counts.get(c, 0)
                if prev == 0:
                    distinct += 1
                counts[c] = prev + 1
                hi += 1
            t_low = t - window_sec
            while lo < hi and ev_t[lo] < t_low:
                c = int(ev_c[lo])
                rem = counts[c] - 1
                if rem == 0:
                    del counts[c]
                    distinct -= 1
                else:
                    counts[c] = rem
                lo += 1
            res[idx] = distinct

        out[pos[q_order]] = res
    return out


def _build_presplit_features(df: pd.DataFrame, wave: pd.DataFrame | None) -> pd.DataFrame:
    """Construct causal pre-split features and standardize columns to English."""
    df = df.rename(columns=RAW_RENAME_MAP).copy()

    # Target & platform baseline (minutes)
    df[TARGET] = (df["arrive_time"] - df["platform_order_time"]) / 60.0
    df[BASELINE] = (df["estimate_arrived_time"] - df["platform_order_time"]) / 60.0

    # Local temporal features (UTC+8)
    ts = pd.to_datetime(df["platform_order_time"], unit="s") + TZ
    df["order_hour"] = ts.dt.hour.astype(np.int32)
    df["day_of_week"] = ts.dt.dayofweek.astype(np.int32)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.int8)
    minute_of_day = df["order_hour"] * 60 + ts.dt.minute
    df["sin_order_hour"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    df["cos_order_hour"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)

    # Operational and kitchen latencies (minutes)
    df["est_meal_prep_min"] = (df["estimate_meal_prepare_time"] - df["platform_order_time"]) / 60.0
    df["time_to_grab_min"] = (df["grab_time"] - df["platform_order_time"]) / 60.0
    df["queue_wait_min"] = (df["dispatch_time"] - df["platform_order_time"]) / 60.0
    df["rem_prep_at_dispatch_min"] = (df["estimate_meal_prepare_time"] - df["dispatch_time"]) / 60.0

    # Causal supply/demand sweep features ordered by grab_time
    df = df.sort_values("grab_time", kind="stable").reset_index(drop=True)
    df["courier_active_orders"] = _compute_courier_active_orders(df)
    df["district_demand_20min"] = _compute_district_demand(df, DEMAND_WINDOW_SEC)

    # Extra causal dynamic features (checkout-anchored at platform_order_time and dispatch-anchored at grab_time)
    eta_vals = df[TARGET].to_numpy(dtype=np.float64)
    delay_vals = (df[TARGET] - df[BASELINE]).to_numpy(dtype=np.float64)
    order_t = df["platform_order_time"].to_numpy(dtype=np.int64)
    fetch_t = df["fetch_time"].to_numpy(dtype=np.int64)
    valid_fetch = (fetch_t > 0) & (fetch_t >= order_t)
    prep_vals = np.where(valid_fetch, (fetch_t - order_t) / 60.0, np.nan)

    df["restaurant_active_orders"] = _compute_restaurant_active_orders(df)
    df["district_recent_eta_30min"] = _compute_window_mean_by_group(
        df, "district_id", "arrive_time", "platform_order_time", eta_vals, ETA_WINDOW_SEC
    )
    df["district_recent_delay_30min"] = _compute_window_mean_by_group(
        df, "district_id", "arrive_time", "platform_order_time", delay_vals, ETA_WINDOW_SEC
    )
    df["district_demand_order_20min"] = _compute_district_demand(
        df, DEMAND_WINDOW_SEC, query_col="platform_order_time", event_col="platform_order_time"
    )
    df["district_active_couriers_order_20min"] = _compute_district_active_couriers(
        df, DEMAND_WINDOW_SEC, query_col="platform_order_time"
    )
    df["supply_demand_ratio_order"] = (
        df["district_demand_order_20min"] / (df["district_active_couriers_order_20min"] + 1.0)
    )
    df["restaurant_recent_prep_min_60min"] = _compute_window_mean_by_group(
        df, "restaurant_id", "fetch_time", "platform_order_time", prep_vals, PREP_WINDOW_SEC, valid_mask=valid_fetch
    )

    df["district_active_couriers_20min"] = _compute_district_active_couriers(
        df, DEMAND_WINDOW_SEC, query_col="grab_time"
    )
    df["supply_demand_ratio"] = (
        df["district_demand_20min"] / (df["district_active_couriers_20min"] + 1.0)
    )
    df["district_recent_eta_dispatch_30min"] = _compute_window_mean_by_group(
        df, "district_id", "arrive_time", "grab_time", eta_vals, ETA_WINDOW_SEC
    )
    df["district_recent_delay_dispatch_30min"] = _compute_window_mean_by_group(
        df, "district_id", "arrive_time", "grab_time", delay_vals, ETA_WINDOW_SEC
    )

    # Courier wave / batching features
    if wave is not None:
        w = wave.copy()
        w["order_list"] = w["order_ids"].str.strip("[]").str.split(", ")
        wave_long = w[["dt", "courier_id", "wave_id", "wave_start_time", "order_list"]].explode("order_list")
        wave_long = wave_long[wave_long["order_list"].str.len() > 0].copy()
        wave_long["order_id"] = wave_long["order_list"].astype("int64")
        wave_long = wave_long.drop(columns="order_list").drop_duplicates(["order_id", "courier_id"])

        df = df.merge(
            wave_long[["order_id", "courier_id", "wave_id", "wave_start_time"]],
            on=["order_id", "courier_id"],
            how="left",
        )
        df["is_in_wave"] = (df["grab_time"] > df["wave_start_time"]).astype(np.int8)
        df["wave_position"] = (
            df.sort_values("grab_time", kind="stable")
            .groupby(["dt", "courier_id", "wave_id"])
            .cumcount()
            .reindex(df.index)
            .fillna(0)
            .astype(np.int32)
        )
    else:
        print("[warning] courier_wave_info_meituan.csv not found: setting is_in_wave=0 and wave_position=0")
        df["is_in_wave"] = np.int8(0)
        df["wave_position"] = np.int32(0)

    # Drop all raw timestamps and internal identifiers to prevent downstream leakage
    return df[PRESPLIT_COLS].copy()


def _load_presplit_base(
    target_min: float,
    target_max: float,
    data_dir: Path,
    use_cache: bool,
    verbose: bool,
) -> pd.DataFrame:
    """Load or build the clean pre-split DataFrame (without target encoding or split-dependent stats)."""
    cache_path = CACHE_DIR / f"base_clean_v5_extra_alvo{target_min:g}-{target_max:g}.parquet"
    source_path = data_dir / WAYBILL_ZIP
    wave_path = data_dir / WAVE_CSV

    cache_fresh = (
        use_cache
        and cache_path.exists()
        and cache_path.stat().st_mtime > source_path.stat().st_mtime
        and (not wave_path.exists() or cache_path.stat().st_mtime > wave_path.stat().st_mtime)
    )
    if cache_fresh:
        df = pd.read_parquet(cache_path)
        if list(df.columns) == PRESPLIT_COLS:
            if verbose:
                print(f"[cache] {cache_path.name}: {len(df):,} rows x {df.shape[1]} pre-split columns")
            return df

    t0 = time.time()
    waybill, wave = _read_raw_files(data_dir)

    # Count prior rejections before discarding ungrabbed waybills
    waybill = waybill.sort_values(["order_id", "dispatch_time"], kind="stable").reset_index(drop=True)
    waybill["prior_rejections"] = waybill.groupby("order_id").cumcount().astype(np.int32)

    df = _clean_waybills(waybill, target_min, target_max, verbose)
    df = _build_presplit_features(df, wave)

    if verbose:
        print(f"Pre-split base ready: {len(df):,} rows x {df.shape[1]} columns ({time.time() - t0:.1f}s)")

    if use_cache:
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            df.to_parquet(cache_path, index=False)
            if verbose:
                print(f"[cache] saved to data/cache/{cache_path.name}")
        except ImportError:
            print("[warning] install pyarrow to enable parquet caching (pip install pyarrow)")

    return df


def _resolve_split_name(split: str) -> str:
    key = split.strip().lower()
    if key not in SPLIT_ALIASES:
        valid = ", ".join(SPLITS)
        raise ValueError(f"Unsupported split={split!r}. Valid options: {valid}")
    return SPLIT_ALIASES[key]


def _split_dataset(
    df: pd.DataFrame,
    split: str,
    seed: int,
    include_val: bool,
    verbose: bool,
    **split_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Split clean DataFrame into (train, val, test) or (train, None, test)."""
    canonical = _resolve_split_name(split)

    if canonical == "70_30":
        tr, rest = train_test_split(df, test_size=0.30, random_state=seed)
        if include_val:
            val, te = train_test_split(rest, test_size=1.0 / 3.0, random_state=seed)
        else:
            val, te = None, rest

    elif canonical == "80_20":
        tr, rest = train_test_split(df, test_size=0.20, random_state=seed)
        if include_val:
            val, te = train_test_split(rest, test_size=0.50, random_state=seed)
        else:
            val, te = None, rest

    elif canonical == "stratified_district":
        tr, rest = train_test_split(
            df, test_size=0.30, stratify=df["district_id"], random_state=seed
        )
        if include_val:
            val, te = train_test_split(
                rest, test_size=1.0 / 3.0, stratify=rest["district_id"], random_state=seed
            )
        else:
            val, te = None, rest

    else:  # "district"
        test_districts = tuple(
            split_kwargs.get("test_districts", split_kwargs.get("distritos_teste", (8, 9)))
        )
        val_districts = tuple(
            split_kwargs.get("val_districts", split_kwargs.get("distritos_validacao", (6, 7)))
        )
        m_te = df["district_id"].isin(test_districts)
        if include_val:
            m_val = df["district_id"].isin(val_districts)
            tr, val, te = df[~(m_te | m_val)], df[m_val], df[m_te]
            assert not (set(tr["district_id"]) & set(val["district_id"])), "District leakage in val"
            assert not (set(val["district_id"]) & set(te["district_id"])), "District leakage between val and test"
        else:
            tr, val, te = df[~m_te], None, df[m_te]
        assert not (set(tr["district_id"]) & set(te["district_id"])), "District leakage in test"

    if verbose:
        n = len(df)
        parts = [("train", tr)]
        if val is not None:
            parts.append(("val", val))
        parts.append(("test", te))
        print(f"Split '{canonical}' (seed={seed}):")
        for name, part in parts:
            print(f"  {name:<6} {len(part):>8,} rows ({len(part) / n:5.1%})")

    return (
        tr.reset_index(drop=True),
        val.reset_index(drop=True) if val is not None else None,
        te.reset_index(drop=True),
    )


def _add_distances(
    tr: pd.DataFrame,
    val: pd.DataFrame | None,
    te: pd.DataFrame,
    distance: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """Compute spatial distances post-split (Manhattan reference latitude uses train only)."""
    if distance == "haversine":
        dist_fn = _haversine_km
    else:
        # Compute mean latitude strictly on train to eliminate spatial data leakage
        ref_lat_rad = float(np.radians(tr["restaurant_lat"].mean() / COORD_SCALE))
        dist_fn = lambda a, b, c, d: _manhattan_km(a, b, c, d, ref_lat_rad)  # noqa: E731

    for part in (tr, val, te):
        if part is None:
            continue
        part["dist_rest_client_km"] = dist_fn(
            part["restaurant_lat"],
            part["restaurant_lng"],
            part["customer_lat"],
            part["customer_lng"],
        )
        part["dist_driver_rest_km"] = dist_fn(
            part["courier_lat"],
            part["courier_lng"],
            part["restaurant_lat"],
            part["restaurant_lng"],
        )
        part["dist_total_driver_trip_km"] = (
            part["dist_driver_rest_km"] + part["dist_rest_client_km"]
        )
    return tr, val, te


def _add_restaurant_target_encoding(
    tr: pd.DataFrame,
    val: pd.DataFrame | None,
    te: pd.DataFrame,
    target: str = TARGET,
    smoothing: float = 10.0,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Compute smoothed historical restaurant ETA (``restaurant_target_enc``) without leakage.

    - ``train``: out-of-fold K-Fold means so each row never sees its own target.
    - ``val`` and ``test``: smoothed means fitted exclusively on ``train``.
    - Unseen restaurants fall back to the global ``train`` mean.
    """
    global_mean = float(tr[target].mean())

    def _smoothed_means(frame: pd.DataFrame) -> pd.Series:
        stats = frame.groupby("restaurant_id")[target].agg(["count", "mean"])
        return (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )

    enc_tr = np.full(len(tr), global_mean, dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_idx, enc_idx in kf.split(tr):
        fold_map = _smoothed_means(tr.iloc[fit_idx])
        enc_tr[enc_idx] = (
            tr["restaurant_id"].iloc[enc_idx].map(fold_map).fillna(global_mean).values
        )

    full_train_map = _smoothed_means(tr)
    tr["restaurant_target_enc"] = enc_tr
    if val is not None:
        val["restaurant_target_enc"] = (
            val["restaurant_id"].map(full_train_map).fillna(global_mean).values
        )
    te["restaurant_target_enc"] = (
        te["restaurant_id"].map(full_train_map).fillna(global_mean).values
    )
    return tr, val, te


def _add_extra_postsplit_features(
    tr: pd.DataFrame,
    val: pd.DataFrame | None,
    te: pd.DataFrame,
    target: str = TARGET,
    smoothing: float = 10.0,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Post-split leak-free imputation of rolling-window NaNs and out-of-fold courier encodings:
    1. Empty rolling windows are filled strictly using `train` statistics (`tr[TARGET].mean()`, `0.0`,
       or the order's `est_meal_prep_min`).
    2. `courier_target_enc` (historical courier delivery time in min) and `courier_speed_enc`
       (historical courier effective speed in km/h) use 5-fold out-of-fold Bayesian smoothing on `train`.
    """
    global_eta_mean = float(tr[target].mean())

    for part in (tr, val, te):
        if part is None:
            continue
        part["district_recent_eta_30min"] = part["district_recent_eta_30min"].fillna(global_eta_mean)
        part["district_recent_eta_dispatch_30min"] = (
            part["district_recent_eta_dispatch_30min"].fillna(global_eta_mean)
        )
        part["district_recent_delay_30min"] = part["district_recent_delay_30min"].fillna(0.0)
        part["district_recent_delay_dispatch_30min"] = (
            part["district_recent_delay_dispatch_30min"].fillna(0.0)
        )
        part["restaurant_recent_prep_min_60min"] = (
            part["restaurant_recent_prep_min_60min"].fillna(part["est_meal_prep_min"])
        )

    # Out-of-fold smoothed encodings for courier_id (ETA in min and effective trip speed in km/h)
    tr_speed = tr["dist_total_driver_trip_km"] / np.maximum(tr[target] / 60.0, 1.0 / 60.0)
    global_speed_mean = float(tr_speed.mean())

    def _smoothed_series(ids: pd.Series, y: pd.Series, prior: float) -> pd.Series:
        stats = y.groupby(ids).agg(["count", "mean"])
        return (stats["count"] * stats["mean"] + smoothing * prior) / (stats["count"] + smoothing)

    enc_eta_tr = np.full(len(tr), global_eta_mean, dtype=np.float64)
    enc_spd_tr = np.full(len(tr), global_speed_mean, dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_idx, enc_idx in kf.split(tr):
        fit_ids = tr["courier_id"].iloc[fit_idx]
        eta_map = _smoothed_series(fit_ids, tr[target].iloc[fit_idx], global_eta_mean)
        spd_map = _smoothed_series(fit_ids, tr_speed.iloc[fit_idx], global_speed_mean)
        target_ids = tr["courier_id"].iloc[enc_idx]
        enc_eta_tr[enc_idx] = target_ids.map(eta_map).fillna(global_eta_mean).values
        enc_spd_tr[enc_idx] = target_ids.map(spd_map).fillna(global_speed_mean).values

    full_eta_map = _smoothed_series(tr["courier_id"], tr[target], global_eta_mean)
    full_spd_map = _smoothed_series(tr["courier_id"], tr_speed, global_speed_mean)

    tr["courier_target_enc"] = enc_eta_tr
    tr["courier_speed_enc"] = enc_spd_tr
    if val is not None:
        val["courier_target_enc"] = val["courier_id"].map(full_eta_map).fillna(global_eta_mean).values
        val["courier_speed_enc"] = val["courier_id"].map(full_spd_map).fillna(global_speed_mean).values
    te["courier_target_enc"] = te["courier_id"].map(full_eta_map).fillna(global_eta_mean).values
    te["courier_speed_enc"] = te["courier_id"].map(full_spd_map).fillna(global_speed_mean).values
    return tr, val, te


def load_meituan(
    split: str = "70_30",
    distance: str = "haversine",
    seed: int = 42,
    target_min: float = 5.0,
    target_max: float = 180.0,
    smoothing: float = 10.0,
    n_folds: int = 5,
    include_val: bool = True,
    use_cache: bool = True,
    verbose: bool = True,
    data_dir: str | Path | None = None,
    **split_kwargs,
) -> tuple[pd.DataFrame, ...]:
    """
    Load, clean, split, and engineer leak-free Meituan delivery features in English.

    Parameters
    ----------
    split : str, default='70_30'
        Partitioning strategy:
        - ``'70_30'``: random 70% train / 20% val / 10% test (or 70/30 if ``include_val=False``).
        - ``'80_20'``: random 80% train / 10% val / 10% test (or 80/20 if ``include_val=False``).
        - ``'stratified_district'``: district-stratified 70/20/10 (or 70/30).
        - ``'district'``: out-of-distribution spatial split (train 0-5 & 10-22, val 6-7, test 8-9).
    distance : {'haversine', 'manhattan'}, default='haversine'
        Distance metric used for spatial route features.
    seed : int, default=42
        Random seed used in splitting and out-of-fold target encoding.
    target_min, target_max : float, default=(5.0, 180.0)
        Allowed delivery duration range in minutes.
    smoothing : float, default=10.0
        Bayesian smoothing factor ``m`` for ``restaurant_target_enc``.
    n_folds : int, default=5
        Number of folds for out-of-fold ``restaurant_target_enc`` on ``train``.
    include_val : bool, default=True
        If True, returns ``(train_df, val_df, test_df)``; otherwise ``(train_df, test_df)``.
    use_cache : bool, default=True
        Cache pre-split clean features under ``data/cache/`` as parquet.
    verbose : bool, default=True
        Print progress and split diagnostics.
    data_dir : str | Path | None, default=None
        Optional override for the directory containing the raw Meituan files.

    Returns
    -------
    tuple[pd.DataFrame, ...]
        ``(train_df, val_df, test_df)`` when ``include_val=True``, or
        ``(train_df, test_df)`` when ``include_val=False``.
    """
    distance = distance.strip().lower()
    if distance not in DISTANCES:
        raise ValueError(f"distance must be one of {DISTANCES}, got {distance!r}")

    resolved_dir = Path(data_dir) if data_dir else DATA_DIR
    df = _load_presplit_base(target_min, target_max, resolved_dir, use_cache, verbose)
    tr, val, te = _split_dataset(df, split, seed, include_val, verbose, **split_kwargs)
    tr, val, te = _add_distances(tr, val, te, distance)
    tr, val, te = _add_restaurant_target_encoding(
        tr,
        val,
        te,
        target=TARGET,
        smoothing=smoothing,
        n_folds=n_folds,
        seed=seed,
    )
    tr, val, te = _add_extra_postsplit_features(
        tr,
        val,
        te,
        target=TARGET,
        smoothing=smoothing,
        n_folds=n_folds,
        seed=seed,
    )

    tr = tr[FINAL_COLS].copy()
    te = te[FINAL_COLS].copy()
    if include_val:
        assert val is not None
        val = val[FINAL_COLS].copy()
        return tr, val, te
    return tr, te


def preparar(
    distancia: str = "haversine",
    split: str = "70_30",
    semente: int = 42,
    alvo_min: float = 5.0,
    alvo_max: float = 180.0,
    usar_cache: bool = True,
    verbose: bool = True,
    **kwargs_split,
) -> tuple[pd.DataFrame, ...]:
    """Backward-compatible Portuguese wrapper around :func:`load_meituan`."""
    return load_meituan(
        split=split,
        distance=distancia,
        seed=semente,
        target_min=alvo_min,
        target_max=alvo_max,
        use_cache=usar_cache,
        verbose=verbose,
        **kwargs_split,
    )
