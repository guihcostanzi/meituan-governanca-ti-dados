#!/usr/bin/env python3
"""Executable data-quality verification harness for the Meituan INFORMS TSL
on-demand food-delivery dataset.

Implements every check in VERIFICATION_SPEC.md across six dimensions:
completeness (C-01..C-14), validity (V-01..V-12), consistency (S-01..S-11),
uniqueness (U-01..U-10), timeliness (T-01..T-07), accuracy (A-01..A-12).

Deterministic, re-runnable, pandas/numpy only, no network.
Every check is wrapped in try/except; failures are recorded as status ERROR
and never abort the run.
"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "quality_results.json")

WB_PATH = os.path.join(ROOT_DIR, "all_waybill_info_meituan_0322.csv")
WV_PATH = os.path.join(ROOT_DIR, "courier_wave_info_meituan.csv")
DR_PATH = os.path.join(ROOT_DIR, "dispatch_rider_meituan.csv")
DW_PATH = os.path.join(ROOT_DIR, "dispatch_waybill_meituan.csv")

# ---------------------------------------------------------------------------
# Constants / documented semantics
# ---------------------------------------------------------------------------
REFERENCE = {
    "orders": 568546,
    "couriers": 4955,
    "merchants": 4962,
    "districts": 23,
    "prebook_rate": 0.039,
    "rejected_couriers": 54,
    "dispatch_nonconformance_rate": 0.0181,
}

# Observed global bounding box (shifted coords, micro-degrees) from spec.
BBOX = {
    "lng_min": 174387310,
    "lng_max": 174999535,
    "lat_min": 45729124,
    "lat_max": 46079853,
}
# Small margin for V-04 (0.5 degree ~ 500k micro-degrees is generous; use 0.05 deg).
BBOX_MARGIN = 50000  # micro-degrees

EXPECTED_DAYS = [20221017, 20221018, 20221019, 20221020,
                 20221021, 20221022, 20221023, 20221024]
WEEKEND_DAYS = {20221022, 20221023}

WB_TIMESTAMP_COLS = [
    "estimate_arrived_time", "dispatch_time", "grab_time", "fetch_time",
    "arrive_time", "estimate_meal_prepare_time", "order_push_time",
    "platform_order_time",
]
WB_COORD_COLS = ["sender_lng", "sender_lat", "recipient_lng", "recipient_lat",
                 "grab_lng", "grab_lat"]
WB_ID_COLS = ["order_id", "waybill_id", "courier_id", "da_id", "poi_id"]
WB_BOOL_COLS = ["is_courier_grabbed", "is_weekend", "is_prebook"]

# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------
DIMENSIONS = ["completeness", "validity", "consistency",
              "uniqueness", "timeliness", "accuracy"]
DIM_PREFIX = {
    "completeness": "C", "validity": "V", "consistency": "S",
    "uniqueness": "U", "timeliness": "T", "accuracy": "A",
}
RESULTS: dict[str, list[dict]] = {d: [] for d in DIMENSIONS}


def record(dim: str, check_id: str, description: str, metric: str,
           value, expected, status: str, details: dict | None = None) -> None:
    """Append a check result to the given dimension."""
    RESULTS[dim].append({
        "check_id": check_id,
        "description": description,
        "metric": metric,
        "value": value,
        "expected": expected,
        "status": status,
        "details": details if details is not None else {},
    })


def run_check(dim: str, check_id: str, description: str, metric: str,
              expected: str, fn) -> None:
    """Execute a check function; on exception record ERROR and continue.

    fn must return (value, status, details).
    """
    try:
        value, status, details = fn()
        record(dim, check_id, description, metric, value, expected, status, details)
    except Exception as exc:  # noqa: BLE001 - spec requires never aborting
        record(dim, check_id, description, metric, None, expected, "ERROR",
               {"error": str(exc), "traceback": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pct(num: float, den: float) -> float:
    """Safe percentage, 0.0 when denominator is zero."""
    if den == 0:
        return 0.0
    return round(100.0 * num / den, 6)


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def parse_int_list(raw) -> tuple[list[int], int]:
    """Parse a list-like string with ast.literal_eval.

    Returns (valid_nonneg_ints, n_failures). Failures include parse errors and
    elements that are not non-negative ints.
    """
    failures = 0
    try:
        parsed = ast.literal_eval(raw)
    except Exception:  # noqa: BLE001
        return [], 1
    if not isinstance(parsed, (list, tuple)):
        return [], 1
    out: list[int] = []
    for el in parsed:
        if isinstance(el, bool) or not isinstance(el, int) or el < 0:
            failures += 1
        else:
            out.append(el)
    return out, failures


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km. Inputs in degrees (float arrays)."""
    r = 6371.0088
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2)
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def dt_to_date(dt: int) -> datetime:
    return datetime.strptime(str(int(dt)), "%Y%m%d")


def day_bounds(dt: int) -> tuple[int, int]:
    """Return (day_start_epoch, day_end_epoch) for a YYYYMMDD int (local naive)."""
    d = dt_to_date(dt)
    start = int(d.replace(tzinfo=timezone.utc).timestamp())
    return start, start + 86400


def quantiles(series: pd.Series, qs=(0.5, 0.9, 0.99)) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {f"p{int(q*100)}": None for q in qs}
    return {f"p{int(q*100)}": float(np.quantile(s, q)) for q in qs}


def iqr_outliers(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {"n": 0, "lower": None, "upper": None, "n_outliers": 0}
    q1 = float(np.quantile(s, 0.25))
    q3 = float(np.quantile(s, 0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_out = int(((s < lower) | (s > upper)).sum())
    return {"n": int(len(s)), "q1": q1, "q3": q3, "iqr": iqr,
            "lower": lower, "upper": upper, "n_outliers": n_out}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    print("[load] reading CSVs ...", flush=True)
    WB = pd.read_csv(WB_PATH, index_col=0)
    WV = pd.read_csv(WV_PATH, index_col=0)
    DR = pd.read_csv(DR_PATH, index_col=0)
    DW = pd.read_csv(DW_PATH, index_col=0)
    print(f"[load] WB={WB.shape} WV={WV.shape} DR={DR.shape} DW={DW.shape}",
          flush=True)
    return WB, WV, DR, DW


# ---------------------------------------------------------------------------
# COMPLETENESS
# ---------------------------------------------------------------------------
def completeness(WB, WV, DR, DW):
    print("[dim] completeness ...", flush=True)

    def c01():
        tables = {"WB": WB, "WV": WV, "DR": DR, "DW": DW}
        per_table = {}
        total = 0
        for name, df in tables.items():
            cols = {}
            for c in df.columns:
                if df[c].dtype == object:
                    n = int(df[c].isna().sum()) + int((df[c].astype(str).str.strip() == "").sum())
                else:
                    n = int(df[c].isna().sum())
                if n:
                    cols[c] = n
                total += n
            per_table[name] = {"n_nan": int(df.isna().sum().sum()), "columns": cols}
        status = "PASS" if total == 0 else "FAIL"
        return total, status, {"per_table": per_table}

    run_check("completeness", "C-01", "Raw null scan: literal NaN/empty per column per table",
              "total_nan_cells", "0", c01)

    def c02():
        tables = {"WB": WB, "WV": WV, "DR": DR, "DW": DW}
        per_table = {}
        sentinel_cols = []
        valid_id_cols = []
        for name, df in tables.items():
            cols = {}
            for c in df.columns:
                if not pd.api.types.is_numeric_dtype(df[c]):
                    continue
                n0 = int((df[c] == 0).sum())
                if n0:
                    if c.endswith("_time") or c.endswith("_lat") or c.endswith("_lng"):
                        kind = "sentinel"
                        sentinel_cols.append(f"{name}.{c}")
                    elif c.endswith("_id"):
                        kind = "valid_id"
                        valid_id_cols.append(f"{name}.{c}")
                    else:
                        kind = "sentinel"
                        sentinel_cols.append(f"{name}.{c}")
                    cols[c] = {"n_zero": n0, "pct": pct(n0, len(df)),
                               "zero_kind": kind}
            per_table[name] = cols
        return len(sentinel_cols), "INFO", {
            "n_columns_with_zero": len(sentinel_cols) + len(valid_id_cols),
            "n_sentinel_columns": len(sentinel_cols),
            "n_valid_id_columns": len(valid_id_cols),
            "sentinel_columns": sentinel_cols,
            "valid_id_columns": valid_id_cols,
            "per_table": per_table,
        }

    run_check("completeness", "C-02", "Sentinel-zero census: count+pct of 0 per numeric column "
              "(annotated sentinel vs valid_id)",
              "n_columns_with_zero", "flag sentinel columns", c02)

    def c03():
        rej = WB[WB.is_courier_grabbed == 0]
        cols = ["grab_time", "fetch_time", "arrive_time", "grab_lat", "grab_lng"]
        all_zero = (rej[cols] == 0).all(axis=1)
        n = int(all_zero.sum())
        p = pct(n, len(rej))
        status = "PASS" if p == 100.0 else "WARN"
        return p, status, {"n_rejected": int(len(rej)), "n_all_zero": n,
                           "pct_all_zero": p}

    run_check("completeness", "C-03", "Rejected-waybill field completeness (all 5 fields zero)",
              "pct_all_zero", "100%", c03)

    def c04():
        acc = WB[WB.is_courier_grabbed == 1]
        cols = ["grab_time", "fetch_time", "arrive_time", "grab_lat", "grab_lng"]
        any_zero = (acc[cols] == 0).any(axis=1)
        n = int(any_zero.sum())
        status = "PASS" if n == 0 else "FAIL"
        return n, status, {"n_accepted": int(len(acc)), "n_any_zero": n,
                           "pct_any_zero": pct(n, len(acc))}

    run_check("completeness", "C-04", "Accepted-waybill field completeness (any of 5 fields zero)",
              "n_any_zero", "0", c04)

    def c05():
        n_arr0 = int((WB.arrive_time == 0).sum())
        n_rej = int((WB.is_courier_grabbed == 0).sum())
        gap = n_arr0 - n_rej
        status = "PASS" if gap == 0 else "WARN"
        return gap, status, {"arrive_time_zero": n_arr0, "rejected": n_rej,
                             "gap": gap}

    run_check("completeness", "C-05", "arrive_time gap: count(arrive_time=0) - count(rejected)",
              "gap", "0 (observed +1 anomaly)", c05)

    def c06():
        n = int((WB.da_id == 0).sum())
        incl = int(WB.da_id.nunique())
        excl = int(WB[WB.da_id != 0].da_id.nunique())
        return n, "INFO", {
            "n_zero": n, "pct": pct(n, len(WB)),
            "distinct_including_zero": incl, "distinct_excluding_zero": excl,
            "reference": REFERENCE["districts"],
            "note": "da_id=0 is a valid anonymized district id, not missing",
        }

    run_check("completeness", "C-06", "da_id value-0 present (valid entity ID, not missing)",
              "n_zero", "~22.8%", c06)

    def c07():
        n = int((WB.poi_id == 0).sum())
        incl = int(WB.poi_id.nunique())
        excl = int(WB[WB.poi_id != 0].poi_id.nunique())
        return n, "INFO", {
            "n_zero": n, "pct": pct(n, len(WB)),
            "distinct_including_zero": incl, "distinct_excluding_zero": excl,
            "reference": REFERENCE["merchants"],
            "note": "poi_id=0 is a valid anonymized merchant id, not missing",
        }

    run_check("completeness", "C-07", "poi_id value-0 present (valid entity ID, not missing)",
              "n_zero", "~757", c07)

    def c08():
        n = int((WB.estimate_meal_prepare_time == 0).sum())
        return n, "INFO", {"n_zero": n, "pct": pct(n, len(WB))}

    run_check("completeness", "C-08", "estimate_meal_prepare_time missingness (=0)",
              "n_zero", "~9,907", c08)

    def c09():
        n = int((WB.courier_id == 0).sum())
        incl = int(WB.courier_id.nunique())
        excl = int(WB[WB.courier_id != 0].courier_id.nunique())
        return n, "INFO", {
            "n_zero": n, "pct": pct(n, len(WB)),
            "distinct_including_zero": incl, "distinct_excluding_zero": excl,
            "reference": REFERENCE["couriers"],
            "note": "courier_id=0 is a valid anonymized courier id, not missing",
        }

    run_check("completeness", "C-09", "courier_id value-0 present (valid entity ID, not missing)",
              "n_zero", "~33", c09)

    def c10():
        counts = WB.groupby("dt").size()
        days = sorted(int(x) for x in counts.index)
        n_days = len(days)
        consecutive = all((days[i + 1] - days[i]) == 1 for i in range(n_days - 1))
        status = "PASS" if (n_days == 8 and consecutive) else "FAIL"
        return n_days, status, {
            "rows_per_day": {str(k): int(v) for k, v in counts.items()},
            "min_rows": int(counts.min()), "max_rows": int(counts.max()),
            "consecutive": consecutive, "days": days,
        }

    run_check("completeness", "C-10", "Daily coverage: rows/day, distinct days, consecutiveness",
              "n_distinct_days", "8 consecutive days", c10)

    def c11():
        wb_orders = set(WB.order_id.unique().tolist())
        dw_orders = DW.order_id.unique().tolist()
        present = sum(1 for o in dw_orders if o in wb_orders)
        p = pct(present, len(dw_orders))
        status = "PASS" if p == 100.0 else "WARN"
        return p, status, {"n_dw_orders": len(dw_orders), "n_present": present,
                           "pct_present": p}

    run_check("completeness", "C-11", "Referential completeness DW.order_id -> WB.order_id",
              "pct_present", "100%", c11)

    def c12():
        wb_wb = set(WB.waybill_id.unique().tolist())
        total = 0
        present = 0
        parse_fail = 0
        for raw in DR.courier_waybills:
            ids, f = parse_int_list(raw)
            parse_fail += f
            total += len(ids)
            present += sum(1 for i in ids if i in wb_wb)
        p = pct(present, total)
        status = "PASS" if p == 100.0 else "WARN"
        return p, status, {"n_ids": total, "n_present": present,
                           "pct_present": p, "parse_failures": parse_fail}

    run_check("completeness", "C-12", "Referential completeness DR.courier_waybills -> WB.waybill_id",
              "pct_present", "100%", c12)

    def c13():
        wb_orders = set(WB.order_id.unique().tolist())
        total = 0
        present = 0
        parse_fail = 0
        for raw in WV.order_ids:
            ids, f = parse_int_list(raw)
            parse_fail += f
            total += len(ids)
            present += sum(1 for i in ids if i in wb_orders)
        p = pct(present, total)
        status = "PASS" if p == 100.0 else "WARN"
        return p, status, {"n_ids": total, "n_present": present,
                           "pct_present": p, "parse_failures": parse_fail}

    run_check("completeness", "C-13", "Referential completeness WV.order_ids -> WB.order_id",
              "pct_present", "100%", c13)

    def c14():
        wb_couriers = set(WB.courier_id.unique().tolist())
        wv_c = WV.courier_id.unique().tolist()
        dr_c = DR.courier_id.unique().tolist()
        wv_present = sum(1 for c in wv_c if c in wb_couriers)
        dr_present = sum(1 for c in dr_c if c in wb_couriers)
        wv_p = pct(wv_present, len(wv_c))
        dr_p = pct(dr_present, len(dr_c))
        status = "PASS" if (wv_p == 100.0 and dr_p == 100.0) else "WARN"
        return {"wv_pct": wv_p, "dr_pct": dr_p}, status, {
            "wv_couriers": len(wv_c), "wv_present": wv_present,
            "dr_couriers": len(dr_c), "dr_present": dr_present,
        }

    run_check("completeness", "C-14", "Cross-table coverage: WV/DR courier_id in WB.courier_id",
              "pct_present", "100%", c14)


# ---------------------------------------------------------------------------
# VALIDITY
# ---------------------------------------------------------------------------
def validity(WB, WV, DR, DW):
    print("[dim] validity ...", flush=True)

    def v01():
        days = sorted(int(x) for x in WB.dt.unique())
        valid_fmt = all(len(str(d)) == 8 and str(d).isdigit() for d in days)
        expected = days == EXPECTED_DAYS
        status = "PASS" if (valid_fmt and expected) else "FAIL"
        return days, status, {"n_days": len(days), "valid_format": valid_fmt,
                              "matches_expected": expected}

    run_check("validity", "V-01", "dt domain: YYYYMMDD, 8 values, Oct 17-24 2022",
              "distinct_days", "8 values Oct 17-24 2022", v01)

    def v02():
        tables = {"WB": WB, "WV": WV, "DR": DR, "DW": DW}
        bad = {}
        total_bad = 0
        for name, df in tables.items():
            for c in df.columns:
                if c in WB_BOOL_COLS:
                    vals = set(int(x) for x in df[c].unique())
                    extra = vals - {0, 1}
                    if extra:
                        bad[f"{name}.{c}"] = sorted(extra)
                        total_bad += int(df[c].isin(extra).sum())
        status = "PASS" if total_bad == 0 else "FAIL"
        return total_bad, status, {"bad_values": bad}

    run_check("validity", "V-02", "Boolean domains: is_courier_grabbed/is_weekend/is_prebook in {0,1}",
              "n_out_of_domain", "0", v02)

    def v03():
        tables = {"WB": WB, "WV": WV, "DR": DR, "DW": DW}
        id_cols = ["order_id", "waybill_id", "courier_id", "da_id", "poi_id", "wave_id"]
        neg = {}
        total = 0
        for name, df in tables.items():
            for c in df.columns:
                if c in id_cols:
                    n = int((df[c] < 0).sum())
                    if n:
                        neg[f"{name}.{c}"] = n
                        total += n
        status = "PASS" if total == 0 else "FAIL"
        return total, status, {"negatives": neg}

    run_check("validity", "V-03", "ID non-negativity across id columns",
              "n_negative", "0", v03)

    def v04():
        lng_min = BBOX["lng_min"] - BBOX_MARGIN
        lng_max = BBOX["lng_max"] + BBOX_MARGIN
        lat_min = BBOX["lat_min"] - BBOX_MARGIN
        lat_max = BBOX["lat_max"] + BBOX_MARGIN
        out = {}
        total = 0
        for c in WB_COORD_COLS:
            s = WB[c]
            nz = s[s != 0]
            if c.endswith("lng"):
                bad = int(((nz < lng_min) | (nz > lng_max)).sum())
            else:
                bad = int(((nz < lat_min) | (nz > lat_max)).sum())
            if bad:
                out[c] = bad
            total += bad
        status = "PASS" if total == 0 else "WARN"
        return total, status, {"out_of_box": out, "margin": BBOX_MARGIN,
                               "bbox": BBOX}

    run_check("validity", "V-04", "Coordinate nonzero & bounds within global bbox +/- margin",
              "n_out_of_box", "0", v04)

    def v05():
        pairs = [("grab_lat", "grab_lng"), ("sender_lat", "sender_lng"),
                 ("recipient_lat", "recipient_lng")]
        mism = {}
        total = 0
        for lat, lng in pairs:
            m = ((WB[lat] == 0) != (WB[lng] == 0))
            n = int(m.sum())
            if n:
                mism[f"{lat}/{lng}"] = n
            total += n
        status = "PASS" if total == 0 else "FAIL"
        return total, status, {"mismatches": mism}

    run_check("validity", "V-05", "Coordinate pairing: lat==0 iff lng==0",
              "n_mismatch", "0", v05)

    def v06():
        out = {}
        total = 0
        bad_index = set()
        for c in WB_TIMESTAMP_COLS:
            s = WB[c]
            nz = s[s != 0]
            if len(nz) == 0:
                continue
            starts = WB.loc[nz.index, "dt"].map(lambda d: day_bounds(int(d))[0])
            ends = WB.loc[nz.index, "dt"].map(lambda d: day_bounds(int(d))[1])
            lo = starts - 86400
            hi = ends + 86400
            mask = (nz < lo) | (nz > hi)
            bad = int(mask.sum())
            if bad:
                out[c] = bad
                bad_index.update(nz.index[mask].tolist())
            total += bad
        prebook_rate = 0.0
        if bad_index:
            prebook_rate = pct(int(WB.loc[list(bad_index), "is_prebook"].sum()),
                               len(bad_index))
        status = "INFO"
        return total, status, {
            "out_of_range": out,
            "n_out_of_range_rows": len(bad_index),
            "prebook_rate": prebook_rate,
            "note": "advance-booking (prebook) explains out-of-range timestamps",
        }

    run_check("validity", "V-06", "Timestamp range within dt day +/- 1 day (prebook explains)",
              "n_out_of_range", "0", v06)

    def v07():
        chain = ["platform_order_time", "order_push_time", "dispatch_time",
                 "grab_time", "fetch_time", "arrive_time"]
        per_pair = {}
        total_viol = 0
        total_eval = 0
        for a, b in zip(chain[:-1], chain[1:]):
            sa = WB[a]
            sb = WB[b]
            mask = (sa != 0) & (sb != 0)
            n_eval = int(mask.sum())
            viol = int((sa[mask] > sb[mask]).sum())
            per_pair[f"{a}->{b}"] = {"evaluable_pairs": n_eval,
                                     "violations": viol,
                                     "pct": pct(viol, n_eval)}
            total_viol += viol
            total_eval += n_eval
        status = "PASS" if total_viol == 0 else "WARN"
        return total_viol, status, {"per_pair": per_pair,
                                    "total_evaluable_pairs": total_eval,
                                    "total_violations": total_viol}

    run_check("validity", "V-07", "Order timeline monotonicity (non-zero pairs only)",
              "n_violations", "0", v07)

    def v08():
        mask = (WB.fetch_time != 0) & (WB.estimate_meal_prepare_time != 0)
        sub = WB[mask]
        viol = sub[sub.fetch_time < sub.estimate_meal_prepare_time]
        n = int(len(viol))
        diff = (sub.estimate_meal_prepare_time - sub.fetch_time)
        diff_v = diff[diff > 0]
        status = "WARN" if n > 0 else "PASS"
        return n, status, {
            "n_evaluable": int(mask.sum()), "n_violations": n,
            "pct": pct(n, int(mask.sum())),
            "magnitude_seconds": quantiles(diff_v) if len(diff_v) else {},
            "max_seconds": int(diff_v.max()) if len(diff_v) else 0,
        }

    run_check("validity", "V-08", "Pre-fetch vs meal-ready: fetch_time < estimate_meal_prepare_time",
              "n_violations", "allowed by spec (WARN/INFO)", v08)

    def v09():
        viol = int((WV.wave_start_time > WV.wave_end_time).sum())
        status = "PASS" if viol == 0 else "FAIL"
        return viol, status, {"n_waves": int(len(WV)), "n_violations": viol}

    run_check("validity", "V-09", "Wave time order: wave_start_time <= wave_end_time",
              "n_violations", "0", v09)

    def v10():
        dt_vals = DR.dispatch_time
        div30 = int((dt_vals % 30 == 0).sum())
        p = pct(div30, len(dt_vals))
        sec_of_min = (dt_vals % 60)
        dist = {str(int(k)): int(v) for k, v in sec_of_min.value_counts().sort_index().items()}
        status = "INFO"
        return p, status, {"n": int(len(dt_vals)), "n_div30": div30,
                           "pct_div30": p, "second_of_minute_dist": dist}

    run_check("validity", "V-10", "Dispatch grid alignment: dispatch_time divisible by 30",
              "pct_div30", "batch cadence", v10)

    def v11():
        counts = WB.groupby("order_id").size()
        dist = {
            "min": int(counts.min()), "max": int(counts.max()),
            "mean": float(counts.mean()),
            "p50": float(np.quantile(counts, 0.5)),
            "p90": float(np.quantile(counts, 0.9)),
            "p99": float(np.quantile(counts, 0.99)),
        }
        n_multi = int((counts > 1).sum())
        n_orders = int(len(counts))
        status = "PASS" if dist["min"] >= 1 else "FAIL"
        return n_multi, status, {"distribution": dist, "n_orders": n_orders,
                                 "n_multi_waybill": n_multi,
                                 "pct_multi": pct(n_multi, n_orders)}

    run_check("validity", "V-11", "Order->waybill cardinality (>=1 waybill per order)",
              "n_orders_multi_waybill", "each order >=1 waybill", v11)

    def v12():
        nz = WB.estimate_arrived_time != 0
        n_zero = int((~nz).sum())
        sub = WB[nz]
        viol = int((sub.estimate_arrived_time < sub.platform_order_time).sum())
        status = "PASS" if (n_zero == 0 and viol == 0) else "WARN"
        return viol, status, {"n_zero": n_zero, "n_violations": viol,
                              "n_evaluable": int(nz.sum())}

    run_check("validity", "V-12", "estimate_arrived_time sanity: non-zero and >= platform_order_time",
              "n_violations", "0", v12)


# ---------------------------------------------------------------------------
# CONSISTENCY
# ---------------------------------------------------------------------------
def consistency(WB, WV, DR, DW):
    print("[dim] consistency ...", flush=True)

    def s01():
        wd = WB.dt.map(lambda d: dt_to_date(int(d)).weekday())
        derived_weekend = wd.isin([5, 6]).astype(int)
        mismatch_mask = derived_weekend != WB.is_weekend
        mismatch = int(mismatch_mask.sum())
        per_day = WB.groupby("dt").is_weekend.mean().round(6).to_dict()
        mm = WB[mismatch_mask]
        prebook_rate = pct(int(mm.is_prebook.sum()), len(mm)) if len(mm) else 0.0
        n_weekday_flagged = int(((derived_weekend == 0) & (WB.is_weekend == 1)).sum())
        n_weekend_unflagged = int(((derived_weekend == 1) & (WB.is_weekend == 0)).sum())
        status = "PASS" if mismatch == 0 else "WARN"
        return mismatch, status, {
            "n_mismatch": mismatch, "pct": pct(mismatch, len(WB)),
            "per_day_flag_rate": {str(k): float(v) for k, v in per_day.items()},
            "prebook_rate_of_mismatches": prebook_rate,
            "n_weekday_flagged": n_weekday_flagged,
            "n_weekend_unflagged": n_weekend_unflagged,
            "note": "most mismatches are prebook rows whose platform_order_time "
                    "calendar day differs from dt",
        }

    run_check("consistency", "S-01", "is_weekend vs dt calendar (Oct 22-23 weekend)",
              "n_mismatch", "0", s01)

    def s02():
        a = int(((WB.is_courier_grabbed == 1) & (WB.grab_time == 0)).sum())
        b = int(((WB.is_courier_grabbed == 0) & (WB.grab_time != 0)).sum())
        status = "PASS" if (a + b) == 0 else "FAIL"
        return a + b, status, {"grabbed1_grab0": a, "grabbed0_grab_nonzero": b}

    run_check("consistency", "S-02", "is_courier_grabbed vs grab_time agreement",
              "n_mismatch", "0", s02)

    def s03():
        acc = WB[(WB.is_courier_grabbed == 1) & (WB.grab_time != 0)]
        order_min_grab = acc.groupby("order_id").grab_time.min().to_dict()
        exact = 0
        mismatch = 0
        n_eval = 0
        parse_fail = 0
        deltas = []
        for raw, wst in zip(WV.order_ids, WV.wave_start_time):
            ids, f = parse_int_list(raw)
            parse_fail += f
            if not ids:
                continue
            mins = [order_min_grab[i] for i in ids if i in order_min_grab]
            if not mins:
                continue
            n_eval += 1
            recomputed = min(mins)
            if recomputed == int(wst):
                exact += 1
            else:
                mismatch += 1
                deltas.append(int(wst) - recomputed)
        p = pct(exact, n_eval)
        status = "WARN" if mismatch > 0 else "PASS"
        delta_stats = {}
        if deltas:
            arr = np.array(deltas, dtype=np.int64)
            delta_stats = {
                "p50": float(np.quantile(arr, 0.5)),
                "p90": float(np.quantile(arr, 0.9)),
                "p99": float(np.quantile(arr, 0.99)),
                "max": int(arr.max()), "min": int(arr.min()),
                "n_negative": int((arr < 0).sum()),
            }
        return p, status, {"n_evaluable_waves": n_eval, "n_exact": exact,
                           "n_mismatch": mismatch, "pct_exact": p,
                           "parse_failures": parse_fail,
                           "mismatch_delta_seconds": delta_stats,
                           "note": "wave_start_time - recomputed_min_grab_time; "
                                   "direction is always >= 0"}

    run_check("consistency", "S-03", "Wave start correctness vs recomputed min(grab_time)",
              "pct_exact", "100% (bug documented)", s03)

    def s04():
        acc = WB[(WB.is_courier_grabbed == 1) & (WB.arrive_time != 0)]
        order_max_arr = acc.groupby("order_id").arrive_time.max().to_dict()
        exact = 0
        mismatch = 0
        n_eval = 0
        for raw, wet in zip(WV.order_ids, WV.wave_end_time):
            ids, _ = parse_int_list(raw)
            if not ids:
                continue
            maxs = [order_max_arr[i] for i in ids if i in order_max_arr]
            if not maxs:
                continue
            n_eval += 1
            if max(maxs) == int(wet):
                exact += 1
            else:
                mismatch += 1
        p = pct(exact, n_eval)
        status = "WARN" if mismatch > 0 else "PASS"
        return p, status, {"n_evaluable_waves": n_eval, "n_exact": exact,
                           "n_mismatch": mismatch, "pct_exact": p}

    run_check("consistency", "S-04", "Wave end correctness vs recomputed max(arrive_time)",
              "pct_exact", "100%", s04)

    def s05():
        wb_orders = set(WB.order_id.unique().tolist())
        n_empty = 0
        n_missing = 0
        total_len = 0
        parse_fail = 0
        for raw in WV.order_ids:
            ids, f = parse_int_list(raw)
            parse_fail += f
            if len(ids) == 0:
                n_empty += 1
            total_len += len(ids)
            n_missing += sum(1 for i in ids if i not in wb_orders)
        n_accepted = int((WB.is_courier_grabbed == 1).sum())
        status = "PASS" if (n_empty == 0 and n_missing == 0) else "WARN"
        return n_missing, status, {
            "n_empty_lists": n_empty, "n_missing_ids": n_missing,
            "total_list_len": total_len, "accepted_waybills": n_accepted,
            "parse_failures": parse_fail,
        }

    run_check("consistency", "S-05", "Wave order set validity: ids exist, non-empty, len>=1",
              "n_missing_ids", "0", s05)

    def s06():
        order_dt = WB.groupby("order_id")["dt"].first().to_dict()
        n_eval = 0
        mismatch = 0
        for raw, wdt in zip(WV.order_ids, WV.index):
            ids, _ = parse_int_list(raw)
            if not ids:
                continue
            n_eval += 1
            if any(order_dt.get(i) != int(wdt) for i in ids):
                mismatch += 1
        p = pct(mismatch, n_eval)
        status = "PASS" if mismatch == 0 else "WARN"
        return p, status, {"n_evaluable_waves": n_eval, "n_mismatch": mismatch,
                           "pct_mismatch": p}

    run_check("consistency", "S-06", "Wave date consistency: order dt == WV.dt",
              "pct_mismatch", "0%", s06)

    def s07():
        wb_by_waybill = WB.set_index("waybill_id").courier_id.to_dict()
        wb_by_order = WB.groupby("order_id").courier_id.apply(set).to_dict()
        wb_order_dispatch = WB.groupby("order_id").dispatch_time.min().to_dict()

        n_eval_wb = 0
        match_wb = 0
        n_eval_order = 0
        match_order = 0
        n_eval_correct = 0
        match_correct = 0
        parse_fail = 0
        for raw, cid, dtime in zip(DR.courier_waybills, DR.courier_id,
                                   DR.dispatch_time):
            ids, f = parse_int_list(raw)
            parse_fail += f
            cid = int(cid)
            dtime = int(dtime)
            for i in ids:
                if i in wb_by_waybill:
                    n_eval_wb += 1
                    if wb_by_waybill[i] == cid:
                        match_wb += 1
                if i in wb_by_order:
                    n_eval_order += 1
                    if cid in wb_by_order[i]:
                        match_order += 1
                    n_eval_correct += 1
                    if (cid in wb_by_order[i]
                            and wb_order_dispatch.get(i, dtime + 1) <= dtime):
                        match_correct += 1
        p_wb = pct(match_wb, n_eval_wb)
        p_order = pct(match_order, n_eval_order)
        p_correct = pct(match_correct, n_eval_correct)
        status = "INFO"
        return p_correct, status, {
            "order_id_join_pct": p_order,
            "waybill_id_join_pct": p_wb,
            "correct_match_pct": p_correct,
            "n_eval_order_join": n_eval_order,
            "n_eval_waybill_join": n_eval_wb,
            "n_eval_correct": n_eval_correct,
            "parse_failures": parse_fail,
            "note": "courier_waybills holds ORDER-vocabulary ids, not waybill ids; "
                    "correct match requires accepted waybill for that order_id with "
                    "courier_id == DR.courier_id and dispatch_time <= DR.dispatch_time",
        }

    run_check("consistency", "S-07", "Courier-waybill ownership (order-vocabulary ids; "
              "field-naming ambiguity)",
              "correct_match_pct", "candidate-vs-assigned nuance", s07)

    def s08():
        wb_wb = set(WB.waybill_id.unique().tolist())
        n_empty = 0
        total = 0
        present = 0
        for raw in DR.courier_waybills:
            ids, _ = parse_int_list(raw)
            if len(ids) == 0:
                n_empty += 1
            total += len(ids)
            present += sum(1 for i in ids if i in wb_wb)
        p = pct(present, total)
        status = "INFO"
        return p, status, {"n_empty_lists": n_empty,
                           "pct_empty": pct(n_empty, len(DR)),
                           "n_ids": total, "n_present": present,
                           "pct_present": p,
                           "note": "empty courier_waybills list = idle candidate"}

    run_check("consistency", "S-08", "DR on-hand set vs WB: ids exist; empty = idle candidate",
              "pct_present", "100%", s08)

    def s09():
        wb_orders = set(WB.order_id.unique().tolist())
        bad = int((~DW.order_id.isin(wb_orders)).sum())
        non_int = int((~DW.order_id.map(lambda x: isinstance(x, (int, np.integer)))).sum())
        status = "PASS" if (bad == 0 and non_int == 0) else "WARN"
        return bad, status, {"n_not_in_wb": bad, "n_non_integer": non_int,
                             "n_rows": int(len(DW))}

    run_check("consistency", "S-09", "DW order existence & integer type",
              "n_bad_rows", "0", s09)

    def s10():
        obs_orders = int(WB.order_id.nunique())
        obs_couriers = int(WB[WB.courier_id != 0].courier_id.nunique())
        obs_merchants = int(WB[WB.poi_id != 0].poi_id.nunique())
        obs_districts = int(WB[WB.da_id != 0].da_id.nunique())
        obs_prebook = float(WB.is_prebook.mean())
        deltas = {
            "orders": {"observed": obs_orders, "reference": REFERENCE["orders"],
                       "delta": obs_orders - REFERENCE["orders"]},
            "couriers": {"observed": obs_couriers, "reference": REFERENCE["couriers"],
                         "delta": obs_couriers - REFERENCE["couriers"]},
            "merchants": {"observed": obs_merchants, "reference": REFERENCE["merchants"],
                          "delta": obs_merchants - REFERENCE["merchants"]},
            "districts": {"observed": obs_districts, "reference": REFERENCE["districts"],
                          "delta": obs_districts - REFERENCE["districts"]},
            "prebook_rate": {"observed": round(obs_prebook, 6),
                             "reference": REFERENCE["prebook_rate"],
                             "delta": round(obs_prebook - REFERENCE["prebook_rate"], 6)},
        }
        status = "INFO"
        return deltas, status, {"deltas": deltas}

    run_check("consistency", "S-10", "Universe reconciliation vs published reference",
              "deltas", "match reference", s10)

    def s11():
        weekend_rows = int(WB.is_weekend.sum())
        p = pct(weekend_rows, len(WB))
        expected = pct(2, 8)
        status = "INFO"
        return p, status, {"weekend_rows": weekend_rows, "pct": p,
                           "expected_pct": expected}

    run_check("consistency", "S-11", "Weekend share vs expectation (2/8 days)",
              "pct_weekend", "25%", s11)


# ---------------------------------------------------------------------------
# UNIQUENESS
# ---------------------------------------------------------------------------
def uniqueness(WB, WV, DR, DW):
    print("[dim] uniqueness ...", flush=True)

    def u01():
        n = int(WB.waybill_id.duplicated().sum())
        status = "PASS" if n == 0 else "FAIL"
        return n, status, {"n_duplicates": n, "n_unique": int(WB.waybill_id.nunique())}

    run_check("uniqueness", "U-01", "waybill_id uniqueness in WB",
              "n_duplicates", "0", u01)

    def u02():
        n = int(WB.duplicated(subset=["dt", "waybill_id"]).sum())
        status = "PASS" if n == 0 else "FAIL"
        return n, status, {"n_duplicates": n}

    run_check("uniqueness", "U-02", "(dt, waybill_id) uniqueness",
              "n_duplicates", "0", u02)

    def u03():
        tables = {"WB": WB, "WV": WV, "DR": DR, "DW": DW}
        out = {}
        total = 0
        for name, df in tables.items():
            n = int(df.duplicated().sum())
            out[name] = n
            total += n
        status = "PASS" if total == 0 else "FAIL"
        return total, status, {"per_table": out}

    run_check("uniqueness", "U-03", "Full-row duplicates per table",
              "n_duplicates", "0", u03)

    def u04():
        n = int(WV.reset_index().duplicated(subset=["dt", "courier_id", "wave_id"]).sum())
        status = "PASS" if n == 0 else "FAIL"
        return n, status, {"n_duplicates": n}

    run_check("uniqueness", "U-04", "(dt, courier_id, wave_id) uniqueness in WV",
              "n_duplicates", "0", u04)
    def u05():
        n = int(DR.reset_index().duplicated(subset=["dt", "dispatch_time", "courier_id"]).sum())
        status = "PASS" if n == 0 else "FAIL"
        return n, status, {"n_duplicates": n}

    run_check("uniqueness", "U-05", "(dt, dispatch_time, courier_id) uniqueness in DR",
              "n_duplicates", "0", u05)

    def u06():
        n = int(DW.reset_index().duplicated(subset=["dt", "dispatch_time", "order_id"]).sum())
        status = "PASS" if n == 0 else "FAIL"
        return n, status, {"n_duplicates": n}

    run_check("uniqueness", "U-06", "(dt, dispatch_time, order_id) uniqueness in DW",
              "n_duplicates", "0", u06)

    def u07():
        counts = WB.groupby("order_id").size()
        n_orders = int(len(counts))
        n_rows = int(len(WB))
        n_multi = int((counts > 1).sum())
        status = "INFO"
        return n_orders, status, {
            "n_orders": n_orders, "n_rows": n_rows,
            "n_multi_waybill": n_multi, "pct_multi": pct(n_multi, n_orders),
            "distribution": {"min": int(counts.min()), "max": int(counts.max()),
                             "mean": float(counts.mean())},
        }

    run_check("uniqueness", "U-07", "order_id cardinality: distinct orders vs rows",
              "n_distinct_orders", "1 order : 1..N waybills", u07)

    def u08():
        counts = DW.groupby("order_id").size()
        n_multi = int((counts > 1).sum())
        n_orders = int(len(counts))
        status = "INFO"
        return n_multi, status, {"n_orders": n_orders, "n_multi_checkpoint": n_multi,
                                 "pct_multi": pct(n_multi, n_orders),
                                 "max_checkpoints": int(counts.max())}

    run_check("uniqueness", "U-08", "Candidate duplication in DW (orders at >1 checkpoint)",
              "n_multi_checkpoint", "expected for delayed assignment", u08)

    def u09():
        sub = WB[WB.poi_id != 0]
        g = sub.groupby("poi_id")[["sender_lng", "sender_lat"]].nunique()
        n_diff = int(((g.sender_lng > 1) | (g.sender_lat > 1)).sum())
        status = "INFO"
        return n_diff, status, {"n_poi": int(sub.poi_id.nunique()),
                                "n_poi_with_differing_coords": n_diff}

    run_check("uniqueness", "U-09", "Duplicate poi_id coords: same poi_id differing coords",
              "n_poi_with_differing_coords", "0", u09)

    def u10():
        wb_pairs = set(zip(WB["dt"].tolist(), WB.courier_id.tolist()))
        wv_pairs = set(zip(WV.index.tolist(), WV.courier_id.tolist()))
        both = wb_pairs & wv_pairs
        status = "INFO"
        return len(both), status, {"wb_day_courier_pairs": len(wb_pairs),
                                   "wv_day_courier_pairs": len(wv_pairs),
                                   "both": len(both)}

    run_check("uniqueness", "U-10", "Courier on same day in both WB and WV (join sanity)",
              "n_common_day_courier_pairs", "descriptive", u10)


# ---------------------------------------------------------------------------
# TIMELINESS
# ---------------------------------------------------------------------------
def timeliness(WB, WV, DR, DW):
    print("[dim] timeliness ...", flush=True)

    def t01():
        days = sorted(int(x) for x in WB.dt.unique())
        span = (dt_to_date(days[-1]) - dt_to_date(days[0])).days + 1
        status = "INFO"
        return span, status, {"min_dt": days[0], "max_dt": days[-1],
                              "span_days": span, "n_distinct_days": len(days)}

    run_check("timeliness", "T-01", "Coverage window: min/max dt, span, distinct days",
              "span_days", "8 days", t01)

    def t02():
        days = sorted(int(x) for x in WB.dt.unique())
        missing = []
        cur = dt_to_date(days[0])
        end = dt_to_date(days[-1])
        present = set(days)
        while cur <= end:
            key = int(cur.strftime("%Y%m%d"))
            if key not in present:
                missing.append(key)
            cur = cur.fromordinal(cur.toordinal() + 1)
        status = "PASS" if not missing else "FAIL"
        return len(missing), status, {"missing_days": missing}

    run_check("timeliness", "T-02", "Gap check: no missing calendar day in window",
              "n_missing_days", "0", t02)

    def t03():
        max_ts = int(WB[WB_TIMESTAMP_COLS].max().max())
        latest = datetime.fromtimestamp(max_ts, tz=timezone.utc)
        now = datetime.now(timezone.utc)
        lag_days = (now - latest).days
        status = "WARN"
        return lag_days, status, {"latest_timestamp": max_ts,
                                  "latest_utc": latest.isoformat(),
                                  "generated_at": now.isoformat(),
                                  "lag_days": lag_days,
                                  "lag_years": round(lag_days / 365.25, 2)}

    run_check("timeliness", "T-03", "Snapshot age: latest timestamp vs generated_at",
              "lag_days", "flag staleness", t03)

    def t04():
        per_day = {}
        for d, g in WB.groupby("dt"):
            ts = g[WB_TIMESTAMP_COLS].to_numpy()
            nz = ts[ts != 0]
            if nz.size == 0:
                per_day[str(int(d))] = {"span": 0, "rows": int(len(g))}
                continue
            per_day[str(int(d))] = {"span": int(nz.max() - nz.min()),
                                    "rows": int(len(g))}
        spans = [v["span"] for v in per_day.values()]
        median_span = float(np.median(spans))
        last_day = str(max(int(x) for x in WB.dt.unique()))
        last_span = per_day[last_day]["span"]
        truncated = last_span < 0.5 * median_span
        status = "WARN" if truncated else "PASS"
        return last_span, status, {"per_day": per_day, "median_span": median_span,
                                   "last_day": last_day, "last_span": last_span,
                                   "truncated": truncated}

    run_check("timeliness", "T-04", "Final-day truncation: per-day event span vs median",
              "last_day_span_seconds", "not truncated", t04)

    def t05():
        out = {}
        total_cross = 0
        for c in WB_TIMESTAMP_COLS:
            s = WB[c]
            nz = s[s != 0]
            if len(nz) == 0:
                continue
            starts = WB.loc[nz.index, "dt"].map(lambda d: day_bounds(int(d))[0])
            ends = WB.loc[nz.index, "dt"].map(lambda d: day_bounds(int(d))[1])
            cross = int(((nz < starts) | (nz >= ends)).sum())
            out[c] = {"min": int(nz.min()), "max": int(nz.max()),
                      "n_cross_midnight": cross}
            total_cross += cross
        status = "INFO"
        return total_cross, status, {"per_column": out,
                                     "total_cross_midnight": total_cross}

    run_check("timeliness", "T-05", "Event-time dispersion: min/max per timestamp, cross-midnight",
              "n_cross_midnight", "descriptive", t05)

    def t06():
        pairs = [
            ("order_push_time", "platform_order_time"),
            ("dispatch_time", "order_push_time"),
            ("grab_time", "dispatch_time"),
            ("arrive_time", "grab_time"),
        ]
        out = {}
        for a, b in pairs:
            mask = (WB[a] != 0) & (WB[b] != 0)
            diff = (WB.loc[mask, a] - WB.loc[mask, b])
            neg = int((diff < 0).sum())
            out[f"{a}-{b}"] = {
                "n_evaluable": int(mask.sum()),
                "median": float(np.median(diff)) if len(diff) else None,
                "p90": float(np.quantile(diff, 0.9)) if len(diff) else None,
                "p99": float(np.quantile(diff, 0.99)) if len(diff) else None,
                "n_negative": neg,
            }
        status = "INFO"
        return out, status, {"latency_profile": out}

    run_check("timeliness", "T-06", "Latency profile: median/p90/p99 of adjacent diffs",
              "latency_profile", "descriptive", t06)

    def t07():
        non_int = {}
        total = 0
        for c in WB_TIMESTAMP_COLS:
            s = WB[c]
            bad = int((s != s.astype(np.int64)).sum())
            if bad:
                non_int[c] = bad
                total += bad
        status = "PASS" if total == 0 else "FAIL"
        return total, status, {"non_integer": non_int}

    run_check("timeliness", "T-07", "Timestamp resolution: all integer seconds",
              "n_non_integer", "0", t07)


# ---------------------------------------------------------------------------
# ACCURACY
# ---------------------------------------------------------------------------
def accuracy(WB, WV, DR, DW):
    print("[dim] accuracy ...", flush=True)

    def a01():
        # (a) rejected-waybill zeroing
        rej = WB[WB.is_courier_grabbed == 0]
        cols = ["grab_time", "fetch_time", "arrive_time", "grab_lat", "grab_lng"]
        a_pct = pct(int((rej[cols] == 0).all(axis=1).sum()), len(rej))
        # (b) wave_start bug
        acc = WB[(WB.is_courier_grabbed == 1) & (WB.grab_time != 0)]
        order_min = acc.groupby("order_id").grab_time.min().to_dict()
        n_eval = 0
        n_mismatch = 0
        for raw, wst in zip(WV.order_ids, WV.wave_start_time):
            ids, _ = parse_int_list(raw)
            mins = [order_min[i] for i in ids if i in order_min]
            if not mins:
                continue
            n_eval += 1
            if min(mins) != int(wst):
                n_mismatch += 1
        b_pct = pct(n_mismatch, n_eval)
        # (c) dispatch non-conformance: does WB contain a waybill for the DW
        # order whose dispatch_time equals the DW checkpoint?
        wb_dispatch_by_order = WB.groupby("order_id").dispatch_time.apply(list).to_dict()
        c_matched = 0
        c_within_300 = 0
        c_missing = 0
        for oid, dtime in zip(DW.order_id, DW.dispatch_time):
            lst = [x for x in wb_dispatch_by_order.get(int(oid), []) if x != 0]
            if not lst:
                c_missing += 1
                continue
            if int(dtime) in lst:
                c_matched += 1
            if min(abs(x - int(dtime)) for x in lst) <= 300:
                c_within_300 += 1
        c_not_matched = int(len(DW)) - c_matched
        c_pct_matched = pct(c_matched, len(DW))
        c_pct_not_matched = pct(c_not_matched, len(DW))
        c_pct_within_300 = pct(c_within_300, len(DW))
        # (d) fetch_time < estimate_meal_prepare_time
        mask = (WB.fetch_time != 0) & (WB.estimate_meal_prepare_time != 0)
        d_n = int((WB.loc[mask, "fetch_time"] < WB.loc[mask, "estimate_meal_prepare_time"]).sum())
        d_pct = pct(d_n, int(mask.sum()))
        # (e) coordinate shift: ranges displaced from real Beijing ~39.9/116.4
        lng_min = int(WB.sender_lng[WB.sender_lng != 0].min())
        lat_min = int(WB.sender_lat[WB.sender_lat != 0].min())
        e_shifted = (lng_min > 1e8) and (lat_min > 1e7)
        status = "INFO"
        return {
            "a_rejected_zeroing_pct": a_pct,
            "b_wave_start_mismatch_pct": b_pct,
            "c_dispatch_not_matched_pct": c_pct_not_matched,
            "d_prefetch_pct": d_pct,
            "e_coords_shifted": e_shifted,
        }, status, {
            "a": {"pct": a_pct, "expected": 100.0},
            "b": {"pct": b_pct, "n_eval": n_eval, "n_mismatch": n_mismatch},
            "c": {"pct_matched": c_pct_matched,
                  "pct_not_matched": c_pct_not_matched,
                  "pct_within_300s": c_pct_within_300,
                  "n_matched": c_matched, "n_not_matched": c_not_matched,
                  "n_within_300s": c_within_300, "n_missing": c_missing,
                  "n_dw": int(len(DW)),
                  "documented": REFERENCE["dispatch_nonconformance_rate"],
                  "measured": "DW (dt, dispatch_time, order_id) matched against WB "
                              "waybills for that order with dispatch_time == checkpoint; "
                              "checkpoint is a batch time so exact equality is rare, "
                              "within-300s is the meaningful conformance proxy"},
            "d": {"pct": d_pct, "n": d_n},
            "e": {"lng_min": lng_min, "lat_min": lat_min,
                  "real_beijing": {"lat": 39.9, "lng": 116.4}},
        }

    run_check("accuracy", "A-01", "Documented anomaly reproduction (a-e)",
              "anomaly_rates", "reproduce documented caveats", a01)

    def a02():
        pairs = [
            ("order_push_time", "platform_order_time"),
            ("dispatch_time", "order_push_time"),
            ("grab_time", "dispatch_time"),
            ("arrive_time", "grab_time"),
        ]
        out = {}
        total = 0
        for a, b in pairs:
            mask = (WB[a] != 0) & (WB[b] != 0)
            neg = int((WB.loc[mask, a] < WB.loc[mask, b]).sum())
            out[f"{a}-{b}"] = neg
            total += neg
        status = "PASS" if total == 0 else "WARN"
        return total, status, {"per_pair": out}

    run_check("accuracy", "A-02", "Negative durations across adjacent timestamp diffs",
              "n_negative", "0", a02)

    def a03():
        out = {}
        # arrive - grab
        m = (WB.arrive_time != 0) & (WB.grab_time != 0)
        out["arrive_time-grab_time"] = iqr_outliers(WB.loc[m, "arrive_time"] - WB.loc[m, "grab_time"])
        # fetch - grab
        m = (WB.fetch_time != 0) & (WB.grab_time != 0)
        out["fetch_time-grab_time"] = iqr_outliers(WB.loc[m, "fetch_time"] - WB.loc[m, "grab_time"])
        # dispatch - platform
        m = (WB.dispatch_time != 0) & (WB.platform_order_time != 0)
        out["dispatch_time-platform_order_time"] = iqr_outliers(
            WB.loc[m, "dispatch_time"] - WB.loc[m, "platform_order_time"])
        # wave duration
        out["wave_duration"] = iqr_outliers(WV.wave_end_time - WV.wave_start_time)
        # extreme examples
        m = (WB.arrive_time != 0) & (WB.grab_time != 0)
        dur = WB.loc[m, "arrive_time"] - WB.loc[m, "grab_time"]
        extreme = int((dur > np.quantile(dur, 0.999)).sum())
        status = "INFO"
        return out, status, {"outliers": out, "n_extreme_arrive_grab": extreme,
                             "p999_arrive_grab": float(np.quantile(dur, 0.999))}

    run_check("accuracy", "A-03", "Duration outliers: IQR bounds for key durations",
              "outlier_stats", "descriptive", a03)

    def a04():
        m = (WB.arrive_time != 0) & (WB.estimate_arrived_time != 0)
        sub = WB[m]
        on_time = int((sub.arrive_time <= sub.estimate_arrived_time).sum())
        p = pct(on_time, len(sub))
        diff = sub.arrive_time - sub.estimate_arrived_time
        status = "INFO"
        return p, status, {"n_evaluable": int(len(sub)), "n_on_time": on_time,
                           "pct_on_time": p,
                           "diff_distribution": quantiles(diff),
                           "n_late": int((diff > 0).sum())}

    run_check("accuracy", "A-04", "On-time delivery rate: arrive_time <= estimate_arrived_time",
              "pct_on_time", "majority on-time", a04)

    def a05():
        m = ((WB.fetch_time != 0) & (WB.arrive_time != 0)
             & (WB.sender_lat != 0) & (WB.recipient_lat != 0))
        sub = WB[m]
        if len(sub) == 0:
            return 0, "INFO", {"n_evaluable": 0}
        dist_km = haversine_km(
            sub.sender_lat.to_numpy() / 1e6, sub.sender_lng.to_numpy() / 1e6,
            sub.recipient_lat.to_numpy() / 1e6, sub.recipient_lng.to_numpy() / 1e6)
        dur_h = (sub.arrive_time.to_numpy() - sub.fetch_time.to_numpy()) / 3600.0
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = np.where(dur_h > 0, dist_km / dur_h, np.nan)
        impossible = int(np.nansum(speed > 120.0))
        zero_dist_time = int(np.sum((dist_km < 0.001) & (dur_h > 0)))
        status = "WARN" if (impossible > 0 or zero_dist_time > 0) else "PASS"
        return impossible, status, {
            "n_evaluable": int(len(sub)),
            "n_impossible_speed_gt120": impossible,
            "pct_impossible": pct(impossible, len(sub)),
            "n_zero_distance_with_time": zero_dist_time,
            "distance_km": quantiles(pd.Series(dist_km)),
            "speed_kmh": quantiles(pd.Series(speed[~np.isnan(speed)])),
        }

    run_check("accuracy", "A-05", "Geodesic/implied-speed check (haversine, >120 km/h flag)",
              "n_impossible_speed", "0", a05)

    def a06():
        m = ((WB.grab_lat != 0) & (WB.grab_lng != 0)
             & (WB.sender_lat != 0) & (WB.sender_lng != 0))
        sub = WB[m]
        if len(sub) == 0:
            return 0, "INFO", {"n_evaluable": 0}
        d = haversine_km(
            sub.grab_lat.to_numpy() / 1e6, sub.grab_lng.to_numpy() / 1e6,
            sub.sender_lat.to_numpy() / 1e6, sub.sender_lng.to_numpy() / 1e6)
        implausible = int((d > 5.0).sum())
        status = "WARN" if implausible > 0 else "PASS"
        return implausible, status, {
            "n_evaluable": int(len(sub)),
            "n_gt_5km": implausible, "pct_gt_5km": pct(implausible, len(sub)),
            "distance_km": quantiles(pd.Series(d)),
            "max_km": float(d.max()),
        }

    run_check("accuracy", "A-06", "Courier-at-pickup proximity (grab vs sender, >5km flag)",
              "n_gt_5km", "0", a06)

    def a07():
        m = (WB.estimate_meal_prepare_time != 0) & (WB.platform_order_time != 0)
        sub = WB[m]
        horizon = sub.estimate_meal_prepare_time - sub.platform_order_time
        impossible = int((horizon < 0).sum())
        status = "WARN" if impossible > 0 else "PASS"
        return impossible, status, {
            "n_evaluable": int(len(sub)), "n_impossible_before_order": impossible,
            "horizon_seconds": quantiles(horizon),
            "median_minutes": float(np.median(horizon)) / 60.0,
            "p90_minutes": float(np.quantile(horizon, 0.9)) / 60.0,
        }

    run_check("accuracy", "A-07", "Meal-prep plausibility vs platform_order_time",
              "n_impossible", "0", a07)

    def a08():
        coord_ok = True
        coord_bad = {}
        for c in WB_COORD_COLS:
            nz = WB[c][WB[c] != 0]
            if len(nz) and int(nz.abs().max()) < 1_000_000:
                coord_ok = False
                coord_bad[c] = int(nz.abs().max())
        ts_ok = True
        ts_bad = {}
        for c in WB_TIMESTAMP_COLS:
            nz = WB[c][WB[c] != 0]
            if len(nz):
                mx = int(nz.max())
                # seconds ~1.6e9; ms ~1.6e12
                if mx > 1e11:
                    ts_ok = False
                    ts_bad[c] = mx
        status = "PASS" if (coord_ok and ts_ok) else "FAIL"
        return {"coords_microdeg": coord_ok, "times_seconds": ts_ok}, status, {
            "coord_bad": coord_bad, "ts_bad": ts_bad}

    run_check("accuracy", "A-08", "Precision/scale: coords micro-degrees, times seconds",
              "scale_ok", "coords>1e6, times<1e11", a08)

    def a09():
        pre = WB[WB.is_prebook == 1]
        m = (pre.platform_order_time != 0) & (pre.order_push_time != 0)
        sub = pre[m]
        lead = sub.order_push_time - sub.platform_order_time
        far = int((lead > 3600).sum())
        status = "INFO"
        return far, status, {"n_prebook": int(len(pre)),
                             "n_lead_gt_1h": far,
                             "pct_lead_gt_1h": pct(far, len(sub)),
                             "lead_seconds": quantiles(lead)}

    run_check("accuracy", "A-09", "Cross-field business rule: prebook lead time",
              "n_prebook_lead_gt_1h", "descriptive", a09)

    def a10():
        wb_c = set(WB[WB.courier_id != 0].courier_id.unique().tolist())
        wv_c = set(WV.courier_id.unique().tolist())
        dr_c = set(DR.courier_id.unique().tolist())
        status = "INFO"
        return {"wb": len(wb_c), "wv": len(wv_c), "dr": len(dr_c)}, status, {
            "wb_couriers": len(wb_c), "wv_couriers": len(wv_c),
            "dr_couriers": len(dr_c),
            "wv_not_wb": len(wv_c - wb_c), "dr_not_wb": len(dr_c - wb_c),
            "reference_couriers": REFERENCE["couriers"],
            "reference_rejected_couriers": REFERENCE["rejected_couriers"],
        }

    run_check("accuracy", "A-10", "Internal reference consistency: courier counts WB/WV/DR",
              "courier_counts", "reconcile with reference", a10)

    def a11():
        daily = WB.groupby("dt").size()
        cov = float(daily.std() / daily.mean()) if daily.mean() else 0.0
        wave_dur = WV.wave_end_time - WV.wave_start_time
        status = "INFO"
        return cov, status, {
            "daily_volume": {str(k): int(v) for k, v in daily.items()},
            "min": int(daily.min()), "max": int(daily.max()),
            "cov": round(cov, 6),
            "wave_duration_seconds": quantiles(wave_dur),
        }

    run_check("accuracy", "A-11", "Distributional plausibility: daily volume stability & wave duration",
              "daily_volume_cov", "stable", a11)

    def a12():
        acc = WB[WB.is_courier_grabbed == 1]
        n = int(((acc.grab_lat == 0) & (acc.grab_lng == 0)).sum())
        status = "FAIL" if n > 0 else "PASS"
        return n, status, {"n_accepted_with_zero_grab_coord": n,
                           "n_accepted": int(len(acc))}

    run_check("accuracy", "A-12", "Impossible coordinate == 0 for accepted waybills",
              "n_zero_coord_accepted", "0", a12)


# ---------------------------------------------------------------------------
# Summary / output
# ---------------------------------------------------------------------------
def build_summary() -> dict:
    summary = {}
    for i, dim in enumerate(DIMENSIONS, start=1):
        counts = {"n_pass": 0, "n_warn": 0, "n_fail": 0, "n_info": 0, "n_error": 0}
        for c in RESULTS[dim]:
            st = c["status"]
            if st == "PASS":
                counts["n_pass"] += 1
            elif st == "WARN":
                counts["n_warn"] += 1
            elif st == "FAIL":
                counts["n_fail"] += 1
            elif st == "INFO":
                counts["n_info"] += 1
            else:
                counts["n_error"] += 1
        summary[f"{i}_{dim}"] = counts
    return summary


def print_console_summary(summary: dict) -> None:
    print("\n" + "=" * 72)
    print("QUALITY VERIFICATION SUMMARY")
    print("=" * 72)
    for key, counts in summary.items():
        print(f"  {key:20s} PASS={counts['n_pass']:2d} "
              f"WARN={counts['n_warn']:2d} FAIL={counts['n_fail']:2d} "
              f"INFO={counts['n_info']:2d} ERROR={counts['n_error']:2d}")

    # Top 10 most alarming findings: FAIL first, then WARN, then ERROR.
    alarming = []
    for dim in DIMENSIONS:
        for c in RESULTS[dim]:
            if c["status"] in ("FAIL", "WARN", "ERROR"):
                alarming.append((c["status"], dim, c))
    order = {"FAIL": 0, "ERROR": 1, "WARN": 2}
    alarming.sort(key=lambda x: order.get(x[0], 3))
    print("\nTOP 10 MOST ALARMING FINDINGS")
    print("-" * 72)
    for status, dim, c in alarming[:10]:
        print(f"  [{status}] {c['check_id']} ({dim}) {c['description']}")
        print(f"        metric={c['metric']} value={c['value']} expected={c['expected']}")
    if not alarming:
        print("  (none)")
    print("=" * 72)


def main() -> int:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    WB, WV, DR, DW = load_data()

    completeness(WB, WV, DR, DW)
    validity(WB, WV, DR, DW)
    consistency(WB, WV, DR, DW)
    uniqueness(WB, WV, DR, DW)
    timeliness(WB, WV, DR, DW)
    accuracy(WB, WV, DR, DW)

    summary = build_summary()

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_counts": {
            "WB": int(len(WB)), "WV": int(len(WV)),
            "DR": int(len(DR)), "DW": int(len(DW)),
        },
        "reference": REFERENCE,
    }
    tables = {
        "WB": {"rows": int(len(WB)), "columns": list(WB.columns)},
        "WV": {"rows": int(len(WV)), "columns": list(WV.columns)},
        "DR": {"rows": int(len(DR)), "columns": list(DR.columns)},
        "DW": {"rows": int(len(DW)), "columns": list(DW.columns)},
    }

    output = {
        "meta": meta,
        "tables": tables,
        "dimensions": RESULTS,
        "summary": summary,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False, default=str)

    print_console_summary(summary)
    print(f"\n[write] results -> {RESULTS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
