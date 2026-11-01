# Meituan INFORMS TSL Dataset — Data-Asset Quality Assessment

**Assessment date:** 2026-10-03
**Scope:** four raw CSVs in this repository (`all_waybill_info_meituan_0322.csv`, `courier_wave_info_meituan.csv`,
`dispatch_rider_meituan.csv`, `dispatch_waybill_meituan.csv`)
**Method:** 66 executable checks across six quality dimensions, defined in
[`VERIFICATION_SPEC.md`](VERIFICATION_SPEC.md), implemented in
[`run_quality_checks.py`](run_quality_checks.py), results in
[`results/quality_results.json`](results/quality_results.json). All results in this report were
independently re-derived and spot-checked against direct pandas queries, and cross-validated against the
challenge's published descriptive statistics (Wang et al., 2025).

---

## 0. Executive summary

The dataset is a **high-quality, internally coherent operational snapshot** of on-demand food delivery:
no literal missing values, unique primary keys, no full-row duplicates, zero violated lifecycle
orderings, and external statistics that reproduce published figures almost exactly (on-time rate
**84.99%** vs published **85.2%**; median delivery delay **−488 s** vs published **−8.2 min**).

It is **immediately usable for non-commercial neural-network research**, but only after a small set of
mandatory preprocessing decisions. The three issues that will silently corrupt models if ignored are:

1. **`0` is overloaded.** In `*_id` columns (0 in a *categorical* feature ≈ 23% of `da_id` values) `0` is a
   **valid anonymized entity**, not missing; in `*_time`/`*_lat`/`*_lng` columns it is a **sentinel for
   "absent"**. Treating `da_id=0` as missing discards real district-0 demand; treating `grab_time=0` as a
   real timestamp corrupts every duration feature.
2. **Rejected waybills are structurally censored** — 13.11% of waybills (85,797) have their entire outcome
   timeline zeroed. They must be modelled separately or excluded, never imputed as real events.
3. **`courier_wave_info` is partially unreliable** — `wave_start_time` disagrees with the true
   `min(grab_time)` for **31.88%** of waves (median error **238 s**, p99 **1,347 s**, max **4.4 h**). It
   cannot be trusted and must be recomputed.

### Quality scorecard

| Dimension | PASS | WARN | FAIL | INFO | Verdict |
|---|---:|---:|---:|---:|---|
| Completeness | 7 | 1 | 1 | 5 | **High** — no nulls; one defective row; documented structural absence |
| Validity | 9 | 1 | 0 | 2 | **High** — domains clean; all exceptions explained by pre-booking |
| Consistency | 4 | 3 | 0 | 4 | **Medium-high** — cross-table joins 100%; wave timing flags |
| Uniqueness | 6 | 0 | 0 | 4 | **Excellent** — keys unique, no duplicates |
| Timeliness | 3 | 1 | 0 | 3 | **Adequate for research, stale for production** (3.94-year lag) |
| Accuracy | 4 | 2 | 0 | 6 | **High** — matches published benchmarks; physical checks pass |

Overall: **20 PASS / 8 WARN / 1 FAIL / 24 INFO**. The single FAIL is one waybill in 568,546.

---

## 1. Origin & provenance

| Attribute | Value |
|---|---|
| Publisher / source | **Meituan** (China's largest on-demand food-delivery platform) |
| Programme | First INFORMS TSL Data-Driven Research Challenge (2024–2025) |
| Geography | Anonymized midsize Chinese city; ~**50 km × 40 km** operating area |
| Period | **2022-10-17 → 2022-10-24**, 8 consecutive days |
| Anonymization | IDs anonymized; **lat/lng uniformly shifted, not scaled** (integer micro-degrees); shift is distance-preserving and distance-impact negligible |
| Licence | **CC BY-NC 4.0** — non-commercial only (academic research, teaching, scientific publishing); **redistribution prohibited**; attribution required ("This research was supported by data provided by Meituan") |
| Documentation | `TSL-Meituan challenge_background and data_20240321.pdf`; `tsl_meituan_2024_data_report_20241015.pdf`; `Wang-etal-2025_...pdf` |

**Provenance confidence:** high. The data is published by the platform itself and accompanied by two
official clarification documents plus a peer-reviewed-style descriptive paper whose aggregate statistics
we reproduce independently (Section 6). This is a *primary* source, not a scraped or third-party mirror.

---

## 2. Meaning — what the asset contains

| File (alias) | Rows | Grain (verified unique key) | Key content |
|---|---:|---|---|
| `all_waybill_info_meituan_0322.csv` (`WB`) | 654,343 | `waybill_id` (unique) | One row per **waybill** = one dispatch attempt. 568,546 distinct orders; an order re-dispatched after rejection gets a new waybill (1–17 per order). Full order lifecycle timestamps + sender/recipient/courier coordinates. |
| `courier_wave_info_meituan.csv` (`WV`) | 206,748 | (`dt`,`courier_id`,`wave_id`) | One row per **wave** = idle→accept→deliver-all→idle cycle. Start/end times and the set of `order_ids`. |
| `dispatch_rider_meituan.csv` (`DR`) | 62,044 | (`dt`,`dispatch_time`,`courier_id`) | Candidate **couriers** at a dispatch checkpoint: position + set of on-hand ids. |
| `dispatch_waybill_meituan.csv` (`DW`) | 15,921 | (`dt`,`dispatch_time`,`order_id`) | **Orders awaiting assignment** at a dispatch checkpoint. |

**Event timeline** (per order/waybill):
`platform_order_time` (customer pays) → `order_push_time` (enters dispatch) → `dispatch_time`
(assigned; may precede acceptance) → `grab_time` (accepted) → `fetch_time` (picked up) →
`arrive_time` (delivered). Predictions: `estimate_arrived_time` (ETA, promised, ≥ order time + 30 min)
and `estimate_meal_prepare_time` (EMR, meal ready).

**Semantics that are easy to get wrong (all verified):**
- `is_courier_grabbed=0` ⇔ rejected waybill ⇒ `grab_time`,`fetch_time`,`arrive_time`,`grab_lat`,`grab_lng` = 0.
- `courier_wave_info` covers `order_ids`; `dispatch_rider.courier_waybills` holds **order-vocabulary** ids
  (not waybill ids — see §5.3).
- `dispatch_time` in DR/DW is a **checkpoint (batch) time**, not an assignment time.
- `estimate_meal_prepare_time` is an *estimate*; `fetch_time < estimate_meal_prepare_time` is normal.

---

## 3. Completeness

**Result: high, with one defective record and one overloaded-zero trap.**

- **No literal missing values** in any of the four files (C-01 = 0 NaN/empty everywhere). Missingness is
  encoded as `0`.
- **⚠️ `0` means different things by column type** (C-02, C-06/07/09). Verified against published
  populations: including the `0` value yields **exactly** the reference cardinalities.

  | Column | Rows = 0 | distinct incl. 0 | distinct excl. 0 | Published | Interpretation |
  |---|---:|---:|---:|---:|---|
  | `da_id` | 148,903 (22.76%) | **23** | 22 | 23 | `0` is a **valid district** |
  | `poi_id` | 757 (0.12%) | **4,962** | 4,961 | 4,962 | `0` is a **valid merchant** |
  | `courier_id` | 33 (0.005%) | **4,955** | 4,954 | 4,955 | `0` is a **valid courier** |
  | `grab_time`/`grab_lat`/… | 85,797 (13.11%) | — | — | — | `0` = **absent** (rejected waybill) |

- **Structural absence is exactly as documented** (C-03): 100% of `is_courier_grabbed=0` rows have all
  five outcome fields zeroed. Consequently those five columns are only **86.89% populated** overall, but
  **100% populated** for accepted waybills (C-04).
- **FAIL C-04 / WARN C-05 — one defective waybill:** `waybill_id=239654` (dt 2022-10-18, order 225004)
  is flagged `is_courier_grabbed=1` with valid `grab_time`/`fetch_time` but `arrive_time=0`. This is the
  only accepted row with an outcome gap (1 / 568,546 = **0.00018%**). Also explains the `arrive_time=0`
  count being exactly 1 higher than the rejected count.
- **`estimate_meal_prepare_time` absent** for 9,907 rows (1.51%) — a platform prediction not always stored.
- **Coverage is complete:** 8/8 consecutive days, 77,027–89,205 rows/day (C-10).
- **Referential completeness is 100%:** every `DW.order_id` (C-11), every `DR` on-hand id (C-12), and
  every `WV.order_ids` element (C-13) resolves to a `WB` record. `WV` couriers (4,901) and `DR`
  candidates (4,085) are **(subsets)** of the 4,955 couriers seen in `WB` (C-14).

---

## 4. Validity

**Result: high. All logical/format constraints hold; the only "violations" are documented pre-booking behaviour.**

- **Domains clean** (V-01/V-02/V-03): 8 valid `dt` dates; all three boolean flags ∈ {0,1}; no negative IDs;
  no negative coordinates (after excluding the sentinel zeros).
- **Coordinates valid** (V-04/V-05): every non-zero coordinate lies inside the observed bounding box
  (`lng ∈ [174,387,310; 174,999,535]`, `lat ∈ [45,729,124; 46,079,853]` micro-degrees — displaced from real
  Beijing by the documented uniform shift); lat/lng zeros always occur in pairs.
- **Lifecycle ordering is flawless** (V-07): across **3,014,321** evaluable adjacent timestamp pairs there
  are **0 violations** of `platform ≤ push ≤ dispatch ≤ grab ≤ fetch ≤ arrive`.
- **WARN V-06 (INFO after cause analysis):** 251 timestamp cells fall >1 day outside their `dt`; **100% are
  pre-booked orders** (placed in advance), which is expected, not an error.
- **WARN V-08:** 44.97% of accepted waybills have `fetch_time < estimate_meal_prepare_time` (p50 206 s,
  p99 632 s). Explicitly expected per Data Report §2.5.
- **Dispatch cadence** (V-10): `dispatch_time` is **not** aligned to a 30 s grid (0% divisible by 30) and
  `DR`/`DW` cover only **3 noon-peak epochs/day** (~11:25, 11:27, 11:30) — a major scope limitation (§7).
- **Re-dispatch cardinality** (V-11/U-07): 1–17 waybills per order, mean 1.15; **10.16%** of orders were
  re-dispatched at least once — an inherent, valid business fact.

---

## 5. Consistency

**Result: medium-high. Cross-table referential joins are perfect; wave timing and two flag semantics carry documented defects.**

- **WARN S-01 — weekend flag vs calendar:** 651 rows (0.099%) disagree with the `dt` weekday
  (396 weekdays flagged weekend, 255 weekend rows unflagged). **91.55% of mismatches are pre-booked** and
  have `platform_order_time` on a different calendar day than `dt` — i.e. the flag is keyed to a different
  date field for cross-day pre-orders. Correct `is_weekend` from `dt`; do not rely on the flag blindly.
- **WARN S-03 — `wave_start_time` is systematically wrong:** only **68.12%** of waves match the recomputed
  `min(grab_time)` of their orders. All 65,904 mismatches are **later** than the true minimum (never
  earlier): p50 **238 s**, p90 **690 s**, p99 **1,347 s**, max **15,692 s (4.4 h)**. This confirms and
  greatly extends the documented §2.2 indexing bug — it is **material** for any wave-duration feature.
- **WARN S-06 — wave date consistency:** 18 waves (0.0087%) contain an order whose `dt` differs from the
  wave `dt` (midnight spillover). Negligible.
- **INFO S-07 — `courier_waybills` vocabulary ambiguity:** joining those ids as `order_id` agrees with the
  candidate courier **99.57%** of the time; joining as `waybill_id` agrees only **0.45%**. The field is
  named "waybills" but stores **order ids**. Using the wrong vocabulary breaks the candidate–order link.
- **INFO S-08 — idle candidates:** 30.09% of candidate couriers have an empty on-hand set (`[]`) — expected
  (idle supply), not a defect.
- **Universe reconciliation (S-10)** matches the published paper exactly once the valid `0` entities are
  retained: **568,546 orders; 4,955 couriers; 4,962 merchants; 23 districts**; pre-book rate **3.68%** vs
  published 3.9%.

---

## 6. Uniqueness

**Result: excellent.**

| Check | Result |
|---|---|
| `waybill_id` unique (U-01) | ✅ 654,343 / 654,343 |
| (`dt`,`waybill_id`) unique (U-02) | ✅ 0 duplicates |
| Full-row duplicates, all tables (U-03) | ✅ 0 |
| (`dt`,`courier_id`,`wave_id`) unique — WV (U-04) | ✅ 0 duplicates |
| (`dt`,`dispatch_time`,`courier_id`) unique — DR (U-05) | ✅ 0 duplicates |
| (`dt`,`dispatch_time`,`order_id`) unique — DW (U-06) | ✅ 0 duplicates |

`order_id` is intentionally non-unique (re-dispatch): 568,546 orders over 654,343 waybills. Candidate
orders/couriers may legitimately appear at multiple checkpoints (U-08). No identity collisions detected.

---

## 7. Timeliness

**Result: adequate for offline research; stale for any operational use.**

- **Coverage** (T-01/T-02): 2022-10-17 → 2022-10-24, 8 distinct days, **no calendar gap**.
- **WARN T-03 — staleness:** latest event is 2022-10-24T19:57Z; as of the 2026 assessment that is a
  **1,439-day (~3.94-year) lag**. Fine for methodological research and retrospective benchmarking;
  **not** representative of current demand, pricing, or courier populations.
- **No truncation** (T-04): the final day (2022-10-24) has a 554,480 s event span vs a 561,670 s median —
  effectively complete.
- **Latency profile** (T-06, seconds, evaluated pairs only): platform→push p50 **13**; push→dispatch p50
  **125** (p99 3,697 for pre-book); dispatch→grab p50 **11** (matching the published ~25.7 s mean
  final-dispatch→acceptance); grab→arrive p50 **1,451 s (~24 min)**.
- **Time resolution** (T-07): all timestamps integer seconds (magnitude ~1.66e9, not milliseconds).
- **⚠️ Dispatch coverage is partial:** DR/DW sample only **3 noon-peak checkpoints per day**, not the full
  day. Dispatch-policy work is restricted to those windows (24 epochs total).

---

## 8. Accuracy

**Result: high. Physical constraints hold and internal statistics reproduce the published paper.**

- **No negative durations** (A-02): 0 negative adjacent differences across all lifecycle pairs.
- **On-time delivery (A-04): 84.99%** of completed waybills arrive at/before ETA; median delay **−488 s
  (−8.1 min)**, late (no buffer) 15.0%. Published figures: **85.2%** early, average delay **−8.2 min**,
  late ratio **14.8%**. Near-perfect external agreement.
- **Physics/geometry pass** (A-05/A-06): **0** impossible speeds (>120 km/h) among 568,545 evaluated
  trips; sender→recipient distance p50 **1.47 km**, p99 **4.99 km** (consistent with the published ~2.0 km
  road-distance average); 7 zero-distance-with-time records; only **38 / 568,546 (0.0067%)** couriers were
  >5 km from the merchant at acceptance (max 19.6 km — plausible long-range accepts).
- **Meal-prep plausibility** (A-07): median prep horizon **12.3 min**; **0** estimates preceding the order.
- **Distributional sanity** (A-11): daily volume CoV **5.9%** (stable); wave duration p50 **1,685 s
  (28.1 min)** — consistent with the published "50.2% of waves ≤ 30 min".
- **Duration outliers** (A-03): expected heavy tails (e.g. grab→arrive IQR outliers 9,598 / 568,545 = 1.7%),
  no impossible values.
- **Documented anomalies reproduced** (A-01): rejected-waybill zeroing 100% ✅; wave-start bug 31.88% ✅;
  pre-fetch<EMR 44.97% ✅; coordinate shift confirmed ✅. Dispatch-checkpoint semantics quantified: exact
  checkpoint match is rare (0.006%) because `dispatch_time` is a batch time, but **93.81%** of DW orders
  map to a WB dispatch within ±300 s.

---

## 9. Limitations

**Licensing**
- CC BY-NC 4.0: **non-commercial only**, **no redistribution**, mandatory attribution. Commercial or
  product use is out of scope.

**Anonymization & scope**
- Coordinates are uniformly shifted: valid for *relative* distances within the city, **not mappable** to
  real geography; no road network, weather, or traffic covariates.
- One city, 8 days, Oct 2022. No seasonal/annual coverage; generalization to other cities/periods is
  unproven. No pricing, demand, recommendation, customer-attribute, or courier-attribute data.
- No ground truth for *why* a courier rejected a waybill.

**Measurement semantics**
- `0` is overloaded (valid `*_id` vs sentinel time/coord) — the single biggest modelling trap.
- `wave_start_time` unreliable for 31.9% of waves (recompute from `grab_time`).
- `is_weekend` and 651 rows' date semantics inconsistent for cross-day pre-book orders.
- `courier_waybills` uses order vocabulary despite its name.
- Dispatch tables cover only 3 noon epochs/day; `dispatch_time` is a checkpoint, not the assignment time.

**Censoring / selection**
- 13.11% rejected waybills have no outcome timeline → any delivery-duration or punctuality model is
  **right-censored and biased toward accepted/available supply** unless rejection is modelled explicitly.
- Sparse tail: 40.2% of couriers and 59.8% of merchants contribute only ~10% of orders (per published
  analysis); rare categories need embeddings/regularization.

**Residual defects (tiny)**
- 1 accepted waybill missing `arrive_time`; 18 waves with cross-day orders; 7 zero-distance trips;
  38 long-distance accepts.

---

## 10. Conditions for using this asset in neural-network training

**Permitted use.** Non-commercial research, teaching, and scientific publication only, with the required
Meituan acknowledgement. No redistribution of the data or derivatives.

**Mandatory preprocessing before any NN use**

1. **Fix the overloaded zero.** Convert `0 → NaN` for `*_time`, `*_lat`, `*_lng`. **Keep `0` as a valid
   category** for `da_id`, `poi_id`, `courier_id` (do *not* impute/delete).
2. **Handle rejected waybills explicitly.** For outcome/duration/punctuality targets, either exclude
   `is_courier_grabbed=0` rows or model acceptance and completion as separate heads. Never impute their
   zeroed outcome fields. Frame the 13.11% as censoring, not missingness.
3. **Recompute `wave_start_time = min(grab_time)`** (and verify `wave_end_time = max(arrive_time)`) before
   using any wave-duration/labour feature.
4. **Recompute `is_weekend` from `dt`**; repair or drop the 651 inconsistent rows if the flag is a feature.
5. **Drop/inspect the one defective waybill** (`waybill_id=239654`).
6. **Treat `courier_waybills` as order ids.** Build the candidate↔order link via `order_id`, respecting the
   `dispatch_time ≤ checkpoint` constraint.
7. **Scope dispatch models to the 24 noon-peak epochs** and treat `dispatch_time` as a checkpoint.
8. **Coordinates:** keep the shifted frame; normalize/standardise; use geodesic (haversine) distance for
   intra-city geometry; do **not** interpret absolute lat/lng.

**Training-protocol recommendations**
- **Split by day, not at random**, to avoid temporal leakage (only 8 days → prefer leave-one-day-out or a
  held-out-day scheme; expect high variance).
- **Standardise per-day or per-district** where volume/seasonality matters (daily volume varies ~16%).
- **Model distribution shift** between weekday (6 days) and weekend (2 days) explicitly.
- **Sequence models (dispatch/courier behaviour):** valid, but restricted to checkpoint windows; use
  `order_ids`/`courier_waybills` sets plus coordinates as the state representation.
- **Sparsity:** use entity embeddings with regularization for merchants/couriers/districts; consider a
  minimum-frequency cutoff (~tail entities carry little gradient signal).
- **Targets:** the natural supervised labels are on-time/ETA-vs-ATA (84.99% positive, ~15% late),
  delay regression (median −8.1 min), and stage durations (grab→fetch, fetch→arrive).
- **Report the licence and preprocessing** with any derived dataset; version it. Do not redistribute.

**Overall fitness.** *Recommended with mandatory preprocessing.* The asset is clean enough for
production-grade research pipelines and is externally validated, provided the three critical traps
(overloaded zero, rejected-waybill censoring, unreliable `wave_start_time`) are handled. Its main
scientific limitation is **scope** (one city, 8 days, noon-peak dispatch sampling), not measurement
quality.

---

## 11. Reproducibility

```
quality/
├── VERIFICATION_SPEC.md          # 66 checks across 6 dimensions (definition)
├── run_quality_checks.py         # executable harness (pandas/numpy only)
├── results/
│   └── quality_results.json      # machine-readable results (per-check status + details)
└── QUALITY_REPORT.md             # this report

# re-run
python3 quality/run_quality_checks.py
```

The harness is deterministic, offline, and wraps every check so a single failure cannot abort the run
(all 66 executed with 0 errors). Findings were spot-checked with independent pandas queries and
cross-validated against the published descriptive statistics.
