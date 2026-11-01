# Data Quality Verification Specification
## Meituan INFORMS TSL On-Demand Food Delivery Dataset

**Purpose.** Define an executable, reproducible set of data-quality checks that transform the raw
Meituan CSVs into an assessed **data asset**. Checks are grouped under six quality dimensions:
*completeness, validity, consistency, uniqueness, timeliness, accuracy*. Every check has a stable
`check_id`, a plain-language description, a deterministic metric, and an expected/observed result so
it can be re-run and audited.

---

## 0. Data assets & grain

| Alias | File | Rows (expected) | Grain / primary key |
|---|---|---|---|
| `WB` | `all_waybill_info_meituan_0322.csv` | 654,343 | `waybill_id` (unique); one order → 1..N waybills |
| `WV` | `courier_wave_info_meituan.csv` | 206,748 | (`dt`, `courier_id`, `wave_id`) |
| `DR` | `dispatch_rider_meituan.csv` | 62,044 | (`dt`, `dispatch_time`, `courier_id`) |
| `DW` | `dispatch_waybill_meituan.csv` | 15,921 | (`dt`, `dispatch_time`, `order_id`) |

### Documented semantics (from the challenge PDFs — authoritative)
- **Origin**: Meituan platform; anonymized midsize Chinese city; 50 km × 40 km area;
  **17–24 Oct 2022** (8 consecutive days). Ref: Wang et al. (2025).
- **Lat/lng are uniformly shifted but not scaled** for privacy; distance impact of the shift is
  negligible (Data Report §2.1). Values are integers scaled by 1e6 (micro-degrees).
- **Missing data is encoded as `0`**, not `NaN`. For non-existent timestamps the value is `0`.
- **Rejected waybills** (`is_courier_grabbed = 0`): `grab_time`, `fetch_time`, `arrive_time`,
  `grab_lat`, `grab_lng` are `0` (Data Report §2.4).
- **`estimate_meal_prepare_time`** = estimated meal *ready* time; `fetch_time` **may be earlier**
  than it — this is expected, not an error (Data Report §2.5).
- **`da_id`** = business district; different `da_id` may geographically overlap; `da_id` may be
  associated with order type (Data Report §2.6).
- **`wave_start_time`** = earliest `grab_time` in the wave; `wave_end_time` = latest `arrive_time`.
  A known **indexing bug** can make `wave_start_time` incorrect (Data Report §2.2) → quantify.
- **Dispatch inputs**: `dispatch_time` is a *checkpoint* time, not the assignment time; candidates
  are not guaranteed to be assigned. **~1.81% of entries may not follow the rule** (Data Report §2.3).
- **Reference population counts** (Wang et al. 2025): 568,546 orders, 4,955 couriers,
  4,962 merchants, 23 business districts, ~3.9% pre-booked orders. Weekends = Oct 22–23.

### Global bounding box (observed, shifted coordinates)
`lng ∈ [174,387,310 ; 174,999,535]`, `lat ∈ [45,729,124 ; 46,079,853]` (micro-degrees).
Non-zero check should fall inside a small margin around these.

### Quality status vocabulary
- `PASS` — check satisfied, no concern.
- `INFO` — descriptive measurement, no pass/fail.
- `WARN` — deviation that is documented/expected or minor (data still usable with care).
- `FAIL` — material violation that affects downstream use; must be handled.

### Output contract
`quality/results/quality_results.json`:
```json
{
  "meta": {"generated_at": "...", "row_counts": {...}, "reference": {...}},
  "tables": {"WB": {...}, "WV": {...}, "DR": {...}, "DW": {...}},
  "dimensions": {
    "completeness": [{"check_id":"C-01","description":"...","metric":"...","value":0,
                      "expected":"...","status":"PASS","details":{}}],
    "validity": [...], "consistency": [...], "uniqueness": [...],
    "timeliness": [...], "accuracy": [...]
  },
  "summary": {"1_completeness": {"n_pass":..,"n_warn":..,"n_fail":..,"n_info":..}, ...}
}
```

---

## 1. COMPLETENESS (`C-*`)
Does the data capture all required records, fields and references?

- **C-01 Raw null scan** — count of literal `NaN`/empty per column, per table. (Expected: 0; missing
  info is encoded as 0.)
- **C-02 Sentinel-zero census** — count + pct of `0` per numeric column, per table. Flag every column
  where `0` is a sentinel rather than a true value.
- **C-03 Rejected-waybill field completeness** — for `WB.is_courier_grabbed=0`, pct of rows where
  `grab_time`,`fetch_time`,`arrive_time`,`grab_lat`,`grab_lng` are all `0`. Expected 100% (by design).
- **C-04 Accepted-waybill field completeness** — for `is_courier_grabbed=1`, count rows with any of
  `grab_time`,`fetch_time`,`arrive_time`,`grab_lat`,`grab_lng` = 0. Expected 0.
- **C-05 `arrive_time` gap** — count `arrive_time=0` minus count `is_courier_grabbed=0`.
  (Already observed: +1 anomaly.)
- **C-06 `da_id` missingness** — count + pct `da_id=0`. (Observed ≈ 22.8%.)
- **C-07 `poi_id` missingness** — count + pct `poi_id=0`. (Observed ≈ 757.)
- **C-08 `estimate_meal_prepare_time` missingness** — count + pct `=0`. (Observed ≈ 9,907.)
- **C-09 `courier_id` missingness (WB)** — count + pct `courier_id=0`. (Observed ≈ 33.)
- **C-10 Daily coverage** — rows per `dt`, distinct days, and whether the 8 days are consecutive
  with no gap. Report rows/day min–max.
- **C-11 Referential completeness DW→WB** — pct of `DW.order_id` present in `WB.order_id`.
- **C-12 Referential completeness DR→WB** — pct of waybill ids in `DR.courier_waybills` present in
  `WB.waybill_id`.
- **C-13 Referential completeness WV→WB** — pct of ids in `WV.order_ids` present in `WB.order_id`.
- **C-14 Cross-table coverage** — pct of `WV.courier_id` present in `WB.courier_id`; pct of
  `DR.courier_id` present in `WB.courier_id`.

## 2. VALIDITY (`V-*`)
Does each value conform to its declared domain, format and logical constraints?

- **V-01 `dt` domain** — set of distinct `dt` values; verify `YYYYMMDD`, 8 values, Oct 17–24 2022.
- **V-02 Boolean domains** — `is_courier_grabbed`,`is_weekend`,`is_prebook` ∈ {0,1} (all tables where
  present); report any other value count.
- **V-03 ID non-negativity** — `order_id`,`waybill_id`,`courier_id`,`da_id`,`poi_id`,`wave_id` ≥ 0;
  report negatives.
- **V-04 Coordinate nonzero & bounds** — for non-zero sender/recipient/grab/rider coords, verify they
  lie within the observed global bounding box ± margin; report out-of-box count.
- **V-05 Coordinate pairing** — `grab_lat==0` iff `grab_lng==0`; count mismatches. Same for sender and
  recipient pairs.
- **V-06 Timestamp range** — every non-zero timestamp ∈ [dt_day_start − 1 day, dt_day_end + 1 day]
  (allow cross-midnight spillover); report out-of-range count per timestamp column.
- **V-07 Order timeline monotonicity** — violations of
  `platform_order_time ≤ order_push_time ≤ dispatch_time ≤ grab_time ≤ fetch_time ≤ arrive_time`
  (evaluated only on non-zero pairs). Report count per adjacent pair.
- **V-08 Pre-fetch vs meal-ready** — count `fetch_time < estimate_meal_prepare_time` (allowed by
  §2.5; quantify magnitude distribution). Status `INFO`/`WARN`, not `FAIL`.
- **V-09 Wave time order** — `wave_start_time ≤ wave_end_time`; count violations.
- **V-10 Dispatch grid alignment** — pct of `dispatch_time` values divisible by 30 (batch cadence);
  report distribution of second-of-minute.
- **V-11 Order→waybill cardinality** — `waybills per order` distribution (min/max/mean/percentiles);
  orders with >1 waybill = re-dispatched. Verify each order has ≥1 waybill.
- **V-12 `estimate_arrived_time` sanity** — non-zero and `≥ platform_order_time`; report violations.

## 3. CONSISTENCY (`S-*`)
Do fields and tables agree with each other?

- **S-01 `is_weekend` vs `dt`** — derive weekday from `dt`; Oct 22–23 are weekend. Count rows where
  flag disagrees with calendar. Report per-day flag rate.
- **S-02 `is_courier_grabbed` vs `grab_time`** — count rows where `is_courier_grabbed=1 & grab_time=0`
  or `=0 & grab_time≠0` (should be 0).
- **S-03 Wave start correctness** — for each wave, recompute `min(grab_time)` over its orders
  (is_courier_grabbed=1) and compare to `wave_start_time`; report exact-match pct and mismatch count
  (documents bug §2.2).
- **S-04 Wave end correctness** — recompute `max(arrive_time)` and compare to `wave_end_time`;
  report exact-match pct.
- **S-05 Wave order set validity** — every id in `WV.order_ids` exists in `WB`; no empty lists; list
  length ≥ 1; sum of list lengths across waves vs accepted-waybill count.
- **S-06 Wave date consistency** — for a wave, the `dt` of all its orders equals `WV.dt`; report
  mismatch pct.
- **S-07 Courier-waybill ownership** — for ids in `DR.courier_waybills`, the `WB.courier_id` equals
  `DR.courier_id`; report mismatch pct (documents candidate-vs-assigned nuance).
- **S-08 DR on-hand set vs WB** — pct of `DR.courier_waybills` ids existing in WB (see C-12) and being
  nonempty; report empty-list count.
- **S-09 DW order existence & type** — `DW.order_id` ∈ WB, and is an integer; report bad rows.
- **S-10 Universe reconciliation** — observed distinct counts vs published reference: orders
  (568,546), couriers (4,955), merchants `poi_id` (4,962), districts `da_id` (23), pre-book rate
  (3.9%). Report deltas. Note `da_id=0` must be excluded before comparing district count.
- **S-11 Weekend share** — pct of waybills on weekend vs expectation (2/8 days).

## 4. UNIQUENESS (`U-*`)
Are keys unique and are duplicates controlled?

- **U-01 `waybill_id` uniqueness (WB)** — duplicates count (expect 0).
- **U-02 (`dt`,`waybill_id`) uniqueness** — duplicates (expect 0).
- **U-03 Full-row duplicates per table** — count (expect 0).
- **U-04 (`dt`,`courier_id`,`wave_id`) uniqueness (WV)** — count (expect 0).
- **U-05 (`dt`,`dispatch_time`,`courier_id`) uniqueness (DR)** — count (expect 0).
- **U-06 (`dt`,`dispatch_time`,`order_id`) uniqueness (DW)** — count (expect 0).
- **U-07 `order_id` cardinality** — distinct orders vs rows; waybill-per-order distribution (re-dispatch
  rate). 1 order : 1..N waybills is expected, not a defect.
- **U-08 Candidate duplication (DW)** — orders appearing at >1 dispatch checkpoint; report pct
  (expected for delayed assignment).
- **U-09 Duplicate `poi_id` coords / `da_id` overlap** — optional descriptive: same poi_id with
  differing coordinates → count (identity integrity).
- **U-10 Courier appearing on same day in both WB and WV** — descriptive join cardinality sanity.

## 5. TIMELINESS (`T-*`)
Is the data current, and is time information internally usable?

- **T-01 Coverage window** — min/max `dt`; span in days; number of distinct days.
- **T-02 Gap check** — verify no missing calendar day in the 8-day window.
- **T-03 Snapshot age** — age of latest timestamp vs `generated_at`; flag staleness for current
  operations. (Data ends 2022; report years of lag.)
- **T-04 Final-day truncation** — max event time per `dt`; test whether the last day is complete
  (compare per-day observed time span / row count to the full-day median). Flag truncated days.
- **T-05 Event-time dispersion** — min/max of each timestamp column within each `dt`; report
  out-of-day (cross-midnight) records.
- **T-06 Latency profile** — distribution (median/p90/p99) of `order_push_time − platform_order_time`,
  `dispatch_time − order_push_time`, `grab_time − dispatch_time`, `arrive_time − grab_time`. Report
  median/p90/p99 and negative counts.
- **T-07 Timestamp resolution** — all timestamps integer seconds; report fractional/non-integer count.

## 6. ACCURACY (`A-*`)
Are values plausible/correct against physical, temporal and business constraints, and do they
reproduce the documented anomalies?

- **A-01 Documented anomaly reproduction** — confirm each documented caveat at its stated rate:
  (a) rejected-waybill zeroing §2.4; (b) wave_start bug §2.2; (c) ~1.81% dispatch non-conformance §2.3;
  (d) `fetch_time < estimate_meal_prepare_time` §2.5; (e) coordinate shift §2.1 (ranges displaced from
  real Beijing ~39.9/116.4).
- **A-02 Negative durations** — count negative values across all adjacent timestamp differences used in
  T-06 (impossible).
- **A-03 Duration outliers** — IQR outlier count + bounds for `arrive_time − grab_time`,
  `fetch_time − grab_time`, `dispatch_time − platform_order_time`, wave duration. Report extreme
  (> p99.9 or physically impossible) examples.
- **A-04 On-time delivery rate** — pct `arrive_time ≤ estimate_arrived_time` among completed waybills;
  compare with plausible platform SLA (expect majority on-time). Report distribution of
  `arrive_time − estimate_arrived_time`.
- **A-05 Geodesic/implied-speed check** — using shifted coords with the *shift treated as negligible*
  (per §2.1) compute haversine distance sender→recipient; from `fetch_time→arrive_time` compute implied
  speed; flag physically impossible speeds (> ~120 km/h) or zero-distance-with-time. Report count/pct
  and speed distribution.
- **A-06 Courier-at-pickup proximity** — haversine distance between `(grab_lat,grab_lng)` and
  `(sender_lat,sender_lng)` at `grab_time`; report distribution and implausible values (> 5 km).
- **A-07 Meal-prep plausibility** — `estimate_meal_prepare_time` relative to `platform_order_time`;
  report median/p90 prep horizon and impossible (before order) count.
- **A-08 Precision/scale consistency** — verify coordinates are integers in micro-degree scale
  (magnitude > 1e6), not degrees; verify times are seconds not ms (magnitude check vs 1e9/1e12).
- **A-09 Cross-field business rule** — `is_prebook=0` → normal lifecycle; count prebook orders whose
  `platform_order_time` lies far before `order_push_time` (prebook behaviour); descriptive.
- **A-10 Internal reference consistency** — courier count in WB vs WV vs DR; flag whether rejected
  couriers (54 per paper that rejected everything) explain the differences.
- **A-11 Distributional plausibility** — daily order volume stability (min/max/CoV across 8 days)
  and wave duration distribution; flag suspicious spikes/drops.
- **A-12 Impossible coordinate == exact 0 for accepted** — accepted waybill with coordinate pair 0
  (covered C-04) is either missing or impossible; cross-report count.

---

## Execution notes for the implementer
- Use **pandas** only (duckdb/pyarrow unavailable). Read CSVs with `dtype` kept as int where possible;
  drop the leading unnamed index column (BOM on WB column 0).
- Parse list-like fields (`order_ids`, `courier_waybills`) with `ast.literal_eval` guarded by try/except;
  validate every element is a non-negative int.
- Adjacent-pair checks in V-07 must **ignore pairs where either side is 0** (structurally absent) and
  must be reported with both the "evaluable pairs" denominator and the violation count.
- Every check must never raise on malformed input: wrap per-check execution, record `"status":"ERROR"`
  with the exception message instead of aborting the run.
- Write results to `quality/results/quality_results.json` and print a concise console summary
  (dimension → pass/warn/fail counts and the top 10 most alarming findings).
- Deterministic and re-runnable; no network; runtime target < 30 min.
