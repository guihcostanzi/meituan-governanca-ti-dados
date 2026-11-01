# Meituan INFORMS TSL Dataset — Privacy & Ethics Assessment

**Assessment date:** 2026-10-06
**Scope:** the four raw CSVs in this repository (`all_waybill_info_meituan_0322.csv`,
`courier_wave_info_meituan.csv`, `dispatch_rider_meituan.csv`, `dispatch_waybill_meituan.csv`;
~939,060 rows total), `License.txt`, and the three official documents
(`TSL-Meituan challenge_background and data_20240321.pdf`,
`tsl_meituan_2024_data_report_20241015.pdf`,
`Wang-etal-2025_...pdf`).

**Method:** each privacy/ethics claim below is grounded in either (a) a direct data probe of the
raw CSVs (figures re-derived with pandas — see [§8 Reproducibility](#8-reproducibility)) or (b) an
explicit passage (or verified absence of a passage) in the official documents. This complements —
and does not repeat — [`QUALITY_REPORT.md`](QUALITY_REPORT.md), which assesses *data quality*; this
report assesses *privacy and ethics*.

---

## 0. Executive summary

This is **pseudonymized, re-identifiable, precise spatio-temporal human-mobility data** — **not
anonymous data**. It contains personal data about at least three populations: **customers,
couriers, and merchants**.

The dominant ethical/legal exposure is the combination of:

1. **Translation-only geographic masking** (coordinates uniformly *shifted*, never scaled), which
   perfectly preserves every pairwise distance and the entire shape of the point cloud, and is
   therefore recoverable by map-matching;
2. **Sub-meter coordinate precision** (~0.11 m at 1 micro-degree) applied to customers' precise
   **residence** points;
3. **A small, single-city population** (4,955 couriers; 568,546 orders in one city over 8 days),
   which makes singling out individuals plausible;
4. **Documentation that materially assists re-identification** (disclosed operating box, counts,
   density heat maps, and example coordinates in the *un-shifted* real-world frame).

There is **no documented consent, ethics review, or data-use agreement** anywhere in the released
materials (verified: no occurrence of *ethic / consent / IRB / GDPR / PIPL / re-identification* in
any of the three PDFs). The only protections cited are ID relabeling and coordinate translation.
The licence is **CC BY-NC 4.0** (non-commercial, no redistribution, attribution required).

**Overall:** usable **only** for non-commercial academic research, with a documented lawful basis
and ethics review, under a strict no-re-identification / no-augmentation / no-publication-of-raw-
coordinates policy. It must be described in publications as **pseudonymized and potentially
re-identifiable**, never as anonymous.

### Risk scorecard

| Dimension | Risk | One-line justification |
|---|---|---|
| Existence of personal data | **High** | Precise customer/courier geolocation + event timestamps under pseudonymous IDs |
| Possibility of re-identification | **High** | Translation-only masking, city fingerprinting, doc coordinate leak, dense IDs, unique spatio-temporal quasi-identifiers |
| Sensitive data | **Moderate–High** | Residence + restaurant choice + timing → health/religion/household inference; worker monitoring |
| Possible biases | **Moderate** | Platform / survivorship / noon-peak / single-city / omitted-covariate biases |
| Underrepresented groups | **Significant** | No customer demographics; tail workers, merchants, and geographies thin |
| Usage limitations | **Restrictive** | CC BY-NC 4.0, no redistribution, no consent/ethics docs, bounded scientific scope |

---

## 1. Existence of personal data — YES, extensive

| Data element (source) | Who it describes | Classification |
|---|---|---|
| `recipient_lat/lng` (WB) | Customer's delivery point = **home/work** | Personal data; precise geolocation; arguably **sensitive** (residence) |
| `sender_lat/lng`, `poi_id` (WB) | Merchant (restaurant) | Business data, re-identifiable to a real establishment |
| `courier_id`, `grab_lat/lng` (WB), `rider_lat/lng` (DR), wave times (WV) | Courier (gig worker) | Personal data; **worker location tracking / monitoring** |
| All event timestamps `platform_order_time … arrive_time` (WB) | Customer + courier + merchant | Personal data (when/where a person acted) |
| `order_id`, `waybill_id`, `da_id`, `wave_id` | Links across the above | Pseudonymous identifiers |

**What is *not* present:** no names, phone numbers, address-as-text, payment, or account data. The
tables contain only numeric IDs, integer coordinates, and Unix-second timestamps.

**Why it is nonetheless personal data:**

- The customers' **precise residence** is the sensitive payload: **350,603 distinct delivery
  coordinates** across 654,343 orders, at **1 micro-degree (~0.11 m) resolution** — finer than a
  single apartment unit. 60% of these coordinates occur exactly once; the most-repeated coordinate
  carries **362** orders.
- Under GDPR Recital 26 (and comparable regimes) data linkable to an individual by any means is
  personal data; **pseudonymized data remains personal data**. Under China's PIPL, location /
  whereabouts is expressly **sensitive personal information** — and the source is a Chinese
  platform.
- No attribute data (age, gender, income) exists for customers or couriers, but high-resolution
  location + time *is itself* the sensitive payload.

**Finding:** personal data exists for three stakeholder populations. There is **no k-anonymity
protection** on the ID space (see §2).

---

## 2. Possibility of re-identification — HIGH / plausible

Five compounding weaknesses, each verified against the data or documents.

### 2.1 Geographic masking is a single 2-D translation
The Data Report (§2.1) states all lat/lng are *"uniformly shifted but not scaled … the impact on
the spherical distance due to the latitude shift can be neglected."* A rigid translation preserves
**all pairwise distances and the entire shape** of the point cloud. It is a shift cipher, not
k-anonymization or noise addition. Standard **map-matching** against a road network, building
footprints, or POI database can recover a translation-invariant match and thereby the shift. Once
the shift is recovered, **every coordinate de-anonymizes at once**.

### 2.2 The city is fingerprinted by the public materials
The disclosed metadata — operating box **50 km × 40 km**, window **17–24 Oct 2022**, counts of
**568,546 orders / 4,955 couriers / 4,962 merchants / 23 districts**, and density heat maps
labelling **three subdistricts** — is sufficient to geo-locate the "anonymized midsize Chinese
city." City-scale identification turns the unknown shift into a solvable alignment problem.

### 2.3 Documentation leaks un-shifted example coordinates
The Background & Data Description uses sample coordinates `sender lat 39996108`, `sender lng
116481447` — i.e. **39.996108 N, 116.481447 E, real Beijing**. The released data occupies a *different*
frame (observed bounds `lng ∈ [174,387,310; 174,999,535]`, `lat ∈ [45,729,124; 46,079,853]`
micro-degrees). If any worked example corresponds to a real record, a single known
plaintext→masked-coordinate pair **solves the translation outright**. At minimum, the public docs
disclose the true coordinate frame and plausible real values.

### 2.4 Sequential, gap-free IDs
All identifiers are dense `0..N-1` relabelings (verified):

| Column | Distinct | Min | Max | Dense (`max+1 == distinct`) |
|---|---:|---:|---:|:--:|
| `order_id` | 568,546 | 0 | 568,545 | ✅ |
| `waybill_id` | 654,343 | 0 | 654,342 | ✅ |
| `courier_id` | 4,955 | 0 | 4,954 | ✅ |
| `poi_id` | 4,962 | 0 | 4,961 | ✅ |
| `da_id` | 23 | 0 | 22 | ✅ |

This is simple re-indexing, not a keyed pseudonymization; it resists nothing and provides no
k-anonymity. If the platform's original mapping ever leaks, linkage is trivial.

### 2.5 Rich quasi-identifiers / singling-out
- **86.81%** of orders (568,057 / 654,343) are unique on `(recipient location, order time)` alone.
- **32.1%** of orders sit at a delivery coordinate that appears exactly **once** in the dataset — a
  near-unique quasi-identifier.
- **116** coordinates carry ≥50 orders (apartments / dorms / offices); max **362** — enabling
  household-level inference.
- Wave structure (`order_ids` sets of size 1–59) plus acceptance timing forms a **behavioural
  fingerprint per courier** in a population of only ~5,000, sufficient to single out workers.

**Finding:** the release design is re-identifiable in principle, and the published descriptive
material materially assists it. **Treat as pseudonymized, not anonymized.**

---

## 3. Sensitive data — YES (inferable), incl. potential special-category inference

No sensitive field is declared, but the data supports **inference of sensitive attributes**:

- **Precise residence / whereabouts** — sensitive under PIPL; high-risk personal data under GDPR.
- **Dietary / health / religious inference** — `poi_id` identifies the exact restaurant of every
  order. Repeated food choices can reveal dietary restrictions, medical conditions (e.g. diabetic /
  renal diets), pregnancy, or religious practice (halal / kosher). This is a route to **GDPR Art. 9
  special-category** inference even though no health field exists.
- **Worker surveillance** — per-acceptance courier coordinates and timings enable productivity and
  pressure analysis of identifiable gig workers (a vulnerable group). The Background document
  frames courier rejection and "willingness to accept" as objects of study; the same data can be
  repurposed for worker discipline.
- **Vulnerable-location exposure** — deliveries to schools, hospitals, shelters, or places of
  worship are exposed at unit-level precision (derivable via map-matching; not separately flagged).
- **Household inference** — repeated orders to one coordinate (up to 362) reveal multi-person
  households and daily routines.

**Finding:** the generator avoided collecting direct sensitive fields, but the geographic +
merchant + temporal detail creates a material inference surface.

---

## 4. Possible biases

| Bias | Evidence | Consequence |
|---|---|---|
| **Platform / single-vendor selection** | Only Meituan orders; no competitor data | Findings are Meituan-specific, not market-wide |
| **Selection on cancellation** | Wang et al. explicitly **exclude merchant-cancelled orders** ("rare in practice") | Survivorship bias in demand / latency estimates |
| **Right-censoring of rejections** | **13.11%** of waybills (`is_courier_grabbed=0`) have the entire outcome timeline zeroed | Duration / punctuality models biased toward accepted supply unless rejection is modelled |
| **Dispatch-table sampling** | Only **3 noon-peak epochs/day × 8 = 24 checkpoints** (~600 orders / ~2,500 couriers each) | Dispatch-policy conclusions valid only for noon peaks |
| **Geographic concentration** | Three dense subdistricts dominate the 50 × 40 km box | Peripheral / rural demand underweighted |
| **Temporal narrowness** | 8 consecutive days, Oct 2022; **~3.9-year lag** | No seasonality/annual cycle; not representative of current behaviour |
| **Active-only courier base** | 54 couriers rejected *all* orders yet are counted; 371 couriers have ≤10 orders | "Average courier" skewed; fragile tail statistics |
| **Omitted covariates** | No weather, traffic, road network, pricing, demand, customer attributes | Omitted-variable bias; attribution errors |
| **Ambiguous `da_id=0`** | **22.76%** of orders (largest group) | If `0` means "unassigned" rather than a district, district-level analyses are biased |

Because the data is internally clean (see [`QUALITY_REPORT.md`](QUALITY_REPORT.md)), these are
**structural/design biases**, not measurement noise. They are largely inherent to a single-platform,
single-city snapshot but must be stated in any publication.

---

## 5. Underrepresented groups

- **Customers — total absence of attributes.** No age, gender, income, disability, or ethnicity, so
  **demographic fairness cannot be assessed at all**; vulnerable consumers (elderly, disabled,
  low-income) are invisible.
- **Couriers (tail).** Highly unequal workload: **40.2% of couriers and 59.8% of merchants generate
  only ~10% of orders** (published analysis); **64 couriers appear with a single order**, 371 with
  ≤10. Part-time, new, night-shift, and non-noon-peak couriers are statistically marginalised
  (dispatch data is noon-only).
- **Merchants (tail).** 4,962 merchants; small / independent restaurants contribute little signal
  versus high-volume chains → minority cuisine types and micro-businesses are underrepresented and
  may be dropped by frequency cutoffs.
- **Geography.** Peripheral districts and less densely populated areas are structurally thin.
- **Temporal working patterns.** Weekend = only **2 of 8** days; night / early-morning operations
  absent (noon-peak sampling); pre-booked orders only **3.68%** (published ~3.9%).
- **Workforce categories.** The 54 all-rejecting couriers and idle dispatch candidates are counted
  but contribute almost no behaviour — a representation imbalance in any behavioural model.

**Finding:** the asset is optimised for noon-peak, high-volume, central-city operations. It cannot
support equity analyses and systematically under-represents workers and businesses at the margins.

---

## 6. Usage limitations

### 6.1 Legal / licensing (binding)
- **CC BY-NC 4.0** (`License.txt`): **non-commercial only** (academic research, teaching, scientific
  publishing); **redistribution prohibited**; attribution mandatory — *"This research was supported
  by data provided by Meituan."* Meituan's terms **override** conflicting third-party licences.
- **No consent, ethics-review, or data-use-agreement documentation exists** in any of the three
  PDFs. Verified absence of the terms *ethic, consent, IRB, GDPR, PIPL, re-identification*. The only
  privacy controls described are ID relabeling and coordinate translation.
- Because the data is **pseudonymized personal data**, a lawful basis (research exemption /
  legitimate interest) is required. Treating it as anonymous and processing it freely would be
  unlawful under GDPR and PIPL. Cross-border use is additionally constrained by PIPL's rules on
  Chinese personal information.

### 6.2 Ethical (recommended prohibitions)
- Do **not** attempt re-identification, map-matching, or augmentation with external datasets to
  locate the city or individuals.
- Do **not** use for worker surveillance, performance scoring, disciplinary, insurance, credit,
  hiring, or any consequential decision about identified couriers or merchants.
- Do **not** publish unit-level coordinates, raw `poi_id`↔location joins, or individual
  trajectories; aggregate / perturb before release.
- Do **not** infer sensitive attributes (health, religion, etc.) from merchant / diet patterns.

### 6.3 Scientific / inferential limits (stated by the authors)
- Research on **pricing, recommendation mechanisms, and competition is not supported** (Wang et al.).
- One city, 8 days, 2022 → no generalization claim to other cities or periods; ~3.9-year staleness
  invalidates operational use.
- No road network, weather, or traffic covariates; no ground truth for *why* a courier rejects.
- Small independent groups (tail entities, 8 days) → high variance; require temporal hold-out
  (split by day), regularization, and entity embeddings.

---

## 7. Conditions for use — recommended research protocol

1. **Classify correctly.** Describe the dataset as **pseudonymized and potentially re-identifiable**,
   never as anonymous, in all publications and DMPs.
2. **Establish a lawful basis** and complete an ethics review / IRB screening before processing,
   covering the courier-monitoring and location aspects.
3. **Minimize.** Do not retain raw `recipient_*` coordinates longer than needed; prefer aggregated or
   perturbed geometries for analysis where possible.
4. **No re-identification / no augmentation.** Forbid linkage to external location datasets.
5. **Aggregate before publishing.** Never release unit-level coordinates, trajectories, or
   `poi_id`↔location joins.
6. **No consequential individual decisions.** Prohibit use for worker discipline, insurance, credit,
   hiring, or any decision about a identified individual.
7. **Report the structural biases.** Platform selection, cancellation exclusion, noon-peak sampling,
   single-city/8-day scope, and the tail-sparsity of couriers/merchants.
8. **Honor the licence.** Non-commercial only, no redistribution, Meituan acknowledgement, and
   record the licence and preprocessing with any derived artifact.

---

## 8. Reproducibility

The privacy-relevant figures in this report were re-derived directly from the raw CSVs with pandas.
Key probes:

```python
import pandas as pd, numpy as np
WB = "all_waybill_info_meituan_0322.csv"

df = pd.read_csv(WB, usecols=["dt","order_id","waybill_id","courier_id","da_id","is_courier_grabbed",
                              "is_prebook","recipient_lng","recipient_lat","platform_order_time"])

# 2.4 dense, gap-free identifiers
for c in ["order_id","waybill_id","courier_id","da_id"]:
    s = df[c]; assert s.max() + 1 == s.nunique()   # dense 0..N-1

# 1 / 2.5 distinct delivery coordinates and k-anonymity of homes
r = df[["recipient_lng","recipient_lat"]]
vc = r.value_counts()
print("distinct delivery coords:", len(vc))          # 350,603
print("coords used once:", (vc == 1).sum())          # 210,214 (60.0%)
print("coords used >=50x:", (vc >= 50).sum())        # 116
print("max orders at one coord:", vc.max())          # 362

# 2.5 spatio-temporal uniqueness
key = list(zip(df.recipient_lng, df.recipient_lat, df.platform_order_time))
print("unique (loc,time) fingerprints:", len(set(key)) / len(df))  # 86.81%

# 4 / 5 shares
print("rejected waybills: %.2f%%" % ((df.is_courier_grabbed == 0).mean() * 100))  # 13.11%
print("prebook: %.2f%%" % (df.is_prebook.mean() * 100))                          # 3.68%
print("da_id=0 share: %.2f%%" % ((df.da_id == 0).mean() * 100))                  # 22.76%
```

Document-level claims (translation-only shift, absence of ethics/consent terms, dispatching
sampling) were verified by full-text extraction of the three PDFs with `pdftotext -layout` and
case-insensitive search for *ethic, consent, IRB, GDPR, PIPL, re-identification, anonym\*,
confidential, privacy, shifted*.

**Provenance confidence:** high — the data is published by the platform itself and accompanied by
two official clarification documents plus a descriptive paper. This is a *primary* source, not a
scraped mirror.

---

## 9. Related documents

- [`QUALITY_REPORT.md`](QUALITY_REPORT.md) — data-asset quality assessment (66 checks, 6 dimensions).
- [`VERIFICATION_SPEC.md`](VERIFICATION_SPEC.md) — executable quality-check definitions.
- `License.txt` — CC BY-NC 4.0 terms.
- `TSL-Meituan challenge_background and data_20240321.pdf` (§2.1 shift; Table 1/2 un-shifted examples).
- `tsl_meituan_2024_data_report_20241015.pdf` (§2.1 shift; §2.4 rejected waybills).
- `Wang-etal-2025_...pdf` (city scale, counts, dispatch sampling, excluded cancellations).
