# Publication-Grade Audit of `julia_descriptive/` Pipeline

**Date:** 2026-06-01
**Method:** 12-agent workflow — scout + 6 parallel per-file reviewers + 3 cross-cutting reviewers + adversarial verifier + synthesis.
**Adversarial verification:** 14/15 findings confirmed, 0 refuted, 1 inconclusive.
**Overall verdict:** `critical_bugs_found` — pipeline is NOT regression-ready.

---

## Top-line summary

The pipeline is a careful descriptive prototype but contains **six confirmed critical defects** that compromise identification of the headline triple-difference. Treating the current `merged_us_eu_matched.parquet` as a regression panel would produce biased β with no clean way to recover the true magnitude. Estimated remediation: **1–2 weeks of focused engineering + re-runs** before the panel can support a publication-grade test.

---

## CRITICAL findings (6) — must fix before any regression

### C1. Portfolio weight uses a GLOBAL denominator, not EU-restricted

- **Location:** `04_us_ownership_european.jl:127-134` (country_total_ct); `05_combine_visualize.jl:339-361, 437-493`
- **Issue:** `w_{i,c,t}` sums over ALL `sec_country` (global), not just EU. US weights are deflated ~6-7× and US/NONUS series use inconsistent denominators.
- **Fix:** Add `WHERE sec_country IN $EU_str` to country_total_holdings; build parallel `portfolio_weight_eu`; designate EU-restricted as the regression input.

### C2. Pre-2003 ChinaExposure silently coerced from NULL to 0

- **Location:** `05_combine_visualize.jl:289-296`
- **Issue:** `COALESCE(e.n_cn_total, 0)` etc. converts the pre-Revere window (1999-2002) into "verified zero exposure" instead of "missing" — directly violates the documented construction fact. Pollutes the ZERO bucket with 4 years of phantom non-exposed firm-quarters.
- **Fix:** `CASE WHEN om.report_date < DATE '2003-03-31' THEN NULL ELSE COALESCE(e.X, 0) END`; add `in_revere_coverage` flag.

### C3. ChinaExposure joined CONTEMPORANEOUSLY at t, not t-1

- **Location:** `05_combine_visualize.jl:302-307`; `02_china_exposure.jl` has no LAG anywhere
- **Issue:** Regression spec requires `ChinaExposure_{i,t-1}` but the merged panel persists only the contemporaneous value. Bucket CTE at 05:579 uses `china_share > 0.20` at the *same* quarter as Δw.
- **Fix:** Add `LAG(china_share) OVER (PARTITION BY sec_entity_id ORDER BY report_date) AS china_share_lag1q` to the merged build; use ONLY the lagged column in any regression interaction.

### C4. china_share numerator and denominator use asymmetric edge sets

- **Location:** `02_china_exposure.jl:217-236` (eu_any_edge Path 2: `src_region <> tgt_region`) vs. `159-186` (eu_china_edge has no analogous filter)
- **Issue:** Denominator drops EU-EU edges from the target firm's count, but numerator does not — inflates china_share for firms with predominantly inbound EU links.
- **Fix:** Rebuild denominator with per-edge dedup; UNION-of-distinct (eu_firm, rel_id) with role tagging, OR keep both EU_SRC and EU_TGT rows for EU-EU edges (count once per endpoint, symmetric with rest of world).

### C5. MOST_RECENT Revere dedup = textbook silent look-ahead

- **Location:** `02_china_exposure.jl:89-112` (ROW_NUMBER PARTITION BY company_id ORDER BY end_d DESC, rn=1); propagated through line 263 (EU universe) and 359-361 (firm_quarter identifiers)
- **Issue:** Stamps the LATEST CUSIP/ISIN/SEDOL/home_region onto every quarter back to 2003. A firm that became EU-listed in 2020 enters the EU universe for 2005. Same pattern family as the firm-MAX leak that cost the **sibling robots.txt paper** once (per `project_canonical_audit_2026_05_17.md` memory) — first occurrence in geoecon.
- **Fix:** Replace single-row dedup with time-versioned join: at each quarter, pick the rev_co_raw row where `start_d <= qend AND (end_d IS NULL OR end_d >= qend)`. Use `FIRST_VALUE(x IGNORE NULLS)` per FIELD.

### C6. EU firm universe is selection-on-outcome

- **Location:** `05_combine_visualize.jl:113-123`; `04:60-74`; `05:547-565`
- **Issue:** EU firms enter the panel only if "held by ≥1 institution in FactSet ownership." But the regression's outcome IS institutional holdings. Extensive-margin exits — the core behavior being measured — silently vanish. The 2.27M observation count is the inner-join output, not a Cartesian grid.
- **Fix:** Build EU firm universe from `Factset_Security_coverage` (ISO_COUNTRY in EU, ACTIVE=1, ISSUE_TYPE in EQ+AD). LEFT-JOIN ownership onto a (firm × country × quarter) grid; zero-fill (i,c,t) cells. Drop `gap_months=6`; require only that t-1 and t+1 exist on the grid.

---

## HIGH-severity findings (16)

Most material:

- **AR(1) fit on quarterly mean, not monthly** as the spec requires (`05:65-98`). Re-fit on monthly series and aggregate residuals.
- **AR(1) coefficients and shock series never persisted to disk** (`05:88-102`) — pipeline is non-reproducible from repo alone.
- **`MAX(adj_shares_out) × MAX(adj_price)` in market-cap** (`04:89-108`) — repeat of the firm-MAX leak pattern.
- **TEST_MODE writes canonical filenames** with no downstream signal — 11 stray logs in `output/` suggest this has happened.
- **NONUS contaminated by tax havens**: at 2018-12-31 LU=28%, IE=18%, CH=10% of NONUS. Many are US-managed UCITS/Cayman.
- **Multi-match aggregation uses `MAX(china_share)`** instead of weighted aggregation; the "MAX is conservative" comment is wrong (MAX is the upper bound).
- **Quarter-end filter uses calendar `LAST_DAY`** — silently drops 2017-12-31 (Sunday), 2016-12-31 (Saturday) if FactSet recorded the snapshot on prior business day.
- **`ownership_share > 1` cells silently dropped** by `BETWEEN 0 AND 1` filter — removes exactly the most-US-loved firms (likely Irish-listed inversions).
- **`exposure_share_by_sec_entity` JOIN missing the DISTINCT** subquery — harmless under MAX, breaks immediately if aggregation switches to SUM/AVG.
- **20.9% of EU sec_entity_ids unmatched to Revere** (10,083 / 12,743), NOT the briefing's 92.5%. Reconcile.
- **±1% hard trim on Δw** combined with `china_share > 0.20` leaves n=1-30 per quarter — extreme moves (the behavior of interest) silently dropped.
- **No atomic writes** in 02/03/04/05 — Ctrl-C / OOM leaves truncated parquets at canonical paths.
- **No duplicate-key check** on holdings panel.
- **All paths hardcoded** to user-specific Windows drives; no Project.toml/Manifest.toml.
- **No dedup audit on rev_rel** — silent inflation risk.
- **`firm_quarter_exposure` LEFT JOIN drops control-group identifiers** (zero-exposure firms get NULL eu_cusip/eu_isin/eu_sedol).

---

## MEDIUM-severity findings (~27)

Highlights:

- **`firm_month_china_exposure.parquet` alias holds QUARTER-END values** — silent type-pun that propagates downstream.
- **Bucket CTE built from US-only rows** — matched-EU firms with no US holding silently dropped from the time series.
- **Entity_type (institutional/retail/sovereign) not filtered** upstream — 13F-specific population not isolated.
- **sec_country redomiciliation fan-out** at boundary quarters.
- **ANY_VALUE / ROW_NUMBER ties non-deterministic** across DuckDB versions/threads.
- **Pre-ETL audit hardcoded SKIP_AUDIT=true** — no early warning for schema drift.
- **MOST_RECENT promises "most recent non-null per FIELD"** in comment but SQL doesn't do that — silent data loss when latest row has NULL.

---

## LOW-severity findings (~17)

Mostly hygiene (parquet compression unpinned, docstrings missing, log files committed alongside outputs, etc.). See full output for details.

---

## Recommended action sequence

1. **Stop using `merged_us_eu_matched.parquet` as a regression panel.** Add a banner in `05_combine_visualize.jl` flagging it as descriptive-only.
2. **Fix C1 (denominator)** — restrict to EU + parallel `portfolio_weight_eu` column.
3. **Fix C2 (pre-2003 NULL)** — drop the COALESCE for pre-2003.
4. **Fix C3 (lag)** — add `LAG(china_share)` window function, use only lagged in any interaction.
5. **Fix C4 (asymmetric edges)** — rebuild denominator with symmetric edge counting.
6. **Fix C5 (MOST_RECENT look-ahead)** — replace with time-versioned join.
7. **Fix C6 (selection-on-outcome universe)** — rebuild from `Factset_Security_coverage`, Cartesian grid + zero-fill.
8. **Refit AR(1) on monthly GPR** and persist coefficients + shock series to disk.
9. **Replace `MAX × MAX` market cap** — most-recent rule, with dispersion diagnostic.
10. **Make TEST_MODE env-driven** and force suffixed filenames.
11. **Add atomic-write pattern** (`.tmp` + mv) to every `COPY ... TO` in 02/03/04/05.
12. **Run weekend-EOM audit** on all 8 holdings chunks.
13. **Build LionShares parent-institution mapping** and reclassify US vs NONUS by ultimate parent.
14. **Profile the 2,660 unmatched EU sec_entity_ids** and reconcile against the briefing's 92.5% claim.
15. **Replace hard ±1% trim** with winsorization; for the headline regression do not trim.
16. **Commit Project.toml/Manifest.toml**; move paths into a config layer; add `README.md` and `run_all.jl`.
17. **Schedule a second adversarial review** after fixes land — at minimum on the corrected `w_{i,c,t}`, time-versioned Revere identifiers, rebuilt EU-universe panel, and monthly AR(1) shock series.

---

## Implication for the current paper draft

Per memory `feedback_update_manual.md`: when fixes land, `DATA_CONSTRUCTION_MANUAL.md` must be updated alongside.

**All current numbers in the deck and `geoecon.tex` should be treated as PROVISIONAL** until re-derived from the corrected panel — particularly:

- The 10 bp US–non-US gap on HIGH-exposure firms (Panel B′) — uses inconsistent denominators across US and NONUS series.
- The corr = −0.155 / slope = −0.015 scatter (Panel C) — uses contemporaneous (not lagged) ChinaExposure and the global-denominator weight.
- The summary stats Table 1 — all stats stored in the panel are affected by C1 (weight) and C6 (selection-on-outcome cell set).
- The 92.5% match-rate claim — actual cascade output is 79.1%.

**The honest framing for next time you meet the advisor:** "The descriptive pipeline is working but I need 1-2 weeks to fix 6 identification-relevant construction bugs before the headline regression is meaningful. Multi-agent audit found them; here's the action list."
