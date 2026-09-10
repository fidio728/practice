# Regression Spec & Robustness Checks — Tier 2 EOM Panel

**Project:** Corporate robots.txt × AI search visibility
**Treatments:** `root_block`, `blocked_any`, `specific_rule`, group-level
blocks (Tier 0 / 1); `search_sitemap_block` (Tier 2)
**Panel:** Monthly, 2021-01 → 2026-03
**Author:** Xueyang Liu
**Last updated:** 2026-04-28

This document defines the main regression for the Tier 0 / 1 / 2 panels
built by the EOM-primary pipeline (Step 2 v3 → 3 → 4 → 6 → 6.5 → 7), and
the precise set of robustness checks that must accompany it for a
high-tier journal submission.

One-line statement:

> The main panel uses the **EOM latest-interpretable primary** spec.
> Robustness checks are **`r2_latest_200_only`**, **`r4_drop_uninterpretable`**,
> and **`r5_strict_prior_revisit`**. **`r1_latest_any`** and **r6** are
> construction diagnostics, not main research specs.

---

## 1. Main spec — EOM-primary latest-interpretable robots policy

### 1.1 Selection rule (Step 2 primary)

For each `(host, year_month)`, walk candidate Wayback snapshots from the
latest in-month timestamp downward; accept the first interpretable
candidate:

```text
200                 -> parse robots.txt
"-" revisit         -> digest already in cache => parse same-digest body
404 / 410           -> allow-all sentinel ("__ALLOW_ALL_404__")
3xx / 403 / 5xx     -> fall through to next-latest candidate
exhausted           -> tier_status = uninterpretable (no carry-forward)
no in-month snapshot -> tier_status = gap
```

### 1.2 File-of-record per stage (EOM-primary main pipeline)

| stage | output | role |
|---|---|---|
| Step 2 | `monthly_snapshots.csv`, `monthly_snapshots_new.csv` | one row per `(host, ym)` with `tier_status`, `digest`, `snapshot_timestamp`, revisit provenance, `r4_keep` |
| Step 3 | `robots_content.json`, `robots_content_new.json` | `digest -> body`. Sentinel `__ALLOW_ALL_404__ -> ""` seeded once. |
| Step 4 | `robots_panel_monthly.csv` | bot-level Tier 0 / 1 panel; `fetch_status` carries Step 2 tier label |
| Step 6 | `sitemap_panel_results_eom_primary.jsonl` | sitemap-URL declarations, 4-tuple-keyed |
| Step 6.5 | `sitemap_url_paths_v2_eom_primary.jsonl` | per-`(nk, ym, dg, ts)` sitemap paths |
| Step 7 | `search_sitemap_block_panel.csv` | Tier 2 panel; 4-tuple join; `ok_404_allow_all` branch |

### 1.3 Tier 0 / 1 main spec

- Source: `robots_panel_monthly.csv`.
- Main treatment / treatment-state variables (per row = host × month × bot):
  - `root_block`
  - `blocked_any`
  - `specific_rule`
  - Group-level aggregates: search / training / assistant / control any-block
- Sample handling:
  - `ok_404_allow_all`: real allow-all; coded `block = 0` (every bot allowed).
  - `gap`, `uninterpretable`: missing; **NOT zero-imputed**.
- Estimator: TWFE with firm + year-month FEs, firm-clustered SE; staggered
  DiD via Callaway-Sant'Anna or Sun-Abraham for first-treated event-time
  curves.

### 1.4 Tier 2 main spec

- Source: `search_sitemap_block_panel.csv` (Step 7 output).
- Main treatment / outcome variables:
  - `search_sitemap_block` (the 0/1 main treatment)
  - Per-bot / per-group sitemap block counts and fractions
- Sample handling:
  - Only `tier2_status == "ok"` cells are used as **sitemap-path-observed**
    sample. These are the only cells whose 0 vs 1 carries information
    about path-level blocking decisions.
  - `tier2_status == "ok_404_allow_all"`: report **separately** as
    definitionally untreated. Do NOT mix into the `ok` denominator in
    summary tables. In regression, present results both with and without
    these cells.
  - `tier2_status ∈ {no_sitemap_paths, no_robots, uninterpretable}`:
    missing; **NOT zero-imputed**.
- Estimator: same as Tier 0 / 1 (TWFE staggered DiD).

### 1.5 Estimating equation (all tiers)

```
y_{i,t} = β · D_{i,t} + α_i + δ_t + γ' X_{i,t} + ε_{i,t}
```

- `D_{i,t}`: tier-specific treatment (e.g., `search_sitemap_block` for Tier 2)
- `α_i`: firm fixed effects
- `δ_t`: year-month fixed effects
- `X_{i,t}`: lagged Compustat **quarterly** controls aligned via `rdq`
  (per supervisor feedback 2026-04-28). Variables: `log(AT)`, `log(SALE)`,
  leverage = `LT/AT`, market-to-book, ROA. Lag ≥ 1 quarter; carry forward
  between releases.
- Cluster SE at firm level.

### 1.6 Sample restrictions (CRSP)

Per supervisor feedback:
- `EXCHD ∈ {1, 2, 3}` (NYSE / AMEX / NASDAQ)
- `SHRCD ∈ {10, 11}` (common stock)
- `L.PRC ≥ 1`
- Drop financials (SIC 6000-6999) in the main spec; keep in robustness.

---

## 2. How to write up the EOM construction

In the main paper:

> We define the monthly robots policy as the latest interpretable archived
> robots.txt state within each host-month. This end-of-month construction
> aligns treatment status with monthly outcomes while avoiding a naive
> latest-capture rule that would classify many revisit or transient-error
> captures as missing.

Then, in the data section / Table 1, report sanity statistics from Step 2
v3 (primary cohort, 2026-04-28 run):

| tier | share |
|---|---|
| `ok_200` | 25.8% |
| `ok_revisit_resolved` | 29.9% |
| `ok_404_allow_all` | 5.1% |
| `uninterpretable` | 3.6% |
| `gap` | 35.5% |

And contrast against the naive latest-any rule (`r1_latest_any`):

> A naive latest-capture rule would classify 41.5% of primary-cohort
> host-months as uninterpretable, mostly due to revisit and transient
> archival statuses. We therefore use the latest *interpretable* state as
> the primary construction.

This makes the EOM rule defensible: it is NOT an arbitrary patch — it is
specifically designed to avoid Wayback CDX status-code mechanics that
would otherwise generate a large fraction of false missingness.

---

## 3. Robustness checks (precise set)

Four specs are produced by Step 2 v3; three are research-grade
robustness, one is diagnostic.

### 3.1 `r2_latest_200_only` — direct-200-only robustness

- Source: `monthly_snapshots_r2_latest_200_only.csv` (and `_new`).
- Selection: walk from latest, accept only `status == 200`. Revisits,
  404/410, 3xx, 5xx all fall through.
- Purpose: show the result is **not** driven by revisit resolution or
  404-allow-all coding.
- Write-up:
  > We re-estimate the main specifications using only host-months whose
  > selected state is a direct 200 capture, excluding revisit resolution
  > and 404 allow-all recovery.

### 3.2 `r4_drop_uninterpretable` — sample-restriction robustness

- Source: `monthly_snapshots_r4_drop_uninterpretable.csv` (and `_new`).
- Selection: identical to primary, but every row carries an `r4_keep`
  flag. `r4_keep == 0` iff `tier_status == "uninterpretable"`. Panel stays
  rectangular at the data layer; sample restriction is applied at the
  regression stage.
- Purpose: confirm the result is not driven by inclusion of cells we
  could not interpret.
- Write-up:
  > We repeat the analysis after excluding host-months for which no
  > interpretable archived robots policy can be recovered.

### 3.3 `r5_strict_prior_revisit` — strict-prior revisit robustness

- Source: `monthly_snapshots_r5_strict_prior_revisit.csv` (and `_new`).
- Selection: like primary, but a `"-"` revisit is accepted only when
  BOTH conditions hold:
  1. A same-host 200 capture with the same digest exists at or before
     the revisit timestamp.
  2. The digest is in the content cache.
- Purpose: respond pre-emptively to the reviewer concern that primary's
  revisit resolution might use future information. r5 completely
  eliminates forward resolution.
- Sanity (already verified in 2026-04-28 run):
  - `r5` forward-resolved revisit count = **0** (by construction).
- Write-up:
  > We require revisit captures to be resolved only when a same-host
  > prior 200 capture with the same digest exists at or before the
  > revisit timestamp.

---

## 4. Construction diagnostics (NOT robustness)

These specs / numbers exist to validate the construction. They are NOT
research robustness checks and should not be presented as substitutes for
the primary spec.

### 4.1 `r1_latest_any` — naive latest-capture diagnostic

- Source: `monthly_snapshots_r1_latest_any.csv`.
- Selection: strict max-timestamp CDX capture, no resolution, no
  fall-through.
- Role: diagnostic only. Goes in the data-construction appendix, not the
  main result table.
- Write-up:
  > A naive latest-capture rule would classify 41.5% of primary-cohort
  > host-months as uninterpretable, mostly due to revisit and transient
  > archival statuses. We therefore use latest interpretable state as the
  > primary construction.

### 4.2 r6 backfill — diagnostic-only, NOT implemented

- We do NOT implement r6 (revisit-with-Step-3-backfill).
- Diagnostic numbers from primary cohort (2026-04-28 run):
  - True coverage gain (cells where r6 would rescue from `uninterpretable`):
    1,721 cells = **0.6%** of all primary cells.
  - Treatment-change risk (cells where r6 would change the chosen tier
    from primary's pick): 10,551 cells.
- Write-up:
  > We do not implement backfill-based revisit recovery because it would
  > change treatment assignment in substantially more cells than it would
  > rescue from missingness.

---

## 5. Other regression-stage robustness (independent of Step 2 spec)

### 5.1 `r3_lag1m` — lagged treatment

- NOT a Step 2 selection-stage change (deliberately moved out of Step 2
  per 2026-04-28 review).
- Implementation: at regression construction, `gen L1_D = L.D` and rerun
  the main regression with `L1_D` in place of `D`.
- Justification: outcomes (CRSP returns, traffic) realize within the
  calendar month; treatment state should pre-date them. Granger-style
  pre-test for robots → outcome direction.

### 5.2 Day-weighted treatment exposure

- For each `(firm, month)`, compute fraction of days the firm was in
  `D = 1` state, weighted across in-month snapshots.
- Treats the EOM rule as a robustness anchor rather than the only
  definition.

### 5.3 Annual vs quarterly Compustat controls

- Main: quarterly + `rdq` lag.
- Robustness: annual + 1-year lag. Show coefficient stability.

### 5.4 Financial-firm inclusion

- Main: drop SIC 6000-6999.
- Robustness: include with a financial-firm-×-treatment interaction.

### 5.5 CRSP filter relaxation

- Robustness: `L.PRC ≥ 0.5`; `SHRCD ⊇ {10,11,12}`.

### 5.6 Pre-trend / parallel-trends tests

- Event study around firm's first-treated month.
- Goodman-Bacon decomposition (flag forbidden 2x2 comparisons under TWFE
  with staggered adoption).
- Sun-Abraham or Callaway-Sant'Anna estimator if pre-trends are non-flat.

### 5.7 Placebo tests

- Random-treatment placebo: shuffle `D` within firm, holding marginal
  frequencies constant; expect null β.
- Pre-period placebo: restrict sample to 2021-01 → 2023-07 (pre-ChatGPT
  Search public availability ~2024-Q4); expect β ≈ 0 since the channel
  did not exist yet.

### 5.8 Heterogeneity

- By firm size (Compustat AT terciles).
- By industry (Fama-French 12).
- By traditional-search baseline (firms already blocking Googlebot vs not).
- By any-AI-bot-blocked baseline (Tier 0 / 1).

### 5.9 Outcome robustness

- Traffic outcomes: IHS vs `log(1 + y)`.
- Returns: monthly raw vs CAPM- or FF3-residualized.

### 5.10 Majority-vote within-month — REJECTED

- Not used. Per supervisor 2026-04-28, this is a snapshot-frequency
  artifact (a host snapped 30 times in a month would dominate over a
  host snapped twice). Mention only as something we considered and
  rejected.

---

## 6. What NOT to do (record of decisions)

- **Do NOT** use the legacy day-15 selection as the main spec or as a
  primary robustness anchor. Day-15 picks the calendar middle of the
  month, not the realized end-of-month state, and was retired
  2026-04-28 per supervisor feedback. The legacy file
  (`monthly_snapshots_day15_backup.csv`, v0 schema) is kept only for
  diagnostic / appendix purposes; if you ever need a day-15 vs EOM audit,
  re-emit day-15 under v3 schema first (otherwise the audit script's
  `assert_v3_schema()` hard-fails).
- **Do NOT** merge the legacy cloud shards into the EOM-primary v2 file.
  The shards (`sitemap_url_paths_v2_legacy_shards.jsonl`, 535 MB) are
  diagnostic only; their day-15 timestamps cannot match the EOM 4-tuple
  key. EOM-primary v2 is rebuilt from scratch by Step 6.5.
- **Do NOT** treat `r1_latest_any` as a substitute research spec — it is
  a construction diagnostic, never a regression spec.
- **Do NOT** code `ok_404_allow_all` as missing. It is a real allow-all
  policy state and contributes a 0 to every block variable.
- **Do NOT** zero-impute `gap`, `uninterpretable`, `no_sitemap_paths`, or
  `no_robots` into the regression. They are missing.
- **Do NOT** add `r6_revisit_with_backfill` to the main pipeline. The
  treatment-change risk (10,551 cells) outweighs the coverage gain
  (1,721 cells, 0.6%).
- **Do NOT** include 3xx redirect-following resolution as a main-spec
  robustness. The redirect target's archived headers are not always
  retrievable, and we don't want to invent treatment from un-retrievable
  data.

---

## 7. Sample-construction transparency table

To accompany the main regression table, produce the following counts walk:

| step | n_hosts | n_firm-months | source |
|---|---|---|---|
| Seed firm list (CRSP common ∩ Compustat) | TBD | — | `bot_paper/build_analysis_sample.py` |
| CDX hosts loaded (`cdx_snapshots*.jsonl`) | 4,330 + 5,053 = 9,383 | — | Step 1 |
| Step 2 rectangular hosts × months (primary) | 9,383 | 591,129 | `monthly_snapshots*.csv` |
| Of which `tier_status ∈ {ok_200, ok_revisit_resolved, ok_404_allow_all}` | TBD | 321,612 | Step 2 summary |
| Step 4 parsed (`fetch_status ∈ {ok_200, ok_revisit_resolved, ok_404_allow_all}`) | TBD | TBD | `robots_panel_monthly.csv` |
| Sitemap declared (Step 6 v1 reaches host) | TBD | TBD | `sitemap_panel_results_eom_primary.jsonl` |
| Sitemap paths fetched clean (Step 6.5 `final_clean`) | TBD | TBD | `sitemap_url_paths_v2_eom_primary.jsonl` |
| Tier 2 `tier2_status == "ok"` | TBD | TBD | `search_sitemap_block_panel.csv` |
| ∩ firm panel (gvkey ↔ host ↔ year_month) | TBD | TBD | `firm_panel_monthly.csv` |
| ∩ CRSP filters | TBD | TBD | regression-stage |
| ∩ Drop financials | TBD | TBD | regression-stage |
| **Final regression sample** | TBD | TBD | regression input |

Every drop step's count must reconcile.

### 7.1 Cohort-coverage caveat (must be reported)

Handoff doc-of-record: 4,427 / 6,669 hosts.
Actual CDX-loaded as of 2026-04-28: 4,330 / 5,053.
- Primary cohort: −97 (~2.2%) — small attrition.
- New cohort: −1,616 (~24.2%) — substantial; investigate before
  submission. Likely cause: hosts with no Wayback snapshots in the
  2021–2026 window, dropped during CDX query. Document in the data
  appendix.

---

## 8. File register (for reproducibility)

| file | role |
|---|---|
| `02_select_monthly.py` | EOM selection (primary + r1/r2/r4/r5) |
| `monthly_snapshots.csv`, `_new` | primary spec, both cohorts |
| `monthly_snapshots_{r1,r2,r4,r5}*.csv` | robustness specs |
| `monthly_snapshots_day15_backup.csv` | legacy day-15 selection (v0 schema; diagnostic only) |
| `03_download_content.py` | tier-status-aware digest fetch; sentinel seed |
| `04_parse_panel.py` | bot-level panel; `fetch_status` = Step 2 tier label |
| `06_collect_sitemaps_for_panel.py` | sitemap declarations under 4-tuple key |
| `06_5_collect_sitemap_paths_monthly.py` | per-`(nk,ym,dg,ts)` sitemap paths |
| `07_build_search_sitemap_block_panel_v3.py` | Tier 2 panel; ok_404_allow_all branch |
| `merge_cloud_shards_into_v2.py` | folds 5 cloud shards into legacy diagnostic file |
| `audit_eom_gap.py` | day-15 vs EOM differ-rate audit (deferred; needs v3-schema baseline) |
| `REGRESSION_SPEC_AND_ROBUSTNESS.md` | this document |
