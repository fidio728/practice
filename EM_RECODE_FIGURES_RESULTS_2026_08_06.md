# EM snapshot rule + zero/missing recode + M1/M2/M3 + Figures A/B — results ledger (2026-08-06)

Workflow: wf_cfa5915e-9d8 (8 agents, all done, verify = ALL SIX ITEMS CONFIRMED, 0 discrepancies).
Full JSON: session tasks/wdpu9bohl.output. Per-agent returns: subagents/workflows/wf_cfa5915e-9d8/journal.jsonl.
Vintage manifest: julia_descriptive/VINTAGE_PREEM.md (70 files archived `_preEM`, 7.6 GB; output/ now junctioned to E:\geoecon_output).

## 1. Construction changes (both advisor-directed, 2026-08-04 meeting + Emanuele email)

**e1 — holdings snapshot (03_eom_etl.jl).** Replaces W=10 as-of window with Emanuele's rule: last
observation within [quarter_start, quarter_end], stamped to quarter-end. `report_date_actual`,
`asof_gap_days` provenance kept. Old W=10 available via DPN_ASOF_WINDOW_DAYS.

**e2 — zero/missing recode (02/06).** Emanuele ruling: in-Revere-but-no-active-links → CN=0;
competitor/partner-only → 0; absent-from-Revere → NULL. Presence rule S1 = record-validity interval
(start_ ≤ q ≤ end_), env-selectable DPN_REVERE_PRESENCE_RULE ∈ {record_interval, record_interval_covered, first_link}.

## 2. Rebuild gates (chain 7/7 exit 0)

| Gate | Result |
|---|---|
| A dup keys / excess rows | 0 / 0; all 186,800,295 old exact-EOM rows recovered |
| B gap distribution | row-wt 70.9% at gap=0, median 0d, max 91d; **MV-wt 84.7% at gap=0**, 12.3% >14d |
| C old-rule weekend damage | weekday capture 81.8% vs weekend 59.5% (22.3pp) |
| D US-share tilt | **−3.77pp → −0.36pp** (P0 asymmetry eliminated) |
| E shock artifacts | 3/3 sha256-identical to _preEM AND _preP0 (shock untouched) |
| F c6 panel | 348,156 → **904,384 rows**; 6,867 → **10,302 firms** (+3,435) |
| G recode audit | 1,695,092 grid firm-quarters NULL→0 (507,231 competitor/partner-only + 1,187,861 no-active-links) |
| P0 bit-identity | all 883,586 old positive-exposure cells preserved exactly (0 lost, 0 drifted) |

Holdings panel: 263,570,485 rows / 9.6 GB (was 208.4M under W=10, 186.8M exact-EOM).

## 3. Attribution decomposition (headline Δw, 3-pairwise-FE, two-way cluster firm_n rd_m)

| Run | β₃ | SE | p | N | firms |
|---|---|---|---|---|---|
| prior P0 (W=10, old missing) | −5.28e-7 | 1.74e-6 | 0.763 | 347,690 | 6,867 |
| A2: snapshot only, old universe | −4.39e-7 | 1.21e-6 | 0.718 | 347,690 | 6,867 |
| A: snapshot + firm adds, old missing rule | −4.38e-7 | 1.21e-6 | 0.719 | 348,576 | 6,897 |
| **B: snapshot + zero-recode (new canonical)** | **−6.92e-7** | **1.05e-6** | **0.512** | **904,294** | **10,302** |

- it+gt spec on B: −9.84e-7, p=0.399.
- Snapshot contributes −17% |β₃| and −30% SE; zero-recode +58% |β₃| and −13% SE; total SE −40%.
- Identifying variation unchanged: cn_lag>0 rows 48,762; 1,607 identifying firms. Added zeros sharpen
  the control arm only; `recoded_zero_rows_with_nonzero_cn_lag = 0`.
- RI (5,000 perms, seed 20260702): old headline p=0.801 → new **0.598**; cum1 0.937 → 0.872;
  cum4 free-perm 0.064 → **0.105** (residual cum4 tension weakens further).
- Independent re-derivation outside Stata: rel. diff ≤ 3.1e-8 on all three β₃.
- 1 of 10 attribution regressions aborted once (Stata batch); fallback route A2 completed; every
  reported cell reconciled verbatim against located Stata logs.

**Conclusion (scoped): under the current per-security in-quarter snapshot, the EU-book denominator
(portfolio_weight_eu), and the record_interval presence rule, β₃ remains insignificant after both
construction changes, with CRVE SEs ~40% smaller than the prior vintage.** NOT yet canonical/final:
pending (i) fund-snapshot grain decision (per-security last-obs stitches multi-date portfolios — see
03 header lines 103-129 and C2d diagnostic: 26.5% of fund-quarters multi-date, 15.3% of MV from
non-last-report rows; completeness diagnostic running 2026-08-08), (ii) global full-portfolio
denominator promotion, (iii) S_{t-1} primary-spec switch, (iv) multi-match aggregation + presence
sensitivity closure (workflow in flight).

## 4. M1/M2/M3 country-quarter measures + Figures A/B (v1)

- 28 countries × 89 quarters (build_country_measures.py → country CSVs).
- Spot: DE 2022Q4 M1=9.07% M2=6.83% M3=1.71%; GB 2022Q4 M1=2.05% M2=4.35% M3=0.85%; FR 2022Q4 M1=4.14%.
- Concentration (Emanuele's point confirmed): median M1/M3 ≈ 3.9–4.2; top-decile firms hold 76–85% of links.
- Figures (plots/): fig_A_firm_zero_vs_positive{,_fixedcohort}, fig_A_firm_quartile_{ew,mcapw},
  fig_A_country_m1_quartiles, fig_B_link_growth_us{2,3} — 7 PDF + 7 PNG + 12 CSVs. t-1 classification,
  CPI-deflated 2020 base, tension background = USA|China (quarter-end-month value).
- Descriptive direction: US book share in positive-tie firms 54.0%→58.2% (2018Q4→2022Q4), zero-tie
  34.1%→27.9%; quartile idx (2022Q4, 100=base): Q4-high 173.6 vs Q1-low 60.2. Fixed-cohort version much
  flatter (47.4%→44.4% vs 40.6%→40.4%) → composition-driven. Fig B: any-US firms add China links faster
  (YoY 17–23%) than no-US firms (2–8%) — no visible pre-emptive de-linking.

## 5. Verify caveats for the written update (wording, not bugs)

1. `revere_coverage_start` is EU-restricted (record-valid AND EU-home-region). The 2015 histogram spike
   mixes genuine Revere additions with home-region reclassification — do NOT describe it as "Revere
   coverage begins".
2. Zero arm = "no ACTIVE EU-classified link", not "no relationship in the raw file" (expired/US-side
   records can exist).
3. Figure background series = quarter-END-MONTH USA|China value, not quarterly average (inherited,
   bit-identical to prior vintages; direction self-documented in CSV).

## 6. Status / pending before "final"

- ~~Figures are v1, NOT final~~ **DONE 2026-08-08 — see §7: MM-FIX v2 applied, chain rerun, verified.**
- After figures: denominator switch (global = main per research_plan.tex, EU = "within-Europe
  reallocation" diagnostic), S_{t-1} promotion to primary, zero_recode_flag column, full battery rerun.
- Multimatch/presence verification workflow (wf_0973468e-885) results: 249/10,212 multi-candidate
  confirmed; 216 ambiguous at winning priority, all CUSIP ties, SEDOL never wins; "descriptive vs
  regression divergence" claim misattributed (06 and build_desc_trend identical rn=1) — real second
  definition lives only in 05 (documented DESCRIPTIVE ONLY); presence: 0/11.1M containment violations,
  88.5% first-link=start_, SCD-2 signature 96.34%, batch pileups (2018-02-05: 18,480 firms).

## 7. MM-FIX v2 + presence sensitivity (2026-08-08, wf_83d53a2f-b04; verify = 6/6 CONFIRMED, 0 discrepancies)

**Multi-match aggregation** (06_cartesian_grid.jl, 06_russia_grid.jl, build_desc_trend_china_links.py,
presence_sens_buckets.py): candidates tied at winning priority (current inputs: **217 entities / 448
pairs, all CUSIP**; crosswalk now 10,575 pairs / 10,344 entities) are SUM-aggregated (numerators and
denominators separately, ratio from sums; presence = union; coverage_start = MIN; recode flags off the
summed counts). Single-winner branch verbatim-identical to HEAD.

| Gate | Result |
|---|---|
| B off-tied bit-identity | grid 2,577,200 cells + c6 7,997,166 cells compared, **0 mismatches** |
| C tied-set recovery | covered fq 8,311 → 12,301 (**+3,990, 0 lost**; grid/c6 window ≤2023Q4: +3,107); NULL→zero 2,659, NULL→pos 448, zero→pos 20 (4 entities), pos→zero 0 |
| D figure deltas | zero-arm share_of_book 2018Q4 .3408→**.3727**, 2020Q1 .3589→**.3916**, 2022Q4 .2791→**.2976**; pos arm 2022Q4 .5817→**.6116** (mass moved from NULL bucket — v1 share panels were materially understated) |
| E headline drift | 3pw −6.92e-7 p=.512 → **−6.74e-7 p=.543**; itgt p=.399→.424; c6 904,384→910,358 rows, firms 10,302 unchanged. Null unchanged |

Verification: crosswalk re-derived with independent SQL (0 set diff); 3 tied entities × 2 quarters
hand-computed from raw Revere CSVs to full precision; all 18,228 tied grid cells recomputed (0 mismatch);
β₃ reproduced outside Stata (rel diff 4.9e-8); falsification — old min-ID path reproduces old artifacts
exactly, lost fq reappear ONLY via aggregation.

**Presence sensitivity (refreshed post-MUST-FIX numbers — use these, not p2's report text):** positive
arm invariant under first_link (24,817 identical fq); zero→NULL flips 168,702 fq / 7,763 firms; zero-arm
book-share moves >1pp ONLY 2007Q3–2011Q4, peak −7.67pp (first_link) / −6.71pp (covered) at 2011Q1;
**post-2012 max 0.58pp; all spot quarters stable**. Verdict: figures' 2018+ narrative rule-insensitive;
caption caveat for pre-2012 zero arm; WRDS confirmation blocking only for early-sample zero-arm claims.

**Stale-artifact rotation (2026-08-08):** headline_3pairwise_canonical.csv and audit_c6_panel.{dta,parquet}
still carried the premm vintage (chain didn't include build_audit_panel_f1f2f7.py / run_headline_3pairwise.do)
→ rotated to *_premm so downstream .do files (run_ddd_nofe_bil, run_fourgroup, run_direction_split,
run_tercile_3pairwise) fail loudly instead of reading stale numbers. Regenerate in the v3 bundled rebuild;
verify_attribution_em.py hardcoded Stata targets also premm-vintage — update there. New-vintage headline
lives in gate_e_headline_mmfix.csv.

**Figures v2 status: mm-fix + presence closed (constraint C4 satisfied at current construction). Still
gated on the v3 bundled rebuild (snapshot grain + global denominator + S_{t-1}) before advisor-final.**
