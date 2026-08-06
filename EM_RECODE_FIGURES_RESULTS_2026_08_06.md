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

**Conclusion: headline null is robust to both advisor-directed construction changes and gains ~40% precision.**

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

- **Figures are v1, NOT final** (user constraint C4): pending (i) multi-match fix — aggregate across
  tied-at-winning-priority candidates (216 CUSIP-tie entities; min-ID pick loses 3,135 firm-quarters)
  harmonized in 06 + build_desc_trend_china_links.py, then rerun 06→c6→figure data = v2 final;
  (ii) presence sensitivity: figure-A zero arm under first_link vs record_interval.
- After figures: denominator switch (global = main per research_plan.tex, EU = "within-Europe
  reallocation" diagnostic), S_{t-1} promotion to primary, zero_recode_flag column, full battery rerun.
- Multimatch/presence verification workflow (wf_0973468e-885) results: 249/10,212 multi-candidate
  confirmed; 216 ambiguous at winning priority, all CUSIP ties, SEDOL never wins; "descriptive vs
  regression divergence" claim misattributed (06 and build_desc_trend identical rn=1) — real second
  definition lives only in 05 (documented DESCRIPTIVE ONLY); presence: 0/11.1M containment violations,
  88.5% first-link=start_, SCD-2 signature 96.34%, batch pileups (2018-02-05: 18,480 firms).
