# P0 REBUILD RESULTS — holdings snapshot as-of rule (2026-08-04)

## What P0 was
03_eom_etl.jl kept only rows with REPORT_DATE == exact calendar quarter-end.
FactSet report dates shift to the prior business day when quarter-end falls on
a weekend, so weekend quarter-ends silently dropped ~40% of funds (2022-12-31
Sat: 58.4% coverage vs 86-87% weekday; 2023-09-30 Sat: 38.6%). The loss was
US/NONUS-asymmetric (dropped Fri batch 26% US vs kept Sat batch 21%), which
group x quarter FE cannot absorb -> bias channel into beta3, plus a fake
"deepest drawdown ever" in the advisor figure.

## The fix (only-change-one-thing; W curve + gates all archived in output/diag_p0_*)
Per (fund_id, fsym_id): latest report <= quarter-end within W=10 days (plateau
W=3..14 measured on full 1999-2023; W=31 imports 28.4% stale rows). report_date
stamped to quarter-end; report_date_actual + asof_gap_days added. New panel
208,418,523 rows (+11.57%); old panel = exact gap=0 subset (186,800,295 rows,
bit-equal). 56 artifacts archived as *_preP0; 16 scripts carry vintage banners.
Infra: 00_setup.jl dbcon now sets max_temp_directory_size=300GB (old 4.3GB
default cap deadlocked 04 on the bigger panel).

## Gates (independently re-derived by adversarial verifier, different algorithm,
## exact to the last dollar)
- Coverage gap weekday-vs-weekend: 22.3->4.2pp full (dragged by ramp-up era
  1999-2005 at 11.3pp, a data property no W fixes - DISCLOSED; candidate
  robustness: sample start 2006); mid 17.7->2.9; modern 34.0->1.6; trend-free
  artifact 11.1->1.1pp.
- US-share selection tilt (the bias channel): -3.41 -> +0.01pp, flat all eras.
- Dollars: 2022Q4 $1.457T->$1.892T; 2021Q4 unchanged $2.40T; nominal YoY
  2022Q4 -39.2% -> -21.2%.
- Shock series: bit-identical to preP0 (265 quarters, max|d|=0.0 on all cols).
- Grid: exposure/shock cols bit-frozen on all 2,548,600 common cells; w moved
  on 18.9%; +48 new firms (204 exposed cells) legitimately enter.
- c6: 348,156 rows / 6,867 firms (was 347,952 / 6,854; +204 rows == new firms'
  exposed cells exactly).

## RESULTS ON THE P0 PANEL (identical shock)
Figure 2 (advisor): 2022Q4 real YoY -42.8% -> -25.9%; 2009Q1 -39.7% BIT-UNCHANGED
(clean-Tuesday placebo); deepest-drawdown ranking REVERSES back to 2009Q1;
2023Q3 -1.0% -> +29.1%. Pattern-3 text sent to advisor needs correction.
Figure 1 (china links): max |d fraction| = 0.22pp over 7,568 cells - immune by
design (same-source numerator/denominator), unchanged.

Headline (run_headline_3pairwise.do + run_ri_3pairwise.py):
| spec | pre-P0 b3 / CRVE p / RI p | post-P0 b3 / CRVE p / RI p |
|---|---|---|
| 3pw headline | +2.746e-6 / 0.110 / 0.308 | -5.28e-7 / 0.763 / 0.801 |
| it+gt | +2.081e-6 / 0.151 / 0.412 | -1.07e-6 / 0.557 / - |
| LP cum1 (3pw) | +4.094e-6 / 0.009 / 0.105 | -1.53e-7 / - / 0.937 |
| LP cum4 (3pw) | +8.826e-6 / 0.053 / 0.039(free) | +6.60e-6 / - / 0.064(free) |
| lead t+1 (3pw) | +1.01e-6 / 0.576 | +2.09e-7 / 0.868 |

READING (three layers):
1. The null SURVIVES and DEEPENS (p 0.11 -> 0.76). Core conclusion unchanged.
2. The uniform-positive-sign pattern DIES: the point estimate flips to tiny
   negative. The old positive lean was substantially the weekend-composition
   artifact (mechanism: weekend quarters under-counted US holdings -> rebound
   quarters read as US buying -> positive tilt).
3. The last free-perm rejection (cum4 0.039) dissolves (0.064) - now dead by
   BOTH the A0 serial-robust inference fix AND the P0 data fix, independently.

## Doc/debt created by P0 (pending)
- EVERYTHING downstream is PRE-P0 vintage pending re-run: fourgroup, direction,
  tercile, sagg, shocklag, riskset, flow, extmargin, Russia, cum4-hardening,
  ddd gate, 07x battery. Doc atomic refresh needed after re-runs.
- Advisor correction: figure 2 + pattern-3 rewrite (draft ready, user to send).
- gpr_ar1_coefficients.csv persists a=0,b=0 (old Julia let-scope bug; shock
  itself unaffected) - fix when touched.
- 04's I_ict now mixes report-date valuations within a quarter cell for
  shifted funds (~1-day price drift on 7.8% of rows) - disclosed, immaterial
  vs the 40% coverage hole it fixes.
