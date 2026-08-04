# SHOCK MENU RESULTS — P1/P1b shock-construction robustness (2026-08-04, post-P0)

Decision memo. Self-contained. Written after the P0 holdings rebuild, on the P0
vintage `c6_panel.dta` (348,156 rows, 6,867 firms, 82 quarters 2003Q3–2023Q4).

**Bottom line.** The P0 deep null survives the entire shock menu under the
design-based arbiter. Zero of 12 randomization-inference columns rejects at
p<.05 under either free permutation or circular shift. Two CRVE columns cross
.05, they correlate 0.892 with each other, and both dissolve under RI. The
direction axis (bidirectional vs USA→China) changes nothing. P1a, P1b and P1c
move from *open threats* to *disclosed and closed*.

---

## 1. Why the menu exists

The headline shock S_t is the monthly level-AR(1) residual of the
Iacoviello–Tong bilateral AI-GPR series `USA|China`, sampled at the quarter-end
month (built in `05_combine_visualize.jl`). Three documented defects:

| id | defect | status after this work package |
|---|---|---|
| **P1a** | quarterly S_t is serially correlated (acf1 = +0.271, LB(1) p = 0.013) because a monthly level-AR(1) under-cleans GPR persistence | re-derived (0.2707 / 0.0125) and **fixed in the preferred variant** (acf1 = −0.129, LB(4) p = 0.354) |
| **P1b** | `USA|China` is directional (US initiator → China respondent); the official data also has `China|USA`; level corr between them only ~0.576 | re-derived (0.5632 full sample, 0.5755 on months ≤ 2023-12) and **tested as a parallel column set** (4 direction columns) |
| **P1c** | the AR is fit on the FULL sample incl. 2024–2026, i.e. look-ahead in a generated regressor (doc F7 disclosure) | **removed** in every B/C/D/E variant (fit ≤ 2023-12) |

Pre-menu headline on the P0 panel: b3 = −5.28e-7, CRVE p = 0.763, RI p = 0.801.

**Nothing existing was modified.** `gpr_quarterly_with_shock.parquet` (frozen,
265 quarters) and `c6_panel.dta` were read-only inputs; both verified
byte-unchanged after the run (parquet sha256 matches its own build-time
`.meta.json` and the archived `*_preP0` twin). All output is in new files.

---

## 2. The menu, and the whiteness diagnostics it is judged on

All variants are built at monthly frequency (except D, which aggregates first),
brought to the 82-quarter panel grid, then standardized to mean 0 / sd 1 over
those 82 quarters so β₃ magnitudes are comparable. Pre-standardization sd is
reported below. Standardizing is affine, so it does not touch acf or Ljung-Box.

Source: `julia_descriptive/output/shock_menu_diagnostics.csv`.

| variant | direction | spec | agg | no-look-ahead | sd_raw | acf1 | LB(4) p | LB(8) p | corr w/ baseline | rank |
|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_existing` | USA\|China | level AR(1), FULL-sample | qtr-end month | no | 2.578 | **+0.2707** | **0.0228** | **0.0040** | 1.000 | — (headline-continuity) |
| `A_baseline_repro` | USA\|China | level AR(1), FULL-sample | qtr-end month | no | 2.578 | +0.2707 | 0.0228 | 0.0040 | 1.000 | — (**gate**) |
| `B_i_lvl_ar1_nla_qend` | USA\|China | level AR(1), fit ≤ 2023-12 | qtr-end month | yes | 2.590 | +0.2676 | 0.0260 | 0.0028 | 0.9996 | 4 |
| `B_ii_lvl_ar1_nla_q3sum` | USA\|China | level AR(1), fit ≤ 2023-12 | 3-month sum | yes | 3.970 | +0.3128 | 3.3e-12 | 2.2e-18 | 0.436 | 5 |
| `C_i_dgpr_ar4_nla_qend` | USA\|China | AR(4) on Δmonthly GPR, fit ≤ 2023-12 | qtr-end month | yes | 2.413 | −0.0309 | 0.2534 | 0.0792 | 0.880 | 3 |
| `C_ii_dgpr_ar4_nla_q3sum` | USA\|China | AR(4) on Δmonthly GPR, fit ≤ 2023-12 | 3-month sum | yes | 3.850 | −0.2136 | 0.2958 | **0.1319** | 0.320 | 2 |
| **`D_q_ar1_nla`** | USA\|China | quarterly MEAN then AR(1), fit ≤ 2023Q4 | native quarterly | yes | 1.786 | **−0.1288** | **0.3544** | 0.0848 | 0.254 | **1 (WINNER)** |
| `E_C_i_bidir` | USA\|China + China\|USA (SUM) | AR(4) on Δmonthly GPR | qtr-end month | yes | 2.815 | +0.0317 | 0.1169 | 0.0401 | 0.829 | parallel |
| `E_C_i_cnus` | China\|USA (diagnostic) | AR(4) on Δmonthly GPR | qtr-end month | yes | 0.887 | +0.1342 | 0.1730 | 0.1486 | 0.170 | parallel |
| `E_B_i_bidir` | USA\|China + China\|USA (SUM) | level AR(1), fit ≤ 2023-12 | qtr-end month | yes | 3.032 | +0.2723 | 0.0138 | 0.0032 | 0.939 | parallel |
| `E_B_i_cnus` | China\|USA (diagnostic) | level AR(1), fit ≤ 2023-12 | qtr-end month | yes | 1.003 | +0.2610 | 0.1006 | 0.3845 | 0.266 | parallel |

Reading of the diagnostics. Level-AR(1) does not whiten GPR at quarterly
frequency, no matter whether it is fit with or without look-ahead: B-i is
essentially the baseline (corr 0.9996) and carries the same +0.27 acf1. The
3-month sum of a level-AR(1) residual is the worst column in the menu (LB p ≈ 0
at every lag); summing re-injects the persistence the AR failed to remove.
Differencing first (C) or aggregating first (D) is what actually whitens.

### Validation gate (A)

`A_baseline_repro` must reproduce the existing `shock_us_cn` to
max|diff| ≤ 1e-8. Achieved **1.776e-15** against both the frozen parquet and the
panel column, on all 82 panel quarters and on all 265 artifact quarters. The
tolerance is hard-coded at `build_shock_menu.py:180`, it was not tuned to the
observed value. The builder's plumbing therefore matches `05_combine_visualize.jl`
bit-for-bit, and every other menu column is trustworthy by construction.

### Re-derivation of prior external measurements

`output/shock_menu_rederivation_check.csv`. Everything quoted in the spec was
recomputed from the raw CSV. Six of eight reproduce. Two do not, and the reason
is now known:

| item | prior | re-derived | verdict |
|---|---|---|---|
| baseline acf1 | 0.271 | 0.2707 | reproduces |
| baseline LB(1) p | 0.013 | 0.0125 | reproduces |
| C-i acf1 / LB(4) p | −0.036 / 0.245 | −0.0309 / 0.2534 | reproduces |
| C-ii acf1 / LB(4) p | −0.215 / 0.292 | −0.2136 / 0.2958 | reproduces |
| **D acf1** | −0.029 | **−0.1288** | **DOES NOT REPRODUCE** |
| **D LB(8) p** | 0.017 | **0.0848** | **DOES NOT REPRODUCE** |
| level corr USA\|China vs China\|USA | 0.576 | 0.5632 full / 0.5755 ≤2023-12 | reproduces |

Cause, confirmed by refit: the prior D numbers came from a **full-sample
(look-ahead)** quarterly AR(1). Refitting full-sample here returns −0.0292 and
0.0175, exact matches. Variant D as built is no-look-ahead, so −0.1288 / 0.0848
govern. This mattered: under the prior (wrong-vintage) numbers D would have
looked whiter than it is, and its LB(8) would have looked *worse* than it is.

---

## 3. The pre-registered selection rule, and what it picks

Verbatim from `build_shock_menu.py` header, written before any regression ran:

> "the preferred NEW construction = the no-look-ahead variant with the whitest
> QUARTERLY residual series on the 82 panel quarters, judged by LB(4) p
> (tie-break: LB(8) p, then acf1 magnitude), within the USA|China direction; the
> direction axis (bidirectional vs directional) is reported as a parallel column
> set, not selected on outcomes. beta3 results play NO role in selection.
> Baseline S_t stays the headline-continuity column regardless (its defects are
> disclosed, not hidden)."

Candidate pool = `no_look_ahead == True` AND `direction == USA|China`, which is
exactly {B-i, B-ii, C-i, C-ii, D}. A and `baseline_existing` are excluded by
look-ahead, the four E columns by direction. Sorting on LB(4) p descending gives
no ties (0.3544 > 0.2958 > 0.2534 > 0.0260 > 3.3e-12).

**Winner: `D_q_ar1_nla`** → menu column `s_D_q_ar1_nla`.
Spec: quarterly MEAN of monthly `USA|China` GPR, then AR(1) at quarterly
frequency, fit on quarters ≤ 2023Q4. Residual is native quarterly, no
aggregation step.

Selection provably preceded estimation on disk: `shock_menu_preferred.txt` and
the diagnostics were written 13:13:46, `shockmenu_vce_diag.csv` at 13:14:55 and
the results CSV after that. β₃ appears nowhere in the selection block. The `.do`
file reads the winner from `shock_menu_preferred.txt` rather than a hard-coded
macro, and echoed the cross-check.

**Honest caveat, keep this in the paper.** The winner is rule-dependent. D wins
on LB(4) p, but its LB(8) p (0.0848) is *worse* than the runner-up's
(C-ii, 0.1319), and an |acf1|-first rule would have picked C-i instead. The
pre-registered rule is LB(4)-first and was applied as written, so the ordering
stands. But higher-order dependence in the selected series is **not** fully
cleaned. Say so plainly rather than claiming a white shock.

---

## 4. Full results table

Design: `reghdfe dw us_cn us_cn_shock_v, absorb(fq gq ig) vce(cluster firm_n rd_m)`
(3-pairwise, the headline FE set) and `absorb(fq gq)` (it+gt companion).
`us_cn_shock_v = us × cn_lag × S_v` rebuilt per variant on the P0 c6 panel.
3pw: N = 347,690, 6,634 firm clusters, 82 month clusters, df_r = 81.
it+gt: N = 348,156, 6,867 firm clusters.
RI: mirror of `run_ri_3pairwise.py` (US-minus-NONUS collapse, 30-iter bincount
two-way demean, joint 2×2 solve), N_PERM = 5000, SEED = 20260702, 81 circular
shifts, 174,078 firm-quarters, 0 quarters dropped. `arbiter_p = p_circ`.

Baseline first. β₃ in units of 1e-6. RI p from `ri_shockmenu.csv` (3pw only).

| variant | β₃ 3pw (×1e-6) | se (×1e-6) | CRVE p 3pw | β₃ it+gt (×1e-6) | CRVE p it+gt | RI p_free | RI p_circ (arbiter) | verdict |
|---|---|---|---|---|---|---|---|---|
| `baseline_panel_shock_raw` (un-standardized, = current headline) | −0.528 | 1.743 | 0.763 | −1.073 | 0.557 | 0.815 | 0.829 | **null** |
| `s_baseline_existing` (= `s_A_baseline_repro`) | −1.361 | 4.494 | 0.763 | −2.765 | 0.557 | 0.815 | 0.829 | **null** |
| `s_B_i_lvl_ar1_nla_qend` | −1.227 | 4.468 | 0.784 | −2.635 | 0.575 | 0.835 | 0.866 | **null** |
| `s_B_ii_lvl_ar1_nla_q3sum` | −6.734 | 3.951 | 0.092 | −6.339 | 0.097 | 0.223 | 0.329 | **null** (CRVE marginal) |
| `s_C_i_dgpr_ar4_nla_qend` | −2.916 | 3.783 | 0.443 | −4.004 | 0.349 | 0.609 | 0.573 | **null** |
| `s_C_ii_dgpr_ar4_nla_q3sum` | −6.252 | 2.761 | **0.026** | −7.054 | **0.018** | 0.261 | 0.280 | **null under arbiter** (CRVE rejects) |
| **`s_D_q_ar1_nla`** (PREFERRED) | −7.557 | 3.452 | **0.031** | −8.031 | **0.020** | 0.178 | 0.244 | **null under arbiter** (CRVE rejects) |
| `s_E_C_i_bidir` | −1.530 | 3.636 | 0.675 | −2.751 | 0.491 | 0.782 | 0.756 | **null** |
| `s_E_C_i_cnus` | +3.515 | 2.387 | 0.145 | +2.620 | 0.237 | 0.530 | 0.524 | **null** (sign flips) |
| `s_E_B_i_bidir` | +0.448 | 4.063 | 0.913 | −0.954 | 0.822 | 0.940 | 0.915 | **null** |
| `s_E_B_i_cnus` | +3.975 | 2.741 | 0.151 | +3.424 | 0.176 | 0.462 | 0.451 | **null** (sign flips) |

Lead spec (dv = `dw_lead1`), run for baseline + preferred only, as specified:

| variant | β₃ 3pw (×1e-6) | CRVE p 3pw | β₃ it+gt (×1e-6) | CRVE p it+gt |
|---|---|---|---|---|
| `baseline_panel_shock_raw` | +0.177 | 0.890 | −0.332 | 0.790 |
| `s_D_q_ar1_nla` | +2.994 | 0.363 | +1.931 | 0.499 |

`b2` (the `us_cn` main interaction) is null everywhere, p ∈ [0.68, 0.98] on
3pw dv=dw across all 12 columns. Full values in `shockmenu_results.csv`.

Execution health. 28 result rows, all `status = ok`, zero reghdfe failures.
B9 degenerate-VCE guard: `shockmenu_vce_diag.csv` has **56** data rows
(12 variants × 2 FE × 2 coefs for dv=dw, plus 2 × 2 × 2 for dv=dw_lead1), every
one `se_valid = 1`. No degenerate two-way-cluster VCE. Drift gate against
`headline_3pairwise_canonical.csv` printed PASS on both FE sets (relerr 6.5e-8
and 7.0e-8).

---

## 5. The honest reading

### 5.1 Does the P0 null hold across every shock construction?

Yes, under the arbiter. Per-variant verdicts are in the table above. Summary:

- **Nine of eleven columns are null on CRVE as well as RI.** Baseline, B-i,
  C-i, and all four direction columns sit at CRVE p ∈ [0.15, 0.91] and RI
  p_circ ∈ [0.45, 0.92]. Nothing ambiguous there.
- **Two columns cross CRVE .05**: `s_D_q_ar1_nla` (0.031 / 0.020) and
  `s_C_ii_dgpr_ar4_nla_q3sum` (0.026 / 0.018). Both dissolve under RI
  (p_circ 0.244 and 0.280). This is the CRVE-vs-RI divergence the
  pre-registered reading already anticipated, and `p_circ` governs.
- **The two CRVE crossings are not two independent hits.** They correlate
  **0.892** with each other (`shock_menu_corr.csv`), and both correlate weakly
  with the baseline (0.254 and 0.320). Treat them as roughly one column.
- **The preferred variant's lead spec is null and sign-flipped**
  (+2.994e-6, p = 0.363, vs −7.557e-6 contemporaneous). A real lagged
  disengagement effect would not do that. This is corroborating evidence that
  the CRVE crossing is noise, not signal.
- **B-ii is the mechanical warning.** Its CRVE p (0.092) is the third-lowest in
  the menu and its residual is the least white column in the menu
  (LB(8) p = 2e-18). Serial correlation inflating apparent CRVE significance is
  visible right there in the data.

So the null is not an artifact of shock construction. Where CRVE and the
arbiter disagree, the disagreement is concentrated in the two most-whitened
low-correlation-with-baseline columns and does not survive design-based
inference. The paper reports a clean null across the whole menu.

### 5.2 Does the direction axis change anything?

No. Four direction columns, all null on both CRVE and RI.

- Bidirectional sum with the C-i spec: β₃ = −1.530e-6, CRVE p = 0.675,
  RI p_circ = 0.756. It correlates 0.949 with the directional C-i.
- Bidirectional sum with the B-i spec: β₃ = +0.448e-6, CRVE p = 0.913,
  RI p_circ = 0.915. It correlates 0.939 with the baseline.
- `China|USA` alone flips sign in both specs (+3.515e-6 and +3.975e-6) and is
  null in both (RI p_circ 0.524 and 0.451). The sign flip is not a finding, it
  is what a null column with a different noise realization looks like.

The conceptual point stands (the hypothesis is about US–China tension broadly,
and `USA|China` alone is a directional choice), so P1b was a legitimate threat.
It is now a *tested and closed* threat rather than an open one. The
bidirectional sum is the conceptually-matching construction and it gives the
same answer as the baseline. Note also that the bidirectional sum is dominated
by the US-initiated leg: it correlates 0.94–0.95 with the corresponding
directional column and only 0.52–0.56 with `China|USA` alone.

### 5.3 What the menu does and does not license

Licensed:
- "The null is robust to the shock's AR specification (level AR(1), AR(4) on
  first differences, quarterly-frequency AR(1)), to the aggregation rule
  (quarter-end month vs within-quarter sum), to removing the full-sample
  look-ahead, and to the bilateral direction."
- "The serial correlation in the baseline shock (P1a) is real and measured;
  the preferred construction reduces it (LB(4) p 0.023 → 0.354) and the result
  does not change under the arbiter."

Not licensed:
- Any claim that the preferred variant produces a *significant negative* effect.
  It does on CRVE and it does not under RI, and the arbiter governs. Reporting
  the CRVE p without the RI p next to it would be selective.
- Any claim that the selected shock is white. LB(8) p = 0.085.

---

## 6. What goes into the paper

1. **Baseline stays the headline-continuity column.** All headline tables keep
   the existing `shock_us_cn`. Its three defects are disclosed in §9/F7, not
   hidden, and the menu is the evidence that disclosure is sufficient.
2. **`s_D_q_ar1_nla` becomes the primary shock-construction robustness column.**
   Describe it as: quarterly-mean GPR, AR(1) at quarterly frequency, fit through
   2023Q4 only, no look-ahead. Report β₃, CRVE p **and** RI p_circ in the same
   row. State the caveat from §3 (rule-dependent winner, LB(8) not clean).
3. **The full diagnostics table (§2) goes in the appendix.** It is the direct
   answer to a referee asking "your shock is autocorrelated, does that drive
   the result?" The answer is a measured no, across five constructions.
4. **The direction axis is a two-row appendix block**, not a headline column:
   bidirectional sum and `China|USA` alone, for both C-i and B-i.
5. **The full-battery re-run uses baseline + preferred only.** Two shock
   columns, not eleven. Everything downstream of P0 is still pre-P0 vintage and
   has to be re-run anyway (fourgroup, direction, tercile, sagg, shocklag,
   riskset, flow, extmargin, Russia, cum4-hardening, ddd gate, 07x). Adding the
   preferred shock as a second column at that time is cheap; adding the whole
   menu is not, and would invite multiple-comparison problems the menu was
   designed to avoid.
6. **Doc edits are deferred to the atomic refresh** that comes with the
   full-battery re-run, per the work-package spec. No doc was touched here.

---

## 7. Open items

**F7 exact-nesting variant — SUPERSEDED, close it.**
Doc line 81 defers a one-column addition (`US·CN·gpr(M2)`) whose only purpose was
to make the raw-GPR two-interaction spec (`US·CN·GPR_t` + `US·CN·GPR_{t−1}`)
*exactly* nest the monthly-fitted AR(1) residual, because the residual's
autoregressive term is the quarter's second-to-last month, not the previous
quarter. That was a proxy argument for "the look-ahead does not matter." The
menu answers the question directly and better: `B_i_lvl_ar1_nla_qend` is the
same construction as the baseline with the look-ahead physically removed, it
correlates 0.9996 with the baseline, and β₃ moves from −1.361e-6 (p = 0.763) to
−1.227e-6 (p = 0.784). The look-ahead is worth ~1e-7 in β₃ and nothing in
inference. Recommend: mark the deferred exact-nesting column **superseded by the
shock menu (P1c)** in the doc, and cite B-i instead.
Caveat: this closes the *nesting* debt, not the F7 numbers themselves. The F7
two-interaction result in the doc (−8.4e-7 / +2.6e-7, N = 347,952) is **pre-P0
vintage** and still needs re-running on the P0 panel with the rest of the
battery. It stays useful as the one robustness column that uses no AR model at
all.

**Other open items.**
- Full-battery re-run on the P0 panel with baseline + `s_D_q_ar1_nla`. Blocking
  the atomic doc refresh.
- The advisor-facing figure-2 / pattern-3 correction from P0 is still pending
  (unrelated to this package).
- `gpr_ar1_coefficients.csv` still persists a = 0, b = 0 from the old Julia
  let-scope bug. The shock itself is unaffected. The menu writes its own AR
  coefficients into `shock_menu_diagnostics.csv` (`ar_coefs`,
  `ar_n_fit`, `ar_fit_first`, `ar_fit_last`), so that file is now the reliable
  record. Fix the old CSV when 05 is next touched.
- **mtime is a weak read-only signal under OneDrive.** `shockmenu_results.csv`
  acquired a later mtime (13:17:00) than the RI run that consumed it (13:16:27);
  chased down and confirmed metadata-only (the RI log recorded the anchor at
  13:14:55 with 28 rows, and all 12 `b3_stata_anchor` values byte-match the
  current file). Same cause as the phantom "modified" files in session-start git
  snapshots. Recommend writing sha256 sidecars for `c6_panel.dta` the way step
  05 already does for the GPR parquet.

---

## 8. Verification status

Every load-bearing number in this memo was independently re-derived by an
adversarial verifier who wrote a separate shock builder from the spec (own OLS
AR fits, own acf, own Ljung-Box), a separate reghdfe-equivalent (exact
Frisch-Waugh on the US-minus-NONUS collapse with 82 firm-demeaned quarter
dummies), and a separate RI engine. Six assigned items, all CONFIRMED:

- Independent build reproduces the frozen parquet to **1.776e-15** on all 265
  quarters. The gate is real and the 1e-8 tolerance is hard-coded, not tuned.
- All diagnostics (acf1–acf4, LB Q/p at lags 1/4/8, mean, sd) match to **≤ 3.6e-15**.
- Five CRVE β₃ values reproduce to all 7 significant figures. The verifier's
  uncorrected two-way sandwich equals the reported se × 0.99394 = 1/√(82/81) for
  **every** variant, pinning the SE to reghdfe's G/(G−1) min-cluster correction.
- The deterministic circular-shift RI p reproduces **exactly** for all five
  columns re-run, including the preferred variant's 0.243902. Independent
  5000-draw free-permutation p's land within Monte Carlo error.
- The selection rule reproduces mechanically from the diagnostics alone, and
  provably preceded estimation on disk.
- All 11 standardized columns have mean ~1e-17 and sd = 1.0000000000, so β₃ is a
  per-1-sd effect and is comparable across variants. Scaling identity holds:
  −5.279916e-7 × 2.577890 = −1.361104e-6.

**Corrections to the run narrative** (cosmetic, no data claim affected; recorded
so they do not propagate):
- `shockmenu_vce_diag.csv` has **56** data rows, not 58. All 56 are `se_valid = 1`,
  so the guard conclusion is unchanged.
- corr(B-i, baseline) is **0.99961**, not 1. The 3-dp rounding in the run
  narrative made it look like an exact duplicate. The diagnostics CSV is correct.
- The preferred variant's string in both CSVs is `s_D_q_ar1_nla`. The narrative's
  `s_D_q_ar1_nla_PREFERRED` label is not on disk anywhere.

---

## 9. File inventory

New scripts, in `julia_descriptive/`:

| file | role |
|---|---|
| `build_shock_menu.py` | builds all 11 menu columns, runs the gate, computes diagnostics, applies the pre-registered rule (rule verbatim in the header; selection block at ~line 495; `GATE_TOL = 1e-8` at line 180) |
| `run_shock_menu.do` | 28 reghdfe specs (12 variants × 2 FE for dv=dw, 2 variants × 2 FE for dv=dw_lead1), B9 VCE guard, drift gate vs canonical |
| `run_ri_shockmenu.py` | RI for 12 columns, free permutation (5000) + circular shift (81), SEED 20260702 |

New outputs, in `julia_descriptive/output/`:

| file | contents |
|---|---|
| `shock_menu_quarterly.{csv,parquet,dta}` | the 82-quarter grid with all raw + standardized menu columns |
| `shock_menu_diagnostics.csv` | 11 rows: acf1–4, LB Q/p at 1/4/8, mean/sd raw, corr with baseline, AR coefficients and fit window, gate results, selection rank |
| `shock_menu_corr.csv` | 11×11 correlation matrix incl. baseline |
| `shock_menu_rederivation_check.csv` | 8 prior external measurements, re-derived, with reproduce verdicts |
| `shock_menu_preferred.txt` | exactly `s_D_q_ar1_nla\n`; the `.do` reads the winner from here |
| `shockmenu_c6_panel.parquet` | the panel with all `us_cn_shock_v` columns attached |
| `shockmenu_results.csv` | 28 rows: b2/se2/p2, b3/se3/p3, N, cluster counts, df_r, r2, corr with panel shock, status |
| `shockmenu_vce_diag.csv` | 56 rows, B9 degenerate-VCE guard |
| `ri_shockmenu.csv` | 12 rows: b3, p_free, p_circ, arbiter_p, anchor + relerr, degeneracy flags |

Read-only inputs, verified unchanged: `E:/Data/Data/ai_gpr_bilateral_monthly.csv`
(796 months, 1960-01..2026-04), `output/gpr_quarterly_with_shock.parquet`
(265 quarters, frozen), `output/c6_panel.dta` (P0 vintage).
