# B7 fix — full rebuild results (2026-07-22)

**Fix (measure B):** `china_share` = CN CUSTOMER+SUPPLIER links / total CUSTOMER+SUPPLIER
links. Drops COMPETITOR (14.85% of CN edges) and all PARTNER-* (21%) from both
numerator and denominator. Aligns the regression exposure variable with the
descriptive Figure 1 universe and the doc §4.2 estimand.

**Recompute sites (all fixed — china_share is rebuilt in 3 places):**
- `02_china_exposure.jl`: china_share column + new `n_supplychain_links` + `china_share_alltypes` diagnostic.
- `05_combine_visualize.jl:~292`: SUM-aggregated exposure_by_sec_entity (multi-match).
- `06_cartesian_grid.jl:~232`: 1:1-join exposure_by_sec_entity.
- Python builders (build_c6/country/firm_ladder) read china_share directly — no change.

**Panel impact:**
- HIGH cutoff (median positive lagged CN): 0.0476 → **0.0625**
- Estimation universe: 7,928 → **6,854 firms** (−13.5%); china_share_lag1q non-null 462,600 → 347,952 rows.
- Firms dropped = those whose only China links were competitor/partner (no CUST/SUPP link) → NULL exposure.

**Regression results (old → new):**

| spec | old | new (B7) | note |
|---|---|---|---|
| DDD headline β₃ (us_cn_shock, dw, 3-pairwise) | +1.80e-6, p=0.35, N=462,096 | **+2.75e-6 (se 1.70e-6), p=0.110, N=347,490** | still null; point est +50%, p lower |
| Firm ladder M1 (firm+qtr FE, level w) | +9.1e-5, p=0.037 | +9.0e-5, p=0.048 | marginal |
| Firm ladder M2 (firm×grp FE) | −1.6e-5, p=0.739 | −5.4e-5, p=0.324 | null |
| **Firm ladder M3 (fq+gq FE)** | +8.6e-5, **p=0.046** ⭐ | +8.6e-5, **p=0.060** | **loses significance** |
| Firm ladder M4 (3-pairwise) | −2.1e-5, p=0.637 | −6.1e-5, p=0.230 | null |
| Country ladder M1–M4 (Step 1) | all null | all null (us_cn −0.04~−0.01, p≥0.17) | unchanged |

**Bottom line:** the null conclusion is preserved. The cleaner supply-chain measure
gives somewhat LARGER point estimates (β₃ +50%, p 0.35→0.11) but nothing crosses
5%, and the sole starred ladder cell (M3) loses its star. Consistent with, and
slightly reinforcing, the overall null.

**Follow-ups still owed:**
1. Hardcoded "gate" checks in `run_ddd_nofe_bil.do` (β₃=1.80e-6, N=462,096) and any
   other `.do` with locked pre-B7 values must be updated to the new numbers (the
   gate correctly tripped and stopped Stage B/C — those columns not re-run yet).
2. Full doc number-refresh: `Essay2_methodology_full.md` has many pre-B7 hardcoded
   numbers (β₃ table §~310, firm counts, tail menu, F1/F2 rows) now stale.
3. Russia positive-control chain (`02_russia_exposure.jl` etc.) has the SAME B7
   contamination — russia_share also counts all rel_types. Apply the same fix +
   rebuild before citing Russia numbers.
4. RI / bootstrap / did_imputation robustness re-runs on the new panel.
5. The earlier NULLS FIRST dedup fix rode along in this same rebuild.

---

# B10 fix — Russia LP pre-invasion anchoring (2026-07-22)

**Fix:** `build_russia_lp_panel.py` — LP regressor now FIXED at pre-invasion
exposure (russia_share as of 2021Q4 = ru_lag at 2022Q1), held across all
horizons, mirroring the w_base anchor. Previously it used time-varying
russia_share_lag1q, so h>=1 used POST-invasion exposure (contradicting the
docstring's "as of 2021Q4"). Note: russia_share here is still pre-B7
(all-rel-type) — B10 is orthogonal to the Russia B7 contamination.

**Result (permutation p, 20k perms; all beta negative = divestment direction):**

| h | quarter | old perm_p | new perm_p (B10) |
|---|---|---|---|
| 0 | 2022Q1 | 0.042 | 0.040 (marginal) |
| 1 | 2022Q2 | **0.046** | **0.052** (now null at 5%) |
| 2 | 2022Q3 | — | 0.099 |
| 3 | 2022Q4 | — | 0.068 |
| 4–7 | 2023 | — | 0.14–0.19 |

Sample now uniform across horizons (n=5,466, balanced 10,932 rows/horizon) —
the correct event-study design. The previously-cited h=1 significance was
partly an artifact of post-invasion exposure; under correct anchoring only h=0
is marginal. Reinforces B12 (Russia positive control is underpowered); do not
cite the LP h=1 as design validation.

**Russia branch still owed:** B7 rel-type fix on russia_share (02_russia +
06_russia + merged_us_ru rebuild) and B9 (R1 degenerate VCE / SE='.' silent
CSV) — bundle into one Russia rebuild.

---

# B8 fix + flow inference under B7 (2026-07-22)

**B8 bug:** doc §7.6 "not a zero-fill artifact" cited a held-only verification
(+0.00122) that `test_obs_only.do` ran on the superseded `dos` outcome, not the
primary `flow` (the obsonly panel has no flow column). Fixed: `run_flow_heldonly.do`
runs the held-only zero-fill robustness on `flow` (B7 panel). Held-only 3-pairwise
β₃ = +0.00135 (still positive, further from disengagement).

**Bigger finding surfaced (B7 side-effect on flow inference):**
Rebuilding ownership_c6_panel.dta on the B7 grid made the flow two-way-cluster CRVE
UNRELIABLE — primary fq-gq spec has a degenerate/singular VCV (SE not computable,
the B9 pathology on a China spec), 3-pairwise gives p=0.0001. Cause: `flow` is
un-winsorized and fat-tailed (kurtosis 5167, max +9.83 = +983% of float from tiny
lagged denominators).

Resolution:
- Winsorize p1/p99 → kurtosis 5167→5.3, degeneracy gone; CRVE R1 p=0.003, R2 p=0.056.
- **Design-based RI (permute 82 quarter shocks, `run_ri_flow.py`): raw p=0.369,
  winsor p=0.384 — NULL both.**

Conclusion: the flow null STANDS under B7 and is reinforced (the CRVE "significance"
is a fat-tail/few-cluster artifact killed by RI). β₃ positive throughout = wrong sign
for disengagement anyway. New files: run_flow_heldonly.do, run_flow_winsor.do,
run_ri_flow.py.

Follow-up: refresh the pre-B7 flow numbers scattered in §7.6 (+0.00088/p=0.44/N=300,866
at lines 27/80/404/551) to the B7 panel + RI; the conclusion is unchanged.
Also note: B9 (degenerate two-way-cluster VCE) is NOT Russia-only — it hits the
China flow fq-gq spec too; winsorization is the fix there.

---

# B9 fix — degenerate VCE no longer silently written (2026-07-22)

**Bug:** run_russia_headline.do R1 (it+gt, two-way cluster) has a degenerate
CGM VCE (non-PSD/singular -> reghdfe returns missing SE) because the Russia
event has very few treated quarters. The column was written to
russia_headline_results.csv with se='.' and no assertion/diagnostic.

**Fix:** after every spec, detect a missing/zero SE, warn loudly, and write
russia_headline_vce_diag.csv marking each spec's VCE validity; esttab now
carries an addnote pointing to the diag file + run_ri_russia.py. Verified:
r1 flagged se_valid=0 (degenerate), r2/r3 se_valid=1.

**Note:** the degenerate two-way-cluster VCE is NOT Russia-only — the China
flow fq-gq spec (B8 work) has the same pathology, there driven by fat tails and
fixed by winsorization. The valid inference for both is design-based RI. This
run used the pre-B7 c6_panel_russia.dta; the Russia B7 rel_type fix + rebuild
is still owed (bundle with a single Russia rebuild).

---

# Shock-tercile dose menu (advisor request) — built, reviewed, run (2026-08-02)

Advisor's ask (2026-07-02 meeting): replace the 2-sigma tail dummy (4 treated
quarters, MacKinnon-Webb few-cluster pathology) with shock TERCILES
(bottom/middle/top thirds of the 82 quarterly shocks; realized bins 28/27/27).
Written + 3-angle multi-agent review (econ / Stata / RI-consistency; caught a
direction-interpretation error, 4 wrong RI pointers, bin-count mislabels);
fixes applied. Files: run_tercile_3pairwise.do, run_ri_tercile.py.

**Results (B7 panel, MAIN = 3-pairwise fq gq ig, N=347,490; T2 middle = base):**

| coef | b | CRVE p | RI p (5000 perms) |
|---|---|---|---|
| us_cn (middle-tercile base) | -1.02e-5 | 0.570 | — |
| us_cn x T1 (bottom) | +1.37e-5 | 0.535 | — |
| us_cn x T3 (top) | +2.05e-5 | 0.264 | **0.233** |
| T3 - T1 contrast (decoupling => negative) | +6.76e-6 | 0.623 | **0.670** |

Python RI reproduces Stata coefficients to 6 significant figures. No degenerate
VCE (both specs valid; every tercile activates 27-28 month-clusters — the
few-cluster problem is cured, which was the point). All estimates positive =
against the decoupling direction; nothing approaches significance under CRVE or
design-based RI. The dose dimension is a clean null, consistent with the
continuous headline, the tail dummy, and the flow outcome. Supersedes the old
tail menu (retires the B14 no-RI critique).

One-liner for the advisor: "Cutting the shock into terciles (27-28 quarters per
bin, restoring valid clustered inference) shows no differential US response in
any dose bin and no dose monotonicity; point estimates are small, positive, and
insignificant under both clustered and permutation inference."

---

# Direction split — sell-to-China vs buy-from-China (2026-08-02)

Design-agenda rank 1. china_share decomposed additively (common supply-chain
denominator, sell+buy=china_share row-wise, double-asserted) into
china_sell_link_share (revenue-exposure link-count proxy: EU_SRC×CUSTOMER +
CN_SRC×SUPPLIER) and china_buy_link_share (input-dependence proxy: EU_SRC×
SUPPLIER + CN_SRC×CUSTOMER), per the FactSet source-perspective convention
(methodology guide p.3, verified by the pipeline reviewer). Pipeline: 02
(counts+cp+shares+identity assert+reciprocal diagnostics both directions) ->
06 -> build_c6 (sell_lag/buy_lag); 05 descriptive-parity only. 3-Opus review:
no must-fix; all should-fixes applied (corr restricted to cn_lag>0, str12,
dropna+cn, cond(XtX) print, group-invariance assert).

**Descriptives (exposed firm-quarters, n=24,345):** corr(sell,buy)=−0.135;
cells: sell-only 46.9%, buy-only 35.1%, both 18.0% — the two directions are
largely carried by different firms; ample independent variation
(cond(XtX)=24).

**Results (MAIN 3-pairwise, N=347,490; CRVE two-way cluster + RI 5000 perms):**

| stat | b | CRVE p | RI p |
|---|---|---|---|
| β₃(sell) | +2.53e-7 | 0.618 | **0.931** |
| β₃(buy) | +5.50e-6 | 0.147 | **0.136** |
| β₃(sell)−β₃(buy) | −5.25e-6 | 0.198 | **0.205** |
| joint β₃s=β₃b=0 | — | F=2.18, p=0.120 | — |
| pooling (β₂ & β₃ equal) | — | F=0.87, p=0.424 | — |

Indicator (any-sell/any-buy, reciprocal-double-record-immune): all null
(p=0.61/0.54, equality 0.52). Python RI reproduces Stata to 6 sig figs.

**Reading:** the offset/attenuation alternative for the aggregate null is
TESTED and not supported — both directions null, equality and pooling not
rejected, and in the decoupling (negative) direction both CIs are tight
(sell lower bound −0.75e-6, buy −2.0e-6 vs headline scale ~2.7e-6). The only
near-marginal cell is buy-side POSITIVE (+5.5e-6, RI p=0.14) — against the
decoupling direction, consistent with the project-wide pattern. Aggregate
china_share specification is statistically valid to pool.

---

# Active-only four-group design (2026-08-03)

Design-agenda rank 2 v2. Feasibility gate passed (US 2021Q4 passive share
39.78% reproduced the external anchor exactly; NONUS UNKNOWN 2018+ mean 10.2%,
far under the 40% degradation threshold -> symmetric four-group design).
Labeling: Funds.STYLE=='Index' -> PASSIVE; explicit non-Index -> ACTIVE;
missing/unmatched -> UNKNOWN (never active). Funds master ~2018-08 snapshot:
post-2018 subsample = PRIMARY (predetermined labels); full period carries a
look-ahead caveat. build_fourgroup_panel.py: reconciliation exact (label
partition sums back to pooled side, 0 dev), 4 balanced books (695,904 rows,
6,854 firms x 82 quarters), sum-to-1 on the full grid. Written+run by a
5-agent workflow; 3-angle review caught one must-fix (sum-to-1 assert on the
filtered panel would always abort — converted to coverage diagnostic) and the
dilution-algebra wording (multiplier = ACTIVE VALUE SHARE, UNKNOWN dilutes
too); all applied.

**Results (t_cn_s; CRVE two-way cluster + RI 5000 perms):**

| contrast | full period | post-2018 (PRIMARY) |
|---|---|---|
| MAIN US_ACTIVE vs NONUS_ACTIVE | +2.79e-6, CRVE p=0.069, **RI p=0.289** | +1.16e-6, CRVE p=0.376, **RI p=0.561** |
| PASSIVE vs PASSIVE (inert benchmark) | +1.03e-6, p=0.760 | −0.14e-6, p=0.963 |
| US-internal ACTIVE vs PASSIVE | +1.56e-6, p=0.135 | −0.18e-6, p=0.842 |

Python RI reproduces Stata to 6 sig figs both samples; no degenerate VCE; book
coverage diagnostic p50=0.84 (<1 expected on the filtered panel).

**Reading:** the passive-dilution hypothesis FAILS — the pure-active contrast
is not more negative than the pooled headline (+2.79e-6 vs pooled +2.75e-6,
essentially identical), and the primary post-2018 sample is a clean null. The
passive benchmark is near-exactly zero (validating that the label carries
content). The full-period CRVE p=0.069 is positive (against decoupling) and
dies under design-based RI (0.289). Third composition alternative retired:
after direction-offset and passive-dilution, the aggregate null keeps
standing on active money alone.

## §7.6 accounting decomposition — FINAL (2026-08-03)

Producer/consumer contract closed: `run_flow_decomposition.py` is the ONE
producer (parquet + .dta with materialized flow_R/flow_common/flow_diff AND
the winsorized `*_w` columns, cutoffs on the estimation sample) and the RI
engine; `run_flow_decomp_step3.do` is the CRVE companion reading the .dta.

Determinism fixes (both required, verified by two bitwise-identical full runs):
1. duckdb `SET threads=1` — parallel hash aggregation summed floats in
   nondeterministic order; last-bit input noise made ri_p jitter ±0.005.
2. RI panel sorted (firm_str, rdate) before factorize, and RI reads the FROZEN
   parquet artifact (bitwise-shared with the .dta) instead of the in-memory frame.

Cross-validation: CRVE reproduces ALL SIX b3 cells to 5 sig figs (winsor cells
matched only after shipping python winsor columns — `_pctile` vs `np.quantile`
definitional drift plus full-panel-vs-estimation-sample cutoff base had CRVE at
+7.13e-4 vs RI +6.65e-4 on flow_diff_w).

Canonical results (N_PERM=5000, seed 20260702; RI panel 120,123 fq, 82 q,
5,061 firms; identity residual 2.2e-16 on 316,691 fq; corr anchor +0.22 winsor
/ +0.20 stable-float):

| outcome | winsor b3 (RI p) | raw b3 (RI p) |
|---|---|---|
| flow_R      | +3.81e-4 (0.69) | −6.14e-5 (0.99) |
| flow_common | +3.31e-5 (0.97) | +1.14e-4 (0.90) |
| flow_diff   | +6.65e-4 (0.41) | +8.90e-4 (0.37) |

All six null; no degenerate VCE (12/12 se_valid). CRVE flow_diff nominal
significance (p<0.001) = fat-tail/weak-FE artifact, adjudicated null by RI.
Estimand rewrite + this block integrated into Essay2_methodology_full.md §7.6;
four "every point estimate is positive" universal claims carved to
quarterly-shock specs pending sagg re-adjudication.

## Audit fixes 5-10 batch — FINAL (2026-08-03, wf_1c0735b0-24c, 6/6 CONFIRMED)

### Russia positive control on B7 (CUSTOMER+SUPPLIER)
06_russia_grid.jl was silently recomputing the old all-rel-type ratio (same
lesson as China B7: the treatment is recomputed downstream — fix every site).
After the full chain rebuild (347,952 rows / 6,854 firms; ru_lag>0 5.20%):

| spec | b3 | CRVE p | RI/perm p |
|---|---|---|---|
| headline it+gt | −4.2147e-6 | 0.0354 | 0.2425 |
| headline 3-pairwise | −4.2812e-6 | 0.0358 | 0.2456 |
| event-window 2022Q1-Q2 | −3.444e-5 | DEGENERATE (se_valid=0) | placebo share 0.457 |
| LP h=0 | −3.613e-5 | t=−0.88 | perm_p 0.2893 (was 0.040 pre-fix) |
| LP h=1 | −5.125e-5 | t=−1.21 | perm_p 0.2181 (was 0.052 pre-fix) |
| LP h=2..7 | all negative | — | perm_p 0.27–0.41 |

**The LP "significance" was an artifact of the all-relationship-type
denominator.** Only the negative sign survives. CRVE strengthening (p=0.035)
is NOT the honest read for a concentrated event; RI (0.24) is. Russia cannot
be cited as passing design validation; B12 power limitation reinforced.
Old refs (h=0 0.040 / h=1 0.052 / b3=−7.33e-6 p=0.085) = pre-B7-fix, superseded.

### F6 flow on rebuilt ownership panel (2026-08-03)
Rebuild changes nothing: 3pw flow +8.900e-04 (CRVE p=0.0001, fat-tail), RI raw
p=0.3691 / winsor p=0.3841 — flow null stable, NOT stale-contaminated.
Degeneracy is vintage-dependent: this vintage the degenerate spec is the OLD
dos comparison (B9 guard fired, se_valid=0), flow specs all valid.
Kurtosis 5169.7, max 9.828 (983% of float).

### F4 country-pair rebuilt on B7
N=172,860 / 3,208 firm clusters. b1=+2.22e-5: firm-quarter cluster p=0.086,
honest 3-country cluster p=0.246 — still clustering-fragile, not evidence.
b3=−1.21e-6 wrong-sign null (p 0.93–0.97). Conclusion unchanged.

### Infrastructure
- Living canonical artifact: output/headline_3pairwise_canonical.csv
  (3pw +2.745538e-6/1.696663e-6/0.109507/347,490; itgt +2.081030e-6/347,952)
- ddd gate retargeted (+2.746e-6/347,490), GATE PASS |diff|=4.6e-10; Stage B
  no-FE point ests valid but ALL SEs missing (non-PSD; inference-degenerate);
  Stage C1 +2.94e-6 p=0.090; C2 +2.83e-6 p=0.100, us_bil +2.41e-5 p=0.208
- 07d refreshed: spec0 +5.83e-10 p=0.999; spec1 +2.081e-6 p=0.1511; spec4
  +5.38e-7 p=0.411; headers in 07d/07e/07f/07g B7-tagged

## A0 cum4 inference hardening — FINAL (2026-08-03, wf_022fbe28-025, verify CONFIRMED)

Engine: run_cum4_inference.py (gates: h0/cum1/cum4 b3 reproduce canonical to 6
sig figs; free-perm p reproduces up to MC noise under the new seed-aligned
draw stream: cum4 0.039→0.042). Shock serial structure RE-DERIVED: acf1=+0.271,
acf2=+0.163, Ljung-Box Q(1) p=0.0125, Q(4) p=0.0228 — S_t is NOT white; free
permutation is anti-conservative on cumulative outcomes (G5 confirmed).

| horizon | b (e-6) | free | circ-shift | mb-L5/L8 | FWER free/mb | WCB |
|---|---|---|---|---|---|---|
| h0   | +2.746 | 0.311 | 0.305 | 0.258/0.256 | 0.563/0.555 | 0.208 |
| cum1 | +4.094 | 0.105 | 0.159 | 0.117/0.132 | 0.047/0.049 | 0.024 |
| cum2 | +7.131 | 0.022 | 0.073 | 0.042/0.054 | 0.017/0.014 | 0.005 |
| cum3 | +6.467 | 0.082 | 0.232 | 0.123/0.139 | 0.159/0.157 | 0.027 |
| cum4 | +8.826 | 0.042 | 0.110 | 0.077/0.086 | 0.159/0.157 | 0.024 |

VERDICT (verify agent, CONFIRMED): the cum4 rejection does NOT survive
serial-robust inference. Every method built for the G5 serial gap
(circular-shift, moving-block) and the G7 multiplicity gap (max-|t| FWER)
clears cum4 above 0.05; only free-perm (documented anti-conservative) and the
score-based quarter-cluster WCB (cross-sectional correction only, does not
model serial dependence) still reject — both positive-signed, against H2.1.
Arbiter = circular-shift → LP family has NO rejection. G8 delivered: WCB IS
feasible (score-based, O(82·B), Webb weights, B=9999) — "boottest OOM" was a
dense-matrix tooling artifact.
cum2 (new, no pre-registered anchor): most consistently sub-0.05 cell
(free 0.022/FWER 0.014-0.017/WCB 0.005) but null under circular-shift (0.073);
positive sign; disclosed, not the arbiter horizon.
Doc integrated (§0/F1/F10 rewrite + §6.3 + new hardening subsection + §13);
docCheck 2 MUST-FIXes (cum2 vs "sole rejection" overclaims) applied by hand.
