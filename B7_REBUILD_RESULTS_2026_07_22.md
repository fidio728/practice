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
