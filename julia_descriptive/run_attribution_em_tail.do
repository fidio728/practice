*=======================================================================
* run_attribution_em_tail.do  (2026-08-06)
*
* Tail of run_attribution_em.do. The parent do-file aborted at the OPTIONAL
* "Z" arm (regression on the recoded-zero rows alone) with
*     reghdfe_fix_psd():  3301  subscript invalid
* so the LIVING canonical artifact never got refreshed. This file does the
* two remaining things, and documents WHY the Z arm is degenerate rather
* than retrying it:
*
*   (a) DEGENERACY EVIDENCE. In 06_cartesian_grid.jl the zero recode sets
*       china_share = 0 exactly when n_supplychain_links = 0, which is
*       exactly when zero_recode_flag >= 1. Lagged, that means
*           zr_lag >= 1  ->  cn_lag == 0  ->  us_cn == 0 and us_cn_shock == 0
*       on EVERY recoded-zero row. A regression on that subsample has two
*       identically-zero regressors, hence the singular cluster VCV. The
*       tabstat below is the evidence, not an inference.
*       CONSEQUENCE FOR THE ATTRIBUTION: the 555,340 rows the recode adds
*       carry no regressor variation at all. They move b3 only through the
*       fixed effects (they reshape the group x quarter means and add firms
*       to the firm x group set) and through the clustering/df, never
*       through the interaction itself.
*
*   (b) refresh headline_3pairwise_canonical.csv on the post-EM canonical
*       panel. The pre-change twin was archived by hand to
*       headline_3pairwise_canonical_preEM.csv BEFORE this run.
*
* [AMENDED 2026-08-08 — GLOBAL-MAIN + S_{t-1} PRIMARY] This tail is an
* INCIDENT patch kept for the record; the parent run_attribution_em.do now
* completes on its own (Z arm included) and owns the canonical refresh. If
* this tail IS re-run, section (b) must — and now does — write the LOCKED
* 2026-08-08 layout spec,denom,timing,fe,b3,se,p,N with the PRIMARY spec
* dw(global) x us_cn_slag (S_{t-1}); the retired two-row S_t layout would
* clobber the living artifact and trip the layout guard in
* run_ddd_nofe_bil.do. The degeneracy evidence in (a) is timing-invariant:
* zr_lag>=1 -> cn_lag==0 -> us_cn, us_cn_shock AND us_cn_slag are all
* identically zero on the recoded-zero rows.
*=======================================================================

clear all
set more off
set linesize 250

local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`OUT'/audit_c6_panel.dta", clear
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
gen us_cn       = us*cn_lag
* PRIMARY triple = S_{t-1} (2026-08-08); S_t kept as the timing diagnostic.
gen us_cn_slag  = us*cn_lag*s_lag
gen us_cn_shock = us*cn_lag*shock

display _newline "===== (a) regressor variation by zr_lag arm ====="
tabstat cn_lag us_cn us_cn_slag us_cn_shock, by(zr_lag) stat(n mean sd min max) format(%12.6g) columns(statistics)

qui count if zr_lag >= 1 & cn_lag != 0
display "recoded-zero rows (zr_lag>=1) with cn_lag != 0: " %12.0gc r(N) "   (must be 0)"
qui count if zr_lag >= 1
display "recoded-zero rows added by CHANGE 2 (zr_lag>=1): " %12.0gc r(N)
qui count if zr_lag == 0 & cn_lag > 0
display "old-rule rows with strictly positive cn_lag:     " %12.0gc r(N)
qui count if zr_lag == 0 & cn_lag == 0
display "old-rule rows with cn_lag exactly 0:             " %12.0gc r(N)

display _newline "===== (b) refresh the LIVING canonical artifact ====="
* Layout is byte-for-byte the one run_headline_3pairwise.do writes
* (LOCKED 2026-08-08): spec,denom,timing,fe,b3,se,p,N — PRIMARY (global x
* S_{t-1}) + itgt variant + the three remaining 2x2 diagnostic cells.
capture program drop _canrow
program define _canrow
    args cf spec denom timing felab y x3 FE
    qui reghdfe `y' us_cn `x3', absorb(`FE') vce(cluster firm_n rd_m)
    display %-13s "`spec'" %-7s "`denom'" %-5s "`timing'" %-9s "`felab'" ///
        "  b3=" %13.6e _b[`x3'] "  se=" %13.6e _se[`x3'] ///
        "  p=" %8.4f 2*ttail(e(df_r), abs(_b[`x3']/_se[`x3'])) "  N=" %11.0gc e(N)
    file write `cf' "`spec',`denom',`timing',`felab'," ///
        (strtrim(strofreal(_b[`x3'], "%14.6e"))) "," ///
        (strtrim(strofreal(_se[`x3'], "%14.6e"))) "," ///
        (strtrim(strofreal(2*ttail(e(df_r), abs(_b[`x3']/_se[`x3'])), "%9.6f"))) "," ///
        (strtrim(strofreal(e(N), "%15.0f"))) _n
end
tempname hf
file open `hf' using "`OUT'/headline_3pairwise_canonical.csv", write replace
file write `hf' "spec,denom,timing,fe,b3,se,p,N" _n
_canrow `hf' "primary"      "global" "slag" "fq_gq_ig" dw    us_cn_slag  "fq gq ig"
_canrow `hf' "primary_itgt" "global" "slag" "fq_gq"    dw    us_cn_slag  "fq gq"
_canrow `hf' "diag"         "global" "st"   "fq_gq_ig" dw    us_cn_shock "fq gq ig"
* diag global-st at fq gq (itgt): S_t continuity-anchor cell for the
* run_shock_menu.do drift gate (byte-for-byte with run_headline_3pairwise.do).
_canrow `hf' "diag"         "global" "st"   "fq_gq"    dw    us_cn_shock "fq gq"
_canrow `hf' "diag"         "eu"     "slag" "fq_gq_ig" dw_eu us_cn_slag  "fq gq ig"
_canrow `hf' "diag"         "eu"     "st"   "fq_gq_ig" dw_eu us_cn_shock "fq gq ig"
file close `hf'
display "Refreshed `OUT'/headline_3pairwise_canonical.csv (GLOBAL-MAIN 2x2 + itgt-st anchor vintage)"

display _newline "Done."
