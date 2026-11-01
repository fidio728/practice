*=======================================================================
* run_attribution_ladder_v3.do  (2026-08-09, ATTRIBUTION + RI stage, REBUILD v3)
*
* Grain / denominator / timing ATTRIBUTION LADDER on the v3 canonical panel:
*   L0 = mmv2 security grain, EU denom, S_t      -> gate_e_headline_mmfix.csv
*        (NOT rerun here, read by the caller)
*   L1 = fund grain, EU denom, S_t
*   L2 = fund grain, GLOBAL denom, S_t
*   L3 = fund grain, GLOBAL denom, S_{t-1}       = NEW PRIMARY
* The 3pw cells of L1/L2/L3 and the itgt cells of L2/L3 ALREADY live in
* headline_3pairwise_canonical.csv (fresh this chain, 2026-08-09 05:18) and are
* REUSED, not recomputed. This file adds ONLY what is missing:
*   (1) L1 itgt            : dw_eu  x us_cn_shock  absorb(fq gq)
*   (2) diag EU x S_{t-1} itgt : dw_eu x us_cn_slag absorb(fq gq)
*   (3) L3 3pw + itgt RERUNS, solely to (a) hard-gate this session against the
*       living canonical CSV (rel < 1e-6, fail-closed, anchors never hardcoded)
*       and (b) count the sparse identifying set on the true e(sample):
*       rows with cn_lag > 0, distinct treated firms, distinct treated
*       firm-quarters (feedback_treated_firm_count).
* Output: attribution_ladder_v3_cells.csv (NEW artifact, nothing clobbered).
*=======================================================================

clear all
set more off
set linesize 250

local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

*-----------------------------------------------------------------------
* 0. Living anchors from the canonical CSV (fail-closed, never hardcoded)
*-----------------------------------------------------------------------
import delimited using "`OUT'/headline_3pairwise_canonical.csv", clear varnames(1) case(preserve)
confirm variable spec denom timing fe b3 se p N
qui count if spec == "primary" & fe == "fq_gq_ig"
assert r(N) == 1
qui su b3 if spec == "primary" & fe == "fq_gq_ig", meanonly
local B3_PRIM = r(mean)
qui count if spec == "primary_itgt" & fe == "fq_gq"
assert r(N) == 1
qui su b3 if spec == "primary_itgt" & fe == "fq_gq", meanonly
local B3_PRIM_ITGT = r(mean)
display "living anchors: primary 3pw b3 = " %14.6e `B3_PRIM' "   primary itgt b3 = " %14.6e `B3_PRIM_ITGT'

*-----------------------------------------------------------------------
* 1. Canonical v3 panel
*-----------------------------------------------------------------------
use "`OUT'/audit_c6_panel.dta", clear
qui count
display "audit_c6_panel.dta rows: " %12.0gc r(N)
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
gen us_cn       = us*cn_lag
gen us_cn_slag  = us*cn_lag*s_lag
gen us_cn_shock = us*cn_lag*shock

*-----------------------------------------------------------------------
* 2. Row writer: b3/se/p/N + firms/quarters + treated (cn_lag>0) counts
*-----------------------------------------------------------------------
capture program drop _lrow
program define _lrow
    args cf cell denom timing felab y x3 FE
    qui reghdfe `y' us_cn `x3', absorb(`FE') vce(cluster firm_n rd_m)
    local b   = _b[`x3']
    local se  = _se[`x3']
    local p   = 2*ttail(e(df_r), abs(`b'/`se'))
    local N   = e(N)
    tempvar smp tf tq ttf ttq
    qui gen byte `smp' = e(sample)
    qui egen byte `tf' = tag(firm_n) if `smp' == 1
    qui count if `tf' == 1
    local nf = r(N)
    qui egen byte `tq' = tag(rd_day) if `smp' == 1
    qui count if `tq' == 1
    local nq = r(N)
    qui count if `smp' == 1 & cn_lag > 0 & !missing(cn_lag)
    local ntr = r(N)
    qui egen byte `ttf' = tag(firm_n) if `smp' == 1 & cn_lag > 0 & !missing(cn_lag)
    qui count if `ttf' == 1
    local ntf = r(N)
    qui egen byte `ttq' = tag(firm_n rd_day) if `smp' == 1 & cn_lag > 0 & !missing(cn_lag)
    qui count if `ttq' == 1
    local ntfq = r(N)
    file write `cf' "`cell',`denom',`timing',`felab'," ///
        (strtrim(strofreal(`b',   "%14.6e"))) "," ///
        (strtrim(strofreal(`se',  "%14.6e"))) "," ///
        (strtrim(strofreal(`p',   "%9.6f")))  "," ///
        (strtrim(strofreal(`N',   "%15.0f"))) "," ///
        (strtrim(strofreal(`nf',  "%15.0f"))) "," ///
        (strtrim(strofreal(`nq',  "%15.0f"))) "," ///
        (strtrim(strofreal(`ntr', "%15.0f"))) "," ///
        (strtrim(strofreal(`ntf', "%15.0f"))) "," ///
        (strtrim(strofreal(`ntfq',"%15.0f"))) _n
    display %-22s "`cell'" %-7s "`denom'" %-5s "`timing'" %-9s "`felab'" ///
        "  b3=" %13.6e `b' "  se=" %13.6e `se' "  p=" %8.4f `p' ///
        "  N=" %11.0gc `N' "  firms=" %8.0gc `nf' ///
        "  treated_rows=" %10.0gc `ntr' "  treated_firms=" %7.0gc `ntf' ///
        "  treated_fq=" %9.0gc `ntfq'
    c_local last_b3 = `b'
end

tempname cf
file open `cf' using "`OUT'/attribution_ladder_v3_cells.csv", write replace
file write `cf' "cell,denom,timing,fe,b3,se,p,N,n_firms,n_quarters,n_treated_rows,n_treated_firms,n_treated_fq" _n

display _newline "===== L3 RERUNS (coherence gate + treated counts on true e(sample)) ====="
_lrow `cf' "L3_3pw_rerun"  "global" "slag" "fq_gq_ig" dw us_cn_slag "fq gq ig"
local rel = abs((`last_b3' - `B3_PRIM') / `B3_PRIM')
display "coherence gate L3 3pw rerun vs living canonical primary: rel diff = " %10.2e `rel'
assert `rel' < 1e-6

_lrow `cf' "L3_itgt_rerun" "global" "slag" "fq_gq"    dw us_cn_slag "fq gq"
local rel = abs((`last_b3' - `B3_PRIM_ITGT') / `B3_PRIM_ITGT')
display "coherence gate L3 itgt rerun vs living canonical primary_itgt: rel diff = " %10.2e `rel'
assert `rel' < 1e-6

display _newline "===== NEW CELLS (the two EU itgt cells absent from the canonical CSV) ====="
_lrow `cf' "L1_itgt_new"        "eu" "st"   "fq_gq" dw_eu us_cn_shock "fq gq"
_lrow `cf' "diagEU_slag_itgt_new" "eu" "slag" "fq_gq" dw_eu us_cn_slag "fq gq"

file close `cf'
display _newline "Wrote `OUT'/attribution_ladder_v3_cells.csv"
display "Done."
