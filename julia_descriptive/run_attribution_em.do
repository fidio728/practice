*=======================================================================
* run_attribution_em.do  (2026-08-06)
*
* ATTRIBUTION of the 3-pairwise headline across the two advisor-directed
* construction changes:
*   CHANGE 1 = holdings SNAPSHOT rule (quarter rule replaces W=10 as-of)
*   CHANGE 2 = ZERO/MISSING recode of the Revere exposure denominator
*
* Both changes are already baked into the CANONICAL panel. The two are
* separated WITHOUT re-running 02/06 because 06_cartesian_grid.jl carries
* zero_recode_flag_lag1q into the grid and build_audit_panel_f1f2f7.py
* projects it as zr_lag:
*
*   zr_lag == 0  <->  n_supplychain_links_lag1q > 0
*                <->  the OLD guard NULLIF(n_supplychain_links, 0) returned
*                     a NON-NULL ratio  ==  the OLD codable set
*   zr_lag == 1  ->   recoded zero, competitor/partner-only links   (NEW)
*   zr_lag == 2  ->   recoded zero, no active link of any type      (NEW)
*   zr_lag == -1 ->   sentinel for NULL upstream (should not occur on the
*                     estimation sample; counted and reported)
*
* So, holding the NEW snapshot fixed:
*   run A  = `if zr_lag==0`  -> new snapshot + OLD missing rule
*   run B  = full sample     -> new snapshot + zero recode  (CANONICAL)
* and run A vs prior-P0 isolates the SNAPSHOT effect, run B vs run A
* isolates the SAMPLE-EXPANSION (zero-recode) effect.
*
* run A2 is a supplementary strict variant: run A additionally intersected
* with the 6,867-firm universe of the archived pre-change panel
* (c6_panel_preEM.dta), which strips the handful of firms that the SNAPSHOT
* change alone newly admits. Reported for transparency, NOT as the headline
* attribution arm.
*
* Spec is verbatim the headline (GLOBAL-MAIN + S_{t-1} PRIMARY, 2026-08-08):
*   reghdfe dw us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)
* where dw = Δw under the FULL-portfolio (global) denominator and
* us_cn_slag = us×cn_lag×S_{t-1}, with the it+gt (fq gq) comparison reported
* alongside, as in run_headline_3pairwise.do. The EU-denominator (dw_eu) and
* S_t (us_cn_shock) cells enter only the canonical-artifact refresh in
* section 3 (2x2 layout).
*=======================================================================

clear all
set more off
set linesize 250

local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

*-----------------------------------------------------------------------
* 0. Old (pre-change, preEM/W=10) firm universe, for the supplementary A2
*-----------------------------------------------------------------------
use firm_str using "`OUT'/c6_panel_preEM.dta", clear
duplicates drop firm_str, force
gen byte old_univ = 1
qui count
display "PREEM firm universe (c6_panel_preEM.dta): " %9.0gc r(N) " firms"
tempfile oldfirms
qui save `oldfirms', replace

*-----------------------------------------------------------------------
* 1. Canonical NEW panel
*-----------------------------------------------------------------------
use "`OUT'/audit_c6_panel.dta", clear
qui count
display "audit_c6_panel.dta rows (regressor-complete): " %12.0gc r(N)

gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
gen us_cn       = us*cn_lag
* PRIMARY triple = S_{t-1} (2026-08-08); S_t kept for the 2x2 refresh only.
gen us_cn_slag  = us*cn_lag*s_lag
gen us_cn_shock = us*cn_lag*shock

qui merge m:1 firm_str using `oldfirms', keep(master match) gen(_mold)
qui replace old_univ = 0 if old_univ == .
label define OLDU 0 "new-only firm" 1 "in preEM universe"
label values old_univ OLDU

display _newline "===== A. zr_lag census on the LOADED panel (dw may be missing) ====="
tab zr_lag, missing

display _newline "===== B. zr_lag census on the ESTIMATION-ELIGIBLE rows (dw!=.) ====="
tab zr_lag if !missing(dw), missing

display _newline "===== C. pair integrity: zr_lag must be constant within firm-quarter ====="
bysort firm_str rd_day (zr_lag): gen byte zr_mismatch = (zr_lag[1] != zr_lag[_N])
qui count if zr_mismatch == 1
local n_mismatch = r(N)
display "rows where zr_lag differs across US/NONUS inside a firm-quarter: " %12.0gc `n_mismatch'

display _newline "===== D. sentinel check: zr_lag == -1 on estimation-eligible rows ====="
qui count if zr_lag == -1 & !missing(dw)
display "zr_lag==-1 & dw non-missing: " %12.0gc r(N) "   (must be 0)"

display _newline "===== E. old_univ x zr_lag on estimation-eligible rows ====="
tab old_univ zr_lag if !missing(dw), missing

* hard gate AFTER the diagnostics have printed
assert `n_mismatch' == 0

*-----------------------------------------------------------------------
* 2. Reporting helper — writes one CSV row and echoes it
*-----------------------------------------------------------------------
capture program drop _wr
program define _wr
    args cf run lbl FE ifc
    * PRIMARY spec verbatim: dw(global) on us_cn + us_cn_slag (S_{t-1})
    qui reghdfe dw us_cn us_cn_slag `ifc', absorb(`FE') vce(cluster firm_n rd_m)
    local b   = _b[us_cn_slag]
    local se  = _se[us_cn_slag]
    local p   = 2*ttail(e(df_r), abs(`b'/`se'))
    local N   = e(N)
    local b2  = _b[us_cn]
    local se2 = _se[us_cn]
    tempvar smp tf tq
    qui gen byte `smp' = e(sample)
    qui egen byte `tf' = tag(firm_n) if `smp' == 1
    qui count if `tf' == 1
    local nf = r(N)
    qui egen byte `tq' = tag(rd_day) if `smp' == 1
    qui count if `tq' == 1
    local nq = r(N)
    file write `cf' "`run'," "`lbl'," "`FE'," ///
        (strtrim(strofreal(`b',   "%14.6e"))) "," ///
        (strtrim(strofreal(`se',  "%14.6e"))) "," ///
        (strtrim(strofreal(`p',   "%9.6f")))  "," ///
        (strtrim(strofreal(`N',   "%15.0f"))) "," ///
        (strtrim(strofreal(`nf',  "%15.0f"))) "," ///
        (strtrim(strofreal(`nq',  "%15.0f"))) "," ///
        (strtrim(strofreal(`b2',  "%14.6e"))) "," ///
        (strtrim(strofreal(`se2', "%14.6e"))) _n
    display %-6s "`run'" %-46s "`lbl'" %-9s "`FE'" ///
            "  b3=" %13.6e `b' "  se=" %13.6e `se' ///
            "  p=" %8.4f `p' "  N=" %11.0gc `N' ///
            "  firms=" %8.0gc `nf' "  qtrs=" %5.0f `nq'
end

tempname cf
file open `cf' using "`OUT'/attribution_em_snapshot_vs_zerorecode.csv", write replace
file write `cf' "run,label,fe,b3,se3,p3,N,n_firms,n_quarters,b2_us_cn,se2_us_cn" _n

display _newline "===== 3-PAIRWISE HEADLINE ATTRIBUTION ====="
_wr `cf' "B"  "new snapshot + zero recode (CANONICAL)"       "fq gq ig" ""
_wr `cf' "B"  "new snapshot + zero recode (CANONICAL)"       "fq gq"    ""
_wr `cf' "A"  "new snapshot + OLD missing rule (zr_lag==0)"  "fq gq ig" "if zr_lag==0"
_wr `cf' "A"  "new snapshot + OLD missing rule (zr_lag==0)"  "fq gq"    "if zr_lag==0"
_wr `cf' "A2" "run A intersected with preEM firm universe"   "fq gq ig" "if zr_lag==0 & old_univ==1"
_wr `cf' "A2" "run A intersected with preEM firm universe"   "fq gq"    "if zr_lag==0 & old_univ==1"
_wr `cf' "Z"  "zero-recode arm only (zr_lag>=1)"             "fq gq ig" "if zr_lag>=1"
_wr `cf' "Z"  "zero-recode arm only (zr_lag>=1)"             "fq gq"    "if zr_lag>=1"
file close `cf'
display "Wrote `OUT'/attribution_em_snapshot_vs_zerorecode.csv"

*-----------------------------------------------------------------------
* 3. Refresh the LIVING canonical artifact (the preEM twin was archived by
*    hand as headline_3pairwise_canonical_preEM.csv before this run).
*    Layout is byte-for-byte the one run_headline_3pairwise.do writes
*    (LOCKED 2026-08-08): spec,denom,timing,fe,b3,se,p,N — PRIMARY (global x
*    S_{t-1}) + itgt variant + the three remaining 2x2 diagnostic cells.
*-----------------------------------------------------------------------
capture program drop _canrow
program define _canrow
    args cf spec denom timing felab y x3 FE
    qui reghdfe `y' us_cn `x3', absorb(`FE') vce(cluster firm_n rd_m)
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
