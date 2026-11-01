* run_headline_3pairwise.do — adopt the fully-saturated three-way pairwise FE
* (firm×quarter it + group×quarter gt + firm×group ig) as the HEADLINE, per the
* Khwaja-Mian / De Haas saturated design. Reports it+gt+ig (headline) alongside
* it+gt (comparison) for every main specification. The null must survive the
* most demanding FE. Two-way cluster (firm, quarter) throughout.
*
* GLOBAL-MAIN + S_{t-1} PRIMARY (2026-08-08 decision):
*   dw    = Δw with the FULL-portfolio (GLOBAL) denominator — MAIN, per
*           research_plan.tex "Country portfolio weight" (w = I/T over the
*           full book; raw pull confirms funds report their global book).
*   dw_eu = Δw with the EU-restricted denominator — labeled WITHIN-EUROPE
*           REALLOCATION diagnostic (the C1-era outcome).
*   s_lag = S_{t-1} (advisor-directed PRIMARY timing);  shock = S_t (labeled
*           timing diagnostic).
* PRIMARY SPEC:
*   reghdfe dw us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)
* (us, us×S_{t-1}, cn_lag, cn_lag×S_{t-1} are absorbed by the saturated FE;
* the surviving estimands are us_cn = us×cn_lag and the triple
* us_cn_slag = us×cn_lag×S_{t-1}.)  itgt variant: absorb(fq gq).
* The canonical CSV below carries the FULL 2x2 {global, EU} x {S_{t-1}, S_t}
* at the headline FE, so the attribution stage reads all four cells from one
* run.

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

* helper: print b3/se/p/N for a given outcome, TRIPLE regressor, FE set + if
capture program drop _row
program define _row
    args lbl y x3 FE ifc
    qui reghdfe `y' us_cn `x3' `ifc', absorb(`FE') vce(cluster firm_n rd_m)
    display %-24s "`lbl'" %-11s "`FE'" "  b3=" %10.3e _b[`x3'] ///
            "  p=" %6.4f 2*ttail(e(df_r), abs(_b[`x3']/_se[`x3'])) ///
            "  N=" %9.0gc e(N)
end

* helper: write one canonical-CSV row from the stored estimates of a spec.
* Layout (LOCKED 2026-08-08; run_attribution_em.do refresh block must match
* byte-for-byte): spec,denom,timing,fe,b3,se,p,N
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

*=======================================================================
* PART A — w-based specs on the audit panel
*=======================================================================
use "`OUT'/audit_c6_panel.dta", clear
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
gen us_cn        = us*cn_lag
gen us_cn_slag   = us*cn_lag*s_lag
gen us_cn_shock  = us*cn_lag*shock
gen us_cn_gpr    = us*cn_lag*gpr
gen us_cn_gprlag = us*cn_lag*gpr_lag

display _newline "===== PRIMARY: dw(global) x S_{t-1}, 3-pairwise (fq gq ig) vs it+gt (fq gq) ====="
_row "PRIMARY dw x S_t-1"   dw     us_cn_slag  "fq gq ig" ""
_row "PRIMARY dw x S_t-1"   dw     us_cn_slag  "fq gq"    ""
display _newline "===== 2x2 DIAGNOSTIC CELLS {denom} x {timing} at fq gq ig ====="
_row "diag dw x S_t"        dw     us_cn_shock "fq gq ig" ""
_row "diag dw_eu x S_t-1"   dw_eu  us_cn_slag  "fq gq ig" ""
_row "diag dw_eu x S_t"     dw_eu  us_cn_shock "fq gq ig" ""

*=======================================================================
* CANONICAL MACHINE-READABLE ARTIFACT (audit fix 6): the PRIMARY spec (+itgt)
* and the full 2x2 {global, EU} x {S_{t-1}, S_t} at the headline FE.
* Values are READ FROM THE STORED ESTIMATES, never hardcoded. This is the
* LIVING source cited by run_ddd_nofe_bil.do (Stage-A gate, reads the primary
* row), run_fourgroup.do, run_direction_split.do, run_tercile_3pairwise.do and
* verify_attribution_em.py. Re-run this .do (or run_attribution_em.do, which
* refreshes the same layout) to refresh it.
* N differs slightly across cells: dw_eu drops empty-EU-book cells and s_lag
* drops its first observable quarter — both labeled sample facts, not bugs.
*=======================================================================
tempname cf
file open `cf' using "`OUT'/headline_3pairwise_canonical.csv", write replace
file write `cf' "spec,denom,timing,fe,b3,se,p,N" _n
_canrow `cf' "primary"      "global" "slag" "fq_gq_ig" dw    us_cn_slag  "fq gq ig"
_canrow `cf' "primary_itgt" "global" "slag" "fq_gq"    dw    us_cn_slag  "fq gq"
_canrow `cf' "diag"         "global" "st"   "fq_gq_ig" dw    us_cn_shock "fq gq ig"
* diag global-st at fq gq (itgt): S_t continuity-anchor cell for the
* run_shock_menu.do drift gate (its baseline itgt run IS this regression;
* audit-vs-c6 panel estimation samples coincide after reghdfe drops).
_canrow `cf' "diag"         "global" "st"   "fq_gq"    dw    us_cn_shock "fq gq"
_canrow `cf' "diag"         "eu"     "slag" "fq_gq_ig" dw_eu us_cn_slag  "fq gq ig"
_canrow `cf' "diag"         "eu"     "st"   "fq_gq_ig" dw_eu us_cn_shock "fq gq ig"
file close `cf'
display "Wrote `OUT'/headline_3pairwise_canonical.csv (primary + 2x2 + itgt-st anchor layout, 2026-08-08)"

* Timing battery (F1a/F1b/F2) on the MAIN family: global outcomes x S_{t-1}
_row "F1a lead dw_t+1"  dw_lead1 us_cn_slag "fq gq ig" ""
_row "F1a lead dw_t+1"  dw_lead1 us_cn_slag "fq gq"    ""
_row "F1b LP cum1"      cum1     us_cn_slag "fq gq ig" ""
_row "F1b LP cum1"      cum1     us_cn_slag "fq gq"    ""
_row "F1b LP cum2"      cum2     us_cn_slag "fq gq ig" ""
_row "F1b LP cum4"      cum4     us_cn_slag "fq gq ig" ""
_row "F1b LP cum4"      cum4     us_cn_slag "fq gq"    ""
_row "F2 in-span dw"    dw       us_cn_slag "fq gq ig" "if in_span==1"
_row "F2 in-span dw"    dw       us_cn_slag "fq gq"    "if in_span==1"

display _newline "===== F7 GPR two-interaction (3-pairwise vs it+gt) ====="
foreach FE in "fq gq ig" "fq gq" {
    qui reghdfe dw us_cn us_cn_gpr us_cn_gprlag, absorb(`FE') vce(cluster firm_n rd_m)
    display "`FE': b(GPR_t)=" %9.3e _b[us_cn_gpr] " p=" %6.4f 2*ttail(e(df_r),abs(_b[us_cn_gpr]/_se[us_cn_gpr])) ///
            "  b(GPR_t-1)=" %9.3e _b[us_cn_gprlag] " p=" %6.4f 2*ttail(e(df_r),abs(_b[us_cn_gprlag]/_se[us_cn_gprlag]))
}

*=======================================================================
* PART B — ownership FLOW (F6)
* ownership_c6_panel.dta carries shock (S_t) only; derive s_lag (S_{t-1})
* in-file from the quarter-level shock series (shock is quarter-constant;
* contiguity of the quarter sequence is asserted, so [_n-1] on the deduped
* quarter list IS the true previous quarter).
*=======================================================================
use "`OUT'/ownership_c6_panel.dta", clear
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
preserve
keep rd_m shock
duplicates drop
sort rd_m
by rd_m: assert _N == 1
assert rd_m - rd_m[_n-1] == 3 if _n > 1
gen double s_lag = shock[_n-1]
keep rd_m s_lag
tempfile slagmap
qui save `slagmap', replace
restore
qui merge m:1 rd_m using `slagmap', assert(match) nogen
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
gen us_cn       = us*cn_lag
gen us_cn_slag  = us*cn_lag*s_lag
gen us_cn_shock = us*cn_lag*shock
display _newline "===== ownership FLOW (F6), PRIMARY S_{t-1}, 3-pairwise vs it+gt ====="
_row "flow x S_t-1"  flow  us_cn_slag  "fq gq ig" ""
_row "flow x S_t-1"  flow  us_cn_slag  "fq gq"    ""
display _newline "----- flow x S_t (timing diagnostic) -----"
_row "flow x S_t"    flow  us_cn_shock "fq gq ig" ""
display _newline "Done."
