* run_headline_3pairwise.do — adopt the fully-saturated three-way pairwise FE
* (firm×quarter it + group×quarter gt + firm×group ig) as the HEADLINE, per the
* Khwaja-Mian / De Haas saturated design. Reports it+gt+ig (headline) alongside
* it+gt (comparison) for every main specification. The null must survive the
* most demanding FE. Two-way cluster (firm, quarter) throughout.

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

* helper: print b3/se/p/N for us_cn_shock under a given FE set + optional if
capture program drop _row
program define _row
    args lbl y FE ifc
    qui reghdfe `y' us_cn us_cn_shock `ifc', absorb(`FE') vce(cluster firm_n rd_m)
    display %-24s "`lbl'" %-11s "`FE'" "  b3=" %10.3e _b[us_cn_shock] ///
            "  p=" %6.4f 2*ttail(e(df_r), abs(_b[us_cn_shock]/_se[us_cn_shock])) ///
            "  N=" %9.0gc e(N)
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
gen us_cn_shock  = us*cn_lag*shock
gen us_cn_gpr    = us*cn_lag*gpr
gen us_cn_gprlag = us*cn_lag*gpr_lag

display _newline "===== 3-PAIRWISE HEADLINE (fq gq ig) vs it+gt (fq gq) ====="
_row "headline dw"      dw       "fq gq ig" ""
_row "headline dw"      dw       "fq gq"    ""

*=======================================================================
* CANONICAL MACHINE-READABLE ARTIFACT (audit fix 6): the two HEADLINE specs.
* Values are READ FROM THE STORED ESTIMATES of the two regressions below,
* never hardcoded. This is the LIVING source cited by run_ddd_nofe_bil.do
* (Stage-A gate), run_fourgroup.do, run_direction_split.do and
* run_tercile_3pairwise.do. Re-run this .do to refresh it.
*=======================================================================
tempname cf
file open `cf' using "`OUT'/headline_3pairwise_canonical.csv", write replace
file write `cf' "spec,b3,se,p,N" _n
qui reghdfe dw us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
file write `cf' "3pairwise_fq_gq_ig," ///
    (strtrim(strofreal(_b[us_cn_shock], "%14.6e"))) "," ///
    (strtrim(strofreal(_se[us_cn_shock], "%14.6e"))) "," ///
    (strtrim(strofreal(2*ttail(e(df_r), abs(_b[us_cn_shock]/_se[us_cn_shock])), "%9.6f"))) "," ///
    (strtrim(strofreal(e(N), "%15.0f"))) _n
qui reghdfe dw us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
file write `cf' "itgt_fq_gq," ///
    (strtrim(strofreal(_b[us_cn_shock], "%14.6e"))) "," ///
    (strtrim(strofreal(_se[us_cn_shock], "%14.6e"))) "," ///
    (strtrim(strofreal(2*ttail(e(df_r), abs(_b[us_cn_shock]/_se[us_cn_shock])), "%9.6f"))) "," ///
    (strtrim(strofreal(e(N), "%15.0f"))) _n
file close `cf'
display "Wrote `OUT'/headline_3pairwise_canonical.csv"

_row "F1a lead dw_t+1"  dw_lead1 "fq gq ig" ""
_row "F1a lead dw_t+1"  dw_lead1 "fq gq"    ""
_row "F1b LP cum1"      cum1     "fq gq ig" ""
_row "F1b LP cum1"      cum1     "fq gq"    ""
_row "F1b LP cum2"      cum2     "fq gq ig" ""
_row "F1b LP cum4"      cum4     "fq gq ig" ""
_row "F1b LP cum4"      cum4     "fq gq"    ""
_row "F2 in-span dw"    dw       "fq gq ig" "if in_span==1"
_row "F2 in-span dw"    dw       "fq gq"    "if in_span==1"

display _newline "===== F7 GPR two-interaction (3-pairwise vs it+gt) ====="
foreach FE in "fq gq ig" "fq gq" {
    qui reghdfe dw us_cn us_cn_gpr us_cn_gprlag, absorb(`FE') vce(cluster firm_n rd_m)
    display "`FE': b(GPR_t)=" %9.3e _b[us_cn_gpr] " p=" %6.4f 2*ttail(e(df_r),abs(_b[us_cn_gpr]/_se[us_cn_gpr])) ///
            "  b(GPR_t-1)=" %9.3e _b[us_cn_gprlag] " p=" %6.4f 2*ttail(e(df_r),abs(_b[us_cn_gprlag]/_se[us_cn_gprlag]))
}

*=======================================================================
* PART B — ownership FLOW (F6)
*=======================================================================
use "`OUT'/ownership_c6_panel.dta", clear
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
gen us_cn       = us*cn_lag
gen us_cn_shock = us*cn_lag*shock
display _newline "===== ownership FLOW (F6), 3-pairwise vs it+gt ====="
_row "flow"  flow  "fq gq ig" ""
_row "flow"  flow  "fq gq"    ""
display _newline "Done."
