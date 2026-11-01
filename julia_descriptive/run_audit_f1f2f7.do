* [VINTAGE WARNING — EU-era / S_t battery. Stamped by REBUILD v3, 2026-08-08]
* This file builds us_cn_shock-style S_t triples and reads dw straight off the
* rebuilt panels. Since 2026-08-08 the panel column dw is REPURPOSED to the
* GLOBAL full-portfolio-denominator outcome (EU-era dw lives in dw_eu) and the
* PRIMARY timing is S_{t-1} (s_lag). Re-running this file therefore estimates
* GLOBAL dw x S_t — neither the EU-era spec its comments/outputs describe nor
* the current primary. MIGRATE (s_lag + outcome relabel) before citing any new
* output; existing outputs on disk are EU-era/S_t vintage.

* run_audit_f1f2f7.do — remediation regressions for review findings F1, F2, F7.
* Panel: audit_c6_panel.dta. Same FE (firm#quarter, group#quarter) + two-way
* cluster (firm, quarter) as the main spec. Coefficients are on Delta w (raw;
* multiply by 1e6 to match the doc's x10^-6 convention).

clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/audit_c6_panel.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

gen rd_day = dofc(rdate)
format rd_day %td
egen firm_n = group(firm_str)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)

* interactions
gen us_cn        = us * cn_lag
gen us_cn_shock  = us * cn_lag * shock
gen us_cn_gpr    = us * cn_lag * gpr
gen us_cn_gprlag = us * cn_lag * gpr_lag
label var us_cn       "US x CN(t-1)"
label var us_cn_shock "US x CN(t-1) x S_t"

*==============================================================
* CHECK: reproduce the headline (contemporaneous dw ~ S_t) on this panel
*==============================================================
display _newline "===== CHECK: headline dw ~ US x CN x S_t (full grid) ====="
reghdfe dw us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)

*==============================================================
* F1a: LEAD-FLOW — Delta w_{t+1} ~ US x CN x S_t  (pure lagged shock, no look-ahead)
*==============================================================
display _newline "===== F1a: LEAD-FLOW dw_lead1 ~ US x CN x S_t (full grid) ====="
reghdfe dw_lead1 us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store f1a_full

display _newline "===== F1a (in-span only) ====="
reghdfe dw_lead1 us_cn us_cn_shock if in_span==1, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store f1a_span

*==============================================================
* F1b: LOCAL PROJECTION IRF — cum_h = w_{t+h} - w_{t-1} ~ US x CN x S_t, h=0..4
*==============================================================
display _newline "===== F1b: LOCAL PROJECTION IRF (b3 by horizon) ====="
capture postclose lp
postfile lp int h double b3 double se3 double p3 double b2 double p2 double nobs ///
    using "`OUT'/audit_f1_lp_irf.dta", replace
forvalues h = 0/4 {
    quietly reghdfe cum`h' us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
    local b3 = _b[us_cn_shock]
    local se3 = _se[us_cn_shock]
    local t3 = `b3'/`se3'
    local p3 = 2*ttail(e(df_r), abs(`t3'))
    local b2 = _b[us_cn]
    local se2 = _se[us_cn]
    local p2 = 2*ttail(e(df_r), abs(`b2'/`se2'))
    post lp (`h') (`b3') (`se3') (`p3') (`b2') (`p2') (e(N))
    display "  h=`h'  b3=" %9.3e `b3' "  se=" %9.3e `se3' "  p=" %6.4f `p3' "  N=" %9.0gc e(N)
}
postclose lp

*==============================================================
* F2: HEADLINE restricted to firm existence span (drop ~25% phantom zeros)
*==============================================================
display _newline "===== F2: headline dw ~ US x CN x S_t, IN-SPAN ONLY ====="
reghdfe dw us_cn us_cn_shock if in_span==1, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store f2_span

*==============================================================
* F7: replace the AR(1) shock with GPR level + lag (no generated regressor / look-ahead)
*==============================================================
display _newline "===== F7: dw ~ US x CN x GPR_t + US x CN x GPR_{t-1} (full grid) ====="
reghdfe dw us_cn us_cn_gpr us_cn_gprlag, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store f7_full

display _newline "===== F7 (in-span) ====="
reghdfe dw us_cn us_cn_gpr us_cn_gprlag if in_span==1, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store f7_span

*==============================================================
* Export summary
*==============================================================
capture which esttab
if _rc == 0 {
    esttab f1a_full f1a_span f2_span f7_full f7_span using "`OUT'/audit_f1f2f7_results.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) ///
        mtitles("F1a_lead_full" "F1a_lead_span" "F2_dw_span" "F7_gpr_full" "F7_gpr_span") ///
        nonumbers plain
    di "Wrote `OUT'/audit_f1f2f7_results.csv"
}
display _newline "Done."
