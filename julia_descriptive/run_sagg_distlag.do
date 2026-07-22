* run_sagg_distlag.do — within-quarter AGGREGATED shock + distributed-lag joint test.
* S_t^agg = sum of the quarter's three monthly AR(1) innovations (aligns the
* shock's information window with Delta_w's full-quarter accrual window; the
* quarter-end-only stamp discards months 1-2, attenuating beta_3).
* Specs (traditional DDD, two-way cluster firm/quarter):
*   D1: h=0 only   dw ~ US x CN x S_t^agg
*   D2: h=1 only   dw ~ US x CN x S_{t-1}^agg
*   D3: BOTH       + joint Wald ("response at ANY horizon?") + cumulative sum
* Each under 3-pairwise (fq gq ig, headline) and 2-way (fq gq) FE.

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
use "`OUT'/sagg_panel.dta", clear

gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)

gen us_cn        = us * cn_lag
gen us_cn_sagg   = us * cn_lag * s_agg
gen us_cn_saggl1 = us * cn_lag * s_agg_l1
label var us_cn_sagg   "US x CN x S_t^agg (h=0, aligned)"
label var us_cn_saggl1 "US x CN x S_{t-1}^agg (h=1)"

foreach FE in "fq gq ig" "fq gq" {
    display _newline _newline "############ FE = `FE' ############"

    display _newline "=== D1: h=0 only (S_t^agg) ==="
    reghdfe dw us_cn us_cn_sagg, absorb(`FE') vce(cluster firm_n rd_m)
    display "  b3(h0)=" %9.3e _b[us_cn_sagg] "  se=" %9.3e _se[us_cn_sagg] ///
            "  p=" %6.4f 2*ttail(e(df_r),abs(_b[us_cn_sagg]/_se[us_cn_sagg]))

    display _newline "=== D2: h=1 only (S_{t-1}^agg) ==="
    reghdfe dw us_cn us_cn_saggl1, absorb(`FE') vce(cluster firm_n rd_m)
    display "  b3(h1)=" %9.3e _b[us_cn_saggl1] "  se=" %9.3e _se[us_cn_saggl1] ///
            "  p=" %6.4f 2*ttail(e(df_r),abs(_b[us_cn_saggl1]/_se[us_cn_saggl1]))

    display _newline "=== D3: distributed lag (both) + joint Wald + cumulative ==="
    reghdfe dw us_cn us_cn_sagg us_cn_saggl1, absorb(`FE') vce(cluster firm_n rd_m)
    display "  b3(h0)=" %9.3e _b[us_cn_sagg]   "  p=" %6.4f 2*ttail(e(df_r),abs(_b[us_cn_sagg]/_se[us_cn_sagg]))
    display "  b3(h1)=" %9.3e _b[us_cn_saggl1] "  p=" %6.4f 2*ttail(e(df_r),abs(_b[us_cn_saggl1]/_se[us_cn_saggl1]))
    test us_cn_sagg us_cn_saggl1
    display "  JOINT Wald (any horizon): F=" %6.3f r(F) "  p=" %6.4f r(p)
    lincom us_cn_sagg + us_cn_saggl1
    display "  CUMULATIVE (h0+h1): b=" %9.3e r(estimate) "  se=" %9.3e r(se) ///
            "  p=" %6.4f 2*ttail(r(df), abs(r(estimate)/r(se)))
}
display _newline "Done."
