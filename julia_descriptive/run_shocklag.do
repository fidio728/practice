* run_shocklag.do — advisor 2026-06-28 meeting point (a): shock should ALSO be
* lagged, so both CN and S are measured as of t-1 (predetermined relative to
* the Delta w_t outcome), not mixing a lagged CN with a contemporaneous shock.
* Compares Delta w_t ~ US x CN(t-1) x S_t  (current headline)
*      vs   Delta w_t ~ US x CN(t-1) x S_(t-1)  (advisor's spec)
* under both it+gt and the 3-pairwise it+gt+ig FE.
* Also reports the SD-standardized coefficient (advisor point (b)) -- a pure
* linear rescaling, does not change t/p, only the "per 1-SD shock" unit.

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
use "`OUT'/shocklag_panel.dta", clear

gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)

gen us_cn        = us * cn_lag
gen us_cn_shockt   = us * cn_lag * shock_t
gen us_cn_shocktm1 = us * cn_lag * shock_tm1
label var us_cn         "US x CN(t-1)"
label var us_cn_shockt   "US x CN(t-1) x S_t (current headline)"
label var us_cn_shocktm1 "US x CN(t-1) x S_(t-1) (advisor spec)"

* sigma of shock over the 82 estimation-sample quarters (both should match --
* verify shock_t and shock_tm1 have the same population std since it's the
* same series shifted by one quarter within an 82-quarter contiguous window)
egen qtag = tag(rd_m)
summarize shock_t if qtag==1, detail
local sd_t = r(sd)
summarize shock_tm1 if qtag==1, detail
local sd_tm1 = r(sd)
display _newline "sigma(shock_t) = " %9.4f `sd_t' "   sigma(shock_t-1) = " %9.4f `sd_tm1'

display _newline "===== CHECK: reproduce headline S_t, it+gt ====="
reghdfe dw us_cn us_cn_shockt, absorb(fq gq) vce(cluster firm_n rd_m)
display "  b3=" %9.3e _b[us_cn_shockt] "  p=" %6.4f 2*ttail(e(df_r),abs(_b[us_cn_shockt]/_se[us_cn_shockt]))

display _newline "===== CHECK: reproduce headline S_t, 3-pairwise ====="
reghdfe dw us_cn us_cn_shockt, absorb(fq gq ig) vce(cluster firm_n rd_m)
display "  b3=" %9.3e _b[us_cn_shockt] "  p=" %6.4f 2*ttail(e(df_r),abs(_b[us_cn_shockt]/_se[us_cn_shockt]))

display _newline "===== ADVISOR SPEC: S_(t-1), it+gt ====="
reghdfe dw us_cn us_cn_shocktm1, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store lag_itgt
display "  b3=" %9.3e _b[us_cn_shocktm1] "  se=" %9.3e _se[us_cn_shocktm1] "  p=" %6.4f 2*ttail(e(df_r),abs(_b[us_cn_shocktm1]/_se[us_cn_shocktm1]))
display "  b3 per 1-SD shock = " %9.3e _b[us_cn_shocktm1]*`sd_tm1'

display _newline "===== ADVISOR SPEC: S_(t-1), 3-pairwise ====="
reghdfe dw us_cn us_cn_shocktm1, absorb(fq gq ig) vce(cluster firm_n rd_m)
estimates store lag_3pw
display "  b3=" %9.3e _b[us_cn_shocktm1] "  se=" %9.3e _se[us_cn_shocktm1] "  p=" %6.4f 2*ttail(e(df_r),abs(_b[us_cn_shocktm1]/_se[us_cn_shocktm1]))
display "  b3 per 1-SD shock = " %9.3e _b[us_cn_shocktm1]*`sd_tm1'

capture which esttab
if _rc == 0 {
    esttab lag_itgt lag_3pw using "`OUT'/shocklag_results.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) keep(us_cn us_cn_shocktm1) ///
        mtitles("Stm1_itgt" "Stm1_3pairwise") nonumbers plain
    di "Wrote `OUT'/shocklag_results.csv"
}
display _newline "Done."
