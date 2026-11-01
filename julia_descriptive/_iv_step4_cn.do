clear all
set more off
set seed 20260810
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
use "`OUT'/country_panel_step4.dta", clear
gen rd_day = dofc(rdate)
gen q = qofd(rd_day)
format q %tq
tempfile base
qui save `base', replace

use "`OUT'/_indepverify_step4_nocn.dta", clear
gen rd_day = dofc(rdate)
gen q = qofd(rd_day)
keep sec_country q dlh_diff_nocn dlh_diff_nocnhk
tempfile addon
qui save `addon', replace

use `base', clear
merge 1:1 sec_country q using `addon', keep(master match)
tab _merge
drop _merge

gen rd_q = q
egen ctry = group(sec_country)
gen double sxm1 = s_lag*m1_lag
gen double sxm2 = s_lag*m2_lag
gen double sxm3 = s_lag*m3_lag

foreach m of numlist 1/3 {
  di _newline "=== M`m' BLOCK B baseline (dlh_diff) ==="
  qui reghdfe dlh_diff sxm`m' m`m'_lag, absorb(ctry rd_q) vce(cluster ctry)
  di "S4 M`m' base   b=" %12.6f _b[sxm`m'] " se=" %10.6f _se[sxm`m'] " p=" %7.5f 2*ttail(e(df_r),abs(_b[sxm`m']/_se[sxm`m'])) " N=" e(N) " G=" e(N_clust)
  di "=== M`m' BLOCK B CN removed from NONUS ==="
  qui reghdfe dlh_diff_nocn sxm`m' m`m'_lag, absorb(ctry rd_q) vce(cluster ctry)
  di "S4 M`m' noCN   b=" %12.6f _b[sxm`m'] " se=" %10.6f _se[sxm`m'] " p=" %7.5f 2*ttail(e(df_r),abs(_b[sxm`m']/_se[sxm`m'])) " N=" e(N) " G=" e(N_clust)
  di "=== M`m' BLOCK B CN+HK removed ==="
  qui reghdfe dlh_diff_nocnhk sxm`m' m`m'_lag, absorb(ctry rd_q) vce(cluster ctry)
  di "S4 M`m' noCNHK b=" %12.6f _b[sxm`m'] " se=" %10.6f _se[sxm`m'] " p=" %7.5f 2*ttail(e(df_r),abs(_b[sxm`m']/_se[sxm`m'])) " N=" e(N) " G=" e(N_clust)
}
di _newline "S4_IV_DONE"
