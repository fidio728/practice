clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`OUT'/c6_panel.dta", clear
merge 1:1 firm_str hgroup rdate using "`OUT'/_indepverify_cn_dw.dta", keep(master match)
tab _merge
drop _merge

gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
gen us_cn       = us*cn_lag
gen us_cn_slag  = us*cn_lag*s_lag
gen us_cn_shock = us*cn_lag*shock

* how different is the CN-removed outcome from the published one?
gen double d_abs = abs(dw_nocn - dw)
sum d_abs if hgroup=="NONUS", detail
corr dw dw_nocn if hgroup=="NONUS"

di _newline "=== IV1 BASELINE primary (c6_panel, dw, S_t-1) ==="
qui reghdfe dw us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)
di "IV1 b3=" %14.6e _b[us_cn_slag] " se=" %14.6e _se[us_cn_slag] " p=" %8.6f 2*ttail(e(df_r),abs(_b[us_cn_slag]/_se[us_cn_slag])) " N=" e(N)

di _newline "=== IV2 CN REMOVED from NONUS arm (numerator+denominator) ==="
qui reghdfe dw_nocn us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)
di "IV2 b3=" %14.6e _b[us_cn_slag] " se=" %14.6e _se[us_cn_slag] " p=" %8.6f 2*ttail(e(df_r),abs(_b[us_cn_slag]/_se[us_cn_slag])) " N=" e(N)

di _newline "=== IV3 CN+HK REMOVED ==="
qui reghdfe dw_nocnhk us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)
di "IV3 b3=" %14.6e _b[us_cn_slag] " se=" %14.6e _se[us_cn_slag] " p=" %8.6f 2*ttail(e(df_r),abs(_b[us_cn_slag]/_se[us_cn_slag])) " N=" e(N)

di _newline "=== IV4 S_t diagnostic, baseline vs CN-removed ==="
qui reghdfe dw us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
di "IV4a b3=" %14.6e _b[us_cn_shock] " se=" %14.6e _se[us_cn_shock] " p=" %8.6f 2*ttail(e(df_r),abs(_b[us_cn_shock]/_se[us_cn_shock])) " N=" e(N)
qui reghdfe dw_nocn us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
di "IV4b b3=" %14.6e _b[us_cn_shock] " se=" %14.6e _se[us_cn_shock] " p=" %8.6f 2*ttail(e(df_r),abs(_b[us_cn_shock]/_se[us_cn_shock])) " N=" e(N)

di _newline "IV_DONE"
