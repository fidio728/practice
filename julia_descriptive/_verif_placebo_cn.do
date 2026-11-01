clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
use "/tmp/placebo_dw.dta", clear
rename report_date rdate
tempfile pl
qui save `pl', replace

use "`OUT'/audit_c6_panel.dta", clear
merge m:1 firm_str rdate using `pl', keep(master match)
tab _merge
drop _merge
foreach k in cn ch nl in jp ca cnhk {
    replace dw_no`k' = dw if hgroup=="US"
}
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
gen us_cn      = us*cn_lag
gen us_cn_slag = us*cn_lag*s_lag

di _newline "=== VERIF: baseline ==="
reghdfe dw us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)
di "LBL=baseline b3=" %14.6e _b[us_cn_slag] " se=" %14.6e _se[us_cn_slag] " p=" %8.6f 2*ttail(e(df_r),abs(_b[us_cn_slag]/_se[us_cn_slag])) " N=" e(N)

foreach k in cn ch nl in jp ca cnhk {
    di _newline "=== VERIF: drop `k' from NONUS ==="
    reghdfe dw_no`k' us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)
    di "LBL=no`k' b3=" %14.6e _b[us_cn_slag] " se=" %14.6e _se[us_cn_slag] " p=" %8.6f 2*ttail(e(df_r),abs(_b[us_cn_slag]/_se[us_cn_slag])) " N=" e(N)
}
di _newline "DONE_VERIF"
