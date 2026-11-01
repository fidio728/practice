clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
use "/tmp/placebo2.dta", clear
rename report_date rdate
tempfile pl
qui save `pl', replace
use "`OUT'/audit_c6_panel.dta", clear
qui merge m:1 firm_str rdate using `pl', keep(master match) nogen
foreach k in CN GB IE NO LU DE SE FR AU ES IT DK BE AT FI IN {
    qui replace x`k' = dw if hgroup=="US"
}
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
gen us_cn      = us*cn_lag
gen us_cn_slag = us*cn_lag*s_lag
qui reghdfe dw us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)
local sb = _se[us_cn_slag]
di "PL base  se=" %12.4e `sb' "  ratio=1.000"
foreach k in CN GB IE NO LU DE SE FR AU ES IT DK BE AT FI IN {
    qui reghdfe x`k' us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)
    di "PL `k'  b=" %12.4e _b[us_cn_slag] "  se=" %12.4e _se[us_cn_slag] "  ratio=" %6.3f _se[us_cn_slag]/`sb'
}
di "DONE_PL2"
