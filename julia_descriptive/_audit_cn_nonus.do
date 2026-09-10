clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
use "`OUT'/audit_c6_panel.dta", clear
merge 1:1 firm_str hgroup rdate using "`OUT'/_audit_cn_in_nonus_dw.dta", keep(master match)
tab _merge
drop _merge
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
gen us_cn      = us*cn_lag
gen us_cn_slag = us*cn_lag*s_lag
gen us_cn_shock= us*cn_lag*shock

di _newline "===== BASELINE (published primary) ====="
reghdfe dw us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)
di "b3=" %14.6e _b[us_cn_slag] "  se=" %14.6e _se[us_cn_slag] "  p=" %8.6f 2*ttail(e(df_r),abs(_b[us_cn_slag]/_se[us_cn_slag])) "  N=" e(N)

di _newline "===== NONUS arm with CHINA investors REMOVED ====="
reghdfe dw_nocn us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)
di "b3=" %14.6e _b[us_cn_slag] "  se=" %14.6e _se[us_cn_slag] "  p=" %8.6f 2*ttail(e(df_r),abs(_b[us_cn_slag]/_se[us_cn_slag])) "  N=" e(N)

di _newline "===== NONUS arm with CHINA + HONG KONG investors REMOVED ====="
reghdfe dw_nocnhk us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)
di "b3=" %14.6e _b[us_cn_slag] "  se=" %14.6e _se[us_cn_slag] "  p=" %8.6f 2*ttail(e(df_r),abs(_b[us_cn_slag]/_se[us_cn_slag])) "  N=" e(N)

di _newline "===== S_t diagnostic cell, baseline vs CN-removed ====="
reghdfe dw us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
di "b3=" %14.6e _b[us_cn_shock] "  se=" %14.6e _se[us_cn_shock] "  p=" %8.6f 2*ttail(e(df_r),abs(_b[us_cn_shock]/_se[us_cn_shock])) "  N=" e(N)
reghdfe dw_nocn us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
di "b3=" %14.6e _b[us_cn_shock] "  se=" %14.6e _se[us_cn_shock] "  p=" %8.6f 2*ttail(e(df_r),abs(_b[us_cn_shock]/_se[us_cn_shock])) "  N=" e(N)
di _newline "DONE_AUDIT_CN"
