* run_audit_f4f8.do — review fixes F8 (risk-set lag-only) and F4 (country-pair
* β₁ clustering level).

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

*==============================================================
* F8: risk-set with LAG-ONLY membership (no t+1 look-ahead). Compare β₃ to the
* with-lead risk set headline.
*==============================================================
use "`OUT'/c6_panel_riskset_lagonly.dta", clear
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen  rd_m   = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock
display _newline "===== F8: risk-set LAG-ONLY membership (dw ~ US x CN x S_t) ====="
reghdfe dw us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
display "  b3=" %9.3e _b[us_cn_shock] "  se=" %9.3e _se[us_cn_shock] ///
        "  p=" %6.4f 2*ttail(e(df_r), abs(_b[us_cn_shock]/_se[us_cn_shock]))
di "  (with-lead risk set 07g R1 b3 was ~ +1.28e-6-scale null; check stability)"

*==============================================================
* F4: country-pair β₁ (US x S_c) under three clustering levels. S_c varies only
* across GB/DE/FR x quarter, so (firm, quarter) clustering ignores within-country
* serial correlation with only ~3 country units.
*==============================================================
use "`OUT'/c6_panel_country_pair.dta", clear
drop if missing(shock_c)
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen  rd_m   = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ctry_n = group(sec_country)
egen ctryq  = group(sec_country rd_day)

gen us_sc      = us * shock_c
gen us_cn      = us * cn_lag
gen us_cn_sc   = us * cn_lag * shock_c
label var us_sc    "US x S_c (β1)"
label var us_cn    "US x CN (β2)"
label var us_cn_sc "US x CN x S_c (β3)"

tab sec_country
display _newline "===== F4a: country-pair, cluster(firm_n rd_m) [as in doc] ====="
reghdfe dw us_sc us_cn us_cn_sc, absorb(fq gq) vce(cluster firm_n rd_m)
display "  b1(US x S_c)=" %9.3e _b[us_sc] "  se=" %9.3e _se[us_sc] ///
        "  p=" %6.4f 2*ttail(e(df_r), abs(_b[us_sc]/_se[us_sc]))

display _newline "===== F4b: country-pair, cluster(sec_country) [3 clusters, honest level] ====="
reghdfe dw us_sc us_cn us_cn_sc, absorb(fq gq) vce(cluster ctry_n)
display "  b1(US x S_c)=" %9.3e _b[us_sc] "  se=" %9.3e _se[us_sc] ///
        "  p=" %6.4f 2*ttail(e(df_r), abs(_b[us_sc]/_se[us_sc])) "  (df_r=" e(df_r) ")"

display _newline "===== F4c: country-pair, cluster(sec_country rd_m) [country x quarter] ====="
reghdfe dw us_sc us_cn us_cn_sc, absorb(fq gq) vce(cluster ctry_n rd_m)
display "  b1(US x S_c)=" %9.3e _b[us_sc] "  se=" %9.3e _se[us_sc] ///
        "  p=" %6.4f 2*ttail(e(df_r), abs(_b[us_sc]/_se[us_sc]))
display _newline "Done."
