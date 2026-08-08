* [VINTAGE WARNING — EU-era / S_t battery. Stamped by REBUILD v3, 2026-08-08]
* This file builds us_cn_shock-style S_t triples and reads dw straight off the
* rebuilt panels. Since 2026-08-08 the panel column dw is REPURPOSED to the
* GLOBAL full-portfolio-denominator outcome (EU-era dw lives in dw_eu) and the
* PRIMARY timing is S_{t-1} (s_lag). Re-running this file therefore estimates
* GLOBAL dw x S_t — neither the EU-era spec its comments/outputs describe nor
* the current primary. MIGRATE (s_lag + outcome relabel) before citing any new
* output; existing outputs on disk are EU-era/S_t vintage.

* run_ownership_share_verify.do - Re-run the ownership share regression
* This verifies the reported b3 result (b3 = +0.000599, se 0.001165, p=0.609)

clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ram?n Llull/git/practice/julia_descriptive/output/ownership_c6_panel.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ram?n Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

display _newline "=== Summary of ownership_c6_panel ==="
count
tab hgroup
summarize os dos cn_lag shock, detail

gen rd_day = dofc(rdate)
format rd_day %td
egen firm_n = group(firm_str)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen fq     = group(firm_str rd_day)
egen gq     = group(hgroup rd_day)
egen ig     = group(firm_str hgroup)

gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock
label var us_cn       "US x CN(t-1)"
label var us_cn_shock "US x CN(t-1) x S_t"

display _newline _newline "=== R1: headline (absorb fq gq), ownership share ==="
reghdfe dos us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store r1
local b3_r1 = _b[us_cn_shock]
local se3_r1 = _se[us_cn_shock]
local b2_r1 = _b[us_cn]
local p3_r1 = 2 * ttail(e(df_r), abs(`b3_r1'/`se3_r1'))

display _newline "R1 Results:"
display "  b3 (us_cn_shock): " `b3_r1'
display "  se3: " `se3_r1'
display "  p-value: " `p3_r1'
display "  N: " e(N)
display "  Firms: " e(N_clust1)
display "  Quarters: " e(N_clust2)

display _newline _newline "=== R2: + firm x group FE (absorb fq gq ig) ==="
reghdfe dos us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
estimates store r2
local b3_r2 = _b[us_cn_shock]
local se3_r2 = _se[us_cn_shock]
local p3_r2 = 2 * ttail(e(df_r), abs(`b3_r2'/`se3_r2'))

display _newline "R2 Results:"
display "  b3 (us_cn_shock): " `b3_r2'
display "  se3: " `se3_r2'
display "  p-value: " `p3_r2'
display "  N: " e(N)

display _newline _newline "=== Detailed coefficients table ==="
esttab r1 r2, b(%12.4e) se(%12.4e) p(%6.4f) ///
    keep(us_cn us_cn_shock) mtitles("R1_fqgq" "R2_firmXgroup")

capture which esttab
if _rc == 0 {
    esttab r1 r2 using "`OUT'/ownership_share_results_verify.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N, fmt(%9.0gc)) ///
        keep(us_cn us_cn_shock) mtitles("r1_os_fqgq" "r2_os_firmXgroup") nonumbers plain
    di "Wrote results to `OUT'/ownership_share_results_verify.csv"
}

display _newline _newline "=== REPORTED vs OBSERVED ==="
display "REPORTED (from paper):"
display "  R1: b3 = +0.000599, se = 0.001165, p = 0.609"
display "       b2 = -0.00108, p = 0.601"
display "       N = 240,246; firms = 6,854; quarters = 82"
display "OBSERVED (re-run):"
display "  R1: b3 = " `b3_r1' ", se = " `se3_r1' ", p = " `p3_r1'
display "Done."