* [VINTAGE WARNING — EU-era / S_t battery. Stamped by REBUILD v3, 2026-08-08]
* This file builds us_cn_shock-style S_t triples and reads dw straight off the
* rebuilt panels. Since 2026-08-08 the panel column dw is REPURPOSED to the
* GLOBAL full-portfolio-denominator outcome (EU-era dw lives in dw_eu) and the
* PRIMARY timing is S_{t-1} (s_lag). Re-running this file therefore estimates
* GLOBAL dw x S_t — neither the EU-era spec its comments/outputs describe nor
* the current primary. MIGRATE (s_lag + outcome relabel) before citing any new
* output; existing outputs on disk are EU-era/S_t vintage.

* Test regression on observed-only, fully-paired subset
clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/ownership_c6_panel_obsonly_paired.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

display _newline "=== ownership_c6_panel OBSERVED-ONLY, FULLY-PAIRED ==="
count
tab hgroup
summarize os dos, detail

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

display _newline _newline "=== R1_OBS: observed-only (absorb fq gq) ==="
reghdfe dos us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store r1_obs

display _newline _newline "=== R2_OBS: observed-only + firm x group FE ==="
reghdfe dos us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
estimates store r2_obs

display _newline "=== Summary: obs-only vs full panel b3 (us_cn_shock) ==="
foreach m in r1_obs r2_obs {
    estimates restore `m'
    display _newline "--- `m' ---"
    estimates table, b(%12.4e) se(%12.4e) p(%6.4f)
}
display _newline "Done."
