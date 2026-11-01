* run_firm_ladder.do — LADDER STEP 2 (advisor-confirmed): the step-1 country
* specification with the FIRM dimension added. No shock yet (step 3 = triple diff).
* Outcome: w_{i,g,t} LEVEL (zero-filled firm share of group g's European book);
* Delta-w as check. Regressor: US_g x CN_{i,t-1} (firm-level exposure). Control:
* US_g x BIL_{c(i),t} (bilateral US-listing-country relations, 16 countries).
* Main sample: rows with bilateral data (~93%). Cluster: two-way (firm, quarter).

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
use "`OUT'/firm_ladder_panel.dta", clear

gen rd_day = dofc(rdate)
gen rd_m   = mofd(rd_day)
egen firm_n = group(firm_str)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)

gen us_cn  = us * cn_lag
gen us_bil = us * bil_us_c
label var us_cn  "US x CN_{i,t-1}"
label var us_bil "US x BIL_{c(i),t}"

preserve
keep if !missing(bil_us_c)

display _newline "===== M1: simplest (firm FE + quarter FE), LEVEL w ====="
reghdfe w us cn_lag us_cn bil_us_c us_bil, absorb(firm_n rd_m) vce(cluster firm_n rd_m)

display _newline "===== M2: firm x group FE + quarter FE ====="
reghdfe w cn_lag us_cn bil_us_c us_bil, absorb(ig rd_m) vce(cluster firm_n rd_m)

display _newline "===== M3: firm x quarter + group x quarter FE (mirrors headline, no shock) ====="
reghdfe w us_cn us_bil, absorb(fq gq) vce(cluster firm_n rd_m)

display _newline "===== M4: 3-pairwise (fq gq ig) ====="
reghdfe w us_cn us_bil, absorb(fq gq ig) vce(cluster firm_n rd_m)

display _newline "===== M4-delta: outcome = Delta w ====="
reghdfe dw us_cn us_bil, absorb(fq gq ig) vce(cluster firm_n rd_m)
restore

display _newline "===== M4-full: all countries, no BIL control ====="
reghdfe w us_cn, absorb(fq gq ig) vce(cluster firm_n rd_m)

display _newline "Done."
