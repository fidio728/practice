* run_country_ladder.do — LADDER STEP 1 (advisor-confirmed): simplest country-level
* regression. Unit (country c, group g, quarter t). Outcome w_{c,g,t} = share of
* group g's European book in country c (LEVEL, per the confirmed email; delta as check).
* Regressor of interest: US_g x CN_{c,t-1}. Control: bilateral US-country relations
* (BIL_{c,t} = quarterly mean of the monthly USA|c AI-GPR level, all 16 covered
* countries) and its US interaction -- controls for general US-European bilateral
* frictions (advisor example: US-Denmark/Greenland) so they are not misread as
* China avoidance. Main sample: 16 countries with bilateral data.

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
use "`OUT'/country_panel.dta", clear

gen rd_day = dofc(rdate)
gen rd_q   = qofd(rd_day)
format rd_q %tq
egen ctry  = group(sec_country)
egen cg    = group(sec_country us)
egen gq    = group(us rd_q)
egen cq    = group(sec_country rd_q)

gen us_cn  = us * cn_c_lag
gen us_bil = us * bil_us_c
label var us_cn  "US x CN_{c,t-1}"
label var us_bil "US x BIL_{c,t} (US-country relations)"

* ------- main sample: countries with bilateral data -------
preserve
keep if !missing(bil_us_c)

display _newline "===== M1: simplest (country FE + quarter FE), LEVEL w ====="
reghdfe w_c us cn_c_lag us_cn bil_us_c us_bil, absorb(ctry rd_q) vce(cluster ctry rd_q)

display _newline "===== M2: + country x group FE (absorbs US level + structural tilts) ====="
* FIX 2026-07-06: cn_c_lag (CN level) added -- it varies over time within a
* country x group cell, so cg FE does NOT absorb it; omitting it lets us_cn
* pick up the common CN effect (firm ladder's M2 always included cn_lag).
reghdfe w_c cn_c_lag us_cn bil_us_c us_bil, absorb(cg rd_q) vce(cluster ctry rd_q)

display _newline "===== M3: country x quarter + group x quarter FE (KM-style, mirrors firm design) ====="
reghdfe w_c us_cn us_bil, absorb(cq gq) vce(cluster ctry rd_q)

display _newline "===== M4: 3-pairwise (cq gq cg), mirrors firm ladder M4 ====="
reghdfe w_c us_cn us_bil, absorb(cq gq cg) vce(cluster ctry rd_q)

display _newline "===== M2-delta: outcome = change in country weight ====="
reghdfe dw_c cn_c_lag us_cn bil_us_c us_bil, absorb(cg rd_q) vce(cluster ctry rd_q)
restore

* ------- full 28 countries, no bilateral control (coverage check) -------
display _newline "===== M2-full28: no BIL control, all countries ====="
reghdfe w_c cn_c_lag us_cn, absorb(cg rd_q) vce(cluster ctry rd_q)

display _newline "Done. NOTE: only 16 (28) country clusters — treat p-values as descriptive."
