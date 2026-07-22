* run_ladder_2x2_cells.do — add-on cells for the 2x2 (FE x controls) slide tables.
* Complements run_country_ladder.do / run_firm_ladder.do (audited): computes the
* missing cells (no-FE no-BIL, designFE no-BIL, no-FE with-BIL) on the SAME
* bilateral main sample so all columns are comparable. Existing cells reused:
*   country col4 = M3 (cq gq + us_bil)  = -0.059 (0.055)
*   firm    col4 = M3 (fq gq + us_bil)  = +8.6e-5** (4.3e-5)
*   firm    col5 = M4 (fq gq ig + bil)  = -2.1e-5 (4.5e-5)
* No-FE columns include the levels (us, cn, bil) directly — required lower-order
* terms when nothing absorbs them.

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

* ============================ COUNTRY =======================================
use "`OUT'/country_panel.dta", clear
gen rd_day = dofc(rdate)
gen rd_q   = qofd(rd_day)
format rd_q %tq
egen ctry  = group(sec_country)
egen gq    = group(us rd_q)
egen cq    = group(sec_country rd_q)
gen us_cn  = us * cn_c_lag
gen us_bil = us * bil_us_c
keep if !missing(bil_us_c)

display _newline "===== C-P1: no FE, no BIL (levels in) ====="
reghdfe w_c us cn_c_lag us_cn, noabsorb vce(cluster ctry rd_q)

display _newline "===== C-P2: design FE (cq gq), no BIL ====="
reghdfe w_c us_cn, absorb(cq gq) vce(cluster ctry rd_q)

display _newline "===== C-P3: no FE, with BIL ====="
reghdfe w_c us cn_c_lag us_cn bil_us_c us_bil, noabsorb vce(cluster ctry rd_q)

* ============================ FIRM ==========================================
use "`OUT'/firm_ladder_panel.dta", clear
gen rd_day = dofc(rdate)
gen rd_m   = mofd(rd_day)
egen firm_n = group(firm_str)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
gen us_cn  = us * cn_lag
gen us_bil = us * bil_us_c
keep if !missing(bil_us_c)

display _newline "===== F-P1: no FE, no BIL (levels in) ====="
reghdfe w us cn_lag us_cn, noabsorb vce(cluster firm_n rd_m)

display _newline "===== F-P2: design FE (fq gq), no BIL ====="
reghdfe w us_cn, absorb(fq gq) vce(cluster firm_n rd_m)

display _newline "===== F-P3: no FE, with BIL ====="
reghdfe w us cn_lag us_cn bil_us_c us_bil, noabsorb vce(cluster firm_n rd_m)

display _newline "Done — 2x2 add-on cells."
