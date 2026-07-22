* run_ddd_nofe_bil.do — Step-3 slide add-ons, gate-locked.
* STAGE A  gate: reproduce the locked 3-pairwise headline (07e):
*          beta3 = +1.80e-06 (se 1.91e-06), N = 462,096.
* STAGE B  PROPER no-FE column: full factorial with ALL lower-order terms
*          (us, cn_lag, shock, us_x_shock, cn_x_shock) + interactions.
*          The old 07d "No FE" cell omitted every main effect -> unusable.
* STAGE C  BIL footnote numbers: on the bilateral-covered subsample,
*          3-pairwise beta3 without vs with the US x BIL control.
*          (BIL LEVEL is absorbed by firm x quarter FE; only the US
*          differential is addable.)

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

* ---- BIL lookup from the (audited) firm ladder panel ----
use "`OUT'/firm_ladder_panel.dta", clear
keep firm_str rdate bil_us_c
duplicates drop firm_str rdate, force
tempfile bil
save `bil', replace

* ---- C6 zero-filled backward-diff panel (same build as 07d/07e) ----
use "`OUT'/c6_panel.dta", clear
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
egen fq     = group(firm_str rd_day)
gen  rd_m   = mofd(rd_day)
egen gq     = group(hgroup rd_day)
egen ig     = group(firm_str hgroup)

gen us_cn       = us * cn_lag
gen us_shock    = us * shock
gen cn_shock    = cn_lag * shock
gen us_cn_shock = us * cn_lag * shock

* ============ STAGE A: gate (3-pairwise headline must reproduce) ============
display _newline "===== STAGE A GATE: 3-pairwise headline ====="
reghdfe dw us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
local b3 = _b[us_cn_shock]
display "gate beta3 = " %12.4e `b3' "   target 1.80e-06   N=" e(N)
if abs(`b3' - 1.80e-06) > 0.05e-06 | e(N) != 462096 {
    display as error "GATE FAILED — stopping."
    exit 459
}
display "GATE PASS"

* ============ STAGE B: PROPER no-FE column (full factorial) ============
display _newline "===== STAGE B: no-FE, ALL lower-order terms in ====="
reghdfe dw us cn_lag shock us_cn us_shock cn_shock us_cn_shock, ///
    noabsorb vce(cluster firm_n rd_m)

* ============ STAGE C: bilateral subsample, w/o and w/ US x BIL ============
merge m:1 firm_str rdate using `bil', keep(master match) gen(_mb)
gen us_bil = us * bil_us_c

display _newline "===== STAGE C1: 3-pairwise on bilateral subsample (no BIL ctrl) ====="
reghdfe dw us_cn us_cn_shock if !missing(bil_us_c), ///
    absorb(fq gq ig) vce(cluster firm_n rd_m)

display _newline "===== STAGE C2: + US x BIL control ====="
reghdfe dw us_cn us_cn_shock us_bil if !missing(bil_us_c), ///
    absorb(fq gq ig) vce(cluster firm_n rd_m)

display _newline "Done — run_ddd_nofe_bil.do"
