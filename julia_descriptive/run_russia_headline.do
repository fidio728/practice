* run_russia_headline.do — Russia positive control (Third review round, R3-A #1 / P0 #3).
* Isomorphic triple-difference on Russia supply-chain exposure instead of China:
*   dw = b2 (US x RU(t-1)) + b3 (US x RU(t-1) x S^{US-RU}_t) + it + gt (+ ig)
* Same panel construction, same FE, same clustering as the China headline
* (run_headline_3pairwise.do). If the design can detect the known 2022
* Russia divestment (b3 << 0), the China null becomes informative rather than
* merely underpowered.
*
* B9 FIX (2026-07-22): the R1 (it+gt) two-way-cluster CGM VCE is degenerate here
* (non-PSD / singular -> reghdfe returns a MISSING SE) because the Russia event
* concentrates on very few treated quarters. The point estimate is still valid,
* but the CRVE p is NOT. Previously the degenerate column was written to the
* results CSV with se='.' and no flag. We now (a) detect a missing/zero SE after
* every spec, (b) warn loudly, and (c) write russia_headline_vce_diag.csv marking
* each spec's VCE validity, so a degenerate column can never be read as a valid
* inference. The valid design-based inference for these specs is the permutation
* test in run_ri_russia.py.

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`OUT'/c6_panel_russia.dta", clear

display _newline "=== c6_panel_russia ==="
count
tab hgroup
summarize dw ru_lag shock, detail

gen rd_day = dofc(rdate)
format rd_day %td
egen firm_n = group(firm_str)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
count if missing(rd_day)
assert r(N) == 0

gen us_ru       = us * ru_lag
gen us_ru_shock = us * ru_lag * shock
label var us_ru       "US x RU(t-1)"
label var us_ru_shock "US x RU(t-1) x S^{US-RU}_t"

display _newline _newline "=== R1: headline it+gt ==="
reghdfe dw us_ru us_ru_shock, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store r1

display _newline _newline "=== R2: 3-pairwise it+gt+ig ==="
reghdfe dw us_ru us_ru_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
estimates store r2

*==============================================================
* 2022Q1-Q2 event-window check: was there a DISCRETE break in the
* US-NONUS Russia-exposure gap around the invasion (Feb 2022), independent of
* the continuous AR(1) shock? Simple dummy for the two invasion quarters.
*==============================================================
gen post_invasion = (rd_m >= tm(2022m1) & rd_m <= tm(2022m6))
gen us_ru_post = us * ru_lag * post_invasion
label var us_ru_post "US x RU(t-1) x post-invasion(22Q1-Q2)"

display _newline _newline "=== R3: event-window (2022 Q1-Q2) dummy instead of continuous shock, it+gt ==="
reghdfe dw us_ru us_ru_post, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store r3

* --- B9 FIX: detect degenerate VCE (missing/zero SE) and record it, so no
*     column is ever silently read as a valid CRVE inference. ---
tempname fh
file open `fh' using "`OUT'/russia_headline_vce_diag.csv", write replace
file write `fh' "spec,coef,b,se,se_valid" _n
local any_degen 0
foreach m in r1 r2 r3 {
    estimates restore `m'
    local cf "us_ru_shock"
    if "`m'" == "r3" {
        local cf "us_ru_post"
    }
    local bb = _b[`cf']
    local ss = _se[`cf']
    local ok = (!missing(`ss') & `ss' > 0)
    file write `fh' "`m',`cf',`bb',`ss',`ok'" _n
    if `ok' == 0 {
        local any_degen 1
        display as error ">>> `m': DEGENERATE VCE (SE missing/zero) — CRVE p INVALID; use design-based RI (run_ri_russia.py). <<<"
    }
}
file close `fh'
di "Wrote `OUT'/russia_headline_vce_diag.csv"
if `any_degen' == 1 {
    display as error "One or more Russia headline specs had a degenerate two-way-cluster VCE. Their CRVE SE/p in russia_headline_results.csv are NOT valid inference; read the design-based RI (run_ri_russia.py) instead."
}

capture which esttab
if _rc == 0 {
    esttab r1 r2 r3 using "`OUT'/russia_headline_results.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) ///
        keep(us_ru us_ru_shock us_ru_post) ///
        mtitles("it_gt" "3pairwise" "event_2022") nonumbers plain ///
        addnote("VCE validity per spec in russia_headline_vce_diag.csv; degenerate columns -> use run_ri_russia.py")
    di "Wrote `OUT'/russia_headline_results.csv"
}

display _newline "=== b2 / b3 (b / se / p) ==="
foreach m in r1 r2 r3 {
    estimates restore `m'
    display _newline "--- `m' ---"
    estimates table, b(%12.4e) se(%12.4e) p(%6.4f)
}
display _newline "Done."
