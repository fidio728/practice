* run_flow_heldonly.do — B8 FIX (2026-07-22).
* Held-only zero-fill robustness on the FLOW outcome (the §7.6 primary outcome),
* replacing test_obs_only.do which ran the SUPERSEDED `dos` comparison outcome.
* Drops the extensive-margin zeros (group holds nothing: os==0), re-pairs to
* firm-quarters where BOTH US and NONUS are held, and re-runs the triple diff on
* `flow`. Panel = ownership_c6_panel.dta rebuilt on the B7-corrected grid, so
* cn_lag is the CUSTOMER+SUPPLIER china_share.
clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`OUT'/ownership_c6_panel.dta", clear

gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen  rd_m   = mofd(rd_day)
egen fq     = group(firm_str rd_day)
egen gq     = group(hgroup rd_day)
egen ig     = group(firm_str hgroup)
gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock

* B9 convention: no estimates store here, so VCE validity is checked from e()
* right after each reghdfe. A missing/non-positive SE on us_cn / us_cn_shock
* means the CRVE is degenerate for that spec; defer to the RI companion.
* Degeneracy is vintage-dependent (2026-08-03 rebuild: no held-only spec degenerate;
* the OLD dos spec in run_ownership_share.do is) — read the vce_diag CSV, never assume.
* The file handle and any_degen flag survive preserve/restore (locals + open file
* handles are not touched by restore), so held-only rows write to the same CSV.
tempname fh
file open `fh' using "`OUT'/flowheldonly_vce_diag.csv", write replace
file write `fh' "spec,coef,b,se,se_valid" _n
local any_degen 0

display _newline "=== FLOW full grid (reference): 3-pairwise (fq gq ig) ==="
reghdfe flow us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
display "full-grid flow b3 = " %12.4e _b[us_cn_shock] "  p = " %6.4f ///
        (2*ttail(e(df_r), abs(_b[us_cn_shock]/_se[us_cn_shock])))  "  N = " e(N)
foreach cf in us_cn us_cn_shock {
    local bb = _b[`cf']
    local ss = _se[`cf']
    local ok = (!missing(`ss') & `ss' > 0)
    file write `fh' "fullgrid_3pw,`cf',`bb',`ss',`ok'" _n
    if `ok' == 0 {
        local any_degen 1
        display as error ">>> fullgrid_3pw/`cf': DEGENERATE VCE — CRVE p INVALID; use run_ri_flow.py RI. <<<"
    }
}

* ---- Held-only: drop extensive-margin zeros (os==0 / missing), re-pair ----
preserve
keep if os > 0 & !missing(os)
bysort firm_str rd_day: gen _npair = _N
keep if _npair == 2          // re-pair: both US and NONUS still held
drop _npair fq gq ig firm_n
egen firm_n = group(firm_str)
egen fq     = group(firm_str rd_day)
egen gq     = group(hgroup rd_day)
egen ig     = group(firm_str hgroup)
count
display _newline "=== FLOW held-only (os>0, re-paired): absorb fq gq ==="
reghdfe flow us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
display "held-only flow b3 (fq gq) = " %12.4e _b[us_cn_shock] "  p = " %6.4f ///
        (2*ttail(e(df_r), abs(_b[us_cn_shock]/_se[us_cn_shock])))  "  N = " e(N)
foreach cf in us_cn us_cn_shock {
    local bb = _b[`cf']
    local ss = _se[`cf']
    local ok = (!missing(`ss') & `ss' > 0)
    file write `fh' "heldonly_fqgq,`cf',`bb',`ss',`ok'" _n
    if `ok' == 0 {
        local any_degen 1
        display as error ">>> heldonly_fqgq/`cf': DEGENERATE VCE — CRVE p INVALID; use run_ri_flow.py RI. <<<"
    }
}

display _newline "=== FLOW held-only: 3-pairwise (fq gq ig) ==="
reghdfe flow us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
display "held-only flow b3 (3-pw) = " %12.4e _b[us_cn_shock] "  p = " %6.4f ///
        (2*ttail(e(df_r), abs(_b[us_cn_shock]/_se[us_cn_shock])))  "  N = " e(N)
foreach cf in us_cn us_cn_shock {
    local bb = _b[`cf']
    local ss = _se[`cf']
    local ok = (!missing(`ss') & `ss' > 0)
    file write `fh' "heldonly_3pw,`cf',`bb',`ss',`ok'" _n
    if `ok' == 0 {
        local any_degen 1
        display as error ">>> heldonly_3pw/`cf': DEGENERATE VCE — CRVE p INVALID; use run_ri_flow.py RI. <<<"
    }
}
restore

file close `fh'
di "Wrote `OUT'/flowheldonly_vce_diag.csv"
if `any_degen' == 1 {
    display as error "Degenerate two-way-cluster VCE detected; the affected CRVE se/p above are NOT valid inference — use the RI companion (run_ri_flow.py)."
}
display _newline "Done."
