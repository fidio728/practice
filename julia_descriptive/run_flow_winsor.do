* run_flow_winsor.do — winsorized-flow CRVE, to sit next to the RI diagnostic.
* flow is un-winsorized and fat-tailed (kurtosis ~5000); winsorize at p1/p99 and
* re-run the primary (fq gq) and 3-pairwise (fq gq ig) triple-diff on the
* B7-rebuilt panel.
clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
use "`OUT'/ownership_c6_panel.dta", clear

_pctile flow, p(1 99)
gen flow_w = flow
replace flow_w = r(r1) if flow < r(r1)
replace flow_w = r(r2) if flow > r(r2)

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
* Degeneracy is vintage-dependent (2026-08-03 rebuild: no winsor spec degenerate;
* the OLD dos spec in run_ownership_share.do is) — read the vce_diag CSV, never assume.
tempname fh
file open `fh' using "`OUT'/flowwinsor_vce_diag.csv", write replace
file write `fh' "spec,coef,b,se,se_valid" _n
local any_degen 0

display _newline "=== WINSORIZED flow, R1: absorb fq gq ==="
reghdfe flow_w us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
display "wins flow b3 (fq gq) = " %12.4e _b[us_cn_shock] "  se = " %12.4e _se[us_cn_shock]
foreach cf in us_cn us_cn_shock {
    local bb = _b[`cf']
    local ss = _se[`cf']
    local ok = (!missing(`ss') & `ss' > 0)
    file write `fh' "winsor_fqgq,`cf',`bb',`ss',`ok'" _n
    if `ok' == 0 {
        local any_degen 1
        display as error ">>> winsor_fqgq/`cf': DEGENERATE VCE — CRVE p INVALID; use run_ri_flow.py RI. <<<"
    }
}

display _newline "=== WINSORIZED flow, R2: 3-pairwise fq gq ig ==="
reghdfe flow_w us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
display "wins flow b3 (3-pw) = " %12.4e _b[us_cn_shock] "  se = " %12.4e _se[us_cn_shock] ///
        "  p = " %6.4f (2*ttail(e(df_r), abs(_b[us_cn_shock]/_se[us_cn_shock])))
foreach cf in us_cn us_cn_shock {
    local bb = _b[`cf']
    local ss = _se[`cf']
    local ok = (!missing(`ss') & `ss' > 0)
    file write `fh' "winsor_3pw,`cf',`bb',`ss',`ok'" _n
    if `ok' == 0 {
        local any_degen 1
        display as error ">>> winsor_3pw/`cf': DEGENERATE VCE — CRVE p INVALID; use run_ri_flow.py RI. <<<"
    }
}

file close `fh'
di "Wrote `OUT'/flowwinsor_vce_diag.csv"
if `any_degen' == 1 {
    display as error "Degenerate two-way-cluster VCE detected; the affected CRVE se/p above are NOT valid inference — use the RI companion (run_ri_flow.py)."
}
display _newline "Done."
