* [VINTAGE WARNING — EU-era / S_t battery. Stamped by REBUILD v3, 2026-08-08]
* This file builds us_cn_shock-style S_t triples and reads dw straight off the
* rebuilt panels. Since 2026-08-08 the panel column dw is REPURPOSED to the
* GLOBAL full-portfolio-denominator outcome (EU-era dw lives in dw_eu) and the
* PRIMARY timing is S_{t-1} (s_lag). Re-running this file therefore estimates
* GLOBAL dw x S_t — neither the EU-era spec its comments/outputs describe nor
* the current primary. MIGRATE (s_lag + outcome relabel) before citing any new
* output; existing outputs on disk are EU-era/S_t vintage.

* run_ownership_share.do — Essay 2 #5 STEP 3 regression.
* Shares-based outcome: dos = backward Delta(ownership_share), primary-EQ float.
* Same De Haas 4-coef triple difference as the w-based main spec, same FE + clustering.
*   dos = b2 (US x CN_{t-1}) + b3 (US x CN_{t-1} x S_t) + firm#quarter FE + group#quarter FE
* b3 (us_cn_shock) is the make-or-break: b3<0 => US reduce their STAKE more when tension rises.
* Compare to the w-based headline (null b3). Panel: ownership_c6_panel.dta.

clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/ownership_c6_panel.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

display _newline "=== ownership_c6_panel ==="
count
tab hgroup
summarize flow dos os, detail

gen rd_day = dofc(rdate)
format rd_day %td
egen firm_n = group(firm_str)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen fq     = group(firm_str rd_day)
egen gq     = group(hgroup rd_day)
egen ig     = group(firm_str hgroup)
count if missing(rd_day)
assert r(N) == 0

gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock
label var us_cn       "US x CN(t-1)"
label var us_cn_shock "US x CN(t-1) x S_t"

display _newline _newline "=== R1: PRIMARY outcome = FLOW (held_t-held_{t-1})/out_{t-1}, absorb fq gq ==="
reghdfe flow us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store r1

display _newline _newline "=== R2: FLOW + firm x group FE (absorb fq gq ig) ==="
reghdfe flow us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
estimates store r2

display _newline _newline "=== R3: OLD dos (share-of-float change, F6-confounded) for comparison ==="
reghdfe dos us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store r3

*==============================================================
* B9 convention: degenerate two-way-cluster VCE detection + diag CSV.
* A missing/non-positive SE on us_cn / us_cn_shock means the CRVE column is NOT
* valid inference for that spec; defer to the RI companion. Degeneracy is
* vintage-dependent: on the 2026-08-03 rebuild the fq-gq FLOW CRVE is valid
* (se_valid=1) and the OLD dos comparison spec is the degenerate one — read
* ownshare_vce_diag.csv for the current vintage, never assume which spec.
*==============================================================
tempname fh
file open `fh' using "`OUT'/ownshare_vce_diag.csv", write replace
file write `fh' "spec,coef,b,se,se_valid" _n
local any_degen 0
foreach m in r1 r2 r3 {
    estimates restore `m'
    foreach cf in us_cn us_cn_shock {
        local bb = _b[`cf']
        local ss = _se[`cf']
        local ok = (!missing(`ss') & `ss' > 0)
        file write `fh' "`m',`cf',`bb',`ss',`ok'" _n
        if `ok' == 0 {
            local any_degen 1
            display as error ">>> `m'/`cf': DEGENERATE VCE — CRVE p INVALID; use run_ri_flow.py RI. <<<"
        }
    }
}
file close `fh'
di "Wrote `OUT'/ownshare_vce_diag.csv"
if `any_degen' == 1 {
    display as error "Degenerate two-way-cluster VCE detected; affected CRVE columns in ownership_share_results.csv are NOT valid inference — use the RI companion (run_ri_flow.py)."
}

capture which esttab
if _rc == 0 {
    esttab r1 r2 r3 using "`OUT'/ownership_share_results.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) ///
        keep(us_cn us_cn_shock) mtitles("flow_fqgq" "flow_firmXgroup" "dos_compare") nonumbers plain
    di "Wrote `OUT'/ownership_share_results.csv"
}

display _newline "=== b2 / b3 (b / se / p) ==="
foreach m in r1 r2 r3 {
    estimates restore `m'
    display _newline "--- `m' ---"
    estimates table, b(%12.4e) se(%12.4e) p(%6.4f)
}
display _newline "Done."
