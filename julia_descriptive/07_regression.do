* 07_regression.do
* Headline triple-difference β_3 on the C6 zero-filled panel.
*
* Spec:  Δw_{b,i,c,t} = β_2·US_c × CN_{i,t-1} + β_3·US_c × CN_{i,t-1} × Shock_t
*                    + α_{i,t} + α_{c×t} + ε
*
* All other De Haas 4-coefficient terms are absorbed by the high-dim FE:
*   - β_0·US_c            absorbed by α_{c×t}  (US_c constant within c)
*   - β_1·US_c × Shock_t   absorbed by α_{c×t}  (Shock_t constant within t)
*   - CN_{i,t-1}           absorbed by α_{i,t}  (CN_lag constant within (i,t))
*   - CN_{i,t-1} × Shock_t absorbed by α_{i,t}  (Shock_t constant within t)
*
* Controls: no separate control variables are added — every firm-quarter level
* and holder-group-quarter level control is absorbed by α_{i,t} + α_{c×t}.

clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/c6_panel.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

display _newline "=== Sample composition ==="
count
display _newline "Holder group:"
tab hgroup
display _newline "Quarters covered:"
sum rdate, format

* Numeric IDs for FE / clustering
* pandas to_stata writes datetime64 as %tc (milliseconds); convert to daily
* with dofc() before extracting month/quarter.
gen rd_day = dofc(rdate)
format rd_day %td

egen firm_n   = group(firm_str)
egen fq       = group(firm_str rd_day)
gen  rd_m     = mofd(rd_day)
format rd_m %tm
egen gq       = group(hgroup rd_day)

* Sanity: rd_day must not be all missing
count if missing(rd_day)
assert r(N) == 0

* Pre-compute interaction terms explicitly so reghdfe doesn't auto-drop
gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock

* Label for output
label var us_cn       "US × CN(t-1)"
label var us_cn_shock "US × CN(t-1) × Shock"

* ============================================================
* Spec 1: minimal identified pair (β_2 + β_3) with full FE stack
*         α_{i,t} + α_{c×t}
* ============================================================
display _newline _newline "=== Spec 1: β_2 + β_3, FE = (firm × quarter) + (hgroup × quarter) ==="
reghdfe dw us_cn us_cn_shock, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store m1

* ============================================================
* Spec 2: triple only (no β_2) — does β_3 survive without γ?
* ============================================================
display _newline _newline "=== Spec 2: β_3 only, FE = (firm × quarter) + (hgroup × quarter) ==="
reghdfe dw us_cn_shock, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store m2

* ============================================================
* Spec 3: full De Haas 4-coefficient form (let reghdfe drop absorbed terms)
* ============================================================
display _newline _newline "=== Spec 3: full c.us##c.cn_lag##c.shock with all FE ==="
reghdfe dw c.us##c.cn_lag##c.shock, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store m3

* ============================================================
* Spec 4: weaker FE (separate firm + quarter, no interactions) — just for
*         comparison to show how much β_3 changes when FE saturate.
* ============================================================
display _newline _newline "=== Spec 4: weaker FE = firm + quarter (no interactions) ==="
reghdfe dw us_cn us_cn_shock, ///
    absorb(firm_n rd_m) ///
    vce(cluster firm_n rd_m)
estimates store m4

* ============================================================
* Summary table
* ============================================================
display _newline _newline "=== Summary table ==="

* Use esttab if available; otherwise fall back to estimates table
capture which esttab
if _rc == 0 {
    esttab m4 m3 m1 m2 using "`OUT'/07_reghdfe_results.txt", replace ///
        cells("b(fmt(6) star) se(fmt(6))") ///
        stats(N r2 r2_a, fmt(%9.0gc %6.4f %6.4f) labels("N" "R²" "Adj R²")) ///
        keep(us_cn us_cn_shock) ///
        title("β_3 on C6 zero-filled panel — different FE specifications") ///
        mtitles("Sep firm+qtr" "Full saturation" "β_2 + β_3" "β_3 only") ///
        note("Cluster SE on (firm, quarter). C6 zero-filled panel, n ≈ 450k.")
    di "Wrote esttab output to `OUT'/07_reghdfe_results.txt"
}
else {
    estimates table m4 m3 m1 m2, b(%9.6f) se(%9.6f) ///
        stats(N r2 r2_a) keep(us_cn us_cn_shock)
}

* Always also print to log for capture
display _newline "=== FINAL HEADLINE β_3 estimate (Spec 1, recommended) ==="
estimates restore m1
estimates table, b(%9.6f) se(%9.6f) p(%6.4f)

display _newline "Done."
