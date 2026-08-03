* 07e_firmgroup_tail.do
* SUPERSEDED by run_tercile_3pairwise.do (advisor 2026-08-02); retained for
*   provenance/few-cluster-invalidity (F3) only
* Two robustness blocks on the backward-diff C6 panel (c6_panel.dta):
*   (A) Add firm x group FE  mu_{i,g}  -- the "missing third pairwise" FE
*       (advisor's "another two-way interaction"). absorb(fq gq ig).
*   (B) Tail-dummy shock: replace continuous S_t with 1[z_t > k] one-sided
*       (escalation = right tail), k = 1.645 / 2 / 3. Report treated-quarter
*       counts at each k (power diagnostic).
*
* Headline reference: beta_3 = +1.28 (SE 1.66, p=0.443), N=462,564.
*
* Design notes / self-review:
*   - sigma_S computed over the 82 DISTINCT quarters (tag(rd_m)), NOT over the
*     462k rows, so it is not weighted by firms-per-quarter.
*   - one-sided right tail: H1 is about tension RISING, so ShockTail = 1 when
*     z_t > k (large unexpected escalation).
*   - firm x group FE (ig) is the third pairwise FE among {firm, group, quarter};
*     beta_2, beta_3 stay identified off within-(firm,group) time variation in
*     CN_{t-1} and S_t.
*   - Stata gotcha: dofc() before mofd() (pandas %tc). z>k guarded for missing.

clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/c6_panel.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

display _newline "=== Sample ==="
count

* keys
gen rd_day = dofc(rdate)
format rd_day %td
egen firm_n = group(firm_str)
egen fq     = group(firm_str rd_day)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen gq     = group(hgroup rd_day)
egen ig     = group(firm_str hgroup)      // NEW: firm x group

count if missing(rd_day)
assert r(N) == 0

* continuous-shock interactions
gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock
label var us_cn       "US x CN(t-1)"
label var us_cn_shock "US x CN(t-1) x S_t (continuous)"

* ============================================================
* (A) firm x group FE robustness
* ============================================================
display _newline _newline "=== A0: headline (absorb fq gq) -- reference ==="
reghdfe dw us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store base

display _newline _newline "=== A1: + firm x group FE (absorb fq gq ig) ==="
reghdfe dw us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
estimates store mig

* ============================================================
* (B) Tail-dummy shock
*     sigma over DISTINCT quarters; one-sided right (escalation)
* ============================================================
egen qtag = tag(rd_m)
quietly summarize shock if qtag==1
local smean = r(mean)
local ssd   = r(sd)
local nq    = r(N)
display _newline "=== Shock distribution over distinct quarters ==="
display "  n quarters = `nq'"
display "  mean(S_t)  = " %9.5f `smean'
display "  sd(S_t)    = " %9.5f `ssd'

gen z = (shock - `smean')/`ssd'

display _newline "=== Tail-dummy treated-quarter counts (one-sided right) ==="
foreach k in 1645 2000 3000 {
    local kk = `k'/1000
    gen byte tail`k' = (z > `kk') if !missing(z)
    quietly count if qtag==1 & tail`k'==1
    local ntq = r(N)
    quietly count if tail`k'==1
    local nrows = r(N)
    display "  k=`kk':  treated quarters = `ntq'  (of `nq')   treated rows = `nrows'"
    gen us_cn_tail`k' = us * cn_lag * tail`k'
    label var us_cn_tail`k' "US x CN(t-1) x 1[S_t > `kk' SD]"
}

display _newline _newline "=== B1: tail k=1.645 ==="
reghdfe dw us_cn us_cn_tail1645, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store t1645

display _newline _newline "=== B2: tail k=2.0 (headline tail) ==="
reghdfe dw us_cn us_cn_tail2000, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store t2000

display _newline _newline "=== B3: tail k=3.0 ==="
reghdfe dw us_cn us_cn_tail3000, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store t3000

* ============================================================
* Summary
* ============================================================
display _newline _newline "=== Summary: continuous + firm-group FE + tail menu ==="
capture which esttab
if _rc == 0 {
    esttab base mig t1645 t2000 t3000 using "`OUT'/07e_firmgroup_tail.txt", replace ///
        cells("b(fmt(%9.3e) star) se(fmt(%9.3e))") ///
        stats(N r2, fmt(%9.0gc %6.4f) labels("N" "R2")) ///
        keep(us_cn us_cn_shock us_cn_tail1645 us_cn_tail2000 us_cn_tail3000) ///
        order(us_cn us_cn_shock us_cn_tail1645 us_cn_tail2000 us_cn_tail3000) ///
        mtitles("base" "+firmXgroup" "tail k=1.645" "tail k=2" "tail k=3") ///
        title("Essay 2 robustness: firm-group FE and tail-dummy shock") ///
        note("Coefs x10^-6. Two-way cluster (firm, quarter). Tail one-sided right (escalation).")

    esttab base mig t1645 t2000 t3000 using "`OUT'/07e_firmgroup_tail.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) ///
        keep(us_cn us_cn_shock us_cn_tail1645 us_cn_tail2000 us_cn_tail3000) ///
        mtitles("base" "firmXgroup" "tail1645" "tail2000" "tail3000") ///
        nonumbers plain
    di "Wrote `OUT'/07e_firmgroup_tail.txt and .csv"
}

* on-screen lock check of each beta_3
display _newline "=== beta_3 across specs (b / se / p) ==="
foreach m in base mig t1645 t2000 t3000 {
    estimates restore `m'
    display _newline "--- spec: `m' ---"
    estimates table, b(%12.4e) se(%12.4e) p(%6.4f)
}

display _newline "Done."
