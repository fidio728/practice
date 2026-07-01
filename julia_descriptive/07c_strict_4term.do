* 07c_strict_4term.do
* Strict De Haas 4-term symmetric notation, slide form:
*
*   Delta_w_{i,g,t} = beta_0 * US_g
*                   + beta_1 * US_g * S_t
*                   + beta_2 * US_g * CN_{i,t-1}
*                   + beta_3 * US_g * CN_{i,t-1} * S_t
*                   + alpha_{i,t}  (firm x quarter)
*                   + gamma_{g,t}  (group x quarter)
*                   + eps_{i,g,t}
*
* Two specs, hand-built interaction terms, NO extra robustness columns.
*
* Spec M (MAIN, full panel, US-CN shock):
*   reghdfe dw us_lvl us_s us_cn us_cn_s, absorb(fq gq) vce(cluster firm_n rd_m)
*   Expected omissions:
*     us_lvl  -> omitted (absorbed by gamma_{g,t})
*     us_s    -> omitted (S_t has no host-country variation -> absorbed by gamma_{g,t})
*     us_cn   -> IDENTIFIED (beta_2)
*     us_cn_s -> IDENTIFIED (beta_3) HEADLINE
*
* Spec P (COUNTRY-PAIR, GB+DE+FR subsample, shock_c):
*   reghdfe dw us_lvl us_s_c us_cn us_cn_s_c, absorb(fq gq) vce(cluster firm_n rd_m)
*   Expected omissions:
*     us_lvl    -> omitted (absorbed by gamma_{g,t})
*     us_s_c    -> IDENTIFIED (beta_1) NEW vs Spec M
*     us_cn     -> IDENTIFIED (beta_2)
*     us_cn_s_c -> IDENTIFIED (beta_3) HEADLINE
*
* Stata gotchas (per project notes):
*   - pandas to_stata writes datetime64 as %tc (ms). MUST apply dofc() before mofd().
*   - In c6_panel.dta the shock column is named "shock". In
*     c6_panel_country_pair.dta the columns are "shock_us_cn" and "shock_c".

clear all
set more off

local DTA_M "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/c6_panel.dta"
local DTA_P "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/c6_panel_country_pair.dta"
local OUT   "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

* ============================================================
* SPEC M: MAIN (full panel, US-CN shock)
* ============================================================
use "`DTA_M'", clear

display _newline "=== Spec M: sample composition (full panel) ==="
count
display _newline "Holder group:"
tab hgroup
display _newline "Quarters covered:"
sum rdate, format

* Numeric IDs for FE / clustering.
* pandas to_stata writes datetime64 as %tc; dofc() before mofd().
gen rd_day = dofc(rdate)
format rd_day %td

egen firm_n = group(firm_str)
egen fq     = group(firm_str rd_day)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen gq     = group(hgroup rd_day)

* Sanity
count if missing(rd_day)
assert r(N) == 0

* Hand-built 4 interaction terms matching slide notation.
* In this file the AR(1) residual column is literally named "shock".
gen us_lvl  = us
gen us_s    = us * shock
gen us_cn   = us * cn_lag
gen us_cn_s = us * cn_lag * shock

label var us_lvl  "US_g (beta_0)"
label var us_s    "US_g x S_t (beta_1)"
label var us_cn   "US_g x CN(t-1) (beta_2)"
label var us_cn_s "US_g x CN(t-1) x S_t (beta_3)"

display _newline _newline "=== Spec M: reghdfe dw us_lvl us_s us_cn us_cn_s, absorb(fq gq) ==="
reghdfe dw us_lvl us_s us_cn us_cn_s, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store specM

* ============================================================
* SPEC P: COUNTRY-PAIR (GB+DE+FR subsample, shock_c)
* ============================================================
use "`DTA_P'", clear

display _newline "=== Spec P: sample composition (GB+DE+FR subsample) ==="
count
display _newline "Listing country (must be GB/DE/FR only):"
tab sec_country
display _newline "Holder group:"
tab hgroup
display _newline "Quarters covered:"
sum rdate, format

* Hard guard: only GB/DE/FR firms allowed
gen byte _bad_country = !inlist(sec_country, "GB", "DE", "FR")
count if _bad_country == 1
assert r(N) == 0
drop _bad_country

* Numeric IDs for FE / clustering (same recipe as Spec M).
gen rd_day = dofc(rdate)
format rd_day %td

egen firm_n = group(firm_str)
egen fq     = group(firm_str rd_day)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen gq     = group(hgroup rd_day)

count if missing(rd_day)
assert r(N) == 0

* Diagnostic on shock_c coverage
count if missing(shock_c)
display "rows with missing shock_c: " r(N)

* Hand-built 4 interaction terms matching slide notation,
* with shock replaced by country-pair shock_c.
gen us_lvl    = us
gen us_s_c    = us * shock_c
gen us_cn     = us * cn_lag
gen us_cn_s_c = us * cn_lag * shock_c

label var us_lvl    "US_g (beta_0)"
label var us_s_c    "US_g x S_{c,t} (beta_1)"
label var us_cn     "US_g x CN(t-1) (beta_2)"
label var us_cn_s_c "US_g x CN(t-1) x S_{c,t} (beta_3)"

display _newline _newline "=== Spec P: reghdfe dw us_lvl us_s_c us_cn us_cn_s_c, absorb(fq gq) ==="
reghdfe dw us_lvl us_s_c us_cn us_cn_s_c, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store specP

* ============================================================
* Side-by-side summary table (Spec M and Spec P)
* ============================================================
display _newline _newline "=== Side-by-side: Spec M | Spec P ==="

capture which esttab
if _rc == 0 {
    esttab specM specP using "`OUT'/07c_strict_4term_results.txt", replace ///
        cells("b(fmt(6) star) se(fmt(6))") ///
        stats(N r2 r2_a, fmt(%9.0gc %6.4f %6.4f) labels("N" "R2" "Adj R2")) ///
        keep(us_lvl us_s us_s_c us_cn us_cn_s us_cn_s_c) ///
        order(us_lvl us_s us_s_c us_cn us_cn_s us_cn_s_c) ///
        title("Strict 4-term form (slide notation): Spec M vs Spec P") ///
        mtitles("Spec M (main, S_t)" "Spec P (country-pair, S_{c,t})") ///
        note("FE = firm x qtr + hgroup x qtr. Cluster SE on (firm, quarter). " ///
             "Spec M: full panel, US-CN AR(1) residual. " ///
             "Spec P: GB+DE+FR subsample, country-pair AR(1) residual S_{c,t}. " ///
             "us_lvl absorbed by gamma_{g,t} in both. us_s absorbed in Spec M (no c-variation in S_t).")

    esttab specM specP using "`OUT'/07c_strict_4term_results.csv", replace ///
        cells("b(fmt(6)) se(fmt(6)) p(fmt(4))") ///
        stats(N r2 r2_a, fmt(%9.0gc %6.4f %6.4f)) ///
        keep(us_lvl us_s us_s_c us_cn us_cn_s us_cn_s_c) ///
        order(us_lvl us_s us_s_c us_cn us_cn_s us_cn_s_c) ///
        mtitles("SpecM_main" "SpecP_country_pair") ///
        nonumbers plain

    di "Wrote esttab outputs to `OUT'/07c_strict_4term_results.txt and .csv"
}
else {
    estimates table specM specP, b(%9.6f) se(%9.6f) p(%6.4f) ///
        stats(N r2 r2_a) ///
        keep(us_lvl us_s us_s_c us_cn us_cn_s us_cn_s_c)
}

* ============================================================
* Final headline display
* ============================================================
display _newline "=== FINAL HEADLINE: Spec M (main, full panel) ==="
estimates restore specM
estimates table, b(%9.6f) se(%9.6f) p(%6.4f)

display _newline "=== FINAL HEADLINE: Spec P (country-pair, GB+DE+FR) ==="
estimates restore specP
estimates table, b(%9.6f) se(%9.6f) p(%6.4f)

display _newline "Done."
