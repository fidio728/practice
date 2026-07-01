* 07b_country_pair_robustness.do
* Country-pair GPR shock robustness for the GB+DE+FR subsample.
*
* Robustness redesign:
*   Main spec uses Shock_t = AR(1) residual of USA|China bilateral GPR_AI,
*   applied identically to every EU listing country. This do-file replaces
*   the homogeneous Shock_t with country-pair-specific shocks S_{c,t}
*   constructed as AR(1) residuals of (UK|China), (Germany|China),
*   (France|China), and merges S_{c,t} to the panel by (sec_country, quarter).
*
* Identification with c.us##c.cn_lag##c.shock_c and FE = alpha_{i,t} + gamma_{g,t}:
*   - beta_0 * S_{c,t}            absorbed by alpha_{i,t} (c(i) constant in t)
*   - beta_1 * US_g * S_{c,t}     IDENTIFIED (NEW vs main)
*   - CN_{i,t-1}                  absorbed by alpha_{i,t}
*   - US_g * CN_{i,t-1}           IDENTIFIED (= main beta_2)
*   - CN * S_{c,t}                absorbed by alpha_{i,t}
*   - US * CN * S_{c,t}           IDENTIFIED (= main beta_3 with new shock)
* So three coefficients survive on the pooled subsample.
*
* Spec C3 reruns the MAIN US-CN shock (shock_us_cn) on the SAME subsample
* (GB+DE+FR firms) so the country-pair result is benchmarked against the
* homogeneous-shock baseline on identical observations.
*
* Cluster on (firm, quarter); NOT on country (only 3 countries -> too few).
*
* Stata gotcha (from 07_regression.do): pandas to_stata writes datetime64
* as %tc (milliseconds); MUST apply dofc() before mofd().

clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/c6_panel_country_pair.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

* ============================================================
* Sanity counts
* ============================================================
display _newline "=== Sample composition (country-pair robustness subsample) ==="
count
display _newline "Holder group:"
tab hgroup
display _newline "Listing country (must be GB/DE/FR only):"
tab sec_country
display _newline "Quarters covered:"
sum rdate, format

* Hard assertion: only GB/DE/FR firms allowed
gen byte _bad_country = !inlist(sec_country, "GB", "DE", "FR")
count if _bad_country == 1
assert r(N) == 0
drop _bad_country

* Numeric IDs for FE / clustering
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

* Diagnostic: how many rows have missing shock_c or shock_us_cn?
count if missing(shock_c)
display "rows with missing shock_c: " r(N)
count if missing(shock_us_cn)
display "rows with missing shock_us_cn: " r(N)

* ============================================================
* Pre-compute interaction terms (clean names for esttab)
* ============================================================
gen us_cn          = us * cn_lag
gen us_s_c         = us * shock_c
gen us_cn_s_c      = us * cn_lag * shock_c
gen us_cn_s_main   = us * cn_lag * shock_us_cn

label var us_cn        "US x CN(t-1)"
label var us_s_c       "US x Shock_c"
label var us_cn_s_c    "US x CN(t-1) x Shock_c"
label var us_cn_s_main "US x CN(t-1) x Shock_USA-CN"

* Per-country flags
gen byte _gb = (sec_country == "GB")
gen byte _de = (sec_country == "DE")
gen byte _fr = (sec_country == "FR")

* ============================================================
* Spec C1: explicit triple-interaction form (screen display only)
*          reghdfe auto-omits absorbed terms; serves as a sanity check
*          that surviving coefs match Spec C2's hand-built form.
* ============================================================
display _newline _newline "=== Spec C1: c.us##c.cn_lag##c.shock_c (screen check) ==="
reghdfe dw c.us##c.cn_lag##c.shock_c, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store c1

* ============================================================
* Spec C2 (HEADLINE): explicit beta_1 + beta_2 + beta_3 with clean names
* ============================================================
display _newline _newline "=== Spec C2 (HEADLINE): us_cn + us_s_c + us_cn_s_c ==="
reghdfe dw us_cn us_s_c us_cn_s_c, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store c2

* ============================================================
* Spec C3: MAIN US-CN shock on the SAME GB+DE+FR subsample (benchmark)
* ============================================================
display _newline _newline "=== Spec C3: MAIN US-CN shock on GB+DE+FR subsample ==="
reghdfe dw us_cn us_cn_s_main, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store c3

* ============================================================
* Per-country breakdowns: within one country, S_{c,t} varies only in t,
* so US x Shock_c is constant within (g,t) and gets absorbed by gq.
* Only us_cn and us_cn_s_c are identified per-country.
* ============================================================
display _newline _newline "=== Spec C1-GB: GB-only subsample ==="
capture noisily reghdfe dw us_cn us_cn_s_c if _gb == 1, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
if _rc == 0 estimates store c1_gb

display _newline _newline "=== Spec C1-DE: DE-only subsample ==="
capture noisily reghdfe dw us_cn us_cn_s_c if _de == 1, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
if _rc == 0 estimates store c1_de

display _newline _newline "=== Spec C1-FR: FR-only subsample ==="
capture noisily reghdfe dw us_cn us_cn_s_c if _fr == 1, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
if _rc == 0 estimates store c1_fr

* ============================================================
* Summary tables (esttab if available)
* ============================================================
display _newline _newline "=== Summary tables ==="

capture which esttab
if _rc == 0 {
    * Main table: C2 (country-pair shock, HEADLINE) vs C3 (main shock benchmark)
    esttab c2 c3 using "`OUT'/07b_country_pair_results.txt", replace ///
        cells("b(fmt(6) star) se(fmt(6))") ///
        stats(N r2 r2_a, fmt(%9.0gc %6.4f %6.4f) labels("N" "R2" "Adj R2")) ///
        keep(us_cn us_s_c us_cn_s_c us_cn_s_main) ///
        title("Country-pair GPR shock robustness - GB+DE+FR subsample") ///
        mtitles("C2: country-pair shock" "C3: main USA-CN shock") ///
        note("FE = firm x qtr + hgroup x qtr. Cluster SE on (firm, quarter). Subsample: sec_country in GB/DE/FR.")

    * Per-country breakdown table
    esttab c2 c1_gb c1_de c1_fr using "`OUT'/07b_country_pair_by_country.txt", replace ///
        cells("b(fmt(6) star) se(fmt(6))") ///
        stats(N r2 r2_a, fmt(%9.0gc %6.4f %6.4f) labels("N" "R2" "Adj R2")) ///
        keep(us_cn us_s_c us_cn_s_c) ///
        title("Country-pair GPR shock - per-country breakdown") ///
        mtitles("All (C2)" "GB only" "DE only" "FR only") ///
        note("FE = firm x qtr + hgroup x qtr. us_s_c absorbed within single country.")

    * CSV version
    esttab c2 c3 using "`OUT'/07b_country_pair_results.csv", replace ///
        cells("b(fmt(6)) se(fmt(6)) p(fmt(4))") ///
        stats(N r2 r2_a, fmt(%9.0gc %6.4f %6.4f)) ///
        keep(us_cn us_s_c us_cn_s_c us_cn_s_main) ///
        mtitles("C2_country_pair" "C3_main_shock") ///
        nonumbers plain

    di "Wrote esttab outputs to `OUT'/07b_country_pair_*.txt and .csv"
}
else {
    estimates table c2 c3, b(%9.6f) se(%9.6f) ///
        stats(N r2 r2_a) keep(us_cn us_s_c us_cn_s_c us_cn_s_main)
    estimates table c2 c1_gb c1_de c1_fr, b(%9.6f) se(%9.6f) ///
        stats(N r2 r2_a) keep(us_cn us_cn_s_c)
}

* ============================================================
* Final headline: Spec C2 (country-pair shock) coefficients
* ============================================================
display _newline "=== FINAL HEADLINE: Spec C2 coefficients (beta_1, beta_2, beta_3) ==="
estimates restore c2
estimates table, b(%9.6f) se(%9.6f) p(%6.4f)

display _newline "Done."
