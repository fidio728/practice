* 07d_three_spec_table.do
* Slide-9 headline three-column table on the C6 zero-filled (backward-diff) panel.
*
* Columns (left to right in slide):
*   (1) Col 1  - SPEC 0  -  NO FE BASELINE       -> estimates m0
*       reghdfe dw us_cn us_cn_shock, noabsorb vce(cluster firm_n rd_m)
*       Pure OLS with two-way cluster; returns _cons; "unconditional
*       differential association" interpretation of beta_2 and beta_3.
*       (Note: noabsorb is documented as a no-op in current reghdfe but
*        the syntax is harmless; coefs match `regress` exactly.)
*
*   (2) Col 2  - SPEC 1  -  HEADLINE             -> estimates m1
*       reghdfe dw us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
*       fq = group(firm_str rd_day), gq = group(hgroup rd_day).
*       Already locked (B7 2026-08-03): beta_3 = +2.081e-06, SE = 1.44e-06, p = 0.151,
*       N = 347,952, R2 = 0.6235, F(2,81) = 1.31 p = 0.276.
*
*   (3) Col 3  - SPEC 4  -  WEAK FE              -> estimates m4
*       reghdfe dw us_cn us_cn_shock, absorb(firm_n rd_m) vce(cluster firm_n rd_m)
*       Already locked (B7 2026-08-03): beta_3 = +5.38e-07, SE = 6.51e-07, p = 0.411,
*       N = 347,952, R2 = 0.0073.
*
* BLOCKING fix from adversarial review: reghdfe does NOT save e(p_F),
* so we add `estadd scalar p_F = Ftail(e(df_m), e(df_r), e(F))` after
* each regression so the esttab Prob>F row is populated.

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

* Panel keys (identical to 07_regression.do)
gen rd_day = dofc(rdate)
format rd_day %td

egen firm_n   = group(firm_str)
egen fq       = group(firm_str rd_day)
gen  rd_m     = mofd(rd_day)
format rd_m %tm
egen gq       = group(hgroup rd_day)

count if missing(rd_day)
assert r(N) == 0

gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock

label var us_cn       "us_cn"
label var us_cn_shock "us_cn_shock"

* ============================================================
* Col 1 / Spec 0: NO FE baseline (pure OLS, two-way cluster).
* ============================================================
display _newline _newline "=== Col 1 / Spec 0: NO FE baseline (reghdfe, noabsorb) ==="
reghdfe dw us_cn us_cn_shock, ///
    noabsorb ///
    vce(cluster firm_n rd_m)
estadd scalar p_F = Ftail(e(df_m), e(df_r), e(F))
estimates store m0

display _newline "Spec 0 -- F and R2 (for slide footer):"
display "  F            = " %12.4f e(F)
display "  df_m         = " %12.0f e(df_m)
display "  df_r         = " %12.0f e(df_r)
display "  r2           = " %12.6f e(r2)
display "  r2_a         = " %12.6f e(r2_a)
display "  N            = " %12.0f e(N)

display _newline "Spec 0 -- full coefficient table including _cons:"
estimates table m0, b(%12.4e) se(%12.4e) p(%6.4f)

* ============================================================
* Col 2 / Spec 1: HEADLINE -- alpha_{i,t} + alpha_{c x t}
* ============================================================
display _newline _newline "=== Col 2 / Spec 1: HEADLINE, FE = (firm x quarter) + (hgroup x quarter) ==="
reghdfe dw us_cn us_cn_shock, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estadd scalar p_F = Ftail(e(df_m), e(df_r), e(F))
estimates store m1

display _newline "Spec 1 -- F and R2 (lock check):"
display "  F            = " %12.4f e(F)
display "  df_m         = " %12.0f e(df_m)
display "  df_r         = " %12.0f e(df_r)
display "  r2           = " %12.6f e(r2)
display "  N            = " %12.0f e(N)

* ============================================================
* Col 3 / Spec 4: WEAK FE -- firm + quarter (no interactions)
* ============================================================
display _newline _newline "=== Col 3 / Spec 4: WEAK FE, absorb(firm_n rd_m) ==="
reghdfe dw us_cn us_cn_shock, ///
    absorb(firm_n rd_m) ///
    vce(cluster firm_n rd_m)
estadd scalar p_F = Ftail(e(df_m), e(df_r), e(F))
estimates store m4

display _newline "Spec 4 -- F and R2 (lock check):"
display "  F            = " %12.4f e(F)
display "  df_m         = " %12.0f e(df_m)
display "  df_r         = " %12.0f e(df_r)
display "  r2           = " %12.6f e(r2)
display "  N            = " %12.0f e(N)

* ============================================================
* Side-by-side table: order (m0, m1, m4) = slide cols (1, 2, 3).
* Drop _cons from keep() to keep table clean (per reviewer P2).
* ============================================================
display _newline _newline "=== Slide-9 three-column headline table ==="

capture which esttab
if _rc == 0 {
    esttab m0 m1 m4 using "`OUT'/07d_three_spec_results.txt", replace ///
        cells("b(fmt(%9.3e) star) se(fmt(%9.3e))") ///
        stats(N r2 r2_a F p_F, ///
              fmt(%9.0gc %6.4f %6.4f %9.4f %6.4f) ///
              labels("N" "R2" "Adj R2" "F" "Prob>F")) ///
        keep(us_cn us_cn_shock) ///
        order(us_cn us_cn_shock) ///
        title("Slide 9 -- three-column headline table on C6 zero-filled (backward) panel") ///
        mtitles("(1) No FE" "(2) Headline" "(3) Weak FE") ///
        nonumbers ///
        note("Two-way cluster (firm, quarter) in all columns. Coefficients in scientific notation; slide rescales by 10^-6.")

    esttab m0 m1 m4 using "`OUT'/07d_three_spec_results.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2 r2_a F p_F, ///
              fmt(%9.0gc %6.4f %6.4f %9.4f %6.4f) ///
              labels("N" "R2" "Adj R2" "F" "Prob>F")) ///
        keep(us_cn us_cn_shock) ///
        order(us_cn us_cn_shock) ///
        mtitles("SpecM0_no_FE" "SpecM1_headline" "SpecM4_weak_FE") ///
        nonumbers plain

    di "Wrote esttab outputs to:"
    di "  `OUT'/07d_three_spec_results.txt"
    di "  `OUT'/07d_three_spec_results.csv"
}
else {
    di as error "esttab not installed -- falling back to estimates table"
    estimates table m0 m1 m4, b(%12.4e) se(%12.4e) p(%6.4f) ///
        stats(N r2 r2_a F) keep(us_cn us_cn_shock)
}

* Echo all three for eyeball verification
display _newline "=== Lock-check: Spec 0 (No FE) ==="
estimates restore m0
estimates table, b(%12.4e) se(%12.4e) p(%6.4f)

display _newline "=== Lock-check: Spec 1 (Headline) ==="
estimates restore m1
estimates table, b(%12.4e) se(%12.4e) p(%6.4f)

display _newline "=== Lock-check: Spec 4 (Weak FE) ==="
estimates restore m4
estimates table, b(%12.4e) se(%12.4e) p(%6.4f)

display _newline "Done."
