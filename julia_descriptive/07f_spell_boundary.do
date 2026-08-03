* 07f_spell_boundary.do
* Advisor's conditional / spell-boundary sample (comment 3): held quarters +
* one boundary zero at each entry/exit, conditional on firm-groups the group
* ever held. Deep never-held zeros and never-held firms are excluded.
* Sample: c6_panel_spell.dta (296,590 rows, 6,355 firms, uneven US/NONUS).
*
* Compare to the full-grid headline (beta_3 = +2.081, p=0.151, N=347,952) (B7 2026-08-03).

clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/c6_panel_spell.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

display _newline "=== Spell-boundary conditional sample ==="
count
display _newline "Holder group (now uneven -- US engages fewer firm-quarters):"
tab hgroup
display _newline "Quarters:"
sum rdate, format

* keys
gen rd_day = dofc(rdate)
format rd_day %td
egen firm_n = group(firm_str)
egen fq     = group(firm_str rd_day)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen gq     = group(hgroup rd_day)
egen ig     = group(firm_str hgroup)

count if missing(rd_day)
assert r(N) == 0

gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock
label var us_cn       "US x CN(t-1)"
label var us_cn_shock "US x CN(t-1) x S_t"

* ============================================================
* Spec 1: headline on conditional sample
* ============================================================
display _newline _newline "=== S1: headline (absorb fq gq), conditional sample ==="
reghdfe dw us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store s1

* ============================================================
* Spec 2: + firm x group FE
* ============================================================
display _newline _newline "=== S2: + firm x group FE (absorb fq gq ig) ==="
reghdfe dw us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
estimates store s2

* ============================================================
* Summary
* ============================================================
display _newline _newline "=== Summary (conditional / spell-boundary sample) ==="
capture which esttab
if _rc == 0 {
    esttab s1 s2 using "`OUT'/07f_spell_boundary.txt", replace ///
        cells("b(fmt(%9.3e) star) se(fmt(%9.3e))") ///
        stats(N r2, fmt(%9.0gc %6.4f) labels("N" "R2")) ///
        keep(us_cn us_cn_shock) ///
        mtitles("cond: fq gq" "cond: + firmXgroup") ///
        title("Essay 2: advisor conditional (spell-boundary) sample") ///
        note("Coefs x10^-6. Two-way cluster (firm, quarter). Held + one boundary zero per entry/exit, conditional on engagement.")
    esttab s1 s2 using "`OUT'/07f_spell_boundary.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) ///
        keep(us_cn us_cn_shock) mtitles("cond_fqgq" "cond_firmgroup") nonumbers plain
    di "Wrote `OUT'/07f_spell_boundary.txt and .csv"
}

display _newline "=== beta_2 / beta_3 (b / se / p) ==="
foreach m in s1 s2 {
    estimates restore `m'
    display _newline "--- `m' ---"
    estimates table, b(%12.4e) se(%12.4e) p(%6.4f)
}

display _newline "Done."
