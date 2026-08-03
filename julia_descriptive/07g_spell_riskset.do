* 07g_spell_riskset.do
* CORRECTED conditional sample: firm-quarter risk set (both groups kept), so the
* within-firm-quarter US-vs-NONUS comparison is preserved (no group singletons).
* Sample: c6_panel_riskset.dta (268,282 rows, 5,564 firms; B7 2026-08-03).
* Supersedes 07f (per-group selection, which broke the pairing).
* Reconciliation note: pre-B7 references disagreed. 342,262 rows was cited with both
* 7,928 and 6,355 firms in different places (alongside a balanced 171,131/171,131
* US/NONUS split). The current verified counts are 268,282 rows / 5,564 firms; the old
* numbers are retained only as historical markers.
*
* Compare: full grid beta_3 = +2.081 (p=0.151, N=347,952) (B7 2026-08-03);
*          per-group spell (07f, biased) beta_3 = +4.38 (p=0.441, 45,672 singletons).

clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/c6_panel_riskset.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

display _newline "=== Risk-set conditional sample (balanced, paired) ==="
count
tab hgroup
sum rdate, format

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

display _newline _newline "=== R1: headline (absorb fq gq), risk-set sample ==="
reghdfe dw us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store r1

display _newline _newline "=== R2: + firm x group FE (absorb fq gq ig) ==="
reghdfe dw us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
estimates store r2

display _newline _newline "=== Summary (risk-set conditional sample) ==="
capture which esttab
if _rc == 0 {
    esttab r1 r2 using "`OUT'/07g_spell_riskset.txt", replace ///
        cells("b(fmt(%9.3e) star) se(fmt(%9.3e))") ///
        stats(N r2, fmt(%9.0gc %6.4f) labels("N" "R2")) ///
        keep(us_cn us_cn_shock) ///
        mtitles("riskset: fq gq" "riskset: + firmXgroup") ///
        title("Essay 2: firm-quarter risk-set conditional sample (paired US/NONUS)") ///
        note("Coefs x10^-6. Two-way cluster (firm, quarter). Both groups kept for any firm-quarter with a spell nearby.")
    esttab r1 r2 using "`OUT'/07g_spell_riskset.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) ///
        keep(us_cn us_cn_shock) mtitles("riskset_fqgq" "riskset_firmgroup") nonumbers plain
    di "Wrote `OUT'/07g_spell_riskset.txt and .csv"
}

display _newline "=== beta_2 / beta_3 (b / se / p) ==="
foreach m in r1 r2 {
    estimates restore `m'
    display _newline "--- `m' ---"
    estimates table, b(%12.4e) se(%12.4e) p(%6.4f)
}

display _newline "Done."
