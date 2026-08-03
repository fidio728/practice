* run_tercile_3pairwise.do — shock-TERCILE triple-difference (bottom/middle/top
* thirds of the 82 distinct quarterly US-NONUS shocks S_t), under the 3-pairwise
* FE (fq gq ig), consistent with the new headline FE and with the retired tail menu
* (run_tail_3pairwise.do).
*
* DESIGN PROVENANCE. Advisor request: replace the 2-sigma tail dummy (few-treated-
* cluster inference problem) with shock TERCILES (bottom/middle/top ~33% each; 82
* quarters -> realized 28/27/27, the boundary quarter falls in T1 because
* D_T1 uses <= c1) to (a) cure the few-treated-cluster pathology and (b) test
* dose-monotonicity. Decoupling prediction: beta3(T3) < beta3(T1), i.e. the
* lincom contrast us_cn_t3 - us_cn_t1 NEGATIVE (US pulls away from high-CN firms
* MORE in top-shock quarters); a positive contrast runs AGAINST decoupling. Three-strand defence (LOCKED):
* Antoniou 2013 JFQA uses a 3-bin sentiment split as prior art for tercile designs;
* Cattaneo 2024 AER (binscatter) motivates coarsening a continuous regressor into a
* few dose bins; MacKinnon-Webb motivates moving off a sparse-treated dummy toward a
* design where every bin is well-populated for cluster-robust inference.
*
* PANEL. c6_panel.dta — B7-fixed rebuild (2026-07-22): 347,952 rows, 6,854 firms,
* 82 quarters (2003Q3-2023Q4), fully-paired US/NONUS. Columns: firm_str,
* hgroup('US'/'NONUS'), rdate(%tc), dw(backward dW), cn_lag(china_share_lag1q,
* CUST+SUPP), shock(S_t, constant within quarter), us(0/1).
*
* SPEC LOGIC (isomorphic to the retired tail menu, run_tail_3pairwise.do). Tercile dummies D_T1(bottom)
* and D_T3(top); T2(middle) is the omitted base. Under the saturated FE
* (fq=firm x quarter, gq=group x quarter, ig=firm x group) every lower-order term
* (cn_lag, D_T1, D_T3, us, and their firm/quarter/group interactions) is absorbed;
* the surviving estimands are us_cn (=us*cn_lag) and the two triple interactions
* us_cn_t1 (=us*cn_lag*D_T1) and us_cn_t3 (=us*cn_lag*D_T3). Two-way cluster
* (firm_n rd_m). CRITICAL GOTCHA: the terciles MUST be cut on the 82 DISTINCT
* quarterly shock values (quarter level, tag(rd_m)); cutting via xtile on the
* 347,952 rows would let unequal firm counts per quarter bias the percentiles.
*
* B9 CONVENTION (see run_russia_headline.do B9 FIX): after every spec, detect a
* missing/zero SE on the triple interactions, warn loudly, and record VCE validity
* to output/tercile_vce_diag.csv so a degenerate CRVE column can never be read as
* valid inference. Design-based inference is the permutation test in
* run_ri_tercile.py (which re-cuts terciles on each permuted shock vector,
* bin sizes staying 28/27/27 by construction). Note: a CGM-repaired non-PSD
* two-way-cluster VCE passes the missing-SE check but is itself fragile — the
* RI is the inference anchor either way.

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
use "`OUT'/c6_panel.dta", clear

gen rd_day = dofc(rdate)
format rd_day %td
egen firm_n = group(firm_str)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
count if missing(rd_day)
assert r(N) == 0

gen us_cn = us * cn_lag
label var us_cn "US x CN(t-1)"

*==============================================================
* TERCILE CONSTRUCTION — cut on the 82 DISTINCT quarters, NOT the 347,952 rows.
* shock is constant within quarter, so one tagged row per quarter carries S_t.
* Compute the 33.33/66.67 percentiles over the 82 tagged values, then apply the
* two scalar cutpoints back at row level (equivalent to a merge, since S_t is
* quarter-constant). D_T1 = bottom third (S <= c1), D_T3 = top third (S > c2),
* T2 = middle third = omitted base.
*==============================================================
egen qtag = tag(rd_m)
quietly count if qtag==1
display _newline "Distinct quarters (should be 82): " r(N)

_pctile shock if qtag==1, percentiles(33.33333 66.66667)
local c1 = r(r1)
local c2 = r(r2)

gen byte D_T1 = (shock <= `c1') if !missing(shock)
gen byte D_T3 = (shock >  `c2') if !missing(shock)
label var D_T1 "shock bottom tercile"
label var D_T3 "shock top tercile"

* report quarter counts per tercile (realized 28/27/27) and the cutpoints
quietly count if qtag==1 & D_T1==1
local n1 = r(N)
quietly count if qtag==1 & D_T1==0 & D_T3==0
local n2 = r(N)
quietly count if qtag==1 & D_T3==1
local n3 = r(N)
display "Tercile cut on 82 quarters: c1(p33.33)=" %9.4f `c1' "  c2(p66.67)=" %9.4f `c2'
display "Quarters per tercile  T1(bottom)=`n1'  T2(middle,base)=`n2'  T3(top)=`n3'  (expect 28/27/27)"
if (`n1'+`n2'+`n3') != 82 {
    display as error ">>> tercile quarter counts do not sum to 82 — check for tied/missing shock. <<<"
    exit 459
}
* balance guard: a cutpoint bug that dumps quarters into one bin still sums to
* 82, so additionally require every bin to hold 25-29 quarters.
if (`n1' < 25 | `n1' > 29 | `n2' < 25 | `n2' > 29 | `n3' < 25 | `n3' > 29) {
    display as error ">>> tercile bins badly imbalanced (`n1'/`n2'/`n3') — cutpoint bug. <<<"
    exit 459
}

gen us_cn_t1 = us * cn_lag * D_T1
gen us_cn_t3 = us * cn_lag * D_T3
label var us_cn_t1 "US x CN(t-1) x bottom tercile"
label var us_cn_t3 "US x CN(t-1) x top tercile"

*==============================================================
* MAIN spec (3-pairwise fq gq ig) and CONTRAST spec (fq gq).
*==============================================================
display _newline _newline "=== MAIN: 3-pairwise fq gq ig ==="
reghdfe dw us_cn us_cn_t1 us_cn_t3, absorb(fq gq ig) vce(cluster firm_n rd_m)
estimates store m_main

* dose-monotonicity: top-tercile vs bottom-tercile triple-diff.
* Decoupling predicts this contrast NEGATIVE (more US pull-back under top-tercile
* shocks); a positive estimate runs against decoupling.
display _newline "--- monotonicity: lincom us_cn_t3 - us_cn_t1 (top vs bottom; decoupling => negative) ---"
lincom us_cn_t3 - us_cn_t1

display _newline _newline "=== CONTRAST: fq gq (no ig) ==="
reghdfe dw us_cn us_cn_t1 us_cn_t3, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store m_ctrl

*==============================================================
* B9 FIX: detect degenerate VCE (missing/zero SE) on the triple interactions and
* record per-spec VCE validity, so no CRVE column is ever silently read as valid.
*==============================================================
tempname fh
file open `fh' using "`OUT'/tercile_vce_diag.csv", write replace
file write `fh' "spec,coef,b,se,se_valid" _n
local any_degen 0
foreach m in m_main m_ctrl {
    estimates restore `m'
    foreach cf in us_cn_t1 us_cn_t3 {
        local bb = _b[`cf']
        local ss = _se[`cf']
        local ok = (!missing(`ss') & `ss' > 0)
        file write `fh' "`m',`cf',`bb',`ss',`ok'" _n
        if `ok' == 0 {
            local any_degen 1
            display as error ">>> `m'/`cf': DEGENERATE VCE (SE missing/zero) — CRVE p INVALID; use design-based RI (run_ri_tercile.py). <<<"
        }
    }
}
file close `fh'
di "Wrote `OUT'/tercile_vce_diag.csv"
if `any_degen' == 1 {
    display as error "One or more tercile specs had a degenerate two-way-cluster VCE. Their CRVE SE/p in tercile_results.csv are NOT valid inference; read the design-based RI (run_ri_tercile.py) instead."
}

*==============================================================
* Results table.
*==============================================================
capture which esttab
if _rc == 0 {
    esttab m_main m_ctrl using "`OUT'/tercile_results.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) ///
        keep(us_cn us_cn_t1 us_cn_t3) ///
        mtitles("3pairwise" "fq_gq") nonumbers plain ///
        addnote("T2(middle) is base; terciles cut on 82 distinct quarters (bins 28/27/27). MAIN drops 462 singletons vs CONTRAST (differing N). VCE validity per spec in tercile_vce_diag.csv; degenerate columns -> design-based RI in run_ri_tercile.py (re-cuts terciles per permuted shock).")
    di "Wrote `OUT'/tercile_results.csv"
}

display _newline "=== b / se / p per spec ==="
foreach m in m_main m_ctrl {
    estimates restore `m'
    display _newline "--- `m' ---"
    estimates table, b(%12.4e) se(%12.4e) p(%6.4f) keep(us_cn us_cn_t1 us_cn_t3)
}
display _newline "Done."
