* run_fourgroup.do — Essay 2 ACTIVE-ONLY four-group design (2026-08-02).
*
* MOTIVE. The pooled US-vs-NONUS headline mixes active, passive, and
* unlabeled institutions. Passive/index funds track a benchmark and are
* expected to respond only weakly (not exactly zero: flows, index changes,
* corporate actions and relative prices still move their weights). Because the
* pooled book aggregates ALL labels, the decomposition is
*   beta3_pooled ~ s_active*beta3_active + s_passive*beta3_passive
*                  + s_unknown*beta3_unknown,
* with s_g the group's book VALUE share — so the dilution multiplier on the
* active response is the ACTIVE VALUE SHARE (UNKNOWN mass dilutes too), not
* (1 - p_passive). US passive book share rose 13%->40% (2010->2021) vs NONUS
* 7%->20%, so the dilution is larger and growing on the US side. This file isolates the ACTIVE book and asks
* whether the ACTIVE-only US-vs-NONUS differential response is MORE negative than
* the pooled headline (the dilution prediction), stated with restraint: an
* active beta3 more negative than pooled is CONSISTENT with dilution, not proof.
*
* DESIGN (locked). Four groups grp = side x label:
*   side  = US (investor_country=='US') vs NONUS (other EU-holding countries)
*   label = ACTIVE (Funds.STYLE non-missing & != 'Index')
*           PASSIVE (Funds.STYLE == 'Index')
*   UNKNOWN (STYLE missing / fund unmatched) is NEVER treated as active and is
*   excluded from both the ACTIVE and PASSIVE books upstream. Each group's book
*   self-normalizes: w_g(firm) = I_g(firm) / sum_EU I_g, so dw is the backward
*   change of a within-group weight. Snapshot of the Funds master is ~2018-08;
*   the label is therefore PREDETERMINED for report quarters at/after 2018m8 and
*   carries look-ahead before it. The post-2018 subsample is the PRIMARY report;
*   the full-period column is disclosed with the look-ahead caveat.
*
* MAIN test: US_ACTIVE vs NONUS_ACTIVE, 3-pairwise DDD under the saturated FE
*   (firm x quarter, grp x quarter, firm x grp), two-way cluster (firm, month):
*       dw = b2 * (us_act x cn_lag) + b3 * (us_act x cn_lag x S) + fq + gq + ig
*   us_act = 1{grp=='US_ACTIVE'}. cn_lag (firm x quarter constant) and cn_lag x S
*   are absorbed by fq; us_act and us_act x S by gq/ig. b3 is the estimand.
* MENU (isomorphic FE logic, each restricted to two of the four groups):
*   (1) PASSIVE vs PASSIVE  — treat = 1{US_PASSIVE}; expected near-zero/inert.
*   (2) US-internal ACTIVE vs PASSIVE — treat = 1{US_ACTIVE} within {US_*};
*       tests active-minus-passive responsiveness holding side fixed.
*   pooled headline is the two-group US/NONUS c6 spec (separate panel,
*   c6_panel.dta) and is the DILUTION BENCHMARK cited in the addnote below.
* Every column reports N, #firms, #(firm clusters)=#firms, #(month clusters).
* B9 convention: degenerate two-way-cluster VCE detection + output diag CSV.

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
use "`OUT'/fourgroup_panel.dta", clear

*==============================================================
* Expected schema (asserted): firm_str(str), grp(str in the four values),
* rdate(%tc quarter), dw(double outcome), cn_lag(double, china_share_lag1q),
* shock(double S_t, constant within quarter). Optional level weight `w` for the
* book-sum reconciliation assertion.
*==============================================================
confirm string variable firm_str grp
confirm numeric variable dw cn_lag shock
gen rd_day = dofc(rdate)
format rd_day %td
gen rd_m = mofd(rd_day)
format rd_m %tm
count if missing(rd_day)
assert r(N) == 0

* grp must be exactly the four locked groups
gen byte _okgrp = inlist(grp, "US_ACTIVE", "NONUS_ACTIVE", "US_PASSIVE", "NONUS_PASSIVE")
assert _okgrp == 1
drop _okgrp

* derive side / label from grp (do not depend on panel carrying them)
gen byte us_side  = (grp == "US_ACTIVE" | grp == "US_PASSIVE")
gen byte is_activ = (grp == "US_ACTIVE" | grp == "NONUS_ACTIVE")

* cn_lag sanity + shock constant within quarter (belt after the builder)
assert cn_lag >= 0 & cn_lag <= 1 if !missing(cn_lag)
bysort rd_m (shock): assert shock == shock[1] if !missing(shock)
* load-bearing FE-absorption precondition: cn_lag must be GROUP-INVARIANT
* within firm x quarter (fq absorbs the bare cn_lag only if so)
bysort firm_str rd_day (cn_lag): assert cn_lag == cn_lag[1] if !missing(cn_lag)

* integer FE codes (built globally; reghdfe drops unused levels per subsample)
egen firm_n = group(firm_str)
egen fq = group(firm_str rd_day)
egen gq = group(grp rd_day)
egen ig = group(firm_str grp)

* PREDETERMINED-LABEL subsample: report month at/after the 2018-08 Funds snapshot.
* rd_m >= tm(2018m8) keeps 2018Q3 (report month 9) onward; earlier quarters carry
* look-ahead in the label and are full-period-only.
gen byte post2018 = (rd_m >= tm(2018m8)) if !missing(rd_m)
gen byte all1 = 1
label var post2018 "report month >= 2018m8 (label predetermined)"

*==============================================================
* Book-weight reconciliation (conditional on the panel carrying level weight w):
* each non-empty group's book weights sum ~1 per (grp, quarter).
* The stronger identity  I_ACTIVE + I_PASSIVE + I_UNKNOWN == pooled side  is an
* I_ict-level check owned by the panel BUILDER (UNKNOWN is excluded here), noted
* for the external reviewer; it cannot be re-derived from this estimation panel.
*==============================================================
* COVERAGE DIAGNOSTIC ONLY (no assertion): this .dta is the cn_lag-filtered
* estimation panel (~6,854 of 12,743 universe firms), while w is normalized on
* the FULL universe grid. Sum_firm(w) per (grp,quarter) here is therefore the
* exposure-matched share of each book (expected < 1), NOT 1. The sum-to-1
* invariant is asserted builder-side on the full grid (reconciliation B in
* build_fourgroup_panel.py); re-asserting it here would always abort (the
* build_c6_panel.py simplex-check comment warns about exactly this).
capture confirm numeric variable w
if _rc == 0 {
    preserve
    collapse (sum) wsum = w, by(grp rd_day)
    quietly summarize wsum, detail
    display _newline "exposure-matched book coverage Sum_firm(w) per (grp,quarter):" ///
        _newline "    min=" %6.4f r(min) "  p50=" %6.4f r(p50) "  max=" %6.4f r(max) ///
        "   (expected < 1; sum-to-1 holds on the FULL grid, asserted in the builder)"
    restore
}
else {
    display _newline "NOTE: no level weight `w' in panel; book-sum==1 invariant is asserted by the builder."
}

*==============================================================
* Matched-share decay disclosure. The share of each side's book matched to a
* fund STYLE drifts as the 2018-08 snapshot ages (US ~95% -> ~90%). If the panel
* carries a per-row matched-share column, summarize it here; otherwise print the
* documented figures so the decay is visible in the run log.
*==============================================================
capture confirm numeric variable matched_share
if _rc == 0 {
    display _newline "=== matched-share by side x year (snapshot 2018-08 ages) ==="
    gen int _yr = year(rd_day)
    table us_side _yr, statistic(mean matched_share) nototals
    drop _yr
}
else {
    display _newline "NOTE: matched-share decay (US ~95% -> ~90% across the window) is reported"
    display        "      by the panel builder; no per-row matched_share column in this panel."
}

*==============================================================
* Spec runner. Restrict to two grp values; treat = 1{grp==treated}; build the
* two DDD regressors with consistent NAMES (t_cn, t_cn_s) so esttab stacks cleanly
* across menus; report N / #firms / cluster counts.
*==============================================================
capture program drop run_spec
program define run_spec
    * ename  : estimates name
    * g0 g1  : the two grp values kept (g1 = treated)
    * svar   : 0/1 sample selector variable (all1 or post2018)
    * slab   : sample label for the log
    args ename g0 g1 svar slab
    capture drop t_cn t_cn_s _touse
    gen byte   _touse  = inlist(grp, "`g0'", "`g1'") & `svar' == 1
    gen double t_cn    = (grp == "`g1'") * cn_lag
    gen double t_cn_s  = (grp == "`g1'") * cn_lag * shock
    label var t_cn   "treat x CN(t-1)"
    label var t_cn_s "treat x CN(t-1) x S_t"
    display _newline "=== `ename' [`g1' vs `g0', `slab'] : dw ~ t_cn + t_cn_s, absorb(fq gq ig) ==="
    reghdfe dw t_cn t_cn_s if _touse, absorb(fq gq ig) vce(cluster firm_n rd_m)
    estimates store `ename'
    * distinct firms / quarters actually used
    quietly count if e(sample)
    local nobs = r(N)
    quietly levelsof firm_n if e(sample), local(_f)
    local nfirm : word count `_f'
    quietly levelsof rd_m if e(sample), local(_q)
    local nq : word count `_q'
    display "    N=" %9.0gc `nobs' "  firms=" %6.0f `nfirm' ///
            "  firm-clusters=" %6.0f e(N_clust1) "  month-clusters=" %5.0f e(N_clust2) ///
            "  quarters=" %4.0f `nq'
    display "    b3(t_cn_s)=" %10.3e _b[t_cn_s] "  se=" %10.3e _se[t_cn_s] ///
            "  p=" %6.4f 2*ttail(e(df_r), abs(_b[t_cn_s]/_se[t_cn_s]))
end

*==============================================================
* MAIN: US_ACTIVE vs NONUS_ACTIVE — full period + post-2018 (primary).
*==============================================================
run_spec m_act_full "NONUS_ACTIVE" "US_ACTIVE" all1     "full-period (look-ahead disclosed)"
run_spec m_act_post "NONUS_ACTIVE" "US_ACTIVE" post2018 "post-2018 PRIMARY"

*==============================================================
* MENU 1: PASSIVE vs PASSIVE (US_PASSIVE vs NONUS_PASSIVE) — inert benchmark.
*==============================================================
run_spec m_pas_full "NONUS_PASSIVE" "US_PASSIVE" all1     "full-period"
run_spec m_pas_post "NONUS_PASSIVE" "US_PASSIVE" post2018 "post-2018"

*==============================================================
* MENU 2: US-internal ACTIVE vs PASSIVE (treat = US_ACTIVE, side fixed = US).
*==============================================================
run_spec m_usi_full "US_PASSIVE" "US_ACTIVE" all1     "full-period"
run_spec m_usi_post "US_PASSIVE" "US_ACTIVE" post2018 "post-2018"

*==============================================================
* MAIN vs pooled comparability report (dimensions only; pooled lives in c6_panel).
*==============================================================
display _newline "=== MAIN (post-2018) panel vs pooled c6_panel comparability ==="
estimates restore m_act_post
quietly count if e(sample)
display "    MAIN post-2018: N=" %9.0gc r(N) "  (vs the pooled two-group c6_panel"
display "    headline: MAIN covers a smaller book PRIMARILY because UNKNOWN- and"
display "    passive-style holdings are excluded from the active book (self-"
display "    normalized, covered share < 1); snapshot matched-share attrition is a"
display "    secondary driver. Dimensions reported per column above)."

*==============================================================
* B9: degenerate two-way-cluster VCE detection on the DDD triple t_cn_s.
*==============================================================
tempname fh
file open `fh' using "`OUT'/fourgroup_vce_diag.csv", write replace
file write `fh' "spec,coef,b,se,se_valid" _n
local any_degen 0
foreach m in m_act_full m_act_post m_pas_full m_pas_post m_usi_full m_usi_post {
    estimates restore `m'
    foreach cf in t_cn t_cn_s {
        local bb = _b[`cf']
        local ss = _se[`cf']
        local ok = (!missing(`ss') & `ss' > 0)
        file write `fh' "`m',`cf',`bb',`ss',`ok'" _n
        if `ok' == 0 {
            local any_degen 1
            display as error ">>> `m'/`cf': DEGENERATE VCE (SE missing/zero) — CRVE p INVALID. RI coverage: run_ri_fourgroup.py covers the MAIN contrast (m_act_*); menu contrasts need the same fold with their own group pair. <<<"
        }
    }
}
file close `fh'
di "Wrote `OUT'/fourgroup_vce_diag.csv"
if `any_degen' == 1 {
    display as error "One or more four-group specs had a degenerate two-way-cluster VCE. Their CRVE SE/p in fourgroup_results.csv are NOT valid inference. run_ri_fourgroup.py provides design-based RI for the MAIN contrast; a degenerate MENU column requires extending the RI to that group pair before it can be cited."
}

*==============================================================
* Results table.
*==============================================================
capture which esttab
if _rc == 0 {
    esttab m_act_full m_act_post m_pas_full m_pas_post m_usi_full m_usi_post ///
        using "`OUT'/fourgroup_results.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) ///
        keep(t_cn t_cn_s) ///
        mtitles("ACT_full" "ACT_post" "PAS_full" "PAS_post" "USint_full" "USint_post") ///
        nonumbers plain ///
        addnote("t_cn_s is the DDD triple (treat x CN(t-1) x S_t). ACT = US_ACTIVE vs NONUS_ACTIVE (MAIN); PAS = US_PASSIVE vs NONUS_PASSIVE (inert benchmark); USint = US-internal ACTIVE vs PASSIVE. Funds master snapshot ~2018-08: labels are PREDETERMINED at/after 2018m8, so *_post columns are the PRIMARY report and *_full carry a look-ahead caveat. UNKNOWN-style funds are excluded from both books; each group book self-normalizes. Dilution hypothesis: pooled beta3 ~ value-share-weighted average of ACT/PAS/UNKNOWN betas, so the multiplier on the active response is the ACTIVE VALUE SHARE (UNKNOWN mass dilutes too); ACT beta3 more negative than pooled is CONSISTENT with dilution, not proof, and PAS beta3 is expected weak, not exactly zero. VCE validity per spec in fourgroup_vce_diag.csv; degenerate columns -> design-based RI in run_ri_fourgroup.py.")
    di "Wrote `OUT'/fourgroup_results.csv"
}

display _newline "=== b / se / p per spec (triple = t_cn_s) ==="
foreach m in m_act_full m_act_post m_pas_full m_pas_post m_usi_full m_usi_post {
    estimates restore `m'
    display _newline "--- `m' ---"
    estimates table, b(%12.4e) se(%12.4e) p(%6.4f) keep(t_cn t_cn_s)
}
display _newline "Done."
