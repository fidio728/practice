* run_flow_decomp_step3.do
* =============================================================================
* Essay 2 §7.6 — STEP 3 (CRVE version) of the accounting decomposition of
* observed institutional and residual ownership flows.
*
* POSITIONING (locked, do NOT re-open): this is a DESCRIPTIVE BOUNDARY, not an
* identification threat and not an "interference lower bound". It applies ONLY
* to the shares-based flow (§7.6), never to the within-portfolio weight w of the
* §6 main spec (no additive identity there).
*
* Accounting identity (per firm-quarter, common LAGGED float denominator):
*     flow_US + flow_NONUS + flow_R = (out_t - out_{t-1})/out_{t-1} == float_growth
* R = the unobserved residual sector (retail + insiders + non-reporting
* institutions + strategic holders). NOT to be called "retail" in the paper.
*
* WHAT THIS .do DOES: the CRVE (reghdfe two-way-cluster) companion to the
* design-based RI. Three firm-quarter outcomes:
*     flow_R       residual-sector flow (the counterparty margin)
*     flow_common  = flow_US + flow_NONUS  (common institutional flow)
*     flow_diff    = flow_US - flow_NONUS  (the differential the §6 estimand is)
* regressed on cn_lag and cn_lag x shock.
*
* WEAK-FE DESIGN (deliberate, locked): the treatment CN_{t-1} x S_t lives at the
* firm-quarter level. A firm#quarter FE would ABSORB it entirely, so this step
* cannot use the fq FE of the main spec. It uses firm FE + quarter FE only. That
* is a weaker design on purpose; its role is descriptive, and inference is
* anchored by the permutation RI (run_flow_decomposition.py, its RI step), NOT
* by these CRVE p's.
*
* WORLD-B BOUNDARY (locked): "US want to sell but price absorbs it, quantity does
* not move" is NOT identifiable from quantity data. A quantity null is the correct
* answer to a quantity question; separating "no demand change" from "demand change
* absorbed by price" requires the H2.2 PRICE evidence. Do NOT write "we win under
* either world".
*
* -----------------------------------------------------------------------------
* DEPENDENCY (Stata cannot read parquet): run_flow_decomposition.py is the ONE
* producer AND the RI engine. Its step [6/6] lands BOTH
*     output/flow_decomposition_panel.parquet   (for its own RI step)
*     output/flow_decomposition_panel.dta        (for THIS do-file)
* at the FIRM-QUARTER level (ONE row per firm-quarter, groups already collapsed;
* .dta keeps only rows with the full flow set). Columns: firm_str, rdate (%tc),
* flow_us, flow_nonus, flow_R, flow_common, flow_diff, float_growth, cn_lag,
* shock. The python side mirrors build_ownership_share_c6_panel.py's BAD
* firm-quarter rule (any group os>1 OR combined os>1 => current AND lagged
* nulled for both groups) and the fixed-lagged-float flow construction, and
* asserts the identity to float tolerance. This do-file re-asserts it as a belt.
* =============================================================================

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
use "`OUT'/flow_decomposition_panel.dta", clear

display _newline "=== flow_decomposition_panel (firm-quarter level) ==="
count
summarize flow_R flow_common flow_diff cn_lag shock, detail

* --- firm-quarter uniqueness: this panel must be collapsed, one row per fq ---
egen firm_n = group(firm_str)
gen  rd_day = dofc(rdate)
format rd_day %td
gen  rd_m   = mofd(rd_day)
format rd_m %tm
count if missing(rd_day)
assert r(N) == 0
isid firm_str rd_day
* (isid also guarantees fq FE would be a singleton per row => must NOT absorb it)

* --- belt-and-suspenders identity re-assert (python is the primary guard) ---
capture confirm variable float_growth
if _rc == 0 {
    gen double _id_gap = abs(flow_common + flow_R - float_growth)
    quietly summarize _id_gap
    display "identity max |flow_common + flow_R - float_growth| = " %12.4e r(max)
    assert _id_gap < 1e-6 if !missing(flow_common, flow_R, float_growth)
    drop _id_gap
}
else {
    display as text "note: float_growth not in panel; identity asserted python-side only."
}

* --- shock must be constant within a quarter (weak-FE design relies on it) ---
quietly {
    bysort rd_m (shock): gen byte _sk = shock != shock[1] & !missing(shock, shock[1])
    count if _sk
}
assert r(N) == 0
drop _sk

gen double cn_x_s = cn_lag * shock
label var cn_lag "CN(t-1)"
label var cn_x_s "CN(t-1) x S_t"

*==============================================================
* Estimation: 3 outcomes x {winsorized MAIN, raw APPENDIX}.
* Winsorized columns (*_w) are SHIPPED by run_flow_decomposition.py: p1/p99
* cutoffs computed on the ESTIMATION sample (cn_lag & shock non-missing), the
* same base and quantile definition as its RI step. Do NOT recompute here with
* _pctile — np.quantile vs _pctile definitional drift plus the full-panel
* cutoff base broke the CRVE/RI cross-check (CRVE 7.13e-4 vs RI 6.65e-4 on
* flow_diff before this fix; raw specs matched to 5 sig figs throughout).
* FE = firm + quarter (NOT firm#quarter). Two-way cluster (firm, quarter).
*==============================================================
foreach y in flow_R flow_common flow_diff {
    confirm variable `y'_w
}

foreach y in flow_R flow_common flow_diff {
    local t : subinstr local y "flow_" ""

    display _newline _newline "=== `y' (WINSORIZED, MAIN): firm + quarter FE ==="
    reghdfe `y'_w cn_lag cn_x_s, absorb(firm_n rd_m) vce(cluster firm_n rd_m)
    estimates store `t'_w
    display "`y'_w  b(cn_x_s) = " %12.4e _b[cn_x_s] "  se = " %12.4e _se[cn_x_s]

    display _newline "=== `y' (RAW, APPENDIX): firm + quarter FE ==="
    reghdfe `y' cn_lag cn_x_s, absorb(firm_n rd_m) vce(cluster firm_n rd_m)
    estimates store `t'_raw
    display "`y'   b(cn_x_s) = " %12.4e _b[cn_x_s] "  se = " %12.4e _se[cn_x_s]
}

*==============================================================
* B9 convention: degenerate two-way-cluster VCE detection + diag CSV.
* A missing/non-positive SE on cn_x_s means the CRVE column is NOT valid
* inference for that spec; defer to the RI companion.
*==============================================================
tempname fh
file open `fh' using "`OUT'/flowdecomp_vce_diag.csv", write replace
file write `fh' "spec,coef,b,se,se_valid" _n
local any_degen 0
foreach m in R_w R_raw common_w common_raw diff_w diff_raw {
    estimates restore `m'
    foreach cf in cn_lag cn_x_s {
        local bb = _b[`cf']
        local ss = _se[`cf']
        local ok = (!missing(`ss') & `ss' > 0)
        file write `fh' "`m',`cf',`bb',`ss',`ok'" _n
        if `ok' == 0 {
            local any_degen 1
            display as error ">>> `m'/`cf': DEGENERATE VCE — CRVE p INVALID; use run_flow_decomposition.py RI. <<<"
        }
    }
}
file close `fh'
di "Wrote `OUT'/flowdecomp_vce_diag.csv"
if `any_degen' == 1 {
    display as error "Degenerate two-way-cluster VCE detected; affected CRVE columns in flowdecomp_results.csv are NOT valid inference — use the RI companion."
}

*==============================================================
* esttab -> flowdecomp_results.csv (winsorized MAIN cols first, raw APPENDIX after)
*==============================================================
capture which esttab
if _rc == 0 {
    esttab R_w common_w diff_w R_raw common_raw diff_raw ///
        using "`OUT'/flowdecomp_results.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) ///
        keep(cn_lag cn_x_s) ///
        mtitles("flowR_w" "flowcommon_w" "flowdiff_w" "flowR_raw" "flowcommon_raw" "flowdiff_raw") ///
        nonumbers plain ///
        addnote("DESCRIPTIVE BOUNDARY (accounting decomposition of observed institutional and residual ownership flows), NOT an identification threat and NOT an interference lower bound; applies to shares-based flow only, never to the within-portfolio weight w. WEAK-FE by design: the treatment CN(t-1)xS_t is a firm-quarter attribute that a firm#quarter FE would fully absorb, so this step uses firm + quarter FE only. Inference is anchored by the design-based permutation RI (run_flow_decomposition.py), not by these CRVE p-values; VCE validity per spec in flowdecomp_vce_diag.csv. Winsorized (p1/p99) columns are MAIN, raw columns are the fat-tail appendix. WORLD-B boundary: 'demand falls but price absorbs it, quantity does not move' is NOT identifiable from quantity data; separating no-demand-change from price-absorbed-demand-change requires the H2.2 price evidence.")
    di "Wrote `OUT'/flowdecomp_results.csv"
}

display _newline "=== cn_x_s across specs (b / se / p) ==="
foreach m in R_w common_w diff_w R_raw common_raw diff_raw {
    estimates restore `m'
    display _newline "--- `m' ---"
    estimates table, keep(cn_lag cn_x_s) b(%12.4e) se(%12.4e) p(%6.4f)
}
display _newline "Done."
