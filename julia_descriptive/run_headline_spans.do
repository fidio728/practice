* run_headline_spans.do — TASK e4: PRIMARY spec across existence-span samples.
* Feeds the 8/21 advisor proposal (promote span-restricted to MAIN or keep as
* robustness — THEIR call; both reported).
*
* PRIMARY spec (identical to run_headline_3pairwise.do, LOCKED 2026-08-08):
*   reghdfe dw us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)
*   dw = Δw GLOBAL full-portfolio denominator (MAIN); s_lag = S_{t-1} PRIMARY.
*   b3 = us_cn_slag (us × cn_lag × S_{t-1}); us/cn_lag mains absorbed by FE.
*
* Samples:
*   (a) full_grid     — current canonical: every ever-held EU firm × 82 quarters
*                       (zero-filled Cartesian grid; 42% of cells outside the
*                       firm's holdings-based existence span).
*   (b) holdings_span — in_span==1: report_date within [first_active,
*                       last_active], first/last = min/max quarter with
*                       I_ict>0 (outcome-derived; the reviewer's objection).
*   (c) coverage_span — inside entity-level spans built from listing/delisting
*                       dates in Factset_Security_coverage.gz.
*                       BLOCKED 2026-08-10: the delivered own_sec_coverage
*                       extract has NO listing/delisting/coverage dates
*                       (13 cols; ADJDATE = corporate-actions vintage per
*                       FactSet V5 User Guide p.16; ACTIVE = undated snapshot
*                       flag). See build_coverage_spans.py docstring. This
*                       file AUTO-DETECTS output/entity_coverage_spans.csv
*                       (sec_entity_id,span_start,span_end as YYYY-MM-DD) and
*                       runs (c) the moment a symbology delivery makes it
*                       buildable; until then the (c) row is written with
*                       missing values and reason "no_usable_dates".
*
* Output: output/headline_span_variants.csv
*   sample,b3,se,p,N,n_firms   (n_firms = e(N_clust1), firm clusters in the
*                               estimation sample; two-way cluster firm_n rd_m)

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

* ----------------------------------------------------------------------------
* ROTATION DISCIPLINE (r1 must-fix M4i, 2026-08-10): REFUSE to clobber the
* canonical CSV (mirror of run_country_panel.do's guard). The auto-detect of
* entity_coverage_spans.csv below guarantees a future re-run once symbology
* arrives, so a bare `write replace' would silently swallow the delivered run.
* ----------------------------------------------------------------------------
capture confirm file "`OUT'/headline_span_variants.csv"
if _rc == 0 {
    display as error "target output/headline_span_variants.csv already exists —"
    display as error "rename it to headline_span_variants_r2pre.csv (rotation rule) before re-running."
    error 602
}

* ----------------------------------------------------------------------------
* CANONICAL DRIFT GATE ANCHOR (r1 must-fix M4ii, 2026-08-10; fail-closed, the
* run_shock_menu.do §0 / run_ri_shockmenu.py CANON_ANCHORS convention). The
* full_grid row below IS the canonical PRIMARY cell
* (primary/global/slag/fq_gq_ig of headline_3pairwise_canonical.csv), so its
* b3 is HARD-ASSERTED against that cell to 6 significant figures after the
* regression runs; a re-run on a stale audit_c6_panel.dta must abort rather
* than silently emit a plausible 3-row CSV.
* ----------------------------------------------------------------------------
local ANCHORCSV "`OUT'/headline_3pairwise_canonical.csv"
capture confirm file "`ANCHORCSV'"
if _rc != 0 {
    display as error "CANONICAL ANCHOR FILE NOT FOUND: `ANCHORCSV'"
    display as error "  Run run_headline_3pairwise.do first. Refusing to run unprotected (fail-closed)."
    exit 459
}
quietly import delimited "`ANCHORCSV'", clear varnames(1) case(preserve) stringcols(1)
capture confirm variable spec denom timing fe b3
if _rc != 0 {
    display as error "STALE ANCHOR LAYOUT in `ANCHORCSV' — expected the LOCKED 2026-08-08"
    display as error "columns spec,denom,timing,fe,b3,... Re-run run_headline_3pairwise.do."
    exit 459
}
quietly count if spec == "primary" & denom == "global" & timing == "slag" & fe == "fq_gq_ig"
if r(N) != 1 {
    display as error "ANCHOR ROW MISSING/AMBIGUOUS: primary/global/slag/fq_gq_ig (`r(N)' rows, must be 1)."
    display as error "  Stale vintage — re-run run_headline_3pairwise.do. Refusing to run unprotected."
    exit 459
}
quietly summarize b3 if spec == "primary" & denom == "global" & timing == "slag" & fe == "fq_gq_ig", meanonly
local APRIM_B3 = r(mean)
clear
display "canonical PRIMARY anchor (primary/global/slag/fq_gq_ig): b3=" %14.6e `APRIM_B3'
display "  -> the full_grid row will be HARD-ASSERTED against it to 6 significant figures."

use "`OUT'/audit_c6_panel.dta", clear
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
gen us_cn      = us*cn_lag
gen us_cn_slag = us*cn_lag*s_lag

* ---- (c) sample flag, only if a real spans file exists --------------------
local have_cov = 0
capture confirm file "`OUT'/entity_coverage_spans.csv"
if _rc == 0 {
    preserve
    import delimited "`OUT'/entity_coverage_spans.csv", clear varnames(1) stringcols(_all)
    keep sec_entity_id span_start span_end
    rename sec_entity_id firm_str
    gen long sp0 = date(span_start, "YMD")
    gen long sp1 = date(span_end,   "YMD")
    assert !missing(sp0) & !missing(sp1)
    keep firm_str sp0 sp1
    tempfile covspans
    qui save `covspans', replace
    restore
    qui merge m:1 firm_str using `covspans', keep(master match)
    qui count if _merge == 3
    display "coverage spans matched rows: " r(N)
    gen byte in_cov_span = _merge == 3 & inrange(rd_day, sp0, sp1)
    drop _merge
    local have_cov = 1
}
else {
    display as error "entity_coverage_spans.csv NOT FOUND - sample (c) blocked" ///
        " (own_sec_coverage carries no listing/delisting dates; see build_coverage_spans.py)"
}

* ---- run the three samples, write the machine-readable CSV ----------------
tempname cf
file open `cf' using "`OUT'/headline_span_variants.csv", write replace
file write `cf' "sample,b3,se,p,N,n_firms" _n

capture program drop _spanrow
program define _spanrow, rclass
    args cf lbl ifc
    qui reghdfe dw us_cn us_cn_slag `ifc', absorb(fq gq ig) vce(cluster firm_n rd_m)
    local b3 = _b[us_cn_slag]
    local se = _se[us_cn_slag]
    local p  = 2*ttail(e(df_r), abs(`b3'/`se'))
    display %-14s "`lbl'" "  b3=" %10.3e `b3' "  se=" %10.3e `se' ///
            "  p=" %6.4f `p' "  N=" %9.0gc e(N) "  firms=" %6.0f e(N_clust1)
    file write `cf' "`lbl'," ///
        (strtrim(strofreal(`b3', "%14.6e"))) "," ///
        (strtrim(strofreal(`se', "%14.6e"))) "," ///
        (strtrim(strofreal(`p',  "%9.6f"))) "," ///
        (strtrim(strofreal(e(N), "%15.0f"))) "," ///
        (strtrim(strofreal(e(N_clust1), "%15.0f"))) _n
    return scalar b3 = `b3'
end

display _newline "===== PRIMARY dw x S_{t-1} (fq gq ig, 2-way cluster) across span samples ====="
_spanrow `cf' "full_grid"     ""
* ---- drift gate (M4ii): the full_grid row IS the canonical primary cell ----
local FG_B3 = r(b3)
local RELERR = abs(`FG_B3'/`APRIM_B3' - 1)
display "  [DRIFT GATE primary/global/slag/fq_gq_ig] anchor b3=" %14.6e `APRIM_B3' ///
        "  observed b3=" %14.6e `FG_B3' "  relerr=" %9.2e `RELERR'
if `RELERR' > 1e-6 {
    display as error "=============================================================="
    display as error "DRIFT GATE FAILED — the full_grid b3 does not match the canonical"
    display as error "primary/global/slag/fq_gq_ig cell to 6 significant figures."
    display as error "  living source : `ANCHORCSV'"
    display as error "  anchor b3     : `APRIM_B3'"
    display as error "  observed b3   : `FG_B3'"
    display as error "  STALE PANEL OR WRONG VINTAGE — audit_c6_panel.dta does not"
    display as error "  reproduce the canonical primary regression. Refusing to emit"
    display as error "  a plausible-looking span CSV built on the wrong data."
    display as error "=============================================================="
    file close `cf'
    capture erase "`OUT'/headline_span_variants.csv"
    exit 459
}
display "  -> PASS (full_grid b3 matches the canonical primary cell)."
_spanrow `cf' "holdings_span" "if in_span==1"
if `have_cov' == 1 {
    _spanrow `cf' "coverage_span" "if in_cov_span==1"
}
else {
    file write `cf' "coverage_span,.,.,.,.,." _n
    display as error "coverage_span row written as missing (reason: no_usable_dates)"
}
file close `cf'
display _newline "Wrote `OUT'/headline_span_variants.csv"
