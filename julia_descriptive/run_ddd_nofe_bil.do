* run_ddd_nofe_bil.do — Step-3 slide add-ons, gate-locked.
* STAGE A  gate: reproduce the locked 3-pairwise headline.
*          GLOBAL-MAIN + S_{t-1} PRIMARY (2026-08-08): the headline is
*          dw(global full-portfolio denominator) on us_cn + us_cn_slag
*          (us×cn_lag×S_{t-1}), absorb(fq gq ig), two-way cluster.
*          The gate target is NO LONGER HARDCODED: it is read at runtime from
*          the LIVING canonical artifact output/headline_3pairwise_canonical.csv
*          (row spec=="primary", fe=="fq_gq_ig"), produced by
*          run_headline_3pairwise.do / run_attribution_em.do from stored
*          estimates. Historical hardcoded gates (P0 -5.279916e-07 N=347,690;
*          pre-P0 +2.746e-06 N=347,490) are retired — vintage drift between a
*          comment and the artifact was exactly the failure mode.
* STAGE B  PROPER no-FE column: full factorial with ALL lower-order terms
*          (us, cn_lag, s_lag, us_x_slag, cn_x_slag) + interactions.
*          The old 07d "No FE" cell omitted every main effect -> unusable.
* STAGE C  BIL footnote numbers: on the bilateral-covered subsample,
*          3-pairwise beta3 without vs with the US x BIL control.
*          (BIL LEVEL is absorbed by firm x quarter FE; only the US
*          differential is addable.)

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

* ---- Gate target from the living canonical artifact (never hardcode) ----
import delimited using "`OUT'/headline_3pairwise_canonical.csv", ///
    clear varnames(1) case(lower)
keep if spec == "primary" & fe == "fq_gq_ig"
if _N != 1 {
    display as error "canonical CSV has no unique primary/fq_gq_ig row — stale pre-2026-08-08 layout? Re-run run_headline_3pairwise.do."
    exit 459
}
local tgt_b3 = b3[1]
local tgt_N  = n[1]
display "gate target from canonical CSV: b3=" %12.4e `tgt_b3' "  N=" %12.0gc `tgt_N'

* ---- BIL lookup from the (audited) firm ladder panel ----
use "`OUT'/firm_ladder_panel.dta", clear
keep firm_str rdate bil_us_c
duplicates drop firm_str rdate, force
tempfile bil
save `bil', replace

* ---- C6 zero-filled backward-diff panel (same build as 07d/07e) ----
* dw = GLOBAL-denominator Δw (MAIN); dw_eu = EU diagnostic; s_lag = S_{t-1}.
use "`OUT'/c6_panel.dta", clear
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
egen fq     = group(firm_str rd_day)
gen  rd_m   = mofd(rd_day)
egen gq     = group(hgroup rd_day)
egen ig     = group(firm_str hgroup)

gen us_cn       = us * cn_lag
gen us_slag     = us * s_lag
gen cn_slag     = cn_lag * s_lag
gen us_cn_slag  = us * cn_lag * s_lag

* ============ STAGE A: gate (3-pairwise headline must reproduce) ============
display _newline "===== STAGE A GATE: 3-pairwise headline (global dw x S_{t-1}) ====="
reghdfe dw us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)
local b3 = _b[us_cn_slag]
display "gate beta3 = " %12.4e `b3' "   target " %12.4e `tgt_b3' "   N=" e(N) " (target " %12.0gc `tgt_N' ")"
if abs(`b3' - `tgt_b3') > 0.05e-06 | e(N) != `tgt_N' {
    display as error "GATE FAILED — stopping."
    exit 459
}
display "GATE PASS"

* ============ STAGE B: PROPER no-FE column (full factorial) ============
display _newline "===== STAGE B: no-FE, ALL lower-order terms in (S_{t-1} timing) ====="
reghdfe dw us cn_lag s_lag us_cn us_slag cn_slag us_cn_slag, ///
    noabsorb vce(cluster firm_n rd_m)

* ============ STAGE C: bilateral subsample, w/o and w/ US x BIL ============
merge m:1 firm_str rdate using `bil', keep(master match) gen(_mb)
gen us_bil = us * bil_us_c

display _newline "===== STAGE C1: 3-pairwise on bilateral subsample (no BIL ctrl) ====="
reghdfe dw us_cn us_cn_slag if !missing(bil_us_c), ///
    absorb(fq gq ig) vce(cluster firm_n rd_m)

display _newline "===== STAGE C2: + US x BIL control ====="
reghdfe dw us_cn us_cn_slag us_bil if !missing(bil_us_c), ///
    absorb(fq gq ig) vce(cluster firm_n rd_m)

display _newline "Done — run_ddd_nofe_bil.do"
