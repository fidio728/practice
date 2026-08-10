* ============================================================================
* run_firm_wls_heterogeneity.do — firm-level WLS, run LAST and labelled as a
* HETEROGENEITY DIAGNOSTIC ONLY (task e2 item 5, 2026-08-10).
* CODE ONLY at time of writing: nothing here has been executed.
*
* ============================================================================
* WHAT THIS IS, AND WHAT IT IS NOT
* ============================================================================
* THE QUESTION IT ANSWERS, and the only one:
*     "Do LARGER POSITIONS respond more?"
* Weighting the firm-level primary spec by lagged holdings tilts the estimate
* toward big positions.  If the weighted and the SAME-SAMPLE unweighted
* coefficients differ, the response is heterogeneous in position size.  That is
* the entire content.  "Same-sample" is load-bearing: the weighted fit also
* DELETES the entry margin, so the full-sample-unweighted vs weighted gap
* confounds a sample change with the size tilt.  See THREE RUNGS below.
*
* WHAT IT IS NOT — stated here because the claim it replaces was already made
* once and withdrawn:
*   It is NOT a reconciliation of the country-level result.  It does NOT
*   reproduce, adjudicate, explain or validate the country coefficient.  The
*   country outcome is a SUM of firm deltas; WLS minimises a WEIGHTED SUM OF
*   SQUARED RESIDUALS.  Those are different objects, and their equivalence needs
*   conditions this design does not satisfy.  Any sentence of the form "the WLS
*   reproduces the country result" is FALSE and must not be written.
*
* ============================================================================
* THE TWO CAVEATS ON THE WEIGHT (written into every row of the output CSV)
* ============================================================================
* (i)  DOUBLE-COUNTING OF SIZE.  The outcome dw = Delta w is ALREADY size-scaled
*      — w is the position's share of the group's book, so a big position
*      mechanically produces big w-changes.  Weighting by lagged w therefore
*      counts size twice: once inside the dependent variable and again in the
*      weight.  The weighted coefficient is not "the same effect, better
*      estimated"; it is a different, size-tilted, estimand.
* (ii) THE ENTRY MARGIN IS DISCARDED.  A new position has w_{t-1} = 0, so
*      aweight gives it weight zero and Stata drops it.  Every entry event —
*      exactly the margin the country decomposition isolates — is deleted from
*      the WLS sample.  The count of dropped zero-weight rows and the number of
*      those that are entry events are reported in the CSV.
*
* ============================================================================
* SPECIFICATION
* ============================================================================
* The firm-level PRIMARY spec, unchanged apart from the weight:
*     reghdfe dw us_cn us_cn_slag [aw=WT], absorb(fq gq ig) vce(cluster firm_n rd_m)
*   dw          = delta_w_global (GLOBAL denominator, MAIN)
*   us_cn       = us x cn_lag
*   us_cn_slag  = us x cn_lag x S_{t-1}   <- the reported coefficient
*   fq/gq/ig    = firm x quarter, group x quarter, firm x group
* WT is run in two flavours because "lagged holdings" is ambiguous:
*   w_prev      lagged PORTFOLIO WEIGHT (the flavour caveat (i) is about)
*   i_prev_usd  lagged DOLLAR holdings  (the literal reading of "holdings")
* Both are reported; neither is promoted.
*
* ----------------------------------------------------------------------------
* THREE RUNGS, NOT TWO (added 2026-08-10 — external review item 10)
* ----------------------------------------------------------------------------
* The previous vintage compared the FULL-SAMPLE unweighted fit against the
* weighted fit.  Those two differ by TWO things at once: an aweight of zero
* DELETES every w_prev == 0 row (the entry margin), and the surviving rows are
* then re-weighted.  A gap between them could be either.  The ladder is now
*   (i)   unweighted, FULL sample                 [spec = unweighted_full]
*   (ii)  unweighted, w_prev > 0 SAME sample      [spec = unweighted_wpos]
*   (iii) WLS,        w_prev > 0 SAME sample      [spec = wls_wpos]
* (i) -> (ii) is the SAMPLE change (the entry margin leaving).
* (ii) -> (iii) is the WEIGHTING change, and ONLY (ii) vs (iii) isolates size
* tilting.  Both deltas are computed and written (delta_sample, delta_weight),
* and rung (ii) is defined as reghdfe's OWN e(sample) from rung (iii), so the
* two really are the same rows rather than approximately the same rows (gated).
*
* A NOTE THAT TRAVELS WITH THE OUTPUT (better_test column).  Even rung (ii) vs
* (iii) is an indirect test of "do larger positions respond more?".  A SIZE-BIN
* specification (interact us_cn_slag with lagged-size terciles/quintiles and
* read the bin coefficients) or a continuous INTERACTION (us_cn_slag x
* ln(lagged size)) tests the same question DIRECTLY, keeps the entry margin in
* the sample as its own bin, and yields a coefficient with an interpretable
* sign and standard error instead of a difference between two estimands.  This
* file does not run one; it says so where the numbers are read.
*
* DRIFT GATE: the UNWEIGHTED row must reproduce the living canonical primary
* cell in output/headline_3pairwise_canonical.csv (spec==primary, denom==global,
* timing==slag, fe==fq_gq_ig) to 1e-6 relative.  If it does not, the panel or the
* canonical CSV is stale and this file refuses to write.
*
* INPUT : output/audit_c6_panel.dta            (canonical firm panel)
*         output/firm_wls_weights.dta          (e1 companion — see DEPENDENCY)
*         output/headline_3pairwise_canonical.csv  (living drift anchor)
* OUTPUT: output/firm_wls_heterogeneity.csv
* Rotation discipline: refuses to overwrite; rotate to *_cwpre first.
*
* ---------------------------------------------------------------------------
* DEPENDENCY (named, not worked around): audit_c6_panel.dta does NOT carry a
* level or lagged-level column — build_audit_panel_f1f2f7.py projects w and
* w_prev only to FORM cum0..cum4 and does not keep them.  So the weight must
* come from e1, which already reads merged_us_eu_zero_filled.parquet:
*     output/firm_wls_weights.dta
*         firm_str  str   = sec_entity_id
*         hgroup    str   = holder_group
*         rdate     num   %tc quarter-end
*         w_prev    num   = w_prev_global      (lagged portfolio weight)
*         i_prev_usd num  = lagged I_ict in USD (lagged dollar holdings)
* If the panel in memory already carries w_prev, that is used and the companion
* file is not required.  If neither exists this file HARD-FAILS with this
* message rather than substituting some other weight.
* ---------------------------------------------------------------------------
* ============================================================================

clear all
set more off

local SRC "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
local LIB "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/_country_weight_lib.do"
include "`LIB'"

local CAV1 "weight double-counts size: dw is already size-scaled (w = share of the group book)"
local CAV2 "aweight=0 on new positions: the ENTRY margin is deleted from the WLS sample"
local ROLE "HETEROGENEITY DIAGNOSTIC ONLY (do larger positions respond more?) — NOT a reconciliation or adjudication of the country-level result"
local BETTER "a SIZE-BIN spec (us_cn_slag x lagged-size bins) or a continuous us_cn_slag x ln(lagged size) interaction tests size heterogeneity DIRECTLY, keeps the entry margin as its own bin, and returns a coefficient with a sign and an SE"

* ----------------------------------------------------------------------------
* [0] Guards
* ----------------------------------------------------------------------------
cw_guard, target("`OUT'/firm_wls_heterogeneity.csv") ///
          pre("`OUT'/firm_wls_heterogeneity_cwpre.csv")

capture confirm file "`OUT'/audit_c6_panel.dta"
if _rc != 0 {
    display as error "missing output/audit_c6_panel.dta — run build_audit_panel_f1f2f7.py first."
    error 601
}
capture confirm file "`OUT'/headline_3pairwise_canonical.csv"
if _rc != 0 {
    display as error "missing output/headline_3pairwise_canonical.csv — run"
    display as error "  run_headline_3pairwise.do first; refusing to run without a living"
    display as error "  drift anchor for the unweighted baseline."
    error 601
}

* ----------------------------------------------------------------------------
* [0b] PROVENANCE — SHA256 of this script, the library and every input.
*      Printed here and written into the CSV, so a diagnostic can be tied to the
*      exact bytes that produced it (mtimes cannot: OneDrive rewrites them and
*      output/ is a junction to E:).
* ----------------------------------------------------------------------------
display as text "PROVENANCE — SHA256 of this script and of every input artifact"
cw_prov_print, dir("`SRC'") name("run_firm_wls_heterogeneity.do") tag("SCRIPT")
local SHA_SELF "`r(sha256)'"
cw_prov_print, dir("`SRC'") name("_country_weight_lib.do") tag("LIBRARY")
local SHA_LIB "`r(sha256)'"
cw_prov_print, dir("`OUT'") name("audit_c6_panel.dta") tag("INPUT ")
local SHA_PANEL "`r(sha256)'"
cw_prov_print, dir("`OUT'") name("firm_wls_weights.dta") tag("INPUT ")
local SHA_WLSW "`r(sha256)'"
cw_prov_print, dir("`OUT'") name("headline_3pairwise_canonical.csv") tag("ANCHOR")
local SHA_ANCH "`r(sha256)'"

* ----------------------------------------------------------------------------
* [1] Living drift anchor (read at run time, NEVER hardcoded).
*     Read FIRST, into empty memory, so no preserve/restore is needed.
* ----------------------------------------------------------------------------
quietly import delimited using "`OUT'/headline_3pairwise_canonical.csv", ///
    clear varnames(1) case(preserve)
foreach v in spec denom timing fe b3 {
    capture confirm variable `v'
    if _rc != 0 {
        display as error "headline_3pairwise_canonical.csv lacks the LOCKED 2026-08-08"
        display as error "  column `v' — stale layout; re-run run_headline_3pairwise.do."
        error 459
    }
}
quietly keep if spec == "primary" & denom == "global" & timing == "slag" & fe == "fq_gq_ig"
if _N != 1 {
    display as error "no unique primary/global/slag/fq_gq_ig row in the canonical CSV."
    error 459
}
local B3_ANCHOR = b3[1]
display as text "canonical primary anchor b3 = `B3_ANCHOR'"

* ----------------------------------------------------------------------------
* [2] Panel + weights
* ----------------------------------------------------------------------------
use "`OUT'/audit_c6_panel.dta", clear

capture confirm variable w_prev
local has_wprev = (_rc == 0)
if !`has_wprev' {
    capture confirm file "`OUT'/firm_wls_weights.dta"
    if _rc != 0 {
        display as error "DEPENDENCY NOT DELIVERED: no weight column and no"
        display as error "  output/firm_wls_weights.dta."
        display as error "  audit_c6_panel.dta does not carry w_prev (build_audit_panel_f1f2f7.py"
        display as error "  uses it to form cum0..cum4 and drops it), so the lagged-holdings"
        display as error "  weight must be supplied by e1 as:"
        display as error "     firm_str, hgroup, rdate (%tc), w_prev, i_prev_usd"
        display as error "  REFUSING to substitute a different weight — a WLS whose weight is"
        display as error "  not the lagged position is not the diagnostic that was specified."
        error 601
    }
    quietly merge m:1 firm_str hgroup rdate using "`OUT'/firm_wls_weights.dta", ///
        keep(master match) keepusing(w_prev i_prev_usd) generate(_mw)
    quietly count if _mw == 1
    local n_nomatch = r(N)
    display as text "weight merge: `n_nomatch' panel rows without a weight (they drop from"
    display as text "  every weighted fit; the unweighted baseline is unaffected)."
    quietly drop _mw
    capture confirm variable w_prev
    if _rc != 0 {
        display as error "firm_wls_weights.dta carries no w_prev column — dependency contract broken."
        error 111
    }
}
capture confirm variable i_prev_usd
local has_iprev = (_rc == 0)

* ----------------------------------------------------------------------------
* [3] Primary-spec construction (identical to run_headline_3pairwise.do)
* ----------------------------------------------------------------------------
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
gen double us_cn      = us*cn_lag
gen double us_cn_slag = us*cn_lag*s_lag

* ---- entry-margin census: what the aweight will delete --------------------
quietly count if !missing(dw, cn_lag, s_lag)
local n_est = r(N)
quietly count if !missing(dw, cn_lag, s_lag) & w_prev == 0
local n_zero_w = r(N)
quietly count if !missing(dw, cn_lag, s_lag) & w_prev == 0 & dw != 0
local n_entry = r(N)
quietly count if !missing(dw, cn_lag, s_lag) & missing(w_prev)
local n_miss_w = r(N)
display as text "estimation-sample rows: `n_est'"
display as text "  zero-weight rows deleted by aweight: `n_zero_w'  " ///
                "(of which entry events, w_prev==0 & dw!=0: `n_entry')"
display as text "  missing-weight rows: `n_miss_w'"

* ----------------------------------------------------------------------------
* [4] Fits
* ----------------------------------------------------------------------------
tempfile results
tempname P
* rung   = which step of the (i) full / (ii) same-sample unweighted / (iii) WLS
*          ladder this row is; delta_sample and delta_weight decompose the gap.
* WIDTHS ARE NOT ARBITRARY: "unweighted_full" is 15 characters and a str14 slot
* would SILENTLY truncate it to "unweighted_ful", and the provenance rows put a
* 32-character file name in `rung'.  Stata truncates strings in postfile without
* a warning, so every width here is set from the longest literal it must hold.
postfile `P' str20 spec str12 weight str32 rung double(b3 se p) long(N n_firms) ///
    double(b3_unweighted b3_samesample b3_anchor delta_sample delta_weight) ///
    long(n_zero_weight_rows n_entry_rows_dropped) ///
    str110 role str100 caveat_size str100 caveat_entry str200 better_test ///
    str64 sha256 using "`results'"

* ---- (0) UNWEIGHTED baseline, FULL sample + drift gate --------------------
display _newline "===== RUNG (i): UNWEIGHTED, FULL sample (must reproduce the canonical cell) ====="
qui reghdfe dw us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)
local b_un  = _b[us_cn_slag]
local se_un = _se[us_cn_slag]
local p_un  = 2*ttail(e(df_r), abs(`b_un'/`se_un'))
local N_un  = e(N)
tempvar smp0
quietly gen byte `smp0' = e(sample)
quietly levelsof firm_n if `smp0' == 1, local(_lf)
local nf_un : word count `_lf'
local rel = abs(`b_un'/`B3_ANCHOR' - 1)
display as text "  b3=" %12.5e `b_un' "  p=" %6.4f `p_un' "  N=" %9.0gc `N_un' ///
                "  rel vs canonical anchor=" %8.2e `rel'
if `rel' > 1e-6 {
    display as error "DRIFT GATE FAILED: the unweighted baseline (`b_un') does not"
    display as error "  reproduce the canonical primary cell (`B3_ANCHOR'), rel=`rel'."
    display as error "  Stale panel or stale canonical CSV — refusing to write a"
    display as error "  heterogeneity diagnostic against a moving baseline."
    error 459
}
post `P' ("unweighted_full") ("none") ("(i) full sample") ///
         (`b_un') (`se_un') (`p_un') (`N_un') (`nf_un') ///
         (`b_un') (.) (`B3_ANCHOR') (.) (.) (0) (0) ///
         ("`ROLE'") ("`CAV1'") ("`CAV2'") ("`BETTER'") ("`SHA_SELF'")

* ---- (1) THE LADDER on the lagged PORTFOLIO WEIGHT ------------------------
* Order matters: the WEIGHTED fit is run FIRST so that rung (ii) can be defined
* as reghdfe's OWN e(sample) from it.  Reconstructing "w_prev > 0 and the model
* variables non-missing" by hand would silently differ whenever reghdfe also
* drops a singleton FE group.
display _newline "===== RUNG (iii): WLS [aw = w_prev] — lagged portfolio weight ====="
qui reghdfe dw us_cn us_cn_slag [aw=w_prev], absorb(fq gq ig) vce(cluster firm_n rd_m)
local b_w  = _b[us_cn_slag]
local se_w = _se[us_cn_slag]
local p_w  = 2*ttail(e(df_r), abs(`b_w'/`se_w'))
local N_w  = e(N)
tempvar smp1
quietly gen byte `smp1' = e(sample)
quietly levelsof firm_n if `smp1' == 1, local(_lf)
local nf_w : word count `_lf'

display _newline "===== RUNG (ii): UNWEIGHTED on the WLS sample (the missing middle rung) ====="
qui reghdfe dw us_cn us_cn_slag if `smp1' == 1, absorb(fq gq ig) vce(cluster firm_n rd_m)
local b_ss  = _b[us_cn_slag]
local se_ss = _se[us_cn_slag]
local p_ss  = 2*ttail(e(df_r), abs(`b_ss'/`se_ss'))
local N_ss  = e(N)
tempvar smp1b
quietly gen byte `smp1b' = e(sample)
quietly levelsof firm_n if `smp1b' == 1, local(_lf)
local nf_ss : word count `_lf'
if `N_ss' != `N_w' {
    display as error "SAME-SAMPLE GATE FAILED: the unweighted middle rung estimates on"
    display as error "  `N_ss' rows but the WLS on `N_w'.  Rung (ii) exists precisely to hold"
    display as error "  the sample fixed; if it does not, (ii) vs (iii) no longer isolates the"
    display as error "  weighting and the decomposition below would be wrong.  Investigate"
    display as error "  (aweight handling, singleton dropping) — do not report."
    error 459
}
local d_sample = `b_ss' - `b_un'
local d_weight = `b_w'  - `b_ss'
display as text "  (i)   unweighted, full      b3=" %12.5e `b_un' "  N=" %9.0gc `N_un'
display as text "  (ii)  unweighted, w>0 same  b3=" %12.5e `b_ss' "  N=" %9.0gc `N_ss'
display as text "  (iii) WLS,        w>0 same  b3=" %12.5e `b_w'  "  N=" %9.0gc `N_w'
display as text "  SAMPLE change (i)->(ii)  = " %12.5e `d_sample' ///
                "   [the entry margin leaving]"
display as text "  WEIGHT change (ii)->(iii)= " %12.5e `d_weight' ///
                "   [THE size-tilt number — the only one that speaks to heterogeneity]"
post `P' ("unweighted_wpos") ("none") ("(ii) same sample") ///
         (`b_ss') (`se_ss') (`p_ss') (`N_ss') (`nf_ss') ///
         (`b_un') (`b_ss') (`B3_ANCHOR') (`d_sample') (.) (`n_zero_w') (`n_entry') ///
         ("`ROLE'") ("`CAV1'") ("`CAV2'") ("`BETTER'") ("`SHA_SELF'")
post `P' ("wls_wpos") ("w_prev") ("(iii) weighted") ///
         (`b_w') (`se_w') (`p_w') (`N_w') (`nf_w') ///
         (`b_un') (`b_ss') (`B3_ANCHOR') (`d_sample') (`d_weight') (`n_zero_w') (`n_entry') ///
         ("`ROLE'") ("`CAV1'") ("`CAV2'") ("`BETTER'") ("`SHA_SELF'")

* ---- (2) THE SAME LADDER on lagged DOLLAR holdings ------------------------
if `has_iprev' {
    display _newline "===== RUNG (iii): WLS [aw = i_prev_usd] — lagged dollar holdings ====="
    quietly count if !missing(dw, cn_lag, s_lag) & i_prev_usd == 0
    local n_zero_i = r(N)
    quietly count if !missing(dw, cn_lag, s_lag) & i_prev_usd == 0 & dw != 0
    local n_entry_i = r(N)
    qui reghdfe dw us_cn us_cn_slag [aw=i_prev_usd], absorb(fq gq ig) vce(cluster firm_n rd_m)
    local b_i  = _b[us_cn_slag]
    local se_i = _se[us_cn_slag]
    local p_i  = 2*ttail(e(df_r), abs(`b_i'/`se_i'))
    local N_i  = e(N)
    tempvar smp2
    quietly gen byte `smp2' = e(sample)
    quietly levelsof firm_n if `smp2' == 1, local(_lf)
    local nf_i : word count `_lf'

    display _newline "===== RUNG (ii): UNWEIGHTED on the dollar-WLS sample ====="
    qui reghdfe dw us_cn us_cn_slag if `smp2' == 1, absorb(fq gq ig) vce(cluster firm_n rd_m)
    local b_iss  = _b[us_cn_slag]
    local se_iss = _se[us_cn_slag]
    local p_iss  = 2*ttail(e(df_r), abs(`b_iss'/`se_iss'))
    local N_iss  = e(N)
    tempvar smp2b
    quietly gen byte `smp2b' = e(sample)
    quietly levelsof firm_n if `smp2b' == 1, local(_lf)
    local nf_iss : word count `_lf'
    if `N_iss' != `N_i' {
        display as error "SAME-SAMPLE GATE FAILED (dollar weight): `N_iss' vs `N_i' rows."
        error 459
    }
    local d_sample_i = `b_iss' - `b_un'
    local d_weight_i = `b_i'   - `b_iss'
    display as text "  (ii)  unweighted, i>0 same  b3=" %12.5e `b_iss' "  N=" %9.0gc `N_iss'
    display as text "  (iii) WLS,        i>0 same  b3=" %12.5e `b_i'   "  N=" %9.0gc `N_i'
    display as text "  SAMPLE change= " %12.5e `d_sample_i' "   WEIGHT change= " %12.5e `d_weight_i'
    post `P' ("unweighted_ipos") ("none") ("(ii) same sample") ///
             (`b_iss') (`se_iss') (`p_iss') (`N_iss') (`nf_iss') ///
             (`b_un') (`b_iss') (`B3_ANCHOR') (`d_sample_i') (.) (`n_zero_i') (`n_entry_i') ///
             ("`ROLE'") ("`CAV1'") ("`CAV2'") ("`BETTER'") ("`SHA_SELF'")
    post `P' ("wls_ipos") ("i_prev_usd") ("(iii) weighted") ///
             (`b_i') (`se_i') (`p_i') (`N_i') (`nf_i') ///
             (`b_un') (`b_iss') (`B3_ANCHOR') (`d_sample_i') (`d_weight_i') ///
             (`n_zero_i') (`n_entry_i') ///
             ("`ROLE'") ("`CAV1'") ("`CAV2'") ("`BETTER'") ("`SHA_SELF'")
}
if !`has_iprev' {
    display as text "i_prev_usd not available — the dollar-weight flavour is ABSENT from the"
    display as text "  CSV (not imputed).  e1 can supply it in firm_wls_weights.dta."
}

* ---- (3) PROVENANCE rows --------------------------------------------------
post `P' ("provenance") ("lib") ("_country_weight_lib.do") ///
         (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) ///
         ("") ("") ("") ("") ("`SHA_LIB'")
post `P' ("provenance") ("panel") ("audit_c6_panel.dta") ///
         (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) ///
         ("") ("") ("") ("") ("`SHA_PANEL'")
post `P' ("provenance") ("weights") ("firm_wls_weights.dta") ///
         (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) ///
         ("") ("") ("") ("") ("`SHA_WLSW'")
post `P' ("provenance") ("anchor") ("headline_3pairwise_canonical.csv") ///
         (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) ///
         ("") ("") ("") ("") ("`SHA_ANCH'")

postclose `P'

* ----------------------------------------------------------------------------
* [5] Write
* ----------------------------------------------------------------------------
use "`results'", clear
order spec weight rung b3 se p N n_firms b3_unweighted b3_samesample b3_anchor ///
      delta_sample delta_weight n_zero_weight_rows n_entry_rows_dropped ///
      role caveat_size caveat_entry better_test sha256
list spec weight rung b3 p N delta_sample delta_weight, noobs
export delimited using "`OUT'/firm_wls_heterogeneity.csv", replace

display as result _newline "{hline 78}"
display as result "HEADER STAMPED WITH THIS ARTIFACT (also in the role/caveat_*/better_test columns)"
display as result "{hline 78}"
display as result "ROLE : `ROLE'"
display as result "CAVEAT (i)  : `CAV1'"
display as result "CAVEAT (ii) : `CAV2'"
display as result "THREE RUNGS, AND ONLY ONE COMPARISON IS THE DIAGNOSTIC:"
display as result "  (i) unweighted full  ->  (ii) unweighted on w>0  =  delta_sample"
display as result "      This is the ENTRY MARGIN leaving.  It is a SAMPLE change and says"
display as result "      nothing about size heterogeneity."
display as result "  (ii) unweighted on w>0 -> (iii) WLS on w>0       =  delta_weight"
display as result "      THIS is the size tilt.  Quote delta_weight, never b3(wls) - b3(full),"
display as result "      which mixes the two."
display as result "BETTER TEST (stated, not run here): `BETTER'"
display as result "Read the weighted-vs-unweighted GAP as evidence on size heterogeneity"
display as result "only.  It says nothing about whether the country-level coefficient is"
display as result "right, and it cannot be used to adjudicate between the two levels."
display as result "PROVENANCE: spec==provenance rows carry the SHA256 of the library, the"
display as result "panel, the weights and the anchor; every estimation row carries this"
display as result "script's own SHA256 in the sha256 column."
display as text  "wrote output/firm_wls_heterogeneity.csv"
display "DONE run_firm_wls_heterogeneity.do"
