* ============================================================================
* run_country_weight_ddd.do — the LOCKED country-level PORTFOLIO-WEIGHT DDD
* (task e2 item 1, 2026-08-10).  CODE ONLY at time of writing: nothing in this
* file has been executed.
*
* ---------------------------------------------------------------------------
* WHAT IS BEING ESTIMATED
* ---------------------------------------------------------------------------
* OUTCOME.  W_{c,g,t} = SUM_{i in c} w_{i,g,t} = country c's share of investor
* group g's GLOBAL equity book;  dW_{c,g,t} = W_{c,g,t} - W_{c,g,t-1}.
* The MAIN outcome is the GLOBAL-denominator dW (dw_global).  The EU-denominator
* twin (dw_eu) is a WITHIN-EUROPE REALLOCATION diagnostic.  d ln(USD holdings)
* (dlog_usd) is SUPPLEMENTARY and is a DIFFERENT ESTIMAND — it mixes price,
* quantity and composition changes and is NEVER to be called a net flow.
*
* SPECIFICATION (locked by the researcher; implemented verbatim):
*   dW_{c,g,t} = beta * US_g x M_{k,c,t-1} x S_{t-1}
*              +  gam * US_g x M_{k,c,t-1}
*              + alpha_{cg} + alpha_{ct} + alpha_{gt} + e_{c,g,t}
*   for k in {M1, M2, M3}.  Two-way cluster (country, quarter).
*
*   reghdfe dw_global usm`k' usm`k's, absorb(cg ct gt) vce(cluster ctry qtr)
*
* The three pairwise FE are the EXACT country-level analogue of the firm-level
* headline's fq (firm x quarter) / gq (group x quarter) / ig (firm x group):
*   cg = country x group   <->  ig
*   ct = country x quarter <->  fq
*   gt = group   x quarter <->  gq
*
* WHAT IS ABSORBED (stated because it is the whole point of the FE choice):
*   alpha_ct kills everything varying only by (c,t) — including the M_{k,c,t-1}
*            MAIN EFFECT and every country-quarter common shock to the country's
*            investable book (this is what the first-pass US-minus-NONUS dlog
*            difference was TRYING to do and could not, because the two groups
*            hold different firms with different weights);
*   alpha_gt kills S_{t-1} and every group-quarter aggregate;
*   alpha_cg kills the country-group level.
*   SURVIVING: US_g x M (the DiD term) and US_g x M x S_{t-1} (the DDD term).
*
* ---------------------------------------------------------------------------
* HYPOTHESIS FAMILY — EXACTLY THREE
* ---------------------------------------------------------------------------
* The MAIN family is {M1, M2, M3} on dw_global under this DDD.  CRVE, the Webb
* score bootstrap and the randomization arbiter are three INFERENCE METHODS for
* those same three hypotheses, not extra hypotheses.  The EU-denominator block,
* the dlog_usd block and the US-only block are SECONDARY / diagnostic and are
* tagged family="secondary" in the CSV precisely so they cannot be folded into
* the family by a later reader.
*
* ---------------------------------------------------------------------------
* INFERENCE
* ---------------------------------------------------------------------------
* se_crve/p_crve : two-way cluster (country, quarter).  ~28 country clusters —
*                  anti-conservative.  Reported, not the arbiter.
* p_wild         : Webb-weight SCORE bootstrap, 9,999 reps, clustered on
*                  COUNTRY.  Engine = the machinery already in
*                  run_country_panel.do, reused through _country_weight_lib.do
*                  and PROVEN equal to it at run time by scoreboot_selfcheck
*                  (which re-derives that file's own M1/us_dlog p_wild).
*                  boottest cannot be used: it refuses to run after reghdfe with
*                  more than one absorbed FE group, and every block here absorbs
*                  two or three (measured 2026-08-10 — it returns WITHOUT r(p)
*                  and WITHOUT an error code, i.e. it fails silently).
*                  One-dimension bootstrap by design; wild_cluster="country" is
*                  stamped on every row.
* ARBITER        : circular-shift randomization inference with max-|t| FWER
*                  across the three M statistics — run_ri_country_ddd.py.  That
*                  file reads THIS file's CSV as its living drift anchor, so run
*                  this one first.
*
* INPUT : output/country_weight_panel.dta   (e1)
* OUTPUT: output/country_weight_ddd.csv
*         columns: family, measure, outcome, spec, b, se_crve, p_crve, N,
*                  n_countries, n_quarters, p_wild, wild_engine, wild_cluster,
*                  wild_reps, t, df_r, seed
* Rotation discipline: refuses to overwrite; rotate to *_cwpre first.
* ============================================================================

clear all
set more off
set seed 20260810

local SRC  "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive"
local OUT  "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
local LIB  "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/_country_weight_lib.do"
local REPS 9999

include "`LIB'"

* ----------------------------------------------------------------------------
* [0] Rotation guard + input presence
* ----------------------------------------------------------------------------
cw_guard, target("`OUT'/country_weight_ddd.csv") pre("`OUT'/country_weight_ddd_cwpre.csv")

capture confirm file "`OUT'/country_weight_panel.dta"
if _rc != 0 {
    display as error "DEPENDENCY: output/country_weight_panel.dta not found."
    display as error "  It is task e1's output.  This file estimates only; it builds nothing."
    error 601
}

* ----------------------------------------------------------------------------
* [0b] PROVENANCE — SHA256 of this script, the library and the panel, into the
*      log.  run_ri_country_ddd.py hashes THIS file's CSV in turn, so the chain
*      panel -> ddd.csv -> ri.csv is traceable end to end by bytes.
* ----------------------------------------------------------------------------
display as text "PROVENANCE — SHA256 of this script and of every input artifact"
cw_prov_print, dir("`SRC'") name("run_country_weight_ddd.do") tag("SCRIPT")
cw_prov_print, dir("`SRC'") name("_country_weight_lib.do") tag("LIBRARY")
cw_prov_print, dir("`OUT'") name("country_weight_panel.dta") tag("INPUT ")

* ----------------------------------------------------------------------------
* [1] REUSE PROOF for the score-bootstrap engine (clears memory; runs first)
* ----------------------------------------------------------------------------
scoreboot_selfcheck, out("`OUT'")
local sb_status "`r(status)'"
display as text "score-bootstrap engine self-check status: `sb_status'"

* ----------------------------------------------------------------------------
* [2] Load + prepare
* ----------------------------------------------------------------------------
use "`OUT'/country_weight_panel.dta", clear
cw_prep
local n_c_all = r(n_countries)
local n_q_all = r(n_quarters)

* optional-outcome availability
local has_eu = 1
capture confirm variable dw_eu
if _rc != 0 {
    local has_eu = 0
}
local has_dlog = 1
capture confirm variable dlog_usd
if _rc != 0 {
    local has_dlog = 0
}

local WENG "scoreboot_webb (Kline-Santos null-restricted cluster scores; Webb 6-pt weights)"

* ----------------------------------------------------------------------------
* [3] Postfile
* ----------------------------------------------------------------------------
tempfile results
tempname P
postfile `P' str9 family str2 measure str10 outcome str20 spec ///
    double(b se_crve p_crve) long N int(n_countries n_quarters) ///
    double p_wild str120 wild_engine str8 wild_cluster long wild_reps ///
    double(t df_r) long seed using "`results'"

* ----------------------------------------------------------------------------
* [4] MAIN FAMILY — dW on the GLOBAL denominator, 3 pairwise FE
*     THE hypothesis family: exactly these three rows.
* ----------------------------------------------------------------------------
display _newline "{hline 78}"
display "MAIN FAMILY: dw_global on US x M_k x S_{t-1}, absorb(cg ct gt), cluster(ctry qtr)"
display "{hline 78}"
forvalues k = 1/3 {
    cw_run, y(dw_global) x3(usm`k's) x2(usm`k') absorb(cg ct gt) clus(ctry qtr)
    local b   = r(b)
    local se  = r(se)
    local p   = r(p)
    local t   = r(t)
    local df  = r(df)
    local N   = r(N)
    local nc  = r(n_countries)
    local nq  = r(n_quarters)
    scoreboot_webb, y(dw_global) xint(usm`k's) ctrl(usm`k') absorb(cg ct gt) ///
                    clustvar(ctry) reps(`REPS')
    local pw  = r(p)
    display as text "  M`k'  b=" %12.5e `b' "  se=" %12.5e `se' ///
                    "  p_crve=" %6.4f `p' "  p_wild=" %6.4f `pw' ///
                    "  N=" %7.0gc `N' "  C=`nc'  Q=`nq'"
    post `P' ("main") ("M`k'") ("dw_global") ("ddd_3pairwise") ///
             (`b') (`se') (`p') (`N') (`nc') (`nq') ///
             (`pw') ("`WENG'") ("country") (`REPS') (`t') (`df') (20260810)
}

* ----------------------------------------------------------------------------
* [5] SECONDARY — EU denominator (within-Europe reallocation DIAGNOSTIC)
* ----------------------------------------------------------------------------
if `has_eu' {
    display _newline "SECONDARY (diagnostic): dw_eu — WITHIN-EUROPE REALLOCATION, not the main estimand"
    forvalues k = 1/3 {
        cw_run, y(dw_eu) x3(usm`k's) x2(usm`k') absorb(cg ct gt) clus(ctry qtr)
        local b = r(b)
        local se = r(se)
        local p = r(p)
        local t = r(t)
        local df = r(df)
        local N = r(N)
        local nc = r(n_countries)
        local nq = r(n_quarters)
        scoreboot_webb, y(dw_eu) xint(usm`k's) ctrl(usm`k') absorb(cg ct gt) ///
                        clustvar(ctry) reps(`REPS')
        local pw = r(p)
        display as text "  M`k'  b=" %12.5e `b' "  p_crve=" %6.4f `p' "  p_wild=" %6.4f `pw' "  N=" %7.0gc `N'
        post `P' ("secondary") ("M`k'") ("dw_eu") ("ddd_3pairwise") ///
                 (`b') (`se') (`p') (`N') (`nc') (`nq') ///
                 (`pw') ("`WENG'") ("country") (`REPS') (`t') (`df') (20260810)
    }
}
if !`has_eu' {
    display as text "SECONDARY dw_eu block SKIPPED — column dw_eu absent from the e1 panel."
}

* ----------------------------------------------------------------------------
* [6] SECONDARY — d log(USD holdings).  DIFFERENT ESTIMAND.  Never a net flow.
* ----------------------------------------------------------------------------
if `has_dlog' {
    display _newline "SECONDARY (different estimand): dlog_usd — mixes price, quantity and"
    display          "composition changes.  Reported for continuity with the first-pass"
    display          "country panel ONLY.  NOT a net flow, NOT in the main family."
    forvalues k = 1/3 {
        cw_run, y(dlog_usd) x3(usm`k's) x2(usm`k') absorb(cg ct gt) clus(ctry qtr)
        local b = r(b)
        local se = r(se)
        local p = r(p)
        local t = r(t)
        local df = r(df)
        local N = r(N)
        local nc = r(n_countries)
        local nq = r(n_quarters)
        scoreboot_webb, y(dlog_usd) xint(usm`k's) ctrl(usm`k') absorb(cg ct gt) ///
                        clustvar(ctry) reps(`REPS')
        local pw = r(p)
        display as text "  M`k'  b=" %12.5e `b' "  p_crve=" %6.4f `p' "  p_wild=" %6.4f `pw' "  N=" %7.0gc `N'
        post `P' ("secondary") ("M`k'") ("dlog_usd") ("ddd_3pairwise") ///
                 (`b') (`se') (`p') (`N') (`nc') (`nq') ///
                 (`pw') ("`WENG'") ("country") (`REPS') (`t') (`df') (20260810)
    }
}
if !`has_dlog' {
    display as text "SECONDARY dlog_usd block SKIPPED — column dlog_usd absent from the e1 panel."
}

* ----------------------------------------------------------------------------
* [7] SECONDARY — US-ONLY, NON-DIFFERENCED.
*     Keep only g = US.  With one group the group-interacted FE collapse:
*       alpha_cg -> country FE, alpha_gt -> quarter FE, alpha_ct is NOT
*       identified (one obs per (c,t)).  So the block is
*         dW^{US}_{c,t} = beta * M_{k,c,t-1} x S_{t-1} + gam * M_{k,c,t-1}
*                       + alpha_c + alpha_t + e
*       S_{t-1} main effect absorbed by alpha_t; M main effect kept as control.
*     THIS BLOCK DOES NOT NET OUT COUNTRY-QUARTER COMMON SHOCKS.  It is the
*     level counterpart of the DDD, reported so the reader can see how much of
*     the DDD comes from the differencing.  It is NOT a headline candidate.
* ----------------------------------------------------------------------------
display _newline "SECONDARY: US-only (non-differenced), absorb(ctry qtr), cluster(ctry qtr)"
forvalues k = 1/3 {
    tempvar msk
    quietly gen double `msk' = m`k'_lag * s_lag
    cw_run, y(dw_global) x3(`msk') x2(m`k'_lag) absorb(ctry qtr) clus(ctry qtr) ifcond(if us == 1)
    local b = r(b)
    local se = r(se)
    local p = r(p)
    local t = r(t)
    local df = r(df)
    local N = r(N)
    local nc = r(n_countries)
    local nq = r(n_quarters)
    scoreboot_webb, y(dw_global) xint(`msk') ctrl(m`k'_lag) absorb(ctry qtr) ///
                    clustvar(ctry) ifcond(if us == 1) reps(`REPS')
    local pw = r(p)
    display as text "  M`k'  b=" %12.5e `b' "  p_crve=" %6.4f `p' "  p_wild=" %6.4f `pw' "  N=" %7.0gc `N'
    post `P' ("secondary") ("M`k'") ("dw_global") ("usonly_c_t_fe") ///
             (`b') (`se') (`p') (`N') (`nc') (`nq') ///
             (`pw') ("`WENG'") ("country") (`REPS') (`t') (`df') (20260810)
}

postclose `P'

* ----------------------------------------------------------------------------
* [8] Write + caveats
* ----------------------------------------------------------------------------
use "`results'", clear
order family measure outcome spec b se_crve p_crve N n_countries n_quarters ///
      p_wild wild_engine wild_cluster wild_reps t df_r seed
list family measure outcome spec b p_crve p_wild N, noobs sepby(outcome)
export delimited using "`OUT'/country_weight_ddd.csv", replace

display as result _newline "{hline 78}"
display as result "READING RULES stamped with this artifact"
display as result "{hline 78}"
display as result " 1. MAIN FAMILY = the three family==main rows only (M1/M2/M3 on dw_global)."
display as result "    CRVE, score bootstrap and RI are three INFERENCE METHODS for those same"
display as result "    three hypotheses — not six or nine hypotheses."
display as result " 2. p_crve is two-way (country, quarter) with ~28 country clusters and is"
display as result "    anti-conservative.  p_wild is bootstrapped over COUNTRY only (no standard"
display as result "    two-way score bootstrap exists); wild_cluster records that."
display as result " 3. The ARBITER is run_ri_country_ddd.py (circular-shift RI, max-|t| FWER"
display as result "    across the three M statistics).  Run it after this file; it reads this"
display as result "    CSV as its living drift anchor."
display as result " 4. dw_eu = within-Europe reallocation diagnostic.  dlog_usd = a DIFFERENT"
display as result "    ESTIMAND (price + quantity + composition), never a net flow.  US-only ="
display as result "    non-differenced level block that does NOT net out country-quarter common"
display as result "    shocks.  None of the three may be promoted into the main family."
display as result " 5. score-bootstrap engine self-check: `sb_status'"
display as text  "wrote output/country_weight_ddd.csv"
display "DONE run_country_weight_ddd.do"
