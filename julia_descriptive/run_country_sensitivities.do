* ============================================================================
* run_country_sensitivities.do — the three REPORTED sensitivities of the
* country-level portfolio-weight DDD (task e2 item 3, 2026-08-10).
* CODE ONLY at time of writing: nothing here has been executed.
*
* ---------------------------------------------------------------------------
* THE ONE RULE THAT GOVERNS THIS FILE
* ---------------------------------------------------------------------------
* NOTHING HERE MAY SELECT A HEADLINE.  Every row is a REPORTED sensitivity of
* the locked main family (M1/M2/M3 on dw_global, absorb(cg ct gt), two-way
* cluster).  In particular:
*   - a leave-one-country-out fit is NEVER a headline.  Dropping the country
*     that moves the coefficient most and reporting the remainder is exactly
*     the practice this file exists to make visible, so the LOO block reports
*     the RANGE and the LARGEST SINGLE-COUNTRY CHANGE and nothing else is
*     licensed;
*   - the outlier-excluded twin is reported ALONGSIDE the main panel, both
*     rows, never instead of it;
*   - the M2-common-coverage restriction is a SAMPLE comparability check, not
*     a better sample.
*
* ---------------------------------------------------------------------------
* THE THREE SENSITIVITIES
* ---------------------------------------------------------------------------
* (a) LEAVE-ONE-COUNTRY-OUT.  For each k and each country c0, re-fit dropping
*     c0.  Reported: b for every c0, plus summary rows carrying min/max/range
*     and the largest |b_loo - b_full| with the country that produced it.
*     WHY IT MATTERS HERE: the outcome is a SHARE OF A GLOBAL BOOK, so a single
*     large-weight country (GB, DE, FR, CH) mechanically carries much of the
*     cross-sectional variance in dW.
*
* (b) IMPOSSIBLE-POSITION OUTLIERS.  The v3.1 DQ residual is the set of
*     sub-threshold rows that fail adj_holding > adj_shares_out and that the
*     shipped (market-cap-proxy-based) filter deliberately leaves in.  Its size
*     is NOT quoted here from a memo: e1 re-derives it at run time from
*     output/dq_variants/dq_census.csv (V2 \ V1) and stamps the count, the
*     dollars and the most concentrated holder-country-quarter into
*     country_weight_gate_summary.csv.  On the current v3.1 data that residual is
*     301 rows / $30.38bn, NOT the "373 rows / $47.0bn / 196 cells / Indonesia
*     2017Q4 = 65.9%" of the older memo, which was measured on a superseded
*     vintage.  At FIRM level these rows are noise; COUNTRY AGGREGATION AMPLIFIES
*     THEM, which is the whole reason this sensitivity exists.
*
*     HOW THE TWIN IS DELIVERED.  e1 does NOT ship a separate _nodq panel.  It
*     rebuilds the weights under the PROXY-FREE regime (drop every row with
*     adj_holding > adj_shares_out AND adj_shares_out > 0, regardless of size =
*     the V2 regime of robustness_dq_filter_variants.py) and carries the result
*     as a TWIN OUTCOME COLUMN, dw_global_x, in the SAME panel — numerator AND
*     denominator both rebuilt, per e1's exactness proof.  Running the twin as an
*     outcome on the same rows is strictly better than a separate panel here: the
*     regressors, the FE and the estimation sample are held FIXED, so the
*     with-vs-without comparison isolates the outcome change instead of confusing
*     it with a sample change.  Both legs are run; both are reported.
*
*     SAMPLE GATE (added 2026-08-10, external review item 8).  "Held FIXED" is
*     now ENFORCED, not assumed.  Both legs are estimated on the COMMON
*     INTERSECTION (rows where BOTH outcomes are non-missing), the main-only and
*     twin-only counts are posted into the CSV, and the two fits are HARD-GATED
*     to the same N.  The twin's b_ref is the main outcome ON THAT SAME SAMPLE,
*     so delta_vs_ref is an outcome effect and cannot absorb a sample effect.
*     The previous vintage printed the mismatch and carried on.
*
* (c) M2 COMMON COVERAGE — A COMMON *FIRM* SET, NOT A COMMON CELL SET.
*     (rewritten 2026-08-10 after the external review; do not re-litigate.)
*
*     WHAT WAS WRONG.  This block used to re-fit M1 and M3 `if m2cov == 1',
*     where m2cov was the country-quarter flag !missing(m2_lag).  That ALIGNS
*     THE CELLS and nothing else.  Inside a kept cell, M1 and M3 were still
*     averaged over ALL firms while M2 was averaged over the market-cap-covered
*     ones, so a country-quarter in which 10% of firms carry a market cap still
*     passed the restriction and the "common sample" was common in name only.
*
*     WHAT IS DONE NOW.  e1 section [5b] RECOMPUTES M1 and M3 inside every
*     country-quarter over EXACTLY the firms M2 uses (non-missing, strictly
*     positive market cap) and ships them as m1cov_lag / m3cov_lag, with
*     m2cov_lag == m2_lag because M2 already IS the covered-firm measure.
*     cw_prep builds usm{k}c / usm{k}cs from those, and THIS block fits the DDD
*     on them.  The three measures are therefore compared on ONE FIRM SET.
*     The old cell-set restriction is still run, as a REPORTED CONTRAST row
*     (variant restrict_to_m2_cellset), so the reader can see how little it did.
*
*     TWO COVERAGE FACTS, TWO GRAINS, BOTH STATED (cw_prep re-derives both):
*       COUNTRY-QUARTER GRAIN — at v3.1 M2 has FEWER missing country-quarter
*         cells (27) than M1 (106).  "M2 is the sparse one" is FLATLY WRONG at
*         this grain: M1's denominator SUM(sc_links) is what goes to zero.
*       FIRM GRAIN — M2's market-cap coverage is only 42.9% of firm-quarters in
*         2023, so at firm grain M2 really is built on a minority of the firms
*         M1 and M3 see.  That is the comparability problem (c) measures.
*       The two are not in conflict.  They are different grains, and quoting
*       either alone misdescribes the data.
*
* INPUT : output/country_weight_panel.dta   (e1: dw_global, the COMMON-FIRM-SET
*                                            measures m{1,2,3}cov_lag, AND the
*                                            proxy-free twin outcome dw_global_x)
* OUTPUT: output/country_weight_sensitivity.csv
*         columns: sens, measure, outcome, variant, dropped_country, b, se_crve,
*                  p_crve, N, n_countries, n_quarters, b_ref, delta_vs_ref,
*                  reldelta_vs_ref, t, df_r, note
*         sens=="provenance" rows carry a file name in `variant' and its SHA256
*         in `note', so the CSV names the bytes that produced it.
* Rotation discipline: refuses to overwrite; rotate to *_cwpre first.
* ============================================================================

clear all
set more off
set seed 20260810

local SRC "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
local LIB "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/_country_weight_lib.do"

include "`LIB'"

* ----------------------------------------------------------------------------
* [0] Guards
* ----------------------------------------------------------------------------
cw_guard, target("`OUT'/country_weight_sensitivity.csv") ///
          pre("`OUT'/country_weight_sensitivity_cwpre.csv")

capture confirm file "`OUT'/country_weight_panel.dta"
if _rc != 0 {
    display as error "DEPENDENCY: output/country_weight_panel.dta not found (task e1)."
    error 601
}

* ----------------------------------------------------------------------------
* [0b] PROVENANCE — SHA256 of this script, of the library it includes and of
*      every input artifact.  Printed here and posted into the CSV below, so a
*      set of sensitivity numbers can be tied to the exact bytes that made it.
* ----------------------------------------------------------------------------
display as text "PROVENANCE — SHA256 of this script and of every input artifact"
cw_prov_print, dir("`SRC'") name("run_country_sensitivities.do") tag("SCRIPT")
local SHA_SELF "`r(sha256)'"
cw_prov_print, dir("`SRC'") name("_country_weight_lib.do") tag("LIBRARY")
local SHA_LIB "`r(sha256)'"
cw_prov_print, dir("`OUT'") name("country_weight_panel.dta") tag("INPUT ")
local SHA_PANEL "`r(sha256)'"

tempfile results
tempname P
* NOTE: outcome is str14, not str10 — the twin outcome name "dw_global_x" is 11
* characters and a str10 slot would SILENTLY TRUNCATE it to "dw_global_", which
* would read in the CSV as if the outlier leg had been run on the main outcome.
postfile `P' str12 sens str2 measure str14 outcome str28 variant str12 dropped_country ///
    double(b se_crve p_crve) long N int(n_countries n_quarters) ///
    double(b_ref delta_vs_ref reldelta_vs_ref t df_r) str90 note using "`results'"

* ----------------------------------------------------------------------------
* [1] Reference fits on the MAIN panel (the locked main family)
* ----------------------------------------------------------------------------
use "`OUT'/country_weight_panel.dta", clear
cw_prep
* capture the coverage census NOW: r() is overwritten by the first cw_run
local NA_M1   = r(na_cells_m1)
local NA_M2   = r(na_cells_m2)
local NA_M3   = r(na_cells_m3)
local FIRMCOV = r(firm_cov_share)
local N_CELLS = r(n_countries) * r(n_quarters)

forvalues k = 1/3 {
    cw_run, y(dw_global) x3(usm`k's) x2(usm`k') absorb(cg ct gt) clus(ctry qtr)
    local bref`k' = r(b)
    display as text "reference M`k': b=" %12.5e r(b) "  p=" %6.4f r(p) "  N=" %7.0gc r(N)
    post `P' ("reference") ("M`k'") ("dw_global") ("main_panel_full") ("") ///
             (r(b)) (r(se)) (r(p)) (r(N)) (r(n_countries)) (r(n_quarters)) ///
             (r(b)) (0) (0) (r(t)) (r(df)) ///
             ("locked main family: absorb(cg ct gt), vce(cluster ctry qtr)")
}

* ----------------------------------------------------------------------------
* [2] (a) LEAVE-ONE-COUNTRY-OUT
*     NOTE ON WHY THE LOOP DROPS BY sec_country AND NOT BY ctry: ctry is an
*     egen group code that would silently re-map if the panel's country set
*     ever changed; the human-readable ISO code is the stable key and is what
*     lands in the CSV.
* ----------------------------------------------------------------------------
display _newline "{hline 78}"
display "(a) LEAVE-ONE-COUNTRY-OUT — reported as a range, NEVER as a headline"
display "{hline 78}"
quietly levelsof sec_country, local(CTRIES)
local n_ctry : word count `CTRIES'
display as text "countries: `n_ctry'"

* one reusable flag: passing a quoted string condition through an option would
* nest double quotes inside ifcond(); a byte flag cannot be mis-parsed.
tempvar keep
quietly gen byte `keep' = 1

forvalues k = 1/3 {
    local bmin = .
    local bmax = .
    local dmax = 0
    local cmax ""
    foreach c0 of local CTRIES {
        quietly replace `keep' = (sec_country != "`c0'")
        cw_run, y(dw_global) x3(usm`k's) x2(usm`k') absorb(cg ct gt) ///
                clus(ctry qtr) ifcond(if `keep' == 1)
        local b = r(b)
        local d = `b' - `bref`k''
        local rd = .
        if `bref`k'' != 0 {
            local rd = `d'/`bref`k''
        }
        if `bmin' == . {
            local bmin = `b'
            local bmax = `b'
        }
        if `b' < `bmin' {
            local bmin = `b'
        }
        if `b' > `bmax' {
            local bmax = `b'
        }
        if abs(`d') > `dmax' {
            local dmax = abs(`d')
            local cmax "`c0'"
        }
        post `P' ("loo") ("M`k'") ("dw_global") ("drop_one_country") ("`c0'") ///
                 (`b') (r(se)) (r(p)) (r(N)) (r(n_countries)) (r(n_quarters)) ///
                 (`bref`k'') (`d') (`rd') (r(t)) (r(df)) ///
                 ("one country dropped; NOT a headline candidate")
    }
    local rng = `bmax' - `bmin'
    display as text "  M`k' LOO: b in [" %12.5e `bmin' ", " %12.5e `bmax' ///
                    "]  range=" %12.5e `rng' "  largest single-country change=" ///
                    %12.5e `dmax' " (`cmax')"
    * summary rows: the two numbers the spec asks to REPORT
    post `P' ("loo_summary") ("M`k'") ("dw_global") ("range_min") ("") ///
             (`bmin') (.) (.) (.) (.) (.) (`bref`k'') (`bmin'-`bref`k'') (.) (.) (.) ///
             ("min b across the `n_ctry' LOO fits")
    post `P' ("loo_summary") ("M`k'") ("dw_global") ("range_max") ("") ///
             (`bmax') (.) (.) (.) (.) (.) (`bref`k'') (`bmax'-`bref`k'') (.) (.) (.) ///
             ("max b across the `n_ctry' LOO fits")
    post `P' ("loo_summary") ("M`k'") ("dw_global") ("largest_single_change") ("`cmax'") ///
             (.) (.) (.) (.) (.) (.) (`bref`k'') (`dmax') (.) (.) (.) ///
             ("max |b_loo - b_full|, attained by dropping this country")
}

* ----------------------------------------------------------------------------
* [3] (c) M2 COMMON COVERAGE — A COMMON *FIRM* SET
*     THE REAL RESTRICTION.  usm{k}c / usm{k}cs are built by cw_prep from e1's
*     m{k}cov_lag: M1 and M3 RECOMPUTED inside each country-quarter over exactly
*     the market-cap-covered firms M2 is built on (M2 needs no restriction — it
*     already IS that measure, which cw_prep and e1 both gate).  Fitting the DDD
*     on these three puts all three measures on ONE FIRM SET.
*
*     The OLD cell-set restriction is ALSO run, as a reported contrast, so the
*     difference between "same cells" and "same firms" is visible in the CSV
*     rather than argued about in prose.  It is NOT the (c) sensitivity.
* ----------------------------------------------------------------------------
display _newline "{hline 78}"
display "(c) M2 COMMON COVERAGE — all three measures on the SAME FIRM SET"
display "{hline 78}"
quietly count
local n_rows_all = r(N)
quietly count if m2cov_cell == 1
local n_cell = r(N)
quietly count if !missing(m1cov_lag) & !missing(m2cov_lag) & !missing(m3cov_lag)
local n_cov3 = r(N)
display as text "  panel rows: `n_rows_all' | old CELL-set flag == 1: `n_cell'" ///
                " | all three COMMON-FIRM-SET measures present: `n_cov3'"

* ---- (c.1) THE SENSITIVITY: common firm set --------------------------------
forvalues k = 1/3 {
    cw_run, y(dw_global) x3(usm`k'cs) x2(usm`k'c) absorb(cg ct gt) clus(ctry qtr)
    local b = r(b)
    local d = `b' - `bref`k''
    local rd = .
    if `bref`k'' != 0 {
        local rd = `d'/`bref`k''
    }
    display as text "  M`k' COMMON FIRM SET: b=" %12.5e `b' "  p=" %6.4f r(p) ///
                    "  N=" %7.0gc r(N) "  (own-sample b=" %12.5e `bref`k'' ")"
    post `P' ("m2coverage") ("M`k'") ("dw_global") ("common_firm_set") ("") ///
             (`b') (r(se)) (r(p)) (r(N)) (r(n_countries)) (r(n_quarters)) ///
             (`bref`k'') (`d') (`rd') (r(t)) (r(df)) ///
             ("M1/M3 RECOMPUTED on the market-cap-covered firms M2 uses; M2 unchanged")
}

* ---- (c.2) REPORTED CONTRAST: the old cell-set restriction ------------------
* Kept so the weak restriction and the real one sit side by side in one CSV.
* Reading rule: a cell-set row that agrees with the own-sample row proves
* NOTHING about firm-set comparability — that is the whole point.
forvalues k = 1/3 {
    cw_run, y(dw_global) x3(usm`k's) x2(usm`k') absorb(cg ct gt) ///
            clus(ctry qtr) ifcond(if m2cov_cell == 1)
    local b = r(b)
    local d = `b' - `bref`k''
    local rd = .
    if `bref`k'' != 0 {
        local rd = `d'/`bref`k''
    }
    display as text "  M`k' old CELL set  : b=" %12.5e `b' "  p=" %6.4f r(p) ///
                    "  N=" %7.0gc r(N) "  (contrast only)"
    post `P' ("m2coverage") ("M`k'") ("dw_global") ("restrict_to_m2_cellset") ("") ///
             (`b') (r(se)) (r(p)) (r(N)) (r(n_countries)) (r(n_quarters)) ///
             (`bref`k'') (`d') (`rd') (r(t)) (r(df)) ///
             ("WEAK cell-set restriction, REPORTED CONTRAST ONLY — not sensitivity (c)")
}

* ---- (c.3) the two coverage facts, at their two grains, on the record -------
* Both grains, never one alone: at COUNTRY-QUARTER grain M2 has FEWER missing
* cells than M1, while at FIRM grain M2 is built on a minority of the firms.
forvalues k = 1/3 {
    post `P' ("coverage") ("M`k'") ("n_na_cells") ("country_quarter_grain") ("") ///
             (`NA_M`k'') (.) (.) (`N_CELLS') (.) (.) (.) (.) (.) (.) (.) ///
             ("b=n missing country-quarter cells, N=n cells; compare M1 vs M2 before calling either sparse")
}
post `P' ("coverage") ("M2") ("cov_share") ("firm_grain") ("") ///
         (`FIRMCOV') (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) ///
         ("share of firm-quarters with a market cap = the firm set M2 is built on")

* ----------------------------------------------------------------------------
* [4] (b) IMPOSSIBLE-POSITION OUTLIER TWIN (proxy-free leg)
*     Runs on the MAIN panel still in memory: the twin is an OUTCOME COLUMN
*     (dw_global_x), rebuilt by e1 from outlier-excluded weights with BOTH the
*     numerator and the denominator re-formed.  Regressors, FE and sample are
*     held fixed BY CONSTRUCTION (same rows) AND BY GATE (same estimation N), so
*     the comparison isolates the outcome change.  Three rows per measure reach
*     the CSV: the full-panel reference (block [1]), the main outcome on the
*     common sample, and the twin on that same common sample.
* ----------------------------------------------------------------------------
display _newline "{hline 78}"
display "(b) IMPOSSIBLE-POSITION OUTLIERS — with vs without the proxy-free drop"
display "{hline 78}"
local has_twin = 1
capture confirm variable dw_global_x
if _rc != 0 {
    local has_twin = 0
}
if `has_twin' {
    * ---- SAMPLE GATE (external review item 8) ------------------------------
    * An earlier vintage merely PRINTED the main-vs-twin sample mismatch and
    * carried on.  A twin fitted on different rows differs from the main leg by
    * SAMPLE as well as by OUTCOME, and "with vs without the outlier rows" then
    * confounds the two.  So: both legs are estimated on the COMMON INTERSECTION
    * (both outcomes non-missing), the mismatch counts are posted, and the two
    * fits are hard-gated to the SAME N — the number that actually decides
    * whether the comparison isolates the outcome change.
    quietly count if !missing(dw_global) & missing(dw_global_x)
    local n_only_main = r(N)
    quietly count if missing(dw_global) & !missing(dw_global_x)
    local n_only_twin = r(N)
    tempvar both
    quietly gen byte `both' = !missing(dw_global) & !missing(dw_global_x)
    quietly count if `both' == 1
    local n_both = r(N)
    display as text "  rows: main-only `n_only_main' | twin-only `n_only_twin' | " ///
                    "BOTH `n_both'  -> both legs are fitted on the BOTH set"
    post `P' ("outlier") ("") ("sample_census") ("common_intersection") ("") ///
             (`n_both') (.) (.) (`n_only_main') (.) (.) (`n_only_twin') (.) (.) (.) (.) ///
             ("b=n_both, N=n_main_only, b_ref=n_twin_only; both legs run on the intersection")

    forvalues k = 1/3 {
        * (i) MAIN outcome on the intersection: the reference the twin must be
        *     read against.  It equals the full-sample reference only when the
        *     mismatch counts above are zero.
        cw_run, y(dw_global) x3(usm`k's) x2(usm`k') absorb(cg ct gt) ///
                clus(ctry qtr) ifcond(if `both' == 1)
        local bmain = r(b)
        local Nmain = r(N)
        local dmain = `bmain' - `bref`k''
        post `P' ("outlier") ("M`k'") ("dw_global") ("main_on_common_sample") ("") ///
                 (`bmain') (r(se)) (r(p)) (`Nmain') (r(n_countries)) (r(n_quarters)) ///
                 (`bref`k'') (`dmain') (.) (r(t)) (r(df)) ///
                 ("reference leg for the twin: SAME rows, SAME regressors, canonical outcome")

        * (ii) TWIN outcome on the SAME rows
        cw_run, y(dw_global_x) x3(usm`k's) x2(usm`k') absorb(cg ct gt) ///
                clus(ctry qtr) ifcond(if `both' == 1)
        local b = r(b)
        local Ntwin = r(N)
        local d = `b' - `bmain'
        local rd = .
        if `bmain' != 0 {
            local rd = `d'/`bmain'
        }
        display as text "  M`k' outlier-excluded: b=" %12.5e `b' "  p=" %6.4f r(p) ///
                        "  N=" %7.0gc `Ntwin' "  (same-sample main b=" %12.5e `bmain' ///
                        ", full-panel main b=" %12.5e `bref`k'' ")"
        if `Ntwin' != `Nmain' {
            display as error "SAMPLE GATE FAILED for M`k': the twin leg estimates on `Ntwin'"
            display as error "  rows and the same-sample main leg on `Nmain', even after"
            display as error "  restricting to the rows where BOTH outcomes are present."
            display as error "  The with-vs-without comparison would then confound an outcome"
            display as error "  change with a sample change, which is the one thing this"
            display as error "  sensitivity exists to avoid.  Fix e1 (dw_global_x must be"
            display as error "  present on exactly the rows dw_global is) — do not report it."
            error 459
        }
        post `P' ("outlier") ("M`k'") ("dw_global_x") ("excl_adjhold_gt_sharesout") ("") ///
                 (`b') (r(se)) (r(p)) (`Ntwin') (r(n_countries)) (r(n_quarters)) ///
                 (`bmain') (`d') (`rd') (r(t)) (r(df)) ///
                 ("proxy-free leg on the COMMON sample; b_ref = same-sample main outcome")
    }
}
if !`has_twin' {
    display as error "DEPENDENCY MISSING: column dw_global_x in country_weight_panel.dta."
    display as error "  Sensitivity (b) requires e1's proxy-free outlier-excluded TWIN"
    display as error "  OUTCOME (V2 regime of robustness_dq_filter_variants.py: drop every"
    display as error "  row with adj_holding > adj_shares_out AND adj_shares_out > 0,"
    display as error "  regardless of size).  The other two sensitivities are still written;"
    display as error "  the outlier rows are ABSENT from the CSV rather than faked."
    forvalues k = 1/3 {
        post `P' ("outlier") ("M`k'") ("dw_global_x") ("UNAVAILABLE") ("") ///
                 (.) (.) (.) (.) (.) (.) (`bref`k'') (.) (.) (.) (.) ///
                 ("dw_global_x absent from the e1 panel — dependency not delivered")
    }
}

* ----------------------------------------------------------------------------
* [4b] PROVENANCE rows — the CSV names the bytes that produced it
* ----------------------------------------------------------------------------
post `P' ("provenance") ("") ("sha256") ("run_country_sensitivities.do") ("") ///
         (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) ("`SHA_SELF'")
post `P' ("provenance") ("") ("sha256") ("_country_weight_lib.do") ("") ///
         (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) ("`SHA_LIB'")
post `P' ("provenance") ("") ("sha256") ("country_weight_panel.dta") ("") ///
         (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) (.) ("`SHA_PANEL'")

postclose `P'

* ----------------------------------------------------------------------------
* [5] Write
* ----------------------------------------------------------------------------
use "`results'", clear
order sens measure outcome variant dropped_country b se_crve p_crve N ///
      n_countries n_quarters b_ref delta_vs_ref reldelta_vs_ref t df_r note
list sens measure variant dropped_country b delta_vs_ref if sens != "loo", noobs
export delimited using "`OUT'/country_weight_sensitivity.csv", replace

display as result _newline "{hline 78}"
display as result "REPORTING RULES stamped with this artifact"
display as result "{hline 78}"
display as result " * sens==loo rows exist to be READ AS A RANGE.  No single LOO fit is a"
display as result "   headline, and the loo_summary rows are the only two numbers the write-up"
display as result "   should quote per measure (range, largest single-country change)."
display as result " * sens==outlier gives BOTH legs, estimated on the COMMON INTERSECTION of"
display as result "   the two outcomes and hard-gated to the same N, so the difference is an"
display as result "   OUTCOME change and not a sample change.  Read the twin against the"
display as result "   variant==main_on_common_sample row, not against the full-panel reference."
display as result " * sens==m2coverage: variant==common_firm_set IS sensitivity (c) — M1 and M3"
display as result "   recomputed on the market-cap-covered firms M2 uses.  variant=="
display as result "   restrict_to_m2_cellset is the OLD, WEAK cell-set restriction, reported"
display as result "   only as a contrast; agreement there proves nothing about firm-set"
display as result "   comparability.  Neither makes M2's sample the preferred sample."
display as result " * sens==coverage carries BOTH coverage facts.  At COUNTRY-QUARTER grain M2"
display as result "   has FEWER missing cells than M1 (M1's SUM(sc_links) denominator is what"
display as result "   goes to zero); at FIRM grain M2 is built on a minority of firms (42.9% of"
display as result "   firm-quarters carry a market cap in 2023).  Quote both or neither."
display as result " * Inference here is CRVE only.  The arbiter for the main family remains"
display as result "   run_ri_country_ddd.py; these rows are descriptive stability, not tests."
display as result " * sens==provenance carries the SHA256 of this script, of the library and of"
display as result "   the panel, so these numbers name the bytes that produced them."
display as text  "wrote output/country_weight_sensitivity.csv"
display "DONE run_country_sensitivities.do"
