*==============================================================================
* run_shock_menu.do — ESTIMATION LEG of the SHOCK MENU (defects P1a / P1b / P1c)
*==============================================================================
* WHAT THIS IS. The headline shock S_t (05_combine_visualize.jl) is the monthly
* level-AR(1) residual of the Iacoviello-Tong bilateral USA|China GPR, sampled at
* the quarter-end month. Three documented defects motivate a MENU of alternative
* constructions:
*   P1a  the quarterly S_t series is serially correlated (acf1=+0.271,
*        LB(1) p=0.013, measured) — monthly level-AR(1) under-cleans GPR's
*        persistence, so the "shock" is partly predictable.
*   P1b  USA|China is DIRECTIONAL (US initiator -> China respondent); the official
*        data also carries China|USA and the two levels correlate only ~0.576,
*        while the hypothesis is about US-China tension BROADLY.
*   P1c  the AR is fit on the FULL sample incl. 2024-2026 -> look-ahead in a
*        generated regressor (doc F7 disclosure).
*
* THIS FILE DOES NOT BUILD THE MENU. build_shock_menu.py (w1) builds every variant
* at monthly frequency, brings it to the 82-quarter panel grid, standardizes each
* to mean 0 / sd 1 OVER THE 82 PANEL QUARTERS, and writes the diagnostics. This
* file only ESTIMATES: it merges the menu onto the P0 c6 panel, rebuilds the DDD
* triple per variant, and runs the 3-pairwise + it+gt battery.
*
* PRE-REGISTERED SELECTION RULE (ECHO of the builder header — the AUTHORITATIVE
* copy lives verbatim in build_shock_menu.py, and it was fixed BEFORE any
* regression ran). The preferred NEW construction = the no-look-ahead variant with
* the whitest QUARTERLY residual series on the 82 panel quarters, judged by
* LB(4) p (tie-break: LB(8) p, then acf1 magnitude), within the USA|China
* direction; the direction axis (bidirectional vs directional) is reported as a
* parallel column set, NOT selected on outcomes. beta3 results play NO role in
* selection. The baseline S_t stays the headline-continuity column regardless —
* its defects are disclosed, not hidden.
*   => NOTHING in this .do may be used to pick the preferred variant. The winner
*      is resolved MECHANICALLY by build_shock_menu.py, which writes the menu
*      COLUMN name to `OUT'/shock_menu_preferred.txt. The `PREFERRED' macro below
*      carries the same string and is CROSS-CHECKED against that file in section
*      0: if the two disagree, this file aborts rather than let a hand-edit
*      silently override the pre-registered selection.
*
* INPUT CONTRACT (the interface — this file does not wait on w1's internals):
*   `OUT'/shock_menu_quarterly.dta   (preferred)   or
*   `OUT'/shock_menu_quarterly.csv   (fallback)
*   STATA CANNOT READ PARQUET (same constraint as run_flow_decomp_step3.do), so
*   build_shock_menu.py MUST write a .dta (or .csv) sibling of
*   shock_menu_quarterly.parquet. One row per quarter. Columns:
*     - a quarter key: `quarter_end' (string "YYYY-MM-DD", %td, or %tc)
*     - one numeric column per STANDARDIZED variant; names are enumerated
*       GENERICALLY at runtime (see the resolver below), so w1 is free to name
*       them, subject only to the documented exclusion patterns.
*   Panel: `OUT'/c6_panel.dta (P0 vintage, 348,156 rows; rdate is %tc).
*
* OUTPUTS (new files only; no existing artifact is modified):
*   `OUT'/shockmenu_results.csv    one row per (variant, dv, FE set)
*   `OUT'/shockmenu_vce_diag.csv   B9 degenerate-two-way-cluster-VCE guard
*   Under SMOKE=1 BOTH gain a `_SMOKE' filename suffix and every row carries
*   smoke=1 plus the firm count, so a syntax-pass CSV can never be mistaken on
*   disk for the real one (run_ri_shockmenu.py reads shockmenu_results.csv as its
*   b3 anchor).
*   `OUT'/shock_menu_preferred.txt is an INPUT, written by build_shock_menu.py:
*   the mechanical hand-off of the pre-registered winner's MENU COLUMN name
*   (s_-prefixed), cross-checked against the `PREFERRED' macro below.
*
* INFERENCE. CRVE two-way cluster(firm, month) is the reported SE. The
* design-based ARBITER is run_ri_shockmenu.py (US-minus-NONUS collapse, free
* permutation + circular shift), which reads the b3 written here as its anchor.
* Where the B9 guard flags a degenerate VCE, the CRVE p in shockmenu_results.csv
* is NOT valid inference — read the RI p instead.
*
* DRIFT ANCHOR (LOCKED 2026-08-08 layout; fail-closed since REBUILD v3). Before
* the battery this file READS the canonical S_t diagnostic cells
* (spec=="diag" & denom=="global" & timing=="st", fe fq_gq_ig / fq_gq) from
* `OUT'/headline_3pairwise_canonical.csv (the LIVING SOURCE, same artifact cited
* by run_fourgroup.do) and hard-asserts the baseline_panel_shock_raw b3 against
* them to 6 significant figures for BOTH FE sets — those cells ARE this battery's
* baseline regressions (dw global x S_t). A missing file, stale layout, or absent
* anchor rows ABORT with exit 459: the menu must never run vintage-unprotected.
* Stale panel vintages sit in the same directory as c6_panel.dta, so a silent
* panel-vintage swap would otherwise produce a full menu of plausible-looking
* numbers with nothing to catch it. b3 is order-invariant, so an exact gate is
* legitimate; p and N are printed beside the anchors rather than gated, because
* reghdfe's singleton drop can legitimately move df_r.
*
* HONESTY. Every number verbatim from the run. Specs that fail reghdfe are
* written to the CSV with status=FAILED and reported as failures, never dropped
* silently.
*
*==============================================================================
* PRE-REGISTERED READING (header note, NOT a conclusion)
*==============================================================================
* Fixed BEFORE any menu regression ran; the AUTHORITATIVE copy is the identically
* worded block in build_shock_menu.py. The menu yields ~12 variant labels x 2 FE
* sets x 2 p-flavours, so what each outcome pattern MEANS is committed here in
* advance rather than narrated after the fact.
*
*  (a) ALL variants null, incl. the pre-registered preferred variant and the
*      bidirectional E_C_i_bidir column => the P0 deep null (b3=-5.28e-7,
*      CRVE p=0.763, RI p=0.801) is NOT an artifact of shock construction, and
*      P1a/P1b/P1c are DISCLOSED-AND-CLOSED defects, not open threats.
*  (b) The PREFERRED variant REJECTS while baseline does not => a
*      construction-sensitivity result on the pre-registered column ONLY, with the
*      caveat that the preferred column correlates just ~0.25 with the baseline
*      shock — close to an INDEPENDENT test, not a perturbation of the headline.
*  (c) BASELINE rejects while no no-look-ahead variant does => the headline is a
*      look-ahead / serial-correlation artifact and the NLA column GOVERNS.
*  (d) An E direction column rejects while its USA|China twin does not => a
*      direction-axis result reported in parallel, NEVER promoted to headline,
*      since E was excluded from selection by design.
*  (e) Any SINGLE one of ~12 columns crossing p<.05 with the rest null is roughly
*      ONE EXPECTED FALSE POSITIVE at this family size. Only the pre-registered
*      preferred variant and the continuity baseline carry weight, and p_circ
*      (run_ri_shockmenu.py) governs over p_free wherever they diverge.
*==============================================================================

clear all
set more off
set varabbrev off

local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

*------------------------------------------------------------------------------
* SMOKE default OFF. When 1, keep <=200 firms for a fast SYNTAX-ONLY pass
* (pairing preserved because the filter is by firm; inference is NOT valid).
*
* SMOKE IS SELF-DOCUMENTING ON DISK. A smoke run must never be mistakable for a
* real one by a downstream consumer: run_ri_shockmenu.py ingests
* shockmenu_results.csv as its b3 anchor, and a byte-indistinguishable 200-firm
* CSV would silently become the anchor of record (the project's documented
* stale-artifact failure mode). Two independent markers:
*   (1) the output FILENAMES gain a `_SMOKE' suffix, so a smoke run cannot
*       overwrite -- or be read as -- the real artifact at all; and
*   (2) every CSV row carries `smoke' and `n_firms' columns, mirroring the way
*       run_ri_shockmenu.py already records n_perm/seed per row.
* The canonical-headline drift gate is also downgraded to a warning under SMOKE
* (a 200-firm subsample cannot reproduce the full-panel b3 and must not pretend to).
*------------------------------------------------------------------------------
local SMOKE 0

local SFX ""
if `SMOKE' == 1 local SFX "_SMOKE"
local RESCSV "`OUT'/shockmenu_results`SFX'.csv"
local VCECSV "`OUT'/shockmenu_vce_diag`SFX'.csv"

*------------------------------------------------------------------------------
* PREFERRED-VARIANT MACRO. Filled from the pre-registered whiteness rule (whitest
* LB(4) p among the no-look-ahead USA|China variants), which build_shock_menu.py
* resolves. NOTHING in this .do may be used to pick it.
*
* IMPORTANT: the value is the MENU COLUMN name (s_-prefixed, as the resolver below
* yields), NOT the diagnostics `variant' key. shock_menu_diagnostics.csv names the
* winner `D_q_ar1_nla' while the resolver yields `s_D_q_ar1_nla'; writing the bare
* key here fails the PREF_IDX match and silently SKIPS the preferred lead spec
* while looking wired up.
*
* Resolution order (documented contract, macro first):
*   (1) the macro below, if non-empty;
*   (2) else `OUT'/shock_menu_preferred.txt, written by build_shock_menu.py.
* CONSISTENCY GUARD: if BOTH resolve and DISAGREE, this file ABORTS. A hand-set
* macro must never silently override the mechanically pre-registered selection
* (e.g. after build_shock_menu.py is re-run and the winner changes).
*------------------------------------------------------------------------------
local PREFERRED "s_D_q_ar1_nla"

*--- resolve (2) and cross-check against (1). Fail FAST, before the battery. ----
local PREF_FILE ""
capture confirm file "`OUT'/shock_menu_preferred.txt"
if _rc == 0 {
    tempname pf0
    file open `pf0' using "`OUT'/shock_menu_preferred.txt", read
    file read `pf0' line0
    file close `pf0'
    * trim() strips blanks but NOT a stray CR; subinstr removes it explicitly.
    local PREF_FILE = trim(subinstr(subinstr(`"`line0'"', char(13), "", .), char(10), "", .))
}
if "`PREF_FILE'" != "" {
    display _newline "PREFERRED variant read from shock_menu_preferred.txt: `PREF_FILE'"
}
if "`PREFERRED'" == "" & "`PREF_FILE'" != "" {
    local PREFERRED "`PREF_FILE'"
    display "PREFERRED macro was empty -> adopted the pre-registered file value: `PREFERRED'"
}
if "`PREFERRED'" != "" & "`PREF_FILE'" != "" & "`PREFERRED'" != "`PREF_FILE'" {
    display as error "================================================================"
    display as error "PREFERRED MISMATCH — refusing to run."
    display as error "  hand-set macro          : `PREFERRED'"
    display as error "  pre-registered file     : `PREF_FILE'"
    display as error "  shock_menu_preferred.txt is the MECHANICAL output of the"
    display as error "  pre-registered whiteness rule. A hand-set macro must never"
    display as error "  silently override it (e.g. after build_shock_menu.py re-ran"
    display as error "  and the winner changed). Fix the macro or re-run the builder."
    display as error "================================================================"
    exit 459
}
display "PREFERRED variant (menu column) in force: `PREFERRED'"

*==============================================================================
* 0. DRIFT ANCHOR — read the canonical S_t diagnostic cells from the LIVING
*    SOURCE output/headline_3pairwise_canonical.csv (LOCKED 2026-08-08 layout:
*    spec,denom,timing,fe,b3,se,p,N). The battery below runs dw x S_t, so its
*    anchors are the TWO diag/global/st cells:
*      fe=="fq_gq_ig" -> baseline_panel_shock_raw at fq gq ig (3pw)
*      fe=="fq_gq"    -> baseline_panel_shock_raw at fq gq    (itgt)
*    (the canonical writers run the identical regressions on audit_c6_panel;
*    reghdfe's estimation sample coincides with the c6_panel run, so a
*    6-sig-fig b3 gate is legitimate).
*    FAIL-CLOSED (REBUILD v3, 2026-08-08): a missing file, a stale layout, or
*    absent anchor rows ABORT with exit 459. The old behaviour — HAVE_ANCHOR=0
*    and a self-DISABLED gate — was exactly the stale-vintage failure mode the
*    layout guard exists to kill; a full menu must never run unprotected.
*==============================================================================
local ANCHORCSV "`OUT'/headline_3pairwise_canonical.csv"
local A3_B3 .
local A3_P  .
local A3_N  .
local AI_B3 .
local AI_P  .
local AI_N  .
* (nothing is in memory yet — `clear all' above — so no preserve/restore needed;
*  section 1 loads the menu into a fresh dataset immediately after.)
capture confirm file "`ANCHORCSV'"
if _rc != 0 {
    display as error "CANONICAL ANCHOR FILE NOT FOUND: `ANCHORCSV'"
    display as error "  Run run_headline_3pairwise.do (or run_attribution_em.do) first."
    display as error "  Refusing to run the menu without vintage protection (fail-closed)."
    exit 459
}
quietly import delimited "`ANCHORCSV'", clear varnames(1) case(preserve) stringcols(1)
capture confirm variable spec denom timing fe b3 se p N
if _rc != 0 {
    display as error "STALE ANCHOR LAYOUT in `ANCHORCSV' — expected the LOCKED 2026-08-08"
    display as error "columns spec,denom,timing,fe,b3,se,p,N. Re-run run_headline_3pairwise.do."
    exit 459
}
quietly count if spec == "diag" & denom == "global" & timing == "st" & fe == "fq_gq_ig"
local _n3 = r(N)
quietly count if spec == "diag" & denom == "global" & timing == "st" & fe == "fq_gq"
local _ni = r(N)
if `_n3' != 1 | `_ni' != 1 {
    display as error "ANCHOR ROWS MISSING/AMBIGUOUS in `ANCHORCSV':"
    display as error "  diag/global/st/fq_gq_ig rows: `_n3'   diag/global/st/fq_gq rows: `_ni'  (each must be 1)"
    display as error "  The diag itgt-st row was added to all three writers on 2026-08-08"
    display as error "  (REBUILD v3); a CSV without it is a stale vintage. Re-run"
    display as error "  run_headline_3pairwise.do. Refusing to run unprotected (fail-closed)."
    exit 459
}
quietly summarize b3 if spec == "diag" & denom == "global" & timing == "st" & fe == "fq_gq_ig", meanonly
local A3_B3 = r(mean)
quietly summarize p  if spec == "diag" & denom == "global" & timing == "st" & fe == "fq_gq_ig", meanonly
local A3_P  = r(mean)
quietly summarize N  if spec == "diag" & denom == "global" & timing == "st" & fe == "fq_gq_ig", meanonly
local A3_N  = r(mean)
quietly summarize b3 if spec == "diag" & denom == "global" & timing == "st" & fe == "fq_gq", meanonly
local AI_B3 = r(mean)
quietly summarize p  if spec == "diag" & denom == "global" & timing == "st" & fe == "fq_gq", meanonly
local AI_P  = r(mean)
quietly summarize N  if spec == "diag" & denom == "global" & timing == "st" & fe == "fq_gq", meanonly
local AI_N  = r(mean)
local HAVE_ANCHOR 1
clear
display _newline "=== CANONICAL S_t DRIFT ANCHOR (living source: headline_3pairwise_canonical.csv, diag/global/st) ==="
display "  3pw_fq_gq_ig : b3=" %14.6e `A3_B3' "  p=" %8.6f `A3_P' "  N=" %12.0fc `A3_N'
display "  itgt_fq_gq   : b3=" %14.6e `AI_B3' "  p=" %8.6f `AI_P' "  N=" %12.0fc `AI_N'
display "  -> baseline_panel_shock_raw b3 will be HARD-ASSERTED to 6 significant figures."

*==============================================================================
* 1. LOAD THE MENU + RESOLVE THE VARIANT COLUMN LIST GENERICALLY
*==============================================================================
local MENUDTA "`OUT'/shock_menu_quarterly.dta"
local MENUCSV "`OUT'/shock_menu_quarterly.csv"

local menusrc ""
capture confirm file "`MENUDTA'"
if _rc == 0 {
    use "`MENUDTA'", clear
    local menusrc "`MENUDTA'"
}
else {
    capture confirm file "`MENUCSV'"
    if _rc == 0 {
        import delimited "`MENUCSV'", clear varnames(1) case(preserve)
        local menusrc "`MENUCSV'"
    }
    else {
        display as error "================================================================"
        display as error "MENU NOT FOUND. Stata cannot read parquet."
        display as error "  looked for: `MENUDTA'"
        display as error "         and: `MENUCSV'"
        display as error "build_shock_menu.py must write a .dta (or .csv) sibling of"
        display as error "shock_menu_quarterly.parquet before this .do can run."
        display as error "================================================================"
        exit 601
    }
}
display _newline "Menu loaded from: `menusrc'   (`c(N)' quarter rows, `c(k)' columns)"

*--- quarter key -> Stata %tq integer (handles string / %td / %tc) -------------
capture confirm variable quarter_end
if _rc {
    display as error "menu has no `quarter_end' column — the quarter key is part of the contract."
    exit 111
}
capture confirm string variable quarter_end
if _rc == 0 {
    gen double _qe_day = date(quarter_end, "YMD")
    local qsrc "string YMD"
}
else {
    local qfmt : format quarter_end
    if strpos("`qfmt'", "%tc") > 0 {
        gen double _qe_day = dofc(quarter_end)
        local qsrc "%tc -> dofc()"
    }
    else {
        gen double _qe_day = quarter_end
        local qsrc "%td (as-is)"
    }
}
format _qe_day %td
quietly count if missing(_qe_day)
if r(N) > 0 {
    display as error "`r(N)' menu rows have an unparseable quarter_end (`qsrc') — aborting."
    exit 459
}
gen int qtr = qofd(_qe_day)
format qtr %tq
quietly duplicates report qtr
if r(unique_value) != r(N) {
    display as error "menu quarter key is not unique (`r(N)' rows, `r(unique_value)' distinct quarters)."
    exit 459
}
display "  quarter key parsed as: `qsrc'"

*--- variant resolver ----------------------------------------------------------
* Preference order: S_* -> s_* -> every numeric column. Then the documented
* exclusions (metadata, raw GPR levels, pre-standardization / diagnostic columns).
* The resolved list is PRINTED so a naming drift in w1 can never pass unnoticed.
local cand ""
capture ds S_*
if _rc == 0 local cand "`r(varlist)'"
if "`cand'" == "" {
    capture ds s_*
    if _rc == 0 local cand "`r(varlist)'"
}
local resolver "S_* / s_* prefix"
if "`cand'" == "" {
    quietly ds, has(type numeric)
    local cand "`r(varlist)'"
    local resolver "all numeric columns minus exclusions"
}

local EXCL_EXACT "qtr _qe_day quarter_end quarter qdate q date year yr n obs nobs index _merge _mm"
local EXCL_PFX   "gpr raw_ sd_ acf lb_ ljung nq_ n_ p_"
local EXCL_SFX   "_raw _sd _unstd _pre _prestd _nostd _level"

local VARIANTS ""
foreach v of local cand {
    local lv = strlower("`v'")
    local skip 0
    foreach d of local EXCL_EXACT {
        if "`lv'" == "`d'" local skip 1
    }
    foreach pfx of local EXCL_PFX {
        local Lp = length("`pfx'")
        if substr("`lv'", 1, `Lp') == "`pfx'" local skip 1
    }
    foreach sfx of local EXCL_SFX {
        local L  = length("`lv'")
        local Ls = length("`sfx'")
        if `L' > `Ls' {
            if substr("`lv'", `L'-`Ls'+1, `Ls') == "`sfx'" local skip 1
        }
    }
    capture confirm numeric variable `v'
    if _rc local skip 1
    if `skip' == 0 local VARIANTS "`VARIANTS' `v'"
}
local VARIANTS = trim("`VARIANTS'")
local NV : word count `VARIANTS'

display _newline "=== MENU VARIANT RESOLVER (resolver: `resolver') ==="
if `NV' == 0 {
    display as error "no variant columns resolved from `menusrc' — check w1's column naming."
    exit 459
}
forval i = 1/`NV' {
    local v : word `i' of `VARIANTS'
    quietly summarize `v'
    display "  [`i'] `v'   n=" %4.0f r(N) "  mean=" %9.3e r(mean) "  sd=" %9.3e r(sd)
}
display "  -> `NV' variant column(s) will be estimated."

keep qtr `VARIANTS'
tempfile MENU
quietly save "`MENU'"

*==============================================================================
* 2. LOAD THE P0 PANEL, BUILD KEYS / FE / LEAD OUTCOME
*==============================================================================
use "`OUT'/c6_panel.dta", clear

if `SMOKE' == 1 {
    egen long _fid = group(firm_str)
    quietly keep if _fid <= 200
    drop _fid
    display as error "SMOKE MODE: restricted to <=200 firms — SYNTAX CHECK ONLY, results NOT valid."
}

*--- %tc gotcha: c6_panel.dta writes rdate as %tc (ms since 1960-01-01) via
*--- pandas to_stata(convert_dates={'rdate':'tc'}). dofc() is REQUIRED; applying
*--- mofd()/qofd() straight to a %tc value silently yields garbage. Guarded so a
*--- future %td vintage cannot pass through the wrong branch.
local rfmt : format rdate
if strpos("`rfmt'", "%tc") > 0 {
    gen double rd_day = dofc(rdate)
    local dsrc "%tc -> dofc()"
}
else if strpos("`rfmt'", "%td") > 0 {
    gen double rd_day = rdate
    local dsrc "%td (as-is)"
}
else {
    display as error "rdate has format `rfmt' — expected %tc (or %td). Refusing to guess."
    exit 459
}
format rd_day %td
gen int rd_m = mofd(rd_day)
gen int qtr  = qofd(rd_day)
format qtr %tq
display _newline "panel rdate handled as: `dsrc'"

egen long firm_n = group(firm_str)
quietly summarize firm_n, meanonly
local NFIRMS = r(max)
display "firms in the estimation panel: `NFIRMS'   (SMOKE=`SMOKE')"
egen long fq     = group(firm_str rd_day)
egen long gq     = group(hgroup rd_day)
egen long ig     = group(firm_str hgroup)

gen double us_cn          = us * cn_lag
gen double us_cn_base_raw = us * cn_lag * shock
label var us_cn          "us x CN(t-1)"
label var us_cn_base_raw "us x CN(t-1) x S_t  (BASELINE, UNSTANDARDIZED — continuity anchor)"

*--- lead outcome dw_{t+1}. c6_panel.dta carries no dw_lead1, so it is built here
*--- on the (firm x group) time series. tsset's F. operator respects the time
*--- index, so a gap yields missing rather than a silently misaligned lead.
capture tsset ig qtr
if _rc {
    display as error "tsset ig qtr failed (rc=`=_rc') — duplicate (firm,group,quarter)? lead spec will be SKIPPED."
    gen double dw_lead1 = .
    local HAS_LEAD 0
}
else {
    gen double dw_lead1 = F.dw
    local HAS_LEAD 1
    quietly count if !missing(dw_lead1)
    display "dw_lead1 built: `r(N)' non-missing of `c(N)' rows"
}

quietly levelsof qtr, local(_pq)
local NQ_PANEL : word count `_pq'
display "panel quarters: `NQ_PANEL'"

*==============================================================================
* 3. MERGE THE MENU. Every panel quarter must be covered — an unmatched quarter
*    would silently drop rows and change the estimation sample per variant.
*==============================================================================
* Name-collision guard: a menu column sharing a panel variable name (e.g. a
* variant literally called `shock') would make merge abort with a bare r(108);
* fail here instead, naming the offender.
local CLASH ""
foreach v of local VARIANTS {
    capture confirm variable `v'
    if _rc == 0 local CLASH "`CLASH' `v'"
}
if "`CLASH'" != "" {
    display as error "menu variant name(s) collide with panel variables:`CLASH'"
    display as error "  -> rename them in build_shock_menu.py (they must not shadow c6_panel columns)."
    exit 110
}

merge m:1 qtr using "`MENU'", gen(_mm)
quietly count if _mm == 1
local n_unmatched = r(N)
if `n_unmatched' > 0 {
    quietly levelsof qtr if _mm == 1, local(_missq)
    display as error "MENU DOES NOT COVER THE PANEL: `n_unmatched' panel rows have no menu row."
    display as error "  uncovered quarters (%tq codes): `_missq'"
    exit 459
}
quietly count if _mm == 2
local n_menuonly = r(N)
quietly drop if _mm == 2
drop _mm
display "menu merge: all `NQ_PANEL' panel quarters covered; `n_menuonly' menu-only quarter(s) dropped."

*--- per-variant triple + correlation with the panel's own raw shock ------------
egen byte _qtag = tag(qtr)
forval i = 1/`NV' {
    local v : word `i' of `VARIANTS'
    gen double sv`i' = us * cn_lag * `v'
    label var sv`i' "us x CN(t-1) x `v'"
    quietly corr shock `v' if _qtag == 1
    local rho`i' = r(rho)
}

*==============================================================================
* 4. ESTIMATION BATTERY + B9 GUARD
*    3-pairwise (fq gq ig) = headline FE; it+gt (fq gq) = companion.
*    Two-way cluster(firm, month) throughout.
*==============================================================================
* SMOKE marker travels ON DISK, in the filename AND in every row (see the SMOKE
* note at the top): a 200-firm CSV must never be mistakable for the real anchor.
tempname rh vh
file open `rh' using "`RESCSV'", write replace
file write `rh' "variant,dv,fe,b2_us_cn,se2,p2,b3_triple,se3,p3,N,n_clust_firm,n_clust_month,df_r,r2,corr_S_with_panel_shock,status,smoke,n_firms" _n
file open `vh' using "`VCECSV'", write replace
file write `vh' "variant,dv,fe,coef,b,se,se_valid,smoke,n_firms" _n

global SM_FAILED ""
global SM_DEGEN  ""

*------------------------------------------------------------------------------
* Row writer, expanded inline (Stata file handles are passed by name; keeping the
* writes inline keeps the audit trail flat and avoids handle-scope surprises).
*------------------------------------------------------------------------------
capture program drop _smrun
program define _smrun, rclass
    * args: dv triple FE
    args dv triple FE
    capture qui reghdfe `dv' us_cn `triple', absorb(`FE') vce(cluster firm_n rd_m)
    if _rc {
        return local status "FAILED_rc`=_rc'"
        return scalar ok = 0
        exit
    }
    return scalar ok  = 1
    return scalar b2  = _b[us_cn]
    return scalar se2 = _se[us_cn]
    return scalar b3  = _b[`triple']
    return scalar se3 = _se[`triple']
    return scalar N   = e(N)
    return scalar c1  = e(N_clust1)
    return scalar c2  = e(N_clust2)
    return scalar dfr = e(df_r)
    return scalar r2  = e(r2)
    return local status "ok"
end

display _newline "=============================================================================="
display "SHOCK MENU BATTERY — dv=dw, FE in {fq gq ig (3pw headline), fq gq (it+gt)}"
display "=============================================================================="

*--- ordered spec list: baseline continuity anchor first, then the menu ---------
local SPEC_LBL  "baseline_panel_shock_raw"
local SPEC_TRIP "us_cn_base_raw"
local SPEC_RHO  "1"
forval i = 1/`NV' {
    local v : word `i' of `VARIANTS'
    local SPEC_LBL  "`SPEC_LBL' `v'"
    local SPEC_TRIP "`SPEC_TRIP' sv`i'"
    local SPEC_RHO  "`SPEC_RHO' `rho`i''"
}
local NSPEC : word count `SPEC_LBL'

forval s = 1/`NSPEC' {
    local lbl  : word `s' of `SPEC_LBL'
    local trip : word `s' of `SPEC_TRIP'
    local rho  : word `s' of `SPEC_RHO'
    display _newline "--- `lbl' ---"
    foreach FE in "fq gq ig" "fq gq" {
        local felab = cond("`FE'" == "fq gq ig", "3pw_fq_gq_ig", "itgt_fq_gq")
        _smrun dw `trip' "`FE'"
        if r(ok) == 0 {
            local st "`r(status)'"
            display as error "    `felab': reghdfe `st' — reported as a FAILURE, not dropped."
            global SM_FAILED "$SM_FAILED `lbl':`felab'"
            file write `rh' "`lbl',dw,`felab',,,,,,,,,,,,`rho',`st',`SMOKE',`NFIRMS'" _n
            file write `vh' "`lbl',dw,`felab',us_cn,,,0,`SMOKE',`NFIRMS'" _n
            file write `vh' "`lbl',dw,`felab',triple,,,0,`SMOKE',`NFIRMS'" _n
            continue
        }
        local b2  = r(b2)
        local se2 = r(se2)
        local b3  = r(b3)
        local se3 = r(se3)
        local NN  = r(N)
        local c1  = r(c1)
        local c2  = r(c2)
        local dfr = r(dfr)
        local rr2 = r(r2)
        local p2 = .
        local p3 = .
        if (`se2' > 0 & !missing(`se2')) local p2 = 2*ttail(`dfr', abs(`b2'/`se2'))
        if (`se3' > 0 & !missing(`se3')) local p3 = 2*ttail(`dfr', abs(`b3'/`se3'))

        * ---- B9: degenerate two-way-cluster VCE detection ----
        local ok2 = (!missing(`se2') & `se2' > 0)
        local ok3 = (!missing(`se3') & `se3' > 0)
        file write `vh' "`lbl',dw,`felab',us_cn,`b2',`se2',`ok2',`SMOKE',`NFIRMS'" _n
        file write `vh' "`lbl',dw,`felab',triple,`b3',`se3',`ok3',`SMOKE',`NFIRMS'" _n
        if (`ok2' == 0 | `ok3' == 0) {
            global SM_DEGEN "$SM_DEGEN `lbl':`felab'"
            display as error "    >>> `felab': DEGENERATE VCE (SE missing/zero) — CRVE p INVALID. Use run_ri_shockmenu.py. <<<"
        }

        file write `rh' "`lbl',dw,`felab'," ///
            (cond(missing(`b2'), "", strtrim(strofreal(`b2', "%14.6e")))) "," ///
            (cond(missing(`se2'), "", strtrim(strofreal(`se2', "%14.6e")))) "," ///
            (cond(missing(`p2'), "", strtrim(strofreal(`p2', "%9.6f"))))  "," ///
            (cond(missing(`b3'), "", strtrim(strofreal(`b3', "%14.6e")))) "," ///
            (cond(missing(`se3'), "", strtrim(strofreal(`se3', "%14.6e")))) "," ///
            (cond(missing(`p3'), "", strtrim(strofreal(`p3', "%9.6f"))))  "," ///
            (cond(missing(`NN'), "", strtrim(strofreal(`NN', "%15.0f")))) "," ///
            (cond(missing(`c1'), "", strtrim(strofreal(`c1', "%15.0f")))) "," ///
            (cond(missing(`c2'), "", strtrim(strofreal(`c2', "%15.0f")))) "," ///
            (cond(missing(`dfr'), "", strtrim(strofreal(`dfr', "%15.0f")))) "," ///
            (cond(missing(`rr2'), "", strtrim(strofreal(`rr2', "%9.6f"))))  "," ///
            (cond(missing(`rho'), "", strtrim(strofreal(`rho', "%9.6f"))))  ",ok,`SMOKE',`NFIRMS'" _n

        display "    `felab': b3=" %10.3e `b3' "  se=" %10.3e `se3' ///
                "  p=" %6.4f `p3' "  N=" %9.0gc `NN' ///
                "  fclust=" %6.0f `c1' "  mclust=" %4.0f `c2'

        *---------------------------------------------------------------------
        * DRIFT GATE (baseline_panel_shock_raw only). b3 on the UNSTANDARDIZED
        * panel shock IS the canonical P0 headline coefficient, so it must equal
        * headline_3pairwise_canonical.csv to 6 significant figures. b3 is
        * order-invariant, so an exact gate is legitimate. p / N are PRINTED
        * beside the anchors, not gated (reghdfe singleton drops can move df_r).
        * A mismatch means a stale panel or the wrong vintage (c6_panel_preP0.dta
        * lives in the same directory) — abort rather than emit a full menu of
        * plausible-looking numbers built on the wrong data.
        *---------------------------------------------------------------------
        if `s' == 1 & `HAVE_ANCHOR' == 1 {
            local ANC_B3 = cond("`felab'" == "3pw_fq_gq_ig", `A3_B3', `AI_B3')
            local ANC_P  = cond("`felab'" == "3pw_fq_gq_ig", `A3_P',  `AI_P')
            local ANC_N  = cond("`felab'" == "3pw_fq_gq_ig", `A3_N',  `AI_N')
            local RELERR = abs(`b3'/`ANC_B3' - 1)
            display "    [DRIFT GATE] `felab': anchor b3=" %14.6e `ANC_B3' ///
                    "  observed b3=" %14.6e `b3' "  relerr=" %9.2e `RELERR'
            display "                 anchor p=" %8.6f `ANC_P' "  observed p=" %8.6f `p3' ///
                    "   |  anchor N=" %12.0fc `ANC_N' "  observed N=" %12.0fc `NN'
            if `SMOKE' == 1 {
                display as error "                 SMOKE run (`NFIRMS' firms) — gate DOWNGRADED to a warning."
            }
            else if `RELERR' > 1e-6 {
                display as error "=============================================================="
                display as error "DRIFT GATE FAILED — `felab' b3 does not match the canonical P0"
                display as error "headline (relerr=`RELERR' > 1e-6)."
                display as error "  living source : `ANCHORCSV'"
                display as error "  anchor b3     : `ANC_B3'"
                display as error "  observed b3   : `b3'"
                display as error "  STALE PANEL OR WRONG VINTAGE — check that `OUT'/c6_panel.dta"
                display as error "  is the P0 vintage and not c6_panel_preP0.dta. Refusing to"
                display as error "  continue: every menu number below would be untrustworthy."
                display as error "=============================================================="
                file close `rh'
                file close `vh'
                exit 459
            }
            else {
                display "                 -> PASS (b3 matches to 6 significant figures)."
            }
        }
    }
    display "    corr(S_variant, panel shock) over the `NQ_PANEL' panel quarters = " %8.6f `rho'
}

*==============================================================================
* 5. LEAD SPEC (dw_{t+1}) — baseline + the PREFERRED variant ONLY.
*    The preferred variant comes from the pre-registered diagnostics rule, never
*    from anything computed above.
*==============================================================================
* `PREFERRED' was resolved and cross-checked in section 0 (macro vs
* shock_menu_preferred.txt), so a mismatch has already aborted this run.

local PREF_IDX 0
if "`PREFERRED'" != "" {
    forval i = 1/`NV' {
        local v : word `i' of `VARIANTS'
        if "`v'" == "`PREFERRED'" local PREF_IDX = `i'
    }
    if `PREF_IDX' == 0 {
        display as error "PREFERRED='`PREFERRED'' is not among the resolved menu variants — lead spec for it SKIPPED."
    }
}
else {
    display as error _newline "PREFERRED variant NOT SET (macro empty and shock_menu_preferred.txt absent)."
    display as error "  -> preferred-variant LEAD spec SKIPPED (reported as skipped, not silently omitted)."
}

if `HAS_LEAD' == 0 {
    display as error "dw_lead1 unavailable — ALL lead specs skipped."
    file write `rh' "ALL,dw_lead1,NA,,,,,,,,,,,,,SKIPPED_no_lead,`SMOKE',`NFIRMS'" _n
}
else {
    display _newline "=============================================================================="
    display "LEAD SPEC — dv=dw_lead1 (pure lagged-shock outcome, no look-ahead in the DV)"
    display "=============================================================================="
    local LEAD_LBL  "baseline_panel_shock_raw"
    local LEAD_TRIP "us_cn_base_raw"
    local LEAD_RHO  "1"
    if `PREF_IDX' > 0 {
        local LEAD_LBL  "`LEAD_LBL' `PREFERRED'"
        local LEAD_TRIP "`LEAD_TRIP' sv`PREF_IDX'"
        local LEAD_RHO  "`LEAD_RHO' `rho`PREF_IDX''"
    }
    else {
        file write `rh' "PREFERRED_UNSET,dw_lead1,NA,,,,,,,,,,,,,SKIPPED_preferred_not_set,`SMOKE',`NFIRMS'" _n
    }
    local NLEAD : word count `LEAD_LBL'
    forval s = 1/`NLEAD' {
        local lbl  : word `s' of `LEAD_LBL'
        local trip : word `s' of `LEAD_TRIP'
        local rho  : word `s' of `LEAD_RHO'
        display _newline "--- `lbl' (lead) ---"
        foreach FE in "fq gq ig" "fq gq" {
            local felab = cond("`FE'" == "fq gq ig", "3pw_fq_gq_ig", "itgt_fq_gq")
            _smrun dw_lead1 `trip' "`FE'"
            if r(ok) == 0 {
                local st "`r(status)'"
                display as error "    `felab': reghdfe `st' — reported as a FAILURE."
                global SM_FAILED "$SM_FAILED `lbl':lead:`felab'"
                file write `rh' "`lbl',dw_lead1,`felab',,,,,,,,,,,,`rho',`st',`SMOKE',`NFIRMS'" _n
                file write `vh' "`lbl',dw_lead1,`felab',us_cn,,,0,`SMOKE',`NFIRMS'" _n
                file write `vh' "`lbl',dw_lead1,`felab',triple,,,0,`SMOKE',`NFIRMS'" _n
                continue
            }
            local b2  = r(b2)
            local se2 = r(se2)
            local b3  = r(b3)
            local se3 = r(se3)
            local NN  = r(N)
            local c1  = r(c1)
            local c2  = r(c2)
            local dfr = r(dfr)
            local rr2 = r(r2)
            local p2 = .
            local p3 = .
            if (`se2' > 0 & !missing(`se2')) local p2 = 2*ttail(`dfr', abs(`b2'/`se2'))
            if (`se3' > 0 & !missing(`se3')) local p3 = 2*ttail(`dfr', abs(`b3'/`se3'))
            local ok2 = (!missing(`se2') & `se2' > 0)
            local ok3 = (!missing(`se3') & `se3' > 0)
            file write `vh' "`lbl',dw_lead1,`felab',us_cn,`b2',`se2',`ok2',`SMOKE',`NFIRMS'" _n
            file write `vh' "`lbl',dw_lead1,`felab',triple,`b3',`se3',`ok3',`SMOKE',`NFIRMS'" _n
            if (`ok2' == 0 | `ok3' == 0) {
                global SM_DEGEN "$SM_DEGEN `lbl':lead:`felab'"
                display as error "    >>> `felab': DEGENERATE VCE — CRVE p INVALID. <<<"
            }
            file write `rh' "`lbl',dw_lead1,`felab'," ///
                (cond(missing(`b2'), "", strtrim(strofreal(`b2', "%14.6e")))) "," ///
                (cond(missing(`se2'), "", strtrim(strofreal(`se2', "%14.6e")))) "," ///
                (cond(missing(`p2'), "", strtrim(strofreal(`p2', "%9.6f"))))  "," ///
                (cond(missing(`b3'), "", strtrim(strofreal(`b3', "%14.6e")))) "," ///
                (cond(missing(`se3'), "", strtrim(strofreal(`se3', "%14.6e")))) "," ///
                (cond(missing(`p3'), "", strtrim(strofreal(`p3', "%9.6f"))))  "," ///
                (cond(missing(`NN'), "", strtrim(strofreal(`NN', "%15.0f")))) "," ///
                (cond(missing(`c1'), "", strtrim(strofreal(`c1', "%15.0f")))) "," ///
                (cond(missing(`c2'), "", strtrim(strofreal(`c2', "%15.0f")))) "," ///
                (cond(missing(`dfr'), "", strtrim(strofreal(`dfr', "%15.0f")))) "," ///
                (cond(missing(`rr2'), "", strtrim(strofreal(`rr2', "%9.6f"))))  "," ///
                (cond(missing(`rho'), "", strtrim(strofreal(`rho', "%9.6f"))))  ",ok,`SMOKE',`NFIRMS'" _n
            display "    `felab': b3=" %10.3e `b3' "  se=" %10.3e `se3' ///
                    "  p=" %6.4f `p3' "  N=" %9.0gc `NN'
        }
    }
}

file close `rh'
file close `vh'
display _newline "Wrote `RESCSV'"
display "Wrote `VCECSV'"

*==============================================================================
* 6. CLOSING SUMMARY — failures and degenerate VCEs surfaced, never buried.
*==============================================================================
display _newline "=============================================================================="
if "$SM_FAILED" != "" {
    display as error "SPECS THAT FAILED reghdfe (status=FAILED_* in the CSV): $SM_FAILED"
}
else {
    display "All specs estimated (no reghdfe failures)."
}
if "$SM_DEGEN" != "" {
    display as error "DEGENERATE two-way-cluster VCE in: $SM_DEGEN"
    display as error "  -> their CRVE p in shockmenu_results.csv is NOT valid inference."
    display as error "  -> read the design-based p from run_ri_shockmenu.py (ri_shockmenu.csv)."
}
else {
    display "B9 guard: no degenerate two-way-cluster VCE detected."
}
display _newline "REMINDER: variant selection is governed ONLY by the pre-registered whiteness"
display "rule on shock_menu_diagnostics.csv. Nothing in shockmenu_results.csv may be"
display "used to choose the preferred construction. The baseline column stays the"
display "headline-continuity anchor regardless of what the menu shows."
display "PREFERRED variant in force this run: `PREFERRED'  (from shock_menu_preferred.txt)"
* HAVE_ANCHOR is always 1 here: section 0 exits 459 (fail-closed) when the
* canonical anchor cannot be read, so an unprotected run cannot reach this line.
display "DRIFT GATE: baseline b3 matched headline_3pairwise_canonical.csv (diag/global/st) to 6 sig figs."
if `SMOKE' == 1 {
    display as error "SMOKE=1 (`NFIRMS' firms): outputs carry the _SMOKE suffix and smoke=1 rows."
    display as error "  These numbers are a SYNTAX PASS. They are NOT valid inference and must"
    display as error "  NOT be used as the run_ri_shockmenu.py anchor."
}

display _newline "=============================================================================="
display "PRE-REGISTERED READING (fixed in this file's header BEFORE any regression ran)"
display "=============================================================================="
display "  (a) ALL variants null, incl. the preferred variant and the bidirectional"
display "      E_C_i_bidir column => the P0 deep null (b3=-5.28e-7, CRVE p=0.763,"
display "      RI p=0.801) is NOT an artifact of shock construction; P1a/P1b/P1c are"
display "      DISCLOSED-AND-CLOSED defects, not open threats."
display "  (b) PREFERRED rejects while baseline does not => construction-sensitivity"
display "      result on the pre-registered column ONLY. It correlates just ~0.25 with"
display "      the baseline shock, so it is close to an INDEPENDENT test, not a"
display "      perturbation of the headline."
display "  (c) BASELINE rejects while no no-look-ahead variant does => the headline is"
display "      a look-ahead / serial-correlation artifact; the NLA column GOVERNS."
display "  (d) An E direction column rejects while its USA|China twin does not => a"
display "      direction-axis result, reported in parallel, NEVER promoted to headline"
display "      (E was excluded from the selection rule by design)."
display "  (e) Any SINGLE one of ~12 columns crossing p<.05 with the rest null is"
display "      roughly ONE EXPECTED FALSE POSITIVE at this family size. Only the"
display "      pre-registered preferred variant and the continuity baseline carry"
display "      weight, and p_circ governs over p_free wherever they diverge."
display "=============================================================================="
display "Done."
