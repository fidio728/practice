* run_extensive_margin.do — Essay 2 EXTENSIVE-MARGIN outcomes (A1/G1, 2026-08-03).
*
* MOTIVE. Every existing Essay-2 outcome is value/weight-based (dw, flow, share).
* The divestment literature's FIRST margin — the COUNT of distinct holders
* (Hong-Kacperczyk holder counts, Chen-Hong-Stein breadth) — was never built:
* 04_us_ownership_european.jl collapses fund_id away (SUM(adj_mv) GROUP BY) before
* any count can form. This file estimates the US-vs-NONUS differential response of
* the extensive margin to the China shock, MIRRORING THE HEADLINE conventions
* exactly so the count result is directly comparable to the weight headline:
*   reghdfe DV us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
* (3-pairwise saturated FE) with the it+gt (fq gq) companion for every DV.
*
* PRE-REGISTERED READING (design note, NOT a conclusion). The partial-divestment
* sign on us_cn_shock is DV-SPECIFIC: `exit` is a held->0 (full-drop) indicator and
* therefore moves OPPOSITE to the count/breadth DVs. For US high-CN firms on the
* shock, divestment predicts:
*     b3 < 0  for d_breadth, d_nh, init   (breadth / holders / initiations fall)
*     b3 > 0  for exit                    (held positions drop to zero MORE often)
*   A NEGATIVE exit b3 means FEWER US exits = anti-divestment, NOT divestment;
*   reading the exit column with the breadth sign is the project's wrong-sign trap.
*   null across all four                  = the intensive-margin null's completeness
*       is strengthened (the margin with the strongest descriptive prior also nulls).
*   sign FLIPPED from divestment (breadth/holders/inits UP, exits DOWN for US)
*                                         = anti-H2.1, same direction as the weight margin.
*   No new subsample splits beyond the riskset robustness (predetermined-moderator
*   rule). The pre-B7 §5.5 descriptive +11.26pp gap is SUPERSEDED — NEVER CITE it.
*
* DVs (per the locked panel contract):
*   (a) d_breadth  PRIMARY   + d_nh companion   — paired, all firm-quarters
*   (b) exit LPM             + init LPM         — CONDITIONED on held_lag, so the
*       US-NONUS pairing breaks: an unpaired firm-quarter cell is an fq-FE
*       singleton and contributes nothing under the 3-pairwise FE. We therefore
*       RESTRICT exit/init to firm-quarters where BOTH group rows satisfy the
*       conditioning (both-held for exit, neither-held for init), disclose the
*       four-cell counts, and report the sample cost. See extmargin_fourcell.csv.
*       ESTIMAND CAVEATS for exit/init (they condition on a LAGGED OUTCOME state):
*         - b3 is a conditional exit/initiation HAZARD among the conditioned-in
*           set (exit rate | held at t-1; init rate | not held at t-1), NOT an
*           unconditional divestment/entry probability.
*         - the both-held / neither restriction selects toward firms held by BOTH
*           US and NONUS at t-1, i.e. dual-held (larger, more-visible) firms; given
*           the ever-held coverage gap in the descriptive refresh below (full B7
*           grid: US ~56.7% vs NONUS ~92.1% of firms ever held, a ~35pp gap) the
*           both-held intersection is a NARROW, visibility-selected subset, so the
*           exit estimand does NOT generalize to "US divestment" broadly.
*
* ROBUSTNESS. Rerun ALL DVs on the riskset-conditional (engaged) subsample
* (merge on c6_panel_riskset.dta firm-quarter keys). Count/exit outcomes are MORE
* exposed to the US-13F-vs-NONUS coverage asymmetry (coverage-driven disappearance
* reads as an exit); the engaged subsample is the disclosed — but PARTIAL —
* mitigation: it restricts the ROW set only, so it mitigates false exits but does
* NOT recover unfiled counts nor adjust the n_active denominator. Note the threat
* is concentrated OFF the PRIMARY: d_breadth is a RATIO (n_holders/n_active) that
* largely cancels a uniform reporting lag because numerator and denominator shrink
* together, whereas d_nh / exit / init are unnormalized COUNT outcomes that carry
* the reporting-lag exposure directly.
*
* INFERENCE. CRVE two-way cluster(firm, month). B9 degenerate-VCE guard writes
* extmargin_vce_diag.csv. The design-based ARBITER is run_ri_extensive.py
* (US-minus-NONUS collapse, free-permutation + circular-shift RI); its b3 is
* anchored to the 3-pairwise b3 written here (extmargin_b3_anchor.csv).
*
* HONESTY. Every number verbatim from the run; failures reported as failures.

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

* SMOKE default OFF. When 1, keep <=200 firms for a fast SYNTAX-ONLY pass
* (pairing preserved because we filter by firm; inference is NOT valid — do not cite).
local SMOKE 0

use "`OUT'/extensive_margin_panel.dta", clear

if `SMOKE' == 1 {
    egen long _fid = group(firm_str)
    quietly keep if _fid <= 200
    drop _fid
    display as error "SMOKE MODE: restricted to <=200 firms — SYNTAX CHECK ONLY, results NOT valid."
}

*==============================================================
* Schema asserts against the LOCKED PANEL CONTRACT. Any silent deviation from the
* headline conventions invalidates the comparison, so we check hard.
*==============================================================
confirm string variable firm_str
confirm numeric variable us n_holders n_active breadth d_breadth d_nh held_lag exit init cn_lag shock

gen rd_day = dofc(rdate)
format rd_day %td
gen rd_m = mofd(rd_day)
format rd_m %tm
count if missing(rd_day)
assert r(N) == 0

* us in {0,1}; grid is zero-filled so every firm-quarter carries BOTH group rows
assert inlist(us, 0, 1)
bysort firm_str rd_day: assert _N == 2
by firm_str rd_day: assert us[1] != us[2]

* cn_lag is a firm-quarter share, GROUP-INVARIANT within firm-quarter (fq absorbs
* the bare cn_lag only if so); shock is one common value per quarter.
assert cn_lag >= 0 & cn_lag <= 1 if !missing(cn_lag)
bysort firm_str rd_day (cn_lag): assert cn_lag == cn_lag[1] if !missing(cn_lag)
bysort rd_m (shock): assert shock == shock[1] if !missing(shock)

* Contract domains: exit defined (non-missing) ONLY where held_lag==1;
*                   init defined (non-missing) ONLY where held_lag==0.
assert missing(exit) if held_lag == 0
assert missing(exit) if missing(held_lag)
assert !missing(exit) if held_lag == 1
assert missing(init) if held_lag == 1
assert missing(init) if missing(held_lag)
assert !missing(init) if held_lag == 0

* n_active is a GROUP x QUARTER denominator (Chen-Hong-Stein): constant within
* (us, quarter). breadth == n_holders / n_active on non-degenerate cells.
bysort us rd_day (n_active): assert n_active == n_active[1]
assert n_holders >= 0 & n_active >= 0

*==============================================================
* DESCRIPTIVE REFRESH on the B7 grid (doubles as the live motivating fact).
* "cell" = firm x group x quarter row. Zero-fill share = share of cells with
* n_holders==0, by group; the US-minus-NONUS gap is the headline descriptive.
* Reported at BOTH firm-quarter and firm level (claims-match-data convention).
* NB: the pre-B7 +11.26pp figure is SUPERSEDED — this recompute replaces it.
*==============================================================
tempname dh
file open `dh' using "`OUT'/extmargin_descriptive.csv", write replace
file write `dh' "group,level,n_zero,n_total,share_zero" _n
gen byte zero_nh = (n_holders == 0)
display _newline "=== DESCRIPTIVE: extensive-margin zero-fill share (B7 grid) ==="
forval u = 0/1 {
    local grp = cond(`u' == 1, "US", "NONUS")
    quietly count if us == `u'
    local tot = r(N)
    quietly count if us == `u' & zero_nh == 1
    local z = r(N)
    local share`u' = `z' / `tot'
    file write `dh' "`grp',firm_quarter_cell,`z',`tot',`=`z'/`tot''" _n
    display "  [`grp'] firm-quarter cells with n_holders==0: `z' / `tot'  (share=" %6.4f `z'/`tot' ")"
}
display "  US-minus-NONUS zero-fill gap (firm-quarter): " %7.4f (`share1' - `share0') ///
        "   (pre-B7 +11.26pp SUPERSEDED — do not cite)"

* firm level: share of firms with >=1 zero-holder quarter, by group
bysort firm_str us: egen byte _anyzero = max(zero_nh)
egen byte _fu_tag = tag(firm_str us)
forval u = 0/1 {
    local grp = cond(`u' == 1, "US", "NONUS")
    quietly count if _fu_tag == 1 & us == `u'
    local ftot = r(N)
    quietly count if _fu_tag == 1 & us == `u' & _anyzero == 1
    local fz = r(N)
    file write `dh' "`grp',firm,`fz',`ftot',`=`fz'/`ftot''" _n
    display "  [`grp'] firms with >=1 zero-holder quarter: `fz' / `ftot'  (share=" %6.4f `fz'/`ftot' ")"
}
file close `dh'
drop _anyzero _fu_tag
di "Wrote `OUT'/extmargin_descriptive.csv"

display _newline "=== breadth distribution by group (us=0 NONUS, us=1 US) ==="
tabstat breadth, by(us) stat(mean p50 sd min max n)

*==============================================================
* Both-conditioning helpers for exit/init. heldlag_{us,nonus} broadcast each
* group's held_lag to both rows of the firm-quarter (egen max ignores the
* other-group missing; if a group's own held_lag is missing it stays missing).
*==============================================================
bysort firm_str rd_day: egen byte heldlag_us    = max(cond(us == 1, held_lag, .))
bysort firm_str rd_day: egen byte heldlag_nonus = max(cond(us == 0, held_lag, .))
gen byte samp_exit = (heldlag_us == 1 & heldlag_nonus == 1)   /* both-held */
gen byte samp_init = (heldlag_us == 0 & heldlag_nonus == 0)   /* neither-held */
label var samp_exit "both group rows held_lag==1 (exit estimation cell)"
label var samp_init "both group rows held_lag==0 (init estimation cell)"

*==============================================================
* FOUR-CELL disclosure (firm-quarter level): both-held / US-only / NONUS-only /
* neither, over the domain where BOTH group held_lag are non-missing. Report how
* much sample each conditioning restriction costs.
*==============================================================
egen byte _fqtag = tag(firm_str rd_day)
gen byte _dom = (!missing(heldlag_us) & !missing(heldlag_nonus))
quietly count if _fqtag == 1
local n_fq = r(N)
quietly count if _fqtag == 1 & _dom == 1
local n_dom = r(N)
quietly count if _fqtag == 1 & heldlag_us == 1 & heldlag_nonus == 1
local c_both = r(N)
quietly count if _fqtag == 1 & heldlag_us == 1 & heldlag_nonus == 0
local c_uso = r(N)
quietly count if _fqtag == 1 & heldlag_us == 0 & heldlag_nonus == 1
local c_nuso = r(N)
quietly count if _fqtag == 1 & heldlag_us == 0 & heldlag_nonus == 0
local c_nei = r(N)

tempname fc
file open `fc' using "`OUT'/extmargin_fourcell.csv", write replace
file write `fc' "cell,n_firmquarters,pct_of_domain" _n
file write `fc' "both_held,`c_both',`=`c_both'/`n_dom''" _n
file write `fc' "US_only,`c_uso',`=`c_uso'/`n_dom''" _n
file write `fc' "NONUS_only,`c_nuso',`=`c_nuso'/`n_dom''" _n
file write `fc' "neither,`c_nei',`=`c_nei'/`n_dom''" _n
file write `fc' "domain_total,`n_dom'," _n
file write `fc' "all_firmquarters,`n_fq'," _n
file close `fc'

display _newline "=== FOUR-CELL held_lag cross-tab (firm-quarter; domain = both non-missing) ==="
display "    both-held=`c_both'  US-only=`c_uso'  NONUS-only=`c_nuso'  neither=`c_nei'"
display "    domain total (both held_lag non-missing)=`n_dom'  of `n_fq' firm-quarters"
display "    exit restriction (both-held) keeps `c_both'/`n_dom' domain firm-quarters (costs `=`n_dom'-`c_both'')"
display "    init restriction (neither)   keeps `c_nei'/`n_dom' domain firm-quarters (costs `=`n_dom'-`c_nei'')"
di "Wrote `OUT'/extmargin_fourcell.csv"
drop _fqtag _dom

*==============================================================
* Riskset-conditional (engaged) subsample flag. Merge on firm-quarter keys of
* c6_panel_riskset.dta (fully paired risk set: held at t OR t-1 OR t+1). If the
* file is absent, the riskset robustness is SKIPPED (flagged, not silent).
*==============================================================
local has_rs 0
capture confirm file "`OUT'/c6_panel_riskset.dta"
if _rc == 0 {
    preserve
    use firm_str rdate using "`OUT'/c6_panel_riskset.dta", clear
    duplicates drop firm_str rdate, force
    tempfile rs
    save "`rs'"
    restore
    merge m:1 firm_str rdate using "`rs'", keep(master match) gen(_mrs)
    gen byte risk1 = (_mrs == 3)
    drop _mrs
    local has_rs 1
    quietly count if risk1 == 1
    display _newline "riskset merge: `r(N)' rows flagged risk1==1 (engaged subsample)"
}
else {
    gen byte risk1 = 0
    display as error "NOTE: c6_panel_riskset.dta not found — riskset robustness SKIPPED."
}

*==============================================================
* Integer FE codes (built globally; reghdfe drops unused levels per subsample).
* us IS the group here (binary), so gq = group x quarter, ig = firm x group.
*==============================================================
egen firm_n = group(firm_str)
egen fq = group(firm_str rd_day)
egen gq = group(us rd_day)
egen ig = group(firm_str us)
gen double us_cn       = us * cn_lag
gen double us_cn_shock = us * cn_lag * shock
label var us_cn       "us x CN(t-1)"
label var us_cn_shock "us x CN(t-1) x S_t  (DDD triple = estimand b3)"

*==============================================================
* Spec runner. Same regressors/FE/VCE as the headline; only DV, FE set and the
* (optional) sample selector change. Reports N / #firms / cluster counts / b3.
*==============================================================
capture program drop run_ext
program define run_ext
    * ename : estimates name
    * dv    : dependent variable
    * FE    : absorb() set, quoted ("fq gq ig" or "fq gq")
    * ifc   : if-condition, quoted ("" or "if samp_exit==1 & risk1==1")
    * slab  : sample/spec label
    args ename dv FE ifc slab
    capture qui reghdfe `dv' us_cn us_cn_shock `ifc', absorb(`FE') vce(cluster firm_n rd_m)
    if _rc {
        display as error _newline "=== `ename' [`dv' | `FE' | `slab'] : reghdfe FAILED rc=`=_rc' (insufficient obs / collinear / degenerate subsample) — SKIPPED ==="
        global EXT_FAILED "$EXT_FAILED `ename'"
        exit
    }
    estimates store `ename'
    quietly count if e(sample)
    local nobs = r(N)
    quietly levelsof firm_n if e(sample), local(_f)
    local nfirm : word count `_f'
    quietly levelsof rd_m if e(sample), local(_q)
    local nq : word count `_q'
    display _newline "=== `ename' [`dv' | `FE' | `slab'] ==="
    display "    N=" %9.0gc `nobs' "  firms=" %6.0f `nfirm' ///
            "  firm-clusters=" %6.0f e(N_clust1) "  month-clusters=" %5.0f e(N_clust2) ///
            "  quarters=" %4.0f `nq'
    local bb = _b[us_cn_shock]
    local ss = _se[us_cn_shock]
    if (`ss' > 0 & !missing(`ss')) {
        display "    b3(us_cn_shock)=" %10.3e `bb' "  se=" %10.3e `ss' ///
                "  p=" %6.4f 2*ttail(e(df_r), abs(`bb'/`ss'))
    }
    else {
        display as error "    b3(us_cn_shock)=" %10.3e `bb' ///
            "  se=DEGENERATE (missing/zero) — CRVE p INVALID; RI arbiter in run_ri_extensive.py"
    }
end

global EXT_FAILED ""

*==============================================================
* (a) PRIMARY d_breadth + companion d_nh — paired, all firm-quarters.
*==============================================================
run_ext eb_db_3pw   d_breadth "fq gq ig" ""              "full 3pw PRIMARY"
run_ext eb_db_itgt  d_breadth "fq gq"    ""              "full it+gt"
run_ext eb_dnh_3pw  d_nh      "fq gq ig" ""              "full 3pw"
run_ext eb_dnh_itgt d_nh      "fq gq"    ""              "full it+gt"

*==============================================================
* (b) exit LPM on the both-held paired cell; init LPM on the neither paired cell.
*==============================================================
run_ext eb_exit_3pw  exit "fq gq ig" "if samp_exit==1" "both-held 3pw"
run_ext eb_exit_itgt exit "fq gq"    "if samp_exit==1" "both-held it+gt"
run_ext eb_init_3pw  init "fq gq ig" "if samp_init==1" "neither 3pw"
run_ext eb_init_itgt init "fq gq"    "if samp_init==1" "neither it+gt"

*==============================================================
* ROBUSTNESS — riskset-conditional (engaged) subsample, all DVs.
*==============================================================
if `has_rs' == 1 {
    run_ext eb_db_3pw_rs   d_breadth "fq gq ig" "if risk1==1" "riskset 3pw"
    run_ext eb_db_itgt_rs  d_breadth "fq gq"    "if risk1==1" "riskset it+gt"
    run_ext eb_dnh_3pw_rs  d_nh      "fq gq ig" "if risk1==1" "riskset 3pw"
    run_ext eb_dnh_itgt_rs d_nh      "fq gq"    "if risk1==1" "riskset it+gt"
    run_ext eb_exit_3pw_rs  exit "fq gq ig" "if samp_exit==1 & risk1==1" "both-held riskset 3pw"
    run_ext eb_exit_itgt_rs exit "fq gq"    "if samp_exit==1 & risk1==1" "both-held riskset it+gt"
    run_ext eb_init_3pw_rs  init "fq gq ig" "if samp_init==1 & risk1==1" "neither riskset 3pw"
    run_ext eb_init_itgt_rs init "fq gq"    "if samp_init==1 & risk1==1" "neither riskset it+gt"
}

*==============================================================
* Build the estimate list (riskset columns appended only if the merge ran).
*==============================================================
local all_ests "eb_db_3pw eb_db_itgt eb_dnh_3pw eb_dnh_itgt eb_exit_3pw eb_exit_itgt eb_init_3pw eb_init_itgt"
if `has_rs' == 1 {
    local all_ests "`all_ests' eb_db_3pw_rs eb_db_itgt_rs eb_dnh_3pw_rs eb_dnh_itgt_rs eb_exit_3pw_rs eb_exit_itgt_rs eb_init_3pw_rs eb_init_itgt_rs"
}

* Keep only specs whose estimates actually stored (a failed reghdfe stores none).
local ok_ests ""
foreach m of local all_ests {
    capture estimates restore `m'
    if _rc == 0 {
        local ok_ests "`ok_ests' `m'"
    }
}
if "$EXT_FAILED" != "" {
    display as error "SPECS THAT FAILED reghdfe (excluded from tables; report as failures): $EXT_FAILED"
}

*==============================================================
* B9: degenerate two-way-cluster VCE detection on us_cn and the DDD triple.
*==============================================================
tempname vh
file open `vh' using "`OUT'/extmargin_vce_diag.csv", write replace
file write `vh' "spec,coef,b,se,se_valid" _n
local any_degen 0
foreach m of local ok_ests {
    estimates restore `m'
    foreach cf in us_cn us_cn_shock {
        local bb = _b[`cf']
        local ss = _se[`cf']
        local ok = (!missing(`ss') & `ss' > 0)
        file write `vh' "`m',`cf',`bb',`ss',`ok'" _n
        if `ok' == 0 {
            local any_degen 1
            display as error ">>> `m'/`cf': DEGENERATE VCE (SE missing/zero) — CRVE p INVALID. Use the design-based RI (run_ri_extensive.py) for this DV. <<<"
        }
    }
}
* Record failed specs too (no valid CRVE inference exists for them).
foreach m in $EXT_FAILED {
    file write `vh' "`m',reghdfe_failed,.,.,0" _n
}
file close `vh'
di "Wrote `OUT'/extmargin_vce_diag.csv"
if `any_degen' == 1 {
    display as error "One or more extensive-margin specs had a degenerate two-way-cluster VCE; their CRVE SE/p in extmargin_results.csv are NOT valid inference — read the RI p-values instead."
}

*==============================================================
* DRIFT-ANCHOR CSV: the 3-pairwise b3 per DV (full + riskset), read FROM the
* stored estimates (never hardcoded). run_ri_extensive.py reads this and prints
* its collapsed b3 next to the Stata b3 (values are NEW, so no hard gate).
*==============================================================
tempname ah
file open `ah' using "`OUT'/extmargin_b3_anchor.csv", write replace
file write `ah' "dv,sample,fe,b3,se,p,N" _n
local anch_ests "eb_db_3pw eb_dnh_3pw eb_exit_3pw eb_init_3pw"
local anch_dv   "d_breadth d_nh exit init"
local anch_samp "full full full full"
if `has_rs' == 1 {
    local anch_ests "`anch_ests' eb_db_3pw_rs eb_dnh_3pw_rs eb_exit_3pw_rs eb_init_3pw_rs"
    local anch_dv   "`anch_dv' d_breadth d_nh exit init"
    local anch_samp "`anch_samp' riskset riskset riskset riskset"
}
local nanch : word count `anch_ests'
forval i = 1/`nanch' {
    local est  : word `i' of `anch_ests'
    local dv   : word `i' of `anch_dv'
    local samp : word `i' of `anch_samp'
    capture estimates restore `est'
    if _rc == 0 {
        local pp = .
        if (_se[us_cn_shock] > 0 & !missing(_se[us_cn_shock])) {
            local pp = 2*ttail(e(df_r), abs(_b[us_cn_shock]/_se[us_cn_shock]))
        }
        file write `ah' "`dv',`samp',fq_gq_ig," ///
            (strtrim(strofreal(_b[us_cn_shock], "%14.6e"))) "," ///
            (strtrim(strofreal(_se[us_cn_shock], "%14.6e"))) "," ///
            (strtrim(strofreal(`pp', "%9.6f"))) "," ///
            (strtrim(strofreal(e(N), "%15.0f"))) _n
    }
}
file close `ah'
di "Wrote `OUT'/extmargin_b3_anchor.csv"

*==============================================================
* Results table.
*==============================================================
capture which esttab
if _rc == 0 & "`ok_ests'" != "" {
    esttab `ok_ests' using "`OUT'/extmargin_results.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) ///
        keep(us_cn us_cn_shock) ///
        nonumbers plain ///
        addnote("us_cn_shock is the DDD triple (us x CN(t-1) x S_t). PRE-REGISTERED sign is DV-SPECIFIC because exit is a held->0 (full-drop) indicator, opposite the count/breadth DVs: partial divestment for US = us_cn_shock<0 for d_breadth/d_nh/init AND us_cn_shock>0 for exit; a NEGATIVE exit coefficient means FEWER US exits = anti-divestment, NOT divestment. DVs: d_breadth = Delta(n_holders/n_active) PRIMARY, d_nh = Delta n_holders companion (both paired, all firm-quarters); exit/init are LPMs conditioned on held_lag (a LAGGED OUTCOME state), so their coefficient is a conditional exit/initiation HAZARD among the conditioned-in set (not an unconditional divestment/entry probability), and US-NONUS pairing breaks so they are RESTRICTED to firm-quarters where BOTH group rows satisfy the conditioning (exit: both-held; init: neither-held) to avoid fq-FE singletons — this both-held/neither restriction selects toward dual-held (larger, more-visible) firms and NARROWS the estimand away from US divestment broadly; four-cell counts and sample cost in extmargin_fourcell.csv. 3pw = firm x quarter + group x quarter + firm x group (headline); itgt = firm x quarter + group x quarter (companion). Two-way cluster(firm, month). VCE validity per spec in extmargin_vce_diag.csv; degenerate columns -> design-based RI in run_ri_extensive.py (free-permutation + circular-shift), which is the ARBITER; b3 anchored in extmargin_b3_anchor.csv. Riskset (_rs) columns rerun on the engaged subsample (held at t OR t-1 OR t+1) — a PARTIAL mitigation for the US-13F-vs-NONUS coverage asymmetry: it restricts ROWS only (coverage-driven disappearance can read as an exit) and does NOT recover unfiled counts nor adjust the n_active denominator. The coverage threat is concentrated OFF the PRIMARY: d_breadth is a ratio that largely cancels a uniform reporting lag (numerator and denominator shrink together), whereas d_nh/exit/init are unnormalized counts that carry it. Pre-B7 +11.26pp descriptive gap is SUPERSEDED — do not cite.")
    di "Wrote `OUT'/extmargin_results.csv"
}

display _newline "=== b / se / p per spec (triple = us_cn_shock) ==="
foreach m of local ok_ests {
    estimates restore `m'
    display _newline "--- `m' ---"
    estimates table, b(%12.4e) se(%12.4e) p(%6.4f) keep(us_cn us_cn_shock)
}
display _newline "Done."
