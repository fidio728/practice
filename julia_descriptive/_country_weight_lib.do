* ============================================================================
* _country_weight_lib.do — SHARED library for the COUNTRY-LEVEL PORTFOLIO-
* WEIGHT DDD (task e2, 2026-08-10).  DEFINES PROGRAMS ONLY.  It loads no data,
* estimates nothing and writes nothing; `include' it from a runner.
*
* WHY A LIBRARY.  Three runners (run_country_weight_ddd.do,
* run_country_sensitivities.do and — through the same variable contract —
* the Python legs) must build the SAME FE ids, the SAME interaction terms and
* the SAME estimator.  Duplicating that construction across .do files is the
* documented drift failure mode of this project, so there is exactly ONE
* definition here.
*
* PROGRAMS
*   cw_prep               validate output/country_weight_panel.dta in memory and
*                         build ctry / grp / qtr / cg / ct / gt / us /
*                         usm{1,2,3} / usm{1,2,3}s (own-sample measures) and
*                         usm{1,2,3}c / usm{1,2,3}cs (the COMMON-FIRM-SET
*                         measures), plus the weak cell-set flag m2cov_cell.
*   cw_run                one reghdfe + returned scalars (b, se, t, p, df, N,
*                         n_countries, n_quarters).  The estimator of record.
*   cw_sha256             SHA256 of one file (provenance; see PROVENANCE below).
*   scoreboot_webb        Webb-weight score bootstrap — the SAME Kline-Santos
*                         machinery as run_country_panel.do, generalised over
*                         absorb() / cluster() / a control VARLIST.
*   scoreboot_selfcheck   proves the generalised copy reproduces
*                         run_country_panel.do's engine bit-for-bit.
*
* ---------------------------------------------------------------------------
* PROVENANCE (external-review item, 2026-08-10)
* ---------------------------------------------------------------------------
* Every runner must print the SHA256 of ITSELF and of each input artifact, so a
* set of numbers can be tied to the exact bytes that produced it.  mtimes cannot
* do that here: OneDrive rewrites them on sync and output/ is a junction to E:.
* Stata has no SHA256 function (`checksum' is a CRC, not a digest), so cw_sha256
* shells out to certutil, which ships with Windows.
*
* WHY cw_sha256 TAKES dir() AND name() SEPARATELY, AND cd's.  The project path
* contains a non-ASCII character ("Universitat Ramón Llull").  A process
* inherits its working directory as UTF-16, but a COMMAND LINE is handed to cmd
* in the console codepage (936 on this machine), where the CP1252 byte for "ó"
* is a lead byte and the path is destroyed.  Passing only an ASCII BASENAME from
* inside the directory sidesteps the conversion entirely.  If the hash still
* cannot be obtained the program returns "UNAVAILABLE" and the caller prints a
* loud warning — provenance never aborts an estimation run, but its absence is
* never silent either.
*
* ---------------------------------------------------------------------------
* SCORE-BOOTSTRAP PROVENANCE (read this before touching the mata block).
* run_country_panel.do carries the original `_step4_scoreboot' / `step4_scorewcb'
* pair, written because boottest REFUSES to run after reghdfe with more than one
* absorbed FE group (measured 2026-08-10: returns without r(p) and without an
* error code, so p_wild came back all-missing while the log claimed the boottest
* engine).  Every design in this task absorbs at least two FE groups, so that
* incompatibility is structural, and the score bootstrap is the engine of record.
*
* The mata core below is a VERBATIM lift of `_step4_scoreboot' — same Webb
* 6-point weights, same statistic t = sum(s_c)/sqrt(sum(s_c^2)), same
* null-restricted-score construction, same rep convention.  Only the wrapper is
* generalised (absorb / cluster / multi-var control).  It is NOT a second
* implementation: `scoreboot_selfcheck' below RE-RUNS run_country_panel.do's own
* Step-4 M1/us_dlog cell through THIS code at THAT file's seed and hard-fails
* unless the p matches the value already stored in country_panel_step4.csv.  So
* "reuse" is verified numerically at run time, not asserted in a comment.
* FOLLOW-UP (stated, not silent): at run_country_panel.do's next rotation it
* should be switched to `include "_country_weight_lib.do"' + `scoreboot_webb'
* so that even the source copy disappears and one definition remains.
* ============================================================================

* ---------------------------------------------------------------------------
* cw_sha256 — SHA256 of ONE file, via certutil.  See the PROVENANCE block above
* for why dir() and name() are separate and why the program cd's.
*   returns  r(sha256)  64 lower-case hex chars, or "MISSING", or "UNAVAILABLE"
* ---------------------------------------------------------------------------
capture program drop cw_sha256
program define cw_sha256, rclass
    syntax , dir(string) name(string)
    return local sha256 "UNAVAILABLE"
    capture confirm file "`dir'/`name'"
    if _rc != 0 {
        return local sha256 "MISSING"
        exit
    }
    local pwd0 : pwd
    tempfile hout
    capture cd "`dir'"
    if _rc != 0 {
        exit
    }
    * certutil writes:  "SHA256 hash of <name>:" / <64 hex> / "CertUtil: ..."
    capture shell certutil -hashfile "`name'" SHA256 > "`hout'" 2>&1
    quietly cd "`pwd0'"
    local hex ""
    tempname fh
    capture file open `fh' using "`hout'", read text
    if _rc != 0 {
        exit
    }
    file read `fh' line
    local eof = r(eof)
    while `eof' == 0 {
        * compound quotes throughout: a certutil line can legitimately contain a
        * double quote, and a bare "`s'" would then terminate the string early
        local s = subinstr(trim(`"`macval(line)'"'), " ", "", .)
        if strlen(`"`macval(s)'"') == 64 & "`hex'" == "" {
            if regexm(lower(`"`macval(s)'"'), "^[0-9a-f]+$") {
                local hex = lower(`"`macval(s)'"')
            }
        }
        file read `fh' line
        local eof = r(eof)
    }
    file close `fh'
    if "`hex'" != "" {
        return local sha256 "`hex'"
    }
end

* ---------------------------------------------------------------------------
* cw_prov_print — hash one file, print the line, return r(sha256).  Keeps the
* three-line ritual out of every runner.
* ---------------------------------------------------------------------------
capture program drop cw_prov_print
program define cw_prov_print, rclass
    syntax , dir(string) name(string) [tag(string)]
    if "`tag'" == "" {
        local tag "INPUT "
    }
    cw_sha256, dir("`dir'") name("`name'")
    local h "`r(sha256)'"
    display as text "  `tag'  " %-34s "`name'" "  `h'"
    if "`h'" == "UNAVAILABLE" {
        display as error "  [provenance] SHA256 UNAVAILABLE for `name' — certutil did not"
        display as error "    return a digest.  The numbers below are NOT tied to file bytes;"
        display as error "    record the run some other way before citing them."
    }
    return local sha256 "`h'"
end

* ---------------------------------------------------------------------------
* mata core — VERBATIM from run_country_panel.do::_step4_scoreboot
* ---------------------------------------------------------------------------
capture mata: mata drop _scoreboot_webb_core()
mata:
void _scoreboot_webb_core(string scalar svar, real scalar reps)
{
    real colvector S, u, w, ws
    real scalar G, tobs, tr, cnt, r
    S = st_data(., svar)
    G = rows(S)
    tobs = sum(S) / sqrt(sum(S:^2))
    cnt = 0
    for (r=1; r<=reps; r++) {
        u = runiform(G, 1)
        w = J(G, 1, 0)
        w = w - sqrt(1.5) * (u:<(1/6))
        w = w - (u:>=(1/6) :& u:<(2/6))
        w = w - sqrt(0.5) * (u:>=(2/6) :& u:<(3/6))
        w = w + sqrt(0.5) * (u:>=(3/6) :& u:<(4/6))
        w = w + (u:>=(4/6) :& u:<(5/6))
        w = w + sqrt(1.5) * (u:>=(5/6))
        ws = w :* S
        tr = sum(ws) / sqrt(sum(ws:^2))
        if (abs(tr) >= abs(tobs)) {
            cnt = cnt + 1
        }
    }
    st_numscalar("__sbw_p", cnt/reps)
    st_numscalar("__sbw_t", tobs)
    st_numscalar("__sbw_G", G)
}
end

* ---------------------------------------------------------------------------
* scoreboot_webb — generalised wrapper.
*   y        outcome
*   xint     the regressor being tested (H0: its coefficient = 0)
*   ctrl     control VARLIST kept in both the unrestricted and the restricted
*            fit (may be empty)
*   absorb   reghdfe absorb() spec (any number of FE groups)
*   clustvar SINGLE clustering variable for the bootstrap (see CLUSTER NOTE)
*
* CLUSTER NOTE.  The CRVE of record for the country DDD is TWO-WAY (country,
* quarter).  A score bootstrap over two non-nested cluster dimensions has no
* standard construction, so p_wild is bootstrapped over ONE dimension — the
* small one, country (28 clusters), which is where the few-cluster problem
* actually lives.  Every consumer must stamp wild_cluster="country" in its
* output; p_wild is a supplement to the CRVE column, and the RANDOMIZATION
* ARBITER (run_ri_country_ddd.py) is what governs the reported verdict.
* ---------------------------------------------------------------------------
capture program drop scoreboot_webb
program define scoreboot_webb, rclass
    syntax , y(varname) xint(varname) absorb(string) clustvar(varname) ///
             [ctrl(varlist) ifcond(string) reps(integer 9999)]
    tempvar insmp er xt s
    quietly {
        reghdfe `y' `xint' `ctrl' `ifcond', absorb(`absorb') vce(cluster `clustvar')
        gen byte `insmp' = e(sample)
        * null-restricted residuals (H0: coefficient on `xint' = 0)
        reghdfe `y' `ctrl' if `insmp', absorb(`absorb') residuals(`er')
        * FWL: partial the tested regressor on the FE + controls
        reghdfe `xint' `ctrl' if `insmp', absorb(`absorb') residuals(`xt')
        gen double `s' = `er' * `xt'
        preserve
        keep if `insmp'
        collapse (sum) __sbw_sc = `s', by(`clustvar')
        mata: _scoreboot_webb_core("__sbw_sc", `reps')
        restore
    }
    return scalar p    = scalar(__sbw_p)
    return scalar tobs = scalar(__sbw_t)
    return scalar G    = scalar(__sbw_G)
    return local engine "scoreboot_webb (Kline-Santos null-restricted cluster scores, Webb 6-pt weights)"
    capture scalar drop __sbw_p
    capture scalar drop __sbw_t
    capture scalar drop __sbw_G
end

* ---------------------------------------------------------------------------
* scoreboot_selfcheck — REUSE PROOF.
* Re-runs run_country_panel.do's FIRST estimation cell (M1, spec us_dlog) with
* the generalised program above, at that file's own seed (20260810), and
* compares the p to the value already written in country_panel_step4.csv.
*
* Why the seed line is exact: run_country_panel.do sets the seed at the top,
* does only deterministic data prep, and its estimation loop's FIRST iteration
* is M1/us_dlog — so the first mata runiform() draws in that file come off a
* freshly seeded stream, exactly like the first draws here.
*
* This program CLEARS the dataset in memory.  Call it BEFORE loading the
* country weight panel.  It hard-fails on mismatch; if the Step-4 artifacts are
* absent it prints SKIPPED and returns r(status)="skipped" (a missing anchor is
* not a licence to run silently — the caller must print the status).
* ---------------------------------------------------------------------------
capture program drop scoreboot_selfcheck
program define scoreboot_selfcheck, rclass
    syntax , out(string) [reltol(real 1e-12)]
    return local status "skipped"
    capture confirm file "`out'/country_panel_step4.dta"
    local rc1 = _rc
    capture confirm file "`out'/country_panel_step4.csv"
    local rc2 = _rc
    if `rc1' != 0 | `rc2' != 0 {
        display as text "[scoreboot self-check] SKIPPED — country_panel_step4.{dta,csv} " ///
                        "not both present, so the generalised score bootstrap cannot be " ///
                        "proven equal to run_country_panel.do's engine on this machine."
        exit
    }
    * ---- the anchor p_wild from the canonical Step-4 CSV --------------------
    * (no preserve/restore: this program is documented as CLEARING memory, and
    *  it is called before the analysis panel is loaded)
    quietly import delimited using "`out'/country_panel_step4.csv", clear varnames(1) case(preserve)
    capture confirm variable p_wild
    if _rc != 0 {
        display as error "[scoreboot self-check] country_panel_step4.csv has no p_wild column"
        error 459
    }
    quietly keep if measure == "M1" & spec == "us_dlog"
    if _N != 1 {
        display as error "[scoreboot self-check] no unique M1/us_dlog row in country_panel_step4.csv"
        error 459
    }
    local p_anchor = p_wild[1]
    * ---- re-derive it through THIS library ---------------------------------
    quietly use "`out'/country_panel_step4.dta", clear
    quietly gen rd_day = dofc(rdate)
    quietly gen rd_q   = qofd(rd_day)
    quietly egen ctry  = group(sec_country)
    quietly gen double sxm1 = s_lag * m1_lag
    set seed 20260810
    scoreboot_webb, y(dlog_us) xint(sxm1) ctrl(m1_lag) absorb(ctry rd_q) ///
                    clustvar(ctry) reps(9999)
    local p_here = r(p)
    local d = abs(`p_here' - `p_anchor')
    display as text "[scoreboot self-check] run_country_panel.do M1/us_dlog p_wild: " ///
                    "anchor=`p_anchor'  re-derived=`p_here'  |diff|=`d'"
    if `d' > `reltol' {
        display as error "[scoreboot self-check] FAILED: the generalised score bootstrap does"
        display as error "  NOT reproduce run_country_panel.do's engine on its own Step-4 cell."
        display as error "  Two possible causes, both of which must be investigated by hand:"
        display as error "   (1) the algorithm diverged (a real defect — fix it, do not widen tol);"
        display as error "   (2) the RNG STREAM diverged (a Stata-version change, or an"
        display as error "       RNG-consuming command inserted before the first draw in either"
        display as error "       file).  Cause (2) is not a licence to proceed either: without a"
        display as error "       reproducible stream the p_wild column is not reproducible."
        error 459
    }
    display as text "[scoreboot self-check] PASS — p_wild engine verified against " ///
                    "country_panel_step4.csv (this is the REUSE proof)."
    return local status "pass"
    return scalar p_anchor = `p_anchor'
    return scalar p_here   = `p_here'
end

* ---------------------------------------------------------------------------
* cw_prep — validate the country weight panel IN MEMORY and build every id and
* interaction the DDD needs.  Fails loudly and by NAME on a missing column, so
* an e1 contract gap is reported as a dependency, never worked around.
*
* DEPENDENCY CONTRACT (output/country_weight_panel.dta, from e1):
*   sec_country   str   ISO-2 security country c
*   holder_group  str   'US' / 'NONUS'
*   rdate         num   quarter-end, %tc (or %td) formatted
*   dw_global     num   dW_{c,g,t} on the GLOBAL denominator          MAIN
*   dw_eu         num   dW on the EU denominator                      diagnostic
*   dlog_usd      num   d ln(USD holdings)                            supplementary
*   m1_lag m2_lag m3_lag        num  M_{k,c,t-1}, each on its OWN firm set
*   m1cov_lag m2cov_lag m3cov_lag  num  the SAME three measures RECOMPUTED on the
*                                    COMMON FIRM SET (the market-cap-covered
*                                    firms inside each country-quarter, i.e.
*                                    exactly the firms M2 is built on) — e1 [5b]
*   nfirm_cov_lag nfirm_all_lag num  covered / all firm counts behind those
*   s_lag         num   S_{t-1}   (quarter-constant)
*   shock         num   S_t       (optional; timing diagnostic)
*   null_class    num   0 fully-non-null / 1 all-null / 2 partial-null (optional)
*
* ---------------------------------------------------------------------------
* SENSITIVITY (c): TWO OBJECTS AT TWO GRAINS.  DO NOT CONFUSE THEM.
* ---------------------------------------------------------------------------
* m2cov_cell (built here) = !missing(m2_lag).  This is the OLD, WEAK restriction
*   the external review rejected on 2026-08-10.  It aligns the country-quarter
*   CELLS and NOTHING ELSE: inside a kept cell, M1 and M3 are still averaged over
*   ALL firms while M2 is averaged over the market-cap-covered ones, so a country
*   in which 10% of firms carry a market cap still passes.  It is kept ONLY so
*   the weak and the real restriction can be reported side by side.
* usm{k}c / usm{k}cs (built here from m{k}cov_lag) = the REAL restriction: all
*   three measures recomputed on the SAME FIRM SET.  This is what sensitivity (c)
*   must use.
*
* TWO COVERAGE FACTS, BOTH TRUE, AT DIFFERENT GRAINS — both are printed below,
* and both must be stated whenever coverage is discussed:
*   COUNTRY-QUARTER GRAIN: at v3.1 M2 has FEWER country-quarter NAs (27) than M1
*     (106).  So "M2 is the sparse one" is FLATLY WRONG at country grain — M1's
*     denominator SUM(sc_links) is what goes to zero.
*   FIRM GRAIN: M2's market-cap coverage is only 42.9% of firm-quarters in 2023.
*     So at firm grain M2 really is built on a minority of the firms M1 and M3
*     see — which is the comparability problem sensitivity (c) exists to measure,
*     and which the cell-set restriction did nothing about.
* Neither number is quoted from this comment at run time: both are RE-DERIVED
* from the panel in memory, printed, and returned.
* ---------------------------------------------------------------------------
capture program drop cw_prep
program define cw_prep, rclass
    syntax [, quiet]
    foreach v in sec_country holder_group rdate dw_global m1_lag m2_lag m3_lag s_lag {
        capture confirm variable `v'
        if _rc != 0 {
            display as error "DEPENDENCY: country_weight_panel.dta has no column `v'."
            display as error "  e1 must emit it (see the contract block in _country_weight_lib.do)."
            error 111
        }
    }
    * -- the COMMON-FIRM-SET measures are a HARD dependency (fail-closed) -----
    * Without them the (c) sensitivity can only fall back to the cell-set
    * restriction the review rejected, and it would do so SILENTLY.  A panel
    * lacking these columns predates e1 section [5b] and must be rebuilt.
    foreach v in m1cov_lag m2cov_lag m3cov_lag {
        capture confirm variable `v'
        if _rc != 0 {
            display as error "DEPENDENCY: country_weight_panel.dta has no column `v'."
            display as error "  This is the COMMON-FIRM-SET measure built by"
            display as error "  build_country_weight_panel.py section [5b]: M1/M2/M3 recomputed on"
            display as error "  the market-cap-covered firms inside each country-quarter."
            display as error "  A panel without it can only support the CELL-set restriction that"
            display as error "  the 2026-08-10 review rejected, and this library refuses to fall"
            display as error "  back to it silently.  Rebuild the panel with the current e1."
            error 111
        }
    }
    * -- optional columns: presence is reported, absence is not fatal --------
    foreach v in dw_eu dlog_usd shock {
        capture confirm variable `v'
        if _rc != 0 {
            display as text "  NOTE: optional column `v' absent — specs using it will be skipped."
        }
    }

    * -- derived names must not silently collide with an e1 column.  They are
    *    deterministic functions of the key columns, so rebuilding is safe, but
    *    the drop is PRINTED rather than done behind the reader's back.
    foreach v in rd_day qtr us ctry grp cg ct gt m2cov_cell usm1 usm2 usm3 usm1s usm2s usm3s usm1c usm2c usm3c usm1cs usm2cs usm3cs {
        capture confirm variable `v'
        if _rc == 0 {
            display as text "  NOTE: rebuilding derived column `v' (it already existed in the panel)."
            quietly drop `v'
        }
    }

    * -- date handling: accept %tc (project convention) or %td --------------
    local rfmt : format rdate
    quietly gen long rd_day = .
    if strpos("`rfmt'", "%tc") > 0 | strpos("`rfmt'", "%tC") > 0 {
        quietly replace rd_day = dofc(rdate)
    }
    if strpos("`rfmt'", "%td") > 0 {
        quietly replace rd_day = rdate
    }
    quietly count if missing(rd_day)
    if r(N) > 0 {
        display as error "rdate carries format `rfmt' — expected %tc (project convention) or %td."
        error 459
    }
    quietly gen int qtr = qofd(rd_day)
    format qtr %tq

    * -- ids -----------------------------------------------------------------
    quietly gen byte us = (holder_group == "US")
    quietly count if !inlist(holder_group, "US", "NONUS")
    if r(N) > 0 {
        display as error "holder_group has values outside {US, NONUS} — panel contract broken."
        error 459
    }
    quietly egen ctry = group(sec_country)
    quietly egen grp  = group(holder_group)
    quietly egen cg   = group(ctry grp)
    quietly egen ct   = group(ctry qtr)
    quietly egen gt   = group(grp qtr)

    * -- key uniqueness ------------------------------------------------------
    tempvar dup
    quietly bysort ctry grp qtr: gen long `dup' = _N
    quietly count if `dup' != 1
    if r(N) > 0 {
        display as error "duplicate (country, group, quarter) rows — panel key is not unique."
        error 459
    }

    * -- balance (NOT required by the FE design; required by the RI collapse
    *    cross-check in run_ri_country_ddd.py, so it is reported, not enforced)
    tempvar ng
    quietly bysort ctry qtr: gen byte `ng' = _N
    quietly count if `ng' != 2
    local n_unbal = r(N)
    local balanced = (`n_unbal' == 0)
    if !`balanced' {
        display as text "  NOTE: {res:`n_unbal'} (country, quarter) cells do not carry both groups."
        display as text "        The 3-pairwise FE still identify the DDD; the difference-collapse"
        display as text "        cross-check in the RI script will be reported as unavailable."
    }

    * -- S_{t-1} must be one common value per quarter ------------------------
    * sd() ignores missing, so a quarter that is PARTLY missing would pass the
    * dispersion test; the missing census below is therefore not optional.  A
    * missing s_lag makes reghdfe drop the row silently, and it makes the
    * circular shift in run_ri_country_ddd.py abort outright (a rotation cannot
    * be defined on a gappy series), so the count is put on the record here.
    tempvar nsq
    quietly bysort qtr: egen `nsq' = sd(s_lag)
    quietly count if `nsq' > 1e-12 & !missing(`nsq')
    if r(N) > 0 {
        display as error "s_lag varies within a quarter — expected a single common S_{t-1}."
        error 459
    }
    quietly count if missing(s_lag)
    local n_s_miss = r(N)
    if `n_s_miss' > 0 {
        display as text "  NOTE: s_lag missing on `n_s_miss' rows — reghdfe drops them; the RI"
        display as text "        script refuses to rotate a gappy shock series.  Fix in e1 if"
        display as text "        the gap is not the panel's first quarter."
    }

    * -- contiguous quarter grid (required before any lag/rotation claim) ----
    preserve
    quietly keep qtr
    quietly duplicates drop
    quietly sort qtr
    tempvar gap
    quietly gen int `gap' = qtr - qtr[_n-1] if _n > 1
    quietly count if `gap' != 1 & _n > 1
    local n_gap = r(N)
    local n_q   = _N
    * build the printable span OUTSIDE the display string: nesting double quotes
    * inside a `=string(...,"%tq")' expression inside a display string silently
    * terminates the string early.
    local q1s = string(qtr[1],  "%tq")
    local qNs = string(qtr[_N], "%tq")
    restore
    if `n_gap' > 0 {
        display as error "quarter grid is NOT contiguous (`n_gap' gaps) — a positional lag or a"
        display as error "  circular shift would not respect calendar time.  Refusing."
        error 459
    }

    * -- identity fail-closed defence (external review item 7) ---------------
    * e1 now ABORTS its build if the MAIN (global) family carries a partial-NULL
    * cell, because on such a cell dW != sum_i dw_i by an unbounded amount and
    * the main regression does not exclude it.  This is the consumer-side half
    * of that promise: a panel built by an older e1 could still contain one, and
    * it would enter the headline unnoticed.  Checked here, once, for every
    * runner that includes this library.
    capture confirm variable null_class
    if _rc == 0 {
        quietly count if null_class == 2 & !missing(dw_global)
        local n_pn = r(N)
        if `n_pn' > 0 {
            display as error "IDENTITY FAIL-CLOSED: `n_pn' estimation rows carry null_class == 2"
            display as error "  (partial-NULL cell: some firm deltas missing, so dW is NOT the sum"
            display as error "  of firm deltas on that cell).  build_country_weight_panel.py aborts"
            display as error "  on this for the GLOBAL family, so this panel predates that gate."
            display as error "  Rebuild the panel.  Excluding the rows here instead would CHANGE"
            display as error "  THE ESTIMAND to 'dW over country-quarters in which every firm delta"
            display as error "  is observed', which is a decision for the write-up, not for a"
            display as error "  library."
            error 459
        }
    }

    * -- interactions --------------------------------------------------------
    * OWN-SAMPLE legs: each measure on the firms it is naturally defined over.
    forvalues k = 1/3 {
        quietly gen double usm`k'  = us * m`k'_lag
        quietly gen double usm`k's = us * m`k'_lag * s_lag
        label var usm`k'  "US_g x M`k'_{c,t-1}"
        label var usm`k's "US_g x M`k'_{c,t-1} x S_{t-1}"
    }
    * COMMON-FIRM-SET legs (sensitivity (c) done properly): all three measures
    * recomputed by e1 on the market-cap-covered firms inside each
    * country-quarter, i.e. on ONE FIRM SET rather than one cell set.
    forvalues k = 1/3 {
        quietly gen double usm`k'c  = us * m`k'cov_lag
        quietly gen double usm`k'cs = us * m`k'cov_lag * s_lag
        label var usm`k'c  "US_g x M`k'^cov_{c,t-1} (common firm set)"
        label var usm`k'cs "US_g x M`k'^cov_{c,t-1} x S_{t-1} (common firm set)"
    }
    * WEAK cell-set flag, kept only for the side-by-side contrast row.
    quietly gen byte m2cov_cell = !missing(m2_lag)
    label var m2cov_cell "OLD cell-set restriction (aligns cells only) — NOT sensitivity (c)"

    quietly levelsof ctry, local(_lc)
    local n_c : word count `_lc'
    local n_rows = _N
    quietly count if m2cov_cell == 1
    local n_m2 = r(N)
    display as text "[cw_prep] `n_rows' rows | `n_c' countries | `n_q' quarters " ///
                    "(`q1s' .. `qNs') | balanced=`balanced'"

    * -- THE TWO COVERAGE FACTS, RE-DERIVED, AT THEIR TWO GRAINS -------------
    display as text "          COVERAGE, country-quarter grain (missing country-quarter cells):"
    preserve
    quietly keep if us == 1
    foreach k in 1 2 3 {
        quietly count if missing(m`k'_lag)
        local na`k' = r(N)
    }
    quietly count
    local n_cells = r(N)
    restore
    display as text "            of `n_cells' country-quarters: M1 missing `na1' | " ///
                    "M2 missing `na2' | M3 missing `na3'"
    if `na2' <= `na1' {
        display as text "            => at THIS grain M2 is NOT the sparse measure (M1 is): the"
        display as text "               framing 'M2 is the sparse one' is wrong at country grain."
    }
    capture confirm variable nfirm_cov_lag
    local has_nf = (_rc == 0)
    if `has_nf' {
        capture confirm variable nfirm_all_lag
        local has_nf = (_rc == 0)
    }
    if `has_nf' {
        quietly summarize nfirm_cov_lag if us == 1, meanonly
        local s_cov = r(sum)
        quietly summarize nfirm_all_lag if us == 1, meanonly
        local s_all = r(sum)
        local shr = .
        * missing > 0 is TRUE in Stata, so the missing test is not optional here
        if `s_all' > 0 & !missing(`s_all') & !missing(`s_cov') {
            local shr = `s_cov'/`s_all'
        }
        display as text "          COVERAGE, FIRM grain (the set M2 is actually built on):"
        display as text "            `s_cov' of `s_all' firm-quarters carry a market cap " ///
                        "(share=" %6.4f `shr' ")"
        display as text "            => at THIS grain M2 IS built on a minority of the firms M1/M3"
        display as text "               see.  BOTH facts must be stated; they are different grains."
    }
    display as text "          sensitivity (c) uses usm{k}c / usm{k}cs (COMMON FIRM SET)."
    display as text "          m2cov_cell (`n_m2' rows) is the OLD cell-set flag, kept only as a"
    display as text "          reported contrast — it is NOT the (c) restriction."

    return scalar n_countries   = `n_c'
    return scalar n_quarters    = `n_q'
    return scalar balanced      = `balanced'
    return scalar n_s_missing   = `n_s_miss'
    return scalar n_m2cov_cell  = `n_m2'
    return scalar na_cells_m1   = `na1'
    return scalar na_cells_m2   = `na2'
    return scalar na_cells_m3   = `na3'
    if `has_nf' {
        return scalar firm_cov_share = `shr'
    }
end

* ---------------------------------------------------------------------------
* cw_run — THE estimator of record.  One reghdfe, scalars returned.
*   y        outcome
*   x3       the term being reported (the triple, or the interaction in the
*            US-only block)
*   x2       lower-order control kept in the fit (may be empty)
*   absorb   FE spec — "cg ct gt" for the DDD (the country-level analogue of
*            the firm-level fq/gq/ig), "ctry qtr" for the US-only block
*   clus     vce(cluster ...) spec — "ctry qtr" (two-way) for every block
* ---------------------------------------------------------------------------
capture program drop cw_run
program define cw_run, rclass
    syntax , y(varname) x3(varname) absorb(string) clus(string) ///
             [x2(varname) ifcond(string)]
    tempvar smp
    quietly reghdfe `y' `x2' `x3' `ifcond', absorb(`absorb') vce(cluster `clus')
    quietly gen byte `smp' = e(sample)
    local b  = _b[`x3']
    local se = _se[`x3']
    local df = e(df_r)
    local N  = e(N)
    local t  = `b'/`se'
    local p  = 2*ttail(`df', abs(`t'))
    quietly levelsof ctry if `smp' == 1, local(_lc)
    quietly levelsof qtr  if `smp' == 1, local(_lq)
    local nc : word count `_lc'
    local nq : word count `_lq'
    return scalar b  = `b'
    return scalar se = `se'
    return scalar p  = `p'
    return scalar t  = `t'
    return scalar df = `df'
    return scalar N  = `N'
    return scalar n_countries = `nc'
    return scalar n_quarters  = `nq'
end

* ---------------------------------------------------------------------------
* cw_guard — rotation discipline (project rule): never overwrite a canonical
* artifact.  Rename it to *_cwpre first; REFUSE if that slot is already taken.
* ---------------------------------------------------------------------------
capture program drop cw_guard
program define cw_guard
    * option syntax (not `args'): the project path contains spaces
    * ("OneDrive - Universitat ..."), and named string options remove any doubt
    * about tokenization.
    syntax , target(string) pre(string)
    capture confirm file "`target'"
    if _rc == 0 {
        capture confirm file "`pre'"
        if _rc == 0 {
            display as error "REFUSING: `target' exists AND the rotation slot `pre' is"
            display as error "  already occupied.  Two live vintages — resolve by hand."
            error 602
        }
        display as error "`target' already exists — rename it to `pre' (rotation rule)"
        display as error "  before re-running."
        error 602
    }
end
