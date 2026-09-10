version 17
clear all
set more off
set linesize 255

local out_dir "C:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice"
capture log close botfamily
log using "`out_dir'/bot_family_twfe_bjs_internal.txt", name(botfamily) replace text

local event_csv "`out_dir'/bot_family_event_estimation.csv"
local results_csv "`out_dir'/bot_family_twfe_bjs_results.csv"

capture which reghdfe
if _rc {
    di as error "reghdfe is required. Install with: ssc install reghdfe"
    exit 499
}
capture which did_imputation
if _rc {
    di as error "did_imputation is required. Install with: ssc install did_imputation"
    exit 499
}

tempname posth
postfile `posth' str40 comparison str20 estimator str80 term ///
    double b se p n firms treated_firms control_firms using "`out_dir'/bot_family_twfe_bjs_results_tmp.dta", replace

program define post_twfe_coef
    args posth comparison term coefname n firms treated control
    tempname bb vv
    scalar `bb' = .
    scalar `vv' = .
    capture scalar `bb' = _b[`coefname']
    capture scalar `vv' = _se[`coefname']
    scalar pp = 2 * ttail(e(df_r), abs(`bb' / `vv'))
    post `posth' ("`comparison'") ("TWFE") ("`term'") (`bb') (`vv') (pp) (`n') (`firms') (`treated') (`control')
end

program define post_bjs_tau
    args posth comparison term n firms treated control
    tempname bb vv
    scalar `bb' = .
    scalar `vv' = .
    capture scalar `bb' = _b[tau]
    if _rc {
        capture scalar `bb' = _b[effect]
    }
    capture scalar `vv' = _se[tau]
    if _rc {
        capture scalar `vv' = _se[effect]
    }
    capture scalar pp = 2 * normal(-abs(`bb' / `vv'))
    if _rc {
        scalar pp = .
    }
    if `vv' == 0 & `bb' == 0 {
        scalar `bb' = .
        scalar `vv' = .
        scalar pp = .
    }
    post `posth' ("`comparison'") ("BJS") ("`term'") (`bb') (`vv') (pp) (`n') (`firms') (`treated') (`control')
end

program define run_pair
    syntax, COMP(name) G1(name) G2(name) LAB1(string) LAB2(string) POSTH(name)

    di as text "============================================================"
    di as text "PAIR: `comp' | `lab1' vs `lab2'"
    di as text "============================================================"

    preserve
        gen byte arm = .
        replace arm = 0 if `g1'_clean_never == 1 & `g2'_clean_never == 1
        replace arm = 1 if `g1'_canon_treated == 1 & `g2'_clean_never == 1
        replace arm = 2 if `g2'_canon_treated == 1 & `g1'_clean_never == 1
        keep if inlist(arm, 0, 1, 2)

        gen double first_g1_n = ym(floor(`g1'_first_block / 100), mod(`g1'_first_block, 100)) if arm == 1
        gen double first_g2_n = ym(floor(`g2'_first_block / 100), mod(`g2'_first_block, 100)) if arm == 2
        format first_g1_n first_g2_n %tm

        gen byte post_g1 = (arm == 1 & filing_ym_n > first_g1_n) if !missing(filing_ym_n)
        replace post_g1 = 0 if missing(post_g1)
        gen byte post_g2 = (arm == 2 & filing_ym_n > first_g2_n) if !missing(filing_ym_n)
        replace post_g2 = 0 if missing(post_g2)

        keep if !(arm == 1 & filing_ym_n == first_g1_n)
        keep if !(arm == 2 & filing_ym_n == first_g2_n)

        count
        local n_all = r(N)
        egen tag_all = tag(gvkey_id)
        count if tag_all == 1
        local firms_all = r(N)
        egen tag_g1 = tag(gvkey_id) if arm == 1
        count if tag_g1 == 1
        local firms_g1 = r(N)
        egen tag_g2 = tag(gvkey_id) if arm == 2
        count if tag_g2 == 1
        local firms_g2 = r(N)
        egen tag_ctl = tag(gvkey_id) if arm == 0
        count if tag_ctl == 1
        local firms_ctl = r(N)
        drop tag_all tag_g1 tag_g2 tag_ctl
        di as result "exclusive sample: N=" `n_all' " firms=" `firms_all' " `lab1' firms=" `firms_g1' " `lab2' firms=" `firms_g2' " clean-control firms=" `firms_ctl'

        capture noisily reghdfe y_w post_g1 post_g2, absorb(gvkey_id filing_ym_n) vce(cluster gvkey_id)
        if _rc {
            di as error "TWFE failed for `comp' rc=" _rc
        }
        else {
            post_twfe_coef `posth' "`comp'" "`lab1'" "post_g1" e(N) e(N_clust) `firms_g1' `firms_ctl'
            post_twfe_coef `posth' "`comp'" "`lab2'" "post_g2" e(N) e(N_clust) `firms_g2' `firms_ctl'
            capture noisily lincom post_g1 - post_g2
            if _rc {
                post `posth' ("`comp'") ("TWFE") ("diff_`lab1'_minus_`lab2'") ///
                    (.) (.) (.) (e(N)) (e(N_clust)) (`firms_g1') (`firms_ctl')
            }
            else {
                post `posth' ("`comp'") ("TWFE") ("diff_`lab1'_minus_`lab2'") ///
                    (r(estimate)) (r(se)) (r(p)) (e(N)) (e(N_clust)) (`firms_g1') (`firms_ctl')
            }
        }

        tempfile pairdata
        save `pairdata', replace

        foreach a in 1 2 {
            use `pairdata', clear
            if `a' == 1 {
                local lab "`lab1'"
                keep if arm == 0 | arm == 1
                gen double g_n = first_g1_n if arm == 1
            }
            else {
                local lab "`lab2'"
                keep if arm == 0 | arm == 2
                gen double g_n = first_g2_n if arm == 2
            }

            collapse (mean) y_w (firstnm) g_n arm, by(gvkey_id filing_ym_n)
            count
            local n_bjs = r(N)
            egen tag_all = tag(gvkey_id)
            count if tag_all == 1
            local f_bjs = r(N)
            egen tag_t = tag(gvkey_id) if !missing(g_n)
            count if tag_t == 1
            local t_bjs = r(N)
            egen tag_c = tag(gvkey_id) if missing(g_n)
            count if tag_c == 1
            local c_bjs = r(N)
            drop tag_all tag_t tag_c

            capture noisily did_imputation y_w gvkey_id filing_ym_n g_n, ///
                autosample cluster(gvkey_id) maxit(5000) tol(1e-5)
            if _rc {
                di as error "BJS failed for `comp' `lab' rc=" _rc
                post `posth' ("`comp'") ("BJS") ("`lab'") (.) (.) (.) (`n_bjs') (`f_bjs') (`t_bjs') (`c_bjs')
            }
            else {
                post_bjs_tau `posth' "`comp'" "`lab'" `n_bjs' `f_bjs' `t_bjs' `c_bjs'
            }
        }
    restore
end

import delimited using "`event_csv'", clear varnames(1) bindquote(strict) stringcols(1)
destring filing_ym_int filing_ym_n abn_amihud_mean_0_5, replace force

foreach v of varlist *_first_block *_canon_treated *_left_censored *_clean_never {
    destring `v', replace force
}

egen gvkey_id = group(gvkey)

summ abn_amihud_mean_0_5, detail
scalar p1 = r(p1)
scalar p99 = r(p99)
gen double y_w = abn_amihud_mean_0_5
replace y_w = p1 if y_w < p1 & !missing(y_w)
replace y_w = p99 if y_w > p99 & !missing(y_w)

run_pair, comp(core_ai_vs_traditional) ///
    g1(core_ai_search) g2(traditional_search) ///
    lab1(core_ai_search) lab2(traditional_search) posth(`posth')

run_pair, comp(broad_ai_vs_traditional) ///
    g1(broad_ai_discovery) g2(traditional_search) ///
    lab1(broad_ai_discovery) lab2(traditional_search) posth(`posth')

run_pair, comp(training_vs_traditional) ///
    g1(training_llm_core) g2(traditional_search) ///
    lab1(training_llm_core) lab2(traditional_search) posth(`posth')

run_pair, comp(oai_search_vs_gptbot) ///
    g1(oai_search) g2(gptbot) ///
    lab1(oai_search) lab2(gptbot) posth(`posth')

postclose `posth'

use "`out_dir'/bot_family_twfe_bjs_results_tmp.dta", clear
export delimited using "`results_csv'", replace

di as text "Results written to `results_csv'"
list, sepby(comparison) abbreviate(24)

log close botfamily
