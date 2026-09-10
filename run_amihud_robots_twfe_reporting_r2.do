/*==============================================================================
  run_amihud_robots_twfe_reporting_r2.do

  Reporting-only TWFE table for robots.txt filing-window Amihud regressions.

  Purpose:
    Re-estimate the TWFE companion specifications and write N, firm count,
    fixed effects, and within R-squared to a clean CSV for manuscript tables.
    This does NOT replace the BJS estimates used for the main treatment effect.

  Output:
    amihud_robots_twfe_reporting_r2.csv
==============================================================================*/

version 17
clear all
set more off
set linesize 200

local panel  `"C:/Users/xl/OneDrive - Universitat Ram`=uchar(243)'n Llull/bot_paper/panel_construction"'
local report `"C:/Users/xl/OneDrive - Universitat Ram`=uchar(243)'n Llull/git/practice"'
local inp    "`panel'/edgar_filing_event_panel.csv"
local robcsv "`panel'/firm_robots_treatment.csv"
local out    "`report'/amihud_robots_twfe_reporting_r2.csv"

tempname fh
file open `fh' using "`out'", write replace
file write `fh' "treatment,window,outcome,b,se,p,n_obs,n_firms,n_canon,n_clean,fe,r2_within" _n

* ---------------------------------------------------------------------------
* Load filing-event panel
* ---------------------------------------------------------------------------
import delimited "`inp'", varnames(1) clear

capture confirm string variable gvkey
if !_rc {
    quietly replace gvkey = string(real(gvkey), "%06.0f")
}
else {
    tostring gvkey, replace
    quietly replace gvkey = string(real(gvkey), "%06.0f")
}

cap drop gvkey_id
egen long gvkey_id = group(gvkey)
format filing_ym_n %tm

* ---------------------------------------------------------------------------
* Merge selected robots.txt treatment columns
* ---------------------------------------------------------------------------
local treatments robots_sb_spec robots_spec robots_any robots_sb_wild robots_wild robots_root_spec

preserve
    import delimited "`robcsv'", varnames(1) stringcols(_all) clear

    capture confirm string variable gvkey
    if !_rc {
        quietly replace gvkey = string(real(gvkey), "%06.0f")
    }
    else {
        tostring gvkey, replace
        quietly replace gvkey = string(real(gvkey), "%06.0f")
    }

    local keepcols "gvkey"
    foreach t of local treatments {
        foreach s in first_block canon_treated clean_never {
            capture confirm variable `t'_`s'
            if !_rc {
                local keepcols "`keepcols' `t'_`s'"
                capture destring `t'_`s', replace force
            }
        }
    }
    keep `keepcols'
    isid gvkey
    tempfile robots
    save `robots', replace
restore

merge m:1 gvkey using `robots', keep(master match) gen(_m_robots)
tab _m_robots
drop _m_robots

foreach t of local treatments {
    quietly replace `t'_canon_treated = 0 if missing(`t'_canon_treated)
    quietly replace `t'_clean_never   = 0 if missing(`t'_clean_never)
}

* ---------------------------------------------------------------------------
* TWFE companion specifications
* ---------------------------------------------------------------------------
local windows 0_1 0_3 0_5 m5_5 6_30 0_30

foreach pfx of local treatments {
    cap drop sub_canon sub_clean sub_first_block sub_filing_in_first sub_post sub_sample tag_sub

    gen byte sub_canon = `pfx'_canon_treated
    gen byte sub_clean = `pfx'_clean_never
    gen long sub_first_block = `pfx'_first_block
    gen byte sub_filing_in_first = (filing_ym_int == sub_first_block) & !missing(sub_first_block)
    gen byte sub_post = (sub_canon == 1) & !missing(sub_first_block) & (filing_ym_int > sub_first_block)
    gen byte sub_sample = (in_sample == 1) & (sub_filing_in_first == 0) & (sub_canon == 1 | sub_clean == 1)

    egen byte tag_sub = tag(gvkey_id) if sub_sample == 1
    quietly count if tag_sub == 1 & sub_canon == 1
    local n_canon = r(N)
    quietly count if tag_sub == 1 & sub_clean == 1
    local n_clean = r(N)

    foreach win of local windows {
        local y abn_amihud_mean_`win'
        capture confirm variable `y'
        if _rc {
            file write `fh' "`pfx',`win',`y',.,.,.,.,.,`n_canon',`n_clean',firm_month_FE,.,missing_outcome" _n
            continue
        }

        cap drop _w
        gen double _w = `y'
        quietly _pctile _w if sub_sample == 1, percentiles(1 99)
        local p1 = r(r1)
        local p99 = r(r2)
        quietly replace _w = `p1' if sub_sample == 1 & _w < `p1' & !missing(_w)
        quietly replace _w = `p99' if sub_sample == 1 & _w > `p99' & !missing(_w)

        capture noisily reghdfe _w sub_post if sub_sample == 1, ///
            absorb(gvkey_id filing_ym_n) vce(cluster gvkey_id)
        if _rc {
            file write `fh' "`pfx',`win',`y',.,.,.,.,.,`n_canon',`n_clean',firm_month_FE,.,failed" _n
            continue
        }

        scalar b = _b[sub_post]
        scalar se = _se[sub_post]
        scalar pv = 2 * ttail(e(df_r), abs(b / se))
        scalar nobs = e(N)
        scalar r2w = e(r2_within)
        quietly tab gvkey_id if e(sample), nofreq
        local nf = r(r)

        file write `fh' "`pfx',`win',`y',`=b',`=se',`=pv',`=nobs',`nf',`n_canon',`n_clean',firm_month_FE,`=r2w'" _n
    }
}

file close `fh'
di _n "Wrote `out'"
