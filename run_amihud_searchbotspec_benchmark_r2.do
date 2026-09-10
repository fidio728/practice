/*==============================================================================
  run_amihud_searchbotspec_benchmark_r2.do

  Reporting-only TWFE benchmark robustness for the SearchBotSpec robots.txt
  treatment.  This aligns the pre-filing benchmark table with the new
  robots.txt homepage treatment rather than the old broad-discovery treatment.

  Output:
    amihud_searchbotspec_benchmark_r2.csv
==============================================================================*/

version 17
clear all
set more off
set linesize 200

local panel  `"C:/Users/xl/OneDrive - Universitat Ram`=uchar(243)'n Llull/bot_paper/panel_construction"'
local report `"C:/Users/xl/OneDrive - Universitat Ram`=uchar(243)'n Llull/git/practice"'
local inp    "`panel'/edgar_filing_event_panel.csv"
local robcsv "`panel'/firm_robots_treatment.csv"
local out    "`report'/amihud_searchbotspec_benchmark_r2.csv"

tempname fh
file open `fh' using "`out'", write replace
file write `fh' "benchmark,window,outcome,b,se,p,n_obs,n_firms,n_post,n_canon,n_clean,fe,r2_within" _n

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
* Merge SearchBotSpec treatment
* ---------------------------------------------------------------------------
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
    destring robots_sb_spec_first_block robots_sb_spec_canon_treated ///
             robots_sb_spec_clean_never, replace force
    keep gvkey robots_sb_spec_first_block robots_sb_spec_canon_treated ///
         robots_sb_spec_clean_never
    isid gvkey
    tempfile robots
    save `robots', replace
restore

merge m:1 gvkey using `robots', keep(master match) gen(_m_robots)
tab _m_robots
drop _m_robots

quietly replace robots_sb_spec_canon_treated = 0 if missing(robots_sb_spec_canon_treated)
quietly replace robots_sb_spec_clean_never   = 0 if missing(robots_sb_spec_clean_never)

gen byte sub_canon = robots_sb_spec_canon_treated
gen byte sub_clean = robots_sb_spec_clean_never
gen long sub_first_block = robots_sb_spec_first_block
gen byte sub_filing_in_first = (filing_ym_int == sub_first_block) & !missing(sub_first_block)
gen byte sub_post = (sub_canon == 1) & !missing(sub_first_block) & (filing_ym_int > sub_first_block)
gen byte sub_sample = (in_sample == 1) & (sub_filing_in_first == 0) & (sub_canon == 1 | sub_clean == 1)

* ---------------------------------------------------------------------------
* TWFE benchmark robustness
* ---------------------------------------------------------------------------
local benches "main b45 b60 b6016"
local windows "0_5 6_30"

foreach bench of local benches {
    foreach win of local windows {
        if "`bench'" == "main" {
            local y "abn_amihud_mean_`win'"
        }
        else {
            local y "abn_amihud_mean_`win'_`bench'"
        }

        capture confirm variable `y'
        if _rc {
            file write `fh' "`bench',`win',`y',.,.,.,.,.,.,.,.,firm_month_FE,.,missing_outcome" _n
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
            file write `fh' "`bench',`win',`y',.,.,.,.,.,.,.,.,firm_month_FE,.,failed" _n
            continue
        }

        scalar b = _b[sub_post]
        scalar se = _se[sub_post]
        scalar pv = 2 * ttail(e(df_r), abs(b / se))
        scalar nobs = e(N)
        scalar r2w = e(r2_within)

        quietly tab gvkey_id if e(sample), nofreq
        local nf = r(r)
        quietly count if e(sample) & sub_post == 1
        local npost = r(N)
        quietly count if e(sample) & sub_canon == 1
        local ncanon = r(N)
        quietly count if e(sample) & sub_clean == 1
        local nclean = r(N)

        file write `fh' "`bench',`win',`y',`=b',`=se',`=pv',`=nobs',`nf',`npost',`ncanon',`nclean',firm_month_FE,`=r2w'" _n
    }
}

file close `fh'
di _n "Wrote `out'"
