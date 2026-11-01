version 17
clear all
set more off
set linesize 160

local panel `"C:/Users/xl/OneDrive - Universitat Ram`=uchar(243)'n Llull/bot_paper/panel_construction"'
local inp   "`panel'/edgar_filing_event_panel.csv"
local cat_inp "`panel'/firm_path_category_blocked_at_treatment.csv"
local out   "`panel'/amihud_irpmc_vs_other_results.csv"
local logf  "`panel'/run_irpmc_vs_other_amihud.log"

cap log close
log using "`logf'", replace text

di _n "=============================================================="
di    "run_irpmc_vs_other_amihud.do -- " c(current_date) " " c(current_time)
di    "=============================================================="

* Load event panel and reproduce canonical sample cleanup.
import delimited "`inp'", varnames(1) clear
foreach v in is_10k is_10q {
    capture confirm string variable `v'
    if !_rc {
        gen byte `v'_n = (`v' == "True")
        drop `v'
        rename `v'_n `v'
    }
}

cap drop gvkey_id
egen long gvkey_id = group(gvkey)
format filing_ym_n %tm

preserve
    use `"`panel'/gvkey_month_edgar_did_reg.dta"', clear
    keep if in_sample == 1
    collapse (max) ever_treated_egen_max = search_sitemap_block, by(gvkey)
    gen byte ever_treated_egen = (ever_treated_egen_max == 1)
    drop ever_treated_egen_max
    rename gvkey gvkey_str
    gen long gvkey_num = real(gvkey_str)
    drop gvkey_str
    rename gvkey_num gvkey
    tempfile egen_flag
    quietly save `egen_flag', replace
restore

merge m:1 gvkey using `egen_flag', keep(master match) gen(_m_egen)
tab _m_egen
replace ever_treated_egen = 0 if missing(ever_treated_egen)
drop _m_egen

gen byte canon_treated = (never_treated == 0)
gen byte status_only   = (never_treated == 1) & (ever_treated_egen == 1)
gen byte clean_never   = (never_treated == 1) & (ever_treated_egen == 0)
assert canon_treated + status_only + clean_never == 1

gen byte primary_sample = (in_sample == 1) ///
    & (filing_in_first_block_month == 0) ///
    & (canon_treated == 1 | clean_never == 1)

* Merge firm-level path-category flags.
preserve
    import delimited "`cat_inp'", varnames(1) stringcols(1) clear
    rename gvkey gvkey_str
    gen long gvkey_num = real(gvkey_str)
    drop gvkey_str
    rename gvkey_num gvkey
    isid gvkey
    tempfile cat_flags
    quietly save `cat_flags', replace
restore

merge m:1 gvkey using `cat_flags', keep(master match) gen(_m_cat)
tab _m_cat

quietly count if primary_sample == 1 & canon_treated == 1 & _m_cat != 3
if r(N) > 0 {
    di as error "FATAL: " r(N) " canonical-treated primary rows did not match category flags."
    assert r(N) == 0
}

foreach cat in ir pmc hr geo csr diversity carbon other {
    replace `cat'_blocked_at_treat = 0 if missing(`cat'_blocked_at_treat)
}
drop _m_cat

* Combined category definitions.
gen byte irpmc_blocked = (ir_blocked_at_treat == 1 | pmc_blocked_at_treat == 1)
gen byte any_othercat_blocked = ///
    (hr_blocked_at_treat == 1 | geo_blocked_at_treat == 1 | ///
     csr_blocked_at_treat == 1 | diversity_blocked_at_treat == 1 | ///
     carbon_blocked_at_treat == 1 | other_blocked_at_treat == 1)
gen byte otheronly_blocked = (irpmc_blocked == 0 & any_othercat_blocked == 1)
gen byte nocat_blocked     = (irpmc_blocked == 0 & any_othercat_blocked == 0)

egen byte tag_canon_firm = tag(gvkey) if primary_sample == 1 & canon_treated == 1
replace tag_canon_firm = 0 if missing(tag_canon_firm)

di _n "Canonical-treated firms in primary sample:"
quietly count if tag_canon_firm == 1
local n_t = r(N)
foreach g in irpmc_blocked otheronly_blocked nocat_blocked {
    quietly count if tag_canon_firm == 1 & `g' == 1
    di "  `g': " %4.0f r(N) " / " %4.0f `n_t' " (" %5.1f 100*r(N)/`n_t' "%)"
}
quietly count if tag_canon_firm == 1 & irpmc_blocked == 0
di "  non_irpmc combined: " %4.0f r(N) " / " %4.0f `n_t' " (" %5.1f 100*r(N)/`n_t' "%)"

* Match main canonical outcome: abnormal Amihud [0,+5], winsorized 1/99.
cap drop amihud_w199
gen double amihud_w199 = abn_amihud_mean_0_5
quietly _pctile amihud_w199 if primary_sample == 1, percentiles(1 99)
local lo = r(r1)
local hi = r(r2)
quietly replace amihud_w199 = `lo' if amihud_w199 < `lo' & !missing(amihud_w199)
quietly replace amihud_w199 = `hi' if amihud_w199 > `hi' & !missing(amihud_w199)

tempname fh
file open `fh' using "`out'", write replace
file write `fh' "model,term,b,se,p,n_obs,n_firms,note" _n

* Model A: IR/PMC vs everything else among canonical-treated firms.
gen byte post_irpmc     = post_strict * irpmc_blocked
gen byte post_nonirpmc  = post_strict * (canon_treated == 1 & irpmc_blocked == 0)

di _n "=== Model A: IR/PMC vs non-IR/PMC ==="
reghdfe amihud_w199 post_irpmc post_nonirpmc if primary_sample == 1, ///
    absorb(gvkey_id filing_ym_n) vce(cluster gvkey_id)
local n = e(N)
local nf = e(N_clust)

foreach term in post_irpmc post_nonirpmc {
    local b  = _b[`term']
    local se = _se[`term']
    local pv = 2*ttail(e(df_r), abs(`b'/`se'))
    file write `fh' "irpmc_vs_nonirpmc,`term',`b',`se',`pv',`n',`nf',group_total" _n
    di "  `term': b=" %7.4f `b' " se=" %7.4f `se' " p=" %6.4f `pv'
}
lincom post_irpmc - post_nonirpmc
file write `fh' "irpmc_vs_nonirpmc,diff_irpmc_minus_nonirpmc,`=r(estimate)',`=r(se)',`=r(p)',`n',`nf',lincom_diff" _n
di "  diff IR/PMC - non-IR/PMC: b=" %7.4f r(estimate) " p=" %6.4f r(p)

* Model B: Split the non-IR/PMC bucket into other-category-only vs no category.
gen byte post_otheronly = post_strict * otheronly_blocked
gen byte post_nocat     = post_strict * nocat_blocked

di _n "=== Model B: IR/PMC vs other-only vs no-category ==="
reghdfe amihud_w199 post_irpmc post_otheronly post_nocat if primary_sample == 1, ///
    absorb(gvkey_id filing_ym_n) vce(cluster gvkey_id)
local n = e(N)
local nf = e(N_clust)

foreach term in post_irpmc post_otheronly post_nocat {
    local b  = _b[`term']
    local se = _se[`term']
    local pv = 2*ttail(e(df_r), abs(`b'/`se'))
    file write `fh' "three_bucket,`term',`b',`se',`pv',`n',`nf',group_total" _n
    di "  `term': b=" %7.4f `b' " se=" %7.4f `se' " p=" %6.4f `pv'
}
lincom post_irpmc - post_otheronly
file write `fh' "three_bucket,diff_irpmc_minus_otheronly,`=r(estimate)',`=r(se)',`=r(p)',`n',`nf',lincom_diff" _n
di "  diff IR/PMC - other-only: b=" %7.4f r(estimate) " p=" %6.4f r(p)

lincom post_irpmc - post_nocat
file write `fh' "three_bucket,diff_irpmc_minus_nocat,`=r(estimate)',`=r(se)',`=r(p)',`n',`nf',lincom_diff" _n
di "  diff IR/PMC - no-category: b=" %7.4f r(estimate) " p=" %6.4f r(p)

lincom post_otheronly - post_nocat
file write `fh' "three_bucket,diff_otheronly_minus_nocat,`=r(estimate)',`=r(se)',`=r(p)',`n',`nf',lincom_diff" _n
di "  diff other-only - no-category: b=" %7.4f r(estimate) " p=" %6.4f r(p)

file close `fh'

di _n "Done. Results: `out'"
di "Log: `logf'"
log close
