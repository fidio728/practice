/*==============================================================================
  run_pead_three_protocol_treatment_sensitivity.do

  Unified treatment-sensitivity runner for three PEAD protocols:
    1. HLT 2009-style analyst forecast error
    2. DVP 2009-style forecast error
    3. Martineau 2022-style PEAD power / microcap setting

  Goal:
    Estimate each protocol under its own variable construction and sample gates,
    but write a common reporting schema so the resulting tables can be compared.

  Treatments:
    robots_spec      Bot-specific robots.txt blocking
    robots_any       Any tracked robots.txt blocking
    robots_sb_spec   Search-bot-specific robots.txt blocking
    broad_discovery  Broad23 discovery cohort

  Output:
    pead_three_protocol_treatment_sensitivity.csv
==============================================================================*/

version 17
clear all
set more off
set linesize 220

local panel  `"C:/Users/xl/OneDrive - Universitat Ram`=uchar(243)'n Llull/bot_paper/panel_construction"'
local report `"C:/Users/xl/OneDrive - Universitat Ram`=uchar(243)'n Llull/git/practice"'
local master "`panel'/pead_master_panel_serious.csv"
local broad  "`panel'/firm_discovery_treatment.csv"
local out    "`report'/pead_three_protocol_treatment_sensitivity.csv"

tempname fh
file open `fh' using "`out'", write replace
file write `fh' "protocol,surprise_form,treatment,slice,window,sample,vce_mode,controls_style,"
file write `fh' "n_obs,n_firms,n_time,n_treat_firms_post,r2_within,"
file write `fh' "beta_b,beta_se,beta_p,alpha_b,alpha_se,alpha_p,"
file write `fh' "gamma_b,gamma_se,gamma_p,implied_scale,implied_effect,fe,vce_used,note" _n

* ---------------------------------------------------------------------------
* Build broad-discovery treatment file for merge
* ---------------------------------------------------------------------------
preserve
    import delimited "`broad'", varnames(1) case(lower) stringcols(_all) clear
    quietly replace gvkey = string(real(gvkey), "%06.0f")
    destring canon_broad_treated_strict clean_never_broad ///
             status_only_broad status_before_timestamp_broad ///
             future_treated_broad, replace force
    gen broad_block_ym = real(subinstr(first_discovery_block_month, "-", "", .))
    keep gvkey canon_broad_treated_strict clean_never_broad ///
         status_only_broad status_before_timestamp_broad future_treated_broad ///
         broad_block_ym
    isid gvkey
    tempfile broad_tf
    save `broad_tf', replace
restore

* ---------------------------------------------------------------------------
* Load master
* ---------------------------------------------------------------------------
import delimited "`master'", varnames(1) case(lower) stringcols(_all) clear

destring bhar_0_1 bhar_2_5 bhar_6_15 bhar_16_30 bhar_31_60 bhar_2_60 ///
         bhar_2_30 bhar_2_45 bhar_2_61 bhar_2_75 bhar_2_90 bhar_3_61 ///
         fe_detail_w fe_detail_decile ///
         fe_detail_dvp_w fe_detail_dvp_q11 ///
         fe_detail_martineau_decile_lag sue_martineau_decile_lag ///
         ann_ym ann_year ann_month rdq_calq ///
         cohort_robots_spec_main cohort_robots_any_main cohort_robots_sb_spec_main ///
         treat_post_robots_spec treat_post_robots_any treat_post_robots_sb_spec ///
         ann_diff_gt30 ann_diff_le5 ann_is_weekend ///
         lm_gate_pass microcap_flag abm_flag ///
         log_atq mb_q leverage_q roa_q tangibility_q log_n_analysts reporting_lag, ///
         replace force

quietly replace gvkey = string(real(gvkey), "%06.0f")
egen long gvkey_id = group(gvkey)

gen int _yr = floor(ann_ym/100)
gen int _mo = mod(ann_ym, 100)
gen ann_ym_n = ym(_yr, _mo)
format ann_ym_n %tm
drop _yr _mo

gen ann_date_d = date(ann_date, "YMD")
format ann_date_d %td
egen long ann_date_id = group(ann_date_d)

gen reporting_lag2 = reporting_lag * reporting_lag
gen reporting_lag3 = reporting_lag2 * reporting_lag

merge m:1 gvkey using `broad_tf', keep(master match) gen(_m_broad)
tab _m_broad
drop _m_broad

gen byte cohort_broad_main = ///
    (((canon_broad_treated_strict == 1) | (clean_never_broad == 1)) ///
     & (status_only_broad == 0) ///
     & (status_before_timestamp_broad == 0) ///
     & (future_treated_broad == 0))

gen byte treat_post_broad = (canon_broad_treated_strict == 1 ///
                             & !missing(broad_block_ym) ///
                             & ann_ym > broad_block_ym)

* ---------------------------------------------------------------------------
* Common controls and slices
* ---------------------------------------------------------------------------
gen byte controls_fe_complete = ///
    ((!missing(log_atq))         ///
     & (!missing(mb_q))           ///
     & (!missing(leverage_q))     ///
     & (!missing(roa_q))          ///
     & (!missing(tangibility_q))  ///
     & (!missing(log_n_analysts)) ///
     & (!missing(reporting_lag))  ///
     & (!missing(reporting_lag2)) ///
     & (!missing(reporting_lag3)))

gen byte slice_full = 1
gen byte slice_abm = (abm_flag == 1)
gen byte slice_microcap = (microcap_flag == 1)

* ---------------------------------------------------------------------------
* Protocol-specific treatment-independent bases
* ---------------------------------------------------------------------------
gen byte base_hlt_cont = (!missing(fe_detail_w) ///
                          & ann_diff_gt30 == 0 ///
                          & controls_fe_complete == 1)
gen byte base_hlt_decile = (!missing(fe_detail_decile) ///
                            & ann_diff_gt30 == 0 ///
                            & controls_fe_complete == 1)

gen byte base_dvp_cont = (!missing(fe_detail_dvp_w) ///
                          & ann_diff_le5 == 1 ///
                          & ann_is_weekend == 0 ///
                          & controls_fe_complete == 1)
gen byte base_dvp_q11 = (!missing(fe_detail_dvp_q11) ///
                         & ann_diff_le5 == 1 ///
                         & ann_is_weekend == 0 ///
                         & controls_fe_complete == 1)

gen byte base_mart_fe = (!missing(fe_detail_martineau_decile_lag) ///
                         & ann_diff_gt30 == 0 ///
                         & ann_is_weekend == 0 ///
                         & lm_gate_pass == 1)
gen byte base_mart_sue = (!missing(sue_martineau_decile_lag) ///
                          & ann_diff_gt30 == 0 ///
                          & lm_gate_pass == 1)

* Per-treatment samples.  Surprise ranks/values stay treatment-independent;
* only the treatment cohort changes.
foreach trt in robots_spec robots_any robots_sb broad {
    if "`trt'" == "robots_spec" local cohort "cohort_robots_spec_main"
    if "`trt'" == "robots_any"  local cohort "cohort_robots_any_main"
    if "`trt'" == "robots_sb"   local cohort "cohort_robots_sb_spec_main"
    if "`trt'" == "broad"       local cohort "cohort_broad_main"

    foreach base in hlt_cont hlt_decile dvp_cont dvp_q11 mart_fe mart_sue {
        gen byte sample_`base'_`trt' = (base_`base' == 1 & `cohort' == 1)
    }
}

* Fixed implied-effect scales.  Continuous scales are p90-p10 in the
* robots_spec full-sample version of each protocol, then reused across
* treatment definitions for comparability.
quietly summarize fe_detail_w if sample_hlt_cont_robots_spec == 1, detail
local scale_hlt_cont = r(p90) - r(p10)
quietly summarize fe_detail_dvp_w if sample_dvp_cont_robots_spec == 1, detail
local scale_dvp_cont = r(p90) - r(p10)
local scale_decile = 9
local scale_q11 = 10

* ---------------------------------------------------------------------------
* Helper
* ---------------------------------------------------------------------------
capture program drop _run_one
program define _run_one
    args fh protocol surprise_form surprise treatment_label treatment_var slice_label slice_var window sample timefe controls_style scale

    local outcome "bhar_`window'"

    if "`controls_style'" == "controls_x_surprise" {
        local rhs c.`surprise'##c.`treatment_var' ///
                  c.log_atq c.mb_q c.leverage_q c.roa_q c.tangibility_q ///
                  c.log_n_analysts c.reporting_lag c.reporting_lag2 c.reporting_lag3 ///
                  c.`surprise'#c.log_atq c.`surprise'#c.mb_q ///
                  c.`surprise'#c.leverage_q c.`surprise'#c.roa_q ///
                  c.`surprise'#c.tangibility_q c.`surprise'#c.log_n_analysts ///
                  c.`surprise'#c.reporting_lag c.`surprise'#c.reporting_lag2 ///
                  c.`surprise'#c.reporting_lag3
    }
    else {
        local rhs c.`surprise'##c.`treatment_var'
    }

    if "`timefe'" == "ann_month" {
        local absorb_fe "gvkey_id ann_ym_n"
        local time_id "ann_ym_n"
        local fe_label "Firm_FE+announcement_month_FE"
    }
    else if "`timefe'" == "rdq_quarter" {
        local absorb_fe "gvkey_id rdq_calq"
        local time_id "rdq_calq"
        local fe_label "Firm_FE+announcement_quarter_FE"
    }
    else {
        di as error "Unknown timefe `timefe'"
        exit 198
    }

    capture confirm variable `outcome'
    if _rc {
        file write `fh' "`protocol',`surprise_form',`treatment_label',`slice_label',`window',`sample',firm_cluster,`controls_style',"
        file write `fh' ".,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,`fe_label',gvkey_id,missing_outcome" _n
        exit
    }

    capture noisily reghdfe `outcome' `rhs' ///
        if (`sample' == 1) & (`slice_var' == 1), ///
        absorb(`absorb_fe') vce(cluster gvkey_id)

    if _rc != 0 {
        file write `fh' "`protocol',`surprise_form',`treatment_label',`slice_label',`window',`sample',firm_cluster,`controls_style',"
        file write `fh' ".,.,.,.,.,.,.,.,.,.,.,.,.,.,.,.,`fe_label',gvkey_id,error_rc_`=_rc'" _n
        exit
    }

    capture lincom c.`surprise'
    if _rc == 0 {
        scalar beta_b = r(estimate)
        scalar beta_se = r(se)
        scalar beta_p = r(p)
    }
    else {
        scalar beta_b = .
        scalar beta_se = .
        scalar beta_p = .
    }

    capture lincom c.`treatment_var'
    if _rc == 0 {
        scalar alpha_b = r(estimate)
        scalar alpha_se = r(se)
        scalar alpha_p = r(p)
    }
    else {
        scalar alpha_b = .
        scalar alpha_se = .
        scalar alpha_p = .
    }

    capture lincom c.`surprise'#c.`treatment_var'
    if _rc == 0 {
        scalar gamma_b = r(estimate)
        scalar gamma_se = r(se)
        scalar gamma_p = r(p)
    }
    else {
        scalar gamma_b = .
        scalar gamma_se = .
        scalar gamma_p = .
    }

    scalar implied_scale = `scale'
    scalar implied_effect = gamma_b * `scale'
    scalar n_obs = e(N)
    scalar r2w = e(r2_within)

    quietly tab gvkey_id if e(sample), nofreq
    local n_firms = r(r)
    quietly tab `time_id' if e(sample), nofreq
    local n_time = r(r)
    quietly tab gvkey_id if e(sample) & `treatment_var' == 1, nofreq
    local n_treat = r(r)

    local bad_se = (missing(gamma_se) | gamma_se <= 1e-12 | missing(gamma_p))
    if `bad_se' {
        file write `fh' "`protocol',`surprise_form',`treatment_label',`slice_label',`window',`sample',firm_cluster,`controls_style',"
        file write `fh' "`=n_obs',`n_firms',`n_time',`n_treat',`=r2w',"
        file write `fh' "`=beta_b',`=beta_se',`=beta_p',"
        file write `fh' "`=alpha_b',`=alpha_se',`=alpha_p',"
        file write `fh' "`=gamma_b',.,.,`=implied_scale',.,`fe_label',gvkey_id,firm_cluster_singular" _n
        exit
    }

    file write `fh' "`protocol',`surprise_form',`treatment_label',`slice_label',`window',`sample',firm_cluster,`controls_style',"
    file write `fh' "`=n_obs',`n_firms',`n_time',`n_treat',`=r2w',"
    file write `fh' "`=beta_b',`=beta_se',`=beta_p',"
    file write `fh' "`=alpha_b',`=alpha_se',`=alpha_p',"
    file write `fh' "`=gamma_b',`=gamma_se',`=gamma_p',"
    file write `fh' "`=implied_scale',`=implied_effect',`fe_label',gvkey_id,ok" _n
end

* ---------------------------------------------------------------------------
* Run matrices
* ---------------------------------------------------------------------------
local trt_labels "robots_spec robots_any robots_sb broad"
local trt_vars   "treat_post_robots_spec treat_post_robots_any treat_post_robots_sb_spec treat_post_broad"

* HLT: original continuous FE plus rank companion, full sample, HLT windows.
foreach form in hlt_cont hlt_decile {
    if "`form'" == "hlt_cont" {
        local protocol "HLT2009"
        local surprise_form "FE_continuous"
        local surprise "fe_detail_w"
        local scale "`scale_hlt_cont'"
    }
    else {
        local protocol "HLT2009"
        local surprise_form "FE_decile_companion"
        local surprise "fe_detail_decile"
        local scale "`scale_decile'"
    }

    forvalues ti = 1/4 {
        local trt : word `ti' of `trt_labels'
        local trtvar : word `ti' of `trt_vars'
        foreach window in 0_1 2_30 2_45 2_61 2_75 2_90 3_61 {
            _run_one `fh' "`protocol'" "`surprise_form'" "`surprise'" "`trt'" "`trtvar'" ///
                "full" "slice_full" "`window'" "sample_`form'_`trt'" ///
                "ann_month" "controls_x_surprise" "`scale'"
        }
    }
}

* DVP: original continuous FE plus sign-split q11, full sample, DVP/HLT windows.
foreach form in dvp_cont dvp_q11 {
    if "`form'" == "dvp_cont" {
        local protocol "DVP2009"
        local surprise_form "FE_continuous_30d"
        local surprise "fe_detail_dvp_w"
        local scale "`scale_dvp_cont'"
    }
    else {
        local protocol "DVP2009"
        local surprise_form "FE_q11_signsplit"
        local surprise "fe_detail_dvp_q11"
        local scale "`scale_q11'"
    }

    forvalues ti = 1/4 {
        local trt : word `ti' of `trt_labels'
        local trtvar : word `ti' of `trt_vars'
        foreach window in 0_1 2_30 2_45 2_61 2_75 2_90 3_61 {
            _run_one `fh' "`protocol'" "`surprise_form'" "`surprise'" "`trt'" "`trtvar'" ///
                "full" "slice_full" "`window'" "sample_`form'_`trt'" ///
                "ann_month" "controls_x_surprise" "`scale'"
        }
    }
}

* Martineau: FE decile and SUE decile, full/all-but-microcap/microcap,
* Martineau short-window grid. No controls, matching the existing D2 layer.
foreach form in mart_fe mart_sue {
    if "`form'" == "mart_fe" {
        local protocol "Martineau2022"
        local surprise_form "FE_decile_lag90d"
        local surprise "fe_detail_martineau_decile_lag"
    }
    else {
        local protocol "Martineau2022"
        local surprise_form "SUE_decile_lag"
        local surprise "sue_martineau_decile_lag"
    }
    local scale "`scale_decile'"

    forvalues ti = 1/4 {
        local trt : word `ti' of `trt_labels'
        local trtvar : word `ti' of `trt_vars'
        foreach slice in full abm microcap {
            local slicevar "slice_`slice'"
            foreach window in 0_1 2_5 6_15 16_30 31_60 2_60 {
                _run_one `fh' "`protocol'" "`surprise_form'" "`surprise'" "`trt'" "`trtvar'" ///
                    "`slice'" "`slicevar'" "`window'" "sample_`form'_`trt'" ///
                    "rdq_quarter" "none" "`scale'"
            }
        }
    }
}

file close `fh'
di _n "Wrote `out'"
