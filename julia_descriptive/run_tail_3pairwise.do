* run_tail_3pairwise.do — re-run the tail-dummy menu (k=1.645/2/3, one-sided
* right) under the 3-pairwise FE (fq gq ig) so the meeting table is consistent
* with the new headline FE. Mirrors 07e_firmgroup_tail.do's construction exactly:
* sigma computed over the 82 distinct quarters (tag(rd_m)), z > k one-sided.
* SUPERSEDED by run_tercile_3pairwise.do (advisor 2026-08-02); retained for
* provenance/few-cluster-invalidity (F3) only

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
use "`OUT'/c6_panel.dta", clear

gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
gen us_cn = us * cn_lag

* sigma over the 82 distinct quarters (NOT over 462k rows)
egen qtag = tag(rd_m)
quietly summarize shock if qtag==1
local mu = r(mean)
local sd = r(sd)
display "sigma_S over " r(N) " quarters = " %9.4f `sd' "  (mean " %9.4f `mu' ")"

gen z = (shock - `mu') / `sd'

foreach k in 1645 2000 3000 {
    local kv = `k'/1000
    gen tail`k' = (z > `kv') if !missing(z)
    gen us_cn_tail`k' = us * cn_lag * tail`k'
    quietly count if tail`k'==1 & qtag==1
    local ntr = r(N)
    display _newline "===== k=`kv' (treated quarters: `ntr'/82), 3-pairwise (fq gq ig) ====="
    reghdfe dw us_cn us_cn_tail`k', absorb(fq gq ig) vce(cluster firm_n rd_m)
    display "  b3=" %9.3e _b[us_cn_tail`k'] "  se=" %9.3e _se[us_cn_tail`k'] ///
            "  p=" %6.4f 2*ttail(e(df_r),abs(_b[us_cn_tail`k']/_se[us_cn_tail`k']))
}
display _newline "Done."
