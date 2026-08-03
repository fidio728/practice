* run_audit_f1b_robust.do — robust inference for the F1b local-projection
* horizons significant under CRVE (h=1 p=0.0166, h=2 p=0.0025, h=4 p=0.0349; B7 2026-08-03).
* Overlapping cumulative windows + only 82 quarter-clusters => CRVE SE likely
* understated. Re-test with wild cluster bootstrap (Webb weights), bootstrapping
* on the QUARTER cluster (the few-cluster / serial-overlap dimension), keeping the
* same two-way (firm, quarter) error clustering and the same firm#quarter +
* group#quarter FE.
* WCB: boottest OOM at 10,000 reps on this machine ([155k-169k x 10k] allocation) -
* WCB p/CI missing; inference for LP horizons is anchored by the design-based RI
* (run_ri_3pairwise.py: cum1 p=0.105, cum4 p=0.0388).

clear all
set more off
local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/audit_c6_panel.dta"
use "`DTA'", clear

gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen  rd_m   = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock

capture which boottest
if _rc ssc install boottest, replace

* absorb ONLY firm#quarter (fq); include group#quarter as explicit i.us#i.rd_m
* dummies so boottest works (it rejects >1 absorbed FE set).
foreach h in 1 2 4 {
    display _newline "===== LP h=`h' : CRVE vs wild-cluster bootstrap ====="
    reghdfe cum`h' us_cn us_cn_shock i.us#i.rd_m, absorb(fq) vce(cluster firm_n rd_m)
    local crve_p = 2*ttail(e(df_r), abs(_b[us_cn_shock]/_se[us_cn_shock]))
    display "  CRVE:  b3=" %9.3e _b[us_cn_shock] "  se=" %9.3e _se[us_cn_shock] "  p=" %6.4f `crve_p'
    * wild cluster bootstrap, Webb weights, bootstrap on quarter (few-cluster / overlap dim)
    boottest us_cn_shock, weighttype(webb) reps(9999) bootcluster(rd_m) nograph
    display "  WCB (Webb, bootcluster=quarter, 9999 reps): p=" %6.4f r(p)
    matrix ci = r(CI)
    display "     95% CI: " %9.3e ci[1,1] "  to  " %9.3e ci[1,2]
}
display _newline "Done."
