* run_direction_split.do — DIRECTION SPLIT of the China exposure (2026-08-02).
* sell = China as customer of the EU firm (revenue-exposure LINK-COUNT proxy);
* buy  = China as supplier to the EU firm (input-dependence LINK-COUNT proxy).
* These are relationship-record shares, NOT revenue/cost shares (revenue_percent
* coverage in Revere is too thin to weight: CUSTOMER ~12.8%, SUPPLIER ~0%).
* Additive identity: sell_lag + buy_lag = cn_lag row-wise (asserted upstream in
* build_c6_panel.py), so this is an exact decomposition of the headline measure.
*
* WHY. The pooled headline cannot distinguish (a) no response from (b) opposing
* directional responses that attenuate or partially offset one another in the
* aggregate specification (the pooled coefficient is a variance-weighted
* projection after FE residualization, not a simple average). Decoupling
* mechanisms differ by direction: retaliation/demand risk (sell) vs supply
* disruption/tariff risk (buy).
*
* SPEC. dw = b2s us*sell + b2b us*buy + b3s us*sell*S + b3b us*buy*S
*            + fq + gq + ig, two-way cluster (firm, month). All lower-order
* terms absorbed exactly as in the headline (same argument, two regressor
* blocks instead of one).
*
* INFERENCE MENU (per external review): equality test b3s=b3b; joint b3s=b3b=0;
* pooling validity b2s=b2b & b3s=b3b; corr(sell_lag, buy_lag); firm-quarter
* cells sell-only/buy-only/both/neither; CIs printed by reghdfe. Design-based
* RI in run_ri_direction.py. INDICATOR robustness (any-sell/any-buy dummies) is
* immune to reciprocal double-records (A->B CUSTOMER + B->A SUPPLIER = one
* economic relation counted twice in link counts).
* B9 convention: degenerate-VCE detection + diag CSV.

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
use "`OUT'/c6_panel.dta", clear

gen rd_day = dofc(rdate)
format rd_day %td
egen firm_n = group(firm_str)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
count if missing(rd_day)
assert r(N) == 0

* additive identity survives to Stata (belt after the python suspenders)
assert abs(sell_lag + buy_lag - cn_lag) < 1e-9

gen us_sell       = us * sell_lag
gen us_buy        = us * buy_lag
gen us_sell_shock = us * sell_lag * shock
gen us_buy_shock  = us * buy_lag  * shock
label var us_sell       "US x SellLink(t-1)"
label var us_buy        "US x BuyLink(t-1)"
label var us_sell_shock "US x SellLink(t-1) x S_t"
label var us_buy_shock  "US x BuyLink(t-1) x S_t"

*==============================================================
* Descriptives the equality tests need for interpretation.
*==============================================================
display _newline "=== sell/buy correlation and cell composition (firm-quarter level, cn_lag>0) ==="
preserve
keep if us == 1
duplicates drop firm_str rd_day, force
* corr restricted to exposed firm-quarters: the cn_lag==0 mass sits at (0,0)
* and would mechanically inflate the correlation toward +1.
corr sell_lag buy_lag if cn_lag > 0
gen byte has_sell = sell_lag > 0 if !missing(sell_lag)
gen byte has_buy  = buy_lag  > 0 if !missing(buy_lag)
gen str12 cell = "neither"
replace cell = "both"      if has_sell & has_buy
replace cell = "sell-only" if has_sell & !has_buy
replace cell = "buy-only"  if !has_sell & has_buy
tab cell if cn_lag > 0
restore

*==============================================================
* MAIN: direction split under 3-pairwise FE.
*==============================================================
display _newline _newline "=== MAIN: dw ~ us_sell + us_buy + us_sell_shock + us_buy_shock, fq gq ig ==="
reghdfe dw us_sell us_buy us_sell_shock us_buy_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
estimates store d_main

display _newline "--- equality of triple terms: b3(sell) = b3(buy) ---"
lincom us_sell_shock - us_buy_shock
display _newline "--- joint zero: b3(sell) = b3(buy) = 0 ---"
test us_sell_shock us_buy_shock
display _newline "--- pooling validity: b2(sell)=b2(buy) AND b3(sell)=b3(buy) ---"
test (us_sell = us_buy) (us_sell_shock = us_buy_shock)

*==============================================================
* INDICATOR robustness: any-sell / any-buy dummies (immune to reciprocal
* double-records). Lower-order absorption argument identical.
*==============================================================
gen byte d_sell = sell_lag > 0 if !missing(sell_lag)
gen byte d_buy  = buy_lag  > 0 if !missing(buy_lag)
gen us_dsell       = us * d_sell
gen us_dbuy        = us * d_buy
gen us_dsell_shock = us * d_sell * shock
gen us_dbuy_shock  = us * d_buy  * shock

display _newline _newline "=== INDICATOR robustness: any-sell / any-buy dummies, fq gq ig ==="
reghdfe dw us_dsell us_dbuy us_dsell_shock us_dbuy_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
estimates store d_ind
display _newline "--- indicator equality: b3(any-sell) = b3(any-buy) ---"
lincom us_dsell_shock - us_dbuy_shock

*==============================================================
* B9: degenerate-VCE detection + diag CSV.
*==============================================================
tempname fh
file open `fh' using "`OUT'/direction_vce_diag.csv", write replace
file write `fh' "spec,coef,b,se,se_valid" _n
local any_degen 0
foreach m in d_main d_ind {
    estimates restore `m'
    local coefs "us_sell_shock us_buy_shock"
    if "`m'" == "d_ind" {
        local coefs "us_dsell_shock us_dbuy_shock"
    }
    foreach cf of local coefs {
        local bb = _b[`cf']
        local ss = _se[`cf']
        local ok = (!missing(`ss') & `ss' > 0)
        file write `fh' "`m',`cf',`bb',`ss',`ok'" _n
        if `ok' == 0 {
            local any_degen 1
            display as error ">>> `m'/`cf': DEGENERATE VCE — CRVE p INVALID; use run_ri_direction.py. <<<"
        }
    }
}
file close `fh'
di "Wrote `OUT'/direction_vce_diag.csv"
if `any_degen' == 1 {
    display as error "Degenerate two-way-cluster VCE detected; CRVE columns in direction_results.csv are NOT valid inference — use run_ri_direction.py."
}

capture which esttab
if _rc == 0 {
    esttab d_main d_ind using "`OUT'/direction_results.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) ///
        keep(us_sell us_buy us_sell_shock us_buy_shock us_dsell us_dbuy us_dsell_shock us_dbuy_shock) ///
        mtitles("link_share" "indicator") nonumbers plain ///
        addnote("sell/buy are LINK-COUNT proxies with a common supply-chain denominator; sell_lag+buy_lag=cn_lag. Indicator column is immune to reciprocal double-records. VCE validity in direction_vce_diag.csv; design-based RI in run_ri_direction.py.")
    di "Wrote `OUT'/direction_results.csv"
}

display _newline "=== b / se / p ==="
foreach m in d_main d_ind {
    estimates restore `m'
    local klist "us_sell us_buy us_sell_shock us_buy_shock"
    if "`m'" == "d_ind" {
        local klist "us_dsell us_dbuy us_dsell_shock us_dbuy_shock"
    }
    display _newline "--- `m' ---"
    estimates table, b(%12.4e) se(%12.4e) p(%6.4f) keep(`klist')
}
display _newline "Done."
