* ============================================================================
* run_country_panel.do — STEP 4 (advisor meeting 2026-08-04): country-by-quarter
* baseline regression, the counterpart of the state-level design in the
* advisors' GFE paper. v3.1 vintage (P0 as-of snapshot + EM zero-recode +
* MM v2 + DQ filter, canonical rebuild 2026-08-09).
*
* Input : output/country_panel_step4.dta   (build_country_panel.py, v3.1)
* Output: output/country_panel_step4.csv   columns: measure, spec, b, se_crve,
*                                          p_crve, p_wild, N, n_countries
*
* Design, for each measure M in {M1, M2, M3} (country c, quarter t):
*   Block A  spec us_dlog:
*     d(log US holdings)_{c,t} = a_c + a_t + b * S_{t-1} x M_{c,t-1}
*                                + g * M_{c,t-1} + e_{c,t}
*   Block B  spec us_minus_nonus_dlog (country-level analogue of the DDD):
*     same RHS, outcome = dlog US - dlog NONUS (differences out country-quarter
*     common shocks to the country's investable book).
*   S_{t-1} main effect is absorbed by the quarter FE; M_{c,t-1} is not and is
*   kept as a control. b (the interaction) is the coefficient of interest.
*
* SMALL-CLUSTER CAVEAT (also printed at run time): only ~28 country clusters.
* (a) headline SE = CRVE clustered by country (anti-conservative with few
*     clusters); (b) wild cluster bootstrap p-values (Webb weights, 9,999
*     reps) via boottest if installed, else a built-in Kline-Santos-style
*     score bootstrap fallback (Webb weights on null-restricted cluster
*     scores). Treat p_wild as the inference of record.
*
* FRESHNESS GATE (two branches, r1 must-fix M3 2026-08-10):
*   (i) ROLLBACK: hard-fails unless I_ict_panel.parquet and
*       country_measures_m1m2m3.csv mtimes are >= 2026-08-09 17:00 (v3.1
*       window) — a pre-v3.1 vintage swapped back in is refused;
*   (ii) REBUILT-AFTER: hard-fails if country_panel_step4.dta is OLDER than
*       either input — a future (v3.2+) upstream rebuild would otherwise pass
*       branch (i) while the .dta silently stays stale, which is this
*       project's documented top failure mode. The builder gates at build
*       time; branch (ii) is the marginal value of gating again at run time.
* ============================================================================

clear all
set more off
set seed 20260810

local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"
local VMIN "20260809170000"

* ----------------------------------------------------------------------------
* [0] Freshness gate. cd into OUT first so the PowerShell probe uses RELATIVE
* paths: the OneDrive project path contains a non-ASCII character that cmd.exe
* can mojibake under the ANSI codepage, but the working directory is inherited
* by the child process without command-line re-encoding.
* ----------------------------------------------------------------------------
cd "`OUT'"
confirm file "I_ict_panel.parquet"
confirm file "country_measures_m1m2m3.csv"
confirm file "gpr_quarterly_with_shock.parquet"
capture confirm file "country_panel_step4.dta"
if _rc != 0 {
    display as error "missing output/country_panel_step4.dta — run build_country_panel.py first"
    error 601
}
capture erase "_step4_mtime_check.txt"
!powershell -NoProfile -Command "Set-Content -Path '_step4_mtime_check.txt' -Encoding ascii -Value ((Get-Item 'I_ict_panel.parquet').LastWriteTime.ToString('yyyyMMddHHmmss') + ' ' + (Get-Item 'country_measures_m1m2m3.csv').LastWriteTime.ToString('yyyyMMddHHmmss') + ' ' + (Get-Item 'country_panel_step4.dta').LastWriteTime.ToString('yyyyMMddHHmmss'))"
capture confirm file "_step4_mtime_check.txt"
if _rc != 0 {
    display as error "freshness gate: PowerShell mtime probe wrote nothing (input missing or shell blocked)"
    error 601
}
tempname fh
file open `fh' using "_step4_mtime_check.txt", read text
file read `fh' mtline
file close `fh'
erase "_step4_mtime_check.txt"
tokenize "`mtline'"
if "`1'" == "" | "`2'" == "" | "`3'" == "" {
    display as error "freshness gate: could not parse mtime probe output: `mtline'"
    error 459
}
* 14-digit fixed-width yyyyMMddHHmmss stamps: string order == time order
if "`1'" < "`VMIN'" {
    display as error "STALE INPUT: I_ict_panel.parquet mtime `1' < `VMIN' (pre-v3.1 vintage)"
    error 459
}
if "`2'" < "`VMIN'" {
    display as error "STALE INPUT: country_measures_m1m2m3.csv mtime `2' < `VMIN' (pre-v3.1 vintage)"
    error 459
}
* branch (ii), r1 must-fix M3: the .dta must be NEWER than both inputs — an
* upstream rebuild AFTER the .dta was built would otherwise pass branch (i)
* silently while run_country_panel.do consumes a stale panel.
if "`3'" < "`1'" | "`3'" < "`2'" {
    display as error "STALE PANEL: country_panel_step4.dta mtime `3' predates an input"
    display as error "  (I_ict_panel.parquet `1' / country_measures_m1m2m3.csv `2')."
    display as error "  The upstream artifacts were rebuilt AFTER the .dta was built —"
    display as error "  re-run build_country_panel.py (rotate the old .dta to *_r2pre first)."
    error 459
}
display as text "freshness gate PASS: I_ict mtime `1', measures mtime `2' (>= `VMIN'); dta mtime `3' >= both inputs"

* rotation discipline: REFUSE to clobber the canonical CSV
capture confirm file "`OUT'/country_panel_step4.csv"
if _rc == 0 {
    display as error "target output/country_panel_step4.csv already exists —"
    display as error "rename it to country_panel_step4_r2pre.csv (rotation rule) before re-running."
    error 602
}

* (the .dta existence check moved INTO the freshness gate above — the mtime
*  probe needs the file to exist before it can stamp it)

* ----------------------------------------------------------------------------
* [1] Score-bootstrap fallback machinery (used only when boottest is absent).
* Kline-Santos-style: t computed from null-restricted cluster scores
* s_c = SUM_i in c [ e_i(H0) * x~_i ], x~ = interaction partialled on FE +
* control (FWL); reference distribution from Webb-weighted cluster scores.
* ----------------------------------------------------------------------------
mata:
void _step4_scoreboot(string scalar svar, real scalar reps)
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
    st_numscalar("__s4pw", cnt/reps)
}
end

capture program drop step4_scorewcb
program define step4_scorewcb, rclass
    syntax, y(varname) xint(varname) ctrl(varname) [reps(integer 9999)]
    tempvar insmp er xt s
    quietly {
        reghdfe `y' `xint' `ctrl', absorb(ctry rd_q) vce(cluster ctry)
        gen byte `insmp' = e(sample)
        * null-restricted residuals (H0: coefficient on `xint' = 0)
        reghdfe `y' `ctrl' if `insmp', absorb(ctry rd_q) residuals(`er')
        * FWL: partial the interaction on the FE + control
        reghdfe `xint' `ctrl' if `insmp', absorb(ctry rd_q) residuals(`xt')
        gen double `s' = `er' * `xt'
        preserve
        keep if `insmp'
        collapse (sum) __s4sc = `s', by(ctry)
        mata: _step4_scoreboot("__s4sc", `reps')
        restore
    }
    return scalar p = scalar(__s4pw)
    capture scalar drop __s4pw
end

* ----------------------------------------------------------------------------
* [2] Data prep
* ----------------------------------------------------------------------------
use "`OUT'/country_panel_step4.dta", clear
gen rd_day = dofc(rdate)
gen rd_q   = qofd(rd_day)
format rd_q %tq
egen ctry  = group(sec_country)

gen double sxm1 = s_lag * m1_lag
gen double sxm2 = s_lag * m2_lag
gen double sxm3 = s_lag * m3_lag
label var sxm1 "S_{t-1} x M1_{c,t-1}"
label var sxm2 "S_{t-1} x M2_{c,t-1}"
label var sxm3 "S_{t-1} x M3_{c,t-1}"

capture which boottest
local has_boottest = (_rc == 0)
* MEASURED INCOMPATIBILITY (2026-08-10, first rb run): boottest IS installed but
* refuses to run after reghdfe with TWO absorbed FE groups ("Doesn't work after
* reghdfe with more than one set of absorbed fixed effects"), returning WITHOUT
* r(p) and WITHOUT an error code — p_wild came back all-missing while the log
* claimed the boottest engine. This design always absorbs ctry + rd_q, so the
* incompatibility is structural, not data-dependent. Force the built-in
* score-bootstrap fallback (the documented engine for exactly this design);
* never mix a silently failing engine into the inference-of-record column.
local has_boottest = 0
if `has_boottest' {
    display as text "wild-bootstrap engine: boottest (Webb weights, 9999 reps)"
}
if !`has_boottest' {
    display as text "wild-bootstrap engine: score-bootstrap fallback (Webb weights, 9999 reps) — boottest incompatible with the 2-way absorb design (see note above)"
}

* ----------------------------------------------------------------------------
* [3] Estimation loop: 3 measures x 2 blocks
* ----------------------------------------------------------------------------
tempfile results
tempname P
postfile `P' str8 measure str24 spec double(b se_crve p_crve p_wild) long N int n_countries using "`results'"

foreach m of numlist 1/3 {
    foreach blk in A B {
        local y ""
        local spec ""
        if "`blk'" == "A" {
            local y "dlog_us"
            local spec "us_dlog"
        }
        if "`blk'" == "B" {
            local y "dlh_diff"
            local spec "us_minus_nonus_dlog"
        }
        display _newline "===== M`m' | `spec' : `y' on sxm`m' + m`m'_lag | ctry FE + quarter FE, cluster(ctry) ====="
        reghdfe `y' sxm`m' m`m'_lag, absorb(ctry rd_q) vce(cluster ctry)
        local b_  = _b[sxm`m']
        local se_ = _se[sxm`m']
        local df_ = e(df_r)
        local N_  = e(N)
        local Nc_ = e(N_clust)
        local p_  = 2*ttail(`df_', abs(`b_'/`se_'))
        local pw_ = .
        if `has_boottest' {
            boottest sxm`m', cluster(ctry) weighttype(webb) reps(9999) nograph
            local pw_ = r(p)
        }
        if !`has_boottest' {
            step4_scorewcb, y(`y') xint(sxm`m') ctrl(m`m'_lag) reps(9999)
            local pw_ = r(p)
        }
        display as text "  -> b=`b_'  se_crve=`se_'  p_crve=`p_'  p_wild=`pw_'  N=`N_'  clusters=`Nc_'"
        post `P' ("M`m'") ("`spec'") (`b_') (`se_') (`p_') (`pw_') (`N_') (`Nc_')
    }
}
postclose `P'

* ----------------------------------------------------------------------------
* [4] Machine-readable CSV + explicit caveat
* ----------------------------------------------------------------------------
use "`results'", clear
list, noobs sepby(measure)
export delimited using "`OUT'/country_panel_step4.csv", replace

display as result _newline "SMALL-CLUSTER CAVEAT: ~28 country clusters only. CRVE (se_crve, p_crve)"
display as result "is anti-conservative with few clusters; the wild cluster bootstrap p_wild"
display as result "(Webb weights) is the inference of record. S_{t-1} main effect absorbed by"
display as result "quarter FE. Block us_minus_nonus_dlog is the country-level DDD analogue."
display as text  "wrote output/country_panel_step4.csv"
display "DONE run_country_panel.do"
