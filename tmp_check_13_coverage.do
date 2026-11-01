clear all
set more off

local root  "C:/Users/xl/OneDrive - Universitat Ram贸n Llull"
local build "`root'/bot_paper/panel_construction"

use "`build'/gvkey_month_tier2_reg.dta", clear
gen long gvkey_n = real(gvkey)

preserve
    use "`build'/gvkey_month_tier1_reg.dta", clear
    keep if inrange(month_yyyymm, 202101, 202512)
    keep gvkey year_month search_block any_block
    rename search_block tier1_search_block
    rename any_block    tier1_any_block
    isid gvkey year_month
    tempfile t1
    save `t1'
restore
merge 1:1 gvkey year_month using `t1', keep(master match) nogen

preserve
    import delimited "`build'/treatment_anatomy_firmmonth.csv", clear varnames(1) case(preserve)
    capture confirm numeric variable gvkey
    if _rc destring gvkey, replace force
    gen long gvkey_n = gvkey
    gen int yy = real(substr(year_month, 1, 4))
    gen int mm = real(substr(year_month, 6, 2))
    gen long ym = ym(yy, mm)
    format ym %tm
    keep gvkey_n ym search16_specific core3_specific training29_specific all69_specific
    foreach v in search16_specific core3_specific training29_specific all69_specific {
        capture confirm string variable `v'
        if !_rc destring `v', replace force
    }
    isid gvkey_n ym
    tempfile anat
    save `anat'
restore
merge 1:1 gvkey_n ym using `anat', keep(master match) nogen

di "=== Panel scope ==="
quietly count
di "rows_all = " r(N)
quietly count if in_sample == 1
di "rows_in_sample = " r(N)
quietly count if in_sample == 1 & tier2_evaluable == 1
di "rows_in_sample_tier2_eval = " r(N)

di _n "=== Coverage in intended estimation samples ==="

foreach v in tier1_search_block tier1_any_block {
    quietly count if in_sample == 1
    local den = r(N)
    quietly count if in_sample == 1 & !missing(`v')
    local num = r(N)
    di "`v' | sample=in_sample | nonmissing = " `num' " / " `den' " | pct = " %6.2f (100*`num'/`den')
}

foreach v in search_sitemap_block search_sitemap_strict_firmwide {
    quietly count if in_sample == 1 & tier2_evaluable == 1
    local den = r(N)
    quietly count if in_sample == 1 & tier2_evaluable == 1 & !missing(`v')
    local num = r(N)
    di "`v' | sample=in_sample&tier2_eval | nonmissing = " `num' " / " `den' " | pct = " %6.2f (100*`num'/`den')
}

foreach v in search16_specific core3_specific training29_specific all69_specific {
    quietly count if in_sample == 1
    local den = r(N)
    quietly count if in_sample == 1 & !missing(`v')
    local num = r(N)
    di "`v' | sample=in_sample | nonmissing = " `num' " / " `den' " | pct = " %6.2f (100*`num'/`den')
}

di _n "=== Optional extra: positive-rate in intended samples ==="

foreach v in tier1_search_block tier1_any_block {
    quietly count if in_sample == 1 & !missing(`v') & `v' == 1
    local num = r(N)
    quietly count if in_sample == 1 & !missing(`v')
    local den = r(N)
    di "`v' | observed sample positive rate = " %6.2f (100*`num'/`den')
}

foreach v in search_sitemap_block search_sitemap_strict_firmwide {
    quietly count if in_sample == 1 & tier2_evaluable == 1 & !missing(`v') & `v' == 1
    local num = r(N)
    quietly count if in_sample == 1 & tier2_evaluable == 1 & !missing(`v')
    local den = r(N)
    di "`v' | observed sample positive rate = " %6.2f (100*`num'/`den')
}

foreach v in search16_specific core3_specific training29_specific all69_specific {
    quietly count if in_sample == 1 & !missing(`v') & `v' == 1
    local num = r(N)
    quietly count if in_sample == 1 & !missing(`v')
    local den = r(N)
    di "`v' | observed sample positive rate = " %6.2f (100*`num'/`den')
}
