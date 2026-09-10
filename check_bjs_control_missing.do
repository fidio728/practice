version 17
clear all
set more off
local panel `"C:/Users/xl/OneDrive - Universitat Ram`=uchar(243)'n Llull/bot_paper/panel_construction"'
use "`panel'/channels_base.dta", clear
di "N total: " _N
foreach cond in "sub_sample==1" "sub_sample==1 & g_zc==1" "sub_sample==1 & g_zc==0" {
    di _n "COND: `cond'"
    count if `cond'
    di "N=" r(N)
    foreach v in x_logat x_lmktcap x_ret x_io _w {
        count if `cond' & missing(`v')
        di "`v'_missing=" r(N)
    }
    count if `cond' & missing(x_logat, x_lmktcap, x_ret, x_io)
    di "any_control_missing=" r(N)
}
