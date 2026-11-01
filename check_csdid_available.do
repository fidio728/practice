version 17
clear all
set more off
cap which csdid
di "which_csdid_rc=" _rc
if _rc==0 which csdid
cap which drdid
di "which_drdid_rc=" _rc
if _rc==0 which drdid
