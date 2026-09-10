*=============================================================================*
* run_wx_crawler_purpose_did.do
*
* DESIGN WX - within-firm cross-crawler-PURPOSE event-time DiD (DDD-flavored).
*
* THE GENUINELY-CAUSAL design from the causal hunt. The dependent variable is
* the firm's robots.txt BLOCKING CHOICE - NEVER a market/price/liquidity
* outcome. Identification is WITHIN firm ACROSS two crawler PURPOSES at the
* same relative event-time, so the firm-level selection / web-presence /
* sophistication / IR-posture confounds that killed every CROSS-firm design are
* DIFFERENCED OUT by firm x event-month fixed effects.
*
* Unit of observation: firm i x crawler-purpose p in {training,search} x
* event-month e, where e = (calendar month) - (launch month of purpose p's MAIN
* user-agent). Each (i,p) row is centered on ITS OWN crawler launch:
*     training = GPTBot         launched 2023-08  (cal 202308)
*     search   = OAI-SearchBot  launched 2024-07  (cal 202407)
* so e = 0 is the launch month for whichever purpose that row tracks.
*
* Outcome (HEADLINE LOCK = cumulative-blocked indicator, monotone step 0->1):
*     block_{i,p,e} = 1[ first_block_{i,p} nonmissing & first_block_{i,p} <= launch_p + e ]
*
* Estimating equation (theory):
*     block_{i,p,e} = b*Post_e + g*(Post_e x Training_p) + alpha_{i,e} + delta_p + eps
*   Post_e is ABSORBED by firm x event-month FE (constant across p within (i,e));
*   delta_p (purpose FE) is the level term; g on (Post x Training) is the ONLY
*   estimable structural parameter.
*
*   OPERATIONAL FORM (validated below - see the THREE FIXES + the FE note):
*     gen byte PostTrain = Post*Training
*     reghdfe block PostTrain, absorb(firmid#e_pos Training) vce(cluster firmid e_pos)
*   g = within-firm, same-relative-instant pp gap in P(training surface fenced)
*       vs P(search surface fenced). g>0 => firms fence training harder/sooner.
*       STRICTLY a blocking-CHOICE statement.
*
*   WHY THE EXPLICIT PRODUCT, NOT c.Post#i.Training (validated, important):
*   with absorb(firmid#e_pos), each (firm,event-month) cell holds exactly TWO
*   rows (training + search) sharing the same e. Under the factor form
*   c.Post#i.Training, reghdfe parameterizes 0.Training#c.Post and
*   1.Training#c.Post and OMITS the 1.Training level for collinearity, reporting
*   g under the 0.Training (search) label with a flipped sign - confusing and
*   fragile. The explicit byte PostTrain = Post*Training plus a SEPARATE
*   absorbed purpose FE (the "Training" absorb term = delta_p) yields a single,
*   correctly-signed, identified coefficient (verified: g=+0.0112, t=8.95).
*
* WHY g STAYS IDENTIFIED despite firm x event-month absorbing nearly
* everything: PostTrain VARIES WITHIN an (i,e) cell - it is 1 for the Training
* row and 0 for the Search row whenever e>=0, and 0 for both when e<0. The
* firm x event-month FE cannot absorb a regressor that differs across the two
* purpose rows it spans. So g is identified off the WITHIN-(i,e) gap. The
* identifying set is firms AT-RISK on BOTH purposes whose onset timing differs;
* pure train-only / search-only firms become singletons in firmid#e_pos and are
* dropped by reghdfe (correct - they cannot speak to a within-firm asymmetry).
* Report the both-arm contributing-firm count so e(N) is not misread.
*
* THREE NON-NEGOTIABLE FIXES (all re-verified on disk in prep for this file):
*   FIX 1 (structural-zero pre-period): MAIN training arm = GPTBot-onset ONLY.
*         gptbot_first_block has ZERO onsets before 202308 -> clean structural
*         zero. training_llm_core has 16 onsets before 202308 (5 in 202101,
*         CCBot/legacy-token contamination) which BREAK the structural zero.
*         => broader training family is a ROBUSTNESS column only.
*   FIX 2 (event-time, not calendar): training UAs launched ~11 months before
*         search UAs, so under CALENDAR FE "training fenced first" is partly a
*         mechanical longer-exposure-window artifact. Centering each purpose on
*         its OWN launch removes the differential exposure window. Calendar-FE
*         spec = ROBUSTNESS only (expected biased toward "training first").
*   FIX 3 (legacy placebo, pre-registered): the training-vs-search asymmetry
*         must be AI-SPECIFIC. A training-vs-LEGACY (Googlebot/bingbot) or
*         search-vs-LEGACY contrast must NOT reproduce g. The proprietary-cost
*         moderator must NOT predict the within-firm legacy-naming gap.
*
* STRUCTURAL-ZERO PRE-PERIOD: a purpose-specific bot cannot be named in
* robots.txt before it publicly exists, so block_{i,p,e}=0 for ALL e<0 BY
* CONSTRUCTION for both MAIN arms (verified: gptbot min onset 202308,
* oai_search min onset 202409 -> 0 pre-period blocked rows). CONSEQUENCE for
* the event study: the lead (k<0) interaction coefficients are MECHANICALLY
* zero (the pre-period outcome has no variance) and the joint pre-trend
* testparm is DEGENERATE (constraints dropped, F undefined). That is the
* identifying discipline working, not a failure: the structural zero IS the
* pre-trend evidence. We capture the degenerate test and flag it rather than
* error out. (The BROAD-family robustness arm, which is contaminated by
* pre-2023-08 onsets, is where a genuine nonzero pre-trend can appear.)
*
* DATA SUFFICIENCY: bot_family_specific_treatments.csv (14,615 firms) + launch
* dates are sufficient to build the cumulative-blocked and new-onset-hazard
* outcomes WITHOUT a firm x month robots panel; the cumulative DV is a
* deterministic step from FAM_first_block and the at-risk/clean restriction is
* fully encoded in FAM_has_valid / FAM_clean_before_block / FAM_left_censored /
* FAM_canon_treated. The absorbing-cumulative assumption (no un-blocking) is a
* stated limitation; the discrete-time hazard reframe is the robustness check.
*
* Estimator: reghdfe LPM (NOT csdid/BJS/did_imputation). This is a 2-arm
* WITHIN-unit purpose contrast at common event-time, NOT staggered unit-cohort
* adoption, so the forbidden-comparison / negative-weight pathology those
* packages fix does not arise; they would mis-specify the estimand. reghdfe is
* the skill-endorsed tool for the interacted high-dim FE absorb(firmid#e),
* which areg/xtreg cannot do.
*
* Clustering: two-way cluster(firmid e) [project convention]. The within-firm
* asymmetry is identified off only ~33 both-arm firms (MAIN) -> few-clusters.
* Wild cluster bootstrap (boottest, cluster(firmid)) is the INFERENCE OF RECORD
* for g and the moderator; analytic two-way SE reported alongside.
*
* HARD RULE: do NOT upgrade g to a price/liquidity/Amihud/market claim.
*
* VERIFIED COUNTS (re-derived on disk for this file):
*   MAIN gptbot x oai_search canon: 392 train-only / 5 search-only / 33 both
*     (14 train-first / 2 search-first / 17 same-month); gptbot canon=425,
*     oai_search canon=38; 0 pre-launch onsets for either -> clean.
*   BROAD training_llm_core x core_ai_search canon: 287/10/169 (97/1/71);
*     16 training_llm_core onsets before 202308 -> FIX 1 confirmed.
*=============================================================================*

version 17
clear all
set more off
set linesize 255

* ---- path local (uchar(243) trick for the accented 'o' in Ramon) -----------
local root    `"C:/Users/xl/OneDrive - Universitat Ram`=uchar(243)'n Llull"'
local out_dir `"`root'/git/practice"'
local bot_dir `"`root'/bot_paper"'

local treat_csv  `"`out_dir'/bot_family_specific_treatments.csv"'
local comp_csv   `"`bot_dir'/compustat_panel.csv"'
local results_csv `"`out_dir'/wx_crawler_purpose_did_results.csv"'

capture log close wx
log using `"`out_dir'/wx_crawler_purpose_did.log"', name(wx) replace text

* ---- dependencies ----------------------------------------------------------
capture which reghdfe
if _rc {
    di as error "reghdfe is required. ssc install reghdfe"
    exit 499
}
capture which boottest
local have_boottest = (_rc==0)
if !`have_boottest' {
    di as error "WARNING: boottest not found (ssc install boottest). " ///
        "Wild cluster bootstrap p-values (inference of record) will be missing."
}

* ---- LAUNCH DATES (calendar YYYYMM), from post_event_ua_dates.csv -----------
* MAIN:        GPTBot 2023-08 (training); OAI-SearchBot 2024-07 (search)
* ROBUSTNESS:  Google-Extended 2023-09, ClaudeBot 2023-09 (broad training);
*              Claude-SearchBot 2024-10 (appendix search) - captured in the
*              training_llm_core / core_ai_search FAMILY columns.
* PLACEBO:     traditional_search (Googlebot/bingbot) - no discrete AI launch,
*              anchored to the relevant AI launch for the placebo contrast.
local L_TRAIN  = 202308     /* GPTBot         */
local L_SEARCH = 202407     /* OAI-SearchBot  */

* ---- event-window & coverage cutoff ----------------------------------------
* >=12 pre-months exist for both purposes (data starts 202101). Cap E_MAX where
* BOTH purposes retain coverage; unbalanced post tails reintroduce the
* exposure-window artifact (a strictly-balanced-support robustness runs below).
local E_MIN = -12
local E_MAX =  18
local COVER_CUTOFF = 202512   /* drop rows whose calendar month exceeds data coverage */

*=============================================================================*
* RESULTS postfile - clear, self-documenting schema
*   spec        : named specification block
*   arm_pair    : which two purposes are contrasted
*   estimator   : LPM / LPM-hazard / eventstudy / pretrend / lincom
*   term        : coefficient / statistic label
*   b se p      : analytic point / two-way-cluster SE / analytic two-sided p
*                 (for pretrend rows b holds the F-stat)
*   p_wild      : wild cluster bootstrap p (inference of record for g)
*   n_obs       : e(N) estimation rows
*   n_firms     : distinct firms (firm cluster count) in estimation
*   n_both_arm  : firms contributing BOTH purpose rows (the identifying set)
*   n_treat_t   : at-risk-clean treated firms, training/AI slot (Training=1)
*   n_treat_s   : at-risk-clean treated firms, search/legacy slot (Training=0)
*   note        : free-text flag
*=============================================================================*
tempname P
postfile `P' str32 spec str24 arm_pair str16 estimator str40 term ///
    double b se p p_wild ///
    double n_obs n_firms n_both_arm n_treat_t n_treat_s ///
    str64 note ///
    using `"`out_dir'/_wx_results_tmp.dta"', replace

*-----------------------------------------------------------------------------*
* Helper program: fit is already run; post the named coefficient `cn', with an
* optional wild cluster bootstrap p on the SAME coefficient. Counts are passed
* in (computed once per spec). Guards the dropped/collinear (b=0,se=0) artifact.
*-----------------------------------------------------------------------------*
program define wx_post
    args ph spec arm est term cn nboth ntt nts wild note
    tempname bb vv pp pw
    scalar `bb' = .
    scalar `vv' = .
    scalar `pp' = .
    scalar `pw' = .
    capture scalar `bb' = _b[`cn']
    capture scalar `vv' = _se[`cn']
    if `vv'<. & `vv'>0 {
        scalar `pp' = 2*ttail(e(df_r), abs(`bb'/`vv'))
    }
    if (`bb'==0 & `vv'==0) | `vv'>=. {
        scalar `bb' = .
        scalar `vv' = .
        scalar `pp' = .
    }
    if `wild'==1 & `bb'<. {
        capture which boottest
        if _rc==0 {
            capture boottest `cn'=0, cluster(firmid) boottype(wild) ///
                reps(9999) nograph noci
            if _rc==0 capture scalar `pw' = r(p)
        }
    }
    local enobs   = e(N)
    local enf     = e(N_clust1)
    if "`enf'"=="" | `enf'==. local enf = e(N_clust)
    post `ph' ("`spec'") ("`arm'") ("`est'") ("`term'") ///
        (`bb') (`vv') (`pp') (`pw') ///
        (`enobs') (`enf') (`nboth') (`ntt') (`nts') ("`note'")
end

*=============================================================================*
* PART A. WIDE FIRM-LEVEL TREATMENT FRAME (with 32-char-truncation fix)
*-----------------------------------------------------------------------------*
* import delimited reads zero-padded gvkey ("001004") as numeric -> consistent
* with compustat (also zero-padded). PROJECT GOTCHA: CSV column names >32 chars
* are SILENTLY TRUNCATED at 32 on import. Three *_clean_before_block columns
* exceed 32 chars and arrive truncated; rename them to clean <fam>_cbb aliases
* immediately, then confirm every needed column resolves.
*-----------------------------------------------------------------------------*
import delimited `"`treat_csv'"', clear varnames(1) case(preserve)

* rename the (truncated) long columns -> short aliases; capture so short
* families (which import intact) are renamed by their full name.
capture rename traditional_search_clean_before_ traditional_search_cbb
capture rename training_llm_core_clean_before_b training_llm_core_cbb
capture rename core_ai_search_clean_before_bloc core_ai_search_cbb
capture rename gptbot_clean_before_block        gptbot_cbb
capture rename oai_search_clean_before_block    oai_search_cbb

foreach fam in gptbot oai_search training_llm_core core_ai_search traditional_search {
    confirm variable `fam'_cbb
    confirm variable `fam'_first_block
    confirm variable `fam'_has_valid
    confirm variable `fam'_left_censored
    confirm variable `fam'_canon_treated
    * normalize first_block to integer YYYYMM (strip any .0 float import)
    capture confirm numeric variable `fam'_first_block
    if _rc destring `fam'_first_block, replace force
    replace `fam'_first_block = floor(`fam'_first_block)
}

keep gvkey ///
    gptbot_first_block gptbot_has_valid gptbot_cbb gptbot_left_censored gptbot_canon_treated ///
    oai_search_first_block oai_search_has_valid oai_search_cbb oai_search_left_censored oai_search_canon_treated ///
    training_llm_core_first_block training_llm_core_has_valid training_llm_core_cbb training_llm_core_left_censored training_llm_core_canon_treated ///
    core_ai_search_first_block core_ai_search_has_valid core_ai_search_cbb core_ai_search_left_censored core_ai_search_canon_treated ///
    traditional_search_first_block traditional_search_has_valid traditional_search_cbb traditional_search_left_censored traditional_search_canon_treated

isid gvkey
tempfile WIDE
save `"`WIDE'"', replace
local WIDEFILE `"`WIDE'"'

*=============================================================================*
* PART B. GENERIC PANEL BUILDER for any {ai_fam (Training=1), other_fam (=0)}
*-----------------------------------------------------------------------------*
* args: aifam otherfam Lai Lother emin emax cutoff
*   - aifam  -> Training=1 stub, centered on launch Lai
*   - otherfam -> Training=0 stub, centered on launch Lother
* For the MAIN/BROAD specs the two purposes have their OWN launches
* (FIX 2: event-time). For the LEGACY PLACEBO the legacy arm is anchored to the
* AI purpose's launch (Lother==Lai), so e is centered identically for both.
*
* At-risk/clean gate per purpose-row (= on-disk canon_treated for blockers):
*     has_valid==1 & (clean_before_block==1 | first_block missing) & left_censored==0
*   Never-blocked canon-clean firms enter as never-fencer controls (block stays
*   0 throughout). Left-censored firms (already blocked before first valid obs)
*   are DROPPED - structural zero unestablishable.
*
* Outcomes built deterministically from first_block + launch (no robots panel):
*   block (cumulative, HEADLINE) = 1 if !missing(fb_p) & fb_p <= cm_pe  [absorbing]
*   onset (new-onset hazard, ROBUSTNESS) = 1 only in onset month; firm leaves
*     the risk set after onset (_post_onset flags rows to drop for the hazard).
*   cm_pe = launch_p + e via ym()/dofm() month arithmetic (NEVER raw YYYYMM).
*   !missing(fb_p) guard is mandatory: "." sorts to +inf, so never-blockers
*     would leak in as blocked without it.
*
* GOTCHA fixed: launch_p and cm_pe hold YYYYMM (~202407) which OVERFLOWS Stata
*   `int' (max 32,740) -> they MUST be `long' or they silently go missing.
*
* Leaves a long panel in memory keyed (gvkey, Training, e) with:
*   firmid Training canon_p fb_p e e_pos cm_pe Post PostTrain
*   e_binned LdT* LgT* block onset _post_onset
*-----------------------------------------------------------------------------*
program define wx_build_panel
    args aifam otherfam Lai Lother emin emax cutoff
    use `"$WX_WIDE"', clear

    * ---- AI / Training=1 stub ----
    preserve
        keep gvkey `aifam'_first_block `aifam'_has_valid `aifam'_cbb ///
             `aifam'_left_censored `aifam'_canon_treated
        rename `aifam'_first_block    fb_p
        rename `aifam'_has_valid      hv_p
        rename `aifam'_cbb            cbb_p
        rename `aifam'_left_censored  lc_p
        rename `aifam'_canon_treated  canon_p
        gen byte Training  = 1
        gen long launch_p  = `Lai'
        tempfile _t
        save `"`_t'"', replace
    restore

    * ---- OTHER / Training=0 stub ----
    keep gvkey `otherfam'_first_block `otherfam'_has_valid `otherfam'_cbb ///
         `otherfam'_left_censored `otherfam'_canon_treated
    rename `otherfam'_first_block    fb_p
    rename `otherfam'_has_valid      hv_p
    rename `otherfam'_cbb            cbb_p
    rename `otherfam'_left_censored  lc_p
    rename `otherfam'_canon_treated  canon_p
    gen byte Training  = 0
    gen long launch_p  = `Lother'

    append using `"`_t'"'

    * ---- at-risk / clean gate per purpose-row ----
    gen byte atrisk = (hv_p==1) & (cbb_p==1 | missing(fb_p)) & (lc_p==0)
    keep if atrisk==1

    * ---- expand to event-months e in [emin,emax] ----
    expand (`emax' - `emin' + 1)
    bysort gvkey Training: gen int e = `emin' + _n - 1

    * ---- calendar month via ym()/dofm() (launch_p, cm_pe are LONG) ----
    gen int  ly    = floor(launch_p/100)
    gen int  lm    = mod(launch_p,100)
    gen int  mi_pe = ym(ly,lm) + e
    gen long cm_pe = year(dofm(mi_pe))*100 + month(dofm(mi_pe))
    drop ly lm mi_pe

    * ---- drop rows beyond data coverage (avoid right-censoring artifact) ----
    drop if cm_pe > `cutoff'

    * ---- outcomes ----
    gen byte block       = (!missing(fb_p)) & (fb_p <= cm_pe)       /* HEADLINE cumulative */
    gen byte onset       = (!missing(fb_p)) & (fb_p == cm_pe)       /* hazard new-onset    */
    gen byte _post_onset = (!missing(fb_p)) & (cm_pe > fb_p)        /* at-risk drop-out    */

    * ---- Post and the EXPLICIT within-cell interaction regressor ----
    gen byte Post      = (e >= 0)
    gen byte PostTrain = Post*Training

    * ---- event-time ids (no negative literals in varnames) ----
    gen int e_pos = e - (`emin')          /* 0..(emax-emin) */
    gen int e_binned = e
    replace e_binned = -12 if e <= -12
    replace e_binned =  12 if e >=  12

    * ---- explicit binned lead/lag x Training dummies (baseline e=-1) ----
    * leads k=12..2 (skip -1 baseline), lags k=0..12. Manual product form is the
    * skill-prescribed fix for uniform event-time timing (pure dummies collinear
    * with firmid#e FE); the Training interaction supplies within-cell variation.
    forvalues k = 12(-1)2 {
        gen byte LdT`k' = (e_binned == -`k')*Training
    }
    forvalues k = 0/12 {
        gen byte LgT`k' = (e_binned == `k')*Training
    }

    egen long firmid = group(gvkey)
    isid gvkey Training e
    compress
end

* expose WIDE path to the program via a global (locals don't cross program scope)
global WX_WIDE `"`WIDEFILE'"'

*-----------------------------------------------------------------------------*
* Helper: compute the per-spec treated-firm counts into locals
*   r(ntt)   training/AI-slot canon firms ; r(nts) search/legacy-slot canon
*   r(nboth) firms canon on BOTH arms (identifying set)
*-----------------------------------------------------------------------------*
program define wx_counts, rclass
    preserve
        keep gvkey Training canon_p
        duplicates drop
        quietly count if Training==1 & canon_p==1
        return scalar ntt = r(N)
        quietly count if Training==0 & canon_p==1
        return scalar nts = r(N)
        keep if canon_p==1
        bysort gvkey: gen byte botharm = (_N==2)
        quietly count if botharm==1
        return scalar nboth = r(N)/2
    restore
end

*=============================================================================*
* PART C. MAIN SPECIFICATION - GPTBot (training) vs OAI-SearchBot (search)
*  Verified: gptbot canon=425 (0 pre-202308 onsets, clean structural zero);
*  oai_search canon=38 (0 pre-launch). Both-arm identifying set = 33 firms.
*=============================================================================*
wx_build_panel gptbot oai_search `L_TRAIN' `L_SEARCH' `E_MIN' `E_MAX' `COVER_CUTOFF'
wx_counts
local NTT = r(ntt)
local NTS = r(nts)
local NBOTH = r(nboth)
di as txt "MAIN counts: n_treat_train=`NTT'  n_treat_search=`NTS'  both-arm firms=`NBOTH'"

* structural-zero sanity (pre-period blocked rows must be 0 in BOTH MAIN arms)
quietly count if block==1 & e<0
di as txt "MAIN structural-zero check: pre-period blocked rows = " r(N) " (must be 0)"

* ---- C1. HEADLINE static LPM ----
* g = within-(i,e) cross-purpose gap. firmid#e_pos absorbs Post, firm
* selection, web presence, sophistication, overall fence-propensity at instant
* e; the separate "Training" absorb term is delta_p (purpose level). Singletons
* (firms at-risk on only one purpose) dropped automatically - report both-arm.
reghdfe block PostTrain, absorb(firmid#e_pos Training) vce(cluster firmid e_pos)
wx_post `P' MAIN_static train_vs_search LPM Post#Training PostTrain ///
    `NBOTH' `NTT' `NTS' 1 "headline cumulative-blocked"

* ---- C2. HEADLINE event study (explicit leads/lags), baseline e=-1 ----
* leads (LdT*) are MECHANICALLY ~0 in the MAIN arm (structural-zero pre-period);
* lags (LgT*) trace the training-vs-search divergence. The joint pre-trend
* testparm is DEGENERATE here (no pre-period outcome variance) - we capture it
* and flag rather than error. Posting each coefficient documents the flat
* pre-period + rising post path.
reghdfe block LdT* LgT*, absorb(firmid#e_binned Training) vce(cluster firmid e_binned)
* leads (k<0)
forvalues k = 12(-1)2 {
    wx_post `P' MAIN_eventstudy train_vs_search eventstudy es_em`k' LdT`k' ///
        `NBOTH' `NTT' `NTS' 0 "lead k=-`k'; structural zero"
}
* lags (k>=0)
forvalues k = 0/12 {
    wx_post `P' MAIN_eventstudy train_vs_search eventstudy es_ep`k' LgT`k' ///
        `NBOTH' `NTT' `NTS' 0 "lag k=+`k'"
}
* joint pre-trend test (degenerate under structural zero -> capture)
capture testparm LdT*
local pt_F = .
local pt_p = .
if _rc==0 {
    local pt_F = r(F)
    local pt_p = r(p)
}
post `P' ("MAIN_eventstudy") ("train_vs_search") ("pretrend") ("testparm_leads") ///
    (`pt_F') (.) (`pt_p') (.) (e(N)) (.) (`NBOTH') (`NTT') (`NTS') ///
    ("DEGENERATE: structural-zero pre-period (leads=0)")
* average post effect (k>=0) via lincom
lincom (LgT0+LgT1+LgT2+LgT3+LgT4+LgT5+LgT6+LgT7+LgT8+LgT9+LgT10+LgT11+LgT12)/13
local apb = r(estimate)
local apse = r(se)
local app = .
if `apse'<. & `apse'>0 local app = 2*ttail(e(df_r),abs(`apb'/`apse'))
post `P' ("MAIN_eventstudy") ("train_vs_search") ("lincom") ("avg_post_k>=0") ///
    (`apb') (`apse') (`app') (.) (e(N)) (.) (`NBOTH') (`NTT') (`NTS') ///
    ("mean of k>=0 lag interactions")

* ---- C3. ROBUSTNESS: discrete-time NEW-ONSET hazard ----
* Cumulative 0->1 trends post-launch by construction; the hazard (new-onset,
* at-risk drop-out) confirms the cumulative trend is not mistaken for the
* effect. Must AGREE ON SIGN of g with C1.
preserve
    drop if _post_onset==1
    reghdfe onset PostTrain, absorb(firmid#e_pos Training) vce(cluster firmid e_pos)
    wx_post `P' MAIN_hazard train_vs_search LPM-hazard Post#Training PostTrain ///
        `NBOTH' `NTT' `NTS' 1 "new-onset hazard; sign must match C1"
restore

* ---- C4. ROBUSTNESS: CALENDAR-FE spec (FIX 2 - expected biased) ----
* absorb firmid#calendar-month instead of firmid#event-month. Training launched
* ~11mo earlier -> reintroduces the differential exposure window; expected
* biased TOWARD "training first". Robustness ONLY.
gen int  cmy = floor(cm_pe/100)
gen int  cmm = mod(cm_pe,100)
gen int  cal_mi = ym(cmy,cmm)
reghdfe block PostTrain, absorb(firmid#cal_mi Training) vce(cluster firmid cal_mi)
wx_post `P' MAIN_calendarFE train_vs_search LPM Post#Training PostTrain ///
    `NBOTH' `NTT' `NTS' 1 "calendar FE - biased toward train-first"
drop cmy cmm cal_mi

* ---- C5. ROBUSTNESS: strictly-balanced-e support ----
* Restrict to event-months e where the LATER-launched purpose (search) still
* has coverage (L_SEARCH + e <= COVER_CUTOFF), dropping the unbalanced post tail
* that could re-inject the exposure-window artifact.
preserve
    gen int s_mi = ym(floor(`L_SEARCH'/100), mod(`L_SEARCH',100)) + e
    gen long s_cm = year(dofm(s_mi))*100 + month(dofm(s_mi))
    quietly summarize e if s_cm <= `COVER_CUTOFF'
    local EBAL = r(max)
    drop if e > `EBAL'
    reghdfe block PostTrain, absorb(firmid#e_pos Training) vce(cluster firmid e_pos)
    wx_post `P' MAIN_balanced_e train_vs_search LPM Post#Training PostTrain ///
        `NBOTH' `NTT' `NTS' 1 "balanced e<=`EBAL'"
restore

* save MAIN panel (with firmid) for the moderator merge
tempfile MAINPANEL
save `"`MAINPANEL'"', replace

*=============================================================================*
* PART D. BROADER-TRAINING ROBUSTNESS - training_llm_core vs core_ai_search
*  BROAD-FAMILY pairing. Verified: 287 train-only / 10 search-only / 169 both
*  (97 train-first / 1 search-first / 71 same-month).
*  CONTAMINATION CAVEAT: training_llm_core has 16 onsets before 202308 (5 in
*  202101, CCBot/legacy-token) BREAKING the structural-zero pre-period -> this
*  is exactly why GPTBot-only is the MAIN training arm (FIX 1). Robustness
*  column ONLY; expect a NONZERO pre-trend signature here (unlike MAIN).
*=============================================================================*
wx_build_panel training_llm_core core_ai_search `L_TRAIN' `L_SEARCH' `E_MIN' `E_MAX' `COVER_CUTOFF'
wx_counts
local NTT_B = r(ntt)
local NTS_B = r(nts)
local NBOTH_B = r(nboth)
quietly count if block==1 & e<0
di as txt "BROAD counts: train=`NTT_B' search=`NTS_B' both-arm=`NBOTH_B'  pre-period blocked rows=" r(N) " (expect >0: contamination)"

reghdfe block PostTrain, absorb(firmid#e_pos Training) vce(cluster firmid e_pos)
wx_post `P' BROAD_static broadtrain_vs_search LPM Post#Training PostTrain ///
    `NBOTH_B' `NTT_B' `NTS_B' 1 "robustness; pre-period contaminated"

* broad event study to EXPOSE the contaminated pre-trend explicitly
reghdfe block LdT* LgT*, absorb(firmid#e_binned Training) vce(cluster firmid e_binned)
capture testparm LdT*
local ptB_F = .
local ptB_p = .
if _rc==0 {
    local ptB_F = r(F)
    local ptB_p = r(p)
}
post `P' ("BROAD_eventstudy") ("broadtrain_vs_search") ("pretrend") ("testparm_leads") ///
    (`ptB_F') (.) (`ptB_p') (.) (e(N)) (.) (`NBOTH_B') (`NTT_B') (`NTS_B') ///
    ("expect FAIL - pre-2023-08 contamination")

*=============================================================================*
* PART E. LEGACY-GOOGLEBOT PLACEBO (FIX 3, pre-registered)
*  The training-vs-search asymmetry must be AI-SPECIFIC. Build a legacy arm
*  (traditional_search = Googlebot/bingbot) with a PLACEBO launch anchored to
*  each AI launch, and run two falsification contrasts:
*    (a) TRAINING vs LEGACY: AI=gptbot (Training=1), legacy (Training=0), BOTH
*        centered on GPTBot launch 202308.
*    (b) SEARCH   vs LEGACY: AI=oai_search (Training=1), legacy (Training=0),
*        BOTH centered on OAI-SearchBot launch 202407.
*  PASS CONDITION: g_TL and g_SL SMALL/NULL relative to the MAIN g, and their
*  pre-trends FLAT. Legacy has onsets back to 202101 -> its blocked LEVEL is
*  HIGH and FLAT around any arbitrary AI launch (the inert benchmark a placebo
*  needs); delta_p (purpose FE) absorbs that level. If a legacy contrast
*  reproduces a sharp post-launch divergence of comparable magnitude, the
*  AI-purpose interpretation FAILS. Mirrors the project's sb_spec-vs-cb_spec
*  placebo logic (within-firm analogue).
*  NOTE: here the "training/AI slot" (Training=1) is the AI purpose and the
*  "search/legacy slot" (Training=0) is traditional_search; n_treat_s is the
*  legacy at-risk-clean count (high).
*=============================================================================*

* ---- E1. TRAINING (gptbot) vs LEGACY, placebo launch = 202308 ----
wx_build_panel gptbot traditional_search `L_TRAIN' `L_TRAIN' `E_MIN' `E_MAX' `COVER_CUTOFF'
wx_counts
local NTT_TL = r(ntt)
local NTS_TL = r(nts)
local NBOTH_TL = r(nboth)
reghdfe block PostTrain, absorb(firmid#e_pos Training) vce(cluster firmid e_pos)
wx_post `P' PLACEBO_train_legacy train_vs_legacy LPM Post#Training PostTrain ///
    `NBOTH_TL' `NTT_TL' `NTS_TL' 1 "PASS = null vs MAIN g; legacy=trad_search"
* placebo pre-trend (should be flat; legacy level absorbed by purpose FE)
reghdfe block LdT* LgT*, absorb(firmid#e_binned Training) vce(cluster firmid e_binned)
capture testparm LdT*
local pTL_F = .
local pTL_p = .
if _rc==0 {
    local pTL_F = r(F)
    local pTL_p = r(p)
}
post `P' ("PLACEBO_train_legacy") ("train_vs_legacy") ("pretrend") ("testparm_leads") ///
    (`pTL_F') (.) (`pTL_p') (.) (e(N)) (.) (`NBOTH_TL') (`NTT_TL') (`NTS_TL') ///
    ("placebo pre-trend should be flat")

* ---- E2. SEARCH (oai_search) vs LEGACY, placebo launch = 202407 ----
wx_build_panel oai_search traditional_search `L_SEARCH' `L_SEARCH' `E_MIN' `E_MAX' `COVER_CUTOFF'
wx_counts
local NTT_SL = r(ntt)
local NTS_SL = r(nts)
local NBOTH_SL = r(nboth)
reghdfe block PostTrain, absorb(firmid#e_pos Training) vce(cluster firmid e_pos)
wx_post `P' PLACEBO_search_legacy search_vs_legacy LPM Post#Training PostTrain ///
    `NBOTH_SL' `NTT_SL' `NTS_SL' 1 "PASS = null vs MAIN g; AI slot=oai_search"
reghdfe block LdT* LgT*, absorb(firmid#e_binned Training) vce(cluster firmid e_binned)
capture testparm LdT*
local pSL_F = .
local pSL_p = .
if _rc==0 {
    local pSL_F = r(F)
    local pSL_p = r(p)
}
post `P' ("PLACEBO_search_legacy") ("search_vs_legacy") ("pretrend") ("testparm_leads") ///
    (`pSL_F') (.) (`pSL_p') (.) (e(N)) (.) (`NBOTH_SL') (`NTT_SL') (`NTS_SL') ///
    ("placebo pre-trend should be flat")

*=============================================================================*
* PART F. PROPRIETARY-COST MODERATOR (SECONDARY / SUGGESTIVE - DEMOTED)
*  Triple interaction g_mod on (PostTrain x PropCost_i), PropCost a PREDETERMINED
*  FY2022 proxy (R&D intensity = xrd/at; intangibles intensity = intan/at),
*  measured PRE all launches -> immune to reverse causality. PropCost enters
*  ONLY through the interaction (its level + Training#PropCost absorbed by the
*  firm-level FE component / purpose structure).
*
*  THIN-CELL DISCIPLINE (mandatory): the search arm is tiny (oai_search
*  canon=38, both-arm=33). After intersecting with non-missing FY2022 Compustat
*  the moderator cell collapses to the order of tens of firms. REPORT n per
*  cell; LABEL the whole leg "suggestive, exploratory, power-limited"; wild
*  bootstrap is the inference of record; NEVER headline. Hoberg-Phillips TNIC
*  not on disk -> omitted, noted as future add.
*=============================================================================*

* ---- F0. build predetermined FY2022 moderators from compustat_panel.csv ----
preserve
    import delimited `"`comp_csv'"', clear varnames(1) case(preserve)
    keep if fyear==2022
    keep gvkey xrd intan at
    * xrd -> 0 when missing-but-reporting (project convention)
    replace xrd = 0 if missing(xrd)
    gen double rd_intensity    = xrd/at   if at>0 & !missing(at)
    gen double intan_intensity = intan/at if at>0 & !missing(at)
    * winsorize each intensity at 1/99
    foreach v in rd_intensity intan_intensity {
        quietly summarize `v', detail
        replace `v' = r(p1)  if `v'<r(p1)  & !missing(`v')
        quietly summarize `v', detail
        replace `v' = r(p99) if `v'>r(p99) & !missing(`v')
    }
    bysort gvkey: keep if _n==1     /* FY2022 unique; enforce */
    isid gvkey
    keep gvkey rd_intensity intan_intensity
    tempfile MOD
    save `"`MOD'"', replace
restore

* ---- F1. moderator on the MAIN GPTBot x OAI-SearchBot panel ----
use `"`MAINPANEL'"', clear
merge m:1 gvkey using `"`MOD'"', keep(master match) nogen
foreach v in rd_intensity intan_intensity {
    quietly summarize `v'
    gen double `v'_std = (`v' - r(mean))/r(sd)
    gen double PT_`v'  = PostTrain*`v'_std      /* explicit triple-difference term */
}

* effective moderator cell counts (firm-level, non-missing FY2022 R&D)
preserve
    keep gvkey Training canon_p rd_intensity
    duplicates drop
    keep if canon_p==1 & !missing(rd_intensity)
    quietly count if Training==1
    local NTT_M = r(N)
    quietly count if Training==0
    local NTS_M = r(N)
    bysort gvkey: gen byte botharm = (_N==2)
    quietly count if botharm==1
    local NBOTH_M = r(N)/2
restore
di as txt "MODERATOR cell (non-missing FY2022 R&D): train=`NTT_M' search=`NTS_M' both-arm=`NBOTH_M'  [SUGGESTIVE - tens of firms]"

foreach v in rd_intensity intan_intensity {
    reghdfe block PostTrain PT_`v', absorb(firmid#e_pos Training) vce(cluster firmid e_pos)
    wx_post `P' MODERATOR_`v' train_vs_search LPM-DDD Post#Train#`v' PT_`v' ///
        `NBOTH_M' `NTT_M' `NTS_M' 1 "SUGGESTIVE power-limited"
    * also record the conditioned main g in this subsample for context
    wx_post `P' MODERATOR_`v' train_vs_search LPM-DDD Post#Train_main PostTrain ///
        `NBOTH_M' `NTT_M' `NTS_M' 0 "main g on moderator subsample"
}

* ---- F2. MODERATOR PLACEBO: PropCost must NOT predict the legacy-naming gap ----
* Re-run g_mod with p in {gptbot(Training=1), legacy}. If proprietary cost
* predicts legacy fencing too, the moderator is a generic sophistication proxy,
* not an AI-training-cost signal.
wx_build_panel gptbot traditional_search `L_TRAIN' `L_TRAIN' `E_MIN' `E_MAX' `COVER_CUTOFF'
merge m:1 gvkey using `"`MOD'"', keep(master match) nogen
foreach v in rd_intensity intan_intensity {
    quietly summarize `v'
    gen double `v'_std = (`v' - r(mean))/r(sd)
    gen double PT_`v'  = PostTrain*`v'_std
}
preserve
    keep gvkey Training canon_p rd_intensity
    duplicates drop
    keep if canon_p==1 & !missing(rd_intensity)
    quietly count if Training==1
    local NTT_MP = r(N)
    quietly count if Training==0
    local NTS_MP = r(N)
    bysort gvkey: gen byte botharm = (_N==2)
    quietly count if botharm==1
    local NBOTH_MP = r(N)/2
restore
foreach v in rd_intensity intan_intensity {
    reghdfe block PostTrain PT_`v', absorb(firmid#e_pos Training) vce(cluster firmid e_pos)
    wx_post `P' MODPLACEBO_`v' train_vs_legacy LPM-DDD Post#Train#`v' PT_`v' ///
        `NBOTH_MP' `NTT_MP' `NTS_MP' 1 "MODERATOR PLACEBO - must be null"
}

*=============================================================================*
* PART G. CLOSE OUT - write results CSV
*=============================================================================*
postclose `P'
use `"`out_dir'/_wx_results_tmp.dta"', clear
order spec arm_pair estimator term b se p p_wild ///
      n_obs n_firms n_both_arm n_treat_t n_treat_s note
label var b           "point estimate (LPM coef; F-stat for pretrend rows)"
label var se          "two-way cluster(firmid,e) analytic SE"
label var p           "analytic two-sided p"
label var p_wild      "wild cluster bootstrap p (inference of record for g)"
label var n_obs       "e(N) estimation rows"
label var n_firms     "distinct firms (firm cluster count) in estimation"
label var n_both_arm  "firms contributing BOTH purpose rows (identifying set)"
label var n_treat_t   "at-risk-clean treated firms, training/AI slot (Training=1)"
label var n_treat_s   "at-risk-clean treated firms, search/legacy slot (Training=0)"
export delimited using `"`results_csv'"', replace
di as result "WROTE: `results_csv'"
list spec arm_pair term b se p p_wild n_both_arm n_treat_t n_treat_s ///
    if estimator!="eventstudy", sepby(spec) noobs

capture erase `"`out_dir'/_wx_results_tmp.dta"'
macro drop WX_WIDE
log close wx
*=============================================================================*
* END run_wx_crawler_purpose_did.do
*=============================================================================*
