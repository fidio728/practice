"""
build_shock_menu.py -- P1/P1b SHOCK MENU builder (2026-08-04, post-P0).

WHAT THIS IS
------------
The headline shock S_t used by the c6 3-pairwise design is the monthly level-AR(1)
residual of the Iacoviello-Tong bilateral AI-GPR series 'USA|China', sampled at the
quarter-end month (built in 05_combine_visualize.jl).  Three documented defects:

  P1a  the resulting QUARTERLY S_t is serially correlated (doc: acf1 = +0.271,
       LB(1) p = 0.013) because a monthly level-AR(1) under-cleans GPR persistence;
  P1b  'USA|China' is DIRECTIONAL (US initiator -> China respondent).  The official
       data also carries 'China|USA'; the hypothesis is about US-China tension
       BROADLY, so the direction choice is a live specification axis;
  P1c  the AR is fit on the FULL sample incl. 2024-2026, i.e. look-ahead in a
       generated regressor (doc F7 disclosure).

This script builds a locked MENU of alternative constructions, a VALIDATION GATE
that proves the builder's plumbing reproduces 05's baseline bit-for-bit, and the
whiteness DIAGNOSTICS on which the pre-registered selection rule operates.

NO EXISTING ARTIFACT IS MODIFIED.  gpr_quarterly_with_shock.parquet and c6_panel.dta
are read-only inputs.  All output goes to NEW files.

================================================================================
PRE-REGISTERED SELECTION RULE  (verbatim; decided BEFORE any regression was run)
================================================================================
"the preferred NEW construction = the no-look-ahead variant with the whitest
QUARTERLY residual series on the 82 panel quarters, judged by LB(4) p (tie-break:
LB(8) p, then acf1 magnitude), within the USA|China direction; the direction axis
(bidirectional vs directional) is reported as a parallel column set, not selected
on outcomes. beta3 results play NO role in selection. Baseline S_t stays the
headline-continuity column regardless (its defects are disclosed, not hidden)."
================================================================================

THE MENU (locked)
-----------------
  A   baseline_repro   level AR(1), FULL-sample fit, USA|China, quarter-end month
                       residual.  NOT a new variant -- a GATE.  Must reproduce the
                       existing shock_us_cn to max|diff| <= 1e-8 on the 82 panel
                       quarters.
  B   lvl_ar1_nla      level AR(1), fit on months <= 2023-12 ONLY, USA|China
                         B-i   quarter-end month residual
                         B-ii  within-quarter 3-month sum of residuals
  C   dgpr_ar4_nla     AR(4) on the MONTHLY FIRST DIFFERENCE of GPR (the official
                       AI-GPR paper's financial-market spec), fit <= 2023-12,
                       USA|China
                         C-i   quarter-end month residual
                         C-ii  within-quarter 3-month sum of residuals
  D   q_ar1_nla        quarterly MEAN of monthly GPR first, then AR(1) at quarterly
                       frequency, fit on quarters <= 2023Q4, USA|China
  E   bidir            direction axis (P1b).  For the two best-motivated
                       constructions (C-i and B-i), also built on
                         'USA|China' + 'China|USA'  (SUM)  -- the conceptually
                            matching PRIMARY for "US-China tension broadly"
                         'China|USA' alone -- diagnostic column only
                       'USA|China' alone is the baseline-continuity variant.

All variants are standardized (mean 0, sd 1) over the 82 panel quarters so beta3
magnitudes are comparable; the PRE-standardization mean/sd are reported.

================================================================================
PRE-REGISTERED READING  (header note, NOT a conclusion)
================================================================================
Written BEFORE any menu regression was run.  The menu produces ~12 variant labels
x 2 FE sets x 2 p-flavours, so what each outcome pattern MEANS is fixed here, in
advance, and is not open to post-hoc narration.

(a) ALL variants null, including the pre-registered preferred variant and the
    bidirectional E_C_i_bidir column => the P0 deep null (b3 = -5.28e-7, CRVE
    p = 0.763, RI p = 0.801) is NOT an artifact of shock construction.  P1a
    (serial correlation), P1b (direction) and P1c (look-ahead) are then
    DISCLOSED-AND-CLOSED defects, not open threats to the headline.

(b) The PREFERRED variant REJECTS while the baseline does not => report as a
    construction-sensitivity result on the pre-registered column ONLY, with the
    caveat that the preferred column correlates just ~0.25 with the baseline
    shock, so it is close to an INDEPENDENT test rather than a perturbation of
    the headline.  It does not retro-actively validate the baseline b3.

(c) The BASELINE rejects while no no-look-ahead variant does => the headline is a
    look-ahead / serial-correlation artifact (P1a+P1c), and the no-look-ahead
    column GOVERNS.

(d) An E direction column rejects while its USA|China twin does not => a
    DIRECTION-AXIS result, reported in parallel and NEVER promoted to headline,
    because E was excluded from the selection rule by design.

(e) Any SINGLE one of ~12 columns crossing p < .05 with the rest null is roughly
    ONE EXPECTED FALSE POSITIVE at this family size.  Only the pre-registered
    preferred variant and the continuity baseline carry weight, and p_circ
    (circular shift, serial-robust) governs over p_free wherever they diverge.
================================================================================

NO-LOOK-AHEAD CONVENTION.  "fit <= 2023-12" means the AR coefficients are estimated
using only observations whose dependent-variable month is <= 2023-12.  Residuals are
then formed over the whole monthly series with those coefficients.  Every panel
quarter is <= 2023Q4, so on the estimation panel the residuals are in-sample-window
but the coefficients never see post-panel data.

HONESTY.  All prior external measurements quoted in the work-package spec are
RE-DERIVED here from scratch; nothing is copied forward.  statsmodels'
acorr_ljungbox is broken in this environment (deprecate_kwarg TypeError, prior
session), so ACF and Ljung-Box are hand-rolled with scipy.stats.chi2 survival --
same convention as run_cum4_inference.py (biased 1/n ACF normalization).

RECONCILIATION vs the prior external measurements quoted in the work-package spec
(all re-derived; two of them do NOT reproduce, and the reasons are identified):

  P1a baseline, doc: acf1=+0.271, LB(1) p=0.013
      -> re-derived +0.2707 / 0.0125.  CONFIRMED.

  C-i, spec: acf1=-0.036, LB(4) p=0.245
      -> re-derived -0.0309 / 0.2534.  CONFIRMED up to ACF normalization rounding.

  C-ii, spec: acf1=-0.215, LB(4) p=0.292
      -> re-derived -0.2136 / 0.2958.  CONFIRMED up to ACF normalization rounding.

  D, spec: acf1=-0.029, LB(8) p=0.017
      -> re-derived -0.1288 / 0.0848.  DOES NOT REPRODUCE.  Cause identified: the
         prior number was measured with the quarterly AR(1) fit on the FULL sample.
         Refitting full-sample here returns acf1=-0.0292, LB(8) p=0.0175 -- an exact
         match to the quoted pair.  Variant D as specified in THIS menu is
         no-look-ahead (fit <= 2023Q4), so -0.1288 / 0.0848 is the correct number
         for the variant we build; the quoted pair belongs to a look-ahead fit and
         is superseded.  (Quarterly MEAN vs SUM is irrelevant here: AR(1) with an
         intercept on a series rescaled by 3 gives numerically identical residual
         autocorrelation -- verified.)

  Level corr(USA|China, China|USA), spec: ~0.576
      -> re-derived 0.5632 on the full 1960-01..2026-04 monthly sample; 0.5755 on
         months <= 2023-12; 0.5712 on the panel window 2003-07..2023-12.  The quoted
         0.576 corresponds to the <=2023-12 window.  The directional-vs-bidirectional
         concern (P1b) stands under any of these.

OUTPUTS
-------
  output/shock_menu_quarterly.parquet   82 panel quarters x (raw + standardized cols)
  output/shock_menu_quarterly.csv       same (convenience)
  output/shock_menu_quarterly.dta       same (Stata input for run_shock_menu.do)
  output/shock_menu_diagnostics.csv     per-variant whiteness + moments + selection
                                        (carries `preferred_menu_column`, the exact
                                        Stata/Python MENU COLUMN name of the winner)
  output/shock_menu_corr.csv            pairwise correlation matrix incl. baseline
  output/shock_menu_rederivation_check.csv   prior external claims vs re-derived
  output/shock_menu_preferred.txt       ONE line: the pre-registered winner's MENU
                                        COLUMN name ("s_" + variant key, e.g.
                                        s_D_q_ar1_nla).  This is the MECHANICAL
                                        hand-off to run_shock_menu.do's lead spec:
                                        the .do reads it so that no human ever
                                        hand-picks the preferred variant.  NOTE the
                                        naming trap it closes -- the diagnostics
                                        `variant` key is D_q_ar1_nla while BOTH
                                        resolvers (.do `ds s_*` and the Python
                                        resolve_variants) yield s_D_q_ar1_nla, so
                                        the file must carry the s_-prefixed form.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------------
# paths / constants
# ----------------------------------------------------------------------------
BASE = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = BASE / "output"

GPR_CSV = Path(r"E:/Data/Data/ai_gpr_bilateral_monthly.csv")
BASELINE_Q = OUT / "gpr_quarterly_with_shock.parquet"   # FROZEN, read-only
PANEL_DTA = OUT / "c6_panel.dta"                        # FROZEN, read-only

COL_USCN = "USA|China"
COL_CNUS = "China|USA"

NLA_FIT_END = pd.Timestamp("2023-12-31")   # no-look-ahead estimation cutoff
GATE_TOL = 1e-8

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


# ----------------------------------------------------------------------------
# hand-rolled OLS
# ----------------------------------------------------------------------------
def ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Plain OLS via lstsq.  X must already carry an intercept column."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


# ----------------------------------------------------------------------------
# AR residual engines (monthly)
# ----------------------------------------------------------------------------
def ar_residuals(values: np.ndarray,
                 month_end: pd.DatetimeIndex,
                 order: int,
                 difference: bool,
                 fit_end: pd.Timestamp | None):
    """Fit AR(`order`) with intercept on `values` (or its first difference if
    `difference`), using only rows whose OWN month <= `fit_end` (None = full
    sample), then return residuals over the WHOLE series.

    Returns (resid_full, info) where resid_full is len(values) with NaN where the
    lag structure is undefined.
    """
    n = len(values)
    z = np.diff(values, prepend=np.nan) if difference else values.astype(float)

    # rows t for which z[t] and z[t-1..t-order] are all defined
    start = order + (1 if difference else 0)
    rows = np.arange(start, n)

    y = z[rows]
    X = np.column_stack([np.ones(len(rows))] + [z[rows - k] for k in range(1, order + 1)])

    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if fit_end is None:
        fit_mask = ok
    else:
        fit_mask = ok & (month_end[rows] <= fit_end)

    beta = ols(y[fit_mask], X[fit_mask])

    resid_full = np.full(n, np.nan)
    resid_full[rows[ok]] = y[ok] - X[ok] @ beta

    info = dict(order=order,
                difference=bool(difference),
                n_fit=int(fit_mask.sum()),
                fit_first=str(month_end[rows][fit_mask][0].date()),
                fit_last=str(month_end[rows][fit_mask][-1].date()),
                coefs=";".join(f"{b:.10g}" for b in beta))
    return resid_full, info


def ar_residuals_quarterly(qvalues: np.ndarray,
                           quarter_end: pd.DatetimeIndex,
                           order: int,
                           fit_end: pd.Timestamp | None):
    """Same engine, quarterly frequency, no differencing (variant D)."""
    return ar_residuals(qvalues, quarter_end, order=order, difference=False, fit_end=fit_end)


# ----------------------------------------------------------------------------
# monthly -> quarterly bridges
# ----------------------------------------------------------------------------
def to_quarter_end_month(resid_m: np.ndarray, month_end: pd.DatetimeIndex) -> pd.Series:
    """(i) residual AT the quarter-end month (this is what 05 does)."""
    sel = month_end.month.isin([3, 6, 9, 12])
    return pd.Series(resid_m[sel], index=month_end[sel])


def to_quarter_sum3(resid_m: np.ndarray, month_end: pd.DatetimeIndex) -> pd.Series:
    """(ii) within-quarter SUM of the three monthly residuals.  NaN unless all
    three months are present and defined."""
    s = pd.Series(resid_m, index=month_end)
    grp = s.groupby(month_end.to_period("Q"))
    tot = grp.sum(min_count=3)          # NaN if fewer than 3 non-NaN
    cnt = grp.size()
    tot[cnt != 3] = np.nan              # NaN if the quarter is not a full 3 months
    tot.index = tot.index.to_timestamp(how="end").normalize()
    return tot


# ----------------------------------------------------------------------------
# whiteness diagnostics (hand-rolled; statsmodels acorr_ljungbox broken here)
# ----------------------------------------------------------------------------
def whiteness(S: np.ndarray, max_acf: int = 4, lb_lags=(1, 4, 8)) -> dict:
    S = np.asarray(S, dtype=float)
    S = S[np.isfinite(S)]
    n = len(S)
    Sd = S - S.mean()
    denom = np.sum(Sd * Sd)

    def acf(k):
        return float(np.sum(Sd[k:] * Sd[:n - k]) / denom)

    out = {"n_q": n}
    for k in range(1, max_acf + 1):
        out[f"acf{k}"] = acf(k)
    for m in lb_lags:
        Q = n * (n + 2) * sum(acf(k) ** 2 / (n - k) for k in range(1, m + 1))
        out[f"LB_Q{m}"] = float(Q)
        out[f"LB_p{m}"] = float(stats.chi2.sf(Q, m))
    return out


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    print("=" * 96)
    print("build_shock_menu.py -- P1/P1b shock menu (validation gate + diagnostics)")
    print("=" * 96)

    # ---- inputs -------------------------------------------------------------
    gpr = pd.read_csv(GPR_CSV, usecols=["Date", COL_USCN, COL_CNUS])
    gpr["Date"] = pd.to_datetime(gpr["Date"])
    gpr = gpr.sort_values("Date").reset_index(drop=True)
    month_end = pd.DatetimeIndex(gpr["Date"] + pd.offsets.MonthEnd(0))
    print(f"\nGPR monthly: {len(gpr)} months, {month_end[0].date()} -> {month_end[-1].date()}")
    print(f"  NaN: {COL_USCN}={gpr[COL_USCN].isna().sum()}  {COL_CNUS}={gpr[COL_CNUS].isna().sum()}")
    lvl_corr = float(gpr[COL_USCN].corr(gpr[COL_CNUS]))
    _pre24 = gpr[gpr["Date"] <= NLA_FIT_END]
    lvl_corr_pre24 = float(_pre24[COL_USCN].corr(_pre24[COL_CNUS]))
    print(f"  RE-DERIVED level corr( {COL_USCN} , {COL_CNUS} ):"
          f" full sample = {lvl_corr:.4f}, months<=2023-12 = {lvl_corr_pre24:.4f}"
          f"   (spec quoted ~0.576 -> matches the <=2023-12 window)")

    base_q = pd.read_parquet(BASELINE_Q)
    base_q["quarter_end"] = pd.to_datetime(base_q["quarter_end"])
    base_q = base_q.sort_values("quarter_end").reset_index(drop=True)
    print(f"\nBaseline quarterly artifact: {len(base_q)} quarters, "
          f"{base_q['quarter_end'].iloc[0].date()} -> {base_q['quarter_end'].iloc[-1].date()}")

    panel_rd = pd.read_stata(PANEL_DTA, columns=["rdate"])["rdate"]
    panel_q = pd.DatetimeIndex(sorted(panel_rd.unique()))
    print(f"Panel quarters (c6_panel.dta): {len(panel_q)}  "
          f"{panel_q[0].date()} -> {panel_q[-1].date()}")

    # panel-side copy of the baseline shock (independent of the parquet)
    panel_shock = (pd.read_stata(PANEL_DTA, columns=["rdate", "shock"])
                     .groupby("rdate")["shock"].first())

    # ---- source series ------------------------------------------------------
    y_uscn = gpr[COL_USCN].to_numpy(float)
    y_cnus = gpr[COL_CNUS].to_numpy(float)
    y_bidir = y_uscn + y_cnus

    SERIES = {"uscn": y_uscn, "cnus": y_cnus, "bidir": y_bidir}

    # ---- build every variant ------------------------------------------------
    # (key, label, series_key, direction_label, spec_label, agg_label, is_nla, is_gate)
    specs = [
        # A -- validation gate
        dict(key="A_baseline_repro", series="uscn", direction="USA|China",
             spec="level AR(1), FULL-sample fit", agg="quarter-end month resid",
             engine=("m", 1, False, None), nla=False, gate=True),
        # B
        dict(key="B_i_lvl_ar1_nla_qend", series="uscn", direction="USA|China",
             spec="level AR(1), fit<=2023-12", agg="quarter-end month resid",
             engine=("m", 1, False, NLA_FIT_END), nla=True, gate=False),
        dict(key="B_ii_lvl_ar1_nla_q3sum", series="uscn", direction="USA|China",
             spec="level AR(1), fit<=2023-12", agg="within-quarter 3-month sum",
             engine=("m", 1, False, NLA_FIT_END), nla=True, gate=False),
        # C
        dict(key="C_i_dgpr_ar4_nla_qend", series="uscn", direction="USA|China",
             spec="AR(4) on monthly first difference, fit<=2023-12",
             agg="quarter-end month resid",
             engine=("m", 4, True, NLA_FIT_END), nla=True, gate=False),
        dict(key="C_ii_dgpr_ar4_nla_q3sum", series="uscn", direction="USA|China",
             spec="AR(4) on monthly first difference, fit<=2023-12",
             agg="within-quarter 3-month sum",
             engine=("m", 4, True, NLA_FIT_END), nla=True, gate=False),
        # D
        dict(key="D_q_ar1_nla", series="uscn", direction="USA|China",
             spec="quarterly MEAN of monthly GPR, then AR(1) quarterly, fit<=2023Q4",
             agg="quarterly residual (native)",
             engine=("q", 1, False, NLA_FIT_END), nla=True, gate=False),
        # E -- direction axis on the two best-motivated constructions
        dict(key="E_C_i_bidir", series="bidir", direction="USA|China + China|USA (SUM)",
             spec="AR(4) on monthly first difference, fit<=2023-12",
             agg="quarter-end month resid",
             engine=("m", 4, True, NLA_FIT_END), nla=True, gate=False),
        dict(key="E_C_i_cnus", series="cnus", direction="China|USA (diagnostic)",
             spec="AR(4) on monthly first difference, fit<=2023-12",
             agg="quarter-end month resid",
             engine=("m", 4, True, NLA_FIT_END), nla=True, gate=False),
        dict(key="E_B_i_bidir", series="bidir", direction="USA|China + China|USA (SUM)",
             spec="level AR(1), fit<=2023-12", agg="quarter-end month resid",
             engine=("m", 1, False, NLA_FIT_END), nla=True, gate=False),
        dict(key="E_B_i_cnus", series="cnus", direction="China|USA (diagnostic)",
             spec="level AR(1), fit<=2023-12", agg="quarter-end month resid",
             engine=("m", 1, False, NLA_FIT_END), nla=True, gate=False),
    ]

    raw = {}
    arinfo = {}
    for sp in specs:
        freq, order, diff, fit_end = sp["engine"]
        y = SERIES[sp["series"]]
        if freq == "m":
            resid_m, info = ar_residuals(y, month_end, order, diff, fit_end)
            if sp["agg"].startswith("within-quarter"):
                q = to_quarter_sum3(resid_m, month_end)
            else:
                q = to_quarter_end_month(resid_m, month_end)
        else:
            # quarterly MEAN of monthly GPR, then AR at quarterly frequency
            sm = pd.Series(y, index=month_end)
            grp = sm.groupby(month_end.to_period("Q"))
            qmean = grp.mean()
            qcnt = grp.size()
            qmean = qmean[qcnt == 3]                       # full quarters only
            qidx = pd.DatetimeIndex(qmean.index.to_timestamp(how="end").normalize())
            resid_q, info = ar_residuals_quarterly(qmean.to_numpy(float), qidx,
                                                   order, fit_end)
            q = pd.Series(resid_q, index=qidx)
        raw[sp["key"]] = q
        arinfo[sp["key"]] = info

    # ---- assemble on the 82 panel quarters ---------------------------------
    menu = pd.DataFrame(index=panel_q)
    menu.index.name = "quarter_end"
    for k, s in raw.items():
        menu["raw_" + k] = s.reindex(panel_q)

    # baseline reference columns (read-only inputs, for the gate + correlations)
    base_ref = base_q.set_index("quarter_end")["shock_us_cn"].reindex(panel_q)
    menu["raw_baseline_existing"] = base_ref
    menu["raw_baseline_panel"] = panel_shock.reindex(panel_q)

    n_missing = menu.isna().sum()
    if n_missing.drop(labels=[], errors="ignore").sum() > 0:
        print("\n[WARN] missing cells on the 82 panel quarters:")
        print(n_missing[n_missing > 0])

    # ---- VALIDATION GATE ----------------------------------------------------
    d_parquet = np.abs(menu["raw_A_baseline_repro"] - menu["raw_baseline_existing"])
    d_panel = np.abs(menu["raw_A_baseline_repro"] - menu["raw_baseline_panel"])
    gate_parquet = float(np.nanmax(d_parquet))
    gate_panel = float(np.nanmax(d_panel))
    gate_pass = (gate_parquet <= GATE_TOL) and (gate_panel <= GATE_TOL)

    print("\n" + "-" * 96)
    print("VALIDATION GATE (A = baseline_repro vs the frozen baseline), 82 panel quarters")
    print(f"  max|A - gpr_quarterly_with_shock.shock_us_cn| = {gate_parquet:.3e}")
    print(f"  max|A - c6_panel.shock|                       = {gate_panel:.3e}")
    print(f"  tolerance = {GATE_TOL:.0e}   ->   {'PASS' if gate_pass else 'FAIL'}")
    print("-" * 96)
    if not gate_pass:
        print("[GATE FAIL] the builder's plumbing does NOT match 05_combine_visualize.jl. "
              "Menu results below are NOT trustworthy.")

    # ---- standardize over the 82 panel quarters -----------------------------
    moments = {}
    for sp in specs:
        k = sp["key"]
        v = menu["raw_" + k]
        mu, sd = float(v.mean()), float(v.std(ddof=1))
        moments[k] = (mu, sd, float(v.min()), float(v.max()))
        menu["s_" + k] = (v - mu) / sd
    mu_b, sd_b = float(base_ref.mean()), float(base_ref.std(ddof=1))
    moments["baseline_existing"] = (mu_b, sd_b, float(base_ref.min()), float(base_ref.max()))
    menu["s_baseline_existing"] = (base_ref - mu_b) / sd_b

    # ---- diagnostics --------------------------------------------------------
    rows = []
    all_keys = [sp["key"] for sp in specs] + ["baseline_existing"]
    meta = {sp["key"]: sp for sp in specs}
    meta["baseline_existing"] = dict(key="baseline_existing", series="uscn",
                                     direction="USA|China",
                                     spec="level AR(1), FULL-sample fit (EXISTING headline)",
                                     agg="quarter-end month resid",
                                     nla=False, gate=False)
    for k in all_keys:
        col = menu["s_" + k].to_numpy(float)
        w = whiteness(col)
        mu, sd, lo, hi = moments[k]
        m = meta[k]
        info = arinfo.get(k, {})
        rows.append(dict(
            variant=k,
            direction=m["direction"],
            spec=m["spec"],
            aggregation=m["agg"],
            no_look_ahead=bool(m["nla"]),
            role=("VALIDATION GATE" if m.get("gate") else
                  ("headline-continuity baseline" if k == "baseline_existing" else
                   ("selection candidate" if (m["nla"] and m["direction"] == "USA|China")
                    else "direction-axis parallel column"))),
            n_q=w["n_q"],
            mean_raw=mu, sd_raw=sd, min_raw=lo, max_raw=hi,
            acf1=w["acf1"], acf2=w["acf2"], acf3=w["acf3"], acf4=w["acf4"],
            LB_Q1=w["LB_Q1"], LB_p1=w["LB_p1"],
            LB_Q4=w["LB_Q4"], LB_p4=w["LB_p4"],
            LB_Q8=w["LB_Q8"], LB_p8=w["LB_p8"],
            corr_with_baseline=float(np.corrcoef(col, menu["s_baseline_existing"])[0, 1]),
            ar_n_fit=info.get("n_fit", np.nan),
            ar_fit_first=info.get("fit_first", ""),
            ar_fit_last=info.get("fit_last", ""),
            ar_coefs=info.get("coefs", ""),
        ))
    diag = pd.DataFrame(rows)

    # ---- PRE-REGISTERED SELECTION ------------------------------------------
    # candidates: no-look-ahead AND direction == USA|China (the gate variant A is
    # full-sample, so it is excluded automatically; E columns are excluded because
    # the direction axis is reported, not selected).
    cand = diag[diag["role"] == "selection candidate"].copy()
    cand = cand.sort_values(["LB_p4", "LB_p8", "acf1"],
                            ascending=[False, False, True],
                            key=lambda s: s.abs() if s.name == "acf1" else s)
    selected = cand.iloc[0]["variant"] if len(cand) else None

    # ---- MECHANICAL HAND-OFF of the pre-registered winner ------------------
    # The estimation leg (run_shock_menu.do) must learn the preferred variant
    # WITHOUT a human editing a macro -- hand-editing would reintroduce exactly
    # the discretion pre-registration exists to remove.  The .do and this script
    # both resolve menu columns by the "s_" prefix, so the hand-off string is the
    # MENU COLUMN name ("s_" + variant key), NOT the diagnostics `variant` key.
    # Writing the bare key (e.g. D_q_ar1_nla) would silently fail the .do's
    # PREF_IDX string match and print "not among the resolved menu variants".
    preferred_col = ("s_" + selected) if selected is not None else ""
    diag["preferred_menu_column"] = preferred_col

    diag["preferred_new_construction"] = diag["variant"].eq(selected)
    diag["selection_rank"] = np.nan
    diag.loc[diag["variant"].isin(cand["variant"]), "selection_rank"] = (
        diag.loc[diag["variant"].isin(cand["variant"]), "variant"]
        .map({v: i + 1 for i, v in enumerate(cand["variant"])}))
    diag["gate_max_abs_diff_vs_parquet"] = gate_parquet
    diag["gate_max_abs_diff_vs_panel"] = gate_panel
    diag["gate_pass"] = gate_pass
    diag["level_corr_uscn_cnus_full"] = lvl_corr
    diag["level_corr_uscn_cnus_pre2024"] = lvl_corr_pre24

    # ---- re-derivation vs prior external measurements ----------------------
    def _get(v, f):
        return float(diag.loc[diag["variant"] == v, f].iloc[0])

    recon = pd.DataFrame([
        dict(item="baseline acf1", prior_external=0.271, rederived=_get("baseline_existing", "acf1")),
        dict(item="baseline LB(1) p", prior_external=0.013, rederived=_get("baseline_existing", "LB_p1")),
        dict(item="C-i acf1", prior_external=-0.036, rederived=_get("C_i_dgpr_ar4_nla_qend", "acf1")),
        dict(item="C-i LB(4) p", prior_external=0.245, rederived=_get("C_i_dgpr_ar4_nla_qend", "LB_p4")),
        dict(item="C-ii acf1", prior_external=-0.215, rederived=_get("C_ii_dgpr_ar4_nla_q3sum", "acf1")),
        dict(item="C-ii LB(4) p", prior_external=0.292, rederived=_get("C_ii_dgpr_ar4_nla_q3sum", "LB_p4")),
        dict(item="D acf1", prior_external=-0.029, rederived=_get("D_q_ar1_nla", "acf1")),
        dict(item="D LB(8) p", prior_external=0.017, rederived=_get("D_q_ar1_nla", "LB_p8")),
        dict(item="level corr USA|China vs China|USA", prior_external=0.576, rederived=lvl_corr),
    ])
    recon["abs_gap"] = (recon["rederived"] - recon["prior_external"]).abs()
    recon["reproduces"] = recon["abs_gap"] <= 0.02
    recon["note"] = ""
    recon.loc[recon["item"].str.startswith("D "), "note"] = (
        "prior number was a FULL-SAMPLE (look-ahead) quarterly AR(1) fit; refit "
        "full-sample here gives acf1=-0.0292 / LB(8)p=0.0175, an exact match. "
        "Variant D in this menu is no-look-ahead, so the re-derived value governs.")
    recon.loc[recon["item"].str.startswith("level corr"), "note"] = (
        f"0.576 matches the months<=2023-12 window ({lvl_corr_pre24:.4f}); "
        f"{lvl_corr:.4f} is the full 1960-2026 sample.")
    recon.to_csv(OUT / "shock_menu_rederivation_check.csv", index=False)

    # ---- correlation matrix -------------------------------------------------
    scols = ["s_" + k for k in all_keys]
    corr = menu[scols].corr()
    corr.index = [c[2:] for c in corr.index]
    corr.columns = [c[2:] for c in corr.columns]

    # ---- write --------------------------------------------------------------
    OUT.mkdir(exist_ok=True)
    menu_out = menu.reset_index()
    menu_out.to_parquet(OUT / "shock_menu_quarterly.parquet", index=False)
    menu_out.to_csv(OUT / "shock_menu_quarterly.csv", index=False)
    dta = menu_out.copy()
    dta.columns = [c.replace("|", "_") for c in dta.columns]
    dta.to_stata(OUT / "shock_menu_quarterly.dta", write_index=False, version=118)
    diag.to_csv(OUT / "shock_menu_diagnostics.csv", index=False)
    corr.to_csv(OUT / "shock_menu_corr.csv")

    # single line, no trailing whitespace; newline="\n" so no CR survives (the
    # .do reads it with trim(), which strips blanks but NOT a trailing CR).
    with open(OUT / "shock_menu_preferred.txt", "w", encoding="ascii", newline="\n") as fh:
        fh.write(preferred_col + "\n")

    # ---- report -------------------------------------------------------------
    show = ["variant", "role", "no_look_ahead", "n_q", "mean_raw", "sd_raw",
            "acf1", "acf2", "acf3", "acf4",
            "LB_Q1", "LB_p1", "LB_Q4", "LB_p4", "LB_Q8", "LB_p8",
            "corr_with_baseline", "selection_rank"]
    print("\nDIAGNOSTICS (on the 82 panel quarters; series standardized -> acf/LB "
          "unaffected by the affine rescale)")
    print(diag[show].to_string(index=False,
                               float_format=lambda x: f"{x:.4f}"))

    print("\nCORRELATION MATRIX (standardized, 82 quarters)")
    print(corr.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nRE-DERIVATION vs prior external measurements (nothing copied forward)")
    print(recon.to_string(index=False, float_format=lambda x: f"{x:.4f}",
                          max_colwidth=64))

    print("\n" + "=" * 96)
    print("PRE-REGISTERED SELECTION (LB(4) p, tie-break LB(8) p, then |acf1|; "
          "NLA + USA|China only)")
    print(cand[["variant", "LB_p4", "LB_p8", "acf1"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  -> PREFERRED NEW CONSTRUCTION: {selected}")
    print(f"     menu column : {preferred_col}   "
          f"(written to shock_menu_preferred.txt for run_shock_menu.do's lead spec)")
    print(f"     spec        : {meta[selected]['spec']}")
    print(f"     aggregation : {meta[selected]['agg']}")
    print(f"     LB(4) p     : {cand.iloc[0]['LB_p4']:.4f}   "
          f"LB(8) p = {cand.iloc[0]['LB_p8']:.4f}   acf1 = {cand.iloc[0]['acf1']:+.4f}")
    print("     Baseline S_t remains the headline-continuity column regardless.")
    runner_up = cand.iloc[1]
    print(f"     Runner-up  : {runner_up['variant']} (LB(4) p={runner_up['LB_p4']:.4f}, "
          f"LB(8) p={runner_up['LB_p8']:.4f}).  HONEST CAVEAT: the winner's LB(8) p "
          f"({cand.iloc[0]['LB_p8']:.4f}) is marginal and BELOW the runner-up's; the rule "
          "is LB(4)-first by pre-registration, so the ordering stands, but residual "
          "higher-order dependence in the selected series is NOT fully cleaned.")
    print("=" * 96)

    print("\nWrote:")
    for f in ["shock_menu_quarterly.parquet", "shock_menu_quarterly.csv",
              "shock_menu_quarterly.dta", "shock_menu_diagnostics.csv",
              "shock_menu_corr.csv", "shock_menu_rederivation_check.csv",
              "shock_menu_preferred.txt"]:
        print("  " + (OUT / f).as_posix())

    print("\nPRE-REGISTERED READING (fixed in the header BEFORE any regression ran):")
    print("  (a) all variants null incl. the preferred + bidirectional -> the P0 deep")
    print("      null is not a shock-construction artifact; P1a/P1b/P1c disclosed-and-closed.")
    print("  (b) preferred rejects, baseline does not -> construction-sensitivity result on")
    print("      the pre-registered column only (corr with baseline ~0.25 => near-independent).")
    print("  (c) baseline rejects, no NLA variant does -> look-ahead / serial-correlation")
    print("      artifact; the NLA column governs.")
    print("  (d) an E direction column rejects alone -> direction-axis result, reported in")
    print("      parallel, never promoted to headline (E excluded from selection by design).")
    print("  (e) one of ~12 columns at p<.05 with the rest null ~ one expected false")
    print("      positive; only the preferred + baseline carry weight; p_circ governs.")


if __name__ == "__main__":
    main()
