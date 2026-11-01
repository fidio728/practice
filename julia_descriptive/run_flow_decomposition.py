# ============================================================================
# PRE-P0 VINTAGE WARNING (P0 holdings-snapshot rebuild, 2026-08-04)
# This script's output is PRE-P0: it was built from the exact-EOM
# holdings_eom.parquet (or from merged_us_eu_zero_filled.parquet built from
# it). The as-of quarter-end selection rule in 03_eom_etl.jl CHANGED the
# panel's fund universe on EVERY quarter. PRE-P0, pending re-run, do not mix
# with post-P0 results. Register: julia_descriptive/VINTAGE_P0.md
# ============================================================================

"""
run_flow_decomposition.py — Essay 2 §7.6: accounting decomposition of observed
institutional and residual ownership FLOWS (design agenda rank3 v2).

POSITIONING (locked, do not change):
  Explanatory *diagnostic*, NOT an identification threat. FLOW ONLY — the within-
  group normalization of the main w-spec has no additive identity, so this
  decomposition does not apply to the headline w regression.

THE IDENTITY (exact, by construction; asserted to 1e-9 relative):

    flow_US + flow_NONUS + flow_R  ==  (out_t - out_{t-1}) / out_{t-1}  ==  float_growth

  with, for every group g in {US, NONUS, R},
        flow_{i,g,t} = ( held_{i,g,t} - held_{i,g,t-1} ) / out_{i,t-1}
  and   held_R = out - held_US - held_NONUS   (residual sector: retail + insiders
        + unreported institutions + strategic stakes; in prose it is the RESIDUAL
        SECTOR, never "retail"). Denominator fixed at the LAGGED float out_{t-1}
        for ALL FOUR terms so a same-quarter buyback/issuance cannot mechanically
        move a flow (mirrors the F6 fix in build_ownership_share_c6_panel.py).

BAD firm-quarter rule (MIRRORED from build_ownership_share_c6_panel.py, mandatory):
  a firm-quarter is BAD if MAX(os_raw) > 1 OR SUM(os_raw) > 1 (os_raw = held/out_t
  on the CURRENT float). A BAD firm-quarter nulls held for BOTH groups at the
  current AND (via the lag) the next quarter. This stops a corrupted float from
  leaking and guarantees held_R >= 0 (SUM os <= 1). Not mirroring it lets a dirty
  float leak in and held_R can go negative.

WHAT THIS SCRIPT DOES (steps 1-3 of the four-step agenda):
  (1) firm-quarter panel flow_US/flow_NONUS/flow_R/float_growth + identity assert
      -> output/flow_decomposition_panel.parquet
  (2) diagnostic table: 3 samples (raw / winsor p1p99 / stable-float out_t==out_{t-1})
      x 2 populations (full / high-CN tercile, cn_lag>0 top tercile) x
      { corr(flow_US,flow_NONUS), corr(flow_US,flow_R),
        slope(flow_NONUS~flow_US), slope(flow_R~flow_US) }
      -> output/flow_decomposition_diag.csv (also printed).
      ANCHOR: high-CN tercile corr(flow_US,flow_NONUS) ~ +0.20 (winsor/stable +0.20~0.23);
      large deviation -> printed WARNING for manual check.
  (3) design-based RI: outcomes flow_R, flow_common=flow_US+flow_NONUS,
      flow_diff=flow_US-flow_NONUS (reference), each on [cn_lag, cn_lag x S] under
      firm+quarter two-way demean (CN x S is firm-quarter level -> fq FE would
      absorb it entirely, so this is the WEAK firm-FE + quarter-FE descriptive
      boundary design). beta_3 = coef on cn_lag x S; permute the quarter shock
      5000x (only the interaction moves; cn_lag is a fixed firm-quarter attribute).
      RI main table uses WINSORIZED flow (kurtosis~5000 fat-tail lesson); raw
      reported as companion columns. -> output/ri_flow_decomposition.csv

  Step 4 (§7.6 estimand rewrite + world-B boundary) is a WRITING task; this script
  only prints the world-B boundary statement so it is not lost.

RI machinery mirrors run_ri_direction.py (bincount two-way demean, N_PERM=5000,
SEED=20260702). Because [cn_lag, cn_lag x S] are correlated, every permutation
re-solves the full 2x2 normal equations (never single-variable FWL).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import duckdb

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
GRID = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
OBS = (OUT / "ownership_share_observed.parquet").as_posix()
FLOAT = (OUT / "ownership_share_float.parquet").as_posix()

PANEL_OUT = OUT / "flow_decomposition_panel.parquet"
DIAG_OUT = OUT / "flow_decomposition_diag.csv"
RI_OUT = OUT / "ri_flow_decomposition.csv"

N_PERM = 5000
DEMEAN_ITERS = 30
SEED = 20260702
WINSOR_P = 0.01
IDENTITY_RTOL = 1e-9
ANCHOR = 0.20
ANCHOR_TOL = 0.12  # |corr - 0.20| beyond this on winsor/stable high-CN -> WARNING


# ----------------------------------------------------------------------------
# STEP 1 — firm-quarter flow panel (BAD rule mirrored, denominator = lagged float)
# ----------------------------------------------------------------------------
def build_panel():
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    # single-thread: parallel hash-aggregation sums floats in nondeterministic
    # order, so held_* differ in the last bits across runs; that input noise
    # made ri_p jitter at the third decimal run-to-run (b3 order-invariant).
    con.execute("SET threads=1")

    print("[1/6] Merge held + primary-EQ float onto the C6 firm-group-quarter grid...")
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE joined AS
    WITH grid AS (
        SELECT sec_entity_id, holder_group, CAST(report_date AS DATE) AS rd
        FROM read_parquet('{GRID}')
    ),
    held AS (
        SELECT sec_entity_id, hgroup AS holder_group,
               CAST(report_date AS DATE) AS rd, shares_held
        FROM read_parquet('{OBS}')
    ),
    flt AS (
        SELECT sec_entity_id, CAST(report_date AS DATE) AS rd, shares_out
        FROM read_parquet('{FLOAT}')
    )
    SELECT g.sec_entity_id, g.holder_group, g.rd,
           f.shares_out,
           CASE WHEN f.shares_out IS NULL THEN NULL
                ELSE COALESCE(h.shares_held, 0) END AS held,
           CASE WHEN f.shares_out IS NULL THEN NULL
                ELSE COALESCE(h.shares_held, 0) / f.shares_out END AS os_raw
    FROM grid g
    LEFT JOIN held h USING (sec_entity_id, holder_group, rd)
    LEFT JOIN flt  f USING (sec_entity_id, rd)
    """)

    print("[2/6] Flag BAD firm-quarters (MAX os>1 OR SUM os>1); null held for BOTH groups...")
    con.execute("""
    CREATE OR REPLACE TEMP TABLE clean AS
    WITH bad AS (
        SELECT sec_entity_id, rd
        FROM joined GROUP BY 1, 2
        HAVING MAX(os_raw) > 1 OR SUM(os_raw) > 1
    )
    SELECT j.sec_entity_id, j.holder_group, j.rd, j.shares_out,
           CASE WHEN b.sec_entity_id IS NOT NULL THEN NULL ELSE j.held END AS held
    FROM joined j
    LEFT JOIN bad b USING (sec_entity_id, rd)
    """)

    print("[3/6] Pivot to firm-quarter wide; held_R = out - held_US - held_NONUS...")
    con.execute("""
    CREATE OR REPLACE TEMP TABLE fq AS
    SELECT sec_entity_id, rd,
           MAX(shares_out)                                          AS out,
           MAX(CASE WHEN holder_group = 'US'    THEN held END)      AS held_us,
           MAX(CASE WHEN holder_group = 'NONUS' THEN held END)      AS held_nonus
    FROM clean GROUP BY 1, 2
    """)

    print("[4/6] Backward lags within firm; flows on FIXED lagged float out_{t-1}...")
    df = con.execute("""
    SELECT
        CAST(sec_entity_id AS VARCHAR)                    AS firm_str,
        CAST(rd AS TIMESTAMP)                             AS rdate,
        out, held_us, held_nonus,
        out - held_us - held_nonus                        AS held_r,
        LAG(out)                        OVER w            AS out_lag,
        LAG(held_us)                    OVER w            AS held_us_lag,
        LAG(held_nonus)                 OVER w            AS held_nonus_lag,
        LAG(out - held_us - held_nonus) OVER w            AS held_r_lag
    FROM fq
    WINDOW w AS (PARTITION BY sec_entity_id ORDER BY rd)
    """).df()

    # cn_lag (firm-quarter) and shock (quarter) from the B7-fixed merged grid
    cn = con.execute(f"""
        SELECT sec_entity_id AS firm_str, CAST(report_date AS TIMESTAMP) AS rdate,
               MIN(china_share_lag1q) AS cn_min, MAX(china_share_lag1q) AS cn_max
        FROM read_parquet('{GRID}') GROUP BY 1, 2
    """).df()
    shk = con.execute(f"""
        SELECT CAST(report_date AS TIMESTAMP) AS rdate,
               MIN(shock_us_cn) AS s_min, MAX(shock_us_cn) AS s_max
        FROM read_parquet('{GRID}') GROUP BY 1
    """).df()
    con.close()

    # guards: cn_lag must be firm-quarter invariant; shock must be quarter invariant
    bad_cn = cn.dropna(subset=["cn_min", "cn_max"])
    assert np.allclose(bad_cn["cn_min"], bad_cn["cn_max"], equal_nan=True), \
        "china_share_lag1q varies within a firm-quarter"
    bad_s = shk.dropna(subset=["s_min", "s_max"])
    assert np.allclose(bad_s["s_min"], bad_s["s_max"], equal_nan=True), \
        "shock_us_cn varies within a quarter"
    cn = cn.assign(cn_lag=cn["cn_min"])[["firm_str", "rdate", "cn_lag"]]
    shk = shk.assign(shock=shk["s_min"])[["rdate", "shock"]]

    df["firm_str"] = df["firm_str"].astype(str)
    df["rdate"] = pd.to_datetime(df["rdate"])
    cn["firm_str"] = cn["firm_str"].astype(str)
    cn["rdate"] = pd.to_datetime(cn["rdate"])
    shk["rdate"] = pd.to_datetime(shk["rdate"])
    df = df.merge(cn, on=["firm_str", "rdate"], how="left").merge(shk, on="rdate", how="left")

    print("[5/6] Compute flows + float_growth; assert the exact identity...")
    out_lag = df["out_lag"].replace(0.0, np.nan)  # float>0; guard anyway
    df["flow_us"] = (df["held_us"] - df["held_us_lag"]) / out_lag
    df["flow_nonus"] = (df["held_nonus"] - df["held_nonus_lag"]) / out_lag
    df["flow_r"] = (df["held_r"] - df["held_r_lag"]) / out_lag
    df["float_growth"] = (df["out"] - df["out_lag"]) / out_lag
    df["is_stable"] = (df["out"] == df["out_lag"])  # stable-float: out_t == out_{t-1}

    # identity assert on rows where all four flow terms are present
    m = df[["flow_us", "flow_nonus", "flow_r", "float_growth"]].notna().all(axis=1)
    lhs = df.loc[m, "flow_us"] + df.loc[m, "flow_nonus"] + df.loc[m, "flow_r"]
    rhs = df.loc[m, "float_growth"]
    denom = np.maximum(1.0, rhs.abs())
    max_rel = float(((lhs - rhs).abs() / denom).max())
    assert max_rel <= IDENTITY_RTOL, (
        f"decomposition identity broken: max relative residual {max_rel:.3e} > {IDENTITY_RTOL}")
    print(f"      identity OK: max relative residual = {max_rel:.3e} on {int(m.sum()):,} firm-quarters")

    print("[6/6] Write firm-quarter panel...")
    keep = ["firm_str", "rdate", "out", "held_us", "held_nonus", "held_r",
            "flow_us", "flow_nonus", "flow_r", "float_growth", "is_stable",
            "cn_lag", "shock"]
    df[keep].to_parquet(PANEL_OUT, index=False)
    print(f"      wrote {PANEL_OUT.name}  ({len(df):,} rows, "
          f"{int(m.sum()):,} with full flow set)")

    # .dta companion for run_flow_decomp_step3.do (CRVE cross-check). Column
    # contract per the .do: flow_R / flow_common / flow_diff (Stata names are
    # case-sensitive), materialized here so producer == consumer.
    dta = df[m][["firm_str", "rdate", "flow_us", "flow_nonus", "flow_r",
                 "float_growth", "cn_lag", "shock"]].copy()
    dta["flow_R"] = dta.pop("flow_r")
    dta["flow_common"] = dta["flow_us"] + dta["flow_nonus"]
    dta["flow_diff"] = dta["flow_us"] - dta["flow_nonus"]
    # Ship the winsorized outcomes too: cutoffs on the ESTIMATION sample (rows
    # with cn_lag & shock non-missing == the RI panel) via the same
    # _winsor_cutoffs used by the RI step. The .do must USE these columns, not
    # recompute with _pctile — np.quantile vs _pctile definitional drift plus
    # the full-panel-vs-estimation-sample base broke the CRVE/RI cross-check.
    est = dta[["cn_lag", "shock"]].notna().all(axis=1)
    for c in ["flow_R", "flow_common", "flow_diff"]:
        lo, hi = _winsor_cutoffs(dta.loc[est, c])
        dta[c + "_w"] = dta[c].clip(lo, hi)
    dta["rdate"] = pd.to_datetime(dta["rdate"])
    dta_path = PANEL_OUT.with_suffix(".dta")
    dta.to_stata(dta_path, write_index=False, convert_dates={"rdate": "tc"},
                 version=118)
    print(f"      wrote {dta_path.name}  ({len(dta):,} complete firm-quarter rows)")
    return df


# ----------------------------------------------------------------------------
# STEP 2 — diagnostic table (3 samples x 2 populations x 4 statistics)
# ----------------------------------------------------------------------------
def _pair(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]

def _corr(a, b):
    a, b = _pair(a, b)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def _slope(y, x):  # slope of y ~ x (univariate OLS): cov(x,y)/var(x)
    x, y = _pair(x, y)
    if len(x) < 3:
        return np.nan
    xc = x - x.mean()
    ssx = float((xc * xc).sum())
    if ssx == 0:
        return np.nan
    return float((xc * (y - y.mean())).sum() / ssx)

def _winsor_cutoffs(s, p=WINSOR_P):
    s = pd.Series(s).dropna()
    return float(s.quantile(p)), float(s.quantile(1 - p))

def diagnostics(df):
    print("\n===== STEP 2: flow-decomposition diagnostics =====")
    # estimation base: all three flows present
    base = df[df[["flow_us", "flow_nonus", "flow_r"]].notna().all(axis=1)].copy()

    # winsor cutoffs from the FULL raw base (global, applied to every population)
    cuts = {c: _winsor_cutoffs(base[c]) for c in ["flow_us", "flow_nonus", "flow_r"]}
    win = base.copy()
    for c in ["flow_us", "flow_nonus", "flow_r"]:
        lo, hi = cuts[c]
        win[c] = win[c].clip(lo, hi)

    # high-CN tercile: among cn_lag>0, top tercile of cn_lag (fixed cutoff from base)
    pos = base.loc[base["cn_lag"] > 0, "cn_lag"]
    cn_cut = float(pos.quantile(2.0 / 3.0)) if len(pos) else np.nan
    print(f"  high-CN tercile cutoff (cn_lag>0, 2/3 quantile) = {cn_cut:.4f}"
          f"   (n cn_lag>0 = {len(pos):,})")

    def hi_mask(frame):
        return (frame["cn_lag"] > 0) & (frame["cn_lag"] >= cn_cut)

    samples = {
        "raw": base,
        "winsor_p1p99": win,
        "stable_float": base[base["is_stable"]].copy(),
    }
    rows = []
    anchor_vals = {}
    for sname, frame in samples.items():
        pops = {"full": frame, "high_cn_tercile": frame[hi_mask(frame)]}
        for pname, sub in pops.items():
            n = int(sub[["flow_us", "flow_nonus", "flow_r"]].notna().all(axis=1).sum())
            stats = {
                "corr_us_nonus": _corr(sub["flow_us"], sub["flow_nonus"]),
                "corr_us_r": _corr(sub["flow_us"], sub["flow_r"]),
                "slope_nonus_us": _slope(sub["flow_nonus"], sub["flow_us"]),
                "slope_r_us": _slope(sub["flow_r"], sub["flow_us"]),
            }
            for stat, val in stats.items():
                rows.append({"sample": sname, "population": pname,
                             "statistic": stat, "value": val, "n": n})
            if pname == "high_cn_tercile" and stats["corr_us_nonus"] == stats["corr_us_nonus"]:
                anchor_vals[sname] = stats["corr_us_nonus"]

    diag = pd.DataFrame(rows)
    diag.to_csv(DIAG_OUT, index=False)

    # pretty print
    piv = diag.pivot_table(index=["sample", "population", "n"],
                           columns="statistic", values="value")
    piv = piv[["corr_us_nonus", "corr_us_r", "slope_nonus_us", "slope_r_us"]]
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(piv.to_string())
    print(f"  wrote {DIAG_OUT.name}")

    # anchor check on winsor / stable-float high-CN corr(flow_US, flow_NONUS) ~ +0.20
    print(f"\n  ANCHOR CHECK (high-CN tercile corr(flow_US, flow_NONUS), expect ~+{ANCHOR:.2f}):")
    for sname in ("winsor_p1p99", "stable_float"):
        v = anchor_vals.get(sname, np.nan)
        flag = "OK" if (v == v and abs(v - ANCHOR) <= ANCHOR_TOL) else "WARNING"
        print(f"    {sname:14s}: {v:+.4f}   [{flag}]")
        if flag == "WARNING":
            print(f"    *** WARNING: {sname} high-CN corr {v:+.4f} deviates from anchor "
                  f"+{ANCHOR:.2f} by > {ANCHOR_TOL:.2f} — manual check required. ***")
    return diag


# ----------------------------------------------------------------------------
# STEP 3 — design-based randomization inference (weak firm-FE + quarter-FE)
# ----------------------------------------------------------------------------
def make_demeaner(firm_c, q_c, iters):
    nf, nqq = firm_c.max() + 1, q_c.max() + 1
    fcount = np.bincount(firm_c, minlength=nf).astype(float)
    qcount = np.bincount(q_c, minlength=nqq).astype(float)

    def demean(x):
        x = x.copy()
        for _ in range(iters):
            x -= (np.bincount(firm_c, weights=x, minlength=nf) / fcount)[firm_c]
            x -= (np.bincount(q_c, weights=x, minlength=nqq) / qcount)[q_c]
        return x
    return demean

def ri_flow(df):
    print("\n===== STEP 3: design-based RI (firm-FE + quarter-FE weak design) =====")
    d = df[df[["flow_us", "flow_nonus", "flow_r", "cn_lag", "shock"]].notna().all(axis=1)].copy()
    # canonical row order BEFORE factorize: df arrives in duckdb scan order,
    # which is nondeterministic across runs; factorize codes then permute, so
    # the same seed maps permutations onto different quarter assignments and
    # ri_p jitters at the third decimal (b3_obs is order-invariant). Sorting
    # pins the permutation stream and makes ri_p exactly reproducible.
    d = d.sort_values(["firm_str", "rdate"]).reset_index(drop=True)
    d["flow_common"] = d["flow_us"] + d["flow_nonus"]
    d["flow_diff"] = d["flow_us"] - d["flow_nonus"]

    firm_c = pd.factorize(d["firm_str"])[0]
    qcode = pd.factorize(d["rdate"])[0]
    nq = int(qcode.max()) + 1
    n_fq = len(d)
    Svec = np.zeros(nq)
    Svec[qcode] = d["shock"].to_numpy(float)
    if d.groupby(qcode)["shock"].nunique().max() != 1:
        raise SystemExit("shock not constant within quarter")
    print(f"  RI panel: {n_fq:,d} firm-quarters, {nq} quarters, "
          f"{pd.Series(firm_c).nunique():,d} firms")

    demean = make_demeaner(firm_c, qcode, DEMEAN_ITERS)

    cn_raw = d["cn_lag"].to_numpy(float)
    cn_dm = demean(cn_raw)  # fixed main-effect regressor across permutations

    # --- EXACT linear reformulation (identical result, ~1000x faster) ---------
    # The two-way demean M is LINEAR and the interaction is x = cn ⊙ S[quarter].
    # Since S[quarter] is a per-quarter constant, x = Σ_q S_q · u_q with
    # u_q[i] = cn_i·1{quarter(i)=q}. Therefore x_dm = M(x) = V @ S where the
    # column V[:,q] = M(u_q) is demeaned ONCE. Each permutation is then just a
    # tiny nq-vector form, not a full iterative demean over n_fq rows.
    V = np.empty((n_fq, nq), dtype=float)
    for q in range(nq):
        u = np.where(qcode == q, cn_raw, 0.0)
        V[:, q] = demean(u)
    G = V.T @ V                      # nq x nq, precomputed once
    g = V.T @ cn_dm                  # nq vector: <cn_dm, x_dm> = g·S
    a = float(cn_dm @ cn_dm)         # <cn_dm, cn_dm> (constant)

    def make_beta3(y_dm):
        # β₃ = coef on x_dm in y_dm ~ [cn_dm, x_dm]; closed-form 2x2 solve.
        p = float(cn_dm @ y_dm)      # <cn_dm, y_dm> (constant)
        h = V.T @ y_dm               # <y_dm, x_dm> = h·S
        def beta3(S):
            c = float(g @ S)         # <cn_dm, x_dm>
            dd = float(S @ G @ S)    # <x_dm, x_dm>
            qy = float(h @ S)        # <y_dm, x_dm>
            det = a * dd - c * c
            if abs(det) < 1e-300:
                # exact singularity: fall back to lstsq on the tiny system
                X = np.column_stack([cn_dm, V @ S])
                return float(np.linalg.lstsq(X, y_dm, rcond=None)[0][1])
            return (a * qy - c * p) / det
        return beta3

    # conditioning check at the observed shock (mirrors run_ri_direction warning)
    _c = float(g @ Svec); _d = float(Svec @ G @ Svec)
    _XtX = np.array([[a, _c], [_c, _d]])
    _cond = np.linalg.cond(_XtX)
    print(f"  cond(XtX) at observed shock = {_cond:.2e}")
    if _cond > 1e8:
        print("  WARNING: XtX ill-conditioned (>1e8) — beta_3 near-collinear "
              "(cn_lag vs cn_lag x S); interpret with care")

    outcomes = ["flow_r", "flow_common", "flow_diff"]
    rows = []
    for oc in outcomes:
        for tag in ("winsor", "raw"):
            y = d[oc].to_numpy(float)
            if tag == "winsor":
                lo, hi = _winsor_cutoffs(y)
                y = np.clip(y, lo, hi)
            y_dm = demean(y)
            beta3 = make_beta3(y_dm)
            b_obs = beta3(Svec)
            rng = np.random.default_rng(SEED)
            cnt = 0
            for _ in range(N_PERM):
                Sp = rng.permutation(Svec)
                if abs(beta3(Sp)) >= abs(b_obs) - 1e-300:
                    cnt += 1
            ri_p = (cnt + 1) / (N_PERM + 1)
            rows.append({"outcome": oc, "flow_form": tag,
                         "b3": b_obs, "ri_p_2side": ri_p, "n_fq": n_fq})
            print(f"    {oc:12s} [{tag:6s}]  b3 = {b_obs:+.6e}   ri_p = {ri_p:.4f}")

    ri = pd.DataFrame(rows)
    # main table = winsorized; raw as companion. Order for readability.
    ri["_ord"] = ri["flow_form"].map({"winsor": 0, "raw": 1})
    ri = ri.sort_values(["_ord", "outcome"]).drop(columns="_ord").reset_index(drop=True)
    ri.to_csv(RI_OUT, index=False)
    print(f"\n  (N_PERM={N_PERM}, iters={DEMEAN_ITERS}, seed={SEED}; MAIN=winsor, raw=companion)")
    print(f"  wrote {RI_OUT.name}")
    print("  cross-check b3 against a reghdfe firm+quarter run before citing — a "
          "material mismatch means a stale panel or broken collapse.")
    return ri


# ----------------------------------------------------------------------------
# STEP 4 — world-B boundary statement (writing anchor; printed so it is not lost)
# ----------------------------------------------------------------------------
def print_world_b_boundary():
    print("\n===== STEP 4: WORLD-B BOUNDARY STATEMENT (for §7.6 prose) =====")
    print(
        "  This is an accounting decomposition of observed institutional and residual\n"
        "  ownership flows, not an interference lower bound. Quantity data identify only\n"
        "  the co-movement of US, non-US, and residual-sector flows. World B — where\n"
        "  US demand falls but price absorbs it and quantities do not move — leaves NO\n"
        "  signature in shares outstanding and CANNOT be identified from quantities.\n"
        "  Distinguishing world B requires the H2.2 price evidence. Do NOT claim the\n"
        "  mechanism 'wins under either world'; the quantity decomposition is silent on\n"
        "  world B by construction.")


if __name__ == "__main__":
    panel = build_panel()
    diagnostics(panel)
    # RI consumes the FROZEN parquet artifact, not the in-memory frame: the RI
    # p is then exactly reproducible given the committed artifact, and shares
    # bitwise-identical values with the .dta the CRVE do-file reads.
    ri_flow(pd.read_parquet(PANEL_OUT))
    print_world_b_boundary()
    print("\nDONE.")
