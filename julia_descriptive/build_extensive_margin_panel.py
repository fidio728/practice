# ============================================================================
# PRE-P0 VINTAGE WARNING (P0 holdings-snapshot rebuild, 2026-08-04)
# This script's output is PRE-P0: it was built from the exact-EOM
# holdings_eom.parquet (or from merged_us_eu_zero_filled.parquet built from
# it). The as-of quarter-end selection rule in 03_eom_etl.jl CHANGED the
# panel's fund universe on EVERY quarter. PRE-P0, pending re-run, do not mix
# with post-P0 results. Register: julia_descriptive/VINTAGE_P0.md
# ============================================================================

"""
build_extensive_margin_panel.py

WORK-PACKAGE A1 (extensive-margin outcomes; G1 of DESIGN_REVIEW_V3_2026_08_03).

WHAT / WHY.
  Every existing ownership outcome in this project is value/weight-based
  (portfolio_weight_eu, delta_w, ownership_share, flow). The divestment
  literature's FIRST margin -- Hong-Kacperczyk holder COUNTS, Chen-Hong-Stein
  BREADTH -- was never constructed here, because 04_us_ownership_european.jl
  collapses fund_id away (SUM(adj_mv) GROUP BY sec_entity_id, sec_country,
  investor_country, report_date -- see 04:141-153) BEFORE any outcome forms, and
  06 inherits that fund-collapsed I_ict. This script goes back to the raw
  holdings_eom panel (which still carries fund_id) and builds the count/breadth
  outcomes onto the SAME C6 zero-filled grid the headline uses, so the extensive
  margin is directly comparable to the intensive (weight) margin.

PRE-REGISTERED READING (written here, NOT as a conclusion; the estimate decides):
  * exit-up / breadth-down for US on (china_exposure x Shock) => partial
    divestment on the extensive margin (holders leave, not just dollars).
  * null => the intensive-margin null's COMPLETENESS is strengthened (no hidden
    extensive-margin action).
  * positive (breadth-up / exit-down for US) => anti-H2.1, same sign as the
    weight margin.
  The section-5.5 descriptive zero-fill gap once ran +11.26pp toward H2.1 on the
  pre-B7 panel; that number is SUPERSEDED and is NEVER cited. The descriptive
  refresh below recomputes the live gap on the current B7 grid.

WHAT THIS FILE DOES (and does not).
  This builds the LOCKED PANEL CONTRACT only: output/extensive_margin_panel.dta
  (+ .parquet twin). Estimation (reghdfe 3-pairwise us_cn us_cn_shock,
  absorb(fq gq ig) vce(cluster firm_n rd_m)), the exit/init pairing restriction,
  the B9 degenerate-VCE guard, and randomization inference (run_ri_3pairwise.py
  machinery) are DOWNSTREAM steps that CONSUME this panel; they are not in here.

MIRRORED CONVENTIONS (any silent deviation invalidates comparison to the
headline; each is cited to the exact source line):
  * holdings path        : output/holdings_eom.parquet          (03:80, 04:43)
  * row-level filters     : sec_entity_id IS NOT NULL AND investor_country IS NOT
                            NULL AND issue_type IN ('EQ','AD')   (04:149-152)
                            -- EQ = common equity, AD = ADR/GDR (US path into EU
                            firms); mirrors 04's I_ict WHERE clause verbatim.
  * holder_group (US/NONUS): CASE WHEN investor_country='US' THEN 'US'
                            ELSE 'NONUS' END                     (06:147, 06:159)
  * firm universe         : the C6 grid's DISTINCT sec_entity_id, i.e.
                            eu_entity_universe (06:64-83); n_holders/n_active are
                            restricted to it, mirroring ict_grouped's
                            `WHERE sec_entity_id IN eu_entity_universe` (06:151).
                            n_holders sums a firm across ALL its sec_country
                            listings (join keys off sec_entity_id, not
                            sec_country) exactly as ict_grouped does (06:145-153).
  * quarter calendar      : the grid's report_date = quarter-end LAST_DAY of
                            Mar/Jun/Sep/Dec 1999-2023 (03:163-164, 06:105-112);
                            holdings_eom already carries exactly these 100
                            quarter-ends, so aggregation on report_date and the
                            join to the grid align 1:1.
  * lag / difference rule : LAG over (firm, holder_group) ORDER BY report_date on
                            the FULL Cartesian grid, exactly the backward-diff
                            convention of delta_w (06:302-304). The grid is
                            complete, so LAG is always the contiguous prior
                            quarter; a datediff==3-months guard enforces
                            contiguity belt-and-suspenders and yields missing at
                            each series' first quarter.
  * cn_lag / shock        : joined VERBATIM from the grid (china_share_lag1q ->
                            cn_lag, shock_us_cn -> shock), never recomputed; a
                            random-sample spot-assert checks exact equality.

DETERMINISM. n_holders / n_active are COUNT(DISTINCT fund_id) -- integer
  aggregation. It is immune to the duckdb parallel float-sum nondeterminism that
  hit the flow build (a fund is either in the set or not; thread order does not
  change an integer set cardinality). Parallel threads are therefore SAFE here.

Usage:
  python build_extensive_margin_panel.py            # FULL build (writes canonical)
  python build_extensive_margin_panel.py --smoke    # 3-quarter slice; *_smoke.* only
"""

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
EOM_PARQUET = OUT / "holdings_eom.parquet"
GRID_PARQUET = OUT / "merged_us_eu_zero_filled.parquet"

EOM_URI = str(EOM_PARQUET).replace("\\", "/")
GRID_URI = str(GRID_PARQUET).replace("\\", "/")

# Contract column order (LOCKED). Lean: only these are written to .dta/.parquet.
CONTRACT_COLS = [
    "firm_str", "rdate", "us",
    "n_holders", "n_active", "breadth",
    "d_breadth", "d_nh",
    "held_lag", "exit", "init",
    "cn_lag", "shock",
]


def log(msg=""):
    print(msg, flush=True)


def build(con, smoke_quarters=None):
    """Build the extensive-margin panel table `panel` in `con`.

    smoke_quarters: if not None, a list of DATE strings (quarter-ends) to which
    both the grid and the holdings aggregation are restricted. Descriptive/lag
    behaviour at the first slice quarter is expected to be missing (no prior row
    in the slice); every other quarter carries true contiguous lags.
    """
    # ---- (0) firm universe + quarter set, taken from the grid itself. --------
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE univ AS
        SELECT DISTINCT sec_entity_id
        FROM read_parquet('{GRID_URI}')
    """)
    n_univ = con.execute("SELECT COUNT(*) FROM univ").fetchone()[0]
    log(f"  firm universe (grid DISTINCT sec_entity_id): {n_univ:,}")

    # Optional smoke slice predicate applied to BOTH the grid and holdings_eom.
    q_filter_eom = ""
    q_filter_grid = ""
    if smoke_quarters is not None:
        qlist = ", ".join(f"DATE '{q}'" for q in smoke_quarters)
        q_filter_eom = f"AND report_date IN ({qlist})"
        q_filter_grid = f"WHERE g.report_date IN ({qlist})"
        log(f"  SMOKE slice quarters: {smoke_quarters}")

    # ---- (1) n_holders: COUNT(DISTINCT fund_id) per firm x group x quarter. ---
    # Mirrors 04 I_ict WHERE clause (04:149-152) + 06 holder_group (06:147) + 06
    # universe filter (06:151). SUM(adj_mv) -> COUNT(DISTINCT fund_id) is the only
    # change; keys off sec_entity_id (not sec_country) exactly like ict_grouped.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE nh AS
        SELECT
            sec_entity_id,
            CASE WHEN investor_country = 'US' THEN 'US' ELSE 'NONUS' END AS holder_group,
            report_date,
            COUNT(DISTINCT fund_id) AS n_holders
        FROM read_parquet('{EOM_URI}')
        WHERE sec_entity_id   IS NOT NULL
          AND investor_country IS NOT NULL
          AND issue_type IN ('EQ','AD')
          AND sec_entity_id IN (SELECT sec_entity_id FROM univ)
          {q_filter_eom}
        GROUP BY 1, 2, 3
    """)

    # ---- (2) n_active: Chen-Hong-Stein denominator (group x quarter). ---------
    # Distinct funds in group g with >=1 EQ/AD position in ANY grid (universe)
    # firm that quarter. Identical row-level filters + universe restriction, so
    # every fund counted in n_holders(i) is by construction counted in n_active
    # (=> breadth = n_holders / n_active in [0,1], asserted below).
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE na AS
        SELECT
            CASE WHEN investor_country = 'US' THEN 'US' ELSE 'NONUS' END AS holder_group,
            report_date,
            COUNT(DISTINCT fund_id) AS n_active
        FROM read_parquet('{EOM_URI}')
        WHERE sec_entity_id   IS NOT NULL
          AND investor_country IS NOT NULL
          AND issue_type IN ('EQ','AD')
          AND sec_entity_id IN (SELECT sec_entity_id FROM univ)
          {q_filter_eom}
        GROUP BY 1, 2
    """)

    # ---- (3) Zero-fill on the FULL C6 grid + breadth. -------------------------
    # LEFT JOIN direction (grid <- nh/na) makes every grid cell present; absent
    # holdings => n_holders 0 (a legitimate zero, not missingness). breadth is
    # NULL only where a whole (group, quarter) has no active funds (n_active=0,
    # e.g. NONUS in early quarters -- mirrors 06's portfolio_weight_eu ELSE NULL
    # when a group's European book is empty, 06:267-269).
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE base AS
        SELECT
            g.sec_entity_id                              AS firm_id,
            g.holder_group                               AS holder_group,
            g.report_date                                AS report_date,
            COALESCE(nh.n_holders, 0)                    AS n_holders,
            COALESCE(na.n_active, 0)                      AS n_active,
            CASE WHEN COALESCE(na.n_active, 0) > 0
                 THEN COALESCE(nh.n_holders, 0)::DOUBLE / na.n_active
                 ELSE NULL END                           AS breadth,
            g.china_share_lag1q                          AS cn_lag,
            g.shock_us_cn                                AS shock
        FROM read_parquet('{GRID_URI}') g
        LEFT JOIN nh
               ON g.sec_entity_id = nh.sec_entity_id
              AND g.holder_group  = nh.holder_group
              AND g.report_date   = nh.report_date
        LEFT JOIN na
               ON g.holder_group  = na.holder_group
              AND g.report_date   = na.report_date
        {q_filter_grid}
    """)

    # ---- (4) Contiguous backward lags + exit/init conditioning. ---------------
    # Window over (firm, group) ordered by quarter -- the delta_w convention
    # (06:302-304). contig requires a prior row exactly one quarter (3 months)
    # back; on the complete grid this fails ONLY at each series' first quarter,
    # giving the contract's missing pattern there.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE panel AS
        WITH lagged AS (
            SELECT
                base.*,
                LAG(n_holders) OVER w AS n_holders_lag,
                LAG(breadth)   OVER w AS breadth_lag,
                LAG(report_date) OVER w AS prev_q
            FROM base
            WINDOW w AS (PARTITION BY firm_id, holder_group ORDER BY report_date)
        )
        SELECT
            CAST(firm_id AS VARCHAR)                         AS firm_str,
            CAST(report_date AS TIMESTAMP)                   AS rdate,
            CASE WHEN holder_group = 'US' THEN 1 ELSE 0 END  AS us,
            n_holders,
            n_active,
            breadth,
            -- contiguity gate: prior row is exactly one quarter back
            (prev_q IS NOT NULL
             AND datediff('month', prev_q, report_date) = 3) AS contig,
            n_holders_lag,
            breadth_lag,
            cn_lag,
            shock
        FROM lagged
    """)

    # Derive the difference / conditional-count outcomes in a final projection so
    # the missing pattern is exactly per contract.
    con.execute("""
        CREATE OR REPLACE TEMP TABLE panel AS
        SELECT
            firm_str, rdate, us, n_holders, n_active, breadth,
            -- d_breadth / d_nh: contiguous quarters only, else missing.
            -- (breadth - breadth_lag is additionally NULL if either breadth is
            --  NULL, i.e. an n_active=0 quarter -- correct: cannot difference a
            --  missing breadth.)
            CASE WHEN contig THEN breadth - breadth_lag END           AS d_breadth,
            CASE WHEN contig THEN (n_holders - n_holders_lag)::DOUBLE END AS d_nh,
            -- held_lag = 1{n_holders_{t-1} > 0}; missing where no contiguous prior.
            CASE WHEN contig THEN CAST(n_holders_lag > 0 AS TINYINT) END AS held_lag,
            -- exit defined ONLY where held_lag==1; init defined ONLY where held_lag==0.
            CASE WHEN contig AND n_holders_lag > 0
                 THEN CAST(n_holders = 0 AS TINYINT) END              AS exit,
            CASE WHEN contig AND n_holders_lag = 0
                 THEN CAST(n_holders > 0 AS TINYINT) END              AS init,
            cn_lag, shock
        FROM panel
    """)
    return con


def run_asserts(df, n_q_expected):
    """Hard structural asserts on the built panel (fail loud, per convention)."""
    log("\n[asserts]")

    # (a) unique key
    assert not df.duplicated(["firm_str", "rdate", "us"]).any(), \
        "duplicate (firm_str, rdate, us) rows"
    log("  (a) unique key (firm_str, rdate, us): OK")

    # (b) paired grid: every firm-quarter has exactly 2 rows (us in {0,1})
    pair = df.groupby(["firm_str", "rdate"])["us"].agg(["nunique", "size"])
    assert (pair["size"] == 2).all() and (pair["nunique"] == 2).all(), \
        f"panel not fully paired: {(pair['size'] != 2).sum():,} firm-quarters lack exactly US+NONUS"
    log(f"  (b) paired grid: all {len(pair):,} firm-quarters have exactly 2 rows (US+NONUS): OK")

    # (c) completeness: each (firm, group) series has exactly n_q_expected rows
    per_series = df.groupby(["firm_str", "us"]).size()
    assert (per_series == n_q_expected).all(), \
        f"incomplete grid: {(per_series != n_q_expected).sum():,} (firm,group) series != {n_q_expected} quarters"
    log(f"  (c) completeness: every (firm,group) series has exactly {n_q_expected} quarters: OK")

    # (d) n_holders >= 0, no NaN; n_active >= 0, no NaN (zero-filled)
    assert df["n_holders"].notna().all() and (df["n_holders"] >= 0).all(), "n_holders NaN/negative"
    assert df["n_active"].notna().all() and (df["n_active"] >= 0).all(), "n_active NaN/negative"
    log("  (d) n_holders>=0, n_active>=0, no NaN (zero-filled): OK")

    # (e) breadth in [0,1] where non-null; n_holders<=n_active where n_active>0
    bm = df["breadth"].notna()
    assert df.loc[bm, "breadth"].between(0, 1).all(), "breadth outside [0,1]"
    pos = df["n_active"] > 0
    assert (df.loc[pos, "n_holders"] <= df.loc[pos, "n_active"]).all(), \
        "n_holders > n_active (subset property violated)"
    # where n_active==0, n_holders must be 0 and breadth NULL
    z = df["n_active"] == 0
    assert (df.loc[z, "n_holders"] == 0).all(), "n_active==0 but n_holders>0"
    assert df.loc[z, "breadth"].isna().all(), "n_active==0 but breadth not NULL"
    log(f"  (e) breadth in [0,1]; n_holders<=n_active; n_active==0 => n_holders==0 & breadth NULL: OK "
        f"({bm.sum():,} non-null breadth cells)")

    # (f) exit/init missingness EXACTLY per contract:
    #     exit non-missing  <=> held_lag==1
    #     init non-missing  <=> held_lag==0
    #     held_lag missing  <=> no contiguous prior (first quarter of series)
    hl1 = (df["held_lag"] == 1)
    hl0 = (df["held_lag"] == 0)
    assert df["exit"].notna().eq(hl1.fillna(False)).all(), \
        "exit non-missingness != {held_lag==1}"
    assert df["init"].notna().eq(hl0.fillna(False)).all(), \
        "init non-missingness != {held_lag==0}"
    assert not (df["exit"].notna() & df["init"].notna()).any(), \
        "exit and init both non-missing on some row"
    # held_lag missing exactly at first quarter of each (firm,group) series
    first_q = df.sort_values(["firm_str", "us", "rdate"]).groupby(["firm_str", "us"]).head(1).index
    is_first = pd.Series(False, index=df.index)
    is_first.loc[first_q] = True
    assert df["held_lag"].isna().eq(is_first).all(), \
        "held_lag missing pattern != first-quarter-of-series"
    log(f"  (f) exit<=>held_lag==1, init<=>held_lag==0, disjoint; held_lag missing only at "
        f"series start ({is_first.sum():,} rows): OK")

    # (g) d_breadth / d_nh missing at series start (and where breadth undifferenced)
    assert df.loc[is_first, "d_nh"].isna().all(), "d_nh not missing at series start"
    assert df.loc[is_first, "d_breadth"].isna().all(), "d_breadth not missing at series start"
    log(f"  (g) d_nh/d_breadth missing at series start: OK "
        f"(d_nh non-null={df['d_nh'].notna().sum():,}, d_breadth non-null={df['d_breadth'].notna().sum():,})")

    # (h) cn_lag same within firm-quarter (firm-level exposure, not group-level);
    #     shock a single common value per quarter (no within-quarter variation).
    cn_nu = df.groupby(["firm_str", "rdate"])["cn_lag"].nunique(dropna=False)
    assert (cn_nu <= 1).all(), "cn_lag varies across US/NONUS within a firm-quarter"
    sh_nu = df.groupby("rdate")["shock"].nunique(dropna=False)
    assert (sh_nu <= 1).all(), "shock varies within a quarter"
    log("  (h) cn_lag constant within firm-quarter; shock constant within quarter: OK")


def spot_assert_grid_join(con, df, n_sample=2000, seed=20260702):
    """Spot-assert cn_lag/shock were joined VERBATIM from the grid (NaN-aware)."""
    log("\n[spot-assert cn_lag/shock vs grid]")
    samp = df.sample(n=min(n_sample, len(df)), random_state=seed)[
        ["firm_str", "rdate", "us", "cn_lag", "shock"]
    ].copy()
    samp["holder_group"] = np.where(samp["us"] == 1, "US", "NONUS")
    con.register("samp", samp)
    chk = con.execute(f"""
        SELECT s.firm_str, s.holder_group, s.cn_lag AS cn_panel, s.shock AS sh_panel,
               g.china_share_lag1q AS cn_grid, g.shock_us_cn AS sh_grid
        FROM samp s
        JOIN read_parquet('{GRID_URI}') g
          ON CAST(g.sec_entity_id AS VARCHAR) = s.firm_str
         AND g.holder_group = s.holder_group
         AND CAST(g.report_date AS TIMESTAMP) = s.rdate
    """).df()
    con.unregister("samp")
    assert len(chk) == len(samp), \
        f"spot-join lost rows: {len(chk)} matched vs {len(samp)} sampled"

    def eq_nan(a, b):
        a = a.to_numpy(dtype="float64")
        b = b.to_numpy(dtype="float64")
        return ((np.isnan(a) & np.isnan(b)) | (a == b)).all()

    assert eq_nan(chk["cn_panel"], chk["cn_grid"]), "cn_lag != grid china_share_lag1q on sample"
    assert eq_nan(chk["sh_panel"], chk["sh_grid"]), "shock != grid shock_us_cn on sample"
    log(f"  {len(chk):,} sampled rows: cn_lag==china_share_lag1q and shock==shock_us_cn (NaN-aware): OK")


def descriptive_refresh(df):
    """B7-grid zero-fill shares (US vs NONUS) + gap + breadth distributions.

    Reported at firm-quarter (cell) AND firm level per the claims-match-data
    convention. The old +11.26pp pre-B7 gap is SUPERSEDED and NOT cited; this is
    the live number.
    """
    log("\n" + "=" * 70)
    log("DESCRIPTIVE REFRESH (B7 grid) -- extensive-margin zero-fill + breadth")
    log("=" * 70)

    grp = {1: "US", 0: "NONUS"}
    n_q = df["rdate"].nunique()

    # ---- firm-quarter (cell) level zero-fill share, full grid + estimable subset
    log("\n[firm-quarter cell level] share of cells with n_holders == 0")
    log(f"  {'group':6s} {'n_cells':>12s} {'n_zero':>12s} {'zero_share':>11s} "
        f"{'n_cells(cn!=NA)':>16s} {'zero_share(cn!=NA)':>19s}")
    cell = {}
    for u in (1, 0):
        sub = df[df["us"] == u]
        n_cells = len(sub)
        n_zero = int((sub["n_holders"] == 0).sum())
        zsh = n_zero / n_cells if n_cells else float("nan")
        est = sub[sub["cn_lag"].notna()]
        n_cells_e = len(est)
        zsh_e = (est["n_holders"] == 0).sum() / n_cells_e if n_cells_e else float("nan")
        cell[u] = (zsh, zsh_e)
        log(f"  {grp[u]:6s} {n_cells:12,d} {n_zero:12,d} {zsh:11.4f} "
            f"{n_cells_e:16,d} {zsh_e:19.4f}")
    gap_cell = cell[1][0] - cell[0][0]
    gap_cell_e = cell[1][1] - cell[0][1]
    log(f"  GAP (US - NONUS) full grid : {gap_cell:+.4f}  ({100*gap_cell:+.2f} pp)")
    log(f"  GAP (US - NONUS) cn!=NA     : {gap_cell_e:+.4f}  ({100*gap_cell_e:+.2f} pp)")
    log("  (US zero-share is expected HIGHER: 13F covers US institutions' EU book")
    log("   more thinly than the NONUS book -> coverage asymmetry, see riskset robustness.)")

    # ---- firm level: ever-held share
    log("\n[firm level] distinct firms EVER held (n_holders>0 in >=1 quarter)")
    n_firms = df["firm_str"].nunique()
    log(f"  total distinct firms in grid: {n_firms:,} (x {n_q} quarters x 2 groups)")
    log(f"  {'group':6s} {'n_firms':>9s} {'ever_held':>10s} {'never_held':>11s} {'ever_share':>11s}")
    firm = {}
    for u in (1, 0):
        sub = df[df["us"] == u]
        ever = sub.groupby("firm_str")["n_holders"].max()
        n_ever = int((ever > 0).sum())
        n_never = int((ever == 0).sum())
        esh = n_ever / n_firms if n_firms else float("nan")
        firm[u] = esh
        log(f"  {grp[u]:6s} {n_firms:9,d} {n_ever:10,d} {n_never:11,d} {esh:11.4f}")
    gap_firm = firm[1] - firm[0]
    log(f"  GAP ever-held share (US - NONUS): {gap_firm:+.4f}  ({100*gap_firm:+.2f} pp)")

    # ---- breadth distribution by group (non-null cells, and held-only cells)
    log("\n[breadth distribution] by group")
    log(f"  {'group':6s} {'scope':10s} {'n':>10s} {'mean':>9s} {'p50':>9s} {'p90':>9s} "
        f"{'p99':>9s} {'mean_nh':>9s}")
    for u in (1, 0):
        sub = df[df["us"] == u]
        for scope, s in (("all cells", sub[sub["breadth"].notna()]),
                         ("held>0", sub[sub["n_holders"] > 0])):
            if len(s):
                b = s["breadth"].dropna()
                log(f"  {grp[u]:6s} {scope:10s} {len(s):10,d} "
                    f"{b.mean():9.4f} {b.quantile(.5):9.4f} {b.quantile(.9):9.4f} "
                    f"{b.quantile(.99):9.4f} {s['n_holders'].mean():9.2f}")
            else:
                log(f"  {grp[u]:6s} {scope:10s} {0:10,d}  (empty)")

    # ---- n_active coverage path (the breadth DENOMINATOR and the exit driver).
    # holdings_eom thins at the panel tail from FactSet reporting lag (recent
    # quarters under-report). A held->0 drop caused by a fund simply not having
    # filed yet reads as a FALSE `exit`. This is the coverage-driven-disappearance
    # threat the spec names; the engaged/riskset subsample is the disclosed
    # downstream mitigation. Disclosed here so the tail is never read as real exit.
    log("\n[n_active coverage] active funds per group-quarter (breadth denominator)")
    na_gq = (df.groupby(["us", "rdate"])["n_active"].first()
               .rename("n_active").reset_index())
    for u in (1, 0):
        s = na_gq[na_gq["us"] == u].sort_values("rdate")
        vals = s["n_active"].to_numpy()
        med = float(np.median(vals)) if len(vals) else float("nan")
        # flag quarters with a >30% quarter-on-quarter drop
        drops = []
        for i in range(1, len(s)):
            prev, cur = vals[i - 1], vals[i]
            if prev > 0 and cur < 0.7 * prev:
                drops.append(f"{str(s['rdate'].iloc[i].date())}({cur:,}<-{prev:,})")
        tail = s.tail(6)
        log(f"  {grp[u]:6s} median={med:,.0f}  range=[{vals.min():,}..{vals.max():,}]  "
            f"n_q={len(s)}")
        log(f"         last-6: " + ", ".join(
            f"{str(r.rdate.date())}={int(r.n_active):,}" for r in tail.itertuples()))
        if drops:
            log(f"         >30% QoQ drops (reporting-lag suspects): " + "; ".join(drops))

    # ---- outcome availability (what estimation will actually see)
    log("\n[estimation-sample availability] non-null outcome x (cn_lag & shock) present")
    est = df[df["cn_lag"].notna() & df["shock"].notna()]
    log(f"  rows with cn_lag & shock present: {len(est):,}")
    for c in ("d_breadth", "d_nh", "exit", "init"):
        log(f"    {c:10s} non-null: {est[c].notna().sum():,}")


def to_contract_frame(con):
    """Fetch the panel, coerce to the LOCKED contract dtypes (lean)."""
    df = con.execute(f"SELECT {', '.join(CONTRACT_COLS)} FROM panel").df()
    df["firm_str"] = df["firm_str"].astype(str)
    df["rdate"] = pd.to_datetime(df["rdate"])
    df["us"] = df["us"].astype("int8")
    df["n_holders"] = pd.to_numeric(df["n_holders"], errors="raise").astype("int32")
    df["n_active"] = pd.to_numeric(df["n_active"], errors="raise").astype("int32")
    for c in ("breadth", "d_breadth", "d_nh", "cn_lag", "shock"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
    # byte columns that carry missing -> pandas nullable Int8 (Stata byte + '.')
    for c in ("held_lag", "exit", "init"):
        df[c] = df[c].astype("Float64").round().astype("Int8")
    return df[CONTRACT_COLS]


def write_twins(df, dta_path, pq_path):
    """Write the .dta (Stata) and its .parquet twin from the same frame."""
    log(f"\n[write] {dta_path.name}")
    df.to_stata(dta_path, write_index=False, convert_dates={"rdate": "tc"}, version=118)
    log(f"        {len(df):,} rows, {df.memory_usage(deep=True).sum()/1024**2:.1f} MB in-mem, "
        f".dta {dta_path.stat().st_size/1024**2:.1f} MB")
    log(f"[write] {pq_path.name}")
    df.to_parquet(pq_path, index=False)
    log(f"        .parquet {pq_path.stat().st_size/1024**2:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description="Build the extensive-margin (holder-count / breadth) panel.")
    ap.add_argument("--smoke", action="store_true",
                    help="Build on the last 3 quarters only; write *_smoke.* and do NOT touch canonical.")
    args = ap.parse_args()

    for p in (EOM_PARQUET, GRID_PARQUET):
        if not p.exists():
            sys.exit(f"Missing required input: {p}")

    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")   # mirror 00_setup dbcon RAM budget (00:86)
    con.execute("SET threads=4")            # COUNT(DISTINCT) is deterministic; parallel OK
    con.execute("SET preserve_insertion_order=false")  # 00:95 -- pipeline through on tight RAM

    # quarter set (from the grid, always the canonical 100 quarter-ends)
    all_q = [str(r[0]) for r in con.execute(
        f"SELECT DISTINCT report_date FROM read_parquet('{GRID_URI}') ORDER BY report_date"
    ).fetchall()]

    if args.smoke:
        smoke_q = all_q[-3:]
        log("=" * 70)
        log("SMOKE BUILD -- last 3 quarters only. Canonical outputs NOT written.")
        log("At the first slice quarter the lag-based outcomes (d_*, held_lag, exit,")
        log("init) are missing by construction (no prior row in the slice); this is")
        log("expected and differs from the full build only at that boundary quarter.")
        log("=" * 70)
        build(con, smoke_quarters=smoke_q)
        df = to_contract_frame(con)
        run_asserts(df, n_q_expected=len(smoke_q))
        spot_assert_grid_join(con, df)
        descriptive_refresh(df)
        write_twins(df,
                    OUT / "extensive_margin_panel_smoke.dta",
                    OUT / "extensive_margin_panel_smoke.parquet")
        # smoke-specific head for eyeballing the lag/exit/init structure
        log("\n[smoke] sample rows for one firm across the 3 quarters:")
        fex = df.loc[df["n_holders"] > 0, "firm_str"].iloc[0]
        cols = ["firm_str", "rdate", "us", "n_holders", "n_active", "breadth",
                "d_breadth", "d_nh", "held_lag", "exit", "init", "cn_lag", "shock"]
        log(df[df["firm_str"] == fex].sort_values(["us", "rdate"])[cols].to_string(index=False))
        log("\nSMOKE DONE. Full build NOT run (per task).")
    else:
        log("=" * 70)
        log("FULL BUILD -- all 100 quarters, full C6 grid.")
        log("=" * 70)
        build(con, smoke_quarters=None)
        df = to_contract_frame(con)
        run_asserts(df, n_q_expected=len(all_q))
        spot_assert_grid_join(con, df)
        descriptive_refresh(df)
        write_twins(df,
                    OUT / "extensive_margin_panel.dta",
                    OUT / "extensive_margin_panel.parquet")
        log("\nFULL BUILD DONE.")

    con.close()


if __name__ == "__main__":
    main()
