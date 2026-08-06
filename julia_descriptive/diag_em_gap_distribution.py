#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diag_em_gap_distribution.py
===========================
READ-ONLY post-build staleness diagnostic for the advisor-directed snapshot rule
(Emanuele, 2026-08-04 meeting: per (fund_id, fsym_id), keep the LAST report
inside the calendar quarter).

WHY IT EXISTS
-------------
The quarter rule expands the estimation universe by admitting intra-quarter
reporters, and it pays for that with STALENESS. 03_eom_etl.jl already writes the
same numbers at build time (Phase C, block C2b -> 03_eom_asof_gap_summary.csv).
This script regenerates them from a BUILT panel, so:
  * the gap distribution can be re-derived without a multi-hour ETL re-run, and
  * the new panel can be put side by side with the archived W=10 vintage
    (holdings_eom_preEM.parquet) in one table.

WHAT IT REPORTS  (row-weighted and MV-weighted, pooled / per quarter / per era)
    share of rows at gap = 0     (a true quarter-end snapshot)
    median gap                   (lower median on the discrete histogram:
                                  the smallest gap whose cumulative share
                                  reaches 0.5; NO interpolation -- same
                                  convention as gap_summary() in 03_eom_etl.jl)
    share of rows at gap > 14d
    max gap                      (must be <= 91 under the quarter rule)

It also re-checks the one hard boundary of the rule: no row may sit outside the
quarter it is stamped to. That is gate M3 in the ETL; repeating it here means a
panel handed over by someone else can be checked without trusting its build log.

WRITES ONLY  output/diag_em_gap_*.csv .  No pipeline artifact is touched.

Usage
-----
    python diag_em_gap_distribution.py
    python diag_em_gap_distribution.py --panel output/holdings_eom.parquet
    python diag_em_gap_distribution.py --compare output/holdings_eom_preEM.parquet
    python diag_em_gap_distribution.py --tag qtr        # suffix on the CSV names
"""

import argparse
import os
import sys
import time

import duckdb
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def connect():
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    # Heavy duckdb work spills to E: by project convention; C: has no room.
    tmp = os.environ.get("DPN_DUCKDB_TEMP_DIR", "E:/duckdb_tmp")
    try:
        os.makedirs(tmp, exist_ok=True)
        con.execute("SET temp_directory='%s'" % tmp.replace("\\", "/"))
        log("duckdb spill dir: %s" % tmp)
    except OSError as exc:
        log("WARNING could not use spill dir %s (%s); falling back to the default" % (tmp, exc))
    return con


def gap_stats(hist, gapcol="asof_gap_days", wcol="n_rows"):
    """Four staleness stats from a (gap, weight) histogram. Lower median."""
    h = hist[[gapcol, wcol]].dropna().sort_values(gapcol)
    tot = float(h[wcol].sum())
    if tot <= 0:
        return dict(total=tot, share_gap0=float("nan"), median_gap=-1,
                    share_gap_gt14=float("nan"), max_gap=-1, mean_gap=float("nan"))
    cum = h[wcol].cumsum() / tot
    reached = h.loc[cum >= 0.5, gapcol]
    return dict(
        total=tot,
        share_gap0=float(h.loc[h[gapcol] == 0, wcol].sum()) / tot,
        median_gap=int(reached.iloc[0]) if len(reached) else int(h[gapcol].iloc[-1]),
        share_gap_gt14=float(h.loc[h[gapcol] > 14, wcol].sum()) / tot,
        max_gap=int(h[gapcol].max()),
        mean_gap=float((h[gapcol] * h[wcol]).sum()) / tot,
    )


def era_of(qend):
    y = pd.Timestamp(qend).year
    if y <= 2005:
        return "1999-2005 ramp-up"
    if y <= 2015:
        return "2006-2015 mid"
    return "2016-2023 modern"


def describe(con, panel, label):
    path = panel.replace("\\", "/")
    if not os.path.exists(panel):
        sys.exit("panel not found: %s" % panel)
    log("scanning %s (%s, %.2f GB)" % (label, panel, os.path.getsize(panel) / 1024 ** 3))

    hist = con.execute("""
        SELECT report_date AS qend, asof_gap_days,
               COUNT(*) AS n_rows, SUM(adj_mv)/1e9 AS mv_b
        FROM read_parquet('%s')
        GROUP BY 1, 2 ORDER BY 1, 2
    """ % path).fetchdf()

    # M3 re-check: no row outside the quarter it is stamped to.
    bad = con.execute("""
        SELECT COUNT(*) AS n_out_of_quarter
        FROM read_parquet('%s')
        WHERE report_date_actual > report_date
           OR report_date_actual < CAST(DATE_TRUNC('quarter', report_date) AS DATE)
    """ % path).fetchone()[0]
    print("  rows outside their stamped quarter: %d   %s"
          % (bad, "[OK]" if bad == 0 else "[FAIL -- the panel double-counts a quarter]"))

    pooled_rows = gap_stats(hist.groupby("asof_gap_days", as_index=False)["n_rows"].sum())
    pooled_mv = gap_stats(hist.groupby("asof_gap_days", as_index=False)["mv_b"].sum(),
                          wcol="mv_b")
    print("  ROW-weighted: share(gap=0)=%6.2f%%  median=%3dd  share(gap>14d)=%6.2f%%  max=%3dd  mean=%5.2fd  (N=%d)"
          % (100 * pooled_rows["share_gap0"], pooled_rows["median_gap"],
             100 * pooled_rows["share_gap_gt14"], pooled_rows["max_gap"],
             pooled_rows["mean_gap"], pooled_rows["total"]))
    print("  MV-weighted : share(gap=0)=%6.2f%%  median=%3dd  share(gap>14d)=%6.2f%%  max=%3dd  mean=%5.2fd"
          % (100 * pooled_mv["share_gap0"], pooled_mv["median_gap"],
             100 * pooled_mv["share_gap_gt14"], pooled_mv["max_gap"], pooled_mv["mean_gap"]))

    per_q = []
    for qend, sdf in hist.groupby("qend"):
        r = gap_stats(sdf)
        m = gap_stats(sdf, wcol="mv_b")
        per_q.append(dict(panel=label, qend=qend, n_rows=int(r["total"]),
                          share_gap0=r["share_gap0"], median_gap=r["median_gap"],
                          share_gap_gt14=r["share_gap_gt14"], max_gap=r["max_gap"],
                          mean_gap=r["mean_gap"],
                          mv_share_gap0=m["share_gap0"],
                          mv_median_gap=m["median_gap"],
                          mv_share_gap_gt14=m["share_gap_gt14"]))
    per_q = pd.DataFrame(per_q).sort_values("qend")

    hist["era"] = hist["qend"].map(era_of)
    per_era = []
    for era, sdf in hist.groupby("era"):
        r = gap_stats(sdf.groupby("asof_gap_days", as_index=False)["n_rows"].sum())
        per_era.append(dict(panel=label, era=era, n_rows=int(r["total"]), **{
            k: r[k] for k in ("share_gap0", "median_gap", "share_gap_gt14", "max_gap", "mean_gap")}))
    per_era = pd.DataFrame(per_era).sort_values("era")

    pooled = pd.DataFrame([
        dict(panel=label, weight="rows", **pooled_rows),
        dict(panel=label, weight="mv_b", **pooled_mv),
    ])
    return pooled, per_q, per_era, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=os.path.join(OUT_DIR, "holdings_eom.parquet"))
    ap.add_argument("--compare", default=None,
                    help="second panel to put side by side (e.g. the archived "
                         "holdings_eom_preEM.parquet W=10 vintage)")
    ap.add_argument("--tag", default="", help="suffix appended to the output CSV names")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    con = connect()
    suffix = ("_" + args.tag) if args.tag else ""

    pooled, per_q, per_era, bad = describe(con, args.panel, "current")
    fails = 1 if bad else 0
    if args.compare:
        p2, q2, e2, bad2 = describe(con, args.compare, "compare")
        pooled = pd.concat([pooled, p2], ignore_index=True)
        per_q = pd.concat([per_q, q2], ignore_index=True)
        per_era = pd.concat([per_era, e2], ignore_index=True)
        fails += 1 if bad2 else 0

    pooled.to_csv(os.path.join(OUT_DIR, "diag_em_gap_pooled%s.csv" % suffix), index=False)
    per_q.to_csv(os.path.join(OUT_DIR, "diag_em_gap_by_quarter%s.csv" % suffix), index=False)
    per_era.to_csv(os.path.join(OUT_DIR, "diag_em_gap_by_era%s.csv" % suffix), index=False)

    print("\n--- pooled ---")
    print(pooled)
    print("\n--- by era ---")
    print(per_era)
    print("\nwrote: diag_em_gap_pooled%s.csv, diag_em_gap_by_quarter%s.csv, "
          "diag_em_gap_by_era%s.csv" % (suffix, suffix, suffix))
    if fails:
        sys.exit("FAILED: %d panel(s) contain rows outside their stamped quarter." % fails)


if __name__ == "__main__":
    main()
