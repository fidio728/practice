#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diag_p0_asof_window.py
======================
READ-ONLY diagnostic for the P0 holdings-snapshot rebuild (as-of quarter-end rule).

WHAT IT DOES
------------
Reads ONLY the raw FactSet parquet chunks (E:/Data/Data/raw_parquet/Factset_FundOwners_*.parquet).
Writes ONLY files named  output/diag_p0_*.csv|.parquet  (no pipeline artifact is touched).

It reproduces, on the FULL 1999-2023 sample, the coverage-vs-W curve that the main
session measured on 2022-2023 only:

    coverage_Q(W) = # distinct funds with ADJ_MV>0 having at least one REPORT_DATE
                    in [qend_Q - W, qend_Q]
                    ---------------------------------------------------------------
                    # distinct funds with ADJ_MV>0 reporting ANYWHERE in quarter Q

W = 0 is the CURRENT (old) rule in 03_eom_etl.jl:
        CAST(REPORT_DATE AS DATE) = LAST_DAY(CAST(REPORT_DATE AS DATE))
    i.e. an exact calendar-quarter-end match.

W = 92 is degenerate BY CONSTRUCTION under this denominator (any report inside the
quarter is within 92 days of the quarter-end), so it must come out at ~100%.  That is
kept deliberately as a consistency anchor on the denominator, not as a candidate W.

It also reports, per quarter and per W:
  * the US share of captured funds            -> GATE 2 (bias-relevant calendar swing)
  * row coverage and MV coverage              -> magnitude of what the old rule drops
  * US-investor x EU-firm nominal $ totals    -> the advisor-figure gate numbers
  * staleness: mean/median as-of gap and the share of captured funds with gap > 14d
  * multi-report funds: share of (fund, quarter) cells with >=2 report dates inside
    the window (this is the only place where "per (fund,fsym)" selection differs from
    "per fund" selection, so it bounds that design choice's impact)

METHOD / COST
-------------
One full pass over the 8 raw chunks aggregating to the (fund_id, report_date) grain
(~10^6 rows), cached to output/diag_p0_fund_report_dates.parquet.  Everything else is
derived from that cache, so re-runs are seconds.  Use --rebuild-cache to force a
re-scan.

Caveat that is deliberately NOT hidden: this diagnostic works at the (fund, date)
grain, so its MV numbers are PRE-dedup (they sum every raw row, exactly like the
current ETL does).  The ETL rebuild adds a ROW_NUMBER dedup at the (fund, fsym) grain;
section 6 measures how big that dedup is on two benchmark quarters.

Usage:
    python diag_p0_asof_window.py [--rebuild-cache] [--skip-dupcheck]
"""

import os
import sys
import time

import duckdb
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 400)

# ----------------------------------------------------------------------------- paths
RAW_DIR = os.environ.get("DPN_RAW_PARQUET_DIR", r"E:\Data\Data\raw_parquet")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

PATTERN = os.path.join(RAW_DIR, "Factset_FundOwners_*.parquet").replace("\\", "/")
CACHE = os.path.join(OUT_DIR, "diag_p0_fund_report_dates.parquet").replace("\\", "/")

# Smoke-test hooks (do NOT set these for the real run): point the scan at a single
# chunk and the outputs at a scratch dir so the whole script can be exercised in
# ~1 min before committing to the full 20 GB pass.
if os.environ.get("DPN_DIAG_PATTERN"):
    PATTERN = os.environ["DPN_DIAG_PATTERN"].replace("\\", "/")
if os.environ.get("DPN_DIAG_OUT"):
    OUT_DIR = os.environ["DPN_DIAG_OUT"]
    os.makedirs(OUT_DIR, exist_ok=True)
    CACHE = os.path.join(OUT_DIR, "diag_p0_fund_report_dates.parquet").replace("\\", "/")

REBUILD = "--rebuild-cache" in sys.argv
SKIP_DUP = "--skip-dupcheck" in sys.argv

# W grid.  0 = the current exact-EOM rule.  92 = whole-quarter anchor.
W_GRID = [0, 3, 7, 10, 14, 31, 92]

# Same 28-country EU list as 00_setup.jl (single source of truth there).
EU_COUNTRIES = ("GB", "DE", "FR", "NL", "CH", "IT", "ES", "SE", "DK", "NO", "FI",
                "BE", "AT", "IE", "LU", "PT", "PL", "CZ", "HU", "GR", "RO", "SK",
                "SI", "BG", "HR", "EE", "LV", "LT")
EU_SQL = "(" + ",".join("'%s'" % c for c in EU_COUNTRIES) + ")"

YEAR_MIN, YEAR_MAX = 1999, 2023


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def rule(title):
    print("\n" + "=" * 100, flush=True)
    print(title, flush=True)
    print("=" * 100, flush=True)


# ----------------------------------------------------------------------------- connect
con = duckdb.connect()
con.execute("SET memory_limit='6GB'")
con.execute("SET threads=4")
con.execute("SET preserve_insertion_order=false")
_tmp = os.path.join(os.environ.get("TEMP", "."), "duckdb_spill_diag_p0")
os.makedirs(_tmp, exist_ok=True)
con.execute("SET temp_directory='%s'" % _tmp.replace("\\", "/"))

log("duckdb %s" % duckdb.__version__)
log("raw pattern: %s" % PATTERN)
log("cache      : %s" % CACHE)

# =============================================================================
# STEP 1 -- build / reuse the (fund_id, report_date) cache
# =============================================================================
if REBUILD or not os.path.exists(CACHE):
    log("building cache (ONE full pass over ~20 GB of raw parquet; expect 10-40 min) ...")
    t0 = time.time()
    con.execute("""
        COPY (
            SELECT
                FACTSET_FUND_ID                       AS fund_id,
                CAST(REPORT_DATE AS DATE)             AS report_date,
                MAX(ISO_COUNTRY)                      AS investor_country,
                COUNT(DISTINCT ISO_COUNTRY)           AS n_investor_country,
                COUNT(*)                              AS n_rows,
                CAST(SUM(ADJ_MV) AS DOUBLE)           AS mv,
                CAST(SUM(CASE WHEN ISO_COUNTRY = 'US'
                              AND SEC_FIRM_ISO_COUNTRY IN %s
                         THEN ADJ_MV ELSE 0 END) AS DOUBLE) AS mv_us_eu
            FROM read_parquet('%s')
            WHERE ADJ_MV IS NOT NULL
              AND ADJ_MV > 0
              AND EXTRACT(YEAR FROM CAST(REPORT_DATE AS DATE)) BETWEEN %d AND %d
            GROUP BY 1, 2
        ) TO '%s' (FORMAT parquet, COMPRESSION zstd)
    """ % (EU_SQL, PATTERN, YEAR_MIN, YEAR_MAX, CACHE))
    log("cache built in %.1f min" % ((time.time() - t0) / 60.0))
else:
    log("cache already present -- reusing (pass --rebuild-cache to force a re-scan)")

con.execute("""
    CREATE OR REPLACE TABLE fd AS
    SELECT fund_id,
           report_date,
           investor_country,
           n_investor_country,
           n_rows,
           mv,
           mv_us_eu,
           CAST(DATE_TRUNC('quarter', report_date) + INTERVAL 3 MONTH - INTERVAL 1 DAY AS DATE) AS qend
    FROM read_parquet('%s')
""" % CACHE)

n_cache, n_funds, n_dates = con.execute(
    "SELECT COUNT(*), COUNT(DISTINCT fund_id), COUNT(DISTINCT report_date) FROM fd"
).fetchone()
log("cache grain rows=%d  distinct funds=%d  distinct report dates=%d" % (n_cache, n_funds, n_dates))

n_multi_ctry = con.execute("SELECT COUNT(*) FROM fd WHERE n_investor_country > 1").fetchone()[0]
if n_multi_ctry:
    log("NOTE: %d (fund, date) cells carry >1 ISO_COUNTRY; MAX() used. Not material for the US share unless large."
        % n_multi_ctry)

con.execute("""
    CREATE OR REPLACE TABLE fdq AS
    SELECT *,
           DATEDIFF('day', report_date, qend) AS gap,
           dayname(qend)                      AS qend_dow,
           CASE WHEN dayname(qend) IN ('Saturday', 'Sunday') THEN 1 ELSE 0 END AS qend_is_weekend
    FROM fd
""")

# =============================================================================
# STEP 2 -- per-quarter denominator (funds reporting ANYWHERE in the quarter)
# =============================================================================
den = con.execute("""
    SELECT qend,
           qend_dow,
           qend_is_weekend,
           COUNT(DISTINCT fund_id)                                          AS n_funds_q,
           COUNT(DISTINCT CASE WHEN investor_country='US' THEN fund_id END) AS n_us_funds_q,
           SUM(n_rows)                                                      AS n_rows_q,
           SUM(mv)/1e9                                                      AS mv_q_b,
           SUM(mv_us_eu)/1e9                                                AS mv_us_eu_q_b
    FROM fdq
    GROUP BY 1,2,3
    ORDER BY 1
""").df()
log("quarters in sample: %d  (%s .. %s)" % (len(den), den.qend.min(), den.qend.max()))

# =============================================================================
# STEP 3 -- coverage-vs-W curve
# =============================================================================
frames = []
for W in W_GRID:
    q = con.execute("""
        SELECT qend,
               COUNT(DISTINCT fund_id)                                          AS n_funds_cap,
               COUNT(DISTINCT CASE WHEN investor_country='US' THEN fund_id END) AS n_us_funds_cap,
               SUM(n_rows)                                                      AS n_rows_cap,
               SUM(mv)/1e9                                                      AS mv_cap_b,
               SUM(mv_us_eu)/1e9                                                AS mv_us_eu_cap_b,
               AVG(gap)                                                         AS mean_gap,
               MEDIAN(gap)                                                      AS median_gap,
               SUM(CASE WHEN gap > 14 THEN n_rows ELSE 0 END)
                   / NULLIF(CAST(SUM(n_rows) AS DOUBLE), 0)                     AS rowshare_gap_gt14
        FROM fdq
        WHERE gap <= %d
        GROUP BY 1
    """ % W).df()
    q["W"] = W
    frames.append(q)

cov = pd.concat(frames, ignore_index=True).merge(den, on="qend", how="right")
cov["W"] = cov["W"].fillna(-1)
for c in ["n_funds_cap", "n_us_funds_cap", "n_rows_cap"]:
    cov[c] = cov[c].fillna(0)
cov["coverage"] = cov.n_funds_cap / cov.n_funds_q
cov["row_coverage"] = cov.n_rows_cap / cov.n_rows_q
cov["mv_coverage"] = cov.mv_cap_b / cov.mv_q_b
cov["us_share_cap"] = cov.n_us_funds_cap / cov.n_funds_cap.replace(0, pd.NA)
cov["us_share_q"] = cov.n_us_funds_q / cov.n_funds_q
cov["year"] = pd.to_datetime(cov.qend).dt.year
cov = cov.sort_values(["W", "qend"]).reset_index(drop=True)

cov_path = os.path.join(OUT_DIR, "diag_p0_asof_coverage_by_quarter.csv")
cov.to_csv(cov_path, index=False)
log("wrote %s (%d rows)" % (cov_path, len(cov)))

# ------------------------------------------------------------------ 3a: the pivot
rule("1. COVERAGE-vs-W BY QUARTER  (captured funds / funds reporting anywhere in the quarter)")
piv = cov.pivot_table(index=["qend", "qend_dow", "qend_is_weekend"], columns="W",
                      values="coverage").reset_index()
piv.columns = [("W%s" % c if isinstance(c, (int, float)) else c) for c in piv.columns]
piv = piv.sort_values("qend")
piv_disp = piv.copy()
for c in piv_disp.columns:
    if str(c).startswith("W"):
        piv_disp[c] = (piv_disp[c] * 100).round(1)
piv_disp["n_funds_q"] = piv_disp.qend.map(den.set_index("qend").n_funds_q)
print(piv_disp.to_string(index=False), flush=True)
piv.to_csv(os.path.join(OUT_DIR, "diag_p0_asof_coverage_pivot.csv"), index=False)

# ------------------------------------------------------- 3b: weekday vs weekend gap
rule("2. GATE 1 -- WEEKDAY vs WEEKEND COVERAGE GAP, sample-wide and by era")


def gap_table(df, label):
    rows = []
    for W in W_GRID:
        s = df[df.W == W]
        wd = s[s.qend_is_weekend == 0].coverage
        we = s[s.qend_is_weekend == 1].coverage
        rows.append(dict(sample=label, W=W,
                         n_wd=len(wd), n_we=len(we),
                         wd_mean=100 * wd.mean(), we_mean=100 * we.mean(),
                         gap_pp=100 * (wd.mean() - we.mean()),
                         wd_min=100 * wd.min(), we_min=100 * we.min(),
                         all_min=100 * s.coverage.min(),
                         all_mean=100 * s.coverage.mean()))
    return pd.DataFrame(rows)


ERAS = [("FULL 1999-2023", cov),
        ("ramp-up 1999-2005", cov[cov.year <= 2005]),
        ("mid 2006-2015", cov[(cov.year >= 2006) & (cov.year <= 2015)]),
        ("modern 2016-2023", cov[cov.year >= 2016]),
        ("main-session window 2022-2023", cov[cov.year >= 2022])]
gaps = pd.concat([gap_table(d, lab) for lab, d in ERAS], ignore_index=True)
print(gaps.round(2).to_string(index=False), flush=True)
gaps.to_csv(os.path.join(OUT_DIR, "diag_p0_asof_weekday_weekend_gap.csv"), index=False)

# ---- 2b: TREND-FREE version of the same gap -------------------------------------
# Sample-wide coverage rises secularly (LionShares ramp-up: ~75% in 2000, ~90% in
# 2022), and weekend quarter-ends are not uniformly spread over that trend, so the
# raw weekday-minus-weekend mean above CONFOUNDS the calendar artifact with the
# trend. The calendar artifact is inherently LOCAL, so measure it locally: how far
# does each quarter sit below the mean of its two neighbouring quarters?
rule("2b. GATE 1, TREND-FREE -- each quarter's coverage MINUS the mean of its two "
     "neighbouring quarters")
loc = cov.sort_values(["W", "qend"]).copy()
loc["nb"] = loc.groupby("W")["coverage"].transform(
    lambda s: (s.shift(1) + s.shift(-1)) / 2.0)
loc["local_dev"] = loc.coverage - loc.nb
rows = []
for W in W_GRID:
    s = loc[(loc.W == W) & loc.local_dev.notna()]
    wd = s[s.qend_is_weekend == 0].local_dev
    we = s[s.qend_is_weekend == 1].local_dev
    rows.append(dict(W=W,
                     wd_local_dev_pp=100 * wd.mean(),
                     we_local_dev_pp=100 * we.mean(),
                     artifact_pp=100 * (wd.mean() - we.mean()),
                     we_worst_pp=100 * we.min()))
locdev = pd.DataFrame(rows)
print(locdev.round(2).to_string(index=False), flush=True)
print("""
artifact_pp = the calendar artifact, net of the secular trend. This is the honest
GATE 1 statistic. (Attenuated where two weekend quarter-ends are adjacent, since a
weekend quarter's neighbour is then also depressed -- so it is a LOWER bound on the
old rule's damage and a conservative test of the fix.)""", flush=True)
locdev.to_csv(os.path.join(OUT_DIR, "diag_p0_asof_local_dev.csv"), index=False)

# ------------------------------------------------------------------ 3c: US-share swing
rule("3. GATE 2 -- US SHARE OF CAPTURED FUNDS, weekday vs weekend, by W and by era")
rows = []
for lab, d in ERAS:
    for W in W_GRID:
        s = d[d.W == W]
        wd = s[s.qend_is_weekend == 0].us_share_cap.astype(float)
        we = s[s.qend_is_weekend == 1].us_share_cap.astype(float)
        base_wd = s[s.qend_is_weekend == 0].us_share_q.astype(float)
        base_we = s[s.qend_is_weekend == 1].us_share_q.astype(float)
        # The bias-relevant quantity is not the level of the US share (it trends),
        # but whether SELECTION tilts it: captured minus the full reporting
        # population, contrasted weekday vs weekend.
        sel_wd = 100 * (wd.mean() - base_wd.mean())
        sel_we = 100 * (we.mean() - base_we.mean())
        rows.append(dict(sample=lab, W=W,
                         us_share_wd=100 * wd.mean(), us_share_we=100 * we.mean(),
                         raw_diff_pp=100 * (wd.mean() - we.mean()),
                         sel_tilt_wd_pp=sel_wd, sel_tilt_we_pp=sel_we,
                         selection_tilt_gap_pp=sel_wd - sel_we))
us_tab = pd.DataFrame(rows)
print(us_tab.round(2).to_string(index=False), flush=True)
print("""
HOW TO READ. us_share_* trends hard over 1999-2023, so raw_diff_pp mixes the
calendar artifact with that trend. sel_tilt_* is the clean statistic: captured US
share MINUS the US share of the full set of funds reporting in the same quarter,
i.e. how much the SELECTION RULE itself tilts investor composition. Under an
unbiased rule both tilts are ~0 and selection_tilt_gap_pp ~ 0. That last column is
the GATE 2 number -- the asymmetry group x quarter FE cannot absorb.""", flush=True)
us_tab.to_csv(os.path.join(OUT_DIR, "diag_p0_asof_us_share.csv"), index=False)

# ------------------------------------------------------------------ 3d: staleness cost
rule("4. STALENESS COST OF A WIDE W  (share of captured ROWS coming from gap > 14 days)")
stale = cov.groupby("W").agg(mean_gap_days=("mean_gap", "mean"),
                             median_gap_days=("median_gap", "mean"),
                             rowshare_gap_gt14=("rowshare_gap_gt14", "mean"),
                             mean_coverage=("coverage", "mean"),
                             mean_row_coverage=("row_coverage", "mean"),
                             mean_mv_coverage=("mv_coverage", "mean")).reset_index()
stale["marginal_cov_pp"] = (stale.mean_coverage.diff() * 100).round(2)
print(stale.round(4).to_string(index=False), flush=True)
stale.to_csv(os.path.join(OUT_DIR, "diag_p0_asof_staleness.csv"), index=False)

# ------------------------------------------------------- 3e: multi-report-in-window
rule("5. PER-(fund,fsym) vs PER-fund SELECTION -- how often does a fund report TWICE "
     "inside the window?")
rows = []
for W in W_GRID:
    r = con.execute("""
        SELECT COUNT(*)                                              AS n_fund_quarters,
               SUM(CASE WHEN nd > 1 THEN 1 ELSE 0 END)               AS n_multi,
               SUM(CASE WHEN nd > 1 THEN 1 ELSE 0 END)
                   / NULLIF(CAST(COUNT(*) AS DOUBLE), 0)             AS share_multi
        FROM (SELECT fund_id, qend, COUNT(DISTINCT report_date) AS nd
              FROM fdq WHERE gap <= %d GROUP BY 1,2)
    """ % W).df()
    r["W"] = W
    rows.append(r)
multi = pd.concat(rows, ignore_index=True)[["W", "n_fund_quarters", "n_multi", "share_multi"]]
print(multi.round(5).to_string(index=False), flush=True)
print("\n(If share_multi is ~0, selecting per (fund,fsym) is equivalent to selecting "
      "per fund and the LOCKED design carries no union-of-two-snapshots risk. If it is "
      "large at some W, the per-(fund,fsym) rule silently unions two portfolios.)",
      flush=True)
multi.to_csv(os.path.join(OUT_DIR, "diag_p0_asof_multireport.csv"), index=False)

# ------------------------------------------------------- 3f: the advisor-figure gate
rule("6. DOLLAR GATE -- US investors x 28 EU countries, nominal ADJ_MV totals ($B), "
     "PRE-dedup")
dol = cov[cov.W.isin([0, 3, 7, 10, 14, 31, 92])].pivot_table(
    index="qend", columns="W", values="mv_us_eu_cap_b").reset_index()
dol.columns = [("W%s_b" % c if isinstance(c, (int, float)) else c) for c in dol.columns]
dol = dol.sort_values("qend")
dol["yoy_W0"] = dol["W0_b"] / dol["W0_b"].shift(4) - 1
dol["yoy_W10"] = dol["W10_b"] / dol["W10_b"].shift(4) - 1
_numcols = [c for c in dol.columns if c != "qend"]


def _fmt(d):
    d = d.copy()
    d[_numcols] = d[_numcols].round(3)
    return d.to_string(index=False)


print("--- Q4 rows only (the advisor figure's drawdown quarters) ---", flush=True)
print(_fmt(dol[pd.to_datetime(dol.qend).dt.month == 12]), flush=True)
print("\n--- last 12 quarters, all ---", flush=True)
print(_fmt(dol.tail(12)), flush=True)
dol.to_csv(os.path.join(OUT_DIR, "diag_p0_asof_us_eu_dollars.csv"), index=False)

# ------------------------------------------------- 6b: absurd-MV integrity flag
# Not a P0 question, but the cache makes it free and it lands on the same dollar
# figures the gates are stated in, so it must not go unreported.
rule("6b. DATA INTEGRITY -- fund-date cells with impossible ADJ_MV totals")
out = con.execute("""
    SELECT fund_id, investor_country, report_date,
           mv/1e9 AS mv_b, mv_us_eu/1e9 AS mv_us_eu_b, n_rows
    FROM fdq WHERE mv > 5e12 ORDER BY mv DESC LIMIT 15
""").df()
tot = con.execute("""
    SELECT COUNT(*) AS n_cells, COUNT(DISTINCT fund_id) AS n_funds,
           MIN(report_date) AS first_date, MAX(report_date) AS last_date,
           SUM(mv)/1e9 AS phantom_mv_b, SUM(mv_us_eu)/1e9 AS phantom_mv_us_eu_b
    FROM fdq WHERE mv > 5e12
""").df()
if tot.n_cells[0] > 0:
    print("Cells where ONE fund reports > $5T on ONE date (a plausible ceiling for the "
          "largest fund family is ~$1.5T):", flush=True)
    print(out.round(2).to_string(index=False), flush=True)
    print("", flush=True)
    print(tot.round(2).to_string(index=False), flush=True)
    print("""
This is a PRE-EXISTING data problem, independent of P0, and it is NOT fixed by the
as-of rule. Check phantom_mv_us_eu_b: if it is 0, the US-investor x EU-firm cell the
P0 gates are stated in is clean and the advisor figure is unaffected. Any statistic
built on GLOBAL or NON-US aggregate MV levels is not.""", flush=True)
    out.to_csv(os.path.join(OUT_DIR, "diag_p0_absurd_mv_cells.csv"), index=False)
else:
    print("No fund-date cell exceeds $5T. Clean.", flush=True)

# =============================================================================
# STEP 4 -- duplicate-key magnitude on two benchmark quarters (needs raw scan)
# =============================================================================
if not SKIP_DUP:
    rule("7. ROW_NUMBER DEDUP MAGNITUDE -- duplicate (fund_id, fsym_id, report_date) "
         "keys on 2 benchmark quarters")
    log("scanning the 2022/2023 chunk for the dedup check (a few minutes) ...")
    dup = con.execute("""
        WITH src AS (
            SELECT FACTSET_FUND_ID AS fund_id, FSYM_ID AS fsym_id,
                   CAST(REPORT_DATE AS DATE) AS report_date, ADJ_MV
            FROM read_parquet('%s')
            WHERE ADJ_MV IS NOT NULL AND ADJ_MV > 0
              AND (CAST(REPORT_DATE AS DATE) BETWEEN DATE '2022-12-21' AND DATE '2022-12-31'
                OR CAST(REPORT_DATE AS DATE) BETWEEN DATE '2021-12-21' AND DATE '2021-12-31')
        ), g AS (
            SELECT fund_id, fsym_id, report_date, COUNT(*) AS c, SUM(ADJ_MV) AS mv
            FROM src GROUP BY 1,2,3
        )
        SELECT EXTRACT(YEAR FROM report_date) AS yr,
               COUNT(*)                                  AS n_keys,
               SUM(c)                                    AS n_rows,
               SUM(CASE WHEN c > 1 THEN 1 ELSE 0 END)    AS n_dup_keys,
               SUM(c) - COUNT(*)                         AS n_excess_rows,
               (SUM(c) - COUNT(*)) / NULLIF(CAST(SUM(c) AS DOUBLE),0) AS excess_row_share,
               SUM(mv)/1e9                               AS mv_all_b,
               SUM(CASE WHEN c > 1 THEN mv ELSE 0 END)/1e9 AS mv_on_dup_keys_b
        FROM g GROUP BY 1 ORDER BY 1
    """ % PATTERN).df()
    print(dup.round(4).to_string(index=False), flush=True)
    print("\n(n_excess_rows / excess_row_share = exactly what the new ROW_NUMBER dedup "
          "will REMOVE relative to the current ETL, which sums duplicates. If this is "
          "non-trivial, the post-fix dollar totals are NOT comparable to the pre-fix "
          "ones on the dedup margin alone.)", flush=True)
    dup.to_csv(os.path.join(OUT_DIR, "diag_p0_asof_dupkeys.csv"), index=False)
else:
    log("dedup check skipped (--skip-dupcheck)")

# =============================================================================
# STEP 5 -- recommendation
# =============================================================================
rule("8. RECOMMENDATION")

full = gaps[gaps["sample"] == "FULL 1999-2023"].set_index("W")
modern = gaps[gaps["sample"] == "modern 2016-2023"].set_index("W")
ld = locdev.set_index("W")
st = stale.set_index("W")
mu = multi.set_index("W")
# Candidate W values.  W=0 is the broken status quo; W=92 is the degenerate
# whole-quarter anchor.  Neither is a candidate.
cands = [W for W in W_GRID if W not in (0, 92)]

print("W   mean_cov%  artifact(pp)  raw_gap_full  raw_gap_modern  marginal_cov(pp)  "
      "rowshare(gap>14)  share_multi", flush=True)
for W in W_GRID:
    print("%-4d %8.2f %13.2f %13.2f %15.2f %17s %17.4f %12.5f" % (
        W, full.loc[W, "all_mean"], ld.loc[W, "artifact_pp"], full.loc[W, "gap_pp"],
        modern.loc[W, "gap_pp"],
        ("%.2f" % st.loc[W, "marginal_cov_pp"]) if pd.notna(st.loc[W, "marginal_cov_pp"]) else "-",
        st.loc[W, "rowshare_gap_gt14"], mu.loc[W, "share_multi"]), flush=True)

# Plateau membership, three criteria:
#   (a) the calendar artifact is closed -> |artifact_pp| < 2pp, TREND-FREE (the raw
#       weekday-minus-weekend mean confounds the artifact with the secular ramp-up,
#       so it is the wrong gate; it is printed above for reference only)
#   (b) widening W further buys almost nothing -> gain to next tested W < 1pp
#   (c) W does not start unioning two portfolios per fund -> share_multi < 2%
# NOTE the "next tested W" for the largest candidate is 92 (the whole-quarter anchor),
# NOT None -- otherwise the largest candidate trivially passes (b).
plateau = []
for i, W in enumerate(cands):
    nxt = cands[i + 1] if i + 1 < len(cands) else 92
    gain = (st.loc[nxt, "mean_coverage"] - st.loc[W, "mean_coverage"]) * 100
    plateau.append((W, abs(ld.loc[W, "artifact_pp"]), gain, mu.loc[W, "share_multi"]))

ok = [t for t in plateau if t[1] < 2.0 and t[2] < 1.0 and t[3] < 0.02]
print("", flush=True)
print("plateau criteria: |trend-free artifact| < 2pp, gain to next tested W < 1pp, "
      "share_multi < 2%", flush=True)
for W, g, gain, sm in plateau:
    verdict = "IN " if (g < 2.0 and gain < 1.0 and sm < 0.02) else "out"
    print("  W=%-3d %s  (gap=%.2fpp, gain_to_next=%.2fpp, share_multi=%.4f)"
          % (W, verdict, g, gain, sm), flush=True)
print("", flush=True)
if ok:
    print("Plateau = {%s}" % ", ".join(str(t[0]) for t in ok), flush=True)
    print("Smallest W inside the plateau: W=%d" % ok[0][0], flush=True)
    if 10 in [t[0] for t in ok]:
        print("W=10 IS inside the measured plateau -> the spec default is CONFIRMED.", flush=True)
    else:
        print("W=10 is NOT inside the measured plateau -> the spec default must MOVE. "
              "Candidates: %s" % ", ".join(str(t[0]) for t in ok), flush=True)
else:
    print("NO W in {%s} satisfies all three plateau criteria on the full sample. "
          "Inspect the by-quarter pivot before choosing."
          % ", ".join(str(c) for c in cands), flush=True)

print("""
Read the numbers above, not this sentence: the recommendation is mechanical
(plateau detection), the judgement call belongs in the write-up.""", flush=True)

# Era caveat -- the spec explicitly asked whether the ramp-up years behave differently.
print("\nERA CHECK (spec asked for this explicitly):", flush=True)
for lab, _ in ERAS:
    g = gaps[gaps["sample"] == lab].set_index("W")
    print("  %-30s  raw wd-we gap: W=0 %6.2fpp -> W=10 %6.2fpp   (mean cov W=10: %5.2f%%)"
          % (lab, g.loc[0, "gap_pp"], g.loc[10, "gap_pp"], g.loc[10, "all_mean"]), flush=True)
print("""  If the ramp-up era keeps a large residual gap at W=10 while the modern era does
  not, the as-of rule is NOT the binding constraint there -- early LionShares funds
  report on scattered mid-quarter dates rather than on the prior business day, so no
  small W can recover them. That is a sample-period caveat, not a reason to widen W.""",
      flush=True)

rule("FILES WRITTEN (all read-only w.r.t. the pipeline)")
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith("diag_p0_"):
        print("  output/%s" % f, flush=True)

con.close()
log("DONE")
