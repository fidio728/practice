# presence_sens_buckets.py
# ===========================================================================
# TASK p2 (presence-rule sensitivity, 2026-08-06) — fully isolated side-run.
# Quantifies how Figure A's zero-vs-positive bucketing depends on the Revere
# presence rule (DPN_REVERE_PRESENCE_RULE in 02_china_exposure.jl (4b)).
#
# Design: the presence rule changes ONLY row existence in
# firm_quarter_china_exposure*.parquet (which cells are codable zeros vs NULL).
# Values on surviving cells are rule-invariant (asserted). So the bucket
# classification of build_fig_ab_data.py (ties at t-1: NULL / zero / positive)
# can be re-derived per rule on the SAME canonical matched ownership universe:
#   canonical grid merged_us_eu_zero_filled.parquet (13,103 firms x 100 q)
#   + persisted crosswalk crosswalk_sec_entity_revere.parquet
#   + per-rule exposure parquet -> china_share -> LAG -> bucket at t.
# The canonical-rule replication is validated cell-by-cell against the grid's
# own china_share / china_share_lag1q before any comparison is made.
#
# (MM-FIX CONSUMER FIX, 2026-08-08, review must-fix.) The crosswalk is NO
# LONGER 1:1 on sec_entity_id: after MM-FIX (06_cartesian_grid.jl, 2026-08-06)
# entities with winning-priority ties carry one row per tied eu_company_id
# (n_tied >= 2). The old code here (a) hard-gated n == distinct(sec_entity_id),
# which crashes on any post-MM-FIX crosswalk, and (b) value-picked china_share
# through a 1:1-assuming LEFT JOIN chain, which would silently fan out (fid, q)
# and break the LAG/bucket logic if the gate were removed. Both replaced with
# the documented consumer contract (06 (EM-FIX-9)/(MM-FIX) block): the gate is
# now PAIR-uniqueness on (sec_entity_id, eu_company_id), and per (fid, q) the
# rule-parquet counts are SUMMED across tied eu_company_ids, with
#   cs_rule = SUM(n_cn_customer + n_cn_supplier) / SUM(n_supplychain_links)
# and the EM-CHANGE-2 ELSE-0.0 recode on a present row with zero summed
# denominator — mirroring 06_cartesian_grid.jl (6b) branch B exactly. For
# unique winners (n_tied = 1) the SUM degenerates to the single candidate's
# row, which reproduces branch A bit-for-bit; the script therefore also still
# works on a pre-MM-FIX (1:1) crosswalk vintage.
#
# READS ONLY canonical + *_flsens / *_ficov side artifacts.
# WRITES ONLY new presence_sens_* CSVs under OUT. Zero canonical writes.
# ===========================================================================
from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
_env_out = os.environ.get("DPN_OUT_DIR", "").strip()
OUT = Path(_env_out).resolve() if _env_out else PROJ / "output"

GRID = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
XW   = (OUT / "crosswalk_sec_entity_revere.parquet").as_posix()

RULE_PARQUETS = {
    "record_interval": (OUT / "firm_quarter_china_exposure.parquet").as_posix(),
    "first_link":      (OUT / "firm_quarter_china_exposure_flsens.parquet").as_posix(),
    "record_interval_covered": (OUT / "firm_quarter_china_exposure_ficov.parquet").as_posix(),
}

SPOT_QUARTERS = ("2018-12-31", "2020-03-31", "2022-12-31")

F_SERIES = OUT / "presence_sens_bucket_series.csv"
F_FLIPS  = OUT / "presence_sens_flip_census.csv"
F_SPOT   = OUT / "presence_sens_spot.csv"
F_MOVES  = OUT / "presence_sens_max_moves.csv"


def main() -> None:
    rules = {r: p for r, p in RULE_PARQUETS.items() if Path(p).is_file()}
    missing = [r for r in RULE_PARQUETS if r not in rules]
    if "record_interval" not in rules or "first_link" not in rules:
        raise FileNotFoundError(f"required exposure parquets missing: {missing}")
    print(f"[0] rules available: {list(rules)}; missing (skipped): {missing}")

    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    tmp = Path("E:/duckdb_tmp")
    if tmp.parent.exists():
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{tmp.as_posix()}'")

    # -- US side of the grid (classification is per firm-quarter; the fig A
    #    US-$-share uses the US holder_group book) --------------------------
    con.execute(f"""
        CREATE OR REPLACE TABLE us AS
        SELECT sec_entity_id AS fid, report_date AS q,
               I_ict AS usd_us,
               COALESCE(portfolio_weight_eu, 0.0) AS w_us,
               china_share AS cs_grid, china_share_lag1q AS cn_lag_grid
        FROM read_parquet('{GRID}')
        WHERE holder_group = 'US'
    """)
    chk = con.sql("""SELECT COUNT(*) n, COUNT(DISTINCT fid) f, COUNT(DISTINCT q) q FROM us""").df().iloc[0]
    if int(chk.n) != int(chk.f) * int(chk.q):
        raise RuntimeError("US grid is not a complete cartesian panel — LAG semantics break")
    print(f"[1] US grid: {int(chk.n):,} rows = {int(chk.f):,} firms x {int(chk.q):,} quarters (complete)")

    # (MM-FIX CONSUMER FIX) crosswalk must be unique on the
    # (sec_entity_id, eu_company_id) PAIR — NOT on sec_entity_id alone:
    # post-MM-FIX vintages legitimately carry n_tied >= 2 rows per entity.
    # A duplicate PAIR would double-count that candidate's links in the SUMs.
    xw = con.sql(f"""
        SELECT COUNT(*) AS n,
               COUNT(DISTINCT sec_entity_id) AS n_entities,
               COUNT(*) - COUNT(DISTINCT sec_entity_id) AS n_tied_extra,
               (SELECT COUNT(*) FROM (
                    SELECT sec_entity_id, eu_company_id
                    FROM read_parquet('{XW}')
                    GROUP BY 1, 2 HAVING COUNT(*) > 1)) AS n_dup_pairs
        FROM read_parquet('{XW}')
    """).df().iloc[0]
    if int(xw.n_dup_pairs) != 0:
        raise RuntimeError(
            f"crosswalk has {int(xw.n_dup_pairs)} duplicate (sec_entity_id, eu_company_id) "
            f"pairs — the per-(fid, q) SUM aggregation would double-count links")
    print(f"[1] crosswalk: {int(xw.n):,} (sec_entity_id, eu_company_id) pairs over "
          f"{int(xw.n_entities):,} entities (pair-unique; {int(xw.n_tied_extra):,} extra "
          f"tied rows from MM-FIX)")

    # -- value invariance across rules on surviving cells -------------------
    canon = rules["record_interval"]
    for r, p in rules.items():
        if r == "record_interval":
            continue
        d = con.sql(f"""
            SELECT COUNT(*) AS n_surviving,
                   COUNT(*) FILTER (WHERE a.china_share IS DISTINCT FROM c.china_share) AS n_drift,
                   COUNT(*) FILTER (WHERE c.eu_company_id IS NULL) AS n_extra_rows
            FROM read_parquet('{p}') a
            LEFT JOIN read_parquet('{canon}') c
              ON c.eu_company_id = a.eu_company_id AND c.quarter_end = a.quarter_end
        """).df().iloc[0]
        if int(d.n_drift) or int(d.n_extra_rows):
            raise RuntimeError(f"rule {r}: value drift {int(d.n_drift)} / extra rows {int(d.n_extra_rows)}")
        print(f"[2] {r}: {int(d.n_surviving):,} rows, all value-identical to canonical, strict subset OK")

    # -- per-rule china_share on the grid + lag + bucket ---------------------
    # (MM-FIX CONSUMER FIX) exp_fid implements the 06 consumer contract at the
    # sec_entity grain: presence = JOIN + GROUP BY union of the tied candidates'
    # validity (a (fid, q) row exists iff ANY tied candidate has a rule-parquet
    # row that quarter); values = ratio-from-SUMS with the EM-CHANGE-2 ELSE-0.0
    # recode — verbatim the china_share expression of 06_cartesian_grid.jl (6b)
    # branch B. n_tied = 1 entities degenerate to branch A bit-for-bit.
    for r, p in rules.items():
        con.execute(f"""
            CREATE OR REPLACE TABLE lab_{r} AS
            WITH exp_fid AS (
                SELECT m.sec_entity_id AS fid, e.quarter_end AS q,
                       CASE WHEN SUM(e.n_supplychain_links) > 0
                            THEN (SUM(e.n_cn_customer) + SUM(e.n_cn_supplier))::DOUBLE
                                 / SUM(e.n_supplychain_links)
                            ELSE 0.0 END AS cs_rule_raw
                FROM read_parquet('{XW}') m
                JOIN read_parquet('{p}') e ON e.eu_company_id = m.eu_company_id
                GROUP BY 1, 2
            ),
            cs AS (
                SELECT u.fid, u.q, u.usd_us, u.w_us, u.cs_grid, u.cn_lag_grid,
                       CASE WHEN u.q < DATE '2003-03-31' THEN NULL
                            ELSE x.cs_rule_raw END AS cs_rule
                FROM us u
                LEFT JOIN exp_fid x ON x.fid = u.fid AND x.q = u.q
            )
            SELECT fid, q, usd_us, w_us, cs_grid, cn_lag_grid, cs_rule,
                   LAG(cs_rule, 1) OVER (PARTITION BY fid ORDER BY q) AS cn_lag_rule,
                   CASE WHEN LAG(cs_rule, 1) OVER (PARTITION BY fid ORDER BY q) IS NULL THEN 'NULL_tie'
                        WHEN LAG(cs_rule, 1) OVER (PARTITION BY fid ORDER BY q) > 0    THEN 'positive_tie'
                        ELSE 'zero_tie' END AS bucket
            FROM cs
        """)

    # -- validation: canonical replication must equal the grid's own columns --
    v = con.sql("""
        SELECT COUNT(*) FILTER (WHERE cs_rule    IS DISTINCT FROM cs_grid)     AS bad_cs,
               COUNT(*) FILTER (WHERE cn_lag_rule IS DISTINCT FROM cn_lag_grid) AS bad_lag
        FROM lab_record_interval
    """).df().iloc[0]
    if int(v.bad_cs) or int(v.bad_lag):
        raise RuntimeError(f"canonical replication FAILED: cs mismatches={int(v.bad_cs)}, lag mismatches={int(v.bad_lag)}")
    print("[3] canonical replication: cs_rule == grid.china_share and lag == china_share_lag1q on every cell OK")

    # -- bucket series per rule ---------------------------------------------
    frames = []
    for r in rules:
        a = con.sql(f"""
            SELECT '{r}' AS rule, q AS quarter_end, bucket,
                   COUNT(*) AS n_firms,
                   SUM(w_us) AS share_of_book,
                   SUM(usd_us) AS usd_us
            FROM lab_{r} GROUP BY 1,2,3 ORDER BY 2,3
        """).df()
        frames.append(a)
    series = pd.concat(frames, ignore_index=True)
    # firm-count share among CLASSIFIABLE (zero+positive) firms
    cls_tot = (series.loc[series.bucket.ne("NULL_tie")]
                     .groupby(["rule", "quarter_end"])["n_firms"].sum()
                     .rename("n_classifiable").reset_index())
    series = series.merge(cls_tot, on=["rule", "quarter_end"], how="left")
    series["firm_share_of_classifiable"] = series.n_firms / series.n_classifiable.where(series.n_classifiable > 0)
    series.to_csv(F_SERIES, index=False)
    print(f"[4] wrote {F_SERIES.name} ({len(series):,} rows)")

    # -- flip census vs canonical -------------------------------------------
    flips = []
    for r in rules:
        if r == "record_interval":
            continue
        f = con.sql(f"""
            SELECT c.bucket AS bucket_canonical, a.bucket AS bucket_{r},
                   COUNT(*) AS n_firm_quarters,
                   COUNT(DISTINCT c.fid) AS n_firms
            FROM lab_record_interval c
            JOIN lab_{r} a ON a.fid = c.fid AND a.q = c.q
            GROUP BY 1,2 ORDER BY 1,2
        """).df()
        f = f.rename(columns={f"bucket_{r}": "bucket_alt"})
        f["rule"] = r
        flips.append(f)
    flips = pd.concat(flips, ignore_index=True)
    flips.to_csv(F_FLIPS, index=False)
    print(f"[5] wrote {F_FLIPS.name}")
    print(flips.to_string(index=False))

    # -- spot table ----------------------------------------------------------
    spot = series.loc[series.quarter_end.astype(str).isin(SPOT_QUARTERS)].copy()
    spot = spot.sort_values(["quarter_end", "rule", "bucket"])
    spot.to_csv(F_SPOT, index=False)
    print(f"[6] wrote {F_SPOT.name}")

    # -- max series moves vs canonical (pp) ---------------------------------
    base = series.loc[series.rule.eq("record_interval"),
                      ["quarter_end", "bucket", "share_of_book", "firm_share_of_classifiable"]]
    moves = []
    for r in rules:
        if r == "record_interval":
            continue
        alt = series.loc[series.rule.eq(r),
                         ["quarter_end", "bucket", "share_of_book", "firm_share_of_classifiable"]]
        m = base.merge(alt, on=["quarter_end", "bucket"], suffixes=("_canon", "_alt"), how="outer")
        for c in ("share_of_book_canon", "share_of_book_alt",
                  "firm_share_of_classifiable_canon", "firm_share_of_classifiable_alt"):
            m[c] = m[c].fillna(0.0)
        m["d_share_of_book_pp"] = 100.0 * (m.share_of_book_alt - m.share_of_book_canon)
        m["d_firm_share_pp"] = 100.0 * (m.firm_share_of_classifiable_alt - m.firm_share_of_classifiable_canon)
        m["rule"] = r
        moves.append(m)
    moves = pd.concat(moves, ignore_index=True)
    moves.to_csv(F_MOVES, index=False)

    print("\n=== MAX ABS MOVES vs canonical (percentage points) ===")
    summ = (moves.groupby(["rule", "bucket"])
                 .agg(max_abs_d_share_of_book_pp=("d_share_of_book_pp", lambda s: s.abs().max()),
                      max_abs_d_firm_share_pp=("d_firm_share_pp", lambda s: s.abs().max()))
                 .reset_index())
    print(summ.to_string(index=False))
    summ.to_csv(OUT / "presence_sens_max_moves_summary.csv", index=False)

    print("\n=== SPOT (US book) ===")
    cols = ["quarter_end", "rule", "bucket", "n_firms", "share_of_book", "firm_share_of_classifiable"]
    print(spot[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    con.close()


if __name__ == "__main__":
    sys.exit(main())
