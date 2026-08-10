# robustness_dq_filter_variants.py
# ===========================================================================
# FOUR-VERSION DQ-FILTER ROBUSTNESS for the COUNTRY-LEVEL aggregates
# (external-reviewer request, 2026-08-10).
#
# The shipped pipeline (04_us_ownership_european.jl, "DQ-FIX" block) drops
# provably-impossible EQ/AD holding rows before building I_ict. The reviewer's
# objection: the shipped rule leans on the INLINE MARKET-CAP PROXY (itself
# built from the same feed), so a proxy error could manufacture the filter's
# own justification. This script rebuilds the country-level aggregates — the
# inputs to fig_A_country_buckets, the country ranking by US-held value
# (Step-4 style), and the per-(investor_country, quarter) totals that feed the
# Step-4 country panel — under FOUR regimes, computed ON THE FLY from
# holdings_eom.parquet. Nothing canonical is touched.
#
# FILTER REGIMES (drop condition, applied to the identified EQ/AD universe:
# sec_entity_id IS NOT NULL AND investor_country IS NOT NULL AND
# issue_type IN ('EQ','AD') — the same universe I_ict is built on; the
# NULL-entity / non-EQ sentinel families never enter ANY version, exactly as
# they never enter I_ict):
#   V0  no DQ filter. Drop nothing.
#   V1  SHIPPED = canonical (04 "DQ-FIX", 2026-08-09):
#           adj_mv > $5bn AND ( adj_mv > inline market cap
#                               OR adj_holding > adj_shares_out )
#       where the inline market cap replicates 04's dq_impossible mc CTE
#       verbatim: primary-EQ share classes (fsym_id = fsym_primary_id,
#       issue_type = 'EQ', shares_out > 0, price > 0), AVG within
#       (sec_entity_id, report_date, fsym_id), SUM across classes — WITHOUT
#       the EM-FIX-2 freshest-gap restriction, exactly like the shipped rule.
#   V2  PROXY-FREE (the version the reviewer demanded): ALL rows with
#           adj_holding > adj_shares_out AND adj_shares_out > 0
#       regardless of size. No market-cap proxy anywhere. This also sweeps in
#       the documented v3.1 RESIDUAL (373 sub-threshold rows / $47.0bn /
#       196 holder-country-quarters; ID 2017Q4 = 65.9% of that
#       country-quarter) that V1 deliberately leaves in.
#   V3  V2 PLUS sentinel-price zero-holding rows:
#           adj_price >= 999999 AND adj_holding = 0
#
# NULL semantics: every comparison is wrapped in COALESCE(..., FALSE), so a
# row with NULL adj_holding / adj_shares_out / adj_price is KEPT in every
# version — identical to the shipped anti-join, where a NULL predicate keeps
# the row out of dq_impossible.
#
# OUTPUTS (all in output/dq_variants/ — new dir, no rotation needed):
#   dq_census.csv
#       version x year: rows / $ dropped. V1 must reproduce the 04 census
#       (2026-08-09: exactly 3 rows, $452.6bn); a drift is printed loudly.
#   dq_country_quarter_deltas_vs_v1.csv
#       version x investor_country x quarter-end:
#         usd_global        SUM(adj_mv) over ALL sec_country  (= the
#                           country_total_holdings_global construction)
#         usd_eu            SUM(adj_mv) over EU sec_country   (= the
#                           country_total_holdings_eu construction)
#         n_rows_kept / n_rows_dropped
#         *_v1, delta_*_vs_v1, delta_pct_*_vs_v1 (per family)
#   dq_country_ranking_us_held.csv
#       version x {2018Q4, 2022Q4} x EU sec_country: US-held value
#       (Step-4 / 04_us_own_by_eu_country_snapshot construction: investor_
#       country = 'US', EU sec_country, NO grid-universe restriction),
#       rank within (version, quarter), plus V1 value/rank deltas.
#   dq_fig_A_country_bucket_shares.csv
#       version x measure (M1/M2/M3) x spot quarter x bucket x holder_group:
#         usd_value               bucket total (grid-universe-restricted
#                                 numerator, mirroring 06's ict_grouped
#                                 patch 3: firms limited to the canonical
#                                 universe, country taken from the GRID, not
#                                 from holdings_eom)
#         share_of_book_global    usd_value / holder group's GLOBAL book
#         share_of_book_eu        usd_value / holder group's EU book
#         (both denominators recomputed under the SAME regime)
#         plus V1 deltas in percentage points.
#       Bucket membership is read from the canonical
#       fig_A_country_bucket_membership.csv (classification is on Revere
#       links at t-1 — holdings-DQ regimes do not move it), so exactly the
#       same countries sit in each bucket across versions: the deltas isolate
#       the FILTER, not reclassification.
#
# BUILT-IN CHECK: under V1 the bucket shares are, by construction, the same
# numbers as fig_A_country_buckets.csv (share_of_book_global family, US
# rows); the script prints the max abs diff when that CSV is available and
# already carries the global columns.
#
# READ-ONLY on canonical inputs (holdings_eom.parquet,
# merged_us_eu_zero_filled.parquet, fig_A_country_bucket_membership.csv,
# fig_A_country_buckets.csv). Writes ONLY under output/dq_variants/.
# Runtime: a few full scans of the 186.8M-row EOM panel, ~5-10 min.
# ===========================================================================

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
_env_out = os.environ.get("DPN_OUT_DIR", "").strip()
OUT = Path(_env_out).resolve() if _env_out else PROJ / "output"

EOM    = (OUT / "holdings_eom.parquet").as_posix()
GRID   = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
F_MEMB = OUT / "fig_A_country_bucket_membership.csv"
F_CTRY = OUT / "fig_A_country_buckets.csv"          # optional, self-check only

OUT_DQ = OUT / "dq_variants"

F_CENSUS  = OUT_DQ / "dq_census.csv"
F_DELTAS  = OUT_DQ / "dq_country_quarter_deltas_vs_v1.csv"
F_RANKING = OUT_DQ / "dq_country_ranking_us_held.csv"
F_BUCKETS = OUT_DQ / "dq_fig_A_country_bucket_shares.csv"

# EU country list, verbatim from 00_setup.jl (single source of truth there;
# copied because this is a Python consumer of a Julia constant).
EU_COUNTRIES = ("GB","DE","FR","NL","CH","IT","ES","SE","DK","NO","FI",
                "BE","AT","IE","LU","PT","PL","CZ","HU","GR","RO","SK",
                "SI","BG","HR","EE","LV","LT")
EU_SQL = "(" + ",".join(f"'{c}'" for c in EU_COUNTRIES) + ")"

RANK_QUARTERS   = ("2018-12-31", "2022-12-31")
BUCKET_QUARTERS = ("2018-12-31", "2020-03-31", "2022-12-31")   # fig_ab spots

VERSIONS = ("v0", "v1", "v2", "v3")


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='12GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    tmp = Path("E:/duckdb_tmp")
    if tmp.parent.exists():
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    return con


def build_base_view(con: duckdb.DuckDBPyConnection) -> None:
    """One view over the identified EQ/AD universe with NULL-safe drop flags.

    mc replicates 04's dq_impossible inline market cap VERBATIM (primary-EQ
    classes, AVG within class, SUM across classes, no EM-FIX-2 restriction).
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE mc AS
        SELECT cls.sec_entity_id, cls.report_date,
               SUM(cls.shares_out * cls.price) AS market_cap
        FROM (
            SELECT sec_entity_id, report_date, fsym_id,
                   AVG(adj_shares_out) AS shares_out, AVG(adj_price) AS price
            FROM read_parquet('{EOM}')
            WHERE fsym_id = fsym_primary_id AND issue_type = 'EQ'
              AND adj_shares_out > 0 AND adj_price > 0
            GROUP BY 1, 2, 3
        ) cls
        GROUP BY 1, 2
    """)
    n_mc = con.sql("SELECT COUNT(*) AS n FROM mc").df()["n"].iloc[0]
    print(f"[0] inline market-cap proxy: {int(n_mc):,} (sec_entity_id, report_date) cells")

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW base AS
        SELECT h.sec_entity_id, h.sec_country, h.investor_country,
               h.report_date, h.adj_mv,
               FALSE AS drop_v0,
               -- V1 = shipped 04 DQ-FIX rule, NULL-safe
               (COALESCE(h.adj_mv > 5e9, FALSE) AND (
                    COALESCE(mc.market_cap IS NOT NULL
                             AND h.adj_mv > mc.market_cap, FALSE)
                 OR COALESCE(h.adj_holding > h.adj_shares_out
                             AND h.adj_shares_out > 0, FALSE)
               )) AS drop_v1,
               -- V2 = proxy-free impossibility, any size
               COALESCE(h.adj_holding > h.adj_shares_out
                        AND h.adj_shares_out > 0, FALSE) AS drop_v2,
               -- V3 = V2 + sentinel-price zero-holding rows
               (COALESCE(h.adj_holding > h.adj_shares_out
                         AND h.adj_shares_out > 0, FALSE)
                OR COALESCE(h.adj_price >= 999999
                            AND h.adj_holding = 0, FALSE)) AS drop_v3
        FROM read_parquet('{EOM}') h
        LEFT JOIN mc ON mc.sec_entity_id = h.sec_entity_id
                    AND mc.report_date   = h.report_date
        WHERE h.sec_entity_id IS NOT NULL
          AND h.investor_country IS NOT NULL
          AND h.issue_type IN ('EQ', 'AD')
    """)


def per_version_sums(expr: str = "adj_mv") -> str:
    """SQL fragment: kept-dollar sums + drop counts for every version."""
    parts = []
    for v in VERSIONS:
        parts.append(
            f"COALESCE(SUM({expr}) FILTER (WHERE NOT drop_{v}), 0) AS usd_{v}")
    return ",\n               ".join(parts)


# ---------------------------------------------------------------------------
def census(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per version x year: rows and dollars DROPPED (full panel, all dates —
    comparable to 04's own dq census)."""
    cols = ",\n               ".join(
        f"COUNT(*) FILTER (WHERE drop_{v}) AS n_drop_{v},\n"
        f"               COALESCE(SUM(adj_mv) FILTER (WHERE drop_{v}), 0) AS mv_drop_{v}"
        for v in VERSIONS)
    wide = con.sql(f"""
        SELECT year(report_date) AS year,
               COUNT(*) AS n_rows_universe,
               {cols}
        FROM base GROUP BY 1 ORDER BY 1
    """).df()
    rows = []
    for v in VERSIONS:
        for _, r in wide.iterrows():
            rows.append({"version": v.upper(), "year": int(r["year"]),
                         "n_rows_universe": int(r["n_rows_universe"]),
                         "n_rows_dropped": int(r[f"n_drop_{v}"]),
                         "usd_dropped": float(r[f"mv_drop_{v}"])})
    cen = pd.DataFrame(rows)          # all version x year cells kept, zeros too
    tot = (cen.groupby("version", as_index=False)
              .agg(n_rows_dropped=("n_rows_dropped", "sum"),
                   usd_dropped=("usd_dropped", "sum")))
    print("\n[1] drop census (totals per version):")
    print(tot.to_string(index=False,
                        float_format=lambda x: f"{x/1e9:,.1f}bn"))
    v1_n = int(tot.loc[tot["version"].eq("V1"), "n_rows_dropped"].iloc[0])
    if v1_n != 3:
        # (r1 advisory A2) HARD ABORT, not a printed warning: every downstream
        # CSV is a delta AGAINST the V1 baseline, so a drifted V1 poisons the
        # whole four-version comparison.
        raise RuntimeError(
            f"V1 drops {v1_n} rows, not the 3 of the 2026-08-09 04 census — "
            "the raw feed (or the replica) changed regime. Refusing to write "
            "any variant tables against a drifted V1 baseline.")
    print("    V1 reproduces the 04 census: 3 rows.")
    return cen


# ---------------------------------------------------------------------------
def country_quarter_totals(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per (investor_country, quarter-end): kept dollars under each version,
    global and EU-restricted — the country_total_ct construction, four ways.
    Restricted to quarter-end stamped dates (the grid's frequency)."""
    g_sums = per_version_sums("adj_mv").replace("usd_", "usd_global_")
    e_sums = per_version_sums(
        f"CASE WHEN sec_country IN {EU_SQL} THEN adj_mv END"
    ).replace("usd_", "usd_eu_")
    n_cols = ",\n               ".join(
        f"COUNT(*) FILTER (WHERE NOT drop_{v}) AS n_kept_{v}" for v in VERSIONS)
    wide = con.sql(f"""
        SELECT investor_country, report_date AS quarter_end,
               COUNT(*) AS n_rows_universe,
               {g_sums},
               {e_sums},
               {n_cols}
        FROM base
        WHERE report_date = last_day(report_date)
          AND month(report_date) IN (3, 6, 9, 12)
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()
    wide["quarter_end"] = pd.to_datetime(wide["quarter_end"])
    print(f"\n[2] country-quarter totals: {len(wide):,} (investor_country, quarter) "
          f"cells, {wide['investor_country'].nunique()} holder countries, "
          f"{wide['quarter_end'].nunique()} quarters")
    return wide


def write_deltas(wide: pd.DataFrame) -> pd.DataFrame:
    keys = ["investor_country", "quarter_end"]
    long_parts = []
    for v in VERSIONS:
        part = wide[keys + ["n_rows_universe", f"usd_global_{v}", f"usd_eu_{v}",
                            f"n_kept_{v}"]].copy()
        part.columns = keys + ["n_rows_universe", "usd_global", "usd_eu", "n_rows_kept"]
        part["version"] = v.upper()
        long_parts.append(part)
    long = pd.concat(long_parts, ignore_index=True)
    long["n_rows_dropped"] = long["n_rows_universe"] - long["n_rows_kept"]

    v1 = (long.loc[long["version"].eq("V1"), keys + ["usd_global", "usd_eu"]]
              .rename(columns={"usd_global": "usd_global_v1", "usd_eu": "usd_eu_v1"}))
    d = long.merge(v1, on=keys, how="left", validate="many_to_one")
    d["delta_usd_global_vs_v1"] = d["usd_global"] - d["usd_global_v1"]
    d["delta_pct_global_vs_v1"] = 100.0 * d["delta_usd_global_vs_v1"] / \
        d["usd_global_v1"].where(d["usd_global_v1"] > 0)
    d["delta_usd_eu_vs_v1"] = d["usd_eu"] - d["usd_eu_v1"]
    d["delta_pct_eu_vs_v1"] = 100.0 * d["delta_usd_eu_vs_v1"] / \
        d["usd_eu_v1"].where(d["usd_eu_v1"] > 0)
    d = d[["version"] + keys +
          ["usd_global", "usd_eu", "n_rows_kept", "n_rows_dropped",
           "usd_global_v1", "delta_usd_global_vs_v1", "delta_pct_global_vs_v1",
           "usd_eu_v1", "delta_usd_eu_vs_v1", "delta_pct_eu_vs_v1"]]
    d = d.sort_values(["version"] + keys).reset_index(drop=True)
    d.to_csv(F_DELTAS, index=False)

    moved = d.loc[d["version"].ne("V1") & (d["delta_usd_global_vs_v1"].abs() > 0)]
    print(f"[2] wrote {F_DELTAS.name} ({len(d):,} rows); "
          f"{len(moved):,} non-V1 cells where the global total moves at all")
    top = moved.reindex(moved["delta_usd_global_vs_v1"].abs()
                        .sort_values(ascending=False).index).head(15)
    if len(top):
        print("    largest movers (any version vs V1):")
        print(top[["version", "investor_country", "quarter_end",
                   "delta_usd_global_vs_v1", "delta_pct_global_vs_v1"]]
              .to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    return d


# ---------------------------------------------------------------------------
def ranking(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Country ranking by US-held value at the two rank quarters, per version.
    Step-4 construction: investor_country='US', EU sec_country, NO universe
    restriction (mirrors 04_us_own_by_eu_country_snapshot / I_ict)."""
    dates = ", ".join(f"DATE '{q}'" for q in RANK_QUARTERS)
    wide = con.sql(f"""
        SELECT sec_country, report_date AS quarter_end,
               {per_version_sums('adj_mv')}
        FROM base
        WHERE investor_country = 'US'
          AND sec_country IN {EU_SQL}
          AND report_date IN ({dates})
        GROUP BY 1, 2
    """).df()
    wide["quarter_end"] = pd.to_datetime(wide["quarter_end"])

    parts = []
    for v in VERSIONS:
        part = wide[["sec_country", "quarter_end", f"usd_{v}"]].copy()
        part.columns = ["sec_country", "quarter_end", "us_held_usd"]
        part["version"] = v.upper()
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    long["rank"] = (long.groupby(["version", "quarter_end"])["us_held_usd"]
                        .rank(ascending=False, method="min").astype(int))

    v1 = (long.loc[long["version"].eq("V1"),
                   ["sec_country", "quarter_end", "us_held_usd", "rank"]]
              .rename(columns={"us_held_usd": "us_held_usd_v1", "rank": "rank_v1"}))
    r = long.merge(v1, on=["sec_country", "quarter_end"], how="left",
                   validate="many_to_one")
    r["delta_usd_vs_v1"] = r["us_held_usd"] - r["us_held_usd_v1"]
    r["delta_pct_vs_v1"] = 100.0 * r["delta_usd_vs_v1"] / \
        r["us_held_usd_v1"].where(r["us_held_usd_v1"] > 0)
    r["rank_change_vs_v1"] = r["rank_v1"] - r["rank"]
    r = (r[["version", "quarter_end", "sec_country", "us_held_usd", "rank",
            "us_held_usd_v1", "rank_v1", "delta_usd_vs_v1", "delta_pct_vs_v1",
            "rank_change_vs_v1"]]
         .sort_values(["version", "quarter_end", "rank"])
         .reset_index(drop=True))
    r.to_csv(F_RANKING, index=False)
    n_moves = int((r["rank_change_vs_v1"] != 0).sum())
    print(f"\n[3] wrote {F_RANKING.name} ({len(r):,} rows); "
          f"{n_moves} country-quarter rank positions differ from V1")
    for q in RANK_QUARTERS:
        t = r.loc[r["quarter_end"].eq(pd.Timestamp(q))
                  & r["version"].isin(["V0", "V2"]) & r["rank"].le(10)]
        print(f"    top-10 under V0/V2 at {q} (rank_change_vs_v1 != 0 rows flag moves):")
        print(t[["version", "sec_country", "us_held_usd", "rank", "rank_change_vs_v1"]]
              .to_string(index=False, float_format=lambda x: f"{x/1e9:,.1f}bn"))
    return r


# ---------------------------------------------------------------------------
def bucket_shares(con: duckdb.DuckDBPyConnection,
                  totals_wide: pd.DataFrame) -> pd.DataFrame:
    """fig_A country bucket shares at the spot quarters, per version.

    Numerator mirrors 06's ict_grouped (patch 3): firms restricted to the
    canonical grid universe, country taken from the GRID. Denominators are the
    holder group's global / EU book recomputed under the same regime (the
    country_total_grouped construction). Membership comes verbatim from the
    canonical fig_A_country_bucket_membership.csv — classification is on
    Revere links, which no holdings-DQ regime touches."""
    if not F_MEMB.is_file():
        raise FileNotFoundError(
            f"{F_MEMB} not found — run build_fig_ab_data.py first.")
    memb = pd.read_csv(F_MEMB, parse_dates=["quarter_end"])
    need = {"measure", "country", "quarter_end", "bucket"}
    if not need.issubset(memb.columns):
        raise RuntimeError(f"{F_MEMB.name} lacks columns {need - set(memb.columns)}")
    spots = [pd.Timestamp(q) for q in BUCKET_QUARTERS]
    memb = memb.loc[memb["quarter_end"].isin(spots),
                    ["measure", "quarter_end", "country", "bucket"]]

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE grid_univ AS
        SELECT DISTINCT sec_entity_id, sec_country
        FROM read_parquet('{GRID}')
    """)
    dup = con.sql("""
        SELECT COUNT(*) AS n FROM (
            SELECT sec_entity_id FROM grid_univ GROUP BY 1 HAVING COUNT(*) > 1)
    """).df()["n"].iloc[0]
    if int(dup) > 0:
        raise RuntimeError(
            f"{int(dup)} sec_entity_id with >1 sec_country in the grid universe "
            "— the canonical-universe uniqueness assert of 06 no longer holds.")

    dates = ", ".join(f"DATE '{q}'" for q in BUCKET_QUARTERS)
    cw = con.sql(f"""
        SELECT g.sec_country AS country, b.report_date AS quarter_end,
               CASE WHEN b.investor_country = 'US' THEN 'US' ELSE 'NONUS' END
                   AS holder_group,
               {per_version_sums('b.adj_mv')}
        FROM base b
        JOIN grid_univ g USING (sec_entity_id)
        WHERE b.report_date IN ({dates})
        GROUP BY 1, 2, 3
    """).df()
    cw["quarter_end"] = pd.to_datetime(cw["quarter_end"])

    # holder-group books (denominators) from the country-quarter totals
    tw = totals_wide.loc[totals_wide["quarter_end"].isin(spots)].copy()
    tw["holder_group"] = np.where(tw["investor_country"].eq("US"), "US", "NONUS")
    den_rows = []
    for v in VERSIONS:
        g = (tw.groupby(["holder_group", "quarter_end"], as_index=False)
               .agg(book_global=(f"usd_global_{v}", "sum"),
                    book_eu=(f"usd_eu_{v}", "sum")))
        g["version"] = v.upper()
        den_rows.append(g)
    den = pd.concat(den_rows, ignore_index=True)

    # long country-level numerators
    parts = []
    for v in VERSIONS:
        p = cw[["country", "quarter_end", "holder_group", f"usd_{v}"]].copy()
        p.columns = ["country", "quarter_end", "holder_group", "usd"]
        p["version"] = v.upper()
        parts.append(p)
    cl = pd.concat(parts, ignore_index=True)

    j = memb.merge(cl, on=["country", "quarter_end"], how="left")
    j["usd"] = j["usd"].fillna(0.0)
    agg = (j.groupby(["version", "measure", "quarter_end", "bucket",
                      "holder_group"], dropna=True, as_index=False)
             .agg(usd_value=("usd", "sum"),
                  n_countries=("country", "nunique")))
    agg = agg.merge(den, on=["version", "holder_group", "quarter_end"],
                    how="left", validate="many_to_one")
    agg["share_of_book_global"] = agg["usd_value"] / \
        agg["book_global"].where(agg["book_global"] > 0)
    agg["share_of_book_eu"] = agg["usd_value"] / \
        agg["book_eu"].where(agg["book_eu"] > 0)

    keys = ["measure", "quarter_end", "bucket", "holder_group"]
    v1 = (agg.loc[agg["version"].eq("V1"),
                  keys + ["usd_value", "share_of_book_global", "share_of_book_eu"]]
             .rename(columns={"usd_value": "usd_value_v1",
                              "share_of_book_global": "share_global_v1",
                              "share_of_book_eu": "share_eu_v1"}))
    out = agg.merge(v1, on=keys, how="left", validate="many_to_one")
    out["delta_usd_vs_v1"] = out["usd_value"] - out["usd_value_v1"]
    out["delta_share_global_vs_v1_pp"] = 100.0 * (out["share_of_book_global"]
                                                  - out["share_global_v1"])
    out["delta_share_eu_vs_v1_pp"] = 100.0 * (out["share_of_book_eu"]
                                              - out["share_eu_v1"])
    out = (out[["version"] + keys +
               ["n_countries", "usd_value", "book_global", "book_eu",
                "share_of_book_global", "share_of_book_eu",
                "usd_value_v1", "delta_usd_vs_v1",
                "delta_share_global_vs_v1_pp", "delta_share_eu_vs_v1_pp"]]
           .sort_values(["version"] + keys)
           .reset_index(drop=True))
    # ---- self-check FIRST, write SECOND (r1 advisory A2): a drifted V1 must
    # ---- never leave a plausible-looking four-version table on disk --------
    if F_CTRY.is_file():
        ship = pd.read_csv(F_CTRY, parse_dates=["quarter_end"])
        if "share_of_book_global" in ship.columns and "measure" in ship.columns:
            s = ship.loc[ship["quarter_end"].isin(spots),
                         ["measure", "quarter_end", "bucket", "holder_group",
                          "share_of_book_global"]]
            chk = out.loc[out["version"].eq("V1")].merge(
                s, on=keys, how="inner", suffixes=("", "_ship"))
            if len(chk):
                diff = (chk["share_of_book_global"]
                        - chk["share_of_book_global_ship"]).abs().max()
                print(f"    V1 vs shipped fig_A_country_buckets.csv "
                      f"(share_of_book_global, {len(chk)} cells): "
                      f"max abs diff {diff:.3e}")
                if not diff < 1e-9:
                    # (r1 advisory A2) HARD ABORT: V1 failing to reproduce the
                    # shipped figure CSV means a vintage drifted somewhere —
                    # the deltas already written above would compare regimes
                    # across vintages, which is meaningless.
                    raise RuntimeError(
                        f"V1 does not reproduce the shipped "
                        f"fig_A_country_buckets.csv (max abs diff {diff:.3e} "
                        f">= 1e-9) — vintage drift; refusing to deliver the "
                        f"four-version tables.")
                print("    [OK]")
        else:
            print("    (self-check skipped: shipped fig_A_country_buckets.csv "
                  "predates the global-denominator switch — re-run "
                  "build_fig_ab_data.py to enable it)")

    out.to_csv(F_BUCKETS, index=False)
    print(f"\n[4] wrote {F_BUCKETS.name} ({len(out):,} rows)")
    mx = out.loc[out["version"].ne("V1"),
                 ["delta_share_global_vs_v1_pp", "delta_share_eu_vs_v1_pp"]].abs().max()
    print(f"    max |share move| vs V1: global {mx.iloc[0]:.4f} pp, "
          f"eu {mx.iloc[1]:.4f} pp")
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    for p in (Path(EOM), Path(GRID), F_MEMB):
        if not Path(p).is_file():
            raise FileNotFoundError(f"missing canonical input: {p}")
    OUT_DQ.mkdir(parents=True, exist_ok=True)

    con = connect()
    build_base_view(con)

    cen = census(con)
    cen.to_csv(F_CENSUS, index=False)
    print(f"[1] wrote {F_CENSUS.name} ({len(cen):,} rows)")

    totals_wide = country_quarter_totals(con)
    write_deltas(totals_wide)
    ranking(con)
    bucket_shares(con, totals_wide)

    con.close()
    print("\nDONE — all outputs under", OUT_DQ)


if __name__ == "__main__":
    sys.exit(main())
