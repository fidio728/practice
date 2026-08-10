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
#       version x EVERY quarter-end x EU sec_country: US-held value
#       (Step-4 / 04_us_own_by_eu_country_snapshot construction: investor_
#       country = 'US', EU sec_country, NO grid-universe restriction),
#       rank within (version, quarter), plus V1 value/rank deltas.
#       (r3 item 1) This used to run on the two spot quarters 2018Q4 / 2022Q4
#       only. Spot-checking rankings at two dates cannot support the claim
#       "the filter does not move the rankings", because the two largest
#       country-quarter dollar anomalies sit at 2017Q4 (Indonesia, -65.9%
#       under V2/V3) and 2020Q4 (Canada, +9.74% under V0) — neither was in
#       the checked set. The comparison now covers every quarter.
#   dq_country_ranking_us_held_SPOT_SUBSET.csv
#       the old two-quarter table, kept verbatim for continuity. SUBSET of
#       the all-quarter file above; not evidence on its own.
#   dq_ranking_quarter_summary.csv
#       version x quarter: n rank positions moved vs V1, max |rank change|
#       and its country, max |delta %| and its country, and whether the
#       ordered / unordered top-5 differs from V1. This is the table the
#       "rankings do not move" claim must be read off.
#   dq_fig_A_country_bucket_shares.csv
#       version x measure (M1/M2/M3) x EVERY membership quarter x bucket x
#       holder_group:
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
#       (r3 item 1) This too used to run on three spot quarters only
#       (2018Q4 / 2020Q1 / 2022Q4), none of which is an anomaly quarter. The
#       old "max share move 0.0010pp / 0.0022pp" number was therefore a
#       three-quarter number, not a full-sample number. It now runs on every
#       quarter present in the membership file.
#   dq_fig_A_country_bucket_shares_SPOT_SUBSET.csv
#       the old three-quarter table, kept verbatim for continuity. SUBSET.
#   dq_bucket_share_quarter_summary.csv
#       version x measure x quarter: max |share move vs V1| on the global and
#       on the EU denominator, with the argmax (bucket, holder_group) for
#       each. The full-sample maxima and their argmax quarters are printed
#       and are what the write-up must quote.
#   dq_named_quarter_rows.csv
#       the anomaly quarters 2017Q4 and 2020Q4, reported explicitly and
#       unconditionally in both families, so they can never again fall
#       outside the checked set. A named quarter that is not on a family's
#       calendar still gets a row, carrying present=0; absence is reported,
#       never silently omitted.
#
# BUILT-IN CHECK: under V1 the bucket shares are, by construction, the same
# numbers as fig_A_country_buckets.csv (share_of_book_global family, US
# rows); the script prints the max abs diff when that CSV is available and
# already carries the global columns. With the all-quarter extension this
# check now covers every shared quarter, not 54 spot cells.
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
# (r3 item 1) all-quarter extension: per-quarter summaries + the retained
# spot subsets + the unconditional named-quarter rows.
F_RANKING_SPOT = OUT_DQ / "dq_country_ranking_us_held_SPOT_SUBSET.csv"
F_BUCKETS_SPOT = OUT_DQ / "dq_fig_A_country_bucket_shares_SPOT_SUBSET.csv"
F_RANK_QSUM    = OUT_DQ / "dq_ranking_quarter_summary.csv"
F_BUCK_QSUM    = OUT_DQ / "dq_bucket_share_quarter_summary.csv"
F_NAMED        = OUT_DQ / "dq_named_quarter_rows.csv"
# (rb MF-1) the membership calendar and the holdings calendar do not coincide:
# membership runs to 2024-03-31, holdings_eom stops at 2023-12-31. A quarter
# with no holdings denominator OUTSIDE the holdings span is a vintage fact, not
# a bug, and must be REPORTED as excluded rather than either aborting the run
# or vanishing silently. A quarter with no denominator INSIDE the span is a
# real calendar mismatch and still aborts.
F_QCENSUS      = OUT_DQ / "dq_bucket_quarter_calendar_census.csv"

# EU country list, verbatim from 00_setup.jl (single source of truth there;
# copied because this is a Python consumer of a Julia constant).
EU_COUNTRIES = ("GB","DE","FR","NL","CH","IT","ES","SE","DK","NO","FI",
                "BE","AT","IE","LU","PT","PL","CZ","HU","GR","RO","SK",
                "SI","BG","HR","EE","LV","LT")
EU_SQL = "(" + ",".join(f"'{c}'" for c in EU_COUNTRIES) + ")"

# (r3 item 1) These two tuples are NO LONGER the estimation set. Both the
# ranking and the bucket-share comparison now run on every quarter. The spot
# lists survive only so the old two-/three-quarter tables can still be
# reproduced verbatim, and they are written to *_SPOT_SUBSET.csv files whose
# names say what they are.
RANK_QUARTERS_SPOT   = ("2018-12-31", "2022-12-31")
BUCKET_QUARTERS_SPOT = ("2018-12-31", "2020-03-31", "2022-12-31")  # fig_ab spots

# Quarters that must appear by name in every summary, whatever the maxima say.
# 2017Q4: Indonesia -65.9% under V2/V3 (the sub-$5bn residual coming out).
# 2020Q4: Canada +9.74% under V0 (a dropped row coming back).
# Neither was in the old spot lists, which is exactly why the old summaries
# could not support a full-sample claim.
ANOMALY_QUARTERS = ("2017-12-31", "2020-12-31")

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
def _argmax_row(g: pd.DataFrame, val_col: str, label_cols: list[str]) -> dict:
    """|max| of val_col within g, plus the labels of the row attaining it.

    Returns zeros / empty labels for an all-NaN or empty group, so a summary
    row exists for every (version, quarter) cell even when nothing moves. When
    the maximum is exactly zero the returned label is the first row of the
    group and carries no meaning; read the label only when the value is
    nonzero.
    """
    a = g[val_col].abs()
    if len(g) == 0 or not np.isfinite(a.to_numpy(dtype=float)).any():
        out = {f"max_abs_{val_col}": 0.0}
        out.update({f"argmax_{c}": "" for c in label_cols})
        return out
    i = a.idxmax()
    out = {f"max_abs_{val_col}": float(a.loc[i])}
    out.update({f"argmax_{c}": g.loc[i, c] for c in label_cols})
    return out


def rank_quarter_summary(r: pd.DataFrame, v1_top5: dict) -> pd.DataFrame:
    """Per (version, quarter): how far the ranking moved away from V1.

    Split out of ranking() so it can be exercised without a DuckDB scan. One
    row per version-quarter, including the quarters where nothing moves — a
    summary that only lists movers cannot support a no-movement claim.
    """
    rows = []
    for (ver, q), g in r.groupby(["version", "quarter_end"], sort=True):
        top5 = list(g.sort_values("rank")["sec_country"].head(5))
        base5 = v1_top5.get(q, [])
        rec = {"version": ver, "quarter_end": q,
               "n_countries": int(g["sec_country"].nunique()),
               "n_rank_moves_vs_v1": int((g["rank_change_vs_v1"] != 0).sum())}
        rec.update(_argmax_row(g, "rank_change_vs_v1", ["sec_country"]))
        rec["argmax_rank_country"] = rec.pop("argmax_sec_country")
        rec.update(_argmax_row(g, "delta_pct_vs_v1", ["sec_country"]))
        rec["argmax_pct_country"] = rec.pop("argmax_sec_country")
        rec["top5_ordered"] = "|".join(top5)
        rec["top5_ordered_differs_from_v1"] = int(top5 != base5)
        rec["top5_set_differs_from_v1"] = int(set(top5) != set(base5))
        rec["is_spot_quarter"] = int(q in [pd.Timestamp(x)
                                           for x in RANK_QUARTERS_SPOT])
        rec["is_anomaly_quarter"] = int(q in [pd.Timestamp(x)
                                              for x in ANOMALY_QUARTERS])
        rows.append(rec)
    qsum = pd.DataFrame(rows).sort_values(["version", "quarter_end"])
    return qsum[["version", "quarter_end", "n_countries", "n_rank_moves_vs_v1",
                 "max_abs_rank_change_vs_v1", "argmax_rank_country",
                 "max_abs_delta_pct_vs_v1", "argmax_pct_country",
                 "top5_ordered", "top5_ordered_differs_from_v1",
                 "top5_set_differs_from_v1", "is_spot_quarter",
                 "is_anomaly_quarter"]].reset_index(drop=True)


def ranking(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Country ranking by US-held value, per version, at EVERY quarter-end.

    Step-4 construction: investor_country='US', EU sec_country, NO universe
    restriction (mirrors 04_us_own_by_eu_country_snapshot / I_ict).

    (r3 item 1) The old version ran on RANK_QUARTERS_SPOT = {2018Q4, 2022Q4}.
    Two dates cannot establish "the filter does not move the rankings" when
    the two largest dollar anomalies in the delta table live at 2017Q4 and
    2020Q4. The ranking is now recomputed at every quarter-end that carries
    US holdings, a per-quarter summary is emitted so the claim can be stated
    over the full sample or refuted, and the two anomaly quarters are printed
    by name whether or not they are the maxima.

    The quarter-end restriction is the same one country_quarter_totals uses,
    so the two families are on one calendar.
    """
    wide = con.sql(f"""
        SELECT sec_country, report_date AS quarter_end,
               {per_version_sums('adj_mv')}
        FROM base
        WHERE investor_country = 'US'
          AND sec_country IN {EU_SQL}
          AND report_date = last_day(report_date)
          AND month(report_date) IN (3, 6, 9, 12)
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
    r["is_anomaly_quarter"] = r["quarter_end"].isin(
        [pd.Timestamp(q) for q in ANOMALY_QUARTERS])
    r["is_spot_quarter"] = r["quarter_end"].isin(
        [pd.Timestamp(q) for q in RANK_QUARTERS_SPOT])
    r = (r[["version", "quarter_end", "sec_country", "us_held_usd", "rank",
            "us_held_usd_v1", "rank_v1", "delta_usd_vs_v1", "delta_pct_vs_v1",
            "rank_change_vs_v1", "is_spot_quarter", "is_anomaly_quarter"]]
         .sort_values(["version", "quarter_end", "rank"])
         .reset_index(drop=True))
    rotate_r3pre(F_RANKING)   # (rb A2) rotate only now, with the content in hand
    r.to_csv(F_RANKING, index=False)

    n_q = r["quarter_end"].nunique()
    n_moves = int((r["rank_change_vs_v1"] != 0).sum())
    print(f"\n[3] wrote {F_RANKING.name} ({len(r):,} rows, ALL {n_q} quarters); "
          f"{n_moves} country-quarter rank positions differ from V1")

    # ---- the retained spot subset, marked as a subset ---------------------
    spot = r.loc[r["is_spot_quarter"]].copy()
    spot.to_csv(F_RANKING_SPOT, index=False)
    print(f"    wrote {F_RANKING_SPOT.name} ({len(spot):,} rows) — SUBSET of the "
          f"above at {', '.join(RANK_QUARTERS_SPOT)}, kept for continuity only")

    # ---- per-quarter summary ---------------------------------------------
    v1_top5 = {q: list(g.sort_values("rank")["sec_country"].head(5))
               for q, g in r.loc[r["version"].eq("V1")].groupby("quarter_end")}
    qsum = rank_quarter_summary(r, v1_top5)
    qsum.to_csv(F_RANK_QSUM, index=False)
    print(f"    wrote {F_RANK_QSUM.name} ({len(qsum):,} version-quarter rows)")

    nv1 = qsum.loc[qsum["version"].ne("V1")]
    tot_moves = int(nv1["n_rank_moves_vs_v1"].sum())
    n_top5_ord = int(nv1["top5_ordered_differs_from_v1"].sum())
    n_top5_set = int(nv1["top5_set_differs_from_v1"].sum())
    print(f"    FULL SAMPLE (non-V1 version-quarters, n={len(nv1):,}): "
          f"{tot_moves} rank moves; {n_top5_ord} quarters where the ORDERED "
          f"top-5 differs; {n_top5_set} where the top-5 SET differs")
    if len(nv1):
        i = nv1["max_abs_delta_pct_vs_v1"].idxmax()
        w = nv1.loc[i]
        print(f"    worst |delta %| anywhere: {w['max_abs_delta_pct_vs_v1']:.4f}% "
              f"({w['version']}, {w['quarter_end'].date()}, "
              f"{w['argmax_pct_country']})")
        j = nv1["max_abs_rank_change_vs_v1"].idxmax()
        w = nv1.loc[j]
        print(f"    worst |rank change| anywhere: "
              f"{w['max_abs_rank_change_vs_v1']:.0f} "
              f"({w['version']}, {w['quarter_end'].date()}, "
              f"{w['argmax_rank_country']})")

    # ---- the named anomaly quarters, unconditionally ----------------------
    for q in ANOMALY_QUARTERS:
        t = qsum.loc[qsum["quarter_end"].eq(pd.Timestamp(q))]
        if not len(t):
            print(f"    [NAMED {q}] NOT PRESENT in the US-held ranking calendar "
                  f"— report this, do not silently omit it")
            continue
        print(f"    [NAMED {q}] ranking summary (all four versions):")
        print(t[["version", "n_rank_moves_vs_v1", "max_abs_rank_change_vs_v1",
                 "max_abs_delta_pct_vs_v1", "argmax_pct_country",
                 "top5_ordered"]]
              .to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    # ---- the old spot printout, kept ---------------------------------------
    for q in RANK_QUARTERS_SPOT:
        t = r.loc[r["quarter_end"].eq(pd.Timestamp(q))
                  & r["version"].isin(["V0", "V2"]) & r["rank"].le(10)]
        print(f"    [SPOT SUBSET] top-10 under V0/V2 at {q} "
              f"(rank_change_vs_v1 != 0 rows flag moves):")
        print(t[["version", "sec_country", "us_held_usd", "rank", "rank_change_vs_v1"]]
              .to_string(index=False, float_format=lambda x: f"{x/1e9:,.1f}bn"))
    return r, qsum


# ---------------------------------------------------------------------------
def bucket_quarter_summary(out: pd.DataFrame) -> pd.DataFrame:
    """Per (version, measure, quarter): the largest Figure-A share move vs V1.

    Split out of bucket_shares() so it can be exercised without a DuckDB scan.
    One row per version-measure-quarter, zeros included, so the full-sample
    maximum and its argmax quarter can be read straight off the file.
    """
    lbl = ["bucket", "holder_group"]
    rows = []
    for (ver, meas, q), g in out.groupby(["version", "measure", "quarter_end"],
                                         sort=True):
        rec = {"version": ver, "measure": meas, "quarter_end": q,
               "n_cells": int(len(g))}
        a = _argmax_row(g, "delta_share_global_vs_v1_pp", lbl)
        rec["max_abs_delta_share_global_pp"] = a["max_abs_delta_share_global_vs_v1_pp"]
        rec["argmax_global_bucket"] = a["argmax_bucket"]
        rec["argmax_global_holder_group"] = a["argmax_holder_group"]
        b = _argmax_row(g, "delta_share_eu_vs_v1_pp", lbl)
        rec["max_abs_delta_share_eu_pp"] = b["max_abs_delta_share_eu_vs_v1_pp"]
        rec["argmax_eu_bucket"] = b["argmax_bucket"]
        rec["argmax_eu_holder_group"] = b["argmax_holder_group"]
        c = _argmax_row(g, "delta_usd_vs_v1", lbl)
        rec["max_abs_delta_usd"] = c["max_abs_delta_usd_vs_v1"]
        rec["is_spot_quarter"] = int(q in [pd.Timestamp(x)
                                           for x in BUCKET_QUARTERS_SPOT])
        rec["is_anomaly_quarter"] = int(q in [pd.Timestamp(x)
                                              for x in ANOMALY_QUARTERS])
        rows.append(rec)
    return (pd.DataFrame(rows)
              .sort_values(["version", "measure", "quarter_end"])
              .reset_index(drop=True))


def bucket_shares(con: duckdb.DuckDBPyConnection,
                  totals_wide: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """fig_A country bucket shares at EVERY membership quarter, per version.

    Numerator mirrors 06's ict_grouped (patch 3): firms restricted to the
    canonical grid universe, country taken from the GRID. Denominators are the
    holder group's global / EU book recomputed under the same regime (the
    country_total_grouped construction). Membership comes verbatim from the
    canonical fig_A_country_bucket_membership.csv — classification is on
    Revere links, which no holdings-DQ regime touches.

    (r3 item 1) The old version ran on BUCKET_QUARTERS_SPOT = {2018Q4, 2020Q1,
    2022Q4}. The reported "max share move 0.0010pp global / 0.0022pp EU" was
    therefore a three-quarter maximum, and the three quarters exclude both
    anomaly quarters (2017Q4 Indonesia, 2020Q4 Canada). The comparison now
    runs on every quarter in the membership file, emits a per-quarter
    max/argmax summary, and prints the anomaly quarters by name.
    """
    if not F_MEMB.is_file():
        raise FileNotFoundError(
            f"{F_MEMB} not found — run build_fig_ab_data.py first.")
    memb = pd.read_csv(F_MEMB, parse_dates=["quarter_end"])
    need = {"measure", "country", "quarter_end", "bucket"}
    if not need.issubset(memb.columns):
        raise RuntimeError(f"{F_MEMB.name} lacks columns {need - set(memb.columns)}")
    memb = memb[["measure", "quarter_end", "country", "bucket"]]
    memb_q = sorted(pd.Timestamp(x) for x in memb["quarter_end"].unique())
    spots = [pd.Timestamp(q) for q in BUCKET_QUARTERS_SPOT]

    # ---- (rb MF-1) CALENDAR RECONCILIATION, BEFORE ANY SCAN ---------------
    # The two families do not share a calendar. Membership (Revere-link
    # classification) runs 2007Q3..2024Q1; holdings_eom, which supplies every
    # denominator, runs 1999Q1..2023Q4. The previous gate demanded a
    # denominator for EVERY membership quarter and therefore aborted on
    # 2024-03-31 — after ten minutes of scans and after main() had already
    # rotated the canonical files away.
    #
    # The distinction that matters:
    #   * a membership quarter OUTSIDE the holdings span is a vintage fact
    #     (the holdings feed simply has not been extended that far). It is
    #     excluded from the comparison and REPORTED as excluded — a census row
    #     with evaluated=0 and a stated reason, never a silent drop.
    #   * a membership quarter INSIDE the holdings span with no denominator is
    #     a genuine calendar mismatch and still aborts.
    hold_q = sorted(pd.Timestamp(x) for x in totals_wide["quarter_end"].unique())
    if not hold_q:
        raise RuntimeError("country_quarter_totals returned no quarters.")
    hold_set = set(hold_q)
    hold_lo, hold_hi = hold_q[0], hold_q[-1]
    inside_missing = [q for q in memb_q
                      if hold_lo <= q <= hold_hi and q not in hold_set]
    if inside_missing:
        raise RuntimeError(
            f"{len(inside_missing)} membership quarters lie INSIDE the "
            f"holdings span {hold_lo.date()}..{hold_hi.date()} but carry no "
            f"country-quarter denominator (first: {inside_missing[0].date()}) "
            "— a real quarter-end calendar mismatch between the two families.")
    out_of_span = [q for q in memb_q if q < hold_lo or q > hold_hi]
    all_q = [q for q in memb_q if q in hold_set]
    if not all_q:
        raise RuntimeError("membership and holdings calendars are disjoint.")

    memb_n = memb.groupby("quarter_end").size()
    cen_rows = []
    for q in memb_q:
        ev = q in hold_set
        cen_rows.append({
            "quarter_end": q, "in_membership": 1,
            "n_membership_rows": int(memb_n.get(q, 0)),
            "in_holdings_calendar": int(q in hold_set),
            "inside_holdings_span": int(hold_lo <= q <= hold_hi),
            "evaluated": int(ev),
            "reason": "" if ev else
                      "outside holdings_eom span (no denominator; "
                      f"holdings end {hold_hi.date()})"})
    for q in hold_q:
        if q in set(memb_q):
            continue
        cen_rows.append({
            "quarter_end": q, "in_membership": 0, "n_membership_rows": 0,
            "in_holdings_calendar": 1, "inside_holdings_span": 1,
            "evaluated": 0,
            "reason": "no fig_A bucket membership at this quarter"})
    qcen = (pd.DataFrame(cen_rows).sort_values("quarter_end")
              .reset_index(drop=True))
    qcen.to_csv(F_QCENSUS, index=False)

    print(f"\n[4] bucket shares. membership calendar {len(memb_q)} quarters "
          f"({memb_q[0].date()}..{memb_q[-1].date()}); holdings calendar "
          f"{len(hold_q)} quarters ({hold_lo.date()}..{hold_hi.date()}); "
          f"EVALUATED on the {len(all_q)} shared quarters "
          f"({all_q[0].date()}..{all_q[-1].date()})")
    if out_of_span:
        print(f"    EXCLUDED (reported, not dropped): "
              f"{len(out_of_span)} membership quarter(s) outside the holdings "
              f"span: {', '.join(q.date().isoformat() for q in out_of_span)}")
    print(f"    wrote {F_QCENSUS.name} ({len(qcen):,} rows) — the calendar "
          f"census; every quarter of either family carries a row.")

    missing_spot = [q for q in spots if q not in set(all_q)]
    if missing_spot:
        raise RuntimeError(
            f"spot quarters absent from the evaluated calendar: {missing_spot}")
    # the two named anomaly quarters must be INSIDE the evaluated set — the
    # whole point of the r3 item-1 fix. If a future vintage pushes one out,
    # abort rather than report an all-quarter maximum that silently omits it.
    anom_missing = [pd.Timestamp(q) for q in ANOMALY_QUARTERS
                    if pd.Timestamp(q) in set(memb_q)
                    and pd.Timestamp(q) not in set(all_q)]
    if anom_missing:
        raise RuntimeError(
            f"named anomaly quarter(s) {[q.date() for q in anom_missing]} are "
            "in the membership file but not evaluable — refusing to report an "
            "all-quarter maximum that excludes them.")
    memb = memb.loc[memb["quarter_end"].isin(all_q)].copy()

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

    dates = ", ".join(f"DATE '{pd.Timestamp(q).date()}'" for q in all_q)
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

    # holder-group books (denominators) from the country-quarter totals.
    # all_q is the reconciled calendar built above, so every element has a
    # denominator by construction; the assert is a cheap tripwire against a
    # later edit that reintroduces an unreconciled quarter list.
    q_set = set(pd.Timestamp(q) for q in all_q)
    missing_den = sorted(q_set - set(totals_wide["quarter_end"].unique()))
    if missing_den:
        raise RuntimeError(
            f"{len(missing_den)} EVALUATED quarters have no country-quarter "
            f"denominator (first: {pd.Timestamp(missing_den[0]).date()}) — the "
            "calendar reconciliation above was bypassed.")
    tw = totals_wide.loc[totals_wide["quarter_end"].isin(q_set)].copy()
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
    out["is_spot_quarter"] = out["quarter_end"].isin(spots)
    out["is_anomaly_quarter"] = out["quarter_end"].isin(
        [pd.Timestamp(q) for q in ANOMALY_QUARTERS])
    out = (out[["version"] + keys +
               ["n_countries", "usd_value", "book_global", "book_eu",
                "share_of_book_global", "share_of_book_eu",
                "usd_value_v1", "delta_usd_vs_v1",
                "delta_share_global_vs_v1_pp", "delta_share_eu_vs_v1_pp",
                "is_spot_quarter", "is_anomaly_quarter"]]
           .sort_values(["version"] + keys)
           .reset_index(drop=True))
    # ---- self-check FIRST, write SECOND (r1 advisory A2): a drifted V1 must
    # ---- never leave a plausible-looking four-version table on disk --------
    # (r3 item 1) the check now runs on every quarter the two files share, not
    # on the three spot quarters.
    if F_CTRY.is_file():
        ship = pd.read_csv(F_CTRY, parse_dates=["quarter_end"])
        if "share_of_book_global" in ship.columns and "measure" in ship.columns:
            s = ship.loc[ship["quarter_end"].isin(q_set),
                         ["measure", "quarter_end", "bucket", "holder_group",
                          "share_of_book_global"]]
            n_dup = int(s.duplicated(subset=keys).sum())
            if n_dup:
                raise RuntimeError(
                    f"{F_CTRY.name} has {n_dup} duplicate rows on "
                    f"{keys} (a second universe or vintage in one file) — the "
                    "self-check cannot be run one-to-one.")
            chk = out.loc[out["version"].eq("V1")].merge(
                s, on=keys, how="inner", suffixes=("", "_ship"),
                validate="one_to_one")
            if len(chk):
                chk = chk.assign(abs_diff=(chk["share_of_book_global"]
                                           - chk["share_of_book_global_ship"]).abs())
                diff = float(chk["abs_diff"].max())
                print(f"    V1 vs shipped fig_A_country_buckets.csv "
                      f"(share_of_book_global, {len(chk)} cells over "
                      f"{chk['quarter_end'].nunique()} shared quarters): "
                      f"max abs diff {diff:.3e}")
                if not diff < 1e-9:
                    worst = (chk.sort_values("abs_diff", ascending=False)
                                .head(10)[["measure", "quarter_end", "bucket",
                                           "holder_group", "abs_diff"]])
                    print("    worst offending cells:")
                    print(worst.to_string(index=False))
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

    rotate_r3pre(F_BUCKETS)   # (rb A2) rotate only now, with the content in hand
    out.to_csv(F_BUCKETS, index=False)
    print(f"    wrote {F_BUCKETS.name} ({len(out):,} rows, ALL "
          f"{out['quarter_end'].nunique()} quarters)")

    # ---- the retained spot subset, marked as a subset ----------------------
    sub = out.loc[out["is_spot_quarter"]].copy()
    sub.to_csv(F_BUCKETS_SPOT, index=False)
    print(f"    wrote {F_BUCKETS_SPOT.name} ({len(sub):,} rows) — SUBSET at "
          f"{', '.join(BUCKET_QUARTERS_SPOT)}, kept for continuity only")

    # ---- per (version, measure, quarter) max / argmax ----------------------
    bsum = bucket_quarter_summary(out)
    bsum.to_csv(F_BUCK_QSUM, index=False)
    print(f"    wrote {F_BUCK_QSUM.name} ({len(bsum):,} version-measure-quarter rows)")

    nv1 = bsum.loc[bsum["version"].ne("V1")]
    for col, tag in (("max_abs_delta_share_global_pp", "global"),
                     ("max_abs_delta_share_eu_pp", "eu")):
        if not len(nv1):
            break
        i = nv1[col].idxmax()
        w = nv1.loc[i]
        bk = w["argmax_global_bucket"] if tag == "global" else w["argmax_eu_bucket"]
        hg = (w["argmax_global_holder_group"] if tag == "global"
              else w["argmax_eu_holder_group"])
        print(f"    FULL SAMPLE max |share move| ({tag} denominator): "
              f"{w[col]:.4f} pp at {w['version']}, {w['measure']}, "
              f"{w['quarter_end'].date()}, {bk}/{hg}")
    # what the OLD three-quarter claim would have said, for contrast
    old = bsum.loc[bsum["version"].ne("V1") & bsum["is_spot_quarter"].eq(1)]
    if len(old):
        print(f"    (old SPOT-ONLY maxima, for contrast: global "
              f"{old['max_abs_delta_share_global_pp'].max():.4f} pp, eu "
              f"{old['max_abs_delta_share_eu_pp'].max():.4f} pp)")

    # ---- named anomaly quarters, printed unconditionally -------------------
    for q in ANOMALY_QUARTERS:
        t = bsum.loc[bsum["quarter_end"].eq(pd.Timestamp(q))]
        if not len(t):
            print(f"    [NAMED {q}] NOT PRESENT in the bucket membership "
                  f"calendar — report this, do not silently omit it")
            continue
        print(f"    [NAMED {q}] bucket-share summary:")
        print(t[["version", "measure", "max_abs_delta_share_global_pp",
                 "max_abs_delta_share_eu_pp", "argmax_global_bucket",
                 "argmax_global_holder_group"]]
              .to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    return out, bsum


# ---------------------------------------------------------------------------
def rotate_r3pre(p: Path) -> None:
    """Project rotation rule: an existing canonical output is renamed to
    *_r3pre before being overwritten, and the run REFUSES if that name is
    already taken. Applied to the two files whose CONTENT CHANGES MEANING with
    the r3 item-1 fix: they used to hold two / three spot quarters and now
    hold every quarter, so the old and new files are not comparable and the
    old one must not be silently replaced."""
    if not p.is_file():
        return
    tgt = p.with_name(p.stem + "_r3pre" + p.suffix)
    if tgt.exists():
        raise RuntimeError(
            f"rotation target {tgt.name} already exists — refusing to "
            f"overwrite it. Move or delete it by hand, deliberately.")
    p.rename(tgt)
    print(f"    rotated {p.name} -> {tgt.name} (spot-era vintage preserved)")


def write_named_quarters(rank_qsum: pd.DataFrame,
                         buck_qsum: pd.DataFrame) -> pd.DataFrame:
    """(r3 item 1) One file that carries the anomaly quarters by name, from
    BOTH summary families, whether or not they attain any maximum AND whether
    or not the quarter is on that family's calendar at all.

    The point is that 2017Q4 and 2020Q4 can never again sit outside the
    checked set. An absent quarter therefore has to leave a row that SAYS it
    is absent (present=0). Emitting nothing would reproduce the original
    defect in a quieter form: a reader of this file could not tell a quarter
    that was checked and did not move from a quarter that was never checked.
    """
    parts = []
    for q in ANOMALY_QUARTERS:
        ts = pd.Timestamp(q)
        for fam, src in (("us_held_ranking", rank_qsum),
                         ("fig_A_bucket_share", buck_qsum)):
            g = src.loc[src["quarter_end"].eq(ts)]
            if len(g):
                parts.append(g.assign(family=fam, present=1))
            else:
                print(f"    [NAMED {q}] absent from the {fam} calendar: "
                      f"emitted as a present=0 row, NOT omitted")
                parts.append(pd.DataFrame([{"family": fam, "version": "",
                                            "quarter_end": ts, "present": 0}]))
    # ANOMALY_QUARTERS is non-empty, so parts is never empty.
    named = pd.concat(parts, ignore_index=True, sort=False)
    front = ["family", "version", "quarter_end", "present"]
    named = named[front + [c for c in named.columns if c not in front]]
    named.to_csv(F_NAMED, index=False)
    n_abs = int((named["present"] == 0).sum())
    print(f"\n[5] wrote {F_NAMED.name} ({len(named):,} rows): the named "
          f"anomaly quarters {', '.join(ANOMALY_QUARTERS)} in both families, "
          f"reported regardless of the maxima "
          f"({n_abs} absent-quarter placeholder rows)")
    return named


# ---------------------------------------------------------------------------
def main() -> None:
    for p in (Path(EOM), Path(GRID), F_MEMB):
        if not Path(p).is_file():
            raise FileNotFoundError(f"missing canonical input: {p}")
    OUT_DQ.mkdir(parents=True, exist_ok=True)

    # (rb A2) PRE-FLIGHT ONLY — no rotation here. The previous version rotated
    # both canonical files at the top of main(), before ten minutes of scans.
    # When a later step aborted, the canonical names were already gone AND the
    # *_r3pre slots were occupied, so the very next run died in rotate_r3pre
    # with "rotation target already exists" and a human had to untangle it by
    # hand. Rotation now happens inside ranking() and bucket_shares(),
    # immediately before each file is written, i.e. only once its replacement
    # content actually exists. What stays here is the cheap check that the
    # slots are free, so a doomed run fails in the first second rather than
    # after the scans.
    for p in (F_RANKING, F_BUCKETS):
        tgt = p.with_name(p.stem + "_r3pre" + p.suffix)
        if p.is_file() and tgt.exists():
            raise RuntimeError(
                f"pre-flight: both {p.name} and {tgt.name} exist. Rotation "
                "would refuse later, after the scans. Move or delete "
                f"{tgt.name} by hand, deliberately, then re-run.")

    con = connect()
    build_base_view(con)

    cen = census(con)
    cen.to_csv(F_CENSUS, index=False)
    print(f"[1] wrote {F_CENSUS.name} ({len(cen):,} rows)")

    totals_wide = country_quarter_totals(con)
    write_deltas(totals_wide)
    _, rank_qsum = ranking(con)
    _, buck_qsum = bucket_shares(con, totals_wide)
    write_named_quarters(rank_qsum, buck_qsum)

    con.close()
    print("\nDONE — all outputs under", OUT_DQ)
    print("READ-ME for the write-up: the ranking and bucket-share claims must "
          "be quoted from dq_ranking_quarter_summary.csv and "
          "dq_bucket_share_quarter_summary.csv, which cover every quarter. "
          "The *_SPOT_SUBSET.csv files are the old two-/three-quarter tables "
          "and are not sufficient evidence on their own.")


if __name__ == "__main__":
    sys.exit(main())
