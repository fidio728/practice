# build_country_measures.py
# ===========================================================================
# ADVISOR DELIVERABLE (Emanuele, meeting 2026-08-04, 11:07-13:26):
# country x quarter China-exposure measures M1 / M2 / M3, plus the M1-vs-M3
# divergence diagnostic (link concentration) he flagged at the same point.
#
# DEFINITIONS (numerator/denominator are B7 = CUSTOMER + SUPPLIER only, and
# are NOT changed here; see 02_china_exposure.jl lines ~868-950):
#
#   firm-level ratio      CN_i,t = (n_cn_customer + n_cn_supplier)
#                                  / n_supplychain_links              in [0,1]
#                        (n_supplychain_links = 0  ->  CN = 0, the EM-CHANGE-2
#                         zero recode; the firm still enters the average)
#
#   M1_c,t  =  SUM_i in c (n_cn_customer + n_cn_supplier)
#              ------------------------------------------
#              SUM_i in c (n_supplychain_links)
#           = link-weighted ("aggregate ratio"). Zero-recode firms contribute
#             0 to BOTH sums, i.e. they are literally invisible to M1. This is
#             exactly the asymmetry that makes M1 and M3 diverge.
#
#   M2_c,t  =  SUM_i (mcap_i,t * CN_i,t) / SUM_i (mcap_i,t)     market-cap wtd
#   M3_c,t  =  (1/N_c,t) * SUM_i CN_i,t                          equal weighted
#
# UNIVERSE RULE (Emanuele, email 2026-08-05, implemented upstream in
# 02_china_exposure.jl EM-CHANGE-2):
#   - firm PIT-present in Revere at t  ->  CN defined (possibly a genuine 0)
#   - firm absent from Revere at t     ->  NO ROW  ->  excluded from all three
#   Point-in-time presence is `revere_pit_present` / `revere_coverage_start`
#   already carried by firm_quarter_china_exposure.parquet, so this script
#   simply inherits it (it does not re-derive coverage starts).
#
# TWO UNIVERSES ARE EMITTED (the `universe` column):
#   'revere_eu_all'     every PIT-present European Revere firm-quarter
#                       (149,526 firms; country = eu_home_region from
#                       eu_revere_universe_qend.parquet). M2 is NOT computable
#                       here -- unlisted/unmatched Revere firms have no
#                       FactSet market cap -- so M2 is left empty.
#   'ownership_matched' the ESTIMATION universe: firms in
#                       merged_us_eu_zero_filled.parquet (the European
#                       ownership book) that are Revere-matched and PIT-present
#                       (country = sec_country from FactSet). All three
#                       measures computable. This is the universe the holdings
#                       figures are drawn on, so it is flagged is_primary=1.
#
# MARKET-CAP COVERAGE CAVEAT (reported, not hidden): market_cap comes from
# marketcap_it.parquet (04_us_ownership_european.jl, PIT-restricted
# shares_out x price). It covers ~43-90% of exposure-defined firm-quarters,
# declining over time as the Revere universe adds small unlisted firms. M2 is
# computed over the covered subset only; n_firms_m2 / n_firms and
# mcap_covered_share are written to the CSV for every cell.
#
# OUTPUTS
#   output/country_measures_m1m2m3.csv          (main deliverable)
#   output/country_measures_link_concentration.csv
#   output/country_measures_m1_vs_m3_divergence.csv
#   output/country_measures_coverage_by_year.csv
#
# Runtime ~1-2 min. No network. No retries: it either completes or raises.
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
OUT.mkdir(parents=True, exist_ok=True)

EXPOSURE = (OUT / "firm_quarter_china_exposure.parquet").as_posix()
REV_QEND = (OUT / "eu_revere_universe_qend.parquet").as_posix()
GRID     = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
MCAP     = (OUT / "marketcap_it.parquet").as_posix()

CSV_MAIN = OUT / "country_measures_m1m2m3.csv"
CSV_CONC = OUT / "country_measures_link_concentration.csv"
CSV_DIV  = OUT / "country_measures_m1_vs_m3_divergence.csv"
CSV_COV  = OUT / "country_measures_coverage_by_year.csv"

# 28 European jurisdictions, verbatim from 00_setup.jl.
EU_COUNTRIES = ("GB", "DE", "FR", "NL", "CH", "IT", "ES", "SE", "DK", "NO",
                "FI", "BE", "AT", "IE", "LU", "PT", "PL", "CZ", "HU", "GR",
                "RO", "SK", "SI", "BG", "HR", "EE", "LV", "LT")
EU_SQL = "(" + ",".join(f"'{c}'" for c in EU_COUNTRIES) + ")"

SPOT_QUARTERS = ("2018-12-31", "2020-03-31", "2022-12-31")


def _require(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"missing input {label}: {path}")


def measures_sql(src: str, universe: str) -> str:
    """Aggregate a firm-quarter relation to country x quarter M1/M2/M3.

    `src` must expose: country, quarter_end, firm_id, cn_links, sc_links,
    cn_share, mcap (mcap may be all-NULL).
    """
    return f"""
    SELECT
        '{universe}'                                       AS universe,
        country,
        quarter_end,
        COUNT(*)                                           AS n_firms,
        COUNT(*) FILTER (WHERE cn_share > 0)               AS n_firms_pos_cn,
        COUNT(*) FILTER (WHERE cn_share = 0)               AS n_firms_zero_cn,
        COUNT(*) FILTER (WHERE sc_links > 0)               AS n_firms_with_sc_links,
        SUM(cn_links)                                      AS sum_cn_links,
        SUM(sc_links)                                      AS sum_sc_links,
        -- M1: link-weighted aggregate ratio. NULL only if the whole country
        -- has zero supply-chain links that quarter (denominator undefined).
        CAST(SUM(cn_links) AS DOUBLE) / NULLIF(SUM(sc_links), 0)   AS M1,
        -- M2: market-cap-weighted mean of the firm ratio, over covered firms.
        SUM(cn_share * mcap) FILTER (WHERE mcap > 0)
            / NULLIF(SUM(mcap) FILTER (WHERE mcap > 0), 0)         AS M2,
        -- M3: equal-weighted mean of the firm ratio, ALL firms in the cell.
        AVG(cn_share)                                              AS M3,
        COUNT(*) FILTER (WHERE mcap > 0)                   AS n_firms_m2,
        SUM(mcap) FILTER (WHERE mcap > 0)                  AS mcap_total_usd
    FROM {src}
    GROUP BY 1, 2, 3
    """


def concentration_sql(src: str, universe: str) -> str:
    """Link-concentration diagnostic: share of supply-chain links held by the
    top decile of firms (ranked by own supply-chain link count) within each
    country x quarter. Ties broken on firm_id so the cut is deterministic.
    The decile is taken over ALL firms in the cell, zero-link firms included,
    because they are part of the population M3 averages over."""
    return f"""
    WITH ranked AS (
        SELECT country, quarter_end, firm_id, cn_links, sc_links,
               ROW_NUMBER() OVER (PARTITION BY country, quarter_end
                                  ORDER BY sc_links DESC, firm_id)      AS rk,
               COUNT(*)     OVER (PARTITION BY country, quarter_end)    AS n_all,
               SUM(sc_links) OVER (PARTITION BY country, quarter_end)   AS tot_sc
        FROM {src}
    )
    SELECT '{universe}' AS universe, country, quarter_end,
           n_all                                                        AS n_firms,
           CAST(CEIL(0.1 * n_all) AS BIGINT)                            AS n_top_decile,
           SUM(sc_links) FILTER (WHERE rk <= CEIL(0.1 * n_all))
               / NULLIF(CAST(SUM(sc_links) AS DOUBLE), 0)               AS top_decile_sc_link_share,
           SUM(cn_links) FILTER (WHERE rk <= CEIL(0.1 * n_all))
               / NULLIF(CAST(SUM(cn_links) AS DOUBLE), 0)               AS top_decile_cn_link_share,
           SUM(POWER(CAST(sc_links AS DOUBLE) / NULLIF(tot_sc, 0), 2))  AS hhi_sc_links,
           MAX(sc_links)                                                AS max_firm_sc_links
    FROM ranked
    GROUP BY 1, 2, 3, 4, 5
    """


def main() -> None:
    for p, lbl in ((EXPOSURE, "firm_quarter_china_exposure"),
                   (REV_QEND, "eu_revere_universe_qend"),
                   (GRID, "merged_us_eu_zero_filled"),
                   (MCAP, "marketcap_it")):
        _require(p, lbl)

    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    tmp = Path("E:/duckdb_tmp")
    if tmp.parent.exists():
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{tmp.as_posix()}'")

    # ------------------------------------------------------------------
    # Universe 1: every PIT-present European Revere firm-quarter.
    # Country = eu_home_region (Revere company home region, as-of the quarter
    # via eu_revere_universe_qend, so it is not a look-ahead constant).
    # ------------------------------------------------------------------
    con.execute(f"""
        CREATE OR REPLACE VIEW u_revere AS
        SELECT u.eu_home_region                                AS country,
               e.quarter_end                                   AS quarter_end,
               e.eu_company_id                                 AS firm_id,
               CAST(e.n_cn_customer + e.n_cn_supplier AS BIGINT) AS cn_links,
               e.n_supplychain_links                           AS sc_links,
               e.china_share                                   AS cn_share,
               CAST(NULL AS DOUBLE)                            AS mcap
        FROM read_parquet('{EXPOSURE}') e
        JOIN read_parquet('{REV_QEND}') u
          ON u.eu_company_id = e.eu_company_id AND u.qend = e.quarter_end
        WHERE u.eu_home_region IN {EU_SQL}
    """)
    n_rev = con.sql("SELECT COUNT(*) c FROM u_revere").fetchone()[0]
    n_exp = con.sql(f"SELECT COUNT(*) c FROM read_parquet('{EXPOSURE}')").fetchone()[0]
    print(f"[1] revere_eu_all firm-quarters: {n_rev:,} "
          f"(exposure panel has {n_exp:,}; join must not duplicate)")
    if n_rev > n_exp:
        raise RuntimeError(
            f"eu_revere_universe_qend join DUPLICATED rows: {n_rev:,} > {n_exp:,}")

    # ------------------------------------------------------------------
    # Universe 2: the estimation universe (European ownership book), matched
    # to Revere and PIT-present. Firm attributes in the grid are identical
    # across holder_group, so holder_group='US' picks exactly one row per
    # firm-quarter. china_share IS NOT NULL == PIT-present (EM-CHANGE-2
    # invariant: NULL is reachable only by absence from Revere).
    # market_cap is joined on the FULL (sec_entity_id, sec_country,
    # report_date) key that 04 built it on.
    # ------------------------------------------------------------------
    con.execute(f"""
        CREATE OR REPLACE VIEW u_owned AS
        SELECT g.sec_country                                   AS country,
               g.report_date                                   AS quarter_end,
               g.sec_entity_id                                 AS firm_id,
               CAST(g.n_cn_customer + g.n_cn_supplier AS BIGINT) AS cn_links,
               g.n_supplychain_links                           AS sc_links,
               g.china_share                                   AS cn_share,
               m.market_cap                                    AS mcap
        FROM read_parquet('{GRID}') g
        LEFT JOIN read_parquet('{MCAP}') m
               ON m.sec_entity_id = g.sec_entity_id
              AND m.sec_country   = g.sec_country
              AND m.report_date   = g.report_date
        WHERE g.holder_group = 'US'
          AND g.china_share IS NOT NULL
          AND g.sec_country IN {EU_SQL}
    """)
    n_own = con.sql("SELECT COUNT(*) c FROM u_owned").fetchone()[0]
    n_own_f = con.sql("SELECT COUNT(DISTINCT firm_id) c FROM u_owned").fetchone()[0]
    print(f"[2] ownership_matched firm-quarters: {n_own:,} over {n_own_f:,} firms")

    # key uniqueness on both universes (a duplicated firm-quarter would
    # silently double-count links into M1)
    for v in ("u_revere", "u_owned"):
        d = con.sql(f"SELECT COUNT(*) c FROM (SELECT firm_id, quarter_end "
                    f"FROM {v} GROUP BY 1,2 HAVING COUNT(*) > 1)").fetchone()[0]
        if d:
            raise RuntimeError(f"{v}: {d:,} duplicate (firm, quarter) keys")
    print("[3] key uniqueness OK on both universes")

    # ------------------------------------------------------------------
    # Measures
    # ------------------------------------------------------------------
    m_rev = con.sql(measures_sql("u_revere", "revere_eu_all")).df()
    m_own = con.sql(measures_sql("u_owned", "ownership_matched")).df()
    meas = pd.concat([m_own, m_rev], ignore_index=True)
    meas["is_primary"] = (meas["universe"] == "ownership_matched").astype("int8")
    meas["mcap_covered_share"] = meas["n_firms_m2"] / meas["n_firms"]

    # bounds: every measure is a share of links, so it must live in [0, 1]
    for col in ("M1", "M2", "M3"):
        bad = meas[col].dropna()
        if len(bad) and (bad.lt(-1e-12).any() or bad.gt(1 + 1e-12).any()):
            raise RuntimeError(f"{col} outside [0,1]: "
                               f"min={bad.min()} max={bad.max()}")
    # link identity: sum_cn_links can never exceed sum_sc_links (the numerator
    # is a subset of the denominator by construction)
    if (meas["sum_cn_links"] > meas["sum_sc_links"]).any():
        raise RuntimeError("sum_cn_links > sum_sc_links in some cells")
    print(f"[4] measures: {len(meas):,} country-quarter rows "
          f"({meas['country'].nunique()} countries, "
          f"{meas['quarter_end'].nunique()} quarters)")

    conc = pd.concat([con.sql(concentration_sql("u_owned", "ownership_matched")).df(),
                      con.sql(concentration_sql("u_revere", "revere_eu_all")).df()],
                     ignore_index=True)
    print(f"[5] concentration diagnostic: {len(conc):,} rows")

    out = meas.merge(
        conc[["universe", "country", "quarter_end", "n_top_decile",
              "top_decile_sc_link_share", "top_decile_cn_link_share",
              "hhi_sc_links", "max_firm_sc_links"]],
        on=["universe", "country", "quarter_end"], how="left", validate="one_to_one")
    out["quarter_end"] = pd.to_datetime(out["quarter_end"])
    out = out.sort_values(["universe", "country", "quarter_end"]).reset_index(drop=True)

    cols = ["universe", "is_primary", "country", "quarter_end",
            "M1", "M2", "M3",
            "n_firms", "n_firms_pos_cn", "n_firms_zero_cn",
            "n_firms_with_sc_links", "n_firms_m2", "mcap_covered_share",
            "sum_cn_links", "sum_sc_links", "mcap_total_usd",
            "n_top_decile", "top_decile_sc_link_share",
            "top_decile_cn_link_share", "hhi_sc_links", "max_firm_sc_links"]
    out[cols].to_csv(CSV_MAIN, index=False)
    print(f"[6] wrote {CSV_MAIN} ({len(out):,} rows)")

    conc.sort_values(["universe", "country", "quarter_end"]).to_csv(CSV_CONC, index=False)
    print(f"    wrote {CSV_CONC} ({len(conc):,} rows)")

    # ------------------------------------------------------------------
    # M1-vs-M3 divergence: the diagnostic Emanuele asked for. M1 weights by
    # links, so a single hub firm with hundreds of links can set a whole
    # country's M1 while M3 gives it 1/N. div = M1 - M3; we report it against
    # the top-decile link share so the mechanism is visible in one table.
    # ------------------------------------------------------------------
    div = out.loc[out["M1"].notna() & out["M3"].notna()].copy()
    div["m1_minus_m3"] = div["M1"] - div["M3"]
    div["m1_over_m3"] = div["M1"] / div["M3"].where(div["M3"] > 0)
    div["m2_minus_m3"] = div["M2"] - div["M3"]
    keep = ["universe", "country", "quarter_end", "M1", "M2", "M3",
            "m1_minus_m3", "m1_over_m3", "m2_minus_m3",
            "top_decile_sc_link_share", "hhi_sc_links",
            "n_firms", "n_firms_pos_cn"]
    div[keep].to_csv(CSV_DIV, index=False)
    print(f"    wrote {CSV_DIV} ({len(div):,} rows)")

    for uni in ("ownership_matched", "revere_eu_all"):
        d = div.loc[div["universe"] == uni]
        if not len(d):
            continue
        r = d[["m1_minus_m3", "top_decile_sc_link_share"]].corr().iloc[0, 1]
        print(f"    {uni}: mean(M1-M3)={d['m1_minus_m3'].mean():+.4f}  "
              f"corr(M1-M3, top-decile link share)={r:+.3f}")

    # ------------------------------------------------------------------
    # Coverage by year (honest reporting of the M2 market-cap hole)
    # ------------------------------------------------------------------
    cov = (out.assign(year=out["quarter_end"].dt.year)
              .groupby(["universe", "year"], as_index=False)
              .agg(n_cells=("country", "size"),
                   n_countries=("country", "nunique"),
                   firms=("n_firms", "sum"),
                   firms_m2=("n_firms_m2", "sum")))
    cov["mcap_covered_share"] = cov["firms_m2"] / cov["firms"]
    cov.to_csv(CSV_COV, index=False)
    print(f"    wrote {CSV_COV} ({len(cov):,} rows)")

    # ------------------------------------------------------------------
    # Spot values for the verifier
    # ------------------------------------------------------------------
    print("\n=== SPOT VALUES (universe=ownership_matched, top-6 countries by n_firms) ===")
    for q in SPOT_QUARTERS:
        s = out[(out["universe"] == "ownership_matched")
                & (out["quarter_end"] == pd.Timestamp(q))]
        s = s.nlargest(6, "n_firms")
        print(f"--- {q}")
        if not len(s):
            print("    (no rows)")
            continue
        print(s[["country", "M1", "M2", "M3", "n_firms", "n_firms_pos_cn",
                 "top_decile_sc_link_share"]].to_string(index=False,
                                                        float_format=lambda x: f"{x:.4f}"))
    print("\n=== SPOT VALUES (universe=revere_eu_all, same countries) ===")
    for q in SPOT_QUARTERS:
        s = out[(out["universe"] == "revere_eu_all")
                & (out["quarter_end"] == pd.Timestamp(q))]
        s = s.nlargest(6, "n_firms")
        print(f"--- {q}")
        print(s[["country", "M1", "M3", "n_firms", "n_firms_pos_cn",
                 "top_decile_sc_link_share"]].to_string(index=False,
                                                        float_format=lambda x: f"{x:.4f}"))
    con.close()


if __name__ == "__main__":
    sys.exit(main())
