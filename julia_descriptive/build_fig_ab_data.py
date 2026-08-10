# build_fig_ab_data.py
# ===========================================================================
# ADVISOR DELIVERABLE (Emanuele, meeting 2026-08-04): the bucket time series
# behind FIGURE A (33:27 + Stefano email) and FIGURE B (37:26).
#
# TIMING CONVENTION (Emanuele 35:49), applied everywhere in this file:
#   classification uses ties at t-1 ; the plotted outcome is measured at t.
# The grid is a complete cartesian firm x quarter x holder_group panel
# (13,103 firms x 100 quarters x 2 groups), so LAG(...,1) is exactly the
# previous CALENDAR quarter and LAG(...,4) exactly four quarters back. That
# is asserted below, because the lag semantics would silently change if the
# panel ever stopped being complete.
#
# ---------------------------------------------------------------- FIGURE A
# Outcome: US institutional holdings of European firms at t, split by the
# firm's China-link intensity at t-1.
#   split = 'zero_vs_positive'  buckets: zero_tie (CN_{t-1} = 0)
#                                        positive_tie (CN_{t-1} > 0)
#   split = 'quartile_ew'       buckets: zero_tie + Q1_low..Q4_high, where the
#                               quartile cutpoints are the UNWEIGHTED (one firm
#                               one vote, i.e. M3-style) 25/50/75th percentiles
#                               of CN_{t-1} AMONG POSITIVE-CN FIRMS in that
#                               quarter. Assignment is by VALUE against the
#                               cutpoints, not NTILE, so tied ratios (very
#                               common: CN is a ratio of small integers) are
#                               never split across buckets. Bucket sizes are
#                               written out so the resulting lumpiness is
#                               visible rather than hidden.
#   split = 'quartile_mcapw'    same, but the cutpoints are MARKET-CAP-WEIGHTED
#                               (M2-style) quantiles of CN_{t-1}. The whole
#                               classification set -- zero bucket included --
#                               is restricted to firms with a market cap at
#                               t-1, so every line in that panel is drawn on
#                               one common universe.
#
#   DOCUMENTED CHOICE (spec asks for it explicitly): in the quartile splits the
#   ZERO bucket is kept as its OWN line rather than being folded into Q1. The
#   zero bucket is not a low tail of the same distribution -- after the
#   EM-CHANGE-2 recode it mixes firms with a positive supply-chain denominator
#   and no China link (flag 0) with firms that have no supply-chain links at
#   all (flags 1 and 2). Merging it into Q1 would blend "measured near-zero"
#   with "nothing to measure".
#
#   Two normalizations, both written:
#     real_usd_2020   CPI-U (CPIAUCNS, quarter-end month, 2020 annual-average
#                     base) deflated level -- SAME convention as
#                     build_desc_trend_us_holdings.py, via the shared
#                     desc_trend_metrics helpers.
#     share_of_book_global / share_of_book_eu / share_of_book
#                     (DENOMINATOR SWITCH, 2026-08-10) BOTH portfolio-weight
#                     families are now carried through every firm/country
#                     aggregation, mirroring the 2026-08-08 GLOBAL-MAIN
#                     decision on the regression side (dw = global family):
#                       share_of_book_global = sum of portfolio_weight_global
#                           over the bucket = share of the holder group's
#                           GLOBAL (full FactSet-identifiable equity) book
#                           sitting in the bucket. MAIN series -- consistent
#                           with the main regression's global denominator.
#                       share_of_book_eu = sum of portfolio_weight_eu = share
#                           of the holder group's EUROPEAN book only. Kept as
#                           the WITHIN-EUROPE REALLOCATION DIAGNOSTIC.
#                       share_of_book = ALIAS of share_of_book_global, kept
#                           under the old name so existing readers keep
#                           working. WARNING: in CSVs written BEFORE
#                           2026-08-10 this column held the EU series; any
#                           comparison against old vintages must use
#                           share_of_book_eu, not share_of_book.
#     idx100          real_usd_2020 indexed to 100 at BASE_QUARTER.
#   The headline figure uses the normalized series; the level version is kept
#   because the advisor asked to see both.
#
#   Country-level variant: countries sorted by M1_{c,t-1} (from
#   build_country_measures.py) into quartiles; bottom vs top quartile, with the
#   middle half shown as a reference line.
#
# ---------------------------------------------------------------- FIGURE B
# Pre-emptive de-linking test: growth in China supply-chain links at t for
# firms that HAD US investors at t-1 vs firms that did not (or had little).
#   split = 'us_2bucket'  no_us  (US holdings = 0 at t-1)
#                         any_us (US holdings > 0 at t-1)
#   split = 'us_3bucket'  no_us / us_low / us_high, where low-vs-high splits
#                         the positive-US firms at that quarter's MEDIAN US
#                         dollar holding. (Median of the LEVEL, not of
#                         ownership_share: ownership_share needs market cap,
#                         whose coverage falls to ~45% late in the sample and
#                         is itself size-correlated, which would confound the
#                         split with firm size twice over.)
#   Outcome: bucket-level TOTAL China customer+supplier link count.
#     growth_yoy = L_b(t) / L_b(t-4) - 1     (headline)
#     growth_qoq = L_b(t) / L_b(t-1) - 1
#   The cohort is FIXED at t-1 and required to have a defined link count at
#   t-4, t-1 and t, so numerator and denominator cover the SAME firms and the
#   growth rate cannot be manufactured by Revere coverage expansion. Firms with
#   zero China links contribute 0 to both sums, so aggregating before dividing
#   avoids any firm-level division by zero.
#
# OUTPUTS (every intermediate kept)
#   output/fig_A_firm_buckets.csv
#   output/fig_A_firm_bucket_cutpoints.csv
#   output/fig_A_country_buckets.csv
#   output/fig_A_country_bucket_membership.csv
#   output/fig_B_link_growth.csv
#   output/fig_B_bucket_cutpoints.csv
#   output/fig_ab_tension_series.csv        (USA|China AI-GPR background)
#   output/fig_ab_spot_values.csv           (verifier spot checks)
#
# Runtime ~2 min. No network (CPI is read from the existing cache). No retries.
# ===========================================================================

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from desc_trend_metrics import deflate_and_add_yoy, normalize_fred_cpi

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
_env_out = os.environ.get("DPN_OUT_DIR", "").strip()
OUT = Path(_env_out).resolve() if _env_out else PROJ / "output"
OUT.mkdir(parents=True, exist_ok=True)

GRID   = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
MCAP   = (OUT / "marketcap_it.parquet").as_posix()
GPR_Q  = (OUT / "gpr_quarterly_with_shock.parquet").as_posix()
CTRY_M = OUT / "country_measures_m1m2m3.csv"
CPI_CACHE = OUT / "cpiaucns_monthly.csv"
CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS"

F_A_FIRM   = OUT / "fig_A_firm_buckets.csv"
F_A_CUTS   = OUT / "fig_A_firm_bucket_cutpoints.csv"
F_A_CTRY   = OUT / "fig_A_country_buckets.csv"
F_A_CMEMB  = OUT / "fig_A_country_bucket_membership.csv"
F_B_GROWTH = OUT / "fig_B_link_growth.csv"
F_B_CUTS   = OUT / "fig_B_bucket_cutpoints.csv"
F_TENSION  = OUT / "fig_ab_tension_series.csv"
F_SPOT     = OUT / "fig_ab_spot_values.csv"

# Indexing base for the normalized (=100) series: the last quarter before the
# 2018 tariff round, so the figure reads "relative to the eve of the trade
# war". Raw levels stay in the CSV, so re-basing needs no re-run.
BASE_QUARTER = pd.Timestamp("2017-12-31")

# Country-level M1 universe used for the country quartile split.
COUNTRY_UNIVERSE = "ownership_matched"
# A country needs at least this many firms in a quarter to be rankable on M1
# (otherwise a single 3-firm country can occupy the top quartile on noise).
MIN_FIRMS_FOR_COUNTRY_RANK = 20

SPOT_QUARTERS = (pd.Timestamp("2018-12-31"),
                 pd.Timestamp("2020-03-31"),
                 pd.Timestamp("2022-12-31"))


# ---------------------------------------------------------------------------
def weighted_quantile(values: np.ndarray, weights: np.ndarray,
                      probs: tuple[float, ...]) -> list[float]:
    """Weighted quantiles by cumulative-weight interpolation.

    Sorted by value; the quantile is the smallest value whose cumulative
    weight share (measured at the MIDPOINT of each observation's weight
    block, the standard convention) reaches p. Falls back to the plain
    quantile when all weights are equal, which keeps the mcap-weighted and
    equal-weighted variants comparable in degenerate cells.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if v.size == 0:
        return [np.nan] * len(probs)
    order = np.argsort(v, kind="mergesort")
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    p_mid = (cw - 0.5 * w) / cw[-1]
    return [float(np.interp(p, p_mid, v)) for p in probs]


def assign_by_cutpoints(x: pd.Series, c25: float, c50: float, c75: float) -> pd.Series:
    """Value-based quartile labels. Ties never straddle a bucket boundary."""
    out = pd.Series("Q4_high", index=x.index, dtype=object)
    out[x <= c75] = "Q3"
    out[x <= c50] = "Q2"
    out[x <= c25] = "Q1_low"
    return out


def load_cpi() -> pd.DataFrame:
    if CPI_CACHE.is_file():
        raw = pd.read_csv(CPI_CACHE)
        print(f"    CPI from cache {CPI_CACHE.name}")
    else:
        print(f"    downloading CPIAUCNS from {CPI_URL}")
        raw = pd.read_csv(CPI_URL)
    cpi = normalize_fred_cpi(raw)
    cpi.to_csv(CPI_CACHE, index=False)
    return cpi


def deflate(frame: pd.DataFrame, groups: list[str], cpi: pd.DataFrame) -> pd.DataFrame:
    """CPI-U deflate `usd_value` to 2020 dollars using the SAME helper (and so
    the same base and the same quarter-end-month convention) as
    build_desc_trend_us_holdings.py."""
    return deflate_and_add_yoy(frame, cpi, group_col=groups,
                               date_col="quarter_end", nominal_col="usd_value",
                               base_year=2020)


def add_index100(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    base = (frame.loc[frame["quarter_end"].eq(BASE_QUARTER), groups + ["real_usd_2020"]]
                 .rename(columns={"real_usd_2020": "_base"}))
    out = frame.merge(base, on=groups, how="left", validate="many_to_one")
    out["idx100"] = 100.0 * out["real_usd_2020"] / out["_base"].where(out["_base"] > 0)
    return out.drop(columns="_base")


# ---------------------------------------------------------------------------
def build_firm_quarter_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per (firm, quarter): US and NONUS holdings side by side, the
    firm's China-link measures, its market cap, and the t-1 / t-4 lags."""
    con.execute(f"""
        CREATE OR REPLACE TABLE fq AS
        WITH us AS (
            -- (2026-08-10) BOTH weight families carried: *_g = global
            -- (full-portfolio) denominator, MAIN; *_eu = EU-restricted
            -- denominator, within-Europe reallocation diagnostic.
            SELECT sec_entity_id, sec_country, report_date,
                   I_ict AS usd_us,
                   portfolio_weight_global AS w_us_g,
                   portfolio_weight_eu     AS w_us_eu,
                   china_share, china_share_lag1q,
                   CAST(n_cn_customer + n_cn_supplier AS BIGINT) AS cn_links,
                   n_supplychain_links, zero_recode_flag, revere_pit_present
            FROM read_parquet('{GRID}') WHERE holder_group = 'US'
        ), nonus AS (
            SELECT sec_entity_id, report_date,
                   I_ict AS usd_nonus,
                   portfolio_weight_global AS w_nonus_g,
                   portfolio_weight_eu     AS w_nonus_eu
            FROM read_parquet('{GRID}') WHERE holder_group = 'NONUS'
        )
        SELECT u.*, n.usd_nonus, n.w_nonus_g, n.w_nonus_eu, m.market_cap AS mcap
        FROM us u
        JOIN nonus n USING (sec_entity_id, report_date)
        LEFT JOIN read_parquet('{MCAP}') m
               ON m.sec_entity_id = u.sec_entity_id
              AND m.sec_country   = u.sec_country
              AND m.report_date   = u.report_date
    """)

    # The lag semantics below assume a COMPLETE cartesian panel. Verify.
    chk = con.sql("""
        SELECT COUNT(*) AS n_rows,
               COUNT(DISTINCT sec_entity_id) AS n_firms,
               COUNT(DISTINCT report_date)   AS n_quarters
        FROM fq""").df().iloc[0]
    if int(chk["n_rows"]) != int(chk["n_firms"]) * int(chk["n_quarters"]):
        raise RuntimeError(
            f"grid is not a complete cartesian panel: {int(chk['n_rows']):,} rows "
            f"!= {int(chk['n_firms']):,} firms x {int(chk['n_quarters']):,} quarters. "
            "LAG(...,k) would no longer mean 'k calendar quarters back'.")
    print(f"[A0] firm-quarter table: {int(chk['n_rows']):,} rows "
          f"({int(chk['n_firms']):,} firms x {int(chk['n_quarters']):,} quarters, complete)")

    df = con.sql("""
        SELECT sec_entity_id AS fid, sec_country AS country,
               report_date AS quarter_end,
               usd_us, w_us_g, w_us_eu, usd_nonus, w_nonus_g, w_nonus_eu,
               china_share AS cn, china_share_lag1q AS cn_lag,
               cn_links, n_supplychain_links AS sc_links,
               zero_recode_flag, mcap,
               LAG(mcap,     1) OVER w AS mcap_lag,
               LAG(usd_us,   1) OVER w AS usd_us_lag,
               LAG(cn_links, 1) OVER w AS cn_links_lag1,
               LAG(cn_links, 4) OVER w AS cn_links_lag4
        FROM fq
        WINDOW w AS (PARTITION BY sec_entity_id ORDER BY report_date)
        ORDER BY sec_entity_id, report_date
    """).df()
    df["quarter_end"] = pd.to_datetime(df["quarter_end"])

    # Independent check that the SQL LAG really is one calendar quarter: the
    # panel's own china_share shifted by one row must equal china_share_lag1q
    # that 06_cartesian_grid.jl computed.
    chk2 = df.copy()
    chk2["cn_shift"] = chk2.groupby("fid")["cn"].shift(1)
    both = chk2["cn_shift"].notna() & chk2["cn_lag"].notna()
    if both.any():
        mx = (chk2.loc[both, "cn_shift"] - chk2.loc[both, "cn_lag"]).abs().max()
        if mx > 1e-12:
            raise RuntimeError(f"row-shift lag disagrees with china_share_lag1q "
                               f"(max abs diff {mx:.3e})")
    n_mismatch = int((chk2["cn_shift"].isna() != chk2["cn_lag"].isna()).sum())
    if n_mismatch:
        raise RuntimeError(f"{n_mismatch:,} rows where the shifted china_share and "
                           f"china_share_lag1q disagree on missingness")
    print("[A0] lag check: row-shift(china_share) == china_share_lag1q on every row")
    return df


# ---------------------------------------------------------------------------
def build_figure_a_firm(df: pd.DataFrame, cpi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Firm-level bucket series for Figure A."""
    cls = df.loc[df["cn_lag"].notna()].copy()          # classifiable at t-1
    print(f"[A1] classifiable firm-quarters (CN_(t-1) non-missing): {len(cls):,} "
          f"over {cls['fid'].nunique():,} firms, "
          f"{cls['quarter_end'].min().date()}..{cls['quarter_end'].max().date()}")

    cut_rows: list[dict] = []
    frames: list[pd.DataFrame] = []

    # ---- split 1: zero vs positive -----------------------------------
    s1 = cls.copy()
    s1["split"] = "zero_vs_positive"
    s1["bucket"] = np.where(s1["cn_lag"] > 0, "positive_tie", "zero_tie")
    frames.append(s1)

    # ---- split 1b: COMPOSITION CONTROL -------------------------------
    # The rolling t-1 classification above cannot separate portfolio
    # REALLOCATION from RECLASSIFICATION: Revere adds relationships over time,
    # so firms migrate from the zero bucket into the positive bucket even if no
    # investor moves a dollar. This variant fixes bucket membership ONCE, on CN
    # measured at BASE_QUARTER, and holds it for the whole sample. It is
    # therefore NOT the t-1 timing convention -- it is the control for it, and
    # any share-of-book movement it shows is reallocation within a frozen set
    # of firms.
    base_lab = (cls.loc[cls["quarter_end"].eq(BASE_QUARTER), ["fid", "cn_lag"]]
                   .assign(bucket=lambda d: np.where(d["cn_lag"] > 0,
                                                     "positive_tie_fixed",
                                                     "zero_tie_fixed"))
                   .drop(columns="cn_lag"))
    if base_lab.empty:
        raise RuntimeError(f"no classifiable firms at BASE_QUARTER {BASE_QUARTER.date()}")
    s1b = df.merge(base_lab, on="fid", how="inner", validate="many_to_one").copy()
    s1b["split"] = "zero_vs_positive_fixed"
    print(f"[A1] composition-control cohort fixed at {BASE_QUARTER.date()}: "
          f"{base_lab['bucket'].value_counts().to_dict()}")
    frames.append(s1b)

    # ---- split 2/3: quartiles among positive-CN firms ----------------
    for split, need_mcap in (("quartile_ew", False), ("quartile_mcapw", True)):
        sub = cls.copy()
        if need_mcap:
            # one common universe for every line in this panel
            sub = sub.loc[sub["mcap_lag"].notna() & (sub["mcap_lag"] > 0)].copy()
        sub["split"] = split
        sub["bucket"] = "zero_tie"
        pos_mask = sub["cn_lag"] > 0
        for q, g in sub.loc[pos_mask].groupby("quarter_end", sort=True):
            if need_mcap:
                c25, c50, c75 = weighted_quantile(
                    g["cn_lag"].to_numpy(), g["mcap_lag"].to_numpy(),
                    (0.25, 0.50, 0.75))
            else:
                c25, c50, c75 = [float(g["cn_lag"].quantile(p))
                                 for p in (0.25, 0.50, 0.75)]
            lab = assign_by_cutpoints(g["cn_lag"], c25, c50, c75)
            sub.loc[lab.index, "bucket"] = lab.to_numpy()
            cut_rows.append({"split": split, "quarter_end": q,
                             "n_positive_firms": int(len(g)),
                             "cut_p25": c25, "cut_p50": c50, "cut_p75": c75,
                             "n_zero_firms": int((sub["quarter_end"].eq(q)
                                                  & ~pos_mask).sum())})
        frames.append(sub)

    tagged = pd.concat(frames, ignore_index=True)

    # ---- aggregate to (split, bucket, holder_group, quarter) ---------
    # (2026-08-10) both weight families summed; share_of_book aliases GLOBAL.
    out = []
    for grp, usd_col, wg_col, we_col in (
            ("US", "usd_us", "w_us_g", "w_us_eu"),
            ("NONUS", "usd_nonus", "w_nonus_g", "w_nonus_eu")):
        a = (tagged.assign(_usd=tagged[usd_col].fillna(0.0),
                           _wg=tagged[wg_col].fillna(0.0),
                           _we=tagged[we_col].fillna(0.0))
                   .groupby(["split", "bucket", "quarter_end"], as_index=False)
                   .agg(usd_value=("_usd", "sum"),
                        share_of_book_global=("_wg", "sum"),
                        share_of_book_eu=("_we", "sum"),
                        n_firms=("fid", "size"),
                        n_firms_held=("_usd", lambda s: int((s > 0).sum())),
                        mean_cn_lag=("cn_lag", "mean"),
                        cn_links_t=("cn_links", "sum")))
        a["holder_group"] = grp
        out.append(a)
    agg = pd.concat(out, ignore_index=True)
    agg["share_of_book"] = agg["share_of_book_global"]   # alias, GLOBAL family

    groups = ["split", "bucket", "holder_group"]
    agg = deflate(agg, groups, cpi)
    agg = add_index100(agg, groups)

    # sanity: within a split and quarter, each share family summed over buckets
    # must not exceed 1 (buckets are disjoint subsets of the same book; the
    # global sum is additionally far below 1 because EU firms are a slice of
    # the global book)
    for shcol in ("share_of_book_global", "share_of_book_eu"):
        tot = agg.groupby(["split", "holder_group", "quarter_end"])[shcol].sum()
        if (tot > 1 + 1e-9).any():
            raise RuntimeError(f"{shcol} sums above 1: max={tot.max():.6f}")

    cuts = pd.DataFrame(cut_rows).sort_values(["split", "quarter_end"])
    return agg.sort_values(groups + ["quarter_end"]).reset_index(drop=True), cuts


# ---------------------------------------------------------------------------
def build_figure_a_country(df: pd.DataFrame, cpi: pd.DataFrame,
                           measure: str = "M1") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Country-level variant: countries split on <measure>_{c,t-1}.

    (2026-08-09) Parameterized over M1/M2/M3 — the 2026-08-04 meeting minute
    commits to "each figure one version per exposure measure"; the caller loops
    over all three and the `measure` column distinguishes them in the CSVs.
    """
    if not CTRY_M.is_file():
        raise FileNotFoundError(
            f"{CTRY_M} not found — run build_country_measures.py first.")
    cm = pd.read_csv(CTRY_M, parse_dates=["quarter_end"])
    cm = cm.loc[cm["universe"].eq(COUNTRY_UNIVERSE),
                ["country", "quarter_end", measure, "n_firms"]].copy()
    cm = cm.loc[cm[measure].notna() & cm["n_firms"].ge(MIN_FIRMS_FOR_COUNTRY_RANK)]

    # classification at t-1 -> stamp forward one quarter to t
    cm["quarter_end"] = cm["quarter_end"] + pd.offsets.QuarterEnd(1)
    cm = cm.rename(columns={measure: "m1_lag", "n_firms": "n_firms_country_lag"})

    memb = []
    for q, g in cm.groupby("quarter_end", sort=True):
        if len(g) < 8:              # need at least 2 countries per quartile
            continue
        c25, c75 = g["m1_lag"].quantile(0.25), g["m1_lag"].quantile(0.75)
        b = pd.Series("mid_half", index=g.index, dtype=object)
        b[g["m1_lag"] > c75] = "top_quartile"
        b[g["m1_lag"] <= c25] = "bottom_quartile"
        gg = g.copy()
        gg["bucket"] = b.to_numpy()
        gg["cut_p25"] = c25
        gg["cut_p75"] = c75
        memb.append(gg)
    memb = pd.concat(memb, ignore_index=True)
    memb["measure"] = measure
    print(f"[A2:{measure}] country classification rows: {len(memb):,} "
          f"({memb['country'].nunique()} countries, {memb['quarter_end'].nunique()} quarters)")

    # (2026-08-10) both weight families carried through the country
    # aggregation; share_of_book aliases the GLOBAL family.
    firm_ct = (df.assign(_us=df["usd_us"].fillna(0.0),
                         _wus_g=df["w_us_g"].fillna(0.0),
                         _wus_e=df["w_us_eu"].fillna(0.0),
                         _nu=df["usd_nonus"].fillna(0.0),
                         _wnu_g=df["w_nonus_g"].fillna(0.0),
                         _wnu_e=df["w_nonus_eu"].fillna(0.0))
                 .groupby(["country", "quarter_end"], as_index=False)
                 .agg(usd_us=("_us", "sum"),
                      w_us_g=("_wus_g", "sum"), w_us_eu=("_wus_e", "sum"),
                      usd_nonus=("_nu", "sum"),
                      w_nonus_g=("_wnu_g", "sum"), w_nonus_eu=("_wnu_e", "sum"),
                      n_firms=("fid", "size")))
    j = memb.merge(firm_ct, on=["country", "quarter_end"], how="inner",
                   validate="one_to_one")

    out = []
    for grp, usd_col, wg_col, we_col in (
            ("US", "usd_us", "w_us_g", "w_us_eu"),
            ("NONUS", "usd_nonus", "w_nonus_g", "w_nonus_eu")):
        a = (j.groupby(["bucket", "quarter_end"], as_index=False)
              .agg(usd_value=(usd_col, "sum"),
                   share_of_book_global=(wg_col, "sum"),
                   share_of_book_eu=(we_col, "sum"),
                   n_countries=("country", "nunique"),
                   n_firms=("n_firms", "sum"),
                   mean_m1_lag=("m1_lag", "mean")))
        a["holder_group"] = grp
        out.append(a)
    agg = pd.concat(out, ignore_index=True)
    agg["share_of_book"] = agg["share_of_book_global"]   # alias, GLOBAL family
    agg["measure"] = measure
    agg["universe"] = COUNTRY_UNIVERSE

    groups = ["bucket", "holder_group"]
    agg = deflate(agg, groups, cpi)
    agg = add_index100(agg, groups)
    return (agg.sort_values(groups + ["quarter_end"]).reset_index(drop=True),
            memb.sort_values(["quarter_end", "country"]).reset_index(drop=True))


# ---------------------------------------------------------------------------
def build_figure_b(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """China-link growth by US-ownership bucket at t-1."""
    coh = df.loc[df["cn_links"].notna()
                 & df["cn_links_lag1"].notna()
                 & df["cn_links_lag4"].notna()].copy()
    # us holdings at t-1: the grid is zero-filled, so a firm nobody in the US
    # held is a genuine 0, not a missing value. Only the very first quarter of
    # the panel has a NULL lag, and those rows are dropped.
    coh = coh.loc[coh["usd_us_lag"].notna()]
    print(f"[B1] cohort rows (links defined at t-4, t-1, t): {len(coh):,} "
          f"over {coh['fid'].nunique():,} firms, "
          f"{coh['quarter_end'].min().date()}..{coh['quarter_end'].max().date()}")

    coh["has_us"] = coh["usd_us_lag"] > 0
    cut_rows = []
    coh["b2"] = np.where(coh["has_us"], "any_us", "no_us")
    coh["b3"] = "no_us"
    for q, g in coh.loc[coh["has_us"]].groupby("quarter_end", sort=True):
        med = float(g["usd_us_lag"].median())
        coh.loc[g.index, "b3"] = np.where(g["usd_us_lag"] > med, "us_high", "us_low")
        cut_rows.append({"quarter_end": q, "n_us_held_firms": int(len(g)),
                         "median_us_usd_lag": med,
                         "n_no_us_firms": int((coh["quarter_end"].eq(q)
                                               & ~coh["has_us"]).sum())})

    frames = []
    for split, col in (("us_2bucket", "b2"), ("us_3bucket", "b3")):
        a = (coh.groupby([col, "quarter_end"], as_index=False)
                .agg(n_firms_cohort=("fid", "size"),
                     cn_links_t=("cn_links", "sum"),
                     cn_links_lag1=("cn_links_lag1", "sum"),
                     cn_links_lag4=("cn_links_lag4", "sum"),
                     n_firms_with_cn_t=("cn_links", lambda s: int((s > 0).sum())),
                     n_firms_with_cn_lag4=("cn_links_lag4", lambda s: int((s > 0).sum())),
                     mean_us_usd_lag=("usd_us_lag", "mean")))
        a = a.rename(columns={col: "bucket"})
        a["split"] = split
        frames.append(a)
    g = pd.concat(frames, ignore_index=True)

    g["growth_qoq"] = g["cn_links_t"] / g["cn_links_lag1"].where(g["cn_links_lag1"] > 0) - 1.0
    g["growth_yoy"] = g["cn_links_t"] / g["cn_links_lag4"].where(g["cn_links_lag4"] > 0) - 1.0
    g["mean_cn_links_t"] = g["cn_links_t"] / g["n_firms_cohort"]
    g["share_firms_with_cn_t"] = g["n_firms_with_cn_t"] / g["n_firms_cohort"]

    return (g.sort_values(["split", "bucket", "quarter_end"]).reset_index(drop=True),
            pd.DataFrame(cut_rows).sort_values("quarter_end").reset_index(drop=True))


# ---------------------------------------------------------------------------
def main() -> None:
    for p in (GRID, MCAP, GPR_Q):
        if not Path(p).is_file():
            raise FileNotFoundError(f"missing input: {p}")

    # ------------------------------------------------------------------
    # ROTATION DISCIPLINE (r1 must-fix M1, 2026-08-10): REFUSE to overwrite
    # any existing output in place. This run FLIPS the semantics of the
    # share_of_book column (pre-2026-08-10 CSVs hold the EU series under that
    # name; from now on it aliases the GLOBAL series — see the WARNING in the
    # header), so a bare in-place rewrite is exactly the silent semantic swap
    # the L62-67 warning describes. Mirror of build_country_panel.py's guard:
    # rename each existing file to *_r2pre first, then re-run.
    # ------------------------------------------------------------------
    _targets = (F_TENSION, F_A_FIRM, F_A_CUTS, F_A_CTRY,
                F_A_CMEMB, F_B_GROWTH, F_B_CUTS, F_SPOT)
    _existing = [p for p in _targets if p.exists()]
    if _existing:
        raise RuntimeError(
            "target output(s) already exist — refusing to overwrite canonical "
            "figure CSVs in place (rotation rule): "
            + ", ".join(p.name for p in _existing)
            + ". Rename each to *_r2pre (e.g. fig_A_firm_buckets_r2pre.csv) "
            "before re-running; pre-2026-08-10 vintages hold the EU series "
            "under the share_of_book name and must stay comparable on disk.")

    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    tmp = Path("E:/duckdb_tmp")
    if tmp.parent.exists():
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{tmp.as_posix()}'")

    print("[0] CPI")
    cpi = load_cpi()

    df = build_firm_quarter_table(con)

    # tension background: USA|China direction of the bilateral AI-GPR index
    ten = con.sql(f"""
        SELECT quarter_end, gpr_us_cn, gpr_global, shock_us_cn
        FROM read_parquet('{GPR_Q}') ORDER BY quarter_end""").df()
    ten["quarter_end"] = pd.to_datetime(ten["quarter_end"])
    ten["direction"] = "USA|China (US -> China attention, Iacoviello-Tong bilateral AI-GPR)"
    ten.to_csv(F_TENSION, index=False)
    print(f"[T ] wrote {F_TENSION.name} ({len(ten):,} rows, "
          f"{ten['quarter_end'].min().date()}..{ten['quarter_end'].max().date()})")

    a_firm, a_cuts = build_figure_a_firm(df, cpi)
    a_firm.to_csv(F_A_FIRM, index=False)
    a_cuts.to_csv(F_A_CUTS, index=False)
    print(f"[A1] wrote {F_A_FIRM.name} ({len(a_firm):,} rows), "
          f"{F_A_CUTS.name} ({len(a_cuts):,} rows)")

    # (2026-08-09) one version per exposure measure, per the 2026-08-04 minute
    _ctry_parts, _memb_parts = [], []
    for _m in ("M1", "M2", "M3"):
        _c, _mb = build_figure_a_country(df, cpi, measure=_m)
        _ctry_parts.append(_c)
        _memb_parts.append(_mb)
    a_ctry = pd.concat(_ctry_parts, ignore_index=True)
    a_memb = pd.concat(_memb_parts, ignore_index=True)
    a_ctry.to_csv(F_A_CTRY, index=False)
    a_memb.to_csv(F_A_CMEMB, index=False)
    print(f"[A2] wrote {F_A_CTRY.name} ({len(a_ctry):,} rows, measures M1/M2/M3), "
          f"{F_A_CMEMB.name} ({len(a_memb):,} rows)")

    b, b_cuts = build_figure_b(df)
    b.to_csv(F_B_GROWTH, index=False)
    b_cuts.to_csv(F_B_CUTS, index=False)
    print(f"[B1] wrote {F_B_GROWTH.name} ({len(b):,} rows), "
          f"{F_B_CUTS.name} ({len(b_cuts):,} rows)")

    # ------------------------------------------------------------------
    # Spot values for the verifier
    # ------------------------------------------------------------------
    spot = []
    us = a_firm.loc[a_firm["holder_group"].eq("US")]
    for q in SPOT_QUARTERS:
        for _, r in us.loc[us["quarter_end"].eq(q)].iterrows():
            spot.append({"figure": "A_firm", "series": f"{r['split']}|{r['bucket']}",
                         "quarter_end": q.date(), "n": r["n_firms"],
                         "real_usd_2020": r["real_usd_2020"],
                         "share_of_book": r["share_of_book"],            # = global
                         "share_of_book_global": r["share_of_book_global"],
                         "share_of_book_eu": r["share_of_book_eu"],
                         "idx100": r["idx100"]})
    cus = a_ctry.loc[a_ctry["holder_group"].eq("US")]
    for q in SPOT_QUARTERS:
        for _, r in cus.loc[cus["quarter_end"].eq(q)].iterrows():
            spot.append({"figure": "A_country", "series": f"{r['measure']}|{r['bucket']}",
                         "quarter_end": q.date(), "n": r["n_countries"],
                         "real_usd_2020": r["real_usd_2020"],
                         "share_of_book": r["share_of_book"],            # = global
                         "share_of_book_global": r["share_of_book_global"],
                         "share_of_book_eu": r["share_of_book_eu"],
                         "idx100": r["idx100"]})
    for q in SPOT_QUARTERS:
        for _, r in b.loc[b["quarter_end"].eq(q)].iterrows():
            spot.append({"figure": "B", "series": f"{r['split']}|{r['bucket']}",
                         "quarter_end": q.date(), "n": r["n_firms_cohort"],
                         "cn_links_t": r["cn_links_t"],
                         "growth_yoy": r["growth_yoy"],
                         "growth_qoq": r["growth_qoq"]})
    spot_df = pd.DataFrame(spot)
    spot_df.to_csv(F_SPOT, index=False)
    print(f"[S ] wrote {F_SPOT.name} ({len(spot_df):,} rows)")

    pd.set_option("display.width", 200)
    print("\n=== FIGURE A (firm level, holder_group = US) spot values ===")
    cols = ["split", "bucket", "quarter_end", "n_firms", "n_firms_held",
            "real_usd_2020", "share_of_book_global", "share_of_book_eu", "idx100"]
    for q in SPOT_QUARTERS:
        print(f"--- {q.date()}")
        t = us.loc[us["quarter_end"].eq(q), cols].copy()
        t["real_usd_2020"] = t["real_usd_2020"] / 1e9
        t = t.rename(columns={"real_usd_2020": "real_bn2020"})
        print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== FIGURE A (country level, M1 quartiles, holder_group = US) spot values ===")
    ccols = ["bucket", "quarter_end", "n_countries", "n_firms", "mean_m1_lag",
             "real_usd_2020", "share_of_book_global", "share_of_book_eu", "idx100"]
    for q in SPOT_QUARTERS:
        print(f"--- {q.date()}")
        t = cus.loc[cus["quarter_end"].eq(q), ccols].copy()
        t["real_usd_2020"] = t["real_usd_2020"] / 1e9
        t = t.rename(columns={"real_usd_2020": "real_bn2020"})
        print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== FIGURE B spot values ===")
    bcols = ["split", "bucket", "quarter_end", "n_firms_cohort", "cn_links_t",
             "cn_links_lag4", "growth_yoy", "growth_qoq", "mean_cn_links_t"]
    for q in SPOT_QUARTERS:
        print(f"--- {q.date()}")
        print(b.loc[b["quarter_end"].eq(q), bcols]
               .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    con.close()


if __name__ == "__main__":
    sys.exit(main())
