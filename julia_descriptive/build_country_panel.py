# ============================================================================
# v3.1 PROVENANCE (P0 as-of holdings snapshot W=10 + EM zero-recode +
# MM v2 + DQ provable-impossibility filter; canonical rebuild 2026-08-09).
# This script reads the CURRENT canonical v3.1 artifacts:
#   output/I_ict_panel.parquet            04_us_ownership_european.jl, post-DQ
#                                         (built 2026-08-09 17:33)
#   output/country_measures_m1m2m3.csv    build_country_measures.py, v3.1
#                                         (built 2026-08-09 18:23)
#   output/gpr_quarterly_with_shock.parquet  shock series (vintage-invariant:
#                                         it is built from GPR inputs only, no
#                                         holdings dependence)
# A HARD mtime gate below (>= 2026-08-09 17:00) refuses to run on pre-v3.1
# vintages of the two holdings-derived inputs.
#
# HISTORY NOTE. The previous version of this file was the PRE-P0 ladder
# Step 1 script (outcome w_{c,g,t}, output country_panel.dta, runner
# run_country_ladder.do). It is preserved in git history
# (git log -- julia_descriptive/build_country_panel.py); its output
# country_panel.dta stays frozen as a pre-P0 artifact and is NOT rebuilt here.
# ============================================================================

"""
build_country_panel.py — STEP 4 (advisor meeting 2026-08-04): the advisors'
country-by-quarter baseline regression panel, the counterpart of the
state-level design in their GFE paper, on v3.1.

Unit: (security country c, quarter t), WIDE in the investor group. From the
post-DQ I_ict_panel.parquet (grain: firm x sec_country x investor_country x
quarter, I_ict = SUM adj_mv over EQ/AD holdings):

  H^{US}_{c,t}    = SUM I_ict over firms in c, investor_country  = 'US'
  H^{NONUS}_{c,t} = SUM I_ict over firms in c, investor_country <> 'US'

Outcomes:
  dlog_us   = ln H^{US}_{c,t} - ln H^{US}_{c,t-1}      MAIN (defined only when
              both quarters have H > 0 and t-1 is the adjacent calendar quarter)
  dlev_us   = (H^{US}_{c,t} - H^{US}_{c,t-1}) / 1e9    ALTERNATIVE (level change,
              $bn, defined on the in-span zero-filled grid)
  dlh_diff  = dlog_us - dlog_nonus                     country-level analogue of
              the DDD (US-minus-NONUS differenced outcome)

Zero-fill rule: within a country's coverage span (first..last quarter with ANY
I_ict row for that country), a missing (c,t) cell is a genuine $0 book and is
zero-filled; outside the span H is missing (coverage, not economics). ln(0) is
undefined, so zero cells enter dlev but drop from dlog.

Regressors merged in:
  m1/m2/m3        country x quarter China-exposure measures, universe =
                  'ownership_matched' (is_primary=1) from
                  country_measures_m1m2m3.csv, plus their 1-quarter lags
                  (calendar lags on the complete country x quarter grid)
  shock, s_lag    shock_us_cn and its 1-quarter lag from
                  gpr_quarterly_with_shock.parquet (series is contiguous
                  1960Q1-2026Q1, so shift(1) IS the calendar lag);
                  S_{t-1} (s_lag) is the PRIMARY timing (advisor-directed)

Output: output/country_panel_step4.dta (28 countries x 100 quarters
1999Q1-2023Q4 = 2,800 rows), consumed by run_country_panel.do.
Rotation discipline: REFUSES to overwrite an existing output; rotate the old
file to *_r2pre first.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
_env_out = os.environ.get("DPN_OUT_DIR", "").strip()
OUT = Path(_env_out).resolve() if _env_out else PROJ / "output"

ICT = OUT / "I_ict_panel.parquet"
ICT_META = OUT / "I_ict_panel.parquet.meta.json"
MEASURES = OUT / "country_measures_m1m2m3.csv"
GPR = OUT / "gpr_quarterly_with_shock.parquet"
DTA = OUT / "country_panel_step4.dta"

# v3.1 canonical rebuild window opened 2026-08-09 17:00 local time.
VMIN = datetime(2026, 8, 9, 17, 0, 0)

# 28 European jurisdictions, verbatim from 00_setup.jl / build_country_measures.py.
EU_COUNTRIES = ("GB", "DE", "FR", "NL", "CH", "IT", "ES", "SE", "DK", "NO",
                "FI", "BE", "AT", "IE", "LU", "PT", "PL", "CZ", "HU", "GR",
                "RO", "SK", "SI", "BG", "HR", "EE", "LV", "LT")
EU_SQL = "(" + ",".join(f"'{c}'" for c in EU_COUNTRIES) + ")"

Q_FIRST, Q_LAST = "1999Q1", "2023Q4"  # v3.1 I_ict span (frozen vintage)


def assert_fresh(p: Path, label: str) -> None:
    """HARD freshness gate: refuse pre-v3.1 vintages."""
    if not p.is_file():
        raise FileNotFoundError(f"missing input {label}: {p}")
    mt = datetime.fromtimestamp(p.stat().st_mtime)
    if mt < VMIN:
        raise RuntimeError(
            f"STALE INPUT {label}: mtime {mt:%Y-%m-%d %H:%M:%S} < "
            f"{VMIN:%Y-%m-%d %H:%M} — pre-v3.1 vintage. Rebuild upstream "
            f"(04_us_ownership_european.jl / build_country_measures.py) first.")


def main() -> None:
    # ------------------------------------------------------------------
    # [0] Freshness + provenance gates (hard-fail, no fallback)
    # ------------------------------------------------------------------
    assert_fresh(ICT, "I_ict_panel.parquet")
    assert_fresh(MEASURES, "country_measures_m1m2m3.csv")
    if not GPR.is_file():
        raise FileNotFoundError(f"missing input gpr_quarterly_with_shock: {GPR}")
    # cross-check: the manifest 04 wrote next to the parquet must also be v3.1
    # (guards against a copied-back file carrying a fresh fs mtime).
    meta = json.loads(ICT_META.read_text(encoding="utf-8"))
    build_ts = datetime.fromisoformat(meta["build_ts"])
    if build_ts < VMIN:
        raise RuntimeError(
            f"STALE MANIFEST: I_ict_panel build_ts {build_ts} < {VMIN} — "
            f"parquet content predates the v3.1 (post-DQ) rebuild.")
    print(f"[0] freshness gate PASS (I_ict build_ts {meta['build_ts']}, "
          f"row_count {meta['row_count']:,})")

    # rotation discipline: never silently clobber the canonical output
    if DTA.exists():
        raise RuntimeError(
            f"target {DTA} already exists — rename it to "
            f"country_panel_step4_r2pre.dta (rotation rule) before re-running.")

    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    tmp = Path("E:/duckdb_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")

    # ------------------------------------------------------------------
    # [1] Country x quarter US / NONUS holdings of EU firms (post-DQ I_ict)
    # ------------------------------------------------------------------
    print("[1] Aggregating I_ict to (sec_country, quarter) x {US, NONUS}...")
    cq = con.execute(f"""
        SELECT sec_country,
               report_date,
               SUM(CASE WHEN investor_country =  'US' THEN I_ict ELSE 0 END) AS h_us,
               SUM(CASE WHEN investor_country <> 'US' THEN I_ict ELSE 0 END) AS h_nonus,
               COUNT(DISTINCT sec_entity_id)                                 AS n_firms_hold
        FROM read_parquet('{ICT.as_posix()}')
        WHERE sec_country IN {EU_SQL}
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()
    con.close()

    cq["report_date"] = pd.to_datetime(cq["report_date"])
    cq["q"] = cq["report_date"].dt.to_period("Q")
    # every report_date must be a calendar quarter-end (v3.1 as-of rule)
    qend = cq["q"].dt.to_timestamp(how="end").dt.normalize()
    bad_dates = cq.loc[qend != cq["report_date"], "report_date"].unique()
    assert len(bad_dates) == 0, f"non-quarter-end report_dates: {bad_dates[:5]}"
    assert not cq.duplicated(["sec_country", "q"]).any()
    assert set(cq["sec_country"]) == set(EU_COUNTRIES), "EU-28 country set drifted"
    assert (cq["h_us"] >= 0).all() and (cq["h_nonus"] >= 0).all()
    # frozen v3.1 span: any drift means a new vintage -> this file needs review
    assert str(cq["q"].min()) == Q_FIRST and str(cq["q"].max()) == Q_LAST, (
        f"I_ict quarter span {cq['q'].min()}..{cq['q'].max()} != frozen v3.1 "
        f"{Q_FIRST}..{Q_LAST} — new vintage, review this script before running.")
    n_q = cq["q"].nunique()
    print(f"    observed cells: {len(cq):,} ({cq['sec_country'].nunique()} countries, "
          f"{n_q} quarters, zero-US cells: {(cq['h_us'] == 0).sum()})")

    # ------------------------------------------------------------------
    # [2] Complete country x quarter grid, coverage spans, outcomes
    # ------------------------------------------------------------------
    print("[2] Complete grid + in-span zero-fill + outcomes...")
    quarters = pd.period_range(Q_FIRST, Q_LAST, freq="Q")
    grid = pd.MultiIndex.from_product(
        [sorted(EU_COUNTRIES), quarters], names=["sec_country", "q"]
    ).to_frame(index=False)
    df = grid.merge(cq.drop(columns=["report_date"]),
                    on=["sec_country", "q"], how="left", validate="1:1")
    df["present"] = df["h_us"].notna().astype("int8")

    span = (df[df["present"] == 1].groupby("sec_country")["q"]
            .agg(q_first="min", q_last="max").reset_index())
    df = df.merge(span, on="sec_country", how="left", validate="m:1")
    df["in_span"] = ((df["q"] >= df["q_first"]) & (df["q"] <= df["q_last"])).astype("int8")
    for g in ("us", "nonus"):
        df[f"h_{g}"] = np.where(df["in_span"] == 1, df[f"h_{g}"].fillna(0.0), np.nan)

    df = df.sort_values(["sec_country", "q"]).reset_index(drop=True)
    # the grid is complete, so groupby(...).diff() IS the calendar-quarter lag
    for g in ("us", "nonus"):
        h = df[f"h_{g}"]
        df[f"_lnh_{g}"] = np.log(h.where(h > 0))
        df[f"dlog_{g}"] = df.groupby("sec_country")[f"_lnh_{g}"].diff()
        df[f"dlev_{g}"] = df.groupby("sec_country")[f"h_{g}"].diff() / 1e9
    df["dlh_diff"] = df["dlog_us"] - df["dlog_nonus"]
    df = df.drop(columns=["_lnh_us", "_lnh_nonus", "q_first", "q_last"])
    n_zero_filled = int(((df["in_span"] == 1) & (df["present"] == 0)).sum())
    print(f"    grid rows: {len(df):,}; in-span zero-filled cells: {n_zero_filled}")

    # ------------------------------------------------------------------
    # [3] M1/M2/M3 (primary universe) + calendar lags
    # ------------------------------------------------------------------
    print("[3] Merging country measures (universe=ownership_matched)...")
    m = pd.read_csv(MEASURES)
    m = m[m["is_primary"] == 1].copy()
    m["q"] = pd.PeriodIndex(pd.to_datetime(m["quarter_end"]), freq="Q")
    assert not m.duplicated(["country", "q"]).any()
    assert set(m["country"]) <= set(EU_COUNTRIES)
    assert m["q"].min() >= quarters[0] and m["q"].max() <= quarters[-1], (
        "measures quarters extend beyond the holdings grid — vintage mismatch")
    m = m.rename(columns={"country": "sec_country",
                          "M1": "m1", "M2": "m2", "M3": "m3"})
    df = df.merge(m[["sec_country", "q", "m1", "m2", "m3", "n_firms", "n_firms_m2"]],
                  on=["sec_country", "q"], how="left", validate="1:1")
    for c in ("m1", "m2", "m3"):
        df[f"{c}_lag"] = df.groupby("sec_country")[c].shift(1)
    print(f"    measure cells merged: {df['m3'].notna().sum():,} "
          f"(M1 missing in {df['m1'].isna().sum() - df['m3'].isna().sum()} "
          f"cells with zero SC links)")

    # ------------------------------------------------------------------
    # [4] Shock series: shock_us_cn (S_t, diagnostic) + s_lag (S_{t-1}, PRIMARY)
    # ------------------------------------------------------------------
    print("[4] Merging shock series...")
    con2 = duckdb.connect()
    g = con2.execute(f"""
        SELECT quarter_end, shock_us_cn
        FROM read_parquet('{GPR.as_posix()}')
        ORDER BY quarter_end
    """).df()
    con2.close()
    g["q"] = pd.PeriodIndex(pd.to_datetime(g["quarter_end"]), freq="Q")
    g = g.sort_values("q").reset_index(drop=True)
    ords = pd.Series(pd.PeriodIndex(g["q"]).asi8)
    assert (ords.diff().dropna() == 1).all(), "gpr quarterly series not contiguous"
    g["s_lag"] = g["shock_us_cn"].shift(1)  # contiguous series -> calendar lag
    df = df.merge(g[["q", "shock_us_cn", "s_lag"]].rename(
        columns={"shock_us_cn": "shock"}), on="q", how="left", validate="m:1")
    assert df["shock"].notna().all() and df["s_lag"].notna().all()

    # ------------------------------------------------------------------
    # [5] Estimation-sample census (printed for the record; Stata re-drops)
    # ------------------------------------------------------------------
    assert len(df) == len(EU_COUNTRIES) * len(quarters)
    print("[5] Estimation-sample census (rows with outcome, s_lag and M lag):")
    for mm in ("m1", "m2", "m3"):
        a = df.dropna(subset=["dlog_us", "s_lag", f"{mm}_lag"])
        b = df.dropna(subset=["dlh_diff", "s_lag", f"{mm}_lag"])
        print(f"    {mm.upper()}: us_dlog N={len(a):,} "
              f"({a['sec_country'].nunique()} countries) | "
              f"us_minus_nonus N={len(b):,} ({b['sec_country'].nunique()} countries)")

    print("    spot values (2018Q4):")
    spot = df[df["q"] == pd.Period("2018Q4", freq="Q")]
    spot = spot[spot["sec_country"].isin(["DE", "GB", "FR"])]
    print(spot[["sec_country", "h_us", "dlog_us", "dlh_diff", "m1", "m3"]]
          .assign(h_us_bn=lambda d: d["h_us"] / 1e9).drop(columns="h_us")
          .to_string(index=False))

    # ------------------------------------------------------------------
    # [6] Write Stata panel
    # ------------------------------------------------------------------
    print("[6] Writing...")
    df["rdate"] = df["q"].dt.to_timestamp(how="end").dt.normalize()
    keep = ["sec_country", "rdate",
            "h_us", "h_nonus", "dlog_us", "dlog_nonus", "dlh_diff",
            "dlev_us", "dlev_nonus",
            "m1", "m2", "m3", "m1_lag", "m2_lag", "m3_lag",
            "shock", "s_lag",
            "n_firms", "n_firms_m2", "n_firms_hold", "present", "in_span"]
    out = df[keep].copy()
    for c in keep:
        if c in ("sec_country", "rdate", "present", "in_span"):
            continue
        out[c] = pd.to_numeric(out[c], errors="raise").astype("float64")
    out.to_stata(DTA, write_index=False, convert_dates={"rdate": "tc"},
                 version=118)
    print(f"    wrote {DTA} ({len(out):,} rows = "
          f"{out['sec_country'].nunique()} countries x "
          f"{out['rdate'].nunique()} quarters)")


if __name__ == "__main__":
    main()
