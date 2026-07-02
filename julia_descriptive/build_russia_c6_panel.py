"""
build_russia_c6_panel.py

ISOMORPHIC copy of build_c6_panel.py for the Russia positive control (Third
external review round, R3-A #1 / P0 #3). Same grid, same asserts, same
estimation-sample filter; only the exposure/shock columns differ (russia_share
instead of china_share). Reads merged_us_ru_zero_filled.parquet (06_russia_grid.jl).

Columns written (matches what run_russia_headline.do `use`s):
  firm_str  (= sec_entity_id, string)
  hgroup    (= holder_group: 'US' / 'NONUS')
  rdate     (= report_date, datetime64 -> Stata %tc)
  dw        (= delta_w)
  ru_lag    (= russia_share_lag1q)
  shock     (= shock_us_ru)
  us        (= 1 if holder_group == 'US' else 0)

Sample filter:
  delta_w IS NOT NULL AND russia_share_lag1q IS NOT NULL AND shock_us_ru IS NOT NULL
"""

from pathlib import Path

import duckdb
import pandas as pd

PROJ_DIR = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT_DIR = PROJ_DIR / "output"

PANEL_PARQUET = OUT_DIR / "merged_us_ru_zero_filled.parquet"
PANEL_DTA = OUT_DIR / "c6_panel_russia.dta"

assert PANEL_PARQUET.exists(), (
    f"Missing input parquet: {PANEL_PARQUET}\nRe-run julia_descriptive/06_russia_grid.jl first."
)

print(f"[1/3] Reading {PANEL_PARQUET}")
panel_uri = str(PANEL_PARQUET).replace("\\", "/")

con = duckdb.connect()
df = con.execute(f"""
    SELECT
        CAST(sec_entity_id AS VARCHAR)                  AS firm_str,
        holder_group                                    AS hgroup,
        CAST(report_date AS TIMESTAMP)                  AS rdate,
        delta_w                                         AS dw,
        russia_share_lag1q                              AS ru_lag,
        shock_us_ru                                     AS shock,
        CASE WHEN holder_group = 'US' THEN 1 ELSE 0 END AS us
    FROM read_parquet('{panel_uri}')
    WHERE delta_w             IS NOT NULL
      AND russia_share_lag1q  IS NOT NULL
      AND shock_us_ru         IS NOT NULL
""").df()

# Same NULL-aware group-quarter weight-sum integrity check as build_c6_panel.py,
# on the FULL grid (not the filtered estimation sample).
_wchk = con.execute(f"""
    WITH cell AS (
        SELECT holder_group, report_date,
               SUM(portfolio_weight_eu)   AS s,
               COUNT(portfolio_weight_eu) AS nn,
               SUM(COALESCE(I_ict, 0))    AS tot_hold
        FROM read_parquet('{panel_uri}')
        GROUP BY 1, 2)
    SELECT
        MAX(CASE WHEN nn > 0 THEN ABS(s - 1) END)          AS max_dev,
        COUNT(CASE WHEN nn > 0 THEN 1 END)                 AS n_checked,
        COUNT(CASE WHEN nn = 0 AND tot_hold > 0 THEN 1 END) AS n_held_but_null
    FROM cell
""").df()
_wdev = _wchk["max_dev"].iloc[0]
_nchk = int(_wchk["n_checked"].iloc[0])
_nbug = int(_wchk["n_held_but_null"].iloc[0])
assert _nbug == 0, f"{_nbug} (group, quarter) cell(s) have positive holdings but all-NULL weights"
assert _wdev < 1e-9, f"weights do not sum to 1: max abs dev {_wdev:.2e} over {_nchk} cells"
con.close()

df["firm_str"] = df["firm_str"].astype(str)
df["hgroup"] = df["hgroup"].astype(str)
df["rdate"] = pd.to_datetime(df["rdate"])
df["dw"] = pd.to_numeric(df["dw"], errors="raise").astype("float64")
df["ru_lag"] = pd.to_numeric(df["ru_lag"], errors="raise").astype("float64")
df["shock"] = pd.to_numeric(df["shock"], errors="raise").astype("float64")
df["us"] = df["us"].astype("int8")

n = len(df)
print(f"[2/3] Filtered panel rows: {n:,}")
print(df["hgroup"].value_counts().to_string())
print(f"       rdate range: {df['rdate'].min()} -> {df['rdate'].max()}")
print(f"       n unique firm_str: {df['firm_str'].nunique():,}")
print(f"       ru_lag>0 share (any Russia exposure): {(df['ru_lag']>0).mean():.4f}")

assert n > 0
assert df["dw"].notna().all()
assert df["ru_lag"].notna().all()
assert df["shock"].notna().all()
assert set(df["hgroup"].unique()) <= {"US", "NONUS"}
assert ((df["hgroup"] == "US") == (df["us"] == 1)).all()
assert not df.duplicated(["firm_str", "hgroup", "rdate"]).any(), "duplicate rows"
_pair = df.groupby(["firm_str", "rdate"])["hgroup"].nunique()
assert (_pair == 2).all(), f"not fully paired: {(_pair != 2).sum():,} firm-quarters"
assert (df.groupby("rdate")["shock"].nunique() == 1).all(), "shock varies within a quarter"
assert df["ru_lag"].between(0, 1).all(), "ru_lag outside [0,1]"
_q = df["rdate"].dt.to_period("Q").drop_duplicates().sort_values()
_expected = pd.period_range(_q.iloc[0], _q.iloc[-1], freq="Q")
assert len(_q) == len(_expected) and (_q.to_numpy() == _expected.to_numpy()).all(), (
    f"gap in quarter coverage: {len(_q)} vs {len(_expected)} expected"
)
print(f"       quarter coverage: {len(_q)} contiguous quarters {_q.iloc[0]} -> {_q.iloc[-1]}")

print(f"[3/3] Writing Stata file: {PANEL_DTA}")
df.to_stata(PANEL_DTA, write_index=False, convert_dates={"rdate": "tc"}, version=118)
print(f"      done. {n:,} rows written.")
