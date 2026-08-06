"""
build_c6_panel.py

Build the Stata-ready C6 firm-quarter-holdergroup panel for 07_regression.do
from the rebuilt merged_us_eu_zero_filled.parquet (now carrying BACKWARD Δw:
delta_w = w_t - w_{t-1}, per slide 4 formula).

Columns written (matches what 07_regression.do `use`s):
  firm_str  (= sec_entity_id, string)
  hgroup    (= holder_group: 'US' / 'NONUS')
  rdate     (= report_date, datetime64 -> Stata %tc)
  dw        (= delta_w)
  cn_lag    (= china_share_lag1q)
  shock     (= shock_us_cn)
  us        (= 1 if holder_group == 'US' else 0)

Sample filter:
  delta_w IS NOT NULL AND china_share_lag1q IS NOT NULL AND shock_us_cn IS NOT NULL

Single duckdb SELECT projects + filters in one shot — no separate scans, no
row-order misalignment.

Stata gotcha: pandas.to_stata writes datetime64 as %tc (milliseconds since
1960-01-01); 07_regression.do already does `gen rd_day = dofc(rdate)`.
"""

import os
from pathlib import Path

import duckdb
import pandas as pd

PROJ_DIR = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
# (EM-FIX-5/7, 2026-08-06) honour the same DPN_OUT_DIR override 00_setup.jl uses,
# so this builder cannot read a stale panel from C: while the Julia chain writes
# the rebuilt one to E:. The .do consumers do NOT honour it — see 00_setup.jl.
_env_out = os.environ.get("DPN_OUT_DIR", "").strip()
OUT_DIR  = Path(_env_out).resolve() if _env_out else PROJ_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
if _env_out:
    print(f"[DPN_OUT_DIR] reading/writing {OUT_DIR} (override in force)")

PANEL_PARQUET = OUT_DIR / "merged_us_eu_zero_filled.parquet"
PANEL_DTA     = OUT_DIR / "c6_panel.dta"

assert PANEL_PARQUET.exists(), (
    f"Missing input parquet: {PANEL_PARQUET}\n"
    f"Re-run julia_descriptive/06_cartesian_grid.jl first."
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
        china_share_lag1q                               AS cn_lag,
        -- (DIRECTION FIX, 2026-08-02) directional link-count shares, lagged;
        -- sell_lag + buy_lag = cn_lag row-wise (additive decomposition).
        sell_share_lag1q                                AS sell_lag,
        buy_share_lag1q                                 AS buy_lag,
        shock_us_cn                                     AS shock,
        CASE WHEN holder_group = 'US' THEN 1 ELSE 0 END AS us,
        -- (EM-FIX-6, 2026-08-06) ATTRIBUTION COLUMNS. The written update has to
        -- separate the SNAPSHOT effect (03's quarter rule) from the SAMPLE-
        -- EXPANSION effect (02's zero recode). 06_cartesian_grid.jl:401 carries
        -- zero_recode_flag_lag1q into the parquet, and flag = 0 is EXACTLY the
        -- pre-change codable set (old rule NULLIF(n_supplychain_links, 0); new
        -- flag = 0 means n_supplychain_links > 0). Projecting it here makes the
        -- attribution a two-line split of ONE build instead of a full 02-06
        -- re-run under two different rules:
        --     run A (new snapshot + OLD missing rule) :  if zr_lag == 0
        --     run B (new snapshot + zero recode)      :  full sample
        -- NULL is mapped to the sentinel -1 in pandas below, because to_stata
        -- would otherwise widen the column to float and write NaN.
        COALESCE(zero_recode_flag_lag1q,   -1)          AS zr_lag,
        COALESCE(n_supplychain_links_lag1q, -1)         AS nsc_lag,
        COALESCE(CAST(revere_pit_present_lag1q AS INTEGER), -1) AS pit_lag
    FROM read_parquet('{panel_uri}')
    WHERE delta_w           IS NOT NULL
      AND china_share_lag1q IS NOT NULL
      AND shock_us_cn       IS NOT NULL
""").df()

# Upstream integrity on the FULL grid (external-review P2): portfolio weights
# must sum to 1 within each (group, quarter). Checked on the full parquet, NOT
# the filtered estimation sample above — filtering drops rows and would break
# the simplex adding-up. (Observed max abs deviation ~4.4e-16 = float noise.)
# Simplex integrity on the FULL grid (external-review P2), NULL-aware.
# portfolio_weight_eu = w = H/T is NULL by construction (06 ELSE NULL) when a
# group's European book is empty that quarter (T_{g,t}=0) — e.g. NONUS holds
# nothing in 1999Q1, I_ict all 0. Such all-NULL cells are LEGITIMATE and allowed.
# The two things that WOULD be bugs, and are hard-failed here:
#   (i)  a cell with positive holdings (SUM(I_ict)>0) but all-NULL weights;
#   (ii) any cell that DOES carry weights whose sum is not 1.
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
assert _nbug == 0, \
    f"{_nbug} (group, quarter) cell(s) have positive holdings but all-NULL weights — 06 weight bug"
assert _wdev < 1e-9, \
    f"weights do not sum to 1 by (group, quarter): max abs dev {_wdev:.2e} over {_nchk} non-empty cells"
con.close()

df["firm_str"] = df["firm_str"].astype(str)
df["hgroup"]   = df["hgroup"].astype(str)
df["rdate"]    = pd.to_datetime(df["rdate"])
df["dw"]       = pd.to_numeric(df["dw"],     errors="raise").astype("float64")
df["cn_lag"]   = pd.to_numeric(df["cn_lag"], errors="raise").astype("float64")
df["sell_lag"] = pd.to_numeric(df["sell_lag"], errors="raise").astype("float64")
df["buy_lag"]  = pd.to_numeric(df["buy_lag"],  errors="raise").astype("float64")
df["shock"]    = pd.to_numeric(df["shock"],  errors="raise").astype("float64")
df["us"]       = df["us"].astype("int8")
# (EM-FIX-6) attribution columns. int8 for the two flags; nsc_lag is a link count
# that can exceed 127, so int32. Sentinel -1 == "not codable / NULL upstream".
df["zr_lag"]   = pd.to_numeric(df["zr_lag"],  errors="raise").astype("int8")
df["pit_lag"]  = pd.to_numeric(df["pit_lag"], errors="raise").astype("int8")
df["nsc_lag"]  = pd.to_numeric(df["nsc_lag"], errors="raise").astype("int32")

n = len(df)
print(f"[2/3] Filtered panel rows: {n:,}")
# (EM-FIX-6) attribution census on the estimation sample actually written.
# zr_lag == 0  -> cell was codable under the OLD missing rule (pre-change arm)
# zr_lag == 1  -> recoded zero, competitor/partner-only links
# zr_lag == 2  -> recoded zero, no active links at all
# zr_lag == -1 -> flag was NULL upstream while cn_lag was not (should be 0 rows;
#                 if not, 06's lag propagation and 02's flag disagree)
print("       EM-FIX-6 attribution census (zr_lag on the estimation sample):")
_zr = df["zr_lag"].value_counts().sort_index()
for _k, _v in _zr.items():
    _lbl = {0: "old-rule codable (pre-change arm)",
            1: "recoded zero: competitor/partner only",
            2: "recoded zero: no active links",
            -1: "NULL upstream (INVESTIGATE)"}.get(int(_k), "unexpected")
    print(f"         zr_lag={int(_k):>3}  n={_v:>12,}  firms={df.loc[df['zr_lag'] == _k, 'firm_str'].nunique():>7,}  {_lbl}")
_n_old = int((df["zr_lag"] == 0).sum())
_f_old = df.loc[df["zr_lag"] == 0, "firm_str"].nunique()
print(f"       run A (new snapshot + OLD missing rule, `if zr_lag==0`): "
      f"N={_n_old:,}  firms={_f_old:,}")
print(f"       run B (new snapshot + zero recode, full sample):        "
      f"N={n:,}  firms={df['firm_str'].nunique():,}")
if int((df["zr_lag"] == -1).sum()) > 0:
    print(f"       !! WARNING: {int((df['zr_lag'] == -1).sum()):,} rows have a non-null cn_lag "
          f"but a NULL zero_recode_flag_lag1q — 02/06 provenance disagreement.")
print("       Breakdown by hgroup:")
print(df["hgroup"].value_counts().to_string())
print("       us flag tab:")
print(df["us"].value_counts().to_string())
print(f"       rdate range: {df['rdate'].min()} -> {df['rdate'].max()}")
print(f"       n unique firm_str: {df['firm_str'].nunique():,}")

# Sanity assertions (hardened per external review — hard fail, not print)
assert n > 0, "Empty panel after filter — check upstream parquet."
assert df["dw"].notna().all(),     "dw has NaN after filter"
assert df["cn_lag"].notna().all(), "cn_lag has NaN after filter"
assert df["shock"].notna().all(),  "shock has NaN after filter"
assert set(df["hgroup"].unique()) <= {"US", "NONUS"}, (
    f"Unexpected hgroup values: {df['hgroup'].unique()}"
)
assert ((df["hgroup"] == "US") == (df["us"] == 1)).all(), "us flag mismatch with hgroup"
# (a) unique key
assert not df.duplicated(["firm_str", "hgroup", "rdate"]).any(), \
    "duplicate (firm_str, hgroup, rdate) rows"
# (b) every firm-quarter paired US + NONUS (within-firm-quarter identification needs both)
_pair = df.groupby(["firm_str", "rdate"])["hgroup"].nunique()
assert (_pair == 2).all(), \
    f"panel not fully paired: {(_pair != 2).sum():,} firm-quarters lack both US and NONUS"
# (c) shock is one common value per quarter (no within-quarter variation)
assert (df.groupby("rdate")["shock"].nunique() == 1).all(), \
    "shock varies within a quarter — expected a single common S_t per quarter"
# (d) cn_lag in [0, 1] (it is a share)
assert df["cn_lag"].between(0, 1).all(), "cn_lag outside [0,1]"
# (d2) DIRECTION FIX: additive decomposition must survive the pipeline —
#      sell_lag + buy_lag == cn_lag on every estimation row (float tolerance).
assert df["sell_lag"].notna().all() and df["buy_lag"].notna().all(), \
    "sell_lag/buy_lag NaN where cn_lag is non-null — 06 lag propagation bug"
assert (df["sell_lag"] + df["buy_lag"] - df["cn_lag"]).abs().max() < 1e-12, \
    "sell_lag + buy_lag != cn_lag — direction split lost additivity in the pipeline"
# (e) quarter coverage is contiguous — no missing quarters in the estimation span
_q = df["rdate"].dt.to_period("Q").drop_duplicates().sort_values()
_expected = pd.period_range(_q.iloc[0], _q.iloc[-1], freq="Q")
assert len(_q) == len(_expected) and (_q.to_numpy() == _expected.to_numpy()).all(), (
    f"gap in quarter coverage: {len(_q)} distinct quarters vs {len(_expected)} expected "
    f"between {_q.iloc[0]} and {_q.iloc[-1]}"
)
print(f"       quarter coverage: {len(_q)} contiguous quarters "
      f"{_q.iloc[0]} -> {_q.iloc[-1]}")

print(f"[3/3] Writing Stata file: {PANEL_DTA}")
df.to_stata(
    PANEL_DTA,
    write_index=False,
    convert_dates={"rdate": "tc"},
    version=118,
)
print(f"      done. {n:,} rows written.")
