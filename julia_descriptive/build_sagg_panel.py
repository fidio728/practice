"""
build_sagg_panel.py — within-quarter AGGREGATED shock (advisor shock-timing
revision, econometric review 2026-07-02).

WHY. Delta_w_t accrues over all three months of quarter t, but the current
S_t is only the quarter-END month's AR(1) innovation: the news surprises of
months 1-2 are absent from the regressor while investors' responses to them
ARE in the outcome -> classical measurement error, attenuating beta_3.
Fix: S_t^agg = sum of the three monthly AR(1) innovations within quarter t
(innovations are serially uncorrelated by construction, so the quarter's
total news surprise is their sum). This aligns the regressor's information
window exactly with the outcome's accrual window.

Also provides S_{t-1}^agg for the distributed-lag joint test
(US x CN x S_t^agg AND US x CN x S_{t-1}^agg in one regression: joint Wald =
"response at ANY horizon?", cumulative sum = total half-year response).

Consistency check built in: the quarter-end-month residual in the monthly file
must equal the main panel's stamped shock_us_cn quarter by quarter (same AR(1),
same source) -- hard-asserted.

Output: output/sagg_panel.dta
"""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
GRID = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
MONTHLY = OUT / "country_pair_shocks_monthly.csv"
DTA = OUT / "sagg_panel.dta"

# ---------------------------------------------------------------
# 1. Monthly residuals -> quarterly aggregate (sum of 3 monthly innovations).
# ---------------------------------------------------------------
m = pd.read_csv(MONTHLY, usecols=["month_end", "shock_us_cn"])
m["month_end"] = pd.to_datetime(m["month_end"])
m["q"] = m["month_end"].dt.to_period("Q")

# quarter must have all 3 monthly residuals to form the aggregate
agg = m.groupby("q").agg(
    s_agg=("shock_us_cn", "sum"),
    n_months=("shock_us_cn", "count"),
    s_qend=("shock_us_cn", "last"),  # last month of quarter = quarter-end residual
).reset_index()
agg = agg[agg["n_months"] == 3].copy()
agg["quarter_end"] = agg["q"].dt.to_timestamp("Q")
agg["s_agg_l1"] = agg["s_agg"].shift(1)
# guard the shift: only valid if the previous row is the immediately preceding quarter
# (Period subtraction returns an offset object, so compare integer ordinals instead)
agg["qi"] = agg["q"].astype("int64")
agg["prev_q_ok"] = agg["qi"].diff() == 1
agg.loc[~agg["prev_q_ok"].fillna(False), "s_agg_l1"] = np.nan

print(f"[1/3] quarters with all 3 monthly residuals: {len(agg)}")

# ---------------------------------------------------------------
# 2. Merge onto the C6 grid + consistency check vs stamped shock.
# ---------------------------------------------------------------
con = duckdb.connect()
con.execute("SET memory_limit='6GB'")
con.register("agg", agg[["quarter_end", "s_agg", "s_agg_l1", "s_qend"]])

df = con.execute(f"""
SELECT
    CAST(g.sec_entity_id AS VARCHAR)                  AS firm_str,
    g.holder_group                                    AS hgroup,
    CAST(g.report_date AS TIMESTAMP)                  AS rdate,
    g.delta_w                                         AS dw,
    g.china_share_lag1q                               AS cn_lag,
    g.shock_us_cn                                     AS shock_stamped,
    a.s_agg, a.s_agg_l1, a.s_qend,
    CASE WHEN g.holder_group = 'US' THEN 1 ELSE 0 END AS us
FROM read_parquet('{GRID}') g
LEFT JOIN agg a ON CAST(g.report_date AS DATE) = CAST(a.quarter_end AS DATE)
WHERE g.delta_w IS NOT NULL
  AND g.china_share_lag1q IS NOT NULL
  AND g.shock_us_cn IS NOT NULL
""").df()
con.close()

# consistency: the monthly file's quarter-end residual == the panel's stamped shock
chk = df.dropna(subset=["s_qend"])
max_dev = (chk["shock_stamped"] - chk["s_qend"]).abs().max()
assert max_dev < 1e-9, f"quarter-end residual mismatch vs stamped shock: {max_dev:.2e}"
print(f"[2/3] consistency check passed: monthly-file quarter-end residual == stamped shock (max dev {max_dev:.1e})")

n0 = len(df)
df = df.dropna(subset=["dw", "cn_lag", "s_agg", "s_agg_l1"])
print(f"      rows: {n0:,} -> {len(df):,} after requiring s_agg + s_agg_l1")

# ---------------------------------------------------------------
# 3. Hard asserts + write.
# ---------------------------------------------------------------
df["firm_str"] = df["firm_str"].astype(str)
df["hgroup"] = df["hgroup"].astype(str)
df["rdate"] = pd.to_datetime(df["rdate"])
for c in ["dw", "cn_lag", "s_agg", "s_agg_l1"]:
    df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
df["us"] = df["us"].astype("int8")

assert not df.duplicated(["firm_str", "hgroup", "rdate"]).any()
_pair = df.groupby(["firm_str", "rdate"])["hgroup"].nunique()
assert (_pair == 2).all(), f"not fully paired: {(_pair != 2).sum():,}"
assert (df.groupby("rdate")["s_agg"].nunique() == 1).all()
assert (df.groupby("rdate")["s_agg_l1"].nunique() == 1).all()
assert df["cn_lag"].between(0, 1).all()

q = df.groupby("rdate")[["s_agg", "s_agg_l1", "shock_stamped"]].first()
print(f"[3/3] quarters: {len(q)}   sigma(s_agg)={q['s_agg'].std(ddof=1):.4f}   "
      f"sigma(stamped)={q['shock_stamped'].std(ddof=1):.4f}")
print(f"      corr(s_agg, stamped quarter-end) = {q['s_agg'].corr(q['shock_stamped']):.3f}")
print(f"      corr(s_agg, s_agg_l1) = {q['s_agg'].corr(q['s_agg_l1']):.3f}  (should be ~0: innovations serially uncorrelated)")

df[["firm_str", "hgroup", "rdate", "dw", "cn_lag", "s_agg", "s_agg_l1", "us"]].to_stata(
    DTA, write_index=False, convert_dates={"rdate": "tc"}, version=118)
print(f"      wrote {DTA.name} ({len(df):,} rows, {df['firm_str'].nunique():,} firms)")
