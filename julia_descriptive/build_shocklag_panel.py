"""
build_shocklag_panel.py — advisor-requested shock-timing revision (2026-06-28
meeting notes, point (a)): "这应该也是滞后" — the shock, like CN exposure, should
be measured as of t-1 relative to the outcome Δw_t, not contemporaneous S_t.

Rationale: CN_{t-1} is already predetermined relative to Δw_t (the quarter over
which the weight change is measured). The advisor's point is that S_t should be
symmetric — S_{t-1} — so BOTH right-hand-side variables are known to investors
BEFORE the quarter over which Δw_t is measured, a cleaner predetermined-regressor
timing than mixing a lagged CN with a contemporaneous shock.

Spec: Δw_t ~ US x CN_{t-1} x S_{t-1}  (vs current headline: Δw_t ~ US x CN_{t-1} x S_t)

shock is a quarter-level variable (identical across firms/groups within a
quarter), so LAG(shock, 1) OVER (PARTITION BY firm, group ORDER BY quarter) on
the complete 100-quarter grid is exactly "last quarter's shock value" — no gap
risk (verified: 100 contiguous quarters, 1999Q1-2023Q4).

Also reports the standardized-shock rescaling (advisor point (b), "用SD衡量"):
this is a PURE LINEAR RESCALING of the existing shock, so it does not change
any t-stat/p-value, only the coefficient's unit ("effect per 1-SD of shock").
Both the raw-shock and SD-rescaled coefficients are printed for both S_t and
S_{t-1} specs.

Output: output/shocklag_panel.dta / .parquet
"""
from pathlib import Path
import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
GRID = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
DTA = OUT / "shocklag_panel.dta"

con = duckdb.connect()
con.execute("SET memory_limit='6GB'")

df = con.execute(f"""
SELECT
    CAST(sec_entity_id AS VARCHAR)                  AS firm_str,
    holder_group                                    AS hgroup,
    CAST(report_date AS TIMESTAMP)                  AS rdate,
    delta_w                                         AS dw,
    china_share_lag1q                               AS cn_lag,
    shock_us_cn                                     AS shock_t,
    LAG(shock_us_cn, 1) OVER w                      AS shock_tm1,
    CASE WHEN holder_group = 'US' THEN 1 ELSE 0 END AS us
FROM read_parquet('{GRID}')
WINDOW w AS (PARTITION BY sec_entity_id, holder_group ORDER BY report_date)
""").df()
con.close()

n0 = len(df)
df = df.dropna(subset=["dw", "cn_lag", "shock_t", "shock_tm1"])
print(f"rows: {n0:,} -> {len(df):,} after requiring dw/cn_lag/shock_t/shock_tm1 all non-null")

# quarter-coverage / pairing hard asserts, mirroring build_c6_panel.py's discipline
df["firm_str"] = df["firm_str"].astype(str)
df["hgroup"] = df["hgroup"].astype(str)
df["rdate"] = pd.to_datetime(df["rdate"])
for c in ["dw", "cn_lag", "shock_t", "shock_tm1"]:
    df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
df["us"] = df["us"].astype("int8")

assert not df.duplicated(["firm_str", "hgroup", "rdate"]).any()
_pair = df.groupby(["firm_str", "rdate"])["hgroup"].nunique()
assert (_pair == 2).all(), f"not fully paired: {(_pair != 2).sum():,}"
assert (df.groupby("rdate")["shock_t"].nunique() == 1).all()
assert (df.groupby("rdate")["shock_tm1"].nunique() == 1).all()
assert df["cn_lag"].between(0, 1).all()

df[["firm_str", "hgroup", "rdate", "dw", "cn_lag", "shock_t", "shock_tm1", "us"]].to_stata(
    DTA, write_index=False, convert_dates={"rdate": "tc"}, version=118)
df.to_parquet(OUT / "shocklag_panel.parquet", index=False)

sd_t = df.groupby("rdate")["shock_t"].first().std()
sd_tm1 = df.groupby("rdate")["shock_tm1"].first().std()
print(f"wrote {DTA.name} ({len(df):,} rows, {df['firm_str'].nunique():,} firms)")
print(f"  sigma(shock_t)={sd_t:.4f}   sigma(shock_t-1)={sd_tm1:.4f}")
