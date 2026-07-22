"""
build_firm_ladder_panel.py — LADDER STEP 2 (advisor-confirmed): same specification
as step 1 but with the FIRM dimension. No shock yet (that is step 3 = the existing
triple diff).

Unit (firm i, group g, quarter t). Outcome: w_{i,g,t} LEVEL (firm's share of group
g's European book, zero-filled), with Delta-w as a check. Regressor of interest:
US_g x CN_{i,t-1} (FIRM-level China exposure now). Control: US_g x BIL_{c(i),t}
(bilateral US-listing-country relations, same construction as step 1: quarterly
mean of the monthly USA|c AI-GPR level, 16 covered countries).

Output: output/firm_ladder_panel.dta
"""
from pathlib import Path
import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
GRID = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
GPR_CSV = r"E:/Data/Data/ai_gpr_bilateral_monthly.csv"
DTA = OUT / "firm_ladder_panel.dta"

ISO2NAME = {
    "GB": "UK", "DE": "Germany", "FR": "France", "PL": "Poland", "IT": "Italy",
    "GR": "Greece", "ES": "Spain", "DK": "Denmark", "PT": "Portugal",
    "BE": "Belgium", "NL": "Netherlands", "IE": "Ireland", "CH": "Switzerland",
    "SE": "Sweden", "AT": "Austria", "NO": "Norway",
}

# ---- bilateral USA|c quarterly means (identical construction to step 1) ----
name_cols = ["Date"] + [f"USA|{n}" for n in ISO2NAME.values()]
g = pd.read_csv(GPR_CSV, usecols=lambda c: c in name_cols)
g["Date"] = pd.to_datetime(g["Date"])
g["q"] = g["Date"].dt.to_period("Q")
qm = g.groupby("q").mean(numeric_only=True).reset_index()
qm["rdate"] = qm["q"].dt.to_timestamp("Q")
bil = qm.melt(id_vars=["rdate"], value_vars=[f"USA|{n}" for n in ISO2NAME.values()],
              var_name="pair", value_name="bil_us_c")
bil["cname"] = bil["pair"].str.replace("USA|", "", regex=False)
name2iso = {v: k for k, v in ISO2NAME.items()}
bil["sec_country"] = bil["cname"].map(name2iso)
bil["rdate"] = pd.to_datetime(bil["rdate"]).dt.normalize()
bil = bil[["sec_country", "rdate", "bil_us_c"]]

# ---- firm-level panel from the grid ----
con = duckdb.connect()
con.execute("SET memory_limit='6GB'")
df = con.execute(f"""
SELECT
    CAST(sec_entity_id AS VARCHAR)                  AS firm_str,
    sec_country,
    holder_group                                    AS hgroup,
    CAST(report_date AS TIMESTAMP)                  AS rdate,
    portfolio_weight_eu                             AS w,
    delta_w                                         AS dw,
    china_share_lag1q                               AS cn_lag,
    CASE WHEN holder_group = 'US' THEN 1 ELSE 0 END AS us
FROM read_parquet('{GRID}')
WHERE portfolio_weight_eu IS NOT NULL
  AND china_share_lag1q  IS NOT NULL
""").df()
con.close()

df["rdate"] = pd.to_datetime(df["rdate"]).dt.normalize()
df = df.merge(bil, on=["sec_country", "rdate"], how="left", validate="m:1")

n_all = len(df)
n_bil = df["bil_us_c"].notna().sum()
print(f"rows (w & lagged CN non-null): {n_all:,}; with bilateral control: {n_bil:,}")
print(f"firms: {df['firm_str'].nunique():,}  quarters: {df['rdate'].nunique()}")

assert not df.duplicated(["firm_str", "hgroup", "rdate"]).any()
assert set(df["hgroup"].unique()) <= {"US", "NONUS"}
assert df["cn_lag"].between(0, 1).all()

df["firm_str"] = df["firm_str"].astype(str)
df["hgroup"] = df["hgroup"].astype(str)
for c in ["w", "dw", "cn_lag", "bil_us_c"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
df["us"] = df["us"].astype("int8")

keep = ["firm_str", "sec_country", "hgroup", "rdate", "w", "dw", "cn_lag", "bil_us_c", "us"]
df[keep].to_stata(DTA, write_index=False, convert_dates={"rdate": "tc"}, version=118)
print(f"wrote {DTA.name} ({n_all:,} rows)")
