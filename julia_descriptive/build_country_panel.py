"""
build_country_panel.py — LADDER STEP 1 (advisor-confirmed specification ladder,
meeting 2026-07-02): the simplest, country-level regression.

  Step 1 (this): unit (country c, group g, quarter t). Outcome w_{c,g,t} = share of
                 group g's European book allocated to country c. Regressor:
                 US_g x CN_{c,t-1} (country-level China exposure). Control:
                 bilateral US-country relations BIL_{c,t} (USA|c AI-GPR for each of
                 the 16 covered countries; guards against general US-Europe frictions
                 being misread as China avoidance -- advisor example: US-Denmark
                 tension over Greenland).
  Step 2: add the firm dimension.  Step 3: add the shock -> current triple diff.

Construction choices (flagged for the advisor):
- w_{c,g,t} = sum over firms listed in c of the zero-filled portfolio weights
  (weights sum to 1 across all Europe per (g,t), so w_c sums to 1 across countries).
- CN_{c,t}: EQUAL-WEIGHTED mean of firm-level china_share over firms in c with
  non-missing exposure (holdings-weighted alternative deferred; equal-weight avoids
  endogenous weighting by the outcome itself). Used lagged (CN_{c,t-1}).
- BIL_{c,t}: quarterly mean of the MONTHLY bilateral AI-GPR level USA|<country>
  (Iacoviello & Tong 2026). Available for 16 of 28 jurisdictions (the missing 12
  are small-holdings CEE + FI/LU); the estimation sample restricts to the 16.

Output: output/country_panel.dta
"""
from pathlib import Path
import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
GRID = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
GPR_CSV = r"E:/Data/Data/ai_gpr_bilateral_monthly.csv"
DTA = OUT / "country_panel.dta"

ISO2NAME = {
    "GB": "UK", "DE": "Germany", "FR": "France", "PL": "Poland", "IT": "Italy",
    "GR": "Greece", "ES": "Spain", "DK": "Denmark", "PT": "Portugal",
    "BE": "Belgium", "NL": "Netherlands", "IE": "Ireland", "CH": "Switzerland",
    "SE": "Sweden", "AT": "Austria", "NO": "Norway",
}

con = duckdb.connect()
con.execute("SET memory_limit='6GB'")

print("[1/4] Aggregating grid to (country, group, quarter)...")
cg = con.execute(f"""
WITH firm_cn AS (  -- one row per (firm, quarter): firm-level exposure (identical across groups)
    SELECT sec_entity_id, sec_country, report_date,
           any_value(china_share) AS cn
    FROM read_parquet('{GRID}')
    GROUP BY 1, 2, 3
),
cty_cn AS (        -- equal-weighted country exposure among firms with non-missing CN
    SELECT sec_country, report_date,
           AVG(cn)                       AS cn_c,
           COUNT(cn)                     AS n_firms_cn,
           COUNT(*)                      AS n_firms_all
    FROM firm_cn
    GROUP BY 1, 2
),
cty_w AS (         -- group g's share of its European book in country c
    SELECT sec_country, holder_group, report_date,
           SUM(COALESCE(portfolio_weight_eu, 0)) AS w_c
    FROM read_parquet('{GRID}')
    GROUP BY 1, 2, 3
)
SELECT w.sec_country, w.holder_group, CAST(w.report_date AS TIMESTAMP) AS rdate,
       w.w_c, c.cn_c, c.n_firms_cn, c.n_firms_all,
       CASE WHEN w.holder_group = 'US' THEN 1 ELSE 0 END AS us
FROM cty_w w
LEFT JOIN cty_cn c USING (sec_country, report_date)
ORDER BY w.sec_country, w.holder_group, w.report_date
""").df()
con.close()

print(f"      rows: {len(cg):,}  countries: {cg['sec_country'].nunique()}  quarters: {cg['rdate'].nunique()}")

print("[2/4] Bilateral USA|country GPR level (quarterly mean of monthly index)...")
name_cols = ["Date"] + [f"USA|{n}" for n in ISO2NAME.values()]
g = pd.read_csv(GPR_CSV, usecols=lambda c: c in name_cols)
g["Date"] = pd.to_datetime(g["Date"])
g["q"] = g["Date"].dt.to_period("Q")
qm = g.groupby("q").mean(numeric_only=True).reset_index()
qm["rdate"] = qm["q"].dt.to_timestamp("Q")
bil_long = qm.melt(id_vars=["rdate"], value_vars=[f"USA|{n}" for n in ISO2NAME.values()],
                   var_name="pair", value_name="bil_us_c")
bil_long["cname"] = bil_long["pair"].str.replace("USA|", "", regex=False)
name2iso = {v: k for k, v in ISO2NAME.items()}
bil_long["sec_country"] = bil_long["cname"].map(name2iso)
bil_long = bil_long[["sec_country", "rdate", "bil_us_c"]]
# normalize both sides to date (grid rdate is quarter-end timestamp)
bil_long["rdate"] = pd.to_datetime(bil_long["rdate"]).dt.normalize()

print("[3/4] Merge + lags...")
cg["rdate"] = pd.to_datetime(cg["rdate"]).dt.normalize()
df = cg.merge(bil_long, on=["sec_country", "rdate"], how="left", validate="m:1")
df = df.sort_values(["sec_country", "holder_group", "rdate"]).reset_index(drop=True)
grp = df.groupby(["sec_country", "holder_group"])
df["cn_c_lag"] = grp["cn_c"].shift(1)
df["w_c_lag"] = grp["w_c"].shift(1)
df["dw_c"] = df["w_c"] - df["w_c_lag"]

est = df.dropna(subset=["w_c", "cn_c_lag"]).copy()
n_bil = est["bil_us_c"].notna().sum()
print(f"      estimation rows (w & lagged CN non-null): {len(est):,}; with bilateral control: {n_bil:,}")
print(f"      countries with bilateral data: {est.loc[est['bil_us_c'].notna(),'sec_country'].nunique()} of {est['sec_country'].nunique()}")

# asserts
assert not est.duplicated(["sec_country", "holder_group", "rdate"]).any()
assert set(est["holder_group"].unique()) <= {"US", "NONUS"}
_w1 = df.groupby(["holder_group", "rdate"])["w_c"].sum()
_dev = (_w1 - 1).abs()
# weights sum to 1 across countries per (group, quarter) wherever the group's book exists
_bad = _dev[(_dev > 1e-6) & (_w1 > 0.5)]
assert len(_bad) == 0, f"country weights do not sum to 1: {_bad.head()}"

print("[4/4] Write...")
keep = ["sec_country", "holder_group", "rdate", "w_c", "dw_c", "cn_c", "cn_c_lag",
        "bil_us_c", "n_firms_cn", "n_firms_all", "us"]
est["rdate"] = pd.to_datetime(est["rdate"])
for c in ["w_c", "dw_c", "cn_c", "cn_c_lag", "bil_us_c"]:
    est[c] = pd.to_numeric(est[c], errors="coerce").astype("float64")
est["us"] = est["us"].astype("int8")
est[keep].to_stata(DTA, write_index=False, convert_dates={"rdate": "tc"}, version=118)
print(f"      wrote {DTA.name} ({len(est):,} rows)")
print(est.groupby("sec_country")["w_c"].mean().sort_values(ascending=False).head(8).to_string())
