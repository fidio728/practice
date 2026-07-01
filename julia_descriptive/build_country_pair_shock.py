"""
build_country_pair_shock.py

Robustness build for `07b_country_pair_robustness.do`.

Replaces the single USA-China AI-GPR shock used in the main spec with a
COUNTRY-PAIR-SPECIFIC shock S_{c,t} computed on each EU-listing country's
own AI-GPR series vs China. Three EU countries covered: GB (UK|China),
DE (Germany|China), FR (France|China). PT excluded (single direction,
magnitude ~0, only 79 firms).

Two artifacts written:
  1. output/country_pair_shocks_monthly.csv  -- diagnostic monthly series
  2. output/c6_panel_country_pair.dta        -- Stata-ready firm-qtr-group panel
     restricted to sec_country IN ('GB','DE','FR') with `shock_c` (country-pair)
     and `shock_us_cn` (original) on identical observations.

AR(1) recipe = byte-for-byte copy of 05_combine_visualize.jl L115-126:
plain OLS on non-missing monthly series, residual at quarter-end month.

REVIEW FIXES applied (vs first draft):
  - month_end via dt.to_period(M).to_timestamp(M) (the np.timedelta64('M')
    arithmetic crashes at runtime)
  - us indicator pulled in the SAME duckdb SELECT as the panel (avoids row-
    order misalignment across two separate parquet scans)
  - shock column kept as `shock_us_cn` (matches main panel naming + do-file ref)
  - assert shock_long key uniqueness before merge
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

PROJ_DIR = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT_DIR  = PROJ_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GPR_CSV       = Path(r"E:/Data/Data/ai_gpr_bilateral_monthly.csv")
PANEL_PARQUET = OUT_DIR / "merged_us_eu_zero_filled.parquet"

SHOCKS_CSV  = OUT_DIR / "country_pair_shocks_monthly.csv"
COEFS_CSV   = OUT_DIR / "country_pair_shocks_coefficients.csv"
PANEL_DTA   = OUT_DIR / "c6_panel_country_pair.dta"

# ISO2 listing country -> AI-GPR CSV column. Direction = first-country-perspective
# matching the main USA|China spec.
COUNTRY_PAIR = {
    "GB": "UK|China",
    "DE": "Germany|China",
    "FR": "France|China",
}

# ---------------------------------------------------------------
# 1. Load raw monthly AI-GPR.
# ---------------------------------------------------------------
print(f"[1/6] Reading raw GPR CSV: {GPR_CSV}")
need_cols = ["Date", "USA|China"] + list(COUNTRY_PAIR.values())
gpr_raw = pd.read_csv(GPR_CSV, usecols=need_cols)
gpr_raw["Date"] = pd.to_datetime(gpr_raw["Date"])
# Use to_period for safe month-end arithmetic (np.timedelta64('M') is rejected
# by recent numpy as "ambiguous duration").
gpr_raw["month_first"] = gpr_raw["Date"].dt.to_period("M").dt.to_timestamp()
gpr_raw["month_end"]   = gpr_raw["Date"].dt.to_period("M").dt.to_timestamp("M")
gpr_raw = gpr_raw.sort_values("month_first").reset_index(drop=True)
print(f"      {len(gpr_raw)} months, "
      f"{gpr_raw['month_first'].min().date()} -> "
      f"{gpr_raw['month_first'].max().date()}")

# ---------------------------------------------------------------
# 2. AR(1) per series. Recipe matches julia 05_combine_visualize.jl L115-126.
# ---------------------------------------------------------------
def fit_ar1(series: pd.Series) -> dict:
    g = series.dropna().to_numpy(dtype=float)
    if len(g) < 3:
        raise ValueError("AR(1) fit needs >=3 non-missing months")
    y  = g[1:]
    yL = g[:-1]
    yL_mean = yL.mean()
    y_mean  = y.mean()
    b_hat = ((yL - yL_mean) * (y - y_mean)).sum() / ((yL - yL_mean) ** 2).sum()
    a_hat = y_mean - b_hat * yL_mean
    resid = y - (a_hat + b_hat * yL)
    aligned = np.concatenate([[np.nan], resid])
    return {"a": a_hat, "b": b_hat, "n": int(len(g)), "resid_aligned": aligned}

# Sanity: interior NaNs would silently shift residuals onto wrong months.
for col in ["USA|China"] + list(COUNTRY_PAIR.values()):
    s = gpr_raw[col]
    nn = s.notna()
    if nn.any():
        first = nn.idxmax()
        if s.iloc[first:].isna().any():
            raise ValueError(f"Interior NaN in '{col}' after {gpr_raw['month_first'].iloc[first].date()}")

print("[2/6] Fitting AR(1) per series (USA|China + 3 country-pairs)")
diag_rows = []
wide_shocks = gpr_raw[["month_first", "month_end"]].copy()

# US baseline first (sanity check vs main spec).
us_fit = fit_ar1(gpr_raw["USA|China"])
us_full = np.full(len(gpr_raw), np.nan)
us_full[gpr_raw["USA|China"].notna().to_numpy().nonzero()[0]] = us_fit["resid_aligned"]
wide_shocks["gpr_us_cn"]   = gpr_raw["USA|China"].to_numpy()
wide_shocks["shock_us_cn"] = us_full
diag_rows.append({"sec_country": "US", "ai_gpr_column": "USA|China",
                  "a_hat": us_fit["a"], "b_hat": us_fit["b"], "n_months": us_fit["n"]})
print(f"      US baseline (USA|China): a={us_fit['a']:.4f}  b={us_fit['b']:.4f}  n={us_fit['n']}")

for ctry_iso, col in COUNTRY_PAIR.items():
    fit = fit_ar1(gpr_raw[col])
    full_resid = np.full(len(gpr_raw), np.nan)
    full_resid[gpr_raw[col].notna().to_numpy().nonzero()[0]] = fit["resid_aligned"]
    wide_shocks[f"gpr_{ctry_iso.lower()}_cn"]   = gpr_raw[col].to_numpy()
    wide_shocks[f"shock_{ctry_iso.lower()}_cn"] = full_resid
    diag_rows.append({"sec_country": ctry_iso, "ai_gpr_column": col,
                      "a_hat": fit["a"], "b_hat": fit["b"], "n_months": fit["n"]})
    print(f"      {ctry_iso} ({col}): a={fit['a']:.4f}  b={fit['b']:.4f}  n={fit['n']}")

wide_shocks["is_quarter_end"] = wide_shocks["month_end"].dt.month.isin([3, 6, 9, 12])

# ---------------------------------------------------------------
# 3. Write diagnostics.
# ---------------------------------------------------------------
print(f"[3/6] Writing diagnostics")
pd.DataFrame(diag_rows).to_csv(COEFS_CSV, index=False)
wide_shocks.to_csv(SHOCKS_CSV, index=False)
print(f"      coefficients -> {COEFS_CSV.name}")
print(f"      monthly      -> {SHOCKS_CSV.name}")

# ---------------------------------------------------------------
# 4. Build the long-format quarter-end shock lookup.
# ---------------------------------------------------------------
qe = wide_shocks.loc[wide_shocks["is_quarter_end"]].copy()
shock_long_rows = []
for ctry_iso in COUNTRY_PAIR:
    col = f"shock_{ctry_iso.lower()}_cn"
    sub = qe[["month_end", col]].rename(columns={col: "shock_c"})
    sub["sec_country"] = ctry_iso
    shock_long_rows.append(sub)
shock_long = pd.concat(shock_long_rows, ignore_index=True)
shock_long = shock_long.rename(columns={"month_end": "quarter_end"})
shock_long["quarter_end"] = pd.to_datetime(shock_long["quarter_end"])

# Belt-and-braces key uniqueness check
assert not shock_long.duplicated(["sec_country", "quarter_end"]).any(), \
    "duplicate (sec_country, quarter_end) key in shock_long"
print(f"[4/6] Country-pair quarter-end shock rows: {len(shock_long)}  "
      f"({shock_long['sec_country'].nunique()} countries x ~"
      f"{shock_long['quarter_end'].nunique()} quarters)")

# ---------------------------------------------------------------
# 5. Load panel + us indicator in ONE duckdb query (avoids row-order
#    misalignment that would happen with two separate parquet scans).
# ---------------------------------------------------------------
print(f"[5/6] Reading panel parquet (single SELECT) -> {PANEL_PARQUET}")
con = duckdb.connect(":memory:")
con.execute("SET memory_limit='6GB'")
panel = con.execute(f"""
    SELECT sec_entity_id           AS firm_str,
           sec_country,
           holder_group             AS hgroup,
           report_date              AS rdate,
           I_ict,
           portfolio_weight_eu      AS w,
           w_prev,
           delta_w                  AS dw,
           china_share              AS cn_share,
           china_share_lag1q        AS cn_lag,
           gpr_us_cn,
           shock_us_cn,
           CASE WHEN investor_country = 'US' THEN 1 ELSE 0 END AS us
    FROM read_parquet('{PANEL_PARQUET.as_posix()}')
    WHERE sec_country IN ('GB','DE','FR')
""").df()
print(f"      panel rows (GB+DE+FR): {len(panel):,}")
print(panel["sec_country"].value_counts().to_string())

# ---------------------------------------------------------------
# 6. Join the country-pair shock by (sec_country, quarter_end).
# ---------------------------------------------------------------
panel["rdate"] = pd.to_datetime(panel["rdate"])
panel["rdate_qe"] = (panel["rdate"].dt.to_period("M")
                                   .dt.to_timestamp("M")
                                   .dt.normalize())
shock_long["quarter_end"] = shock_long["quarter_end"].dt.normalize()

before = len(panel)
panel = panel.merge(
    shock_long.rename(columns={"quarter_end": "rdate_qe"}),
    on=["sec_country", "rdate_qe"],
    how="left",
    validate="m:1",
)
assert len(panel) == before, "merge changed row count"
unmatched = panel["shock_c"].isna().sum()
print(f"[6/6] Joined shock_c. Unmatched (NaN shock_c): {unmatched:,} / {len(panel):,}")
panel = panel.drop(columns=["rdate_qe"])

# Final column order. shock_us_cn kept as-is (matches main panel naming).
out_cols = ["firm_str", "sec_country", "hgroup", "rdate",
            "w", "dw", "cn_share", "cn_lag", "I_ict",
            "gpr_us_cn", "shock_us_cn", "shock_c", "us"]
panel = panel[out_cols]

print(f"      writing Stata file -> {PANEL_DTA}")
panel.to_stata(
    PANEL_DTA,
    write_index=False,
    variable_labels={
        "firm_str":    "FactSet sec_entity_id",
        "sec_country": "EU listing country (GB/DE/FR)",
        "hgroup":      "Holder group",
        "rdate":       "Report date (quarter-end, %tc)",
        "w":           "Portfolio weight (EU)",
        "dw":          "Delta w",
        "cn_share":    "China share",
        "cn_lag":      "China share, lag 1q",
        "I_ict":       "Group holding value, USD (I_{i,c,t})",
        "gpr_us_cn":   "AI-GPR US-CN level",
        "shock_us_cn": "AR(1) residual on USA|China (main shock)",
        "shock_c":     "AR(1) residual on country|China (robustness)",
        "us":          "US investor indicator",
    },
    convert_dates={"rdate": "tc"},
)
print("Done.")
print(f"  Panel rows written: {len(panel):,}")
print(f"  Country breakdown:\n{panel['sec_country'].value_counts().to_string()}")
print(f"  shock_c missing: {panel['shock_c'].isna().sum():,} / {len(panel):,}")
print(f"  shock_us_cn missing: {panel['shock_us_cn'].isna().sum():,} / {len(panel):,}")
