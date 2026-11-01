"""
build_russia_shock.py — US-Russia GPR shock for the Russia positive control.

Isomorphic to the US-China shock: AR(1) conditional-OLS residual on the monthly
`USA|Russia` AI-GPR series (Iacoviello-Tong), quarter-end month = the quarter's
shock. Exactly the recipe in 05_combine_visualize.jl / build_country_pair_shock.py,
only the series changes.

Output: output/russia_shock_monthly.csv  (month_end, gpr_us_ru, shock_us_ru,
        is_quarter_end)
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")
GPR_CSV = Path(r"E:/Data/Data/ai_gpr_bilateral_monthly.csv")
COL = "USA|Russia"

g = pd.read_csv(GPR_CSV, usecols=["Date", COL])
g["Date"] = pd.to_datetime(g["Date"])
g["month_first"] = g["Date"].dt.to_period("M").dt.to_timestamp()
g["month_end"] = g["Date"].dt.to_period("M").dt.to_timestamp("M")
g = g.sort_values("month_first").reset_index(drop=True)

# AR(1) conditional least squares (identical to fit_ar1 in build_country_pair_shock.py)
s = g[COL].dropna().to_numpy(float)
assert len(s) >= 3
y, yL = s[1:], s[:-1]
b = ((yL - yL.mean()) * (y - y.mean())).sum() / ((yL - yL.mean()) ** 2).sum()
a = y.mean() - b * yL.mean()
resid = y - (a + b * yL)
aligned = np.concatenate([[np.nan], resid])  # first obs has no residual

full = np.full(len(g), np.nan)
full[g[COL].notna().to_numpy().nonzero()[0]] = aligned
g["gpr_us_ru"] = g[COL]
g["shock_us_ru"] = full
g["is_quarter_end"] = g["month_end"].dt.month.isin([3, 6, 9, 12])

out = g[["month_end", "gpr_us_ru", "shock_us_ru", "is_quarter_end"]]
out.to_csv(OUT / "russia_shock_monthly.csv", index=False)
print(f"AR(1) USA|Russia: a={a:.4f} b={b:.4f} n={len(s)}  ({g['month_first'].min().date()} -> {g['month_first'].max().date()})")
qe = out[out["is_quarter_end"] & out["shock_us_ru"].notna()]
print(f"quarter-end shocks: {len(qe)}")
# peak (should be 2022Q1/Q2 Ukraine)
top = qe.reindex(qe['shock_us_ru'].abs().sort_values(ascending=False).index).head(5)
print("top-5 |shock| quarters (expect 2022 Ukraine):")
print(top[["month_end", "shock_us_ru"]].to_string(index=False))
print(f"wrote russia_shock_monthly.csv")
