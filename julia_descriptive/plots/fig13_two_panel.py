"""
fig13_two_panel.py

Rebuild fig13 as a TWO-panel figure mirroring the slide-7 identification form:
  Left  panel = beta_1 territory: Delta(US - NONUS) mean dw on ALL firms vs S_t
  Right panel = beta_3 territory: Delta(US - NONUS) mean dw on HIGH-CN firms vs S_t

Panel A (all firms) and Panel B (high-CN firms) cover different quarter counts
(A ~98, B ~82, since the high-CN filter drops early quarters with no exposed
holdings). Raw mean (no winsorisation), Pearson r in upper-left box. The per-panel
"n = ... quarters" printed on the figure is authoritative, not this comment.

Reads the C6 zero-filled panel directly so the ALL-firms series is constructed
from the same source as the regression panel.

Output: C:/Users/xl/Downloads/figures/fig13_diff_vs_shock_c6.png  (see FIG_DIR;
overwrites existing). Underlying series -> output/fig13_diff_{all,high}_c6.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import matplotlib.pyplot as plt

PROJ_DIR = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT_DIR  = PROJ_DIR / "output"
FIG_DIR  = Path(r"C:/Users/xl/Downloads/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PANEL_PARQUET = OUT_DIR / "merged_us_eu_zero_filled.parquet"

# ---------------------------------------------------------------
# Build quarterly diff series for ALL firms (beta_1) and HIGH-CN
# firms (beta_3), pulling directly from the C6 panel via duckdb.
# ---------------------------------------------------------------
print(f"[1/3] Loading panel: {PANEL_PARQUET.name}")
con = duckdb.connect(":memory:")
con.execute("SET memory_limit='6GB'")

# HIGH cutoff = median of china_share_lag1q > 0 across firm-quarters (data-driven,
# matches 06_cartesian_grid.jl)
high_cutoff = con.execute(f"""
    WITH base AS (
        SELECT DISTINCT sec_entity_id, report_date, china_share_lag1q
        FROM read_parquet('{PANEL_PARQUET.as_posix()}')
        WHERE china_share_lag1q IS NOT NULL AND china_share_lag1q > 0
    )
    SELECT QUANTILE_CONT(china_share_lag1q, 0.5) AS med FROM base
""").df()["med"].iloc[0]
print(f"      HIGH cutoff (median of china_share_lag1q > 0): {high_cutoff:.4f}")

# Series A: ALL firms (no exposure filter) — beta_1 territory
all_series = con.execute(f"""
    WITH q AS (
        SELECT report_date,
               investor_country,
               AVG(delta_w) AS mean_dw,
               AVG(shock_us_cn) AS shock_us_cn
        FROM read_parquet('{PANEL_PARQUET.as_posix()}')
        WHERE delta_w IS NOT NULL
        GROUP BY 1, 2
    )
    SELECT report_date,
           MAX(CASE WHEN investor_country='US'    THEN mean_dw END) AS mean_us,
           MAX(CASE WHEN investor_country='NONUS' THEN mean_dw END) AS mean_nonus,
           MAX(shock_us_cn) AS shock_us_cn
    FROM q
    GROUP BY report_date
    ORDER BY report_date
""").df()
all_series["mean_diff"] = all_series["mean_us"] - all_series["mean_nonus"]
all_series["report_date"] = pd.to_datetime(all_series["report_date"])
print(f"      ALL-firms quarterly series: {len(all_series)} quarters")

# Series B: HIGH-CN firms (china_share_lag1q > cutoff) — beta_3 territory
high_series = con.execute(f"""
    WITH q AS (
        SELECT report_date,
               investor_country,
               AVG(delta_w) AS mean_dw,
               AVG(shock_us_cn) AS shock_us_cn
        FROM read_parquet('{PANEL_PARQUET.as_posix()}')
        WHERE delta_w IS NOT NULL
          AND china_share_lag1q IS NOT NULL
          AND china_share_lag1q > {high_cutoff}
        GROUP BY 1, 2
    )
    SELECT report_date,
           MAX(CASE WHEN investor_country='US'    THEN mean_dw END) AS mean_us,
           MAX(CASE WHEN investor_country='NONUS' THEN mean_dw END) AS mean_nonus,
           MAX(shock_us_cn) AS shock_us_cn
    FROM q
    GROUP BY report_date
    ORDER BY report_date
""").df()
high_series["mean_diff"] = high_series["mean_us"] - high_series["mean_nonus"]
high_series["report_date"] = pd.to_datetime(high_series["report_date"])
print(f"      HIGH-CN quarterly series: {len(high_series)} quarters")

# ---------------------------------------------------------------
# 2-panel scatter
# ---------------------------------------------------------------
print("[2/3] Drawing 2-panel figure")
RED = "#d35400"  # match plots_python PAL["red"]

def scatter_panel(ax, df, title):
    d = df.dropna(subset=["mean_diff", "shock_us_cn"]).copy()
    x = pd.to_numeric(d["shock_us_cn"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(d["mean_diff"], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 2:
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                ha="center", va="center")
        return
    ax.scatter(x, y, s=22, color=RED, alpha=0.55, edgecolor="none")
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, intercept + slope * xs, ls="--", color="0.3", lw=1.2)
    corr = float(np.corrcoef(x, y)[0, 1])
    txt = f"n = {n} quarters\ncorr = {corr:+.3f}\nslope = {slope:+.2e}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="0.7"))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("US-CN GPR AR(1) shock  $S_t$")
    ax.set_ylabel(r"quarterly $\Delta($US$-$NONUS$)$ mean $\Delta w$")
    ax.set_title(title, fontsize=10)
    print(f"      {title}: n={n}, corr={corr:+.4f}, slope={slope:+.2e}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

scatter_panel(
    axes[0],
    all_series,
    r"(A) $\beta_1$ territory: ALL firms"
    "\n"
    r"(US$-$NONUS gap, no exposure filter)"
)
scatter_panel(
    axes[1],
    high_series,
    rf"(B) $\beta_3$ territory: HIGH-CN firms"
    "\n"
    rf"($\mathrm{{CN}}_{{t-1}} > {high_cutoff:.3f}$)"
)

fig.suptitle(
    r"Descriptive: $\Delta(\mathrm{US}-\mathrm{NONUS})$ mean $\Delta w$ vs.\ US-CN GPR shock",
    fontsize=11, y=1.00
)
fig.tight_layout()

out_path = FIG_DIR / "fig13_diff_vs_shock_c6.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[3/3] Saved: {out_path}")

# Save the underlying series CSVs under FIG13-OWNED names.
# NOTE: 06_cartesian_grid.jl also writes 05_diff_us_vs_nonus_high_c6.csv with a
# DIFFERENT schema (ts_diff, incl. mean_diff_ws). Writing distinct filenames here
# removes the last-writer-wins collision on that path (external-review fix).
all_series.to_csv(OUT_DIR / "fig13_diff_all_c6.csv", index=False)
high_series.to_csv(OUT_DIR / "fig13_diff_high_c6.csv", index=False)
print(f"      Series saved: fig13_diff_all_c6.csv, fig13_diff_high_c6.csv "
      f"(distinct from 06's 05_diff_* to avoid schema collision)")
