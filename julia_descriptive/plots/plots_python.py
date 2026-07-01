#!/usr/bin/env python
"""
plots_python.py — Generate paper figures from the audit-fixed julia_descriptive
pipeline output CSVs. Each figure is a SINGLE panel saved as its own PNG,
unless two/three panels tell a single story (then a vertical stack of 2
panels). All output goes to C:/Users/xl/Downloads/figures/ so that
C:/Users/xl/Downloads/geoecon.tex picks them up via \\graphicspath{{figures/}}.

Run after 02 + 04 + 05 have produced their CSVs in
julia_descriptive/output/.

Naming convention:
  fig01_eu_firms_with_cn.png            (single panel)
  fig02_mean_china_share.png            (single panel)
  fig03_distribution_cn_relations.png   (single panel)
  fig04_relation_type_composition.png   (single panel)
  fig05_us_eu_engagement.png            (2-panel stack — story unit)
  fig06_distribution_us_ownership.png   (single panel)
  fig07_mean_us_ownership_time.png      (single panel)
  fig08_total_us_held_time.png          (single panel)
  fig09_us_held_by_country.png          (single panel)
  fig10_scatter_own_vs_cn.png           (single panel)
  fig11_us_allocation_by_group.png      (single panel)
  fig12_us_vs_nonus_high.png            (2-panel stack — US vs NONUS + GPR)
  fig13_diff_vs_shock.png               (single panel)
  fig14_coverage_cascade.png            (single panel)
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# Paths and palette
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR.parent / "output"          # julia_descriptive/output
FIG_DIR    = Path(r"C:/Users/xl/Downloads/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

PAL = dict(
    blue   = "#0072B2",
    orange = "#E69F00",
    green  = "#009E73",
    red    = "#D55E00",
    purple = "#CC79A7",
    gray   = "#787878",
)
COLOR_US    = "#0072B2"
COLOR_NONUS = "#E69F00"

# Default style for paper-quality figures
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "legend.frameon": False,
})

def save(fig, name):
    suffix = "_c6" if USE_C6 else ""
    out = FIG_DIR / f"{name}{suffix}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")
    return out

USE_C6 = os.environ.get("DPN_USE_C6", "false").lower() in ("true", "1", "yes")

# CSVs that have a C6 (zero-filled panel) counterpart written by 06_cartesian_grid.jl.
# When DPN_USE_C6=true, these filenames are remapped to the _c6 suffixed version.
C6_REMAP = {
    "05_scatter_own_vs_cn_data.csv":        "05_scatter_own_vs_cn_data_c6.csv",
    "05_within_europe_share_by_group.csv":  "05_within_europe_share_by_group_c6.csv",
    "05_us_vs_nonus_high_share_data.csv":   "05_us_vs_nonus_high_share_data_c6.csv",
    "05_diff_us_vs_nonus_high.csv":         "05_diff_us_vs_nonus_high_c6.csv",
}

def safe_read(filename):
    if USE_C6 and filename in C6_REMAP:
        candidate = OUT_DIR / C6_REMAP[filename]
        if candidate.exists():
            filename = C6_REMAP[filename]
            print(f"  [C6] using {filename}")
    p = OUT_DIR / filename
    if not p.exists():
        print(f"  [SKIP] {filename} not found at {p}")
        return None
    try:
        if str(p).endswith(".parquet"):
            return pd.read_parquet(p)
        return pd.read_csv(p)
    except Exception as e:
        print(f"  [SKIP] could not read {filename}: {e}")
        return None

def fmt_thousand_axis(ax, axis="y"):
    if axis == "y":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

def date_axis(ax, dates):
    """Adaptive date locator: short series get yearly ticks, long get 5-yearly."""
    if dates is None or len(dates) == 0:
        return
    span_years = (pd.to_datetime(dates).max() - pd.to_datetime(dates).min()).days / 365.25
    if span_years > 15:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
    elif span_years > 5:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# ---------------------------------------------------------------------------
# FIG 1-4 — China exposure of EU firms over time
# Source: 02_china_exposure_timeseries.csv + 02_dist_cn_total_2018.csv
# ---------------------------------------------------------------------------
def fig_china_exposure():
    ts   = safe_read("02_china_exposure_timeseries.csv")
    dist = safe_read("02_dist_cn_total_2018.csv")
    if ts is None:
        print("  fig_china_exposure: missing input")
        return

    # The Julia pipeline writes the quarter-end date column with name varying
    # by version: 02_china_exposure_timeseries.csv may have 'month_end' or
    # 'quarter_end'. Pick whichever exists.
    date_col = next((c for c in ("quarter_end", "month_end") if c in ts.columns), None)
    if date_col is None:
        print("  fig_china_exposure: no date column found")
        return
    ts[date_col] = pd.to_datetime(ts[date_col])
    ts = ts.sort_values(date_col)

    # --- fig01: # EU firms with any CN supply-chain link
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts[date_col], ts["n_firms_with_cn"], color=PAL["blue"], lw=1.8)
    ax.set_title("EU firms with at least one Chinese supply-chain link")
    ax.set_ylabel("# firms")
    fmt_thousand_axis(ax, "y")
    date_axis(ax, ts[date_col])
    save(fig, "fig01_eu_firms_with_cn")

    # --- fig02: Mean China share of supply-chain links
    fig, ax = plt.subplots(figsize=(7, 4))
    if "avg_china_share_among_exposed" in ts.columns:
        ax.plot(ts[date_col], ts["avg_china_share_among_exposed"],
                color=PAL["red"], lw=1.8, label="among CN-exposed firms")
        if "avg_china_share" in ts.columns:
            ax.plot(ts[date_col], ts["avg_china_share"],
                    color=PAL["orange"], lw=1.4, ls="--",
                    label="all firms with any link")
        ax.legend(loc="upper left")
        ax.set_ylabel("share (China-links / total links)")
    elif "avg_cn_rels" in ts.columns:
        ax.plot(ts[date_col], ts["avg_cn_rels"], color=PAL["red"], lw=1.8)
        ax.set_ylabel("mean # CN relations per exposed firm")
    ax.set_title("Mean Chinese share of supply-chain links")
    date_axis(ax, ts[date_col])
    save(fig, "fig02_mean_china_share")

    # --- fig03: Distribution of # CN relations per exposed firm (2018-12-31)
    fig, ax = plt.subplots(figsize=(7, 4))
    if dist is not None and len(dist) > 0:
        max_n = min(40, dist["n_cn_total"].max() + 2)
        ax.bar(dist["n_cn_total"], dist["n_firms"],
               color=PAL["blue"], alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.set_xlim(0, max_n)
    ax.set_title("Distribution of # Chinese supply-chain relations per exposed firm (2018-12-31)")
    ax.set_xlabel("# Chinese relations")
    ax.set_ylabel("# firms")
    save(fig, "fig03_distribution_cn_relations")

    # --- fig04: Relation type composition over time
    fig, ax = plt.subplots(figsize=(7, 4))
    type_specs = [
        ("avg_cn_customer", "CUSTOMER", PAL["blue"]),
        ("avg_cn_supplier", "SUPPLIER", PAL["orange"]),
        ("avg_cn_jv",       "JV",       PAL["red"]),
    ]
    for col, label, color in type_specs:
        if col in ts.columns:
            ax.plot(ts[date_col], ts[col], color=color, lw=1.5, label=label)
    if "avg_cn_rels" in ts.columns:
        ax.plot(ts[date_col], ts["avg_cn_rels"], color="black",
                lw=2.0, ls="--", label="TOTAL")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Relation-type composition over time")
    ax.set_ylabel("avg # per exposed firm")
    date_axis(ax, ts[date_col])
    save(fig, "fig04_relation_type_composition")

# ---------------------------------------------------------------------------
# FIG 5 — US-EU engagement (2-panel stack: counts + total value share a story)
# Source: 03_us_x_eu_cells_by_month.csv
# ---------------------------------------------------------------------------
def fig_us_eu_engagement():
    df = safe_read("03_us_x_eu_cells_by_month.csv")
    if df is None:
        print("  fig_us_eu_engagement: missing input")
        return
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values("report_date")
    mv_col = "total_mv_billions" if "total_mv_billions" in df.columns else "total_mv_b"

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True)
    # Panel (a): counts with dual y-axis
    ax = axes[0]
    ax.plot(df["report_date"], df["n_us_funds"], color=COLOR_US, lw=1.6, label="# US funds")
    ax.set_ylabel("# US funds", color=COLOR_US)
    ax.tick_params(axis="y", labelcolor=COLOR_US)
    fmt_thousand_axis(ax, "y")
    axb = ax.twinx()
    axb.plot(df["report_date"], df["n_eu_firms"], color=COLOR_NONUS, lw=1.6, ls="--", label="# EU firms")
    axb.set_ylabel("# EU firms", color=COLOR_NONUS)
    axb.tick_params(axis="y", labelcolor=COLOR_NONUS)
    fmt_thousand_axis(axb, "y")
    axb.grid(False)
    ax.set_title("US-EU engagement: investor and firm counts")

    # Panel (b): total dollar value
    ax = axes[1]
    ax.plot(df["report_date"], df[mv_col], color=PAL["green"], lw=1.8)
    ax.set_ylabel("USD billions")
    ax.set_title("Total US-held EU equity (USD bn)")
    date_axis(ax, df["report_date"])
    save(fig, "fig05_us_eu_engagement")

# ---------------------------------------------------------------------------
# FIG 6-9 — US ownership distribution + by-country
# Source: 04_us_ownership_eu_snapshot.csv, 04_us_own_by_eu_country_snapshot.csv,
#         04_us_ownership_eu_timeseries.csv
# ---------------------------------------------------------------------------
def fig_us_ownership():
    snap = safe_read("04_us_ownership_eu_snapshot.csv")
    bycn = safe_read("04_us_own_by_eu_country_snapshot.csv")
    ts   = safe_read("04_us_ownership_eu_timeseries.csv")

    # --- fig06: Distribution of US ownership share (snapshot)
    if snap is not None and len(snap) > 0:
        x_all = pd.to_numeric(snap["ownership_share"], errors="coerce").dropna().values * 100
        n_dropped = int((x_all < 0.5).sum())
        x = x_all[x_all >= 0.5]
        p50 = float(np.quantile(x_all, 0.50))
        p75 = float(np.quantile(x_all, 0.75))
        p90 = float(np.quantile(x_all, 0.90))
        xlim_top = min(40, float(np.quantile(x, 0.99)) * 1.1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(x, bins=40, color=PAL["blue"], alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axvline(p50, color="black",       ls="--", lw=1)
        ax.axvline(p75, color=PAL["gray"],   ls="--", lw=1)
        ax.axvline(p90, color=PAL["red"],    ls="--", lw=1)
        txt = (f"P50 = {p50:.1f}%\nP75 = {p75:.1f}%\nP90 = {p90:.1f}%\n"
               f"(N={len(x_all)}; {n_dropped} firms <0.5% trimmed from view)")
        ax.text(0.98, 0.96, txt, transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"))
        ax.set_xlim(0, xlim_top)
        ax.set_title("Distribution of US ownership share in EU firms (snapshot)")
        ax.set_xlabel("US ownership share (%)")
        ax.set_ylabel("# firms")
        save(fig, "fig06_distribution_us_ownership")

    # --- fig07: Mean US ownership over time
    if ts is not None and "mean_us_own" in ts.columns:
        ts["report_date"] = pd.to_datetime(ts["report_date"])
        ts = ts.sort_values("report_date")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ts["report_date"], ts["mean_us_own"] * 100, color=COLOR_US, lw=1.8)
        ax.set_title("Mean US institutional ownership of EU firms over time")
        ax.set_ylabel("mean ownership (%)")
        date_axis(ax, ts["report_date"])
        save(fig, "fig07_mean_us_ownership_time")

    # --- fig08: Total US-held EU equity over time
    if ts is not None and "total_us_holding_b" in ts.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(ts["report_date"], ts["total_us_holding_b"], color=PAL["green"], lw=1.8)
        ax.set_title("Total US-held EU equity over time (USD bn)")
        ax.set_ylabel("USD billions")
        date_axis(ax, ts["report_date"])
        save(fig, "fig08_total_us_held_time")

    # --- fig09: US-held EU equity by country (top 15 horizontal bar)
    if bycn is not None and len(bycn) > 0:
        top = bycn.sort_values("total_us_holding_b", ascending=False).head(15).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        ypos = np.arange(len(top))[::-1]
        ax.barh(ypos, top["total_us_holding_b"], color=PAL["blue"], alpha=0.85,
                edgecolor="white", linewidth=0.6)
        ax.set_yticks(ypos)
        ax.set_yticklabels(top["sec_country"])
        ax.set_xlabel("USD billions")
        ax.set_title("Total US-held EU equity by country (USD bn, top 15)")
        for yp, v in zip(ypos, top["total_us_holding_b"]):
            ax.text(v + max(top["total_us_holding_b"]) * 0.01, yp, f"{int(round(v))}",
                    va="center", ha="left", fontsize=9)
        ax.set_xlim(0, top["total_us_holding_b"].max() * 1.12)
        ax.tick_params(axis="y", length=0)
        save(fig, "fig09_us_held_by_country")

# ---------------------------------------------------------------------------
# FIG 10-13 — Pre-regression descriptive support
# Sources: 05_scatter_own_vs_cn_data.csv, 05_within_europe_share_by_group.csv,
#          05_us_vs_nonus_high_share_data.csv, 05_diff_us_vs_nonus_high.csv
# ---------------------------------------------------------------------------
def fig_preregression():
    sc   = safe_read("05_scatter_own_vs_cn_data.csv")
    ts   = safe_read("05_within_europe_share_by_group.csv")
    hi   = safe_read("05_us_vs_nonus_high_share_data.csv")
    diff = safe_read("05_diff_us_vs_nonus_high.csv")

    # --- fig10: Scatter US ownership vs CN exposure (snapshot)
    if sc is not None and len(sc) > 10:
        # Prefer lagged exposure if 05 emits it under that name
        x_col = "china_share_lag1q" if "china_share_lag1q" in sc.columns else "china_share"
        sub = sc.dropna(subset=[x_col, "us_ownership_share"]).copy()
        if len(sub) > 10:
            x = pd.to_numeric(sub[x_col], errors="coerce")
            y = pd.to_numeric(sub["us_ownership_share"], errors="coerce") * 100  # to %
            m = x.notna() & y.notna()
            x, y = x[m].values, y[m].values
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.scatter(x, y, s=8, color=PAL["blue"], alpha=0.35, edgecolor="none")
            if len(x) > 2:
                slope, intercept = np.polyfit(x, y, 1)
                xs = np.linspace(0, 1, 50)
                ax.plot(xs, intercept + slope * xs, ls="--", color="0.3", lw=1.2)
                ax.text(0.97, 0.95, f"slope = {slope:.2f}", transform=ax.transAxes,
                        ha="right", va="top", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, max(50, np.percentile(y, 99) * 1.1))
            ax.set_xlabel("China share of supply-chain links (lagged 1Q)")
            ax.set_ylabel("US ownership share (%)")
            ax.set_title("Cross-section: US ownership vs Chinese supply-chain exposure")
            save(fig, "fig10_scatter_own_vs_cn")

    # --- fig11: US allocation by exposure group
    if ts is not None and "exp_grp" in ts.columns:
        sub = ts[ts.get("investor_country", "US") == "US"].copy()
        if "report_date" in sub.columns:
            sub["report_date"] = pd.to_datetime(sub["report_date"])
            # plot abs_portfolio_weight if present (basis points), else within_europe_share
            y_col = "abs_portfolio_weight" if "abs_portfolio_weight" in sub.columns else "within_europe_share"
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            # Two-group split: median of china_share_lag1q on positive-exposure
            # cells (data-driven), plus MISSING for NULL (pre-Revere / out of coverage).
            color_map = dict(LOW=PAL["blue"], HIGH=PAL["red"], MISSING="0.85")
            for grp, color in color_map.items():
                ssub = sub[sub["exp_grp"] == grp].sort_values("report_date")
                if len(ssub) > 0:
                    # Convert to basis points if values look like raw weights (<1)
                    yv = ssub[y_col].astype(float)
                    if y_col == "within_europe_share":
                        yv = yv * 1e4   # to bp
                    ax.plot(ssub["report_date"], yv, color=color, lw=1.5, label=grp)
            ax.set_title("US portfolio weight on EU firms by China-exposure group")
            ax.set_ylabel("portfolio weight (basis points)")
            ax.legend(title="Exposure (t-1)", loc="upper right", ncol=3, fontsize=9)
            date_axis(ax, sub["report_date"])
            save(fig, "fig11_us_allocation_by_group")

    # --- fig12: Portfolio weight on HIGH-exposure firms, US vs NONUS + GPR overlay
    # Build from ts (both US and NONUS rows present) restricted to HIGH
    if ts is not None and "exp_grp" in ts.columns:
        hi_us    = ts[(ts["exp_grp"] == "HIGH") & (ts["investor_country"] == "US")].copy()
        hi_nonus = ts[(ts["exp_grp"] == "HIGH") & (ts["investor_country"] == "NONUS")].copy()
        if len(hi_us) > 0 and len(hi_nonus) > 0:
            hi_us["report_date"]    = pd.to_datetime(hi_us["report_date"])
            hi_nonus["report_date"] = pd.to_datetime(hi_nonus["report_date"])
            hi_us    = hi_us.sort_values("report_date")
            hi_nonus = hi_nonus.sort_values("report_date")

            # Y values to basis points: abs_portfolio_weight is in raw weight units
            y_col = "abs_portfolio_weight" if "abs_portfolio_weight" in hi_us.columns else "within_europe_share"
            def to_bp(s):
                v = pd.to_numeric(s, errors="coerce").astype(float)
                # if very small (<1), treat as raw weight and scale to bp
                if v.dropna().max() < 1.0:
                    v = v * 1e4
                return v

            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.plot(hi_us["report_date"],    to_bp(hi_us[y_col]),
                    color=COLOR_US, lw=1.6, label="US")
            ax.plot(hi_nonus["report_date"], to_bp(hi_nonus[y_col]),
                    color=COLOR_NONUS, lw=1.6, label="non-US")
            ax.set_ylabel("portfolio weight on HIGH-exposure firms (basis points)")
            # GPR overlay on right axis if column exists
            gpr_col = next((c for c in ("gpr_us_cn",) if c in hi_us.columns), None)
            if gpr_col is not None:
                axb = ax.twinx()
                axb.plot(hi_us["report_date"], hi_us[gpr_col],
                         color=PAL["green"], lw=1.0, ls=":", alpha=0.8)
                axb.set_ylabel("USA|China GPR", color=PAL["green"])
                axb.tick_params(axis="y", labelcolor=PAL["green"])
                axb.grid(False)
            ax.legend(loc="upper left")
            ax.set_title("Portfolio weight on HIGH China-exposure firms: US vs non-US")
            date_axis(ax, hi_us["report_date"])
            save(fig, "fig12_us_vs_nonus_high")

    # --- fig13: Scatter Δ(US - non-US) on HIGH vs GPR AR(1) shock
    if diff is not None and "shock_us_cn" in diff.columns:
        d = diff.dropna(subset=["shock_us_cn"]).copy()
        # Prefer the winsorized diff if available
        # On the C6 zero-filled panel we MUST plot raw mean_diff: winsorisation
        # would cap exactly the extensive-margin tails that C6 exists to expose
        # (raw r = -0.019 vs winsorised r = +0.037 in the current build — the
        # sign flip is mechanical, not informative).
        if USE_C6:
            y_col = "mean_diff" if "mean_diff" in d.columns else "mean_diff_raw"
        else:
            y_col = "mean_diff_ws" if "mean_diff_ws" in d.columns else (
                    "mean_diff" if "mean_diff" in d.columns else "mean_diff_raw")
        if y_col not in d.columns:
            print("  fig13: no diff column found")
        else:
            x = pd.to_numeric(d["shock_us_cn"], errors="coerce")
            y = pd.to_numeric(d[y_col], errors="coerce")
            m = x.notna() & y.notna()
            x, y = x[m].values, y[m].values
            if len(x) > 10:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.scatter(x, y, s=22, color=PAL["red"], alpha=0.55, edgecolor="none")
                slope, intercept = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, intercept + slope * xs, ls="--", color="0.3", lw=1.2)
                corr = float(np.corrcoef(x, y)[0, 1])
                txt = f"corr = {corr:.3f}\nslope = {slope:.4f}"
                ax.text(0.03, 0.95, txt, transform=ax.transAxes,
                        ha="left", va="top", fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"))
                ax.axhline(0, color="0.6", lw=0.8)
                ax.set_xlabel("US-CN GPR AR(1) shock")
                ax.set_ylabel(r"quarterly $\Delta($US$-$non-US$)$ portfolio weight (bp)")
                ax.set_title("Δ(US − non-US) on HIGH-exposure firms vs US-CN GPR shock")
                save(fig, "fig13_diff_vs_shock")

# ---------------------------------------------------------------------------
# FIG 14 — Coverage cascade
# Source: 05_coverage_cascade.csv
# ---------------------------------------------------------------------------
def fig_coverage_cascade():
    cas = safe_read("05_coverage_cascade.csv")
    if cas is None or len(cas) == 0:
        return
    row = cas.iloc[0].to_dict()
    labels = ["Total EU sec_entity_ids",
              "with CUSIP populated",
              "with ISIN populated",
              "with SEDOL populated",
              "matched to Revere"]
    keys = ["n_eu_sec_total", "n_eu_sec_with_cusip", "n_eu_sec_with_isin",
            "n_eu_sec_with_sedol", "n_eu_sec_matched"]
    values = [float(row.get(k, 0)) for k in keys]
    if max(values) == 0:
        print("  fig14: empty cascade row")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ypos = np.arange(len(labels))[::-1]
    ax.barh(ypos, values, color=PAL["blue"], alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    for yp, v in zip(ypos, values):
        ax.text(v + max(values) * 0.01, yp, f"{int(round(v)):,}",
                va="center", ha="left", fontsize=9)
    ax.set_xlim(0, max(values) * 1.12)
    fmt_thousand_axis(ax, "x")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("# distinct EU sec_entity_ids")
    ax.set_title("Coverage cascade: FactSet EU holdings universe to Revere match")
    save(fig, "fig14_coverage_cascade")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"OUT_DIR = {OUT_DIR}")
    print(f"FIG_DIR = {FIG_DIR}")
    print("--- China exposure of EU firms ---")
    fig_china_exposure()
    print("--- US-EU engagement ---")
    fig_us_eu_engagement()
    print("--- US ownership distribution + by country ---")
    fig_us_ownership()
    print("--- Pre-regression descriptive ---")
    fig_preregression()
    print("--- Coverage cascade ---")
    fig_coverage_cascade()
    print("All done.")

if __name__ == "__main__":
    main()
