# plots/plot_fig_ab.py
# ===========================================================================
# FIGURE A and FIGURE B for the advisor update (Emanuele, meeting 2026-08-04).
# Consumes only the CSVs written by build_fig_ab_data.py -- no re-aggregation
# here, so the numbers in the figures and in the CSVs cannot drift apart.
#
# BACKGROUND SERIES, and how it is drawn.
# The advisor asked for the US->China tension series "in the background". It is
# the USA|China direction of the Iacoviello-Tong bilateral AI-GPR index, i.e.
# US-media attention to geopolitical risk involving China -- NOT the China|USA
# direction, which is a different column of the same file. Every figure states
# the direction in the caption.
# It is rendered TWICE, deliberately:
#   (a) as its own top panel, with its own labelled axis, so the level can
#       actually be read; and
#   (b) as a light grey silhouette behind each data panel, drawn on a blended
#       transform (x = data, y = axes fraction). It therefore carries NO second
#       y-scale and no ticks -- it is background context, not a second series.
# This avoids a dual-axis chart, which would be unreadable and is the single
# most common charting error, while still giving the advisor the visual
# overlay he asked for.
#
# COLOR. Taken unchanged from the data-viz reference palette (blue categorical
# slot 1 #2a78d6, orange slot 2 #eb6834, blue ordinal ramp steps 250 #86b6ef
# and 600 #184f95, both at or darker than the documented light-surface ordinal
# floor of step 250). The palette validator ships as a node script and node is
# not installed on this machine, so no new colors were invented: every hex
# below is a documented, already-validated value used in its documented role.
# Encoding is never colour-alone -- each figure carries a legend AND direct
# end-of-line labels, and the ordered arms additionally differ in line weight.
#
# SERIES CAP. Each panel draws at most three lines. The quartile figures keep
# Q2 and Q3 in the CSV but plot only the bottom (Q1_low) and top (Q4_high)
# arms the advisor asked for, with the zero bucket as a recessive reference.
#
# OUTPUTS (PDF + PNG for each):
#   plots/fig_A_firm_zero_vs_positive.{pdf,png}
#   plots/fig_A_firm_quartile_ew.{pdf,png}
#   plots/fig_A_firm_quartile_mcapw.{pdf,png}
#   plots/fig_A_country_{m1,m2,m3}_quartiles.{pdf,png}
#   plots/fig_B_link_growth_us2.{pdf,png}
#   plots/fig_B_link_growth_us3.{pdf,png}
# ===========================================================================

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
_env_out = os.environ.get("DPN_OUT_DIR", "").strip()
OUT = Path(_env_out).resolve() if _env_out else PROJ / "output"
PLOTS = PROJ / "plots"
PLOTS.mkdir(exist_ok=True)

F_A_FIRM   = OUT / "fig_A_firm_buckets.csv"
F_A_CTRY   = OUT / "fig_A_country_buckets.csv"
F_B_GROWTH = OUT / "fig_B_link_growth.csv"
F_TENSION  = OUT / "fig_ab_tension_series.csv"

PLOT_START = pd.Timestamp("2012-03-31")
PLOT_END   = pd.Timestamp("2023-12-31")
BASE_QUARTER = pd.Timestamp("2017-12-31")

TENSION_LABEL = ("Bilateral AI-GPR, USA|China direction "
                 "(US-media geopolitical-risk attention to China)")

# --- palette (documented values, documented roles) -------------------------
C_BG    = "#fcfcfb"
C_TEXT  = "#0b0b0b"
C_TEXT2 = "#52514e"
C_GRID  = "#e8e8e6"
C_ZERO  = "#eb6834"   # categorical slot 2 -- the "no tie" category
C_POS   = "#2a78d6"   # categorical slot 1 -- the "has tie" category
C_LOW   = "#86b6ef"   # blue ordinal step 250 -- bottom arm
C_HIGH  = "#184f95"   # blue ordinal step 600 -- top arm
C_REF   = "#9c9b94"   # recessive reference line (residual / middle half)
C_SHADE = "#d8d7d1"   # background tension silhouette

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "figure.facecolor": C_BG,
    "savefig.facecolor": C_BG,
    "pdf.fonttype": 42,
})


def _load_tension() -> pd.DataFrame:
    t = pd.read_csv(F_TENSION, parse_dates=["quarter_end"])
    return t[(t["quarter_end"] >= PLOT_START) & (t["quarter_end"] <= PLOT_END)]


def _style(ax) -> None:
    ax.set_facecolor(C_BG)
    ax.grid(axis="y", color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT2, labelsize=8, length=3)
    ax.set_xlim(PLOT_START, PLOT_END)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def _shade_tension(ax, ten: pd.DataFrame) -> None:
    """Background silhouette of the tension series: x in data coordinates,
    y in AXES FRACTION. No second y-scale, no ticks, nothing to misread as a
    data value -- the readable version lives in the dedicated top panel."""
    v = ten["gpr_us_cn"].to_numpy(dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    frac = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(ten["quarter_end"], 0.0, frac * 0.97, transform=tr,
                    color=C_SHADE, alpha=0.55, lw=0, zorder=0)


def _tension_panel(ax, ten: pd.DataFrame) -> None:
    ax.fill_between(ten["quarter_end"], 0.0, ten["gpr_us_cn"],
                    color=C_SHADE, alpha=0.85, lw=0)
    ax.plot(ten["quarter_end"], ten["gpr_us_cn"], color=C_TEXT2, lw=1.2)
    _style(ax)
    ax.set_ylabel("index", fontsize=8, color=C_TEXT2)
    ax.set_title(TENSION_LABEL, fontsize=9, color=C_TEXT, loc="left", pad=3)
    ax.set_ylim(bottom=0)


def _draw_series(ax, data: pd.DataFrame, series: list[tuple[str, str, str, float]],
                 ycol: str, ten: pd.DataFrame, title: str, ylabel: str,
                 logy: bool = False, pct: bool = False, hline: float | None = None,
                 label_fmt=lambda v: f"{v:,.0f}") -> None:
    """series = [(bucket, display label, color, linewidth), ...]"""
    _shade_tension(ax, ten)
    xs_end = []
    for bucket, lbl, color, lw in series:
        s = (data.loc[data["bucket"].eq(bucket)]
                 .sort_values("quarter_end")
                 .set_index("quarter_end")[ycol])
        s = s[(s.index >= PLOT_START) & (s.index <= PLOT_END)]
        ax.plot(s.index, s.to_numpy(), color=color, lw=lw, label=lbl,
                solid_capstyle="round", zorder=3)
        last = s.dropna()
        if len(last):
            xs_end.append((last.index[-1], float(last.iloc[-1]), color, lbl))
    if hline is not None:
        ax.axhline(hline, color=C_TEXT2, lw=0.7, alpha=0.6, zorder=1)
    _style(ax)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0),
                                                      numticks=12))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:,.2f}"))
    if pct:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    elif not logy:
        # NB: must not run on a log panel — MaxNLocator would silently replace
        # the LogLocator set above and print near-linear ticks on a log axis.
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.set_title(title, fontsize=9.5, color=C_TEXT, loc="left", pad=3)
    ax.set_ylabel(ylabel, fontsize=8, color=C_TEXT2)
    _end_labels(ax, xs_end, label_fmt)


def _end_labels(ax, xs_end, label_fmt, min_gap: float = 0.075) -> None:
    """Direct end-of-line labels, de-collided.

    Identity must never be carried by colour alone, so every line gets a label;
    but two lines that end at nearly the same value would print their labels on
    top of each other. Positions are converted to axes fraction, sorted, and
    pushed apart to a minimum vertical gap (then clamped back inside the panel),
    so the label order still matches the line order at the right edge."""
    if not xs_end:
        return
    inv = ax.transAxes.inverted()
    items = []
    for _x, y, color, lbl in xs_end:
        if not np.isfinite(y):
            continue
        frac = float(inv.transform(ax.transData.transform((0.0, y)))[1])
        items.append([frac, color, f"{lbl}: {label_fmt(y)}"])
    if not items:
        return
    items.sort(key=lambda r: r[0])
    for i in range(1, len(items)):                      # push up from below
        if items[i][0] - items[i - 1][0] < min_gap:
            items[i][0] = items[i - 1][0] + min_gap
    overflow = items[-1][0] - 1.0
    if overflow > 0:                                    # then clamp downward
        for it in items:
            it[0] -= overflow
        for i in range(len(items) - 2, -1, -1):
            if items[i + 1][0] - items[i][0] < min_gap:
                items[i][0] = items[i + 1][0] - min_gap
    for frac, color, text in items:
        ax.annotate(text, xy=(1.0, frac), xycoords="axes fraction",
                    xytext=(6, 0), textcoords="offset points", color=color,
                    fontsize=7.5, va="center", ha="left",
                    annotation_clip=False, zorder=4)


def _new_figure():
    """4 stacked panels: the readable tension panel on top, three data panels
    below. Margins are set explicitly rather than by tight_layout, because the
    direct end-of-line labels live OUTSIDE the axes and tight_layout would
    either clip them or fight the footnote block."""
    fig, axes = plt.subplots(4, 1, figsize=(11.4, 11.2), sharex=True,
                             height_ratios=[0.58, 1, 1, 1])
    fig.subplots_adjust(left=0.070, right=0.760, top=0.900, bottom=0.150,
                        hspace=0.30)
    return fig, axes


def _finish(fig, axes, path_stem: str, suptitle: str, subtitle: str,
            note: str) -> list[Path]:
    fig.suptitle(suptitle, x=0.010, y=0.984, ha="left", fontsize=12.5,
                 color=C_TEXT, fontweight="bold")
    fig.text(0.010, 0.958, subtitle, ha="left", va="top", fontsize=9.2,
             color=C_TEXT2)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.010, 0.937),
               frameon=False, fontsize=8.4, ncols=len(labels),
               labelcolor=C_TEXT2, handlelength=1.8, columnspacing=1.8)
    fig.text(0.010, 0.012, textwrap.fill(" ".join(note.split()), width=196),
             ha="left", va="bottom", fontsize=7.2, color=C_TEXT2,
             linespacing=1.35)
    written = []
    for ext in ("pdf", "png"):
        p = PLOTS / f"{path_stem}.{ext}"
        fig.savefig(p, dpi=180 if ext == "png" else None, facecolor=C_BG)
        written.append(p)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
TIMING_ROLLING = ("TIMING: buckets are assigned on the firm's China supply-chain "
                  "link share CN at t-1; holdings are measured at t.")


def figure_a_firm(split: str, series: list[tuple[str, str, str, float]],
                  stem: str, subtitle: str, extra_note: str,
                  ten: pd.DataFrame, timing_note: str = TIMING_ROLLING) -> list[Path]:
    d = pd.read_csv(F_A_FIRM, parse_dates=["quarter_end"])
    d = d.loc[d["split"].eq(split) & d["holder_group"].eq("US")]
    if d.empty:
        raise RuntimeError(f"no rows for split={split}")

    fig, axes = _new_figure()
    _tension_panel(axes[0], ten)

    dd = d.copy()
    dd["real_bn"] = dd["real_usd_2020"] / 1e9
    _draw_series(axes[1], dd, series, "real_bn", ten,
                 "US institutional holdings, level (real 2020 USD bn, log scale)",
                 "bn 2020 USD", logy=True,
                 label_fmt=lambda v: f"{v:,.0f}bn")
    _draw_series(axes[2], d, series, "idx100", ten,
                 f"Same, indexed to 100 at {BASE_QUARTER.date()} (headline normalization)",
                 "index", hline=100.0, label_fmt=lambda v: f"{v:,.0f}")
    _draw_series(axes[3], d, series, "share_of_book", ten,
                 "Share of the US investors' whole European book held in the bucket",
                 "share", pct=True, label_fmt=lambda v: f"{100*v:,.1f}%")

    note = (
        f"Sample: European firms in the US/non-US ownership grid, 2012Q1-2023Q4. "
        f"{timing_note} "
        f"CN = (China customer + China supplier links) / (all customer + supplier "
        f"links); a Revere-covered firm with no supply-chain links is a genuine "
        f"zero, a firm absent from Revere at t-1 is unclassified and excluded, so the "
        f"buckets need not exhaust the book. "
        f"Levels are CPI-U deflated (CPIAUCNS, quarter-end month, 2020 annual-average "
        f"base) exactly as in the Figure-2 builder. Background silhouette in the "
        f"three lower panels = {TENSION_LABEL}, scaled to panel height with no "
        f"axis and no ticks; read its level off the top panel. {extra_note}")
    return _finish(fig, axes, stem,
                   "Figure A. US holdings of European firms, by China-link exposure",
                   subtitle, note)


# ---------------------------------------------------------------------------
MEASURE_DESC = {
    "M1": "M1, the country-aggregate China link share (sum of China customer+supplier "
          "links / sum of all customer+supplier links, ownership-matched universe)",
    "M2": "M2, the market-cap-weighted average of firm-level China link shares",
    "M3": "M3, the equal-weighted (1/N) average of firm-level China link shares",
}


def figure_a_country(ten: pd.DataFrame) -> list[Path]:
    # (2026-08-09) one figure per exposure measure, per the 2026-08-04 minute
    # ("each figure has one version per exposure measure (M1, M2, M3)").
    dd_all = pd.read_csv(F_A_CTRY, parse_dates=["quarter_end"])
    if "measure" not in dd_all.columns:
        dd_all["measure"] = "M1"
    paths: list[Path] = []
    for meas in sorted(dd_all["measure"].unique()):
        paths += _figure_a_country_one(ten, dd_all.loc[dd_all["measure"].eq(meas)], meas)
    return paths


def _figure_a_country_one(ten: pd.DataFrame, d: pd.DataFrame, meas: str) -> list[Path]:
    d = d.loc[d["holder_group"].eq("US")]
    series = [("bottom_quartile", f"bottom quartile (low {meas})", C_LOW, 2.0),
              ("mid_half", "middle half", C_REF, 1.4),
              ("top_quartile", f"top quartile (high {meas})", C_HIGH, 2.0)]

    fig, axes = _new_figure()
    _tension_panel(axes[0], ten)
    dd = d.copy()
    dd["real_bn"] = dd["real_usd_2020"] / 1e9
    _draw_series(axes[1], dd, series, "real_bn", ten,
                 "US institutional holdings, level (real 2020 USD bn, log scale)",
                 "bn 2020 USD", logy=True, label_fmt=lambda v: f"{v:,.0f}bn")
    _draw_series(axes[2], d, series, "idx100", ten,
                 f"Same, indexed to 100 at {BASE_QUARTER.date()} (headline normalization)",
                 "index", hline=100.0, label_fmt=lambda v: f"{v:,.0f}")
    _draw_series(axes[3], d, series, "share_of_book", ten,
                 "Share of the US investors' whole European book held in the bucket",
                 "share", pct=True, label_fmt=lambda v: f"{100*v:,.1f}%")

    note = (
        f"Countries are ranked each quarter on {MEASURE_DESC[meas]}, "
        "then split at the cross-country 25th "
        "and 75th percentiles.\n"
        f"TIMING: the ranking uses {meas} at t-1; holdings are measured at t. Countries "
        "with fewer than 20 firms in a quarter are not rankable and are dropped that "
        "quarter, so the three lines need not exhaust the book.\n"
        "The level panel is on a log scale because the two arms differ by roughly two "
        "orders of magnitude in raw dollars -- a mechanical size difference, which is "
        "exactly why the indexed and share panels are the headline. "
        "READ THE PRE-2017 BOTTOM-QUARTILE LINE WITH CARE: bucket membership churns "
        "far more before 2017 than after"
        + (" (under M1: 5.0 of 26 countries change bucket per quarter over 2012-2016, "
           "against 2.0 over 2018-2023; Spain crosses the 25th-percentile cut in 2015Q2 "
           "and alone moves the arm by hundreds of index points)" if meas == "M1" else "")
        + ", so single large countries crossing a percentile cut can move an arm "
        "sharply. The quarter-by-quarter composition (per measure) is in "
        "fig_A_country_bucket_membership.csv. "
        f"Background silhouette = {TENSION_LABEL}, scaled to panel height with no "
        "axis and no ticks; read its level off the top panel.")
    return _finish(fig, axes, f"fig_A_country_{meas.lower()}_quartiles",
                   "Figure A (country level). US holdings by country China-exposure quartile",
                   f"Countries split on {meas}, measured at t-1",
                   note)


# ---------------------------------------------------------------------------
def figure_b(split: str, series: list[tuple[str, str, str, float]],
             stem: str, subtitle: str, ten: pd.DataFrame) -> list[Path]:
    d = pd.read_csv(F_B_GROWTH, parse_dates=["quarter_end"])
    d = d.loc[d["split"].eq(split)]
    if d.empty:
        raise RuntimeError(f"no rows for split={split}")

    # cumulative view: total China links indexed to 100 at the base quarter
    base = (d.loc[d["quarter_end"].eq(BASE_QUARTER), ["bucket", "cn_links_t"]]
              .rename(columns={"cn_links_t": "_b"}))
    d = d.merge(base, on="bucket", how="left", validate="many_to_one")
    d["links_idx100"] = 100.0 * d["cn_links_t"] / d["_b"].where(d["_b"] > 0)

    fig, axes = _new_figure()
    _tension_panel(axes[0], ten)
    _draw_series(axes[1], d, series, "growth_yoy", ten,
                 "Year-over-year growth in China supply-chain links (headline)",
                 "YoY", pct=True, hline=0.0,
                 label_fmt=lambda v: f"{100*v:,.1f}%")
    _draw_series(axes[2], d, series, "links_idx100", ten,
                 f"Total China links, indexed to 100 at {BASE_QUARTER.date()}",
                 "index", hline=100.0, label_fmt=lambda v: f"{v:,.0f}")
    _draw_series(axes[3], d, series, "mean_cn_links_t", ten,
                 "China supply-chain links per firm in the bucket",
                 "links / firm", label_fmt=lambda v: f"{v:,.2f}")

    note = (
        "Pre-emptive de-linking test. TIMING: firms are bucketed on their US "
        "institutional ownership at t-1; the link outcome is measured at t.\n"
        "Growth is computed on a cohort FIXED at t-1 and required to have a defined "
        "China link count at t-4, t-1 and t, so numerator and denominator cover the "
        "same firms and Revere coverage expansion cannot manufacture growth. Links "
        "are summed across the bucket before dividing, so firms with zero China links "
        "contribute 0 rather than an undefined firm-level growth rate.\n"
        "'no US' = zero US institutional holdings at t-1 (the ownership grid is "
        "zero-filled, so this is a genuine zero). Where three buckets are shown, "
        "positive-US firms are split at that quarter's median US dollar holding. "
        f"Background silhouette = {TENSION_LABEL}, scaled to panel height with no "
        "axis and no ticks; read its level off the top panel.")
    return _finish(fig, axes, stem,
                   "Figure B. China supply-chain link growth, by US ownership at t-1",
                   subtitle, note)


# ---------------------------------------------------------------------------
def main() -> None:
    for p in (F_A_FIRM, F_A_CTRY, F_B_GROWTH, F_TENSION):
        if not p.is_file():
            raise FileNotFoundError(f"missing input {p} — run build_fig_ab_data.py first")
    ten = _load_tension()
    print(f"tension background: {len(ten)} quarters, "
          f"gpr_us_cn range [{ten['gpr_us_cn'].min():.3f}, {ten['gpr_us_cn'].max():.3f}]")

    written: list[Path] = []

    written += figure_a_firm(
        "zero_vs_positive",
        [("zero_tie", "zero China ties", C_ZERO, 2.0),
         ("positive_tie", "positive China ties", C_POS, 2.0)],
        "fig_A_firm_zero_vs_positive", "variant (i), zero vs positive ties",
        "Both arms are Revere-covered at t-1; the zero arm is a measured zero, "
        "not a missing value. CAVEAT, and the reason the companion figure exists: "
        "with the rolling t-1 rule the positive arm grows from 144 firms in 2012Q1 "
        "to 766 in 2023Q1 because Revere keeps adding relationships, so part of the "
        "rise in its share of the book is RECLASSIFICATION, not reallocation. "
        "fig_A_firm_zero_vs_positive_fixedcohort freezes bucket membership at "
        "2017Q4 and isolates the reallocation component.", ten)

    written += figure_a_firm(
        "zero_vs_positive_fixed",
        [("zero_tie_fixed", "zero China ties in 2017Q4", C_ZERO, 2.0),
         ("positive_tie_fixed", "positive China ties in 2017Q4", C_POS, 2.0)],
        "fig_A_firm_zero_vs_positive_fixedcohort",
        "variant (i), composition control: membership frozen at 2017Q4",
        "Firms not classifiable in 2017Q4 are excluded throughout. Because the firm "
        "set is frozen, movement here is portfolio reallocation rather than Revere "
        "reclassifying firms into the positive arm; compare against the rolling "
        "version in fig_A_firm_zero_vs_positive.", ten,
        timing_note=("TIMING: this figure DEPARTS from the t-1 convention on purpose. "
                     "Each firm is bucketed ONCE on its CN at 2017Q4 (7,379 zero-tie, "
                     "491 positive-tie) and keeps that bucket in every quarter; only "
                     "the holdings vary with t."))

    quart_series = [("zero_tie", "zero ties (reference)", C_ZERO, 1.4),
                    ("Q1_low", "bottom quartile of CN", C_LOW, 2.0),
                    ("Q4_high", "top quartile of CN", C_HIGH, 2.0)]
    written += figure_a_firm(
        "quartile_ew", quart_series,
        "fig_A_firm_quartile_ew", "variant (ii), equal-weighted quartiles (M3-style)",
        "Quartile cutpoints are the unweighted 25/50/75th percentiles of CN among "
        "positive-CN firms that quarter (M3-style, one firm one vote); assignment is "
        "by value, so tied ratios never straddle a boundary. Q2 and Q3 are in the CSV "
        "but omitted here to keep three lines per panel. The zero bucket is drawn as "
        "its own recessive line rather than folded into Q1: after the zero recode it "
        "mixes firms with a supply-chain denominator and no China link with firms that "
        "have no supply-chain links at all, so it is not the low tail of the same "
        "distribution.", ten)
    written += figure_a_firm(
        "quartile_mcapw", quart_series,
        "fig_A_firm_quartile_mcapw", "variant (ii), market-cap-weighted quartiles (M2-style)",
        "Quartile cutpoints are market-cap-weighted quantiles of CN at t-1 (M2-style). "
        "The whole classification set, zero bucket included, is restricted to firms "
        "with a market cap at t-1 so every line shares one universe; bucket counts are "
        "therefore very uneven by construction and are reported in the CSV.", ten)

    written += figure_a_country(ten)

    written += figure_b(
        "us_2bucket",
        [("no_us", "no US investors at t-1", C_ZERO, 2.0),
         ("any_us", "any US investors at t-1", C_POS, 2.0)],
        "fig_B_link_growth_us2", "two buckets", ten)
    written += figure_b(
        "us_3bucket",
        [("no_us", "no US investors", C_ZERO, 1.6),
         ("us_low", "US-held, below median", C_LOW, 2.0),
         ("us_high", "US-held, above median", C_HIGH, 2.0)],
        "fig_B_link_growth_us3", "three buckets", ten)

    print("\nwrote:")
    for p in written:
        print(f"  {p}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    sys.exit(main())
