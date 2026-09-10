# plots/plot_fig_ab.py
# ===========================================================================
# FIGURE A and FIGURE B for the advisor update (Emanuele, meeting 2026-08-04).
# Consumes only the CSVs written by build_fig_ab_data.py -- no re-aggregation
# here, so the numbers in the figures and in the CSVs cannot drift apart.
#
# BACKGROUND SERIES, and how it is drawn.
# The advisor asked for the US->China tension series "in the background". It is
# the USA|China direction of a bilateral newspaper-text geopolitical-risk index:
# the United States is the ACTING party of the dyad and China the counterparty.
# It is NOT a measure of "US media attention" -- the corpus is a shared newspaper
# panel, not a US-only one -- and it is NOT the China|USA direction, which is a
# different column of the same file. The source citation is still unconfirmed. Every figure states
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
# SERIES CAP. Each panel draws at most three lines. The quartile figures plot the
# lowest and highest CN quartiles -- the low-versus-high contrast requested by
# Stefano on 2026-08-04 -- with the zero bucket as a recessive reference. Q2 and
# Q3 are computed and written to the CSV; their end-of-sample values are quoted in
# the caption so the reader can check monotonicity without a five-line chart.
#
# SHARE PANEL DENOMINATOR (2026-08-10). The share panel plots
# share_of_book_global -- the share of the holder group's GLOBAL (full
# FactSet-identifiable equity) book -- consistent with the main regression's
# global denominator (dw, 2026-08-08 GLOBAL-MAIN decision). The EU-book
# version of the same series (share_of_book_eu) stays in the CSVs as the
# within-Europe reallocation diagnostic and is NOT plotted. The dollar and
# idx100 panels are denominator-free and unchanged.
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
F_A_CHAIN  = OUT / "fig_A_firm_chained.csv"   # built by build_fig_a_chained.py

# Plot the FULL available span of each series rather than a hand-set window, so
# that the early-sample Revere coverage build-out is visible to the reader
# instead of being silently cropped away. _XSTART is set per figure from the
# data itself. No "interpretable window" marker is drawn: the bucket-size panel
# shows where the classification settles, and the reader can judge it directly.
PLOT_START = pd.Timestamp("1999-03-31")
_XSTART = PLOT_START
PLOT_END   = pd.Timestamp("2023-12-31")
BASE_QUARTER = pd.Timestamp("2017-12-31")

TENSION_LABEL = ("Bilateral geopolitical-risk index, USA|China direction "
                 "(United States acting, China counterparty)")

# --- palette (documented values, documented roles) -------------------------
C_BG    = "#fcfcfb"
C_TEXT  = "#0b0b0b"
C_TEXT2 = "#52514e"
C_GRID  = "#e8e8e6"
C_ZERO  = "#eb6834"   # categorical slot 2 -- the "no tie" category
C_POS   = "#2a78d6"   # categorical slot 1 -- the "has tie" category
C_LOW   = "#86b6ef"   # blue ordinal step 250 -- bottom arm
C_Q2    = "#3987e5"   # blue ordinal step 400 -- second quartile
C_Q3    = "#1c5cab"   # blue ordinal step 550 -- third quartile
# Steps 250/400/550/600 are the widest spacing the documented light-surface ramp
# allows for four ordered arms (250 is the light floor, 600 the dark floor). The
# node validator is not installed here, so separation was not machine-checked;
# identity does not rest on colour alone, since every arm carries a direct
# end-of-line label and the top arm is drawn heavier.
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
    return t[(t["quarter_end"] >= _XSTART) & (t["quarter_end"] <= PLOT_END)]


def _style(ax) -> None:
    ax.set_facecolor(C_BG)
    ax.grid(axis="y", color=C_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT2, labelsize=8, length=3)
    ax.set_xlim(_XSTART, PLOT_END)
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
    ax.set_title(_paper_title(TENSION_LABEL), fontsize=9, color=C_TEXT,
                 loc="left", pad=3)
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
    ax.set_title(_paper_title(title), fontsize=9.5, color=C_TEXT,
                 loc="left", pad=3)
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


def _set_xstart(*frames) -> None:
    """Start every axis at the first quarter the figure's own data actually
    covers, so nothing is cropped and the reader sees the full history."""
    global _XSTART
    mins = [f["quarter_end"].min() for f in frames if f is not None and len(f)]
    _XSTART = min(mins) if mins else PLOT_START


def _new_figure(counts_panel: bool = False):
    """Stacked panels sharing one x axis: the readable tension panel on top,
    three data panels below, and optionally a bucket-size panel at the bottom.
    Margins are set explicitly rather than by tight_layout, because the direct
    end-of-line labels live OUTSIDE the axes and tight_layout would either clip
    them or fight the footnote block."""
    if counts_panel:
        fig, axes = plt.subplots(5, 1, figsize=(11.4, 12.9), sharex=True,
                                 height_ratios=[0.58, 1, 1, 1, 0.62])
    else:
        fig, axes = plt.subplots(4, 1, figsize=(11.4, 11.2), sharex=True,
                                 height_ratios=[0.58, 1, 1, 1])
    fig.subplots_adjust(left=0.070, right=0.760, top=0.900, bottom=0.150,
                        hspace=0.30)
    return fig, axes


def _n_fixed(bucket: str) -> str:
    """True fixed-cohort arm size, read from the data so the note cannot go
    stale when the panel is rebuilt."""
    d = pd.read_csv(F_A_FIRM, usecols=["split", "bucket", "n_firms"])
    v = d.loc[d["split"].eq("zero_vs_positive_fixed") & d["bucket"].eq(bucket),
              "n_firms"]
    return f"{int(v.iloc[0]):,}" if len(v) else "n/a"


def _n_roll(quarter_end: str) -> str:
    """Rolling positive-arm firm count at a given quarter, read from the data so
    the caveat text cannot go stale when the panel is rebuilt."""
    d = pd.read_csv(F_A_FIRM, usecols=["split", "bucket", "holder_group",
                                       "quarter_end", "n_firms"])
    v = d.loc[d["split"].eq("zero_vs_positive") & d["bucket"].eq("positive_tie")
              & d["holder_group"].eq("US") & d["quarter_end"].eq(quarter_end), "n_firms"]
    return f"{int(v.iloc[0]):,}" if len(v) else "n/a"


def _counts_panel(ax, data: pd.DataFrame,
                  series: list[tuple[str, str, str, float]],
                  title: str, ylabel: str = "firms",
                  col: str = "n_firms") -> None:
    """Number of firms in each arm, quarter by quarter. This panel is the
    figure's own audit trail: where the counts step up in batches, the arms are
    being repopulated by the relationship data rather than by anything
    economic, and movement in the panels above should be read accordingly."""
    xs_end = []
    for bucket, lbl, color, lw in series:
        s = (data.loc[data["bucket"].eq(bucket)]
                 .sort_values("quarter_end")
                 .set_index("quarter_end")[col])
        s = s[(s.index >= PLOT_START) & (s.index <= PLOT_END)].dropna()
        if s.empty:
            continue
        ax.plot(s.index, s.to_numpy(), color=color, lw=lw, solid_capstyle="round")
        xs_end.append((s.index[-1], float(s.iloc[-1]), color, lbl))
    _style(ax)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel(ylabel, fontsize=8, color=C_TEXT2)
    ax.set_title(_paper_title(title), fontsize=9.5, color=C_TEXT, loc="left", pad=3)
    _end_labels(ax, xs_end, lambda v: f"{v:,.0f}")


# Paper mode: suppress the in-figure title, subtitle and technical footnote
# (the LaTeX caption in Essay2_figures_AB.tex carries all three) and write to
# *_paper.{pdf,png} alongside the standard files. Enable with --paper on the
# command line or DPN_FIG_PAPER=1.
PAPER = ("--paper" in sys.argv) or (os.environ.get("DPN_FIG_PAPER", "0") == "1")


def _paper_title(title: str) -> str:
    """Panel-title rewrites applied only in paper mode: same content, journal
    wording, no internal shorthand ("bucket", "headline", vendor labels)."""
    if not PAPER:
        return title
    rules = [
        ("Bilateral geopolitical-risk index, USA|China direction "
         "(United States acting, China counterparty)",
         "US-China geopolitical tension index "
         "(USA|China direction: United States acting, China counterparty)"),
        (" (headline normalization)", ""),
        (" (headline)", ""),
        ("Share of the US investors' GLOBAL (full FactSet-identifiable "
         "equity) book held in the bucket",
         "Share of the US investors' worldwide equity portfolio "
         "held by the group"),
        ("per firm in the bucket", "per firm in the group"),
    ]
    for old, new in rules:
        title = title.replace(old, new)
    return title


def _finish(fig, axes, path_stem: str, suptitle: str, subtitle: str,
            note: str) -> list[Path]:
    handles, labels = axes[1].get_legend_handles_labels()
    if PAPER:
        fig.subplots_adjust(top=0.945, bottom=0.045)
        fig.legend(handles, labels, loc="upper left",
                   bbox_to_anchor=(0.010, 0.998), frameon=False, fontsize=8.4,
                   ncols=len(labels), labelcolor=C_TEXT2, handlelength=1.8,
                   columnspacing=1.8)
        path_stem = f"{path_stem}_paper"
    else:
        fig.suptitle(suptitle, x=0.010, y=0.984, ha="left", fontsize=12.5,
                     color=C_TEXT, fontweight="bold")
        fig.text(0.010, 0.958, subtitle, ha="left", va="top", fontsize=9.2,
                 color=C_TEXT2)
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
    if "share_of_book_global" not in d.columns:
        raise RuntimeError(
            f"{F_A_FIRM.name} has no share_of_book_global column — it predates the "
            "2026-08-10 global-denominator switch. Re-run build_fig_ab_data.py.")
    d = d.loc[d["split"].eq(split) & d["holder_group"].eq("US")]
    if d.empty:
        raise RuntimeError(f"no rows for split={split}")

    # Every variant carries a bucket-size panel, but they answer different
    # questions. The rolling variants plot how many firms are CLASSIFIED into
    # each arm, which exposes the Revere coverage build-out. The fixed-cohort
    # variant has a constant classified population by construction, so it plots
    # how many of those firms are actually HELD, which exposes the semi-annual
    # reporting pattern of the early 2000s.
    fixed = split.endswith("_fixed")
    show_counts = True
    counts_col = "n_firms_held" if fixed else "n_firms"
    counts_title = ("Number of firms in each bucket actually held (log scale)"
                    if fixed else
                    "Number of firms classified into each bucket (log scale)")
    _set_xstart(d)
    ten = _load_tension()
    fig, axes = _new_figure(counts_panel=show_counts)
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
    # (2026-08-10) GLOBAL denominator = MAIN, matching the regression's dw.
    _draw_series(axes[3], d, series, "share_of_book_global", ten,
                 "Share of the US investors' GLOBAL (full FactSet-identifiable "
                 "equity) book held in the bucket",
                 "share of global book", pct=True,
                 label_fmt=lambda v: f"{100*v:,.2f}%")
    if show_counts:
        _counts_panel(axes[4], d, series, counts_title,
                      ylabel="firms held" if fixed else "firms",
                      col=counts_col)

    note = (
        f"Sample: European firms in the US/non-US ownership grid, "
        f"full available span. "
        f"{timing_note} "
        f"CN = (China customer + China supplier links) / (all customer + supplier "
        f"links); a Revere-covered firm with no supply-chain links is a genuine "
        f"zero, a firm absent from Revere at t-1 is unclassified and excluded, so the "
        f"buckets need not exhaust the book. "
        f"Levels are CPI-U deflated (CPIAUCNS, quarter-end month, 2020 annual-average "
        f"base) exactly as in the Figure-2 builder. "
        f"The share panel uses the GLOBAL (full FactSet-identifiable equity) "
        f"denominator, the same denominator as the main regression; the EU-book "
        f"version of the series (share_of_book_eu in fig_A_firm_buckets.csv) is the "
        f"within-Europe reallocation diagnostic and is not plotted. "
        f"Background silhouette in the "
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


def figure_a_chained(ten: pd.DataFrame) -> list[Path]:
    """Composition control without a freeze date (2026-08-21): a chain-linked
    matched-sample index. Each quarter's growth uses only firms present in that
    quarter and the previous one AND classified in the same arm at both dates,
    so roster changes cannot move the lines by construction. Replaces the
    frozen-cohort variant as the figure shown; the residual it leaves is a
    single number rather than a function of a chosen freeze quarter."""
    d = pd.read_csv(F_A_CHAIN, parse_dates=["quarter_end"])
    series = [("zero", "zero China ties", C_ZERO, 2.0),
              ("positive", "positive China ties", C_POS, 2.0)]

    _set_xstart(d)
    ten = _load_tension()
    fig, axes = plt.subplots(3, 1, figsize=(11.4, 8.6), sharex=True,
                             height_ratios=[0.58, 1.6, 0.62])
    fig.subplots_adjust(left=0.070, right=0.760, top=0.900, bottom=0.150,
                        hspace=0.30)
    _tension_panel(axes[0], ten)
    _draw_series(axes[1], d, series, "chained_idx", ten,
                 f"Chain-linked matched-sample index, {BASE_QUARTER.date()} = 100",
                 "index", hline=100.0, label_fmt=lambda v: f"{v:,.0f}")
    _counts_panel(axes[2], d, series,
                  "Matched firms entering each quarter's growth (log scale)",
                  ylabel="firms", col="matched_firms")

    note = (
        "Composition control WITHOUT a freeze date. For each quarter t and each arm, "
        "growth is computed only over firms present in t-1 and t AND classified in "
        "the same arm (on CN at t-1) at both dates; quarterly growth rates are then "
        "chained. A firm whose links enter Revere, or that switches arms, contributes "
        "nothing in the quarter it moves, so the roster growth visible in "
        "fig_A_firm_zero_vs_positive cannot move these lines. The 2017Q4 = 100 base "
        "is presentation only: it fixes where the lines cross 100, not who is "
        "compared with whom. Values are real 2020 USD before growth is taken. "
        "Pre-2009 zigzag = the odd-quarter reporting gap (many US funds filed "
        "semiannually), which also thins the matched counts in those quarters. "
        "Supersedes the frozen-2017Q4 cohort variant: a sweep over all 52 candidate "
        "freeze quarters left residual gaps from -19.5 to +18.4 index points "
        "(positive in 24 of 52), so no freeze-dependent residual is interpretable.")

    return _finish(fig, axes, "fig_A_firm_zero_vs_positive_chained",
                   "Figure A. US holdings of European firms, by China-link exposure",
                   "variant (i), composition control: chain-linked matched sample",
                   note)


def figure_a_country(ten: pd.DataFrame) -> list[Path]:
    # (2026-08-09) one figure per exposure measure, per the 2026-08-04 minute
    # ("each figure has one version per exposure measure (M1, M2, M3)").
    dd_all = pd.read_csv(F_A_CTRY, parse_dates=["quarter_end"])
    if "share_of_book_global" not in dd_all.columns:
        raise RuntimeError(
            f"{F_A_CTRY.name} has no share_of_book_global column — it predates the "
            "2026-08-10 global-denominator switch. Re-run build_fig_ab_data.py.")
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

    _set_xstart(d)
    ten = _load_tension()
    fig, axes = _new_figure(counts_panel=True)
    _tension_panel(axes[0], ten)
    dd = d.copy()
    dd["real_bn"] = dd["real_usd_2020"] / 1e9
    _draw_series(axes[1], dd, series, "real_bn", ten,
                 "US institutional holdings, level (real 2020 USD bn, log scale)",
                 "bn 2020 USD", logy=True, label_fmt=lambda v: f"{v:,.0f}bn")
    _draw_series(axes[2], d, series, "idx100", ten,
                 f"Same, indexed to 100 at {BASE_QUARTER.date()} (headline normalization)",
                 "index", hline=100.0, label_fmt=lambda v: f"{v:,.0f}")
    # (2026-08-10) GLOBAL denominator = MAIN, matching the regression's dw.
    _draw_series(axes[3], d, series, "share_of_book_global", ten,
                 "Share of the US investors' GLOBAL (full FactSet-identifiable "
                 "equity) book held in the bucket",
                 "share of global book", pct=True,
                 label_fmt=lambda v: f"{100*v:,.2f}%")
    _counts_panel(axes[4], d, series,
                  "Number of countries in each bucket", ylabel="countries",
                  col="n_countries")

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
        "READ THE PRE-2017 BOTTOM-QUARTILE LINE WITH CARE: the pre-2017 arms are far "
        "more sensitive to a single country than the post-2017 ones"
        + (" (under M1: 2.5 countries change bucket per quarter through 2016 and 2.3 "
           "afterward, so turnover itself is flat; but the bottom quartile holds only "
           "6-7 countries, so when Spain crosses the 25th-percentile cut in 2015Q2 it "
           "alone moves that arm from 119 to 687 index points)" if meas == "M1" else "")
        + ", so single large countries crossing a percentile cut can move an arm "
        "sharply. The quarter-by-quarter composition (per measure) is in "
        "fig_A_country_bucket_membership.csv. "
        "The share panel uses the GLOBAL (full FactSet-identifiable equity) "
        "denominator, the same denominator as the main regression; the EU-book "
        "version of the series (share_of_book_eu in fig_A_country_buckets.csv) is "
        "the within-Europe reallocation diagnostic and is not plotted. "
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

    _set_xstart(d)
    ten = _load_tension()
    fig, axes = _new_figure(counts_panel=True)
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
    _counts_panel(axes[4], d, series,
                  "Number of firms in each bucket (log scale)",
                  col="n_firms_cohort")

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
        f"with the rolling t-1 rule the positive arm grows from {_n_roll('2012-03-31')} firms in 2012Q1 "
        f"to {_n_roll('2023-12-31')} in 2023Q4 because Revere keeps adding relationships, so part of the "
        "rise in its share of the book is RECLASSIFICATION, not reallocation. "
        "fig_A_firm_zero_vs_positive_chained removes reclassification by chaining "
        "growth over matched same-arm firms, no freeze date involved.", ten)

    # (2026-08-21) The frozen-2017Q4 cohort variant is retired as the shown
    # composition control: its residual gap was a function of the freeze date
    # (-19.5 to +18.4 index points across the 52 candidate quarters). The chained
    # matched-sample variant needs no date. The zero_vs_positive_fixed split
    # remains in fig_A_firm_buckets.csv for provenance.
    if F_A_CHAIN.is_file():
        written += figure_a_chained(ten)
    else:
        raise FileNotFoundError(
            f"missing {F_A_CHAIN} — run build_fig_a_chained.py first")

    # (2026-08-21) Back to the low-versus-high contrast Stefano asked for in his
    # 2026-08-04 email ("low versus high quartile firms/countries by china-link
    # share"). Drawing all four quartiles was tried and abandoned: four sequential
    # blues on a log axis are not separable by eye, and the middle arms buried the
    # comparison the advisor actually wanted. The monotonicity question the middle
    # arms were meant to answer is settled numerically in the caption instead,
    # where all four end-of-sample index values are reported. Q2 and Q3 stay in
    # fig_A_firm_buckets.csv.
    quart_series = [("zero_tie", "zero ties (reference)", C_ZERO, 1.4),
                    ("Q1_low", "Q1, lowest CN", C_LOW, 2.0),
                    ("Q4_high", "Q4, highest CN", C_HIGH, 2.0)]
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
