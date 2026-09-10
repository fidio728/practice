"""daily_strict_curve_and_regression_panel.py — wrap up the descriptive
side of the daily transition pipeline:

  1. Strict-version monthly adoption: cumulative curves for
     search / search_strict / control + the search_strict − control gap,
     with AI-bot launch dates annotated. (Strict = search transition
     observed AND first_search_block_ts < first_control_block_ts, i.e.,
     AI-search blocked before traditional-search baseline.)

  2. Regression-ready gvkey × calendar-month panel:
       gvkey
       month_yyyymm
       first_search_block_month
       first_search_block_strict_month
       first_control_block_month
       interval_days_search
       timing_precision_search          one of {le7, le14, le30, wide, ""}
       event_time_search                calendar - first_search_block (months)
       event_time_search_strict         calendar - first_search_block_strict
       event_time_control               calendar - first_control_block
       post_search                      1 if event_time_search >= 0 else 0
       post_search_strict               same for strict
       post_control                     same for control (placebo)
     One row per (gvkey, month). Months span the same window as the
     adoption-curve file (= union of months observed for any first_*_block_ts).

Inputs (E:/bot_paper_data):
  gvkey_first_block_dates.csv          (one row per gvkey)
  gvkey_adoption_curve_monthly.csv     (defines month range and at-risk N)

Outputs (E:/bot_paper_data):
  adoption_curves_strict.png
  adoption_gap_strict_minus_control.png
  gvkey_adoption_curve_monthly_strict.csv     (parallel to the non-strict
                                               adoption curve, with strict)
  gvkey_month_panel_for_regression.csv        (the main deliverable)
"""

import csv
import datetime
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("E:/bot_paper_data")
GVKEY_CSV = DATA_DIR / "gvkey_first_block_dates.csv"
ADOPT_CSV = DATA_DIR / "gvkey_adoption_curve_monthly.csv"

PNG_CURVES = DATA_DIR / "adoption_curves_strict.png"
PNG_GAP    = DATA_DIR / "adoption_gap_strict_minus_control.png"
STRICT_ADOPT_CSV = DATA_DIR / "gvkey_adoption_curve_monthly_strict.csv"
PANEL_OUT  = DATA_DIR / "gvkey_month_panel_for_regression.csv"

KEY_DATES = [
    ("2022-11-30", "ChatGPT public release"),
    ("2023-08-07", "GPTBot UA published"),
    ("2023-09-28", "Google-Extended introduced"),
    ("2024-06-10", "Applebot-Extended announced"),
]


def parse_int(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


def yyyymm_to_date(ym):
    return datetime.date(int(ym[:4]), int(ym[4:6]), 1)


def add_month(ym, delta):
    y, m = int(ym[:4]), int(ym[4:6])
    total = y * 12 + (m - 1) + delta
    ny, nm = total // 12, total % 12 + 1
    return f"{ny:04d}{nm:02d}"


def months_diff(later_ym, earlier_ym):
    """later - earlier, in months. Both YYYYMM strings."""
    ly, lm = int(later_ym[:4]), int(later_ym[4:6])
    ey, em = int(earlier_ym[:4]), int(earlier_ym[4:6])
    return (ly - ey) * 12 + (lm - em)


def timing_precision(ival):
    if ival is None:
        return ""
    if ival <= 7:
        return "le7"
    if ival <= 14:
        return "le14"
    if ival <= 30:
        return "le30"
    return "wide"


def is_strict(s_ts, c_ts):
    """search transition observed AND search strictly precedes control
    (or control never observed)."""
    if not s_ts:
        return False
    if c_ts and s_ts >= c_ts:
        return False
    return True


def main():
    # ---- Load adoption curve to fix month range and at-risk N ----
    adopt_rows = list(csv.DictReader(open(ADOPT_CSV, encoding="utf-8")))
    months_seq = [r["month_yyyymm"] for r in adopt_rows]
    n_at_risk = int(adopt_rows[0]["n_at_risk_gvkeys"])
    print(f"month range: {months_seq[0]} to {months_seq[-1]} "
          f"({len(months_seq)} months); at-risk gvkeys = {n_at_risk:,}")

    # Pre-load search/control cum series for plotting
    search_cum   = [int(r["search_cum_count"])   for r in adopt_rows]
    control_cum  = [int(r["control_cum_count"])  for r in adopt_rows]
    search_share  = [float(r["search_cum_share"])  * 100 for r in adopt_rows]
    control_share = [float(r["control_cum_share"]) * 100 for r in adopt_rows]

    # ---- Load gvkey first-block dates ----
    print(f"\nReading {GVKEY_CSV}")
    gv_rows = list(csv.DictReader(open(GVKEY_CSV, encoding="utf-8")))
    print(f"  {len(gv_rows):,} gvkeys")

    # Build strict adoption curves (search_strict by month)
    strict_new_by_month = defaultdict(int)
    n_strict = 0
    for r in gv_rows:
        s_ts = r["first_search_block_ts"]
        c_ts = r["first_control_block_ts"]
        if is_strict(s_ts, c_ts):
            n_strict += 1
            strict_new_by_month[r["first_search_block_month"]] += 1
    print(f"  strict gvkeys: {n_strict:,}")

    strict_cum = []
    cum = 0
    for ym in months_seq:
        cum += strict_new_by_month.get(ym, 0)
        strict_cum.append(cum)
    strict_share = [c / max(1, n_at_risk) * 100 for c in strict_cum]

    # ---- Strict adoption curve PNG ----
    dates = [yyyymm_to_date(m) for m in months_seq]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(dates, search_share,  label="search (broad)",  linewidth=2.0, color="#cb1d3f")
    ax.plot(dates, strict_share,  label="search_strict (search before control)",
            linewidth=2.2, color="#d62728", linestyle="-.")
    ax.plot(dates, control_share, label="control (traditional search)",
            linewidth=1.7, color="#7f7f7f", linestyle="--")
    ymax = max(max(search_share), max(strict_share), max(control_share)) * 1.15
    ax.set_ylim(0, ymax)
    for ds, lbl in KEY_DATES:
        d = datetime.date.fromisoformat(ds)
        if dates[0] <= d <= dates[-1]:
            ax.axvline(d, color="gray", linestyle=":", alpha=0.6, linewidth=0.9)
            ax.text(d, ymax * 0.985, lbl, rotation=90, fontsize=8,
                    verticalalignment="top", horizontalalignment="right",
                    color="dimgray")
    ax.set_xlabel("month")
    ax.set_ylabel("cumulative adoption (% of at-risk gvkeys)")
    ax.set_title(f"AI-search blocking: broad vs strict (search before control), "
                 f"gvkey-level (N={n_at_risk:,})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(PNG_CURVES, dpi=130)
    plt.close(fig)
    print(f"\n  wrote {PNG_CURVES}")

    # ---- Strict gap PNG ----
    gap_share = [s - c for s, c in zip(strict_share, control_share)]
    gap_count = [s - c for s, c in zip(strict_cum, control_cum)]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, gap_share, color="#d62728", linewidth=2.2,
            label="search_strict − control (pp)")
    ax.fill_between(dates, gap_share, 0,
                    where=[g >= 0 for g in gap_share],
                    color="#d62728", alpha=0.13, interpolate=True)
    ax.fill_between(dates, gap_share, 0,
                    where=[g < 0 for g in gap_share],
                    color="#7f7f7f", alpha=0.13, interpolate=True)
    ax.axhline(0, color="black", linewidth=0.6)
    ymax = max(max(gap_share) + 0.5, 1.0)
    ymin = min(min(gap_share) - 0.5, -0.5)
    ax.set_ylim(ymin, ymax)
    for ds, lbl in KEY_DATES:
        d = datetime.date.fromisoformat(ds)
        if dates[0] <= d <= dates[-1]:
            ax.axvline(d, color="gray", linestyle=":", alpha=0.6, linewidth=0.9)
            ax.text(d, ymax * 0.97, lbl, rotation=90, fontsize=8,
                    verticalalignment="top", horizontalalignment="right",
                    color="dimgray")
    ax.text(dates[-1], gap_share[-1],
            f"  {gap_share[-1]:+.2f} pp ({gap_count[-1]:+,} firms)",
            fontsize=9, color="#d62728", verticalalignment="center")
    ax.set_xlabel("month")
    ax.set_ylabel("cumulative adoption gap (percentage points)")
    ax.set_title("Strict AI-specific adoption premium: search_strict − control "
                 "(after dropping bundled-block firms)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(PNG_GAP, dpi=130)
    plt.close(fig)
    print(f"  wrote {PNG_GAP}")

    # ---- Strict adoption curve CSV (parallel to the non-strict one) ----
    print(f"\nWriting {STRICT_ADOPT_CSV}")
    with open(STRICT_ADOPT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["month_yyyymm", "n_at_risk_gvkeys",
                    "search_strict_new_count",
                    "search_strict_cum_count",
                    "search_strict_cum_share"])
        cum = 0
        for ym in months_seq:
            new = strict_new_by_month.get(ym, 0)
            cum += new
            w.writerow([ym, n_at_risk, new, cum,
                        round(cum / max(1, n_at_risk), 6)])

    # ---- Regression-ready (gvkey × month) panel ----
    print(f"\nBuilding regression panel: {len(gv_rows):,} gvkeys × "
          f"{len(months_seq)} months")

    # Pre-compute per-gvkey static fields once
    static_by_gv = {}
    for r in gv_rows:
        gv = r["gvkey"]
        s_ts = r["first_search_block_ts"]
        c_ts = r["first_control_block_ts"]
        s_month = r["first_search_block_month"]
        c_month = r["first_control_block_month"]
        s_strict_month = s_month if is_strict(s_ts, c_ts) else ""
        ival_search = parse_int(r.get("interval_days_search", ""))
        static_by_gv[gv] = {
            "first_search_block_month":        s_month,
            "first_search_block_strict_month": s_strict_month,
            "first_control_block_month":       c_month,
            "interval_days_search":            "" if ival_search is None else ival_search,
            "timing_precision_search":         timing_precision(ival_search),
        }

    # Use a stable gvkey ordering
    all_gvkeys = sorted(static_by_gv.keys())

    fields = [
        "gvkey", "month_yyyymm",
        "first_search_block_month",
        "first_search_block_strict_month",
        "first_control_block_month",
        "interval_days_search",
        "timing_precision_search",
        "event_time_search",
        "event_time_search_strict",
        "event_time_control",
        "post_search",
        "post_search_strict",
        "post_control",
    ]

    n_rows_written = 0
    with open(PANEL_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for gv in all_gvkeys:
            st = static_by_gv[gv]
            s_m  = st["first_search_block_month"]
            ss_m = st["first_search_block_strict_month"]
            c_m  = st["first_control_block_month"]
            for ym in months_seq:
                evt_s  = months_diff(ym, s_m)  if s_m  else ""
                evt_ss = months_diff(ym, ss_m) if ss_m else ""
                evt_c  = months_diff(ym, c_m)  if c_m  else ""
                row = {
                    "gvkey": gv,
                    "month_yyyymm": ym,
                    "first_search_block_month":        s_m,
                    "first_search_block_strict_month": ss_m,
                    "first_control_block_month":       c_m,
                    "interval_days_search":   st["interval_days_search"],
                    "timing_precision_search":st["timing_precision_search"],
                    "event_time_search":        evt_s,
                    "event_time_search_strict": evt_ss,
                    "event_time_control":       evt_c,
                    "post_search":        1 if (s_m  and evt_s  != "" and evt_s  >= 0) else 0,
                    "post_search_strict": 1 if (ss_m and evt_ss != "" and evt_ss >= 0) else 0,
                    "post_control":       1 if (c_m  and evt_c  != "" and evt_c  >= 0) else 0,
                }
                writer.writerow(row)
                n_rows_written += 1
    print(f"  rows written: {n_rows_written:,}")
    print(f"  wrote {PANEL_OUT}")

    # Final cumulative numbers for sanity
    print(f"\nFinal cumulative shares (% of {n_at_risk:,} at-risk gvkeys):")
    print(f"  search broad:        {search_share[-1]:6.2f}%  ({search_cum[-1]:,})")
    print(f"  search_strict:       {strict_share[-1]:6.2f}%  ({strict_cum[-1]:,})")
    print(f"  control:             {control_share[-1]:6.2f}%  ({control_cum[-1]:,})")
    print(f"  search_strict − control: {gap_share[-1]:+6.2f} pp  "
          f"({gap_count[-1]:+,} firms)")


if __name__ == "__main__":
    main()
