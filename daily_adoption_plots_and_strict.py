"""daily_adoption_plots_and_strict.py — produce three deliverables on
top of gvkey_first_block_dates.csv + gvkey_adoption_curve_monthly.csv:

  1. PNG: cumulative adoption curves for search / training / assistant /
     control / any_ai with key AI-bot-launch dates annotated.
  2. PNG: search minus control cumulative gap (the AI-specific premium).
     Negative means control adopters lead search adopters; positive means
     firms are blocking AI-search bots faster than traditional crawlers.
  3. CSV: strict-timing sample-size table —
       cutoff in {all, ≤30d, ≤14d, ≤7d}
       × {first_search_block, first_search_block_strict}
     where _strict = search transition observed AND control either
     unobserved or strictly LATER than search (i.e., the firm blocked
     AI-search before the traditional-search baseline). This isolates
     AI-targeted adoption from blanket "block-all-bots" behavior.

Inputs (E:/bot_paper_data):
  gvkey_first_block_dates.csv         (one row per gvkey)
  gvkey_adoption_curve_monthly.csv    (monthly cumulative)

Outputs (E:/bot_paper_data):
  adoption_curves_by_group.png
  adoption_gap_search_minus_control.png
  adoption_strict_timing_table.csv
  adoption_strict_timing_table.txt    (same content, human-readable)
"""

import csv
import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("E:/bot_paper_data")
GVKEY_CSV = DATA_DIR / "gvkey_first_block_dates.csv"
ADOPT_CSV = DATA_DIR / "gvkey_adoption_curve_monthly.csv"

PNG_CURVES = DATA_DIR / "adoption_curves_by_group.png"
PNG_GAP    = DATA_DIR / "adoption_gap_search_minus_control.png"
CSV_TABLE  = DATA_DIR / "adoption_strict_timing_table.csv"
TXT_TABLE  = DATA_DIR / "adoption_strict_timing_table.txt"

KEY_DATES = [
    ("2022-11-30", "ChatGPT public release"),
    ("2023-08-07", "GPTBot UA published (OpenAI)"),
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


def main():
    # ---- 1. Adoption curves PNG ----
    print(f"Reading {ADOPT_CSV}")
    rows = list(csv.DictReader(open(ADOPT_CSV, encoding="utf-8")))
    months = [r["month_yyyymm"] for r in rows]
    dates = [yyyymm_to_date(m) for m in months]
    n_at_risk = int(rows[0]["n_at_risk_gvkeys"])

    def shares(group):
        return [float(r[f"{group}_cum_share"]) * 100 for r in rows]

    def counts(group):
        return [int(r[f"{group}_cum_count"]) for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(dates, shares("search"),    label="search (AI search bots)",    linewidth=2.0, color="#cb1d3f")
    ax.plot(dates, shares("training"),  label="training (AI scrapers)",     linewidth=1.7, color="#1f77b4")
    ax.plot(dates, shares("assistant"), label="assistant (AI user agents)", linewidth=1.5, color="#2ca02c")
    ax.plot(dates, shares("control"),   label="control (traditional search)", linewidth=1.7, color="#7f7f7f", linestyle="--")
    ax.plot(dates, shares("any_ai"),    label="any AI (search ∪ training ∪ assistant)", linewidth=1.4, color="black", linestyle=":")

    ymax = max(max(shares("search")), max(shares("any_ai"))) * 1.15
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
    ax.set_title(f"Cumulative bot-block adoption by group, gvkey-level "
                 f"(N at-risk = {n_at_risk:,})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(PNG_CURVES, dpi=130)
    plt.close(fig)
    print(f"  wrote {PNG_CURVES}")

    # ---- 2. search - control gap PNG ----
    s_share = shares("search")
    c_share = shares("control")
    gap_share = [s - c for s, c in zip(s_share, c_share)]
    s_count = counts("search")
    c_count = counts("control")
    gap_count = [s - c for s, c in zip(s_count, c_count)]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, gap_share, color="#cb1d3f", linewidth=2.2, label="search − control (pp)")
    ax.fill_between(dates, gap_share, 0, color="#cb1d3f", alpha=0.13)
    ax.axhline(0, color="black", linewidth=0.6)

    ymax = max(max(gap_share) + 0.5, 1.0)
    ymin = min(min(gap_share) - 0.3, -0.5)
    ax.set_ylim(ymin, ymax)

    for ds, lbl in KEY_DATES:
        d = datetime.date.fromisoformat(ds)
        if dates[0] <= d <= dates[-1]:
            ax.axvline(d, color="gray", linestyle=":", alpha=0.6, linewidth=0.9)
            ax.text(d, ymax * 0.97, lbl, rotation=90, fontsize=8,
                    verticalalignment="top", horizontalalignment="right",
                    color="dimgray")

    ax.set_xlabel("month")
    ax.set_ylabel("cumulative adoption gap (percentage points)")
    ax.set_title("AI-specific adoption premium: search-group minus control-group "
                 "cumulative blocking")
    ax.grid(True, alpha=0.3)
    # also annotate final-month numbers
    final_gap_pp = gap_share[-1]
    final_search_count = s_count[-1]
    final_control_count = c_count[-1]
    ax.text(dates[-1], gap_share[-1],
            f"  +{final_gap_pp:.2f} pp  ({final_search_count - final_control_count:+,} firms)",
            fontsize=9, color="#cb1d3f", verticalalignment="center")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(PNG_GAP, dpi=130)
    plt.close(fig)
    print(f"  wrote {PNG_GAP}")

    # ---- 3. Strict timing adoption table ----
    print(f"\nReading {GVKEY_CSV}")
    gv_rows = list(csv.DictReader(open(GVKEY_CSV, encoding="utf-8")))
    print(f"  {len(gv_rows):,} gvkey rows")

    def is_main(r):
        return bool(r.get("first_search_block_ts", ""))

    def is_strict(r):
        s_ts = r.get("first_search_block_ts", "")
        if not s_ts:
            return False
        c_ts = r.get("first_control_block_ts", "")
        # strict = search came strictly before control (or control never)
        if c_ts and s_ts >= c_ts:
            return False
        return True

    cuts = [None, 30, 14, 7]
    table_rows = []
    for cut in cuts:
        n_main = 0
        n_strict = 0
        for r in gv_rows:
            ival = parse_int(r.get("interval_days_search", ""))
            if is_main(r):
                if cut is None or (ival is not None and ival <= cut):
                    n_main += 1
            if is_strict(r):
                if cut is None or (ival is not None and ival <= cut):
                    n_strict += 1
        table_rows.append({
            "cutoff": "all" if cut is None else f"<= {cut}d",
            "first_search_block": n_main,
            "first_search_block_strict": n_strict,
            "strict_share": round(n_strict / max(1, n_main), 4),
        })

    # CSV
    with open(CSV_TABLE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        for r in table_rows:
            writer.writerow(r)
    print(f"  wrote {CSV_TABLE}")

    # text
    lines = []
    lines.append("Strict timing adoption table (gvkey-level, search group)")
    lines.append("")
    lines.append(f"{'cutoff':>10}  {'first_search_block':>20}  "
                 f"{'first_search_block_strict':>30}  {'strict / main':>14}")
    lines.append("-" * 80)
    for r in table_rows:
        lines.append(f"{r['cutoff']:>10}  {r['first_search_block']:>20,}  "
                     f"{r['first_search_block_strict']:>30,}  "
                     f"{r['strict_share']:>14.4f}")
    lines.append("")
    lines.append("Definitions:")
    lines.append("  first_search_block         = gvkeys with any first_search_block_ts observed")
    lines.append("  first_search_block_strict  = same, AND first_search_block_ts < first_control_block_ts")
    lines.append("                                (or control never blocked) — i.e., AI-search blocked")
    lines.append("                                BEFORE traditional-search baseline.")
    lines.append("  cutoff applies to interval_days_search (the timing-precision of the search event).")

    text = "\n".join(lines) + "\n"
    TXT_TABLE.write_text(text, encoding="utf-8")
    print(f"  wrote {TXT_TABLE}")
    print("\n" + text)


if __name__ == "__main__":
    main()
