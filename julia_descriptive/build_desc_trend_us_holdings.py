# build_desc_trend_us_holdings.py
# ---------------------------------------------------------------------------
# Revised Figure 2 for advisor: year-over-year growth in the real (2020-dollar)
# value of US institutional holdings of European listed firms, by country.
#
# Primary source: output/ownership_ict.parquet, whose I_ict column implements
# the full filter/aggregation rule of 04_us_ownership_european.jl Section 1:
#     I_ict = SUM(adj_mv) GROUP BY (sec_entity_id, sec_country,
#                                   investor_country, report_date)
#     WHERE sec_entity_id IS NOT NULL AND investor_country IS NOT NULL
#       AND issue_type IN ('EQ','AD')
# adj_mv is FactSet LionShares ADJ_MV (fund-level, USD), deduplicated on the
# (fund_id, fsym_id, report_date) key upstream (03 Phase C check = 0 dups;
# 04 Section 0 re-checks and hard-errors on dups).
#
# Cross-check: the same aggregation recomputed directly from
# output/holdings_eom.parquet (186.8M rows) via DuckDB streaming aggregation.
# The script HARD-FAILS if the two disagree beyond float tolerance, so the
# numbers in the CSV are never single-sourced.
#
# NOTE (deliberate deviation from 04 Section 5 time series): we do NOT apply
# the `ownership_share BETWEEN 0 AND 1` filter used by
# 04_us_ownership_eu_timeseries.csv. That filter needs market_cap and drops
# legitimate dollar cells where market_cap is missing/odd, so it understates
# the pure dollar total. See NOTES_us_holdings.md.
#
# Inflation adjustment: CPI-U all items, not seasonally adjusted (CPIAUCNS),
# quarter-end month, normalized by the 2020 annual-average CPI.
#
# Outputs:
#   output/desc_trend_us_holdings_real_growth.csv
#   output/cpiaucns_monthly.csv
#   plots/fig_us_holdings_real_yoy_growth.png / .pdf
#   plots/NOTES_us_holdings.md is written separately (documentation).
#
# Rerunnable end-to-end; ~2-6 min (dominated by the holdings_eom cross-check).
# ---------------------------------------------------------------------------

from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from desc_trend_metrics import deflate_and_add_yoy, normalize_fred_cpi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "output"
ICT_PARQUET = BASE / "output" / "ownership_ict.parquet"
EOM_PARQUET = BASE / "output" / "holdings_eom.parquet"
OUT_CSV = OUT_DIR / "desc_trend_us_holdings_real_growth.csv"
CPI_CACHE = OUT_DIR / "cpiaucns_monthly.csv"
CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS"
PLOT_DIR = BASE / "plots"
PLOT_DIR.mkdir(exist_ok=True)
FIG_PNG = PLOT_DIR / "fig_us_holdings_real_yoy_growth.png"
FIG_PDF = PLOT_DIR / "fig_us_holdings_real_yoy_growth.pdf"

# EU_COUNTRIES, verbatim from 00_setup.jl (28 European countries; includes
# GB/CH/NO etc., hence "European", not "EU", in all figure text).
EU_COUNTRIES = ("GB", "DE", "FR", "NL", "CH", "IT", "ES", "SE", "DK", "NO",
                "FI", "BE", "AT", "IE", "LU", "PT", "PL", "CZ", "HU", "GR",
                "RO", "SK", "SI", "BG", "HR", "EE", "LV", "LT")
EU_SQL_TUPLE = "(" + ",".join(f"'{c}'" for c in EU_COUNTRIES) + ")"

COUNTRY_NAMES = {
    "GB": "United Kingdom", "DE": "Germany", "FR": "France",
    "NL": "Netherlands", "CH": "Switzerland", "IT": "Italy", "ES": "Spain",
    "SE": "Sweden", "DK": "Denmark", "NO": "Norway", "FI": "Finland",
    "BE": "Belgium", "AT": "Austria", "IE": "Ireland", "LU": "Luxembourg",
    "PT": "Portugal", "PL": "Poland", "CZ": "Czechia", "HU": "Hungary",
    "GR": "Greece", "RO": "Romania", "SK": "Slovakia", "SI": "Slovenia",
    "BG": "Bulgaria", "HR": "Croatia", "EE": "Estonia", "LV": "Latvia",
    "LT": "Lithuania",
}

MIN_QUARTERS_FOR_PANEL = 8  # countries below this fold into 'Other' (figure only)

# Plot growth from here on (CSV keeps the full 1999- period). Earlier YoY
# values are dominated by LionShares' coverage ramp-up (three-digit-thousand
# percent artifacts), not by investment.
PLOT_START = pd.Timestamp("2005-03-31")

# Uniform y-limits across panels: keeps countries comparable and pushes the
# residual coverage-artifact spikes (small CEE countries) off-scale instead of
# letting them squash their own panel.
YLIM = (-100.0, 200.0)

# Style spec (fixed by caller)
C_LINE = "#2a78d6"
C_BG = "#fcfcfb"
C_TEXT = "#0b0b0b"
C_TEXT2 = "#52514e"
C_GRID = "#e8e8e6"


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")

    ict = str(ICT_PARQUET).replace("\\", "/")
    eom = str(EOM_PARQUET).replace("\\", "/")

    # -----------------------------------------------------------------------
    # 1) Quarter-end calendar: last available EOM report_date per calendar
    #    quarter (in this panel every quarter has exactly one date, the
    #    calendar month-end; the rule is generic anyway).
    # -----------------------------------------------------------------------
    qmap = con.sql(f"""
        SELECT date_trunc('quarter', report_date) AS q,
               MAX(report_date) AS quarter_end
        FROM (SELECT DISTINCT report_date FROM read_parquet('{ict}'))
        GROUP BY 1
    """).df()
    n_quarters = len(qmap)
    qend_list = ",".join(f"DATE '{d}'" for d in sorted(qmap["quarter_end"].astype(str)))
    print(f"[1] {n_quarters} quarter-end dates "
          f"({qmap['quarter_end'].min()} .. {qmap['quarter_end'].max()})")

    # -----------------------------------------------------------------------
    # 2) PRIMARY aggregation from ownership_ict.parquet
    # -----------------------------------------------------------------------
    primary = con.sql(f"""
        SELECT sec_country,
               report_date AS quarter_end,
               SUM(I_ict)                    AS usd_value,
               COUNT(DISTINCT sec_entity_id) AS n_firms_held
        FROM read_parquet('{ict}')
        WHERE investor_country = 'US'
          AND sec_country IN {EU_SQL_TUPLE}
          AND report_date IN ({qend_list})
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()
    print(f"[2] primary (ownership_ict): {len(primary)} country-quarter rows, "
          f"{primary['sec_country'].nunique()} countries")

    # -----------------------------------------------------------------------
    # 3) CROSS-CHECK: recompute directly from holdings_eom.parquet with the
    #    exact 04 Section 1 filters. Must match to float tolerance.
    # -----------------------------------------------------------------------
    check = con.sql(f"""
        SELECT sec_country,
               report_date AS quarter_end,
               SUM(adj_mv)                   AS usd_value,
               COUNT(DISTINCT sec_entity_id) AS n_firms_held
        FROM read_parquet('{eom}')
        WHERE investor_country = 'US'
          AND sec_country IN {EU_SQL_TUPLE}
          AND sec_entity_id IS NOT NULL
          AND issue_type IN ('EQ', 'AD')
          AND report_date IN ({qend_list})
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()

    m = primary.merge(check, on=["sec_country", "quarter_end"],
                      how="outer", suffixes=("_p", "_c"), indicator=True)
    if (m["_merge"] != "both").any():
        bad = m.loc[m["_merge"] != "both"]
        raise RuntimeError(f"cross-check cell mismatch:\n{bad.head(20)}")
    rel = (m["usd_value_p"] - m["usd_value_c"]).abs() / m["usd_value_c"].clip(lower=1.0)
    dn = (m["n_firms_held_p"] - m["n_firms_held_c"]).abs()
    print(f"[3] cross-check vs holdings_eom: max rel diff usd={rel.max():.3e}, "
          f"max abs diff n_firms={dn.max()}")
    if rel.max() > 1e-9 or dn.max() > 0:
        raise RuntimeError(
            "ownership_ict aggregation does not reproduce holdings_eom "
            f"direct aggregation (max rel usd diff {rel.max():.3e}, "
            f"max n_firms diff {dn.max()}). Refusing to write outputs.")

    # -----------------------------------------------------------------------
    # 4) Total Europe (distinct-firm count computed properly, not summed).
    # -----------------------------------------------------------------------
    total_eu = con.sql(f"""
        SELECT report_date AS quarter_end,
               SUM(I_ict)                    AS usd_value,
               COUNT(DISTINCT sec_entity_id) AS n_firms_held
        FROM read_parquet('{ict}')
        WHERE investor_country = 'US'
          AND sec_country IN {EU_SQL_TUPLE}
          AND report_date IN ({qend_list})
        GROUP BY 1 ORDER BY 1
    """).df()
    total_eu["quarter_end"] = pd.to_datetime(total_eu["quarter_end"])

    # -----------------------------------------------------------------------
    # 5) CPI-U deflation and exact same-quarter year-over-year growth.
    # -----------------------------------------------------------------------
    if CPI_CACHE.is_file():
        cpi_raw = pd.read_csv(CPI_CACHE)
        print(f"[5] CPI cache: {CPI_CACHE}")
    else:
        print(f"[5] downloading CPIAUCNS from {CPI_URL}")
        cpi_raw = pd.read_csv(CPI_URL)
    cpi = normalize_fred_cpi(cpi_raw)
    CPI_CACHE.parent.mkdir(exist_ok=True)
    cpi.to_csv(CPI_CACHE, index=False)

    country_series = primary.copy()
    country_series["quarter_end"] = pd.to_datetime(country_series["quarter_end"])
    total_series = total_eu.copy()
    total_series["sec_country"] = "TOTAL_EUROPE"
    combined = pd.concat([country_series, total_series], ignore_index=True, sort=False)
    metrics = deflate_and_add_yoy(combined, cpi, group_col="sec_country")
    metrics["value_currency"] = "USD"
    metrics["real_base_year"] = 2020
    metrics.to_csv(OUT_CSV, index=False)
    print(f"    2020 annual-average CPI={metrics['cpi_2020_annual_avg'].iloc[0]:.3f}")
    print(f"    wrote {OUT_CSV} ({len(metrics)} rows)")

    # -----------------------------------------------------------------------
    # 6) Figure prep: country ordering and panel construction.
    # -----------------------------------------------------------------------
    df = metrics.loc[metrics["sec_country"].ne("TOTAL_EUROPE")].copy()

    nq_by_ctry = df.groupby("sec_country")["quarter_end"].nunique()
    small = sorted(nq_by_ctry[nq_by_ctry < MIN_QUARTERS_FOR_PANEL].index)
    big = list(nq_by_ctry.index)

    last_q = df["quarter_end"].max()
    end_vals = (df[df["quarter_end"] == last_q]
                .set_index("sec_country")["usd_value"])
    # order by end-of-sample value desc; countries absent at the last quarter
    # rank below all present ones, ordered by their own last observed value
    own_last = (df.sort_values("quarter_end").groupby("sec_country").tail(1)
                .set_index("sec_country")["usd_value"])
    order_key = {c: (0, -end_vals[c]) if c in end_vals.index else (1, -own_last[c])
                 for c in big}
    big_sorted = sorted(big, key=lambda c: order_key[c])

    panels = [("Total Europe",
               metrics.loc[metrics["sec_country"].eq("TOTAL_EUROPE"),
                           ["quarter_end", "real_yoy_pct"]])]
    for c in big_sorted:
        panels.append((COUNTRY_NAMES.get(c, c),
                       df.loc[df["sec_country"] == c,
                              ["quarter_end", "real_yoy_pct"]]))
    if small:
        print(f"    sparse countries retained as separate panels: {','.join(small)}")
    print(f"[6] panels: {len(panels)} (Total Europe + {len(big_sorted)} countries)")

    # -----------------------------------------------------------------------
    # 7) Draw
    # -----------------------------------------------------------------------
    ncols = 5
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.1 * nrows + 1.6),
                             sharex=True)
    fig.patch.set_facecolor(C_BG)
    axes = np.atleast_2d(axes)

    full_index = pd.DatetimeIndex(sorted(pd.to_datetime(qmap["quarter_end"])))
    full_index = full_index[full_index >= PLOT_START]
    xmin = full_index.min()
    xmax = full_index.max()

    for i, ax in enumerate(axes.flat):
        if i >= len(panels):
            ax.axis("off")
            continue
        name, pdat = panels[i]
        # Reindex on the full quarter calendar so missing quarters break the
        # line instead of being visually interpolated across gaps.
        s = (pdat.sort_values("quarter_end")
             .set_index("quarter_end")["real_yoy_pct"].reindex(full_index))
        ax.set_facecolor(C_BG)
        ax.axhline(0, color=C_TEXT2, lw=0.7, alpha=0.7)
        ax.plot(s.index, s.values, color=C_LINE, lw=1.8,
                solid_capstyle="round")
        # Isolated observations (both neighbors missing) would be invisible
        # on a line plot; mark them with small dots.
        obs = s.notna().to_numpy()
        prev_na = np.r_[True, ~obs[:-1]]
        next_na = np.r_[~obs[1:], True]
        iso = obs & prev_na & next_na
        if iso.any():
            ax.plot(s.index[iso], s.values[iso], linestyle="none", marker="o",
                    markersize=2.4, color=C_LINE)
        ax.set_title(name, fontsize=10, color=C_TEXT, loc="left", pad=4,
                     fontweight="bold" if name == "Total Europe" else "normal")
        ax.grid(axis="y", color=C_GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(C_GRID)
        ax.tick_params(colors=C_TEXT2, labelsize=8, length=3)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(*YLIM)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        if i % ncols == 0:
            ax.set_ylabel("YoY", fontsize=8, color=C_TEXT2)

    fig.suptitle("Real growth in US institutional holdings of European firms, "
                 "by country", x=0.01, y=0.995, ha="left", fontsize=14,
                 color=C_TEXT, fontweight="bold")
    fig.text(0.01, 0.958,
             "Year-over-year percent change in CPI-U-deflated market value "
             "(2020 USD). Shown from 2005; earlier quarters are dominated by "
             "LionShares coverage ramp-up.\n"
             "Common y-axis clipped at +200%: reporting-coverage expansions "
             "(2014, 2019, 2020; late-starting CEE countries) run off-scale "
             "rather than rescaling their panels.",
             ha="left", fontsize=9, color=C_TEXT2)
    fig.tight_layout(rect=(0, 0.0, 1, 0.955))
    fig.savefig(FIG_PNG, dpi=200, facecolor=C_BG)
    fig.savefig(FIG_PDF, facecolor=C_BG)
    plt.close(fig)
    print(f"[7] wrote {FIG_PNG}\n    wrote {FIG_PDF}")

    # -----------------------------------------------------------------------
    # 8) Headline numbers (from the CSV just written, real YoY percent)
    # -----------------------------------------------------------------------
    csv = pd.read_csv(OUT_CSV, parse_dates=["quarter_end"])
    csv = csv[csv["quarter_end"] >= PLOT_START]  # headline = shown period only
    eu_growth = (csv.loc[csv["sec_country"].eq("TOTAL_EUROPE")]
                 .set_index("quarter_end")["real_yoy_pct"].dropna())

    print("\n=== HEADLINE (real YoY growth) ===")
    print(f"Total Europe: latest={eu_growth.iloc[-1]:.1f}%, "
          f"median={eu_growth.median():.1f}%, "
          f"range=[{eu_growth.min():.1f}%, {eu_growth.max():.1f}%]")
    countries = csv.loc[csv["sec_country"].ne("TOTAL_EUROPE")]
    top3 = (countries[countries['quarter_end'] == countries['quarter_end'].max()]
            .nlargest(3, "usd_value")["sec_country"].tolist())
    for c in top3:
        ts = (countries[countries["sec_country"] == c]
              .set_index("quarter_end")["real_yoy_pct"].dropna().sort_index())
        print(f"{c} ({COUNTRY_NAMES.get(c, c)}): latest={ts.iloc[-1]:.1f}%, "
              f"median={ts.median():.1f}%")
    if small:
        print("Sparse countries retained individually: " + ", ".join(small))


if __name__ == "__main__":
    main()
