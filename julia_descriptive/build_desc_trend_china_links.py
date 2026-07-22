# build_desc_trend_china_links.py
# Revised Figure 1 for advisor: fraction of firms with an ACTIVE China
# supply-chain link (CUSTOMER or SUPPLIER edges in FactSet Revere), by home
# country, quarterly.
#
# Countries: US + the 28 "European" countries from 00_setup.jl EU_COUNTRIES
# (includes GB/CH/NO, i.e. "European", not strictly EU).
#
# Construction (mirrors 02_china_exposure.jl, extended to include US):
#   (1) Revere company master is time-versioned: sentinel end date 4000-01-01
#       -> NULL; dedup to one row per (company_id, start_d) with deterministic
#       tiebreaker (02 section 1c-pre).
#   (2) Edge endpoints classified AS-OF rel_start via ASOF JOIN (02 section 1d,
#       C5 fix: no look-ahead from latest-row static attributes).
#   (3) China edge = rel_type IN ('CUSTOMER','SUPPLIER') AND one endpoint in
#       C_SET (EU28+US) with the other endpoint 'CN', both as-of rel_start.
#       NOTE: deliberately NARROWER than 02's n_cn_total, which includes
#       COMPETITOR and all PARTNER-* types.
#   (4) A link is ACTIVE at quarter-end q iff rel_start <= q AND
#       (rel_end IS NULL OR rel_end >= q)  (02 section 5 convention).
#   (5) The firm's country at q is its home_region AS-OF q (time-versioned,
#       not static). Firms with NULL/other region at q are not counted at q.
#   (6) Count DISTINCT firms per (country, quarter_end).
#
# Two universes for the EU/European lines:
#   all_revere  = every Revere company (main).
#   regression  = Revere companies matched (CUSIP > ISIN > SEDOL priority, as
#                 in 06_cartesian_grid.jl) to a sec_entity_id present in
#                 merged_us_eu_zero_filled.parquet (i.e. ever held by >=1
#                 institution with EU sec_country; selection-on-outcome caveat).
# US line: all_revere only (the regression panel is EU-centric by design).
#
# Window: quarter-ends 2003-06-30 .. 2025-03-31. Revere edges start
# 2003-04-03; data vintage ends 2025-05-03, so 2025Q2 would be right-censored
# and is excluded.
#
# Denominator: all firms in the same Revere universe with a valid time-varying
# home_region in the country at the same quarter-end.
#
# Outputs:
#   output/desc_trend_china_link_fraction.csv
#   plots/fig_china_link_fraction.png / .pdf
#
# Rerun: python build_desc_trend_china_links.py   (no arguments; ~2-4 min)

from pathlib import Path
import sys
import tempfile

import duckdb
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402

from desc_trend_metrics import add_fraction, assert_nested_universe

# ------------------------------------------------------------------
# Paths (pathlib + forward slashes; base dir has spaces + accented chars)
# ------------------------------------------------------------------
BASE = Path(__file__).resolve().parent          # .../julia_descriptive
OUT_DIR = BASE / "output"
PLOT_DIR = BASE / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

REV_DIR = Path("E:/Data/Data/Factset Revere")
CO_CSV = (REV_DIR / "revere_company_wrds.csv").as_posix()
REL_CSV = (REV_DIR / "data_giorgio.csv").as_posix()

EOM_PARQUET = (OUT_DIR / "holdings_eom.parquet").as_posix()
UNIV_STATIC_PARQUET = (OUT_DIR / "eu_revere_universe.parquet").as_posix()
MERGED_PARQUET = (OUT_DIR / "merged_us_eu_zero_filled.parquet").as_posix()

CSV_OUT = OUT_DIR / "desc_trend_china_link_fraction.csv"
FIG_PNG = PLOT_DIR / "fig_china_link_fraction.png"
FIG_PDF = PLOT_DIR / "fig_china_link_fraction.pdf"

for p in (CO_CSV, REL_CSV, EOM_PARQUET, UNIV_STATIC_PARQUET, MERGED_PARQUET):
    if not Path(p).is_file():
        sys.exit(f"FATAL: required input missing: {p}")

# ------------------------------------------------------------------
# Country sets — EU_COUNTRIES copied verbatim from 00_setup.jl
# ------------------------------------------------------------------
EU_COUNTRIES = ("GB", "DE", "FR", "NL", "CH", "IT", "ES", "SE", "DK", "NO",
                "FI", "BE", "AT", "IE", "LU", "PT", "PL", "CZ", "HU", "GR",
                "RO", "SK", "SI", "BG", "HR", "EE", "LV", "LT")
C_SET = ("US",) + EU_COUNTRIES

def sql_tuple(cs):
    return "(" + ",".join(f"'{c}'" for c in cs) + ")"

EU_SQL = sql_tuple(EU_COUNTRIES)
CSET_SQL = sql_tuple(C_SET)

COUNTRY_NAMES = {
    "US": "United States", "GB": "United Kingdom", "DE": "Germany",
    "FR": "France", "NL": "Netherlands", "CH": "Switzerland", "IT": "Italy",
    "ES": "Spain", "SE": "Sweden", "DK": "Denmark", "NO": "Norway",
    "FI": "Finland", "BE": "Belgium", "AT": "Austria", "IE": "Ireland",
    "LU": "Luxembourg", "PT": "Portugal", "PL": "Poland",
    "CZ": "Czech Republic", "HU": "Hungary", "GR": "Greece", "RO": "Romania",
    "SK": "Slovakia", "SI": "Slovenia", "BG": "Bulgaria", "HR": "Croatia",
    "EE": "Estonia", "LV": "Latvia", "LT": "Lithuania",
}

PANEL_START = "2003-06-30"   # first quarter-end >= earliest rel_start (2003-04-03)
PANEL_END = "2025-03-31"     # last full quarter before data vintage end (2025-05-03)

# ------------------------------------------------------------------
# DuckDB
# ------------------------------------------------------------------
spill = Path(tempfile.gettempdir()) / "duckdb_spill_fig1"
spill.mkdir(parents=True, exist_ok=True)
con = duckdb.connect(":memory:")
con.execute("SET memory_limit='6GB'")
con.execute("SET threads=4")
con.execute(f"SET temp_directory='{spill.as_posix()}'")
con.execute("SET preserve_insertion_order=false")

print("(1) Revere company master: load + sentinel normalisation + dedup ...")
con.execute(f"""
    CREATE TABLE rev_co_dedup AS
    SELECT company_id, home_region, start_d, end_d
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY company_id, start_d
                   -- NULLS FIRST: an open row (end_=sentinel -> NULL = still valid)
                   -- must beat a same-start_d zero-length closed row (s,s), the
                   -- WRDS same-day-correction pattern. Independent audit of all 11
                   -- affected firms confirmed NULLS LAST drops live firms
                   -- (02_china_exposure.jl 1c-pre convention undercounts tail
                   -- quarters by <=0.36%; 02's own static dedup uses NULLS FIRST).
                   ORDER BY end_d DESC NULLS FIRST, company_id ASC
               ) AS rn
        FROM (
            SELECT company_id,
                   home_region,
                   CAST(start_ AS DATE) AS start_d,
                   CASE WHEN CAST(end_ AS DATE) >= DATE '4000-01-01' THEN NULL
                        ELSE CAST(end_ AS DATE) END AS end_d
            FROM read_csv_auto('{CO_CSV}', sample_size=-1)
        )
    )
    WHERE rn = 1
""")
n = con.execute("SELECT COUNT(*), COUNT(DISTINCT company_id) FROM rev_co_dedup").fetchone()
print(f"    rev_co_dedup rows={n[0]:,}  companies={n[1]:,}")

print("(2) Quarter calendar ...")
con.execute(f"""
    CREATE TABLE quarters AS
    SELECT LAST_DAY(MAKE_DATE(y, m, 1)) AS qend
    FROM range(2003, 2026) t(y)
    CROSS JOIN (VALUES (3),(6),(9),(12)) AS mt(m)
    WHERE LAST_DAY(MAKE_DATE(y, m, 1))
          BETWEEN DATE '{PANEL_START}' AND DATE '{PANEL_END}'
""")
nq = con.execute("SELECT COUNT(*), MIN(qend), MAX(qend) FROM quarters").fetchone()
print(f"    {nq[0]} quarter-ends: {nq[1]} .. {nq[2]}")

print("(3) Edges: load CUSTOMER/SUPPLIER, classify endpoints AS-OF rel_start ...")
# Filter rel_type EARLY (main definition: true supply-chain edges only;
# COMPETITOR and PARTNER-* excluded — see header).
con.execute(f"""
    CREATE TABLE rel_raw AS
    SELECT source_company_id, target_company_id, rel_type,
           CAST(start_ AS DATE) AS rel_start,
           CASE WHEN CAST(end_ AS DATE) >= DATE '4000-01-01' THEN NULL
                ELSE CAST(end_ AS DATE) END AS rel_end
    FROM read_csv_auto('{REL_CSV}', sample_size=-1)
    WHERE rel_type IN ('CUSTOMER','SUPPLIER')
""")
rr = con.execute("SELECT COUNT(*), MIN(rel_start), MAX(rel_start) FROM rel_raw").fetchone()
print(f"    CUSTOMER/SUPPLIER edges={rr[0]:,}  rel_start {rr[1]} .. {rr[2]}")

# As-of rel_start region for both endpoints (02 section 1d logic: ASOF picks
# latest start_d <= rel_start; NULL-out region if picked row ended before
# rel_start, i.e. edge falls in a coverage gap).
con.execute("""
    CREATE TABLE rel_classified AS
    WITH src_asof AS (
        SELECT r.*, s.home_region AS src_region_raw, s.end_d AS src_end_d
        FROM rel_raw r
        ASOF LEFT JOIN rev_co_dedup s
          ON r.source_company_id = s.company_id AND r.rel_start >= s.start_d
    ),
    both_asof AS (
        SELECT sa.*, t.home_region AS tgt_region_raw, t.end_d AS tgt_end_d
        FROM src_asof sa
        ASOF LEFT JOIN rev_co_dedup t
          ON sa.target_company_id = t.company_id AND sa.rel_start >= t.start_d
    )
    SELECT source_company_id, target_company_id, rel_type, rel_start, rel_end,
           CASE WHEN src_end_d IS NULL OR src_end_d >= rel_start
                THEN src_region_raw ELSE NULL END AS src_region,
           CASE WHEN tgt_end_d IS NULL OR tgt_end_d >= rel_start
                THEN tgt_region_raw ELSE NULL END AS tgt_region
    FROM both_asof
""")

print("(4) China-edge view (firm side in C_SET, counterparty CN, as-of rel_start) ...")
con.execute(f"""
    CREATE TABLE cn_edge AS
    SELECT source_company_id AS firm_id, rel_start, rel_end
    FROM rel_classified
    WHERE src_region IN {CSET_SQL} AND tgt_region = 'CN'
    UNION ALL
    SELECT target_company_id AS firm_id, rel_start, rel_end
    FROM rel_classified
    WHERE src_region = 'CN' AND tgt_region IN {CSET_SQL}
""")
ce = con.execute("SELECT COUNT(*), COUNT(DISTINCT firm_id) FROM cn_edge").fetchone()
print(f"    cn_edge rows={ce[0]:,}  distinct firms={ce[1]:,}")

print("(5) As-of quarter-end home_region for ALL supply-chain-active firms ...")
# Superset probe: every firm that appears on either side of a CUSTOMER/SUPPLIER
# edge (needed both for the numerator counts and for the supply_chain
# denominator below). cn_edge firms are a subset by construction.
con.execute("""
    CREATE TABLE firm_region_q AS
    WITH firms AS (
        SELECT source_company_id AS firm_id FROM rel_raw
        UNION
        SELECT target_company_id FROM rel_raw
    ),
    probe AS (SELECT f.firm_id, q.qend FROM firms f CROSS JOIN quarters q)
    SELECT p.firm_id, p.qend, h.home_region
    FROM probe p
    ASOF LEFT JOIN rev_co_dedup h
      ON p.firm_id = h.company_id AND p.qend >= h.start_d
    WHERE h.company_id IS NOT NULL
      AND (h.end_d IS NULL OR h.end_d >= p.qend)
""")

print("(6) all_revere counts: active edge at qend x as-of country ...")
con.execute(f"""
    CREATE TABLE counts_all AS
    SELECT u.home_region AS country, q.qend AS quarter_end,
           COUNT(DISTINCT e.firm_id) AS n_firms_cn_link
    FROM cn_edge e
    JOIN quarters q
      ON q.qend >= e.rel_start
     AND (e.rel_end IS NULL OR e.rel_end >= q.qend)
    JOIN firm_region_q u
      ON u.firm_id = e.firm_id AND u.qend = q.qend
    WHERE u.home_region IN {CSET_SQL}
    GROUP BY 1, 2
""")

print("(6b) all_revere denominators: all firms with valid as-of country ...")
con.execute(f"""
    CREATE TABLE denominator_all AS
    SELECT h.home_region AS country, q.qend AS quarter_end,
           COUNT(DISTINCT h.company_id) AS n_firms_universe
    FROM rev_co_dedup h
    JOIN quarters q
      ON q.qend >= h.start_d
     AND (h.end_d IS NULL OR h.end_d >= q.qend)
    WHERE h.home_region IN {CSET_SQL}
    GROUP BY 1, 2
""")

print("(6c) supply_chain denominators: firms with >=1 active CUST/SUPP edge ...")
# Coverage-matched denominator (MAIN universe for the figure): firms that have
# at least one ACTIVE CUSTOMER/SUPPLIER relationship (any counterparty) at the
# quarter-end, with a valid as-of home_region. Numerator and denominator are
# built from the same edge data, so Revere's analyst-coverage expansion nets
# out to first order — unlike the all-Revere company master, whose batch
# expansions (US: 2011, 2015) mechanically crash the ratio.
con.execute(f"""
    CREATE TABLE sc_firm_active AS
    SELECT DISTINCT f.firm_id, q.qend
    FROM (
        SELECT source_company_id AS firm_id, rel_start, rel_end FROM rel_raw
        UNION ALL
        SELECT target_company_id, rel_start, rel_end FROM rel_raw
    ) f
    JOIN quarters q
      ON q.qend >= f.rel_start
     AND (f.rel_end IS NULL OR f.rel_end >= q.qend)
""")
con.execute(f"""
    CREATE TABLE denominator_sc AS
    SELECT u.home_region AS country, s.qend AS quarter_end,
           COUNT(DISTINCT s.firm_id) AS n_firms_universe
    FROM sc_firm_active s
    JOIN firm_region_q u
      ON u.firm_id = s.firm_id AND u.qend = s.qend
    WHERE u.home_region IN {CSET_SQL}
    GROUP BY 1, 2
""")
dsc = con.execute("SELECT COUNT(*) FROM denominator_sc").fetchone()[0]
print(f"    denominator_sc cells: {dsc:,}")

print("(7) Regression universe: crosswalk 06 logic (CUSIP > ISIN > SEDOL) ...")
# Identifiers of sec_entity_ids ever appearing with EU sec_country in holdings.
con.execute(f"""
    CREATE TABLE eu_sec_ids AS
    SELECT DISTINCT sec_entity_id,
           NULLIF(TRIM(cusip), '') AS cusip,
           NULLIF(TRIM(isin),  '') AS isin,
           NULLIF(TRIM(sedol), '') AS sedol
    FROM read_parquet('{EOM_PARQUET}')
    WHERE sec_country IN {EU_SQL} AND sec_entity_id IS NOT NULL
""")
con.execute(f"""
    CREATE TABLE reg_firms AS
    WITH univ AS (
        SELECT eu_company_id,
               NULLIF(TRIM(eu_cusip), '') AS eu_cusip,
               NULLIF(TRIM(eu_isin),  '') AS eu_isin,
               NULLIF(TRIM(eu_sedol), '') AS eu_sedol
        FROM read_parquet('{UNIV_STATIC_PARQUET}')
    ),
    all_matches AS (
        SELECT 1 AS prio, s.sec_entity_id, u.eu_company_id
        FROM eu_sec_ids s JOIN univ u ON s.cusip = u.eu_cusip
        WHERE s.cusip IS NOT NULL
        UNION ALL
        SELECT 2, s.sec_entity_id, u.eu_company_id
        FROM eu_sec_ids s JOIN univ u ON s.isin = u.eu_isin
        WHERE s.isin IS NOT NULL
        UNION ALL
        SELECT 3, s.sec_entity_id, u.eu_company_id
        FROM eu_sec_ids s JOIN univ u ON s.sedol = u.eu_sedol
        WHERE s.sedol IS NOT NULL
    ),
    cw AS (
        SELECT sec_entity_id, eu_company_id
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY sec_entity_id
                                         ORDER BY prio ASC, eu_company_id ASC) AS rn
            FROM all_matches
        )
        WHERE rn = 1
    )
    SELECT DISTINCT cw.eu_company_id AS firm_id
    FROM cw
    WHERE cw.sec_entity_id IN (
        SELECT DISTINCT sec_entity_id FROM read_parquet('{MERGED_PARQUET}')
    )
""")
nrf = con.execute("SELECT COUNT(*) FROM reg_firms").fetchone()[0]
print(f"    Revere firms in regression (held) universe: {nrf:,}")

print("(8) regression counts (EU countries only) ...")
con.execute(f"""
    CREATE TABLE counts_reg AS
    SELECT u.home_region AS country, q.qend AS quarter_end,
           COUNT(DISTINCT e.firm_id) AS n_firms_cn_link
    FROM cn_edge e
    JOIN quarters q
      ON q.qend >= e.rel_start
     AND (e.rel_end IS NULL OR e.rel_end >= q.qend)
    JOIN firm_region_q u
      ON u.firm_id = e.firm_id AND u.qend = q.qend
    WHERE u.home_region IN {EU_SQL}
      AND e.firm_id IN (SELECT firm_id FROM reg_firms)
    GROUP BY 1, 2
""")

print("(8b) regression denominators (EU countries only) ...")
con.execute(f"""
    CREATE TABLE denominator_reg AS
    SELECT h.home_region AS country, q.qend AS quarter_end,
           COUNT(DISTINCT h.company_id) AS n_firms_universe
    FROM rev_co_dedup h
    JOIN quarters q
      ON q.qend >= h.start_d
     AND (h.end_d IS NULL OR h.end_d >= q.qend)
    WHERE h.home_region IN {EU_SQL}
      AND h.company_id IN (SELECT firm_id FROM reg_firms)
    GROUP BY 1, 2
""")

print("(9) Assemble CSV (zero-filled full country x quarter grid) ...")
df = con.execute(f"""
    WITH grid_all AS (
        SELECT c.country, q.qend AS quarter_end, 'all_revere' AS universe
        FROM (SELECT UNNEST({list(C_SET)}) AS country) c CROSS JOIN quarters q
    ),
    grid_sc AS (
        SELECT c.country, q.qend AS quarter_end, 'supply_chain' AS universe
        FROM (SELECT UNNEST({list(C_SET)}) AS country) c CROSS JOIN quarters q
    ),
    grid_reg AS (
        SELECT c.country, q.qend AS quarter_end, 'regression' AS universe
        FROM (SELECT UNNEST({list(EU_COUNTRIES)}) AS country) c CROSS JOIN quarters q
    ),
    grid AS (SELECT * FROM grid_all
             UNION ALL SELECT * FROM grid_sc
             UNION ALL SELECT * FROM grid_reg)
    SELECT g.country, g.quarter_end,
           COALESCE(CASE WHEN g.universe = 'regression' THEN r.n_firms_cn_link
                         ELSE a.n_firms_cn_link END, 0) AS n_firms_cn_link,
           COALESCE(CASE g.universe
                        WHEN 'all_revere'   THEN da.n_firms_universe
                        WHEN 'supply_chain' THEN ds.n_firms_universe
                        ELSE dr.n_firms_universe END, 0) AS n_firms_universe,
           g.universe
    FROM grid g
    LEFT JOIN counts_all a
      ON a.country = g.country AND a.quarter_end = g.quarter_end
    LEFT JOIN counts_reg r
      ON g.universe = 'regression' AND r.country = g.country
     AND r.quarter_end = g.quarter_end
    LEFT JOIN denominator_all da
      ON g.universe = 'all_revere' AND da.country = g.country
     AND da.quarter_end = g.quarter_end
    LEFT JOIN denominator_sc ds
      ON g.universe = 'supply_chain' AND ds.country = g.country
     AND ds.quarter_end = g.quarter_end
    LEFT JOIN denominator_reg dr
      ON g.universe = 'regression' AND dr.country = g.country
     AND dr.quarter_end = g.quarter_end
    ORDER BY g.universe, g.country, g.quarter_end
""").df()
df["quarter_end"] = pd.to_datetime(df["quarter_end"])
df = add_fraction(
    df,
    numerator="n_firms_cn_link",
    denominator="n_firms_universe",
    output="fraction_cn_link",
)
assert_nested_universe(
    df,
    outer_label="all_revere",
    inner_label="regression",
    key_cols=["country", "quarter_end"],
    metric_cols=["n_firms_cn_link", "n_firms_universe"],
)
# supply_chain denominator (firms with >=1 active edge) must nest inside the
# all-Revere company universe; numerators are identical between the two.
assert_nested_universe(
    df,
    outer_label="all_revere",
    inner_label="supply_chain",
    key_cols=["country", "quarter_end"],
    metric_cols=["n_firms_cn_link", "n_firms_universe"],
)
df.to_csv(CSV_OUT, index=False)
print(f"    -> {CSV_OUT}  ({len(df):,} rows)")

print("    sanity: numerators are bounded by denominators; regression is nested")

# Cross-check vs 02 output (different definition: EU total, ALL rel_types
# incl COMPETITOR/PARTNER-*, so 02 should be >= our EU total; order-of-
# magnitude check only).
ts02 = OUT_DIR / "02_china_exposure_timeseries.csv"
if ts02.is_file():
    t2 = pd.read_csv(ts02, parse_dates=["quarter_end"])
    ours = (df[df["universe"] == "all_revere"]
            .query("country != 'US'")
            .groupby("quarter_end")["n_firms_cn_link"].sum())
    for qd in ("2003-06-30", "2018-12-31"):
        q = pd.Timestamp(qd)
        v02 = t2.loc[t2["quarter_end"] == q, "n_firms_with_cn"]
        v02 = int(v02.iloc[0]) if len(v02) else None
        print(f"    cross-check {qd}: ours(EU total, CUST+SUPP)={int(ours.get(q, 0))}"
              f"  02(EU total, all rel_types)={v02}")

# ------------------------------------------------------------------
# Plot: fractions in small multiples, US first, one colour, horizontal grid.
# ------------------------------------------------------------------
print("(10) Plot ...")
BG = "#fcfcfb"; INK = "#0b0b0b"; MUTED = "#52514e"
LINE = "#2a78d6"; GRID = "#e8e8e6"

MIN_DENOM_FIRMS = 50  # hide cells whose supply-chain universe is tiny

sc = df[df["universe"] == "supply_chain"].copy()
# Mask (plot only; CSV keeps raw values): fraction is meaningless when the
# denominator is a handful of firms.
sc.loc[sc["n_firms_universe"] < MIN_DENOM_FIRMS, "fraction_cn_link"] = float("nan")
wide_all = (sc.pivot(index="quarter_end", columns="country",
                     values="fraction_cn_link") * 100.0)

# Panels: US first, then EU countries with any unmasked nonzero history,
# by peak desc. Countries fully masked (tiny supply-chain universe) drop out.
eu_present = [c for c in EU_COUNTRIES
              if c in wide_all.columns and wide_all[c].notna().any()
              and wide_all[c].max() > 0]
eu_present.sort(key=lambda c: (-float(wide_all[c].max()), c))
panels = ["US"] + eu_present
dropped = [c for c in EU_COUNTRIES if c not in eu_present]
if dropped:
    print(f"    panels dropped (supply-chain universe always < "
          f"{MIN_DENOM_FIRMS} or never a CN link): {','.join(dropped)}")

ncols = 5
nrows = -(-len(panels) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.1 * nrows),
                         sharex=True, squeeze=False)
fig.patch.set_facecolor(BG)

for k, ax in enumerate(axes.flat):
    if k >= len(panels):
        ax.axis("off")
        continue
    c = panels[k]
    ax.set_facecolor(BG)
    s = wide_all[c]
    ax.plot(s.index, s.values, color=LINE, linewidth=1.8,
            solid_capstyle="round")
    # Isolated unmasked observations would be invisible on a line plot.
    obs = s.notna().to_numpy()
    if obs.any():
        import numpy as _np
        prev_na = _np.r_[True, ~obs[:-1]]
        next_na = _np.r_[~obs[1:], True]
        iso = obs & prev_na & next_na
        if iso.any():
            ax.plot(s.index[iso], s.values[iso], linestyle="none",
                    marker="o", markersize=2.4, color=LINE)
    ax.set_title(COUNTRY_NAMES.get(c, c), fontsize=9.5, color=INK,
                 loc="left", pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=3)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_minor_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.margins(x=0.02)

fig.suptitle("Share of firms with China supply-chain links, by country",
             fontsize=13, color=INK, x=0.01, ha="left", y=0.995)
fig.text(0.01, 0.972,
         "Percent of firms with at least one active supply-chain (customer or "
         f"supplier) relationship in Revere, same country-quarter. Quarters "
         f"with fewer than {MIN_DENOM_FIRMS} such firms are not shown.",
         ha="left", fontsize=8.5, color=MUTED)
fig.supxlabel("")
fig.tight_layout(rect=(0, 0.0, 1, 0.945))

fig.savefig(FIG_PNG, dpi=200, facecolor=BG)
fig.savefig(FIG_PDF, facecolor=BG)
plt.close(fig)
print(f"    -> {FIG_PNG}")
print(f"    -> {FIG_PDF}")

# ------------------------------------------------------------------
# Headline numbers (printed from the CSV-backed dataframe, not hand-derived).
# ------------------------------------------------------------------
print("\nHeadline fractions (supply_chain universe, masked cells excluded):")
masked = sc.dropna(subset=["fraction_cn_link"])
top_eu = (masked[masked["country"] != "US"]
          .groupby("country")["fraction_cn_link"].max()
          .sort_values(ascending=False).head(3).index.tolist())
for c in ["US"] + top_eu:
    s = (masked[masked["country"] == c]
         .set_index("quarter_end")["fraction_cn_link"] * 100)
    if s.empty:
        print(f"  {c}: fully masked")
        continue
    peak_q = s.idxmax()
    print(f"  {c}: first-shown {s.index.min().date()}={s.iloc[0]:.2f}%, "
          f"peak {peak_q.date()}={s.max():.2f}%, "
          f"end {s.index.max().date()}={s.iloc[-1]:.2f}%")

print("\nDONE.")
