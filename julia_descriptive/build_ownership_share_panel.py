"""
build_ownership_share_panel.py  —  Essay 2, task #5 (shares-based outcome), STEP 1-2 only.

Motivation. The main outcome is a MARKET-VALUE portfolio weight w = H(USD)/T(USD),
which mixes trading flow with price changes and portfolio-denominator reallocation.
To support a "US investors do not SELL / do not reduce their stake" reading (not just
"do not reduce portfolio weight"), we need a pure QUANTITY measure. Shares are not
additive across firms (different prices/units), so a shares-based portfolio weight is
meaningless. The right object is the group's OWNERSHIP SHARE of each firm:

    ownership_share_{i,g,t} = ( sum_{b in g} adj_holding_{b,i,t} ) / adj_shares_out_{i,t}

= fraction of firm i's float held by group g. It is immune to price (numerator and
denominator are both in shares) and to portfolio-denominator reallocation (no T term).
Delta(ownership_share) is net buying/selling of firm i by group g as a share of float.

--- MULTI-AGENT REVIEW FIX (must-fix, review wsu1zupkc) ---
A `sec_entity_id` maps to MULTIPLE `fsym_id` security classes (primary equity, other
classes, ADR/GDR). Each class has its OWN adj_shares_out and its OWN adj_holding, in
DIFFERENT units, so pooling them under one sec_entity_id is unit-inconsistent for a
SHARES measure. The first draft pooled all classes and papered over the resulting
"dual-universe" shares_out with MODE(); verified that this left 19,771 sec-quarters
(5.28%) with >1 distinct shares_out (up to 1e12x) and inflated ownership_share ~2.6x on
the 6.7% of dispersed cells. Restricting to `fsym_id = fsym_primary_id` (the primary
equity class, exactly as the reference market-cap in 04_us_ownership_european.jl:171-183)
drives within-(sec,quarter) shares_out dispersion to EXACTLY ZERO, so we no longer need
MODE — adj_shares_out is a constant and AVG() returns it. We further restrict to
issue_type='EQ' (the primary listing float; the market-cap reference is EQ-only), so
numerator and denominator live in the same well-defined universe.

--- LIMITATION this creates (documented, quantified) ---
Restricting to the primary EQ class DROPS holdings on non-primary classes and on ADR/GDR
(AD). Measured on the EQ-primary measure this build uses (the `frac_*_on_primary_eq`
diagnostics): the fraction of European holdings USD on the primary EQ class is 96.68% for
NONUS but only 91.08% for US — i.e. US holds ~8.92% of its European exposure off the
primary class (ADR channel) vs 3.32% for NONUS. This ASYMMETRY matters: if US
investors adjust via ADRs, the primary-class ownership-share test will not see it, which
could bias the "do US sell" test toward a null. The USD portfolio-weight main spec DOES
capture the ADR channel, so the two outcomes are complementary. An ADR-inclusive
ownership-share robustness would need the ADR conversion ratio and is deferred.

This script does STEP 1-2 ONLY: aggregate to (firm, group, quarter), attach the
primary-class float, compute ownership_share, run integrity diagnostics (incl. a
COMBINED US+NONUS > 1 check and a by-country missingness breakdown), and stop. It does
NOT build the Cartesian grid, zero-fill, difference, or run any regression — those are
step 3, and the entry/exit zero-fill convention (mirroring build_c6_panel.py) is a
step-3 design item flagged by the review.

Outputs:
  output/ownership_share_observed.parquet         (observed firm-group-quarter cells)
  output/ownership_share_diagnostics.csv          (one-row summary of data quality)
  output/ownership_share_missing_by_country.csv   (residual shrout missingness by country)
"""

from pathlib import Path
import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
EOM = (OUT / "holdings_eom.parquet").as_posix()
OBS_PARQUET = OUT / "ownership_share_observed.parquet"
FLOAT_PARQUET = OUT / "ownership_share_float.parquet"   # shares_out per (firm, quarter), for step-3 zero-fill
DIAG_CSV = OUT / "ownership_share_diagnostics.csv"
MISS_CSV = OUT / "ownership_share_missing_by_country.csv"

# 28 European listing jurisdictions — MUST match 00_setup.jl EU_COUNTRIES exactly.
EU = ("GB", "DE", "FR", "NL", "CH", "IT", "ES", "SE", "DK", "NO", "FI", "BE",
      "AT", "IE", "LU", "PT", "PL", "CZ", "HU", "GR", "RO", "SK", "SI", "BG",
      "HR", "EE", "LV", "LT")
EU_SQL = "(" + ",".join(f"'{c}'" for c in EU) + ")"

# Primary EQ class only: matches the reference market-cap float rule and makes
# shares_out constant within each security-quarter (numerator + denominator same universe).
BASE_WHERE = f"issue_type = 'EQ' AND sec_country IN {EU_SQL} AND fsym_id = fsym_primary_id"

con = duckdb.connect()
con.execute("SET memory_limit='6GB'")

print("[1/5] Primary-class float per (sec, quarter) — shares_out now constant, AVG()...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE shr AS
SELECT sec_entity_id,
       report_date,
       AVG(adj_shares_out)            AS shares_out,
       COUNT(DISTINCT adj_shares_out) AS n_distinct_shrout,
       COUNT(*)                       AS n_holder_rows
FROM read_parquet('{EOM}')
WHERE {BASE_WHERE}
  AND adj_shares_out IS NOT NULL AND adj_shares_out > 0
GROUP BY 1, 2
""")

# Integrity: after the primary-class filter, shares_out must be a within-cell constant.
_disp = con.execute("SELECT COUNT(*) FROM shr WHERE n_distinct_shrout > 1").fetchone()[0]
assert _disp == 0, f"{_disp} (sec, quarter) cells still have dispersed shares_out after fsym=primary — investigate"

# Persist the primary-EQ float per (firm, quarter) so step 3 can zero-fill the grid:
# a grid cell where a group holds nothing but the firm HAS a valid float -> ownership 0;
# a firm-quarter with NO primary-EQ float -> ownership NULL (excluded). The distinction
# needs this table, not just the observed (held>0) cells.
con.execute(f"COPY (SELECT sec_entity_id, report_date, shares_out FROM shr) "
            f"TO '{FLOAT_PARQUET.as_posix()}' (FORMAT PARQUET)")

print("[2/5] Group (US / NONUS) shares held per (firm, quarter)...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE grp AS
SELECT sec_entity_id,
       any_value(sec_country)                                   AS sec_country,
       report_date,
       CASE WHEN investor_country = 'US' THEN 'US' ELSE 'NONUS' END AS hgroup,
       SUM(adj_holding)                                          AS shares_held,
       COUNT(*)                                                  AS n_positions
FROM read_parquet('{EOM}')
WHERE {BASE_WHERE}
  AND adj_holding IS NOT NULL AND adj_holding > 0
GROUP BY sec_entity_id, report_date,
         CASE WHEN investor_country = 'US' THEN 'US' ELSE 'NONUS' END
""")

print("[3/5] Join + ownership_share, flag impossible (>1)...")
obs = con.execute("""
SELECT g.sec_entity_id,
       g.sec_country,
       g.report_date,
       g.hgroup,
       g.shares_held,
       g.n_positions,
       s.shares_out,
       g.shares_held / NULLIF(s.shares_out, 0)                   AS ownership_share,
       CASE WHEN g.shares_held / NULLIF(s.shares_out, 0) > 1
            THEN 1 ELSE 0 END                                    AS os_impossible,
       CASE WHEN s.shares_out IS NULL THEN 1 ELSE 0 END          AS shrout_missing
FROM grp g
LEFT JOIN shr s USING (sec_entity_id, report_date)
""").df()
obs.to_parquet(OBS_PARQUET, index=False)

print("[4/5] Integrity diagnostics (single-group + COMBINED US+NONUS)...")
# Combined per-(firm, quarter) ownership across BOTH groups must also be <= 1
# (FactSet covers a subset of holders, so institutional ownership cannot exceed float).
# The per-group os_impossible flag misses cases where the combined total > 1 but
# neither group individually exceeds 1 — check that separately.
combined = con.execute("""
SELECT COUNT(*) AS n_combined_gt1
FROM (
    SELECT sec_entity_id, report_date,
           SUM(shares_held) / NULLIF(AVG(shares_out), 0) AS combined_os
    FROM (
        SELECT g.sec_entity_id, g.report_date, g.shares_held, s.shares_out
        FROM grp g LEFT JOIN shr s USING (sec_entity_id, report_date)
    )
    GROUP BY sec_entity_id, report_date
    HAVING SUM(shares_held) / NULLIF(AVG(shares_out), 0) > 1
)
""").df()["n_combined_gt1"].iloc[0]

# ADR/non-primary materiality (what the primary-class restriction drops), by group.
adr = con.execute(f"""
SELECT CASE WHEN investor_country='US' THEN 'US' ELSE 'NONUS' END AS g,
       SUM(CASE WHEN issue_type='EQ' AND fsym_id=fsym_primary_id THEN adj_mv ELSE 0 END)
         / SUM(adj_mv) AS frac_mv_on_primary_eq
FROM read_parquet('{EOM}')
WHERE issue_type IN ('EQ','AD') AND sec_country IN {EU_SQL} AND adj_mv > 0
GROUP BY 1
""").df().set_index("g")["frac_mv_on_primary_eq"].to_dict()

# Residual missingness by country (post-fix) — computed on the observed frame.
mb = (obs.groupby("sec_country")
        .agg(n_cells=("shrout_missing", "size"), frac_shrout_missing=("shrout_missing", "mean"))
        .sort_values("frac_shrout_missing", ascending=False).reset_index())
mb.to_csv(MISS_CSV, index=False)

print("[5/5] Summary...")
n = len(obs)
os_clean = obs.loc[(obs["shrout_missing"] == 0) & (obs["ownership_share"] <= 1), "ownership_share"]
diag = {
    "n_firm_group_quarter_cells": n,
    "n_distinct_firms": obs["sec_entity_id"].nunique(),
    "n_distinct_quarters": obs["report_date"].nunique(),
    "rdate_min": str(obs["report_date"].min()),
    "rdate_max": str(obs["report_date"].max()),
    "frac_cells_shrout_missing": round(float((obs["shrout_missing"] == 1).mean()), 6),
    "n_cells_os_impossible_gt1": int(obs["os_impossible"].sum()),
    "frac_cells_os_impossible_gt1": round(float(obs["os_impossible"].mean()), 8),
    "n_firmquarters_combined_os_gt1": int(combined),
    # distribution over CLEAN cells only (<=1, non-missing) — excludes impossibles
    "os_median_clean": round(float(os_clean.median()), 8),
    "os_p90_clean": round(float(os_clean.quantile(0.90)), 8),
    "os_p99_clean": round(float(os_clean.quantile(0.99)), 8),
    "os_max_clean": round(float(os_clean.max()), 8),
    # ADR / non-primary materiality (dropped by the primary-class restriction)
    "frac_us_holdings_mv_on_primary_eq": round(float(adr.get("US", float("nan"))), 6),
    "frac_nonus_holdings_mv_on_primary_eq": round(float(adr.get("NONUS", float("nan"))), 6),
    "ref_main_panel_firms": 7928,
    "ref_main_panel_quarters": 82,
}
pd.DataFrame([diag]).to_csv(DIAG_CSV, index=False)

print("\n===== ownership_share panel (observed, pre-grid, PRIMARY EQ class) =====")
for k, v in diag.items():
    print(f"  {k:38s} {v}")
print(f"\n  hgroup split:\n{obs['hgroup'].value_counts().to_string()}")
print(f"\n  top-5 countries by residual shrout missingness:\n{mb.head(5).to_string(index=False)}")
print(f"\n  wrote {OBS_PARQUET.name}, {DIAG_CSV.name}, {MISS_CSV.name}")
print("\nSTEP 1-2 complete. Grid / zero-fill / entry-exit convention / difference / "
      "regression = step 3 (not run).")
con.close()
