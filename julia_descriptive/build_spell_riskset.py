# ============================================================================
# PRE-P0 VINTAGE WARNING (P0 holdings-snapshot rebuild, 2026-08-04)
# This script's output is PRE-P0: it was built from the exact-EOM
# holdings_eom.parquet (or from merged_us_eu_zero_filled.parquet built from
# it). The as-of quarter-end selection rule in 03_eom_etl.jl CHANGED the
# panel's fund universe on EVERY quarter. PRE-P0, pending re-run, do not mix
# with post-P0 results. Register: julia_descriptive/VINTAGE_P0.md
# ============================================================================

"""
build_spell_riskset.py

CORRECTED spell-boundary sample (fixes build_spell_boundary.py, which selected
per-(firm,group) and broke the US-vs-NONUS pairing).

Risk set is defined at the FIRM-QUARTER level:
  A firm-quarter (i,t) enters the risk set if EITHER group (US or NONUS) has a
  spell-boundary row there, i.e. is held at t, or held at t-1, or held at t+1.
Once (i,t) is in the risk set, KEEP BOTH group rows -- (i,US,t) and (i,NONUS,t) --
even if one side is a boundary zero or a deep zero. This preserves the within-
firm-quarter US-vs-NONUS comparison that alpha_{i,t} identifies off.

delta_w reused from the full-grid parquet (correct full-grid lag).
Output: output/c6_panel_riskset.dta
"""

from pathlib import Path
import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT  = PROJ / "output"
PARQUET = OUT / "merged_us_eu_zero_filled.parquet"
DTA = OUT / "c6_panel_riskset.dta"
uri = str(PARQUET).replace("\\", "/")

con = duckdb.connect()

print("[1/4] Flagging held + neighbors per (firm, group)...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE flagged AS
SELECT
    sec_entity_id, sec_country, holder_group, investor_country,
    report_date, I_ict, portfolio_weight_eu, delta_w,
    china_share_lag1q, shock_us_cn,
    CAST(I_ict > 0 AS INTEGER) AS held,
    COALESCE(LAG(CAST(I_ict > 0 AS INTEGER))  OVER w, 0) AS held_lag,
    COALESCE(LEAD(CAST(I_ict > 0 AS INTEGER)) OVER w, 0) AS held_lead
FROM read_parquet('{uri}')
WINDOW w AS (PARTITION BY sec_entity_id, holder_group ORDER BY report_date)
""")

print("[2/4] Building FIRM-QUARTER risk set (union over groups)...")
con.execute("""
CREATE OR REPLACE TEMP TABLE riskset AS
SELECT *,
    CASE WHEN held=1 OR held_lag=1 OR held_lead=1 THEN 1 ELSE 0 END AS sb,
    MAX(CASE WHEN held=1 OR held_lag=1 OR held_lead=1 THEN 1 ELSE 0 END)
        OVER (PARTITION BY sec_entity_id, report_date) AS risk
FROM flagged
""")

print("[3/4] Keeping BOTH group rows for risk-set firm-quarters...")
df = con.execute("""
SELECT
    CAST(sec_entity_id AS VARCHAR)                  AS firm_str,
    sec_country,
    holder_group                                    AS hgroup,
    CAST(report_date AS TIMESTAMP)                  AS rdate,
    delta_w                                         AS dw,
    china_share_lag1q                               AS cn_lag,
    shock_us_cn                                     AS shock,
    CASE WHEN holder_group = 'US' THEN 1 ELSE 0 END AS us,
    held, sb,
    CASE WHEN held=1 THEN 'held'
         WHEN sb=1  THEN 'boundary_zero'
         ELSE 'deep_zero_paired' END               AS row_type
FROM riskset
WHERE risk = 1
""").df()

n_all = len(df)
print(f"      risk-set rows (before dropping NULLs): {n_all:,}")
print(f"      hgroup split (should be balanced):\n{df['hgroup'].value_counts().to_string()}")
print(f"      row type:\n{df['row_type'].value_counts().to_string()}")
# verify pairing: every (firm, quarter) has exactly 2 rows
pair = df.groupby(['firm_str','rdate']).size()
print(f"      firm-quarters with exactly 2 rows: {(pair==2).sum():,} / {len(pair):,}"
      f"  (unpaired: {(pair!=2).sum():,})")
# hard fail: risk-set MUST keep both group rows per firm-quarter (external-review fix)
assert (pair == 2).all(), \
    f"risk-set not fully paired before drop: {(pair != 2).sum():,} unpaired firm-quarters"

print("[4/4] Estimation subset: drop missing dw / cn_lag / shock...")
est = df.dropna(subset=["dw", "cn_lag", "shock"]).copy()
n_est = len(est)
print(f"      estimation rows: {n_est:,}  (dropped {n_all - n_est:,})")
print(f"      hgroup split:\n{est['hgroup'].value_counts().to_string()}")
pair2 = est.groupby(['firm_str','rdate']).size()
print(f"      paired firm-quarters after drop: {(pair2==2).sum():,} / {len(pair2):,}"
      f"  (unpaired singletons: {(pair2!=2).sum():,})")
print(f"      unique firms: {est['firm_str'].nunique():,}")
print(f"      rdate range: {est['rdate'].min()} -> {est['rdate'].max()}")

est["firm_str"] = est["firm_str"].astype(str)
est["hgroup"]   = est["hgroup"].astype(str)
est["rdate"]    = pd.to_datetime(est["rdate"])
for c in ["dw", "cn_lag", "shock"]:
    est[c] = pd.to_numeric(est[c], errors="raise").astype("float64")
est["us"] = est["us"].astype("int8")

assert n_est > 0
assert ((est["hgroup"] == "US") == (est["us"] == 1)).all()
# hard asserts (external-review fix): dedup + pairing + shock uniqueness + cn_lag range
assert not est.duplicated(["firm_str", "hgroup", "rdate"]).any(), "duplicate rows in estimation subset"
_p2 = est.groupby(["firm_str", "rdate"])["hgroup"].nunique()
assert (_p2 == 2).all(), \
    f"estimation subset not fully paired: {(_p2 != 2).sum():,} unpaired firm-quarters"
assert (est.groupby("rdate")["shock"].nunique() == 1).all(), "shock varies within a quarter"
assert est["cn_lag"].between(0, 1).all(), "cn_lag outside [0,1]"

keep_cols = ["firm_str", "hgroup", "rdate", "dw", "cn_lag", "shock", "us"]
est[keep_cols].to_stata(DTA, write_index=False, convert_dates={"rdate": "tc"}, version=118)
print(f"      wrote {DTA}  ({n_est:,} rows)")
