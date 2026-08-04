# ============================================================================
# PRE-P0 VINTAGE WARNING (P0 holdings-snapshot rebuild, 2026-08-04)
# This script's output is PRE-P0: it was built from the exact-EOM
# holdings_eom.parquet (or from merged_us_eu_zero_filled.parquet built from
# it). The as-of quarter-end selection rule in 03_eom_etl.jl CHANGED the
# panel's fund universe on EVERY quarter. PRE-P0, pending re-run, do not mix
# with post-P0 results. Register: julia_descriptive/VINTAGE_P0.md
# ============================================================================

"""
build_spell_boundary.py

Advisor's "conditional / spell-boundary" zero-fill (comment 3):
Instead of the full Cartesian grid (every firm x group x quarter zero-filled),
keep only:
  - held quarters (I_ict > 0), AND
  - the ONE boundary zero immediately before each entry and after each exit.
Drop the deep never-held interior, and drop firm-groups the group NEVER held
(the "conditional on engagement" restriction).

Keep rule per (firm, group) series ordered by quarter:
    keep(q) = held(q) OR held(q-1) OR held(q+1)
where held = 1[I_ict > 0]. Series boundaries: missing lag/lead treated as 0.

delta_w is reused from the full-grid parquet (it was computed with the correct
full-grid lag, so the post-exit boundary zero already carries the exit -Delta w).

Output: output/c6_panel_spell.dta  (Stata-ready, same columns as c6_panel.dta).
"""

from pathlib import Path
import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT  = PROJ / "output"
PARQUET = OUT / "merged_us_eu_zero_filled.parquet"
DTA = OUT / "c6_panel_spell.dta"
uri = str(PARQUET).replace("\\", "/")

con = duckdb.connect()

# ---------------------------------------------------------------
# 1. Flag held + neighbors, per (firm, group) series.
# ---------------------------------------------------------------
print("[1/3] Flagging spell-boundary rows...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE flagged AS
SELECT
    sec_entity_id, sec_country, holder_group, investor_country,
    report_date, I_ict, portfolio_weight_eu, delta_w,
    china_share, china_share_lag1q, gpr_us_cn, shock_us_cn,
    CAST(I_ict > 0 AS INTEGER) AS held,
    COALESCE(LAG(CAST(I_ict > 0 AS INTEGER)) OVER w, 0)  AS held_lag,
    COALESCE(LEAD(CAST(I_ict > 0 AS INTEGER)) OVER w, 0) AS held_lead
FROM read_parquet('{uri}')
WINDOW w AS (PARTITION BY sec_entity_id, holder_group ORDER BY report_date)
""")

# ---------------------------------------------------------------
# 2. Keep held + immediate boundary zeros. Build final columns.
# ---------------------------------------------------------------
print("[2/3] Selecting held + boundary zeros...")
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
    held,
    CASE WHEN held = 1 THEN 'held' ELSE 'boundary_zero' END AS row_type
FROM flagged
WHERE held = 1 OR held_lag = 1 OR held_lead = 1
""").df()

n_all = len(df)
print(f"      spell-boundary rows (before dropping NULLs): {n_all:,}")
print(f"      row type:\n{df['row_type'].value_counts().to_string()}")
print(f"      unique firm-groups with a spell: "
      f"{df.groupby(['firm_str','hgroup']).ngroups:,}")
print(f"      unique firms: {df['firm_str'].nunique():,}")

# ---------------------------------------------------------------
# 3. Estimation subset: drop rows with missing dw / cn_lag / shock.
# ---------------------------------------------------------------
print("[3/3] Dropping rows with missing dw / cn_lag / shock for estimation...")
est = df.dropna(subset=["dw", "cn_lag", "shock"]).copy()
n_est = len(est)
print(f"      estimation rows: {n_est:,}  (dropped {n_all - n_est:,})")
print(f"      hgroup split:\n{est['hgroup'].value_counts().to_string()}")
print(f"      row type in estimation sample:\n{est['row_type'].value_counts().to_string()}")
print(f"      unique firms in estimation: {est['firm_str'].nunique():,}")
print(f"      rdate range: {est['rdate'].min()} -> {est['rdate'].max()}")

# types
est["firm_str"] = est["firm_str"].astype(str)
est["hgroup"]   = est["hgroup"].astype(str)
est["rdate"]    = pd.to_datetime(est["rdate"])
for c in ["dw", "cn_lag", "shock"]:
    est[c] = pd.to_numeric(est[c], errors="raise").astype("float64")
est["us"] = est["us"].astype("int8")

# sanity
assert n_est > 0
assert est["dw"].notna().all() and est["cn_lag"].notna().all() and est["shock"].notna().all()
assert ((est["hgroup"] == "US") == (est["us"] == 1)).all()

keep_cols = ["firm_str", "hgroup", "rdate", "dw", "cn_lag", "shock", "us"]
est[keep_cols].to_stata(DTA, write_index=False, convert_dates={"rdate": "tc"}, version=118)
print(f"      wrote {DTA}  ({n_est:,} rows)")
