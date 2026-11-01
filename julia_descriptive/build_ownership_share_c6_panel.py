# ============================================================================
# PRE-P0 VINTAGE WARNING (P0 holdings-snapshot rebuild, 2026-08-04)
# This script's output is PRE-P0: it was built from the exact-EOM
# holdings_eom.parquet (or from merged_us_eu_zero_filled.parquet built from
# it). The as-of quarter-end selection rule in 03_eom_etl.jl CHANGED the
# panel's fund universe on EVERY quarter. PRE-P0, pending re-run, do not mix
# with post-P0 results. Register: julia_descriptive/VINTAGE_P0.md
# ============================================================================

"""
build_ownership_share_c6_panel.py — Essay 2 #5 STEP 3 (panel build).

Merge the primary-EQ ownership onto the SAME C6 grid used by the w-based main spec
(merged_us_eu_zero_filled.parquet), zero-fill, and compute a shares-based flow for the
triple-difference. Structurally identical to c6_panel.dta but with a shares outcome.

--- REVIEW FIX F6 (float-denominator confound) ---
The first version used dos = ownership_share_t − ownership_share_{t−1} with EACH term
divided by its OWN current float: os = held/out_t, dos = held_t/out_t − held_{t−1}/out_{t−1}.
That is NOT denominator-immune: a buyback/issuance (out changes) moves dos even with zero
trading, by −os_{g,t−1}·(Δout/out_t), a term ∝ the group's own lagged ownership level, which
differs across US/NONUS within a firm-quarter and so is NOT absorbed by firm×quarter FE.

The correct pure-trading FLOW fixes the denominator at the LAGGED float:

    flow_{i,g,t} = ( shares_held_{i,g,t} − shares_held_{i,g,t−1} ) / shares_out_{i,t−1}

Numerator = the actual change in the group's share count (real net buying/selling; a firm
buyback does not change it unless the group trades). Denominator = fixed lagged float, so a
same-quarter corporate action cannot mechanically move it. This IS the "net buying/selling as
a fraction of float" object §7.6 intends. We also keep the OLD `dos` (share-of-float change)
as a labelled comparison column, not the primary outcome.

Zero-fill / entry-exit (grid cell with a valid primary-EQ float, group holds nothing → held=0;
firm with no primary-EQ float → NULL, excluded). BAD firm-quarters (any group os>1 OR combined
os>1) are nulled for BOTH groups (current AND lagged) to keep US/NONUS pairing and to stop a
corrupted float from leaking into flow.

Output: output/ownership_c6_panel.dta   (outcome `flow`; `dos` kept for comparison)
"""

from pathlib import Path
import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
GRID = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
OBS = (OUT / "ownership_share_observed.parquet").as_posix()
FLOAT = (OUT / "ownership_share_float.parquet").as_posix()
DTA = OUT / "ownership_c6_panel.dta"

con = duckdb.connect()
con.execute("SET memory_limit='6GB'")

print("[1/4] Merge held + primary-EQ float onto the main C6 grid...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE joined AS
WITH grid AS (
    SELECT sec_entity_id, holder_group, report_date, china_share_lag1q, shock_us_cn
    FROM read_parquet('{GRID}')
),
held AS (
    SELECT sec_entity_id, hgroup AS holder_group, report_date, shares_held
    FROM read_parquet('{OBS}')
),
flt AS (
    SELECT sec_entity_id, report_date, shares_out FROM read_parquet('{FLOAT}')
)
SELECT g.sec_entity_id, g.holder_group, g.report_date,
       g.china_share_lag1q, g.shock_us_cn,
       f.shares_out,
       -- held is 0 where the firm HAS a float but the group holds nothing; NULL where no float
       CASE WHEN f.shares_out IS NULL THEN NULL ELSE COALESCE(h.shares_held, 0) END AS held,
       CASE WHEN f.shares_out IS NULL THEN NULL
            ELSE COALESCE(h.shares_held, 0) / f.shares_out END AS os_raw
FROM grid g
LEFT JOIN held h USING (sec_entity_id, holder_group, report_date)
LEFT JOIN flt  f USING (sec_entity_id, report_date)
""")

print("[2/4] Flag BAD firm-quarters (any group os>1 OR combined>1); null both groups...")
con.execute("""
CREATE OR REPLACE TEMP TABLE clean AS
WITH bad AS (
    SELECT sec_entity_id, report_date
    FROM joined GROUP BY 1, 2
    HAVING MAX(os_raw) > 1 OR SUM(os_raw) > 1
)
SELECT j.sec_entity_id, j.holder_group, j.report_date,
       j.china_share_lag1q, j.shock_us_cn, j.shares_out,
       CASE WHEN b.sec_entity_id IS NOT NULL THEN NULL ELSE j.held   END AS held,
       CASE WHEN b.sec_entity_id IS NOT NULL THEN NULL ELSE j.os_raw END AS os
FROM joined j
LEFT JOIN bad b USING (sec_entity_id, report_date)
""")

print("[3/4] Flow = (held_t - held_{t-1}) / out_{t-1}, backward, within (firm, group)...")
df = con.execute("""
SELECT
    CAST(sec_entity_id AS VARCHAR)                  AS firm_str,
    holder_group                                    AS hgroup,
    CAST(report_date AS TIMESTAMP)                  AS rdate,
    os,
    held,
    -- F6 primary outcome: pure trading flow with FIXED lagged float
    (held - LAG(held) OVER w) / NULLIF(LAG(shares_out) OVER w, 0)  AS flow,
    -- old share-of-float change, kept for comparison (has the F6 float-confound)
    os - LAG(os) OVER w                             AS dos,
    china_share_lag1q                               AS cn_lag,
    shock_us_cn                                     AS shock,
    CASE WHEN holder_group = 'US' THEN 1 ELSE 0 END AS us
FROM clean
WINDOW w AS (PARTITION BY sec_entity_id, holder_group ORDER BY report_date)
""").df()
con.close()

df["firm_str"] = df["firm_str"].astype(str)
df["hgroup"] = df["hgroup"].astype(str)
df["rdate"] = pd.to_datetime(df["rdate"])
for c in ["os", "held", "flow", "dos", "cn_lag", "shock"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
df["us"] = df["us"].astype("int8")

# estimation subset: non-null FLOW (primary) / cn_lag / shock
est = df.dropna(subset=["flow", "cn_lag", "shock"]).copy()

print("[4/4] Hard asserts + write...")
n = len(est)
assert n > 0
assert set(est["hgroup"].unique()) <= {"US", "NONUS"}
assert ((est["hgroup"] == "US") == (est["us"] == 1)).all()
assert not est.duplicated(["firm_str", "hgroup", "rdate"]).any(), "duplicate rows"
_pair = est.groupby(["firm_str", "rdate"])["hgroup"].nunique()
assert (_pair == 2).all(), f"not fully paired: {(_pair != 2).sum():,} firm-quarters"
assert (est.groupby("rdate")["shock"].nunique() == 1).all(), "shock varies within a quarter"
assert est["cn_lag"].between(0, 1).all(), "cn_lag outside [0,1]"
_q = est["rdate"].dt.to_period("Q").drop_duplicates().sort_values()
_exp = pd.period_range(_q.iloc[0], _q.iloc[-1], freq="Q")
assert len(_q) == len(_exp) and (_q.to_numpy() == _exp.to_numpy()).all(), "gap in quarter coverage"

keep = ["firm_str", "hgroup", "rdate", "flow", "dos", "os", "cn_lag", "shock", "us"]
est[keep].to_stata(DTA, write_index=False, convert_dates={"rdate": "tc"}, version=118)

print(f"\n===== ownership_c6_panel (estimation subset, outcome=flow) =====")
print(f"  rows: {n:,}   firms: {est['firm_str'].nunique():,}   quarters: {len(_q)}"
      f"   ({_q.iloc[0]} -> {_q.iloc[-1]})")
print(f"  hgroup split:\n{est['hgroup'].value_counts().to_string()}")
print(f"  flow: mean {est['flow'].mean():.3e}  sd {est['flow'].std():.3e}"
      f"  p1 {est['flow'].quantile(.01):.3e}  p99 {est['flow'].quantile(.99):.3e}")
print(f"  (old dos for comparison: mean {est['dos'].mean():.3e}  sd {est['dos'].std():.3e})")
print(f"  wrote {DTA.name} ({n:,} rows)")
