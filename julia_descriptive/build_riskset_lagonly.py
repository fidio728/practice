"""
build_riskset_lagonly.py — review fix F8.

The main risk-set (build_spell_riskset.py) defines firm-quarter membership as
held at t OR t-1 OR t+1 (LEAD). The t+1 lead conditions membership on a
POST-treatment outcome — the same family of concern as the retracted centered
diff. This builds a LAG-ONLY variant (held at t OR t-1, no look-ahead) so β₃ can
be checked for sensitivity. Everything else matches build_spell_riskset.py.

Output: output/c6_panel_riskset_lagonly.dta
"""
from pathlib import Path
import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
uri = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
DTA = OUT / "c6_panel_riskset_lagonly.dta"

con = duckdb.connect()
con.execute(f"""
CREATE OR REPLACE TEMP TABLE flagged AS
SELECT sec_entity_id, holder_group, report_date, I_ict, delta_w,
       china_share_lag1q, shock_us_cn,
       CAST(I_ict > 0 AS INTEGER) AS held,
       COALESCE(LAG(CAST(I_ict > 0 AS INTEGER)) OVER w, 0) AS held_lag
FROM read_parquet('{uri}')
WINDOW w AS (PARTITION BY sec_entity_id, holder_group ORDER BY report_date)
""")
con.execute("""
CREATE OR REPLACE TEMP TABLE riskset AS
SELECT *,
       MAX(CASE WHEN held=1 OR held_lag=1 THEN 1 ELSE 0 END)
           OVER (PARTITION BY sec_entity_id, report_date) AS risk
FROM flagged
""")
df = con.execute("""
SELECT CAST(sec_entity_id AS VARCHAR) AS firm_str,
       holder_group AS hgroup,
       CAST(report_date AS TIMESTAMP) AS rdate,
       delta_w AS dw, china_share_lag1q AS cn_lag, shock_us_cn AS shock,
       CASE WHEN holder_group='US' THEN 1 ELSE 0 END AS us
FROM riskset WHERE risk = 1
""").df()
con.close()

est = df.dropna(subset=["dw", "cn_lag", "shock"]).copy()
est["firm_str"] = est["firm_str"].astype(str)
est["hgroup"] = est["hgroup"].astype(str)
est["rdate"] = pd.to_datetime(est["rdate"])
for c in ["dw", "cn_lag", "shock"]:
    est[c] = pd.to_numeric(est[c], errors="raise").astype("float64")
est["us"] = est["us"].astype("int8")

# pairing assert (lag-only membership is still firm-quarter -> both groups kept)
_p = est.groupby(["firm_str", "rdate"])["hgroup"].nunique()
assert (_p == 2).all(), f"not fully paired: {(_p != 2).sum():,}"
assert not est.duplicated(["firm_str", "hgroup", "rdate"]).any()

keep = ["firm_str", "hgroup", "rdate", "dw", "cn_lag", "shock", "us"]
est[keep].to_stata(DTA, write_index=False, convert_dates={"rdate": "tc"}, version=118)
print(f"lag-only risk set: {len(est):,} rows, {est['firm_str'].nunique():,} firms, "
      f"{est.groupby(['firm_str','rdate']).ngroups:,} firm-quarters")
print(f"  (compare with-lead risk set c6_panel_riskset.dta = 268,282 rows; B7 2026-08-03)")
print(f"  wrote {DTA.name}")
