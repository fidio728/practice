"""
build_audit_panel_f1f2f7.py — remediation panel for review findings F1, F2, F7.

Built on the SAME C6 grid (merged_us_eu_zero_filled.parquet), adding:
- F1 (timing): dw_lead1 = Delta w_{t+1} = w_{t+1} - w_t (pure lagged-shock outcome:
  regress on US x CN x S_t with NO look-ahead), and cumulative responses
  cum_h = w_{t+h} - w_{t-1} for h = 0..4 for a local-projection IRF.
- F2 (existence span): in_span = 1 iff report_date in [first_active, last_active],
  where first/last_active = min/max quarter with I_ict > 0 for the firm (either group).
  Lets the headline be re-run WITHOUT the ~25% (in_span==0 = 25.03%) pre-IPO /
  post-delisting phantom zeros.
- F7 (generated regressor / full-sample AR(1)): carry the RAW GPR level gpr_us_cn and
  its lag, so the shock can be replaced by US x CN x GPR_t + US x CN x GPR_{t-1}
  (nests every AR(1) (a,b), no generated regressor, no full-sample look-ahead).

Leads/lags are computed on the FULL contiguous grid (Cartesian, so LEAD/LAG over
report_date within (firm, group) is the true adjacent quarter), THEN the in_span flag
is attached — so span-boundary leads are not corrupted.

Output: output/audit_c6_panel.dta
"""

from pathlib import Path
import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
GRID = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
DTA = OUT / "audit_c6_panel.dta"

con = duckdb.connect()
con.execute("SET memory_limit='6GB'")

print("[1/3] Firm existence span (first/last quarter with I_ict>0)...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE span AS
SELECT sec_entity_id,
       MIN(CASE WHEN I_ict > 0 THEN report_date END) AS first_active,
       MAX(CASE WHEN I_ict > 0 THEN report_date END) AS last_active
FROM read_parquet('{GRID}')
GROUP BY 1
""")

print("[2/3] Leads/lags on the full contiguous grid + in_span flag...")
df = con.execute(f"""
WITH g AS (
    SELECT
        CAST(m.sec_entity_id AS VARCHAR)                         AS firm_str,
        m.holder_group                                          AS hgroup,
        CAST(m.report_date AS TIMESTAMP)                        AS rdate,
        m.delta_w                                               AS dw,
        m.portfolio_weight_eu                                   AS w,
        m.w_prev                                                AS w_prev,
        m.china_share_lag1q                                     AS cn_lag,
        m.shock_us_cn                                           AS shock,
        m.gpr_us_cn                                             AS gpr,
        CASE WHEN m.holder_group = 'US' THEN 1 ELSE 0 END       AS us,
        s.first_active, s.last_active,
        LEAD(m.delta_w, 1)            OVER w                     AS dw_lead1,
        LEAD(m.portfolio_weight_eu,1) OVER w                     AS w_l1,
        LEAD(m.portfolio_weight_eu,2) OVER w                     AS w_l2,
        LEAD(m.portfolio_weight_eu,3) OVER w                     AS w_l3,
        LEAD(m.portfolio_weight_eu,4) OVER w                     AS w_l4,
        LAG(m.gpr_us_cn, 1)          OVER w                      AS gpr_lag
    FROM read_parquet('{GRID}') m
    JOIN span s USING (sec_entity_id)
    WINDOW w AS (PARTITION BY m.sec_entity_id, m.holder_group ORDER BY m.report_date)
)
SELECT firm_str, hgroup, rdate, us, dw, cn_lag, shock, gpr, gpr_lag,
       dw_lead1,
       (w      - w_prev) AS cum0,   -- = dw
       (w_l1   - w_prev) AS cum1,
       (w_l2   - w_prev) AS cum2,
       (w_l3   - w_prev) AS cum3,
       (w_l4   - w_prev) AS cum4,
       CASE WHEN rdate BETWEEN first_active AND last_active THEN 1 ELSE 0 END AS in_span
FROM g
WHERE cn_lag IS NOT NULL AND shock IS NOT NULL   -- keep regressor-complete rows; outcomes may be null at edges
""").df()
con.close()

df["firm_str"] = df["firm_str"].astype(str)
df["hgroup"] = df["hgroup"].astype(str)
df["rdate"] = pd.to_datetime(df["rdate"])
for c in ["dw", "cn_lag", "shock", "gpr", "gpr_lag", "dw_lead1",
          "cum0", "cum1", "cum2", "cum3", "cum4"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
df["us"] = df["us"].astype("int8")
df["in_span"] = df["in_span"].astype("int8")

# sanity: cum0 must equal dw where both present
_chk = df.dropna(subset=["dw", "cum0"])
assert (_chk["dw"] - _chk["cum0"]).abs().max() < 1e-9, "cum0 != dw"

print("[3/3] Diagnostics + write...")
n = len(df)
print(f"  rows (regressor-complete): {n:,}")
print(f"  in_span share: {df['in_span'].mean():.4f}  (phantom/out-of-span = {1-df['in_span'].mean():.4f})")
print(f"  dw non-null: {df['dw'].notna().mean():.4f}   dw_lead1 non-null: {df['dw_lead1'].notna().mean():.4f}")
print(f"  cum4 non-null: {df['cum4'].notna().mean():.4f}")
print(f"  gpr non-null: {df['gpr'].notna().mean():.4f}  gpr_lag non-null: {df['gpr_lag'].notna().mean():.4f}")
df.to_stata(DTA, write_index=False, convert_dates={"rdate": "tc"}, version=118)
print(f"  wrote {DTA.name} ({n:,} rows)")
