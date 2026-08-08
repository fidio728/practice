"""
build_audit_panel_f1f2f7.py — remediation panel for review findings F1, F2, F7.

Built on the SAME C6 grid (merged_us_eu_zero_filled.parquet), adding:
- F1 (timing): dw_lead1 = Delta w_{t+1} = w_{t+1} - w_t (pure lagged-shock outcome:
  regress on US x CN x S_t with NO look-ahead), and cumulative responses
  cum_h = w_{t+h} - w_{t-1} for h = 0..4 for a local-projection IRF.

GLOBAL-MAIN (2026-08-08): the MAIN outcome family (dw, dw_lead1, cum0..cum4)
is now built from the FULL-portfolio (global-denominator) weights
portfolio_weight_global / w_prev_global / delta_w_global, per
research_plan.tex "Country portfolio weight". The EU-restricted Δw is kept as
dw_eu (within-Europe reallocation diagnostic; NaN where the EU book is empty
but the global book is not). s_lag = shock_us_cn_lag1q (S_{t-1}) is carried as
the PRIMARY timing regressor alongside shock (S_t, labeled diagnostic), so the
headline .do can run the 2x2 {global, EU} x {S_{t-1}, S_t} from this one panel.
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

import os
from pathlib import Path
import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
# (EM-FIX-5/7, 2026-08-06) same DPN_OUT_DIR override as 00_setup.jl / build_c6_panel.py.
_env_out = os.environ.get("DPN_OUT_DIR", "").strip()
OUT = Path(_env_out).resolve() if _env_out else PROJ / "output"
OUT.mkdir(parents=True, exist_ok=True)
if _env_out:
    print(f"[DPN_OUT_DIR] reading/writing {OUT} (override in force)")
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
        -- (GLOBAL-MAIN, 2026-08-08) MAIN outcome family = global denominator;
        -- EU Δw kept as the labeled within-Europe diagnostic.
        m.delta_w_global                                        AS dw,
        m.delta_w                                               AS dw_eu,
        m.portfolio_weight_global                               AS w,
        m.w_prev_global                                         AS w_prev,
        m.china_share_lag1q                                     AS cn_lag,
        m.shock_us_cn                                           AS shock,
        -- (S_{t-1} PRIMARY, 2026-08-08) lagged shock, true quarter lag on the
        -- contiguous grid (computed in 06).
        m.shock_us_cn_lag1q                                     AS s_lag,
        m.gpr_us_cn                                             AS gpr,
        CASE WHEN m.holder_group = 'US' THEN 1 ELSE 0 END       AS us,
        -- (EM-FIX-6, 2026-08-06) attribution columns — see build_c6_panel.py.
        -- zr_lag == 0 is the pre-change codable set, so run A of the snapshot-vs-
        -- sample-expansion attribution is `if zr_lag==0` on THIS panel too
        -- (07d_three_spec_table.do / run_headline_3pairwise.do / run_ri_3pairwise.py
        -- all read audit_c6_panel). NULL -> sentinel -1.
        COALESCE(m.zero_recode_flag_lag1q,   -1)                AS zr_lag,
        COALESCE(m.n_supplychain_links_lag1q, -1)               AS nsc_lag,
        COALESCE(CAST(m.revere_pit_present_lag1q AS INTEGER), -1) AS pit_lag,
        s.first_active, s.last_active,
        -- (GLOBAL-MAIN) leads/cums on the GLOBAL family, matching dw.
        LEAD(m.delta_w_global, 1)         OVER w                 AS dw_lead1,
        LEAD(m.portfolio_weight_global,1) OVER w                 AS w_l1,
        LEAD(m.portfolio_weight_global,2) OVER w                 AS w_l2,
        LEAD(m.portfolio_weight_global,3) OVER w                 AS w_l3,
        LEAD(m.portfolio_weight_global,4) OVER w                 AS w_l4,
        LAG(m.gpr_us_cn, 1)          OVER w                      AS gpr_lag
    FROM read_parquet('{GRID}') m
    JOIN span s USING (sec_entity_id)
    WINDOW w AS (PARTITION BY m.sec_entity_id, m.holder_group ORDER BY m.report_date)
)
SELECT firm_str, hgroup, rdate, us, dw, dw_eu, cn_lag, shock, s_lag, gpr, gpr_lag,
       zr_lag, nsc_lag, pit_lag,
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
for c in ["dw", "dw_eu", "cn_lag", "shock", "s_lag", "gpr", "gpr_lag", "dw_lead1",
          "cum0", "cum1", "cum2", "cum3", "cum4"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
df["us"] = df["us"].astype("int8")
df["in_span"] = df["in_span"].astype("int8")
# (EM-FIX-6) attribution columns: int8 flags, int32 link count, -1 = NULL upstream.
df["zr_lag"] = pd.to_numeric(df["zr_lag"], errors="raise").astype("int8")
df["pit_lag"] = pd.to_numeric(df["pit_lag"], errors="raise").astype("int8")
df["nsc_lag"] = pd.to_numeric(df["nsc_lag"], errors="raise").astype("int32")

# sanity: cum0 must equal dw where both present
_chk = df.dropna(subset=["dw", "cum0"])
assert (_chk["dw"] - _chk["cum0"]).abs().max() < 1e-9, "cum0 != dw"

print("[3/3] Diagnostics + write...")
n = len(df)
print(f"  rows (regressor-complete): {n:,}")
print(f"  in_span share: {df['in_span'].mean():.4f}  (phantom/out-of-span = {1-df['in_span'].mean():.4f})")
print(f"  dw (global MAIN) non-null: {df['dw'].notna().mean():.4f}   dw_lead1 non-null: {df['dw_lead1'].notna().mean():.4f}")
print(f"  dw_eu (EU diag) non-null:  {df['dw_eu'].notna().mean():.4f}   "
      f"global-only rows (dw ok, dw_eu NaN): {int((df['dw'].notna() & df['dw_eu'].isna()).sum()):,}")
print(f"  s_lag (S_t-1 PRIMARY) non-null: {df['s_lag'].notna().mean():.4f}")
# NULL-semantics direction: dw NULL where dw_eu defined is impossible.
assert int((df["dw"].isna() & df["dw_eu"].notna()).sum()) == 0, \
    "dw (global) NaN on rows where dw_eu defined — impossible under T_global >= T_eu"
print(f"  cum4 non-null: {df['cum4'].notna().mean():.4f}")
print(f"  gpr non-null: {df['gpr'].notna().mean():.4f}  gpr_lag non-null: {df['gpr_lag'].notna().mean():.4f}")
print("  EM-FIX-6 attribution census (zr_lag):")
for _k, _v in df["zr_lag"].value_counts().sort_index().items():
    _lbl = {0: "old-rule codable (pre-change arm)",
            1: "recoded zero: competitor/partner only",
            2: "recoded zero: no active links",
            -1: "NULL upstream (INVESTIGATE)"}.get(int(_k), "unexpected")
    print(f"    zr_lag={int(_k):>3}  n={_v:>12,}  firms={df.loc[df['zr_lag'] == _k, 'firm_str'].nunique():>7,}  {_lbl}")
print(f"  run A (`if zr_lag==0`): N={int((df['zr_lag'] == 0).sum()):,}   "
      f"run B (full): N={len(df):,}")
df.to_stata(DTA, write_index=False, convert_dates={"rdate": "tc"}, version=118)
print(f"  wrote {DTA.name} ({n:,} rows)")
