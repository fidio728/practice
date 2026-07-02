"""
build_russia_lp_panel.py — extend the Russia positive control with a
local-projection cumulative response, since the 2022 sanctions regime escalated
over MANY quarters (not a single-quarter shock). A tight 2-quarter dummy could
be underpowered by construction even if the design has real power to detect the
divestment. Mirrors build_audit_panel_f1f2f7.py's cum_h logic (h=0..8, i.e.
through end of 2023) but anchored on Q4-2021 (the quarter before the invasion)
regardless of firm-level shock timing, since this is an EVENT study around a
known calendar date, not a continuous-shock design.

cum_h = w_{2022Q1 + h} - w_{2021Q4}, for h = 0..7 (covers 2022Q1 through 2023Q4).
Outcome: (US - NONUS) cumulative Delta-w from the quarter before the invasion.
"""
from pathlib import Path
import duckdb
import pandas as pd

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")
GRID = (OUT / "merged_us_ru_zero_filled.parquet").as_posix()

con = duckdb.connect()
con.execute("SET memory_limit='6GB'")

df = con.execute(f"""
WITH g AS (
    SELECT sec_entity_id, holder_group, report_date, portfolio_weight_eu AS w,
           russia_share_lag1q AS ru_lag,
           CASE WHEN holder_group='US' THEN 1 ELSE 0 END AS us
    FROM read_parquet('{GRID}')
),
base AS (
    SELECT sec_entity_id, holder_group, w AS w_base
    FROM g WHERE report_date = DATE '2021-12-31'
),
joined AS (
    SELECT g.sec_entity_id, g.holder_group, g.report_date, g.us, g.ru_lag,
           g.w - b.w_base AS cum_dw
    FROM g
    JOIN base b USING (sec_entity_id, holder_group)
    WHERE g.report_date BETWEEN DATE '2022-03-31' AND DATE '2023-12-31'
)
SELECT * FROM joined
""").df()
con.close()

df["report_date"] = pd.to_datetime(df["report_date"])
n0 = len(df)
df = df.dropna(subset=["cum_dw", "ru_lag"])
print(f"rows: {n0:,} -> {len(df):,} after dropna")
df.to_parquet(OUT / "russia_lp_panel.parquet", index=False)
print("wrote russia_lp_panel.parquet")
print(df["report_date"].value_counts().sort_index().to_string())
