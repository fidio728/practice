"""
diag_active_labels.py — FEASIBILITY GATE for the active-only four-group design
(design agenda rank 2 v2, 2026-08-02).

Three-state fund labeling per the locked v2 rule:
    PASSIVE : Funds.STYLE == 'Index'
    ACTIVE  : Funds.STYLE non-missing and != 'Index'
    UNKNOWN : STYLE missing or fund_id unmatched in the Funds master
FUND_TYPE / MANAGER_STYLE / names are cross-checks only, never overrides.
Missing is NEVER treated as active.

Outputs (per holder side US/NONUS x quarter): passive / active / unknown MV
shares of the European book (I_ict filters: EQ+AD, sec_country EU28,
sec_entity_id non-null) + matched-fund MV share. Decision numbers printed:
  - external anchors (2021Q4 US book): Index ~39.78%, NULL-entity bucket
    ~34.37% of MV with ~96% of it Index — the script must reproduce these;
  - NONUS UNKNOWN share time series -> four-group feasibility vs degraded
    (US_ACTIVE vs US_PASSIVE within-US) design (threshold ~40%);
  - post-2018 matched-share decay (Funds master ~2018-08 snapshot).
Writes output/diag_active_labels.csv.
"""
from pathlib import Path
import duckdb
import pandas as pd

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")
FUNDS = "E:/Data/Data/Factset Ownership/Factset_LionShares_Funds.gz"
EOM = (OUT / "holdings_eom.parquet").as_posix()

EU = ("GB","DE","FR","NL","CH","IT","ES","SE","DK","NO","FI","BE","AT","IE",
      "LU","PT","PL","CZ","HU","GR","RO","SK","SI","BG","HR","EE","LV","LT")
EU_SQL = "(" + ",".join(f"'{c}'" for c in EU) + ")"

con = duckdb.connect()
con.execute("SET memory_limit='6GB'")
con.execute("SET threads=4")

print("[1] Funds master: three-state label per fund ...")
con.execute(f"""
CREATE TEMP TABLE fund_label AS
SELECT FACTSET_FUND_ID AS fund_id,
       STYLE,
       CASE WHEN STYLE = 'Index' THEN 'PASSIVE'
            WHEN STYLE IS NOT NULL AND TRIM(STYLE) != '' THEN 'ACTIVE'
            ELSE 'UNKNOWN' END AS label
FROM read_csv_auto('{FUNDS}', compression='gzip')
""")
lab = con.execute("SELECT label, COUNT(*) n FROM fund_label GROUP BY 1 ORDER BY 2 DESC").df()
print(lab.to_string(index=False))
dup = con.execute("SELECT COUNT(*) - COUNT(DISTINCT fund_id) FROM fund_label").fetchone()[0]
print(f"    duplicate fund_id rows in master: {dup}")
if dup > 0:
    # deterministic dedup: PASSIVE > ACTIVE > UNKNOWN would bias; use strict
    # 'conflict -> UNKNOWN' so an ambiguous fund can never enter a clean book
    con.execute("""
    CREATE OR REPLACE TEMP TABLE fund_label AS
    SELECT fund_id,
           CASE WHEN COUNT(DISTINCT label) = 1 THEN ANY_VALUE(label)
                ELSE 'UNKNOWN' END AS label
    FROM fund_label GROUP BY fund_id
    """)
    print("    deduped: conflicting labels -> UNKNOWN")

print("[2] Join to holdings (I_ict filters), aggregate by side x quarter x label ...")
con.execute(f"""
CREATE TEMP TABLE agg AS
SELECT CASE WHEN h.investor_country = 'US' THEN 'US' ELSE 'NONUS' END AS side,
       h.report_date,
       COALESCE(f.label, 'UNKNOWN') AS label,
       (f.fund_id IS NOT NULL) AS matched,
       SUM(h.adj_mv) AS mv
FROM read_parquet('{EOM}') h
LEFT JOIN fund_label f ON h.fund_id = f.fund_id
WHERE h.sec_country IN {EU_SQL}
  AND h.sec_entity_id IS NOT NULL
  AND h.investor_country IS NOT NULL
  AND h.issue_type IN ('EQ','AD')
GROUP BY 1, 2, 3, 4
""")

df = con.execute("""
SELECT side, report_date,
       SUM(mv) AS mv_total,
       SUM(mv) FILTER (WHERE label='PASSIVE') / SUM(mv) AS passive_sh,
       SUM(mv) FILTER (WHERE label='ACTIVE')  / SUM(mv) AS active_sh,
       SUM(mv) FILTER (WHERE label='UNKNOWN') / SUM(mv) AS unknown_sh,
       SUM(mv) FILTER (WHERE matched)         / SUM(mv) AS matched_sh
FROM agg GROUP BY 1, 2 ORDER BY 1, 2
""").df()
df["report_date"] = pd.to_datetime(df["report_date"])
df.to_csv(OUT / "diag_active_labels.csv", index=False)
print(f"    wrote diag_active_labels.csv ({len(df)} side-quarter rows)")

# ---- external anchor check: 2021Q4 US book ----
print("\n[3] ANCHOR CHECK vs user-verified 2021Q4 US numbers:")
a = df[(df.side == "US") & (df.report_date == "2021-12-31")]
if len(a):
    r = a.iloc[0]
    print(f"    US 2021Q4: passive={100*r.passive_sh:.2f}% (anchor 39.78%)  "
          f"active={100*r.active_sh:.2f}%  unknown={100*r.unknown_sh:.2f}%  "
          f"matched={100*r.matched_sh:.2f}%")

# ---- decision numbers ----
print("\n[4] DECISION NUMBERS")
for side in ["US", "NONUS"]:
    s = df[df.side == side].set_index("report_date")
    for yr in ["2010-12-31", "2015-12-31", "2018-12-31", "2021-12-31", "2023-12-31"]:
        if pd.Timestamp(yr) in s.index:
            r = s.loc[pd.Timestamp(yr)]
            print(f"    {side:5s} {yr[:4]}: passive {100*r.passive_sh:5.1f}%  "
                  f"active {100*r.active_sh:5.1f}%  UNKNOWN {100*r.unknown_sh:5.1f}%  "
                  f"matched {100*r.matched_sh:5.1f}%")
post18 = df[(df.side == "NONUS") & (df.report_date >= "2018-01-01")]
print(f"\n    NONUS UNKNOWN share, 2018+ : mean {100*post18.unknown_sh.mean():.1f}%  "
      f"max {100*post18.unknown_sh.max():.1f}%")
print("    RULE: NONUS unknown >~40% -> degrade to within-US ACTIVE vs PASSIVE design")
