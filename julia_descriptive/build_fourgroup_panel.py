"""
build_fourgroup_panel.py

Essay-2 active-only FOUR-GROUP panel builder (independent of 04/06 main pipeline).

Design (locked, do NOT change):
  * Fund labels from Factset_LionShares_Funds master, on STYLE:
        PASSIVE  = STYLE == 'Index'
        ACTIVE   = STYLE non-missing AND != 'Index'
        UNKNOWN  = STYLE missing/empty  OR  fund_id unmatched to master
    UNKNOWN is NEVER treated as active; UNKNOWN is excluded from the four
    ACTIVE/PASSIVE books but IS aggregated for the reconciliation identity.
  * Four groups = side x label, side = ('US' if investor_country=='US' else
    'NONUS'):  US_ACTIVE, US_PASSIVE, NONUS_ACTIVE, NONUS_PASSIVE.
  * Each book self-normalises:  w_g(firm,t) = I_g(firm,t) / SUM_{EU firms} I_g(t).
    The EU firm universe = the merged parquet's sec_entity set (06 universe),
    so summing I_g over universe firms == the full-EU book total by construction.
  * I_ict filter (locked): sec_entity_id non-null AND investor_country non-null
    AND issue_type IN ('EQ','AD).  I = SUM(adj_mv).  (Identical to 04's I_ict.)
  * Exposure (cn_lag = china_share_lag1q) and shock (shock_us_cn) are lifted
    DISTINCT from merged_us_eu_zero_filled.parquet; both are group-invariant
    (cn_lag per firm-quarter, shock per quarter) and that invariance is ASSERTED
    before extraction.
  * Backward delta_w: dw = w_t - w_{t-1}, per (sec_entity_id, group), computed on
    the FULL zero-filled quarter grid, then the output is filtered.

Outputs:
  output/fund_label.parquet        (fund_id -> label, one row per master fund)
  output/fourgroup_panel.dta       (Stata-ready: firm_str, grp, rdate, dw, w,
                                     cn_lag, shock, active, us)
  output/fourgroup_matched_share.csv   (US/NONUS matched-share decay diagnostic)
  output/fourgroup_build_diag.csv      (reconciliation + weight-sum + comparability)

Reconciliation identities hard-checked here:
  (A) per (side, quarter):  SUM_EU (I_ACTIVE + I_PASSIVE + I_UNKNOWN)
        == pooled side aggregate (independently computed from raw holdings,
        NO label join -> catches any join fan-out).
  (B) each of the 4 books, per quarter, weights sum to 1 (non-empty books);
      no cell with positive holdings but NULL weight.
  (C) comparability with pooled c6_panel firm/quarter dimensions (reported).

Stata gotcha: pandas.to_stata writes datetime64 as %tc; 07-style .do does
`gen rd_day = dofc(rdate)`.
"""

from pathlib import Path

import duckdb
import pandas as pd

PROJ_DIR = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT_DIR  = PROJ_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ inputs ---
EOM_PARQUET   = OUT_DIR / "holdings_eom.parquet"
MERGED_PARQUET = OUT_DIR / "merged_us_eu_zero_filled.parquet"   # 06 universe/exposure/shock
FUNDS_GZ      = Path(r"E:/Data/Data/Factset Ownership/Factset_LionShares_Funds.gz")
C6_PANEL_DTA  = OUT_DIR / "c6_panel.dta"                        # optional, for comparability

# ------------------------------------------------------------------ outputs --
FUND_LABEL_PARQUET = OUT_DIR / "fund_label.parquet"
PANEL_DTA          = OUT_DIR / "fourgroup_panel.dta"
MATCHED_SHARE_CSV  = OUT_DIR / "fourgroup_matched_share.csv"
BUILD_DIAG_CSV     = OUT_DIR / "fourgroup_build_diag.csv"

for p in (EOM_PARQUET, MERGED_PARQUET, FUNDS_GZ):
    assert p.exists(), f"Required input missing: {p}"

eom_uri    = str(EOM_PARQUET).replace("\\", "/")
merged_uri = str(MERGED_PARQUET).replace("\\", "/")
funds_uri  = str(FUNDS_GZ).replace("\\", "/")

# ------------------------------------------------------------------ connect --
con = duckdb.connect()
con.execute("SET memory_limit='6GB'")
con.execute("SET threads=4")
con.execute("SET preserve_insertion_order=false")

print("=" * 72)
print("build_fourgroup_panel.py  —  Essay-2 four-group (side x label) panel")
print("=" * 72)

# ===========================================================================
# (1) Fund labels from the master (three states) -> fund_label.parquet
# ===========================================================================
print("\n[1] Labelling funds from master (PASSIVE=Index, ACTIVE=other, UNKNOWN=missing)")
con.execute(f"""
    CREATE OR REPLACE TABLE fund_label AS
    SELECT
        FACTSET_FUND_ID AS fund_id,
        CASE
            WHEN NULLIF(TRIM(STYLE), '') = 'Index'          THEN 'PASSIVE'
            WHEN NULLIF(TRIM(STYLE), '') IS NOT NULL         THEN 'ACTIVE'
            ELSE 'UNKNOWN'
        END AS label
    FROM read_csv_auto('{funds_uri}', compression='gzip')
""")
# Locked assertion: master has no duplicate fund_id.
_lab = con.execute("""
    SELECT COUNT(*) AS n, COUNT(DISTINCT fund_id) AS u,
           COUNT(*) FILTER (WHERE label='ACTIVE')  AS n_active,
           COUNT(*) FILTER (WHERE label='PASSIVE') AS n_passive,
           COUNT(*) FILTER (WHERE label='UNKNOWN') AS n_unknown
    FROM fund_label
""").df()
_n, _u = int(_lab["n"].iloc[0]), int(_lab["u"].iloc[0])
assert _n == _u, f"Funds master not unique on fund_id ({_n} rows, {_u} unique)"
print(f"    funds: {_n:,} unique fund_id  |  ACTIVE={_lab['n_active'].iloc[0]:,} "
      f"PASSIVE={_lab['n_passive'].iloc[0]:,} UNKNOWN={_lab['n_unknown'].iloc[0]:,}")

# atomic-ish write: tmp then replace
_fl_tmp = str(FUND_LABEL_PARQUET).replace("\\", "/") + ".tmp"
con.execute(f"COPY (SELECT * FROM fund_label) TO '{_fl_tmp}' (FORMAT 'parquet', COMPRESSION 'zstd')")
Path(_fl_tmp.replace("/", "\\")).replace(FUND_LABEL_PARQUET)
print(f"    -> {FUND_LABEL_PARQUET.name}")

# ===========================================================================
# (2) Firm universe (= merged parquet sec_entity set) + quarter calendar
# ===========================================================================
print("\n[2] Firm universe + quarter calendar from merged parquet (06 universe)")
con.execute(f"""
    CREATE OR REPLACE TABLE firm_universe AS
    SELECT DISTINCT sec_entity_id FROM read_parquet('{merged_uri}')
    WHERE sec_entity_id IS NOT NULL
""")
con.execute(f"""
    CREATE OR REPLACE TABLE quarters AS
    SELECT DISTINCT report_date AS rdate FROM read_parquet('{merged_uri}')
""")
_nfirm = int(con.execute("SELECT COUNT(*) n FROM firm_universe").df()["n"].iloc[0])
_nq    = int(con.execute("SELECT COUNT(*) n FROM quarters").df()["n"].iloc[0])
print(f"    firm universe: {_nfirm:,} sec_entity_id  |  quarters: {_nq}")

# ===========================================================================
# (3) Aggregate I to (sec_entity_id, side, label, quarter) over universe firms.
#     LEFT JOIN fund_label (unique fund_id -> no fan-out); unmatched -> UNKNOWN.
# ===========================================================================
print("\n[3] Aggregating I = SUM(adj_mv) by (sec_entity_id, side, label, quarter)")
print("    (locked filter: sec_entity_id & investor_country non-null, issue_type in EQ/AD)")
con.execute(f"""
    CREATE OR REPLACE TABLE grp_agg AS
    SELECT
        h.sec_entity_id,
        CASE WHEN h.investor_country = 'US' THEN 'US' ELSE 'NONUS' END AS side,
        COALESCE(fl.label, 'UNKNOWN') AS label,
        -- in_master splits the UNKNOWN bucket into 'matched-but-no-STYLE'
        -- (fl.fund_id present) vs 'unmatched' (fund_id absent from master),
        -- so the matched-share diagnostic can report the true join rate.
        (fl.fund_id IS NOT NULL) AS in_master,
        h.report_date AS rdate,
        SUM(CAST(h.adj_mv AS DOUBLE)) AS I
    FROM read_parquet('{eom_uri}') h
    LEFT JOIN fund_label fl ON h.fund_id = fl.fund_id
    WHERE h.sec_entity_id IS NOT NULL
      AND h.investor_country IS NOT NULL
      AND h.issue_type IN ('EQ', 'AD')
      AND h.sec_entity_id IN (SELECT sec_entity_id FROM firm_universe)
    GROUP BY 1, 2, 3, 4, 5
""")
_ga = con.execute("""
    SELECT COUNT(*) n,
           COUNT(*) FILTER (WHERE label='ACTIVE')  n_a,
           COUNT(*) FILTER (WHERE label='PASSIVE') n_p,
           COUNT(*) FILTER (WHERE label='UNKNOWN') n_u
    FROM grp_agg
""").df()
print(f"    grp_agg cells: {int(_ga['n'].iloc[0]):,} "
      f"(ACTIVE={int(_ga['n_a'].iloc[0]):,}, PASSIVE={int(_ga['n_p'].iloc[0]):,}, "
      f"UNKNOWN={int(_ga['n_u'].iloc[0]):,})")

# ===========================================================================
# (4) RECONCILIATION (A): sum of the three label books == independent pooled.
#     Pooled is recomputed from raw holdings WITHOUT the label join, so any
#     fan-out from a non-unique join would show up as a mismatch.
# ===========================================================================
print("\n[4] Reconciliation (A): SUM_EU(I_ACT+I_PAS+I_UNK) == pooled side (independent)")
con.execute(f"""
    CREATE OR REPLACE TABLE pooled_side AS
    SELECT
        CASE WHEN h.investor_country = 'US' THEN 'US' ELSE 'NONUS' END AS side,
        h.report_date AS rdate,
        SUM(CAST(h.adj_mv AS DOUBLE)) AS I_pooled
    FROM read_parquet('{eom_uri}') h
    WHERE h.sec_entity_id IS NOT NULL
      AND h.investor_country IS NOT NULL
      AND h.issue_type IN ('EQ', 'AD')
      AND h.sec_entity_id IN (SELECT sec_entity_id FROM firm_universe)
    GROUP BY 1, 2
""")
_recon = con.execute("""
    WITH lab AS (
        SELECT side, rdate, SUM(I) AS I_lab
        FROM grp_agg GROUP BY 1, 2)
    SELECT
        COUNT(*) AS n_cells,
        MAX(ABS(COALESCE(l.I_lab,0) - COALESCE(p.I_pooled,0))) AS max_abs_dev,
        MAX(ABS(COALESCE(l.I_lab,0) - COALESCE(p.I_pooled,0))
            / NULLIF(p.I_pooled,0)) AS max_rel_dev,
        COUNT(*) FILTER (WHERE l.side IS NULL OR p.side IS NULL) AS n_unmatched
    FROM lab l FULL OUTER JOIN pooled_side p USING (side, rdate)
""").df()
_dev = float(_recon["max_abs_dev"].iloc[0])
_rel = float(_recon["max_rel_dev"].iloc[0] or 0.0)
_unm = int(_recon["n_unmatched"].iloc[0])
print(f"    cells={int(_recon['n_cells'].iloc[0]):,}  max_abs_dev={_dev:.3e}  "
      f"max_rel_dev={_rel:.3e}  unmatched_side_qtr={_unm}")
assert _unm == 0, "reconciliation (A): a (side,quarter) exists in one aggregate but not the other"
# absolute dev can be a few units of float on ~1e15 dollar sums -> judge on relative.
assert _rel < 1e-9, f"reconciliation (A) FAILED: label partition != pooled (rel dev {_rel:.3e})"
print("    (A) PASS: label partition reproduces pooled side exactly (no join fan-out)")

# ===========================================================================
# (5) Book totals T_{g,t} (four ACTIVE/PASSIVE books only) + zero-fill grid.
# ===========================================================================
print("\n[5] Book totals + zero-filled grid (firm x 4 groups x quarter)")
con.execute("""
    CREATE OR REPLACE TABLE book_tot AS
    SELECT side, label, rdate, SUM(I) AS T
    FROM grp_agg
    WHERE label IN ('ACTIVE', 'PASSIVE')
    GROUP BY 1, 2, 3
""")
# The 4 book groups with their flags.
con.execute("""
    CREATE OR REPLACE TABLE groups4 AS
    SELECT * FROM (VALUES
        ('US',    'ACTIVE',  1, 1),
        ('US',    'PASSIVE', 0, 1),
        ('NONUS', 'ACTIVE',  1, 0),
        ('NONUS', 'PASSIVE', 0, 0)
    ) AS t(side, label, active, us)
""")
con.execute("""
    CREATE OR REPLACE TABLE grid AS
    SELECT u.sec_entity_id, g.side, g.label, g.active, g.us, q.rdate,
           COALESCE(a.I, 0.0) AS I,
           bt.T AS T,
           CASE WHEN bt.T > 0 THEN COALESCE(a.I, 0.0) / bt.T ELSE NULL END AS w
    FROM firm_universe u
    CROSS JOIN groups4 g
    CROSS JOIN quarters q
    LEFT JOIN grp_agg a
           ON a.sec_entity_id = u.sec_entity_id
          AND a.side  = g.side
          AND a.label = g.label
          AND a.rdate = q.rdate
    LEFT JOIN book_tot bt
           ON bt.side  = g.side
          AND bt.label = g.label
          AND bt.rdate = q.rdate
""")
_ng = int(con.execute("SELECT COUNT(*) n FROM grid").df()["n"].iloc[0])
print(f"    zero-filled grid rows: {_ng:,}  (= {_nfirm:,} firms x 4 x {_nq} quarters)")
assert _ng == _nfirm * 4 * _nq, "grid row count != firms*4*quarters (Cartesian build bug)"

# ---- RECONCILIATION (B): weights sum to 1 per (group, quarter); NULL-aware ----
_wchk = con.execute("""
    WITH cell AS (
        SELECT side, label, rdate,
               SUM(w)               AS s,
               COUNT(w)             AS nn,
               SUM(I)               AS tot_hold
        FROM grid GROUP BY 1, 2, 3)
    SELECT
        MAX(CASE WHEN nn > 0 THEN ABS(s - 1) END)           AS max_dev,
        COUNT(CASE WHEN nn > 0 THEN 1 END)                  AS n_checked,
        COUNT(CASE WHEN nn = 0 AND tot_hold > 0 THEN 1 END) AS n_held_but_null
    FROM cell
""").df()
_wdev = float(_wchk["max_dev"].iloc[0])
_nbug = int(_wchk["n_held_but_null"].iloc[0])
print(f"    (B) weight-sum: max|sum-1|={_wdev:.2e} over "
      f"{int(_wchk['n_checked'].iloc[0]):,} non-empty (group,quarter) books; "
      f"held-but-null cells={_nbug}")
assert _nbug == 0, f"{_nbug} (group,quarter) cell(s) have positive holdings but all-NULL weights"
assert _wdev < 1e-9, f"(B) FAILED: book weights do not sum to 1 (max dev {_wdev:.2e})"

# ===========================================================================
# (6) Exposure (cn_lag) + shock from merged parquet — verify invariance first.
# ===========================================================================
print("\n[6] Extracting cn_lag + shock from merged (after invariance asserts)")
# cn_lag group-invariance: at most one distinct value per (sec_entity, rdate).
_inv = con.execute(f"""
    SELECT COUNT(*) FILTER (WHERE nd > 1) AS n_variant
    FROM (
        SELECT sec_entity_id, report_date, COUNT(DISTINCT china_share_lag1q) AS nd
        FROM read_parquet('{merged_uri}')
        WHERE china_share_lag1q IS NOT NULL
        GROUP BY 1, 2)
""").df()
assert int(_inv["n_variant"].iloc[0]) == 0, \
    "cn_lag varies across holder_group within (sec_entity,rdate) — not group-invariant"
# shock uniqueness per quarter.
_sinv = con.execute(f"""
    SELECT COUNT(*) FILTER (WHERE nd > 1) AS n_variant
    FROM (SELECT report_date, COUNT(DISTINCT shock_us_cn) AS nd
          FROM read_parquet('{merged_uri}') GROUP BY 1)
""").df()
assert int(_sinv["n_variant"].iloc[0]) == 0, "shock_us_cn varies within a quarter in merged"

con.execute(f"""
    CREATE OR REPLACE TABLE exposure_map AS
    SELECT DISTINCT sec_entity_id, report_date AS rdate, china_share_lag1q AS cn_lag
    FROM read_parquet('{merged_uri}')
    WHERE china_share_lag1q IS NOT NULL
""")
con.execute(f"""
    CREATE OR REPLACE TABLE shock_map AS
    SELECT DISTINCT report_date AS rdate, shock_us_cn AS shock
    FROM read_parquet('{merged_uri}')
    WHERE shock_us_cn IS NOT NULL
""")
# guard: extraction stayed unique
_em_u = con.execute("SELECT COUNT(*) n, COUNT(DISTINCT (sec_entity_id, rdate)) u FROM exposure_map").df()
assert int(_em_u["n"].iloc[0]) == int(_em_u["u"].iloc[0]), "exposure_map not unique on (sec_entity,rdate)"
_sm_u = con.execute("SELECT COUNT(*) n, COUNT(DISTINCT rdate) u FROM shock_map").df()
assert int(_sm_u["n"].iloc[0]) == int(_sm_u["u"].iloc[0]), "shock_map not unique on rdate"
print(f"    exposure_map (sec,rdate,cn_lag): {int(_em_u['n'].iloc[0]):,} rows  |  "
      f"shock_map (rdate,shock): {int(_sm_u['n'].iloc[0]):,} rows")

# ===========================================================================
# (7) Backward delta_w on the FULL grid (per firm x group), then attach
#     exposure + shock.  grp = side_label.
# ===========================================================================
print("\n[7] Backward delta_w per (sec_entity, group) + join exposure/shock")
con.execute("""
    CREATE OR REPLACE TABLE panel_full AS
    SELECT
        CAST(g.sec_entity_id AS VARCHAR)          AS firm_str,
        g.side || '_' || g.label                  AS grp,
        g.rdate,
        (g.w - LAG(g.w) OVER (PARTITION BY g.sec_entity_id, g.side, g.label
                              ORDER BY g.rdate))  AS dw,
        g.w                                        AS w,
        e.cn_lag                                   AS cn_lag,
        s.shock                                    AS shock,
        g.active                                   AS active,
        g.us                                       AS us
    FROM grid g
    LEFT JOIN exposure_map e
           ON e.sec_entity_id = g.sec_entity_id AND e.rdate = g.rdate
    LEFT JOIN shock_map s ON s.rdate = g.rdate
""")

# ===========================================================================
# (8) Output filter (group-invariant only) + hard asserts + write .dta.
#     Filter on cn_lag & shock (both group-invariant) so the 4-group balance
#     is preserved.  dw keeps its natural first-quarter NULL (Stata drops it).
# ===========================================================================
print("\n[8] Filtering estimation sample (cn_lag & shock non-null) + asserts")
df = con.execute("""
    SELECT firm_str, grp,
           CAST(rdate AS TIMESTAMP) AS rdate,
           dw, w, cn_lag, shock, active, us
    FROM panel_full
    WHERE cn_lag IS NOT NULL
      AND shock  IS NOT NULL
    ORDER BY firm_str, rdate, grp
""").df()

# ---- comparability with pooled c6_panel (report only) ----
c6_firms = c6_quarters = None
if C6_PANEL_DTA.exists():
    try:
        _c6 = pd.read_stata(C6_PANEL_DTA, columns=["firm_str", "rdate"])
        c6_firms = _c6["firm_str"].nunique()
        c6_quarters = _c6["rdate"].dt.to_period("Q").nunique()
    except Exception as _e:  # noqa: BLE001
        print(f"    (comparability) could not read c6_panel.dta: {_e}")

# type coercions
df["firm_str"] = df["firm_str"].astype(str)
df["grp"]      = df["grp"].astype(str)
df["rdate"]    = pd.to_datetime(df["rdate"])
df["dw"]       = pd.to_numeric(df["dw"],     errors="raise").astype("float64")
df["w"]        = pd.to_numeric(df["w"],      errors="raise").astype("float64")
df["cn_lag"]   = pd.to_numeric(df["cn_lag"], errors="raise").astype("float64")
df["shock"]    = pd.to_numeric(df["shock"],  errors="raise").astype("float64")
df["active"]   = df["active"].astype("int8")
df["us"]       = df["us"].astype("int8")

n = len(df)
print(f"    output rows: {n:,}  |  firms={df['firm_str'].nunique():,}  "
      f"grp={sorted(df['grp'].unique())}")
print(f"    rdate range: {df['rdate'].min()} -> {df['rdate'].max()}")

# --- hard assertions on the emitted panel ---
assert n > 0, "empty panel after filter"
_valid_grp = {"US_ACTIVE", "US_PASSIVE", "NONUS_ACTIVE", "NONUS_PASSIVE"}
assert set(df["grp"].unique()) == _valid_grp, f"unexpected grp values: {df['grp'].unique()}"
# flags consistent with grp
assert (df["us"]     == df["grp"].str.startswith("US_").astype("int8")).all(), "us flag != grp side"
assert (df["active"] == df["grp"].str.endswith("_ACTIVE").astype("int8")).all(), "active flag != grp label"
# unique key (firm, grp, quarter)
assert not df.duplicated(["firm_str", "grp", "rdate"]).any(), "duplicate (firm_str, grp, rdate)"
# GROUPS COMPLETE: every (firm, quarter) carries all four groups
_grpcount = df.groupby(["firm_str", "rdate"])["grp"].nunique()
assert (_grpcount == 4).all(), \
    f"groups not complete: {(_grpcount != 4).sum():,} firm-quarters lack all 4 groups"
# shock unique within quarter
assert (df.groupby("rdate")["shock"].nunique() == 1).all(), "shock varies within a quarter"
# cn_lag in [0,1]
assert df["cn_lag"].between(0, 1).all(), "cn_lag outside [0,1]"
# QUARTERS CONTIGUOUS (panel-wide distinct quarters)
_q = df["rdate"].dt.to_period("Q").drop_duplicates().sort_values()
_expected = pd.period_range(_q.iloc[0], _q.iloc[-1], freq="Q")
assert len(_q) == len(_expected) and (_q.to_numpy() == _expected.to_numpy()).all(), \
    f"gap in quarter coverage: {len(_q)} distinct vs {len(_expected)} expected"
print(f"    quarter coverage: {len(_q)} contiguous quarters {_q.iloc[0]} -> {_q.iloc[-1]}")
print(f"    every firm-quarter carries all 4 groups: OK")

# ===========================================================================
# (9) Diagnostics: matched-share decay (US & NONUS) + build-diag CSV.
# ===========================================================================
print("\n[9] Diagnostics: matched-share decay + comparability")
matched = con.execute("""
    SELECT side,
           EXTRACT(YEAR FROM rdate) AS yr,
           SUM(I) AS I_total,
           -- (a) master JOIN rate: fund_id found in the funds master at all.
           SUM(I) FILTER (WHERE in_master)
               / NULLIF(SUM(I), 0) AS join_match_share,
           -- (b) usable-LABEL share: matched AND STYLE present (ACTIVE/PASSIVE).
           SUM(I) FILTER (WHERE label IN ('ACTIVE','PASSIVE'))
               / NULLIF(SUM(I), 0) AS labeled_share
    FROM grp_agg
    GROUP BY 1, 2
    ORDER BY 1, 2
""").df()
matched.to_csv(MATCHED_SHARE_CSV, index=False)
# The design's "matched share ~95->90% decay" is the JOIN rate (a), which
# decays post-2018 as funds born after the 2018 master snapshot go unmatched.
_us_ms = matched[matched["side"] == "US"].sort_values("yr")
if len(_us_ms) > 0:
    _peak = _us_ms.loc[_us_ms["join_match_share"].idxmax()]
    print(f"    US master-JOIN share: peak {_peak['yr']:.0f}={_peak['join_match_share']:.3f} "
          f"-> {_us_ms['yr'].iloc[-1]:.0f}={_us_ms['join_match_share'].iloc[-1]:.3f} "
          f"(post-snapshot decay); usable-LABEL share {_us_ms['yr'].iloc[-1]:.0f}="
          f"{_us_ms['labeled_share'].iloc[-1]:.3f}. Full series in {MATCHED_SHARE_CSV.name}")

diag_rows = [
    {"metric": "n_funds_master",        "value": _n},
    {"metric": "n_active_funds",        "value": int(_lab["n_active"].iloc[0])},
    {"metric": "n_passive_funds",       "value": int(_lab["n_passive"].iloc[0])},
    {"metric": "n_unknown_funds",       "value": int(_lab["n_unknown"].iloc[0])},
    {"metric": "firm_universe",         "value": _nfirm},
    {"metric": "n_quarters_grid",       "value": _nq},
    {"metric": "reconA_max_rel_dev",    "value": _rel},
    {"metric": "reconB_max_weight_dev", "value": _wdev},
    {"metric": "panel_out_rows",        "value": n},
    {"metric": "panel_out_firms",       "value": int(df["firm_str"].nunique())},
    {"metric": "panel_out_quarters",    "value": int(len(_q))},
    {"metric": "c6_firms",              "value": (int(c6_firms) if c6_firms is not None else -1)},
    {"metric": "c6_quarters",           "value": (int(c6_quarters) if c6_quarters is not None else -1)},
]
pd.DataFrame(diag_rows).to_csv(BUILD_DIAG_CSV, index=False)
if c6_firms is not None:
    print(f"    (C) comparability: fourgroup firms={df['firm_str'].nunique():,}/"
          f"quarters={len(_q)}  vs  c6 firms={c6_firms:,}/quarters={c6_quarters}")
else:
    print("    (C) comparability: c6_panel.dta not found — skipped (build it via build_c6_panel.py)")

# ===========================================================================
# (10) Write Stata file.
# ===========================================================================
print(f"\n[10] Writing {PANEL_DTA}")
df.to_stata(
    PANEL_DTA,
    write_index=False,
    convert_dates={"rdate": "tc"},
    version=118,
)
con.close()
print(f"     done. {n:,} rows written.")
print("\nDONE. Outputs:")
for p in (FUND_LABEL_PARQUET, PANEL_DTA, MATCHED_SHARE_CSV, BUILD_DIAG_CSV):
    print(f"  {p}")
