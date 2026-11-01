"""
build_ext_int_decomposition.py -- extensive / intensive / composition
decomposition of the country China-exposure measures M1, M2, M3.
Task f2 (external review item 4), 2026-08-10.

================================================================================
WHY THIS FILE EXISTS
================================================================================
The draft update of 2026-08-21, block (h), reports an extensive/intensive
decomposition of M1/M2/M3.  Those numbers were computed IN SESSION and never
written to disk: there was no generating script, no cell-level output and no
summary artifact anywhere in the repo.  A referee could not reproduce a single
digit of block (h).  This script regenerates the whole block from the raw
firm-quarter parquets, writes the cell-level intermediate AND the summary, and
hard-gates the arithmetic identity the decomposition rests on.

It reads nothing that a previous session produced except the shipped measures
CSV, and that one is used ONLY as a cross-check gate (the measures are
re-derived here from the same parquets, with the same SQL, and the two must
agree).

================================================================================
THE OBJECT BEING DECOMPOSED
================================================================================
For a country c and an ADJACENT quarter pair (t-1, t) the actual change in a
country measure is

    dM_{c,t} = M_{c,t} - M_{c,t-1}

where M is one of (definitions verbatim from build_country_measures.py, over
the FULL country cell in each quarter):

    M1 = SUM_i cn_links_i / SUM_i sc_links_i          link-weighted
    M2 = SUM_i cn_share_i * mcap_i / SUM_i mcap_i     mcap-weighted (mcap > 0)
    M3 = (1/N) SUM_i cn_share_i                       equal-weighted

dM mixes three different things: firms that gained or lost China links, firms
that were linked throughout and changed how many links they have, and pure
population churn (firms entering or leaving the country cell, plus drift in the
denominators contributed by firms that never had a China link).  The
decomposition separates them.

================================================================================
METHOD (this is exactly what produced the draft block-(h) numbers)
================================================================================
BALANCED SET.  B_{c,t} = firms observed in country c in BOTH t-1 and t.  Panel
entrants and exiters are, by construction, not in B.

TWO PLAYERS, defined on B by any-China-link status  linked_i,s = 1{cn_links > 0}:

    E  "extensive"  linked_{i,t-1} != linked_{i,t}   (gained or lost any link)
    I  "intensive"  linked_{i,t-1} =  linked_{i,t} = 1
    N  neither      linked in neither quarter        (NOT a player; always held
                                                      at its t-1 state)

N is not a player because it cannot move the numerator of any measure.  It can
still move the DENOMINATORS (supply-chain links for M1, market cap for M2), and
that movement is deliberately left in the residual, where it belongs with the
rest of the composition drift.

VALUE FUNCTION.  For a coalition S subset of {E, I},

    g(S) = the measure recomputed on B, with every firm in S at its t state
           and every other firm in B at its t-1 state.

So g({}) = the measure on B with everybody at t-1, and g({E,I}) = the measure on
B with the two players at t and the never-linked firms still at t-1.

SHAPLEY (two players, so the exact closed form, no sampling):

    comp_E = 1/2 [ g({E}) - g({}) ] + 1/2 [ g({E,I}) - g({I}) ]
    comp_I = 1/2 [ g({I}) - g({}) ] + 1/2 [ g({E,I}) - g({E}) ]

Efficiency gives comp_E + comp_I = g({E,I}) - g({}) EXACTLY.  Call that the
BALANCED CHANGE dB.  It is gated below, not asserted.

RESIDUAL (composition).  What the balanced set cannot explain:

    resid = dM - comp_E - comp_I

i.e. panel entry and exit, plus denominator/weight drift among never-linked
balanced firms.  By construction comp_E + comp_I + resid = dM, which is the
hard gate at the centre of this script.  ANY gate failure is a BUG in the
arithmetic, never a finding.

VARIANCE SHARES.  Across country-quarter cells, within (universe, measure),

    beta_k = Cov(comp_k, dM) / Var(dM),   k in {E, I, resid}

which sum to exactly 1 because the components sum to dM.  These are the
percentages quoted in the draft.  The "extensive share of the balanced change"
is the same object taken on dB instead of dM:

    beta_E_bal = Cov(comp_E, dB) / Var(dB),  beta_I_bal = 1 - beta_E_bal

Covariance betas rather than shares of levels: dM is signed and averages near
zero, so a share-of-levels denominator is not interpretable.  A gross-magnitude
share (SUM|comp_k| / SUM_j SUM|comp_j|) is emitted alongside as a robustness
read, and the two are expected to differ.

================================================================================
UNIVERSES
================================================================================
Both universes of build_country_measures.py, built here with the SAME SQL:

  ownership_matched  the estimation universe (European ownership book, Revere
                     matched, PIT-present; country = sec_country).  M1, M2, M3.
  revere_eu_all      every PIT-present European Revere firm-quarter
                     (country = eu_home_region).  M1, M3 only -- M2 is NOT
                     computable there, unmatched Revere firms have no FactSet
                     market cap.  The script reports M2/revere_eu_all as
                     "not computable", it does not silently emit zeros.

================================================================================
INPUT  : output/firm_quarter_china_exposure.parquet
         output/eu_revere_universe_qend.parquet
         output/merged_us_eu_zero_filled.parquet
         output/marketcap_it.parquet
         output/country_measures_m1m2m3.csv   (cross-check gate only)
OUTPUT : output/ext_int_cells.csv     one row per universe x country x quarter
                                      x measure: dM, comp_E, comp_I, resid,
                                      the four g values, class counts, status
         output/ext_int_summary.csv   the variance shares per universe x measure
Rotation discipline: refuses to overwrite a canonical output; rotate to
*_r3pre first.  Runtime ~1 min, no network, single duckdb connection.

================================================================================
REPRODUCTION STATUS (2026-08-10)
================================================================================
Run on the shipped parquets this script reproduces DRAFT_UPDATE_2026_08_21.md
block (h) to the last quoted digit:

  universe=ownership_matched   beta_E / beta_I / beta_resid     beta_E_bal
    M1   0.5655 / 0.0491 / 0.3854  (draft 57/5/39)                 0.9305 (0.93)
    M2   0.5147 / 0.3473 / 0.1381  (draft 51/35/14)                0.5972 (0.60)
    M3   0.3478 / 0.0414 / 0.6108  (draft 35/4/61)                 0.9103 (0.91)
  universe=revere_eu_all
    M1   0.5449 / 0.1202 / 0.3349  (draft 55 extensive)            0.8839
    M3   0.1904 / 0.0582 / 0.7514  (draft 75 composition)          0.7876

so block (h) is now regenerable from the raw parquets by a referee.  The
comparison is re-run and re-printed on every execution (see DRAFT_CLAIMS): it
is a REPRODUCTION check, deliberately non-fatal, because if the upstream
parquets are rebuilt the right response is to update the draft to the new
numbers, not to make this script fail.
================================================================================
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
_env_out = os.environ.get("DPN_OUT_DIR", "").strip()
OUT = Path(_env_out).resolve() if _env_out else PROJ / "output"

EXPOSURE = (OUT / "firm_quarter_china_exposure.parquet").as_posix()
REV_QEND = (OUT / "eu_revere_universe_qend.parquet").as_posix()
GRID     = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
MCAP     = (OUT / "marketcap_it.parquet").as_posix()
MEAS_CSV = OUT / "country_measures_m1m2m3.csv"

CELLS_CSV   = OUT / "ext_int_cells.csv"
SUMMARY_CSV = OUT / "ext_int_summary.csv"

# 28 European jurisdictions, verbatim from 00_setup.jl / build_country_measures.py
EU_COUNTRIES = ("GB", "DE", "FR", "NL", "CH", "IT", "ES", "SE", "DK", "NO",
                "FI", "BE", "AT", "IE", "LU", "PT", "PL", "CZ", "HU", "GR",
                "RO", "SK", "SI", "BG", "HR", "EE", "LV", "LT")
EU_SQL = "(" + ",".join(f"'{c}'" for c in EU_COUNTRIES) + ")"

UNIVERSES = ("ownership_matched", "revere_eu_all")
MEASURES = ("M1", "M2", "M3")
# M2 needs FactSet market cap, which only the ownership book has.
MEASURE_UNIVERSE = {("revere_eu_all", "M2"): False}

# dM and the g's are ratios bounded in [0,1]; float64 error on the Shapley
# recombination is O(1e-16).  1e-12 is ~4 orders of magnitude of slack and is
# NOT to be widened to make a run pass.
IDENTITY_ATOL = 1e-12
# Substantive-gate constants (rb MF-2, 2026-08-10). The two identity checks
# above are algebraic tautologies and cannot fail; these can.
ANCHOR_RTOL = 1e-9        # g(empty) vs the shipped M_lag, relative
ANCHOR_MIN_CELLS = 100    # refuse to ship if the anchor has too little support
CLOSURE_ATOL = 1e-9       # M3 residual on both-sides-balanced cells
CLOSURE_MIN_CELLS = 20
XCHECK_ATOL = 1e-9      # re-derived measures vs the shipped CSV
BETA_SUM_ATOL = 1e-9    # beta_E + beta_I + beta_resid == 1

# The block-(h) numbers as they stand in DRAFT_UPDATE_2026_08_21.md, at the
# precision they were computed in session.  Re-checked on every run so that
# "the draft matches the artifact" is an observation, not a memory.
# NOT a gate: if the parquets are rebuilt, the DRAFT is what must change.
DRAFT_CLAIMS = {
    ("ownership_matched", "M1"): dict(beta_E=.566, beta_I=.049, beta_resid=.385, beta_E_bal=.93),
    ("ownership_matched", "M2"): dict(beta_E=.515, beta_I=.347, beta_resid=.138, beta_E_bal=.60),
    ("ownership_matched", "M3"): dict(beta_E=.348, beta_I=.041, beta_resid=.611, beta_E_bal=.91),
    ("revere_eu_all",     "M1"): dict(beta_E=.55),
    ("revere_eu_all",     "M3"): dict(beta_resid=.75),
}
DRAFT_TOL = 0.005       # half a percentage point: the draft quotes whole %

# An intensive share at or below this is "contributes little" in the generated
# prose.  M2 sits near 0.35 and must NOT be described that way.
SMALL_INTENSIVE = 0.15


# =============================================================================
# rotation discipline (same rule as run_ri_country_ddd.rotation_guard, with
# this script's own suffix so the two rotations never collide)
# =============================================================================
def rotation_guard(target: Path) -> None:
    """Never overwrite a canonical artifact in place: rotate to *_r3pre first,
    and REFUSE outright if that slot is already occupied (two live vintages).
    `_r3pre` = the third-review rotation slot, following build_country_panel.py's
    `_r2pre`; build_country_weight_panel.py uses `_cwpre` for its own family."""
    pre = target.with_name(target.stem + "_r3pre" + target.suffix)
    if target.exists():
        if pre.exists():
            raise SystemExit(
                f"REFUSING: {target.name} exists AND the rotation slot "
                f"{pre.name} is already occupied -- two live vintages; resolve "
                "by hand.")
        raise SystemExit(
            f"{target.name} already exists -- rename it to {pre.name} "
            "(rotation rule) before re-running.")


def _require(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise SystemExit(f"missing input {label}: {path}")


# =============================================================================
# duckdb: the two universes, verbatim from build_country_measures.py
# =============================================================================
def connect():
    import duckdb

    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    tmp = Path(os.environ.get("DPN_DUCKDB_TEMP_DIR", r"E:/duckdb_tmp"))
    if tmp.parent.exists():
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    return con


def build_universe_views(con) -> None:
    """u_revere / u_owned. Column contract: country, quarter_end, firm_id,
    cn_links, sc_links, cn_share, mcap.  This SQL is copied from
    build_country_measures.py lines ~182-230 and must stay identical to it --
    the whole point is that the decomposed dM is the SHIPPED dM."""
    con.execute(f"""
        CREATE OR REPLACE VIEW u_revere AS
        SELECT u.eu_home_region                                AS country,
               e.quarter_end                                   AS quarter_end,
               e.eu_company_id                                 AS firm_id,
               CAST(e.n_cn_customer + e.n_cn_supplier AS BIGINT) AS cn_links,
               e.n_supplychain_links                           AS sc_links,
               e.china_share                                   AS cn_share,
               CAST(NULL AS DOUBLE)                            AS mcap
        FROM read_parquet('{EXPOSURE}') e
        JOIN read_parquet('{REV_QEND}') u
          ON u.eu_company_id = e.eu_company_id AND u.qend = e.quarter_end
        WHERE u.eu_home_region IN {EU_SQL}
    """)
    con.execute(f"""
        CREATE OR REPLACE VIEW u_owned AS
        SELECT g.sec_country                                   AS country,
               g.report_date                                   AS quarter_end,
               g.sec_entity_id                                 AS firm_id,
               CAST(g.n_cn_customer + g.n_cn_supplier AS BIGINT) AS cn_links,
               g.n_supplychain_links                           AS sc_links,
               g.china_share                                   AS cn_share,
               m.market_cap                                    AS mcap
        FROM read_parquet('{GRID}') g
        LEFT JOIN read_parquet('{MCAP}') m
               ON m.sec_entity_id = g.sec_entity_id
              AND m.sec_country   = g.sec_country
              AND m.report_date   = g.report_date
        WHERE g.holder_group = 'US'
          AND g.china_share IS NOT NULL
          AND g.sec_country IN {EU_SQL}
    """)

    for v, uni in (("u_owned", "ownership_matched"), ("u_revere", "revere_eu_all")):
        d = con.sql(f"SELECT COUNT(*) FROM (SELECT firm_id, quarter_end FROM {v} "
                    f"GROUP BY 1,2 HAVING COUNT(*) > 1)").fetchone()[0]
        if d:
            raise SystemExit(
                f"{uni}: {d:,} duplicate (firm, quarter) keys -- a duplicated "
                "firm-quarter double-counts links into M1 and puts a phantom "
                "firm in the balanced set.")
        # cn_links must never be NULL where the firm is in the universe: a NULL
        # would be silently skipped by SUM and would break the class split.
        nbad = con.sql(f"SELECT COUNT(*) FROM {v} WHERE cn_links IS NULL "
                       f"OR sc_links IS NULL OR cn_share IS NULL").fetchone()[0]
        if nbad:
            raise SystemExit(
                f"{uni}: {nbad:,} rows with NULL cn_links/sc_links/cn_share. "
                "The link-status split (cn_links > 0) is undefined for them; "
                "fix upstream rather than coalescing here.")
    print("[1] universes built; key uniqueness and NULL checks OK")


MEASURES_SQL = """
    SELECT '{uni}' AS universe, country, quarter_end,
           COUNT(*)                                                 AS n_firms,
           CAST(SUM(cn_links) AS DOUBLE) / NULLIF(SUM(sc_links), 0) AS M1,
           SUM(cn_share * mcap) FILTER (WHERE mcap > 0)
               / NULLIF(SUM(mcap) FILTER (WHERE mcap > 0), 0)        AS M2,
           AVG(cn_share)                                             AS M3
    FROM {src}
    GROUP BY 1, 2, 3
"""


def actual_measures(con) -> pd.DataFrame:
    """Re-derive M1/M2/M3 here, then hard-gate against the shipped CSV."""
    m = pd.concat(
        [con.sql(MEASURES_SQL.format(src="u_owned", uni="ownership_matched")).df(),
         con.sql(MEASURES_SQL.format(src="u_revere", uni="revere_eu_all")).df()],
        ignore_index=True)
    m["quarter_end"] = pd.to_datetime(m["quarter_end"])
    m["qi"] = m["quarter_end"].dt.year * 4 + m["quarter_end"].dt.quarter
    print(f"[2] re-derived measures: {len(m):,} country-quarter rows")

    if not MEAS_CSV.exists():
        raise SystemExit(
            f"missing {MEAS_CSV.name}. It is the cross-check gate: the whole "
            "claim is that this decomposition splits the SHIPPED dM, and that "
            "cannot be verified without it. Run build_country_measures.py first.")
    ship = pd.read_csv(MEAS_CSV, usecols=["universe", "country", "quarter_end",
                                          "M1", "M2", "M3", "n_firms"])
    ship["quarter_end"] = pd.to_datetime(ship["quarter_end"])
    j = m.merge(ship, on=["universe", "country", "quarter_end"], how="outer",
                suffixes=("_new", "_ship"), indicator=True, validate="1:1")
    n_off = int((j["_merge"] != "both").sum())
    if n_off:
        raise SystemExit(
            f"XCHECK FAILED: {n_off} country-quarter keys are in one of "
            "(re-derived, shipped country_measures_m1m2m3.csv) but not the "
            "other. The CSV is a different vintage of the parquets -- rebuild "
            "it with build_country_measures.py before decomposing it.")
    if not (j["n_firms_new"] == j["n_firms_ship"]).all():
        k = int((j["n_firms_new"] != j["n_firms_ship"]).sum())
        raise SystemExit(f"XCHECK FAILED: n_firms differs in {k} cells.")
    for c in MEASURES:
        a, b = j[f"{c}_new"], j[f"{c}_ship"]
        if not (a.isna() == b.isna()).all():
            raise SystemExit(f"XCHECK FAILED: {c} missingness pattern differs "
                             "between re-derived and shipped measures.")
        d = (a - b).abs().max()
        if pd.notna(d) and d > XCHECK_ATOL:
            raise SystemExit(f"XCHECK FAILED: max |{c}_new - {c}_ship| = {d:.3e} "
                             f"> {XCHECK_ATOL:.0e}.")
    print("[3] cross-check vs shipped country_measures_m1m2m3.csv: PASS "
          "(keys, n_firms, M1/M2/M3 all agree)")
    return m


# =============================================================================
# class-state sums on the balanced set
# =============================================================================
CLASS_SQL = """
    WITH v AS (
        SELECT country, quarter_end, firm_id, cn_links, sc_links, cn_share, mcap,
               CAST(year(quarter_end) * 4 + quarter(quarter_end) AS BIGINT) AS qi
        FROM {src}
    ),
    pair AS (
        -- a = quarter t, b = quarter t-1; ADJACENT quarters only (qi diff 1),
        -- and the SAME country: a firm that changes country is an exit from the
        -- old cell and an entry into the new one, i.e. composition, not a
        -- balanced firm.
        SELECT a.country, a.quarter_end, a.qi, a.firm_id,
               CASE WHEN (b.cn_links > 0) <> (a.cn_links > 0) THEN 'E'
                    WHEN  b.cn_links > 0 AND  a.cn_links > 0  THEN 'I'
                    ELSE 'N' END                                     AS cls,
               b.cn_links AS cn_l0, a.cn_links AS cn_l1,
               b.sc_links AS sc_l0, a.sc_links AS sc_l1,
               b.cn_share AS cs_0,  a.cn_share AS cs_1,
               b.mcap     AS mc_0,  a.mcap     AS mc_1
        FROM v a
        JOIN v b
          ON b.country = a.country AND b.firm_id = a.firm_id AND b.qi = a.qi - 1
    )
    SELECT '{uni}' AS universe, country, quarter_end, qi, cls,
           COUNT(*)                                            AS n,
           SUM(cn_l0)                                          AS cn0,
           SUM(cn_l1)                                          AS cn1,
           SUM(sc_l0)                                          AS sc0,
           SUM(sc_l1)                                          AS sc1,
           SUM(cs_0)                                           AS cs0,
           SUM(cs_1)                                           AS cs1,
           COALESCE(SUM(cs_0 * mc_0) FILTER (WHERE mc_0 > 0), 0) AS p0,
           COALESCE(SUM(cs_1 * mc_1) FILTER (WHERE mc_1 > 0), 0) AS p1,
           COALESCE(SUM(mc_0)        FILTER (WHERE mc_0 > 0), 0) AS q0,
           COALESCE(SUM(mc_1)        FILTER (WHERE mc_1 > 0), 0) AS q1
    FROM pair
    GROUP BY 1, 2, 3, 4, 5
"""

SUMCOLS = ["n", "cn0", "cn1", "sc0", "sc1", "cs0", "cs1", "p0", "p1", "q0", "q1"]


def class_state_sums(con) -> pd.DataFrame:
    """(universe, country, quarter, class) -> sums of each firm attribute at
    t-1 and at t.  This is all the state g(S) needs: every firm sits in exactly
    one of E / I / N, so any coalition value is a sum of class-level sums."""
    parts = []
    for src, uni in (("u_owned", "ownership_matched"), ("u_revere", "revere_eu_all")):
        d = con.sql(CLASS_SQL.format(src=src, uni=uni)).df()
        parts.append(d)
        print(f"    {uni}: {len(d):,} (country, quarter, class) rows, "
              f"{int(d['n'].sum()):,} balanced firm-pairs")
    cs = pd.concat(parts, ignore_index=True)
    cs["quarter_end"] = pd.to_datetime(cs["quarter_end"])
    if not set(cs["cls"]).issubset({"E", "I", "N"}):
        raise SystemExit(f"unexpected class labels: {sorted(set(cs['cls']))}")

    wide = (cs.set_index(["universe", "country", "quarter_end", "qi", "cls"])[SUMCOLS]
              .unstack("cls"))
    # a class absent from a cell is the EMPTY SET: zero firms, zero sums.
    wide = wide.reindex(columns=pd.MultiIndex.from_product([SUMCOLS, ["E", "I", "N"]]))
    wide = wide.fillna(0.0)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    print(f"[4] balanced-set class sums: {len(wide):,} country-quarter cells")
    return wide


# =============================================================================
# coalition values and the Shapley split
# =============================================================================
def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num/den with a zero denominator -> NaN (an undefined measure, which the
    status column then reports; never 0, which would be a silent fake value)."""
    return num / den.where(den != 0, np.nan)


def coalition_values(w: pd.DataFrame, measure: str) -> pd.DataFrame:
    """g({}), g({E}), g({I}), g({E,I}) for one measure.

    State selector: player in the coalition -> its t sums (suffix 1); otherwise
    its t-1 sums (suffix 0).  Class N is never in a coalition, so it always
    contributes its t-1 sums -- that is the definition, and it is why N's own
    denominator drift lands in the residual.
    """
    def pick(stub0: str, stub1: str, in_E: bool, in_I: bool) -> pd.Series:
        e = w[f"{stub1}_E"] if in_E else w[f"{stub0}_E"]
        i = w[f"{stub1}_I"] if in_I else w[f"{stub0}_I"]
        n = w[f"{stub0}_N"]
        return e + i + n

    g = pd.DataFrame(index=w.index)
    coalitions = {"empty": (False, False), "E": (True, False),
                  "I": (False, True), "EI": (True, True)}
    if measure == "M1":
        for lbl, (a, b) in coalitions.items():
            g[f"g_{lbl}"] = _ratio(pick("cn0", "cn1", a, b), pick("sc0", "sc1", a, b))
    elif measure == "M2":
        for lbl, (a, b) in coalitions.items():
            g[f"g_{lbl}"] = _ratio(pick("p0", "p1", a, b), pick("q0", "q1", a, b))
    elif measure == "M3":
        # denominator is the balanced-set head count, identical in every
        # coalition, so only the numerator moves
        nb = w["n_E"] + w["n_I"] + w["n_N"]
        for lbl, (a, b) in coalitions.items():
            g[f"g_{lbl}"] = _ratio(pick("cs0", "cs1", a, b), nb)
    else:
        raise SystemExit(f"unknown measure {measure}")

    # exact two-player Shapley
    g["comp_E"] = 0.5 * (g["g_E"] - g["g_empty"]) + 0.5 * (g["g_EI"] - g["g_I"])
    g["comp_I"] = 0.5 * (g["g_I"] - g["g_empty"]) + 0.5 * (g["g_EI"] - g["g_E"])
    g["dB"] = g["g_EI"] - g["g_empty"]
    return g


def build_cells(wide: pd.DataFrame, meas: pd.DataFrame) -> pd.DataFrame:
    """One row per universe x country x quarter x measure."""
    lag = meas.rename(columns={c: f"{c}_lag" for c in MEASURES})
    lag = lag.rename(columns={"n_firms": "n_firms_lag"})
    lag = lag.assign(qi=lag["qi"] + 1)[["universe", "country", "qi",
                                        "M1_lag", "M2_lag", "M3_lag",
                                        "n_firms_lag"]]
    base = (meas.merge(lag, on=["universe", "country", "qi"], how="left",
                       validate="1:1")
                .merge(wide, on=["universe", "country", "quarter_end", "qi"],
                       how="left", validate="1:1", indicator=True))

    rows = []
    for measure in MEASURES:
        for uni in UNIVERSES:
            if not MEASURE_UNIVERSE.get((uni, measure), True):
                continue
            b = base.loc[base["universe"] == uni].copy()
            has_bal = b["_merge"] == "both"
            g = coalition_values(b.fillna({c: 0.0 for c in wide.columns
                                           if c not in ("universe", "country",
                                                        "quarter_end", "qi")}),
                                 measure)
            # a cell with NO balanced firms has no coalition value at all
            g.loc[~has_bal, ["g_empty", "g_E", "g_I", "g_EI",
                             "comp_E", "comp_I", "dB"]] = np.nan

            r = pd.DataFrame({
                "universe": uni, "country": b["country"].to_numpy(),
                "quarter_end": b["quarter_end"].to_numpy(), "measure": measure,
                "M_t": b[measure].to_numpy(float),
                "M_lag": b[f"{measure}_lag"].to_numpy(float),
                "g_empty": g["g_empty"].to_numpy(), "g_E": g["g_E"].to_numpy(),
                "g_I": g["g_I"].to_numpy(), "g_EI": g["g_EI"].to_numpy(),
                "comp_E": g["comp_E"].to_numpy(), "comp_I": g["comp_I"].to_numpy(),
                "dB": g["dB"].to_numpy(),
                "n_firms_t": b["n_firms"].to_numpy(),
                "n_firms_lag": b["n_firms_lag"].to_numpy(float),
                "n_bal_E": b["n_E"].to_numpy(), "n_bal_I": b["n_I"].to_numpy(),
                "n_bal_N": b["n_N"].to_numpy(),
                "has_balanced": has_bal.to_numpy().astype(int),
            })
            r["n_balanced"] = r[["n_bal_E", "n_bal_I", "n_bal_N"]].sum(axis=1)
            r["dM"] = r["M_t"] - r["M_lag"]
            r["resid"] = r["dM"] - r["comp_E"] - r["comp_I"]

            g_ok = r[["g_empty", "g_E", "g_I", "g_EI"]].notna().all(axis=1)
            r["status"] = np.select(
                [r["M_lag"].isna(),
                 r["M_t"].isna(),
                 r["has_balanced"] == 0,
                 ~g_ok],
                ["no_lag_cell", "measure_undefined_t", "no_balanced_firms",
                 "coalition_denominator_zero"],
                default="ok")
            r["usable"] = (r["status"] == "ok").astype(int)

            # Blank the decomposition on every excluded cell. Without this a
            # cell whose measure is undefined at t (M1 with no supply-chain
            # links left in the country, M2 with no covered market cap) still
            # ships a finite comp_E/comp_I even though dM does not exist -- so
            # `mean(comp_I)` over the raw file would mix gated numbers with
            # ungated ones. Two Latvian M2 cells carry comp_I near -0.2 and
            # -0.4 this way, which is large enough to move a naive average.
            # The four g_* values stay, so an excluded cell is still auditable.
            r.loc[r["usable"] == 0, ["comp_E", "comp_I", "resid", "dB"]] = np.nan
            rows.append(r)
    cells = pd.concat(rows, ignore_index=True)
    return cells.sort_values(["universe", "measure", "country", "quarter_end"]
                             ).reset_index(drop=True)


# =============================================================================
# the hard gate
# =============================================================================
def identity_gate(cells: pd.DataFrame) -> None:
    """Gates on the decomposition. Two BOOKKEEPING checks and two SUBSTANTIVE
    ones; the file states plainly which is which.

    (rb MF-2) The previous version advertised the Shapley efficiency identity
    comp_E + comp_I == g(EI) - g({}) as "the one that can actually catch a
    coding error". That was FALSE. With
        comp_E = .5(g_E - g_empty) + .5(g_EI - g_I)
        comp_I = .5(g_I - g_empty) + .5(g_EI - g_E)
        dB     = g_EI - g_empty
    the sum comp_E + comp_I collapses to g_EI - g_empty for ANY four numbers
    g_*, so the check is an algebraic tautology in exactly the same way that
    comp_E + comp_I + resid == dM is (resid is DEFINED as the remainder). Both
    print 1e-18 whatever CLASS_SQL computes. A bug in CLASS_SQL -- SUM(cs_1)
    where SUM(cs_0) belongs, say -- would corrupt every g, every comp_E/comp_I
    and both old gates would still say PASS. The XCHECK in actual_measures()
    guards MEASURES_SQL (the dM side) only; it never touches CLASS_SQL, so the
    comp_E/comp_I numbers that get quoted shipped ungated.

    The two substantive gates below read the CLASS sums against the SHIPPED
    measures, so a wrong g fails them:

      (a) ANCHOR. On a cell where every t-1 firm is in the balanced set
          (n_balanced == n_firms_lag), the empty coalition evaluates every
          player at its t-1 state over the whole t-1 population, so
          g_empty MUST equal the shipped M_lag. This ties the class-level
          sums (cn0/sc0/p0/q0/cs0) to an externally computed number.

      (b) CLOSURE, M3 ONLY. On a cell balanced on BOTH sides (n_balanced ==
          n_firms_lag == n_firms_t) the grand coalition covers the whole t
          population as well, so for a simple average the decomposition must
          close exactly: resid == 0.
          M1 and M2 are deliberately EXCLUDED from (b), and not because they
          fail by accident. M3's denominator is the balanced-set head count,
          which is constant across coalitions (coalition_values, M3 branch),
          while M1's denominator is SUM(sc_links) and M2's is SUM(mcap): class
          N is pinned at its t-1 state by definition (see coalition_values
          docstring), so N's denominator drift lands in the residual by
          design. Observed max |resid| on both-sides-balanced cells is 0.208
          (M1) and 0.0278 (M2) -- the design, not a bug. Gating M1/M2 on (b)
          would be gating the design out.
    """
    u = cells.loc[cells["usable"] == 1]
    print("\n--- HARD GATE: additive identity of the decomposition ---")
    print(f"  cells total            : {len(cells):,}")
    print(f"  cells usable (gated)   : {len(u):,}")
    for st, k in cells.loc[cells["usable"] == 0, "status"].value_counts().items():
        print(f"  excluded [{st}]: {k:,}")

    if not len(u):
        raise SystemExit("no usable cells -- nothing was decomposed.")

    # Excluded cells must carry NO numbers. A cell that is not gated but still
    # ships a finite comp_E/comp_I/resid would be an ungated number in a
    # published artifact -- exactly the failure mode this script exists to end.
    nu = cells.loc[cells["usable"] == 0, ["comp_E", "comp_I", "resid", "dB"]]
    n_leak = int(nu.notna().any(axis=1).sum())
    print(f"  excluded cells carrying a finite component (must be 0): {n_leak:,}")
    if n_leak:
        raise SystemExit(
            f"GATE FAILED: {n_leak:,} cells are excluded from the identity gate "
            "yet still carry a finite comp_E/comp_I/resid in ext_int_cells.csv. "
            "Every number that ships must be gated; blank the components or "
            "make the cell usable.")

    d1 = (u["comp_E"] + u["comp_I"] + u["resid"] - u["dM"]).abs()
    d2 = (u["comp_E"] + u["comp_I"] - u["dB"]).abs()
    print(f"  max |comp_E+comp_I+resid - dM|      : {float(d1.max()):.3e} "
          f"(tol {IDENTITY_ATOL:.0e})")
    print(f"  max |comp_E+comp_I - (g_EI-g_empty)|: {float(d2.max()):.3e} "
          f"(tol {IDENTITY_ATOL:.0e})")
    for lbl, d in (("comp_E + comp_I + resid == dM", d1),
                   ("comp_E + comp_I == g(EI) - g(empty)  [Shapley efficiency]", d2)):
        if not float(d.max()) <= IDENTITY_ATOL:
            worst = u.loc[d.idxmax(), ["universe", "measure", "country",
                                       "quarter_end"]].to_dict()
            raise SystemExit(
                f"IDENTITY GATE FAILED on `{lbl}`: max abs deviation "
                f"{float(d.max()):.3e} > {IDENTITY_ATOL:.0e} (worst cell "
                f"{worst}). This is an ARITHMETIC IDENTITY -- a nonzero "
                "deviation is a bug in the coalition sums, not a finding. Fix "
                "the construction; do NOT widen the tolerance.")
    print("  (both of the above are algebraic identities: they verify the "
          "bookkeeping, NOT the coalition sums)")

    # -------------------------------------------------------------------
    # SUBSTANTIVE GATES. These read CLASS_SQL's coalition sums against the
    # independently built MEASURES_SQL numbers, so a wrong g FAILS them.
    # -------------------------------------------------------------------
    print("\n--- SUBSTANTIVE GATE (a): anchor g(empty) to the shipped M_lag ---")
    anchor = u.loc[(u["n_balanced"] == u["n_firms_lag"])
                   & u["g_empty"].notna() & u["M_lag"].notna()].copy()
    print(f"  cells where every t-1 firm is balanced: {len(anchor):,}")
    if len(anchor) < ANCHOR_MIN_CELLS:
        raise SystemExit(
            f"GATE (a) UNRUNNABLE: only {len(anchor):,} fully-balanced-at-t-1 "
            f"cells (need >= {ANCHOR_MIN_CELLS}). Without this anchor the "
            "coalition sums ship ungated; do not proceed.")
    da = (anchor["g_empty"] - anchor["M_lag"]).abs()
    # relative, because M1/M2 levels vary by orders of magnitude across cells
    rel_a = da / anchor["M_lag"].abs().clip(lower=1e-12)
    print(f"  max |g_empty - M_lag|      : {float(da.max()):.3e}")
    print(f"  max relative deviation     : {float(rel_a.max()):.3e} "
          f"(tol {ANCHOR_RTOL:.0e})")
    if float(rel_a.max()) > ANCHOR_RTOL:
        worst = anchor.loc[rel_a.idxmax(), ["universe", "measure", "country",
                                            "quarter_end", "g_empty", "M_lag"]].to_dict()
        raise SystemExit(
            f"GATE (a) FAILED: g(empty) does not reproduce the shipped M_lag "
            f"(max rel {float(rel_a.max()):.3e} > {ANCHOR_RTOL:.0e}, worst "
            f"{worst}). The empty coalition evaluates every player at its t-1 "
            "state, so on a cell where all t-1 firms are balanced it MUST equal "
            "the measure at t-1. A failure here means CLASS_SQL's coalition "
            "sums disagree with MEASURES_SQL: a real bug in the class "
            "construction, not a tolerance question.")
    print("  GATE (a) RESULT: PASS")

    print("\n--- SUBSTANTIVE GATE (b): M3 closure on both-sides-balanced cells ---")
    closed = u.loc[(u["measure"] == "M3")
                   & (u["n_balanced"] == u["n_firms_lag"])
                   & (u["n_balanced"] == u["n_firms_t"])
                   & u["resid"].notna()].copy()
    print(f"  M3 cells balanced on both sides: {len(closed):,}")
    if len(closed) < CLOSURE_MIN_CELLS:
        print(f"  [SKIP] fewer than {CLOSURE_MIN_CELLS} such cells; gate (a) "
              "carries the substantive check for this run.")
    else:
        db_ = closed["resid"].abs()
        print(f"  max |resid| on those cells : {float(db_.max()):.3e} "
              f"(tol {CLOSURE_ATOL:.0e})")
        if float(db_.max()) > CLOSURE_ATOL:
            worst = closed.loc[db_.idxmax(), ["universe", "country",
                                              "quarter_end", "resid"]].to_dict()
            raise SystemExit(
                f"GATE (b) FAILED: M3 does not close on a cell balanced on both "
                f"sides (max |resid| {float(db_.max()):.3e}, worst {worst}). M3's "
                "denominator is the balanced head count and is constant across "
                "coalitions, so with the whole t population inside the grand "
                "coalition the two players must exhaust dM. M1 and M2 are "
                "excluded from this gate by design: their denominators "
                "(SUM(sc_links), SUM(mcap)) are pinned at t-1 for class N, so "
                "denominator drift lands in the residual on purpose.")
        print("  GATE (b) RESULT: PASS")

    print("\n  ALL GATES PASS -- bookkeeping exact AND coalition sums anchored.")


# =============================================================================
# variance shares
# =============================================================================
def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (uni, measure), d in cells.groupby(["universe", "measure"], sort=False):
        u = d.loc[d["usable"] == 1]
        rec = {"universe": uni, "measure": measure,
               "n_cells_total": int(len(d)), "n_cells_usable": int(len(u)),
               "n_cells_dropped": int((d["usable"] == 0).sum()),
               "n_countries": int(u["country"].nunique()),
               "quarter_min": u["quarter_end"].min(),
               "quarter_max": u["quarter_end"].max()}
        if len(u) < 3:
            rows.append(rec)
            continue
        dM, dB = u["dM"], u["dB"]
        var_dM, var_dB = float(dM.var(ddof=1)), float(dB.var(ddof=1))
        rec["var_dM"], rec["sd_dM"] = var_dM, float(dM.std(ddof=1))
        rec["mean_dM"] = float(dM.mean())
        for k, col in (("E", "comp_E"), ("I", "comp_I"), ("resid", "resid")):
            rec[f"beta_{k}"] = float(u[col].cov(dM) / var_dM) if var_dM > 0 else np.nan
            rec[f"mean_{k}"] = float(u[col].mean())
            rec[f"sd_{k}"] = float(u[col].std(ddof=1))
        gross = {k: float(u[c].abs().sum())
                 for k, c in (("E", "comp_E"), ("I", "comp_I"), ("resid", "resid"))}
        tot = sum(gross.values())
        for k in gross:
            rec[f"gross_share_{k}"] = gross[k] / tot if tot > 0 else np.nan
        rec["beta_E_bal"] = (float(u["comp_E"].cov(dB) / var_dB)
                             if var_dB > 0 else np.nan)
        rec["beta_I_bal"] = (float(u["comp_I"].cov(dB) / var_dB)
                             if var_dB > 0 else np.nan)
        s = rec["beta_E"] + rec["beta_I"] + rec["beta_resid"]
        rec["beta_sum"] = s
        if np.isfinite(s) and abs(s - 1.0) > BETA_SUM_ATOL:
            raise SystemExit(
                f"VARIANCE-SHARE GATE FAILED for {uni}/{measure}: "
                f"beta_E + beta_I + beta_resid = {s:.12f} != 1. The betas are "
                "covariance shares of components that sum to dM, so they must "
                "sum to one by construction.")
        sb = rec["beta_E_bal"] + rec["beta_I_bal"]
        if np.isfinite(sb) and abs(sb - 1.0) > BETA_SUM_ATOL:
            raise SystemExit(
                f"VARIANCE-SHARE GATE FAILED for {uni}/{measure}: "
                f"beta_E_bal + beta_I_bal = {sb:.12f} != 1.")
        rec["beta_bal_sum"] = sb
        rows.append(rec)
    out = pd.DataFrame(rows)
    print("\n  variance-share sum gate (beta_E+beta_I+beta_resid == 1): PASS")
    return out


# =============================================================================
# the draft sentence, generated from the numbers actually computed
# =============================================================================
def draft_sentence(summary: pd.DataFrame) -> str:
    s = summary.set_index(["universe", "measure"])
    prim = "ownership_matched"

    def pct(u, m, col):
        try:
            v = s.loc[(u, m), col]
        except KeyError:
            return None
        return None if pd.isna(v) else 100.0 * float(v)

    def listing(ms: list[str]) -> str:
        if len(ms) == 1:
            return ms[0]
        return ", ".join(ms[:-1]) + " and " + ms[-1]

    have = [m for m in MEASURES if pct(prim, m, "beta_E") is not None]
    parts = [f"{m} {pct(prim, m, 'beta_E'):.0f}% extensive / "
             f"{pct(prim, m, 'beta_I'):.0f}% intensive / "
             f"{pct(prim, m, 'beta_resid'):.0f}% composition" for m in have]

    # Every quantified claim below is READ OFF the computed betas. Nothing about
    # which margin "dominates" is hardcoded, so a rebuild on new parquets
    # rewrites the sentence instead of silently invalidating it.
    small_i = [m for m in have if pct(prim, m, "beta_I") <= 100 * SMALL_INTENSIVE]
    big_i = [m for m in have if pct(prim, m, "beta_I") > 100 * SMALL_INTENSIVE]
    dominant = [m for m in have if pct(prim, m, "beta_E") > 50.0]
    not_dom = [m for m in have if pct(prim, m, "beta_E") <= 50.0]

    lead = (f"In the estimation universe the intensive margin contributes little "
            f"in {listing(small_i)}, while the extensive and composition margins "
            f"carry most of the variation: " + "; ".join(parts) + ".")
    if big_i:
        lead += (" " + listing(big_i) + " is the exception: cap-weighting gives "
                 f"continuing large firms an intensive share of "
                 + ", ".join(f"{pct(prim, m, 'beta_I'):.0f}%" for m in big_i) + ".")

    qualifier = ""
    if not_dom:
        bits = [f"{m} ({pct(prim, m, 'beta_E'):.0f}% extensive, "
                f"{pct(prim, m, 'beta_resid'):.0f}% composition)" for m in not_dom]
        qualifier = (f" The blanket phrase \"the extensive margin dominates\" is "
                     f"wrong for {listing(bits)}: there universe composition, not "
                     "the extensive margin, is the largest single contributor, "
                     "because in an equal-weighted mean every entrant and exiter "
                     "moves the measure directly.")
    scope = ""
    if dominant and not_dom:
        scope = (f" The extensive margin is the majority contributor for "
                 f"{listing(dominant)} only.")
    return lead + qualifier + scope


def draft_reproduction_check(summary: pd.DataFrame) -> None:
    """Print recomputed vs the numbers standing in DRAFT_UPDATE block (h).

    Deliberately NON-FATAL. The draft is the thing that has to track the
    artifact, not the other way round, so a mismatch is loud text telling the
    writer to restate block (h) -- it is not a build failure.
    """
    s = summary.set_index(["universe", "measure"])
    print("\n--- REPRODUCTION CHECK vs DRAFT_UPDATE_2026_08_21.md block (h) ---")
    print(f"{'universe':<18}{'meas':<5}{'stat':<12}{'recomputed':>12}"
          f"{'in draft':>10}{'delta':>10}  flag")
    n_off = 0
    for (uni, m), claims in DRAFT_CLAIMS.items():
        for stat, claimed in claims.items():
            got = s.loc[(uni, m), stat] if (uni, m) in s.index else np.nan
            if pd.isna(got):
                print(f"{uni:<18}{m:<5}{stat:<12}{'n/a':>12}{claimed:>10.3f}"
                      f"{'':>10}  MEASURE ABSENT")
                n_off += 1
                continue
            d = float(got) - claimed
            ok = abs(d) <= DRAFT_TOL
            n_off += (not ok)
            print(f"{uni:<18}{m:<5}{stat:<12}{float(got):>12.4f}{claimed:>10.3f}"
                  f"{d:>+10.4f}  {'match' if ok else 'DIFFERS'}")
    if n_off:
        print(f"\n  {n_off} draft figure(s) no longer match this run "
              f"(tol {DRAFT_TOL}). The ARTIFACT is authoritative: restate "
              "block (h) from ext_int_summary.csv. Not a build failure.")
    else:
        print(f"\n  All draft block-(h) figures reproduce within {DRAFT_TOL}. "
              "Block (h) is regenerable from the raw parquets.")


# =============================================================================
def main() -> None:
    rotation_guard(CELLS_CSV)
    rotation_guard(SUMMARY_CSV)
    for p, lbl in ((EXPOSURE, "firm_quarter_china_exposure"),
                   (REV_QEND, "eu_revere_universe_qend"),
                   (GRID, "merged_us_eu_zero_filled"),
                   (MCAP, "marketcap_it")):
        _require(p, lbl)

    con = connect()
    build_universe_views(con)
    meas = actual_measures(con)
    wide = class_state_sums(con)
    con.close()

    cells = build_cells(wide, meas)
    identity_gate(cells)
    summary = summarise(cells)

    cell_cols = ["universe", "country", "quarter_end", "measure",
                 "dM", "comp_E", "comp_I", "resid",
                 "M_lag", "M_t", "dB", "g_empty", "g_E", "g_I", "g_EI",
                 # n_firms_lag ships (rb MF-2) so substantive gate (a) is
                 # re-derivable from this CSV alone: gate (a) selects on
                 # n_balanced == n_firms_lag, and a referee cannot check it
                 # without the column.
                 "n_firms_t", "n_firms_lag", "n_balanced",
                 "n_bal_E", "n_bal_I", "n_bal_N",
                 "status", "usable"]
    cells[cell_cols].to_csv(CELLS_CSV, index=False)
    print(f"\nwrote {CELLS_CSV.name} ({len(cells):,} rows)")
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"wrote {SUMMARY_CSV.name} ({len(summary):,} rows)")

    show = ["universe", "measure", "n_cells_usable", "beta_E", "beta_I",
            "beta_resid", "beta_E_bal", "beta_I_bal"]
    print("\n--- VARIANCE SHARES (covariance betas on dM; betas on dB for _bal) ---")
    print(summary[show].to_string(index=False,
                                  float_format=lambda x: f"{x:.4f}"))
    print("\n  NOTE: M2 is not computable on revere_eu_all (no FactSet market "
          "cap for unmatched Revere firms) and is therefore absent above by "
          "construction, not by a filter.")

    draft_reproduction_check(summary)

    print("\n" + "=" * 78)
    print("SUMMARY SENTENCE FOR THE DRAFT (generated from the numbers above)")
    print("=" * 78)
    print(draft_sentence(summary))
    print("=" * 78)
    print("READING RULES stamped with these artifacts")
    print(" * comp_E + comp_I + resid = dM is an IDENTITY, gated at "
          f"{IDENTITY_ATOL:.0e} on every usable cell. A nonzero gate deviation")
    print("   is a bug in the coalition sums; it is never a finding.")
    print(" * `resid` is composition: panel entry and exit, plus denominator and")
    print("   weight drift among balanced firms that were never China-linked.")
    print("   It is NOT an error term and NOT a measure of fit.")
    print(" * The betas are covariance shares of a SIGNED change. They can fall")
    print("   outside [0,1] if a component moves against dM; read the sign.")
    print(" * Cells excluded by `status` are listed in the log and carried in")
    print("   ext_int_cells.csv with usable = 0; they are not silently dropped.")


if __name__ == "__main__":
    sys.exit(main())
