"""
diag_p0_compare.py — READ-ONLY old-vs-new validation for the P0 holdings
snapshot rebuild (exact-EOM filter -> as-of-W selection).

Run this AFTER archive_preP0.py --apply AND after 03/04 have been rebuilt.
It never writes to any pipeline artifact; it only reads them and emits
diag_p0_*.csv summaries next to them.

WHAT IT CHECKS
  GATE 1  Fund-coverage gap closure. Per calendar quarter, coverage =
          (distinct funds captured in the panel) / (distinct funds reporting
          anywhere in that calendar quarter, raw FactSet, ADJ_MV>0). Reported
          per quarter with the quarter-end weekday, then aggregated into
          weekday-mean minus weekend-mean, BEFORE and AFTER. The target is gap
          CLOSURE, not an absolute level. Weekday coverage RISING after the fix
          is expected (early reporters are now included) and is NOT a failure.

          GATED PER ERA, NOT POOLED. A pooled 1999-2023 mean hides the era that
          matters: measured at W=10 the pooled gap is 4.18pp but ramp-up
          1999-2005 is 11.26pp (does NOT close) against mid 2006-2015 = 2.92pp
          and modern 2016-2023 = 1.63pp. The ramp-up era is INSIDE the
          estimation sample (06_cartesian_grid.jl builds the grid over
          1999-2023), so a pooled PASS would certify a fix that fails on a
          quarter of the panel. Reported additionally TREND-FREE (each
          quarter's coverage minus the mean of its two neighbours), because
          coverage trends secularly with the LionShares ramp-up and weekend
          quarter-ends are not uniformly spread over that trend.

  GATE 2  US-share calendar swing. Per quarter, US fraction of captured funds,
          before and after, plus the weekday-vs-weekend difference in that
          share. Pre-fix this swings with the calendar (the dropped Friday
          batch is ~26% US vs ~21% US in the kept Saturday batch) — an
          asymmetry group x quarter FE cannot absorb. Post-fix it must be flat.

          GATED ON THE SELECTION TILT, PER ERA. The raw US share trends hard
          over 1999-2023 (44% in the ramp-up era, 25-26% in the modern era), so
          a weekday-minus-weekend contrast of the RAW share mixes the calendar
          artifact with that trend, and pools to a PASS by sign cancellation
          between opposite-signed eras (measured at W=10: FULL -0.35pp, but
          ramp-up +0.69, mid -1.13, modern -1.40 — both post-2006 eras breach
          1.0pp). The clean statistic is the SELECTION TILT: captured US share
          MINUS the US share of ALL funds reporting in the same quarter. Under
          an unbiased rule both weekday and weekend tilts are ~0, so the
          weekday-vs-weekend gap in the tilt is the bias-relevant number. The
          raw diff is still reported per era alongside.

  GATE 3  Dollar levels. Per-quarter US-investor x 28-European-country nominal
          totals, old vs new, on TWO definitions:
            (a) RAW   = SUM(adj_mv) over holdings_eom, no issue_type filter.
            (b) EQ/AD = SUM(I_ict) over ownership_ict (04 applies
                        issue_type IN ('EQ','AD')).
          WHICH ONE THE SPEC ANCHORS USE — verified on the live pre-P0
          artifacts on 2026-08-04, not assumed:
                          RAW adj_mv     EQ/AD I_ict
            2021-12-31    $2.3965T       $2.3781T
            2022-12-31    $1.4575T       $1.4471T
          The spec's exact-filter figures are "$1,457B" (2022Q4) and "$2,397B"
          (2021Q4) — i.e. the RAW definition, to the dollar. The EQ/AD series
          runs 0.71-0.78% lower. So the spec bands ($1.85-1.95T for 2022Q4,
          ~$2.40T for 2021Q4) are applied to RAW; the EQ/AD series is reported
          alongside with the same band scaled by the measured wedge, because
          naively gating EQ/AD on the RAW band would fail at the lower edge for
          a definition reason rather than a data reason.
          Anchors: 2022Q4 total in $1.85-1.95T; 2022Q4 nominal YoY about -21%
          (+/-2pp); 2021Q4 total approximately unchanged (~$2.40T).

  GATE 4  Structure. Exactly one row per (fund_id, fsym_id, report_date) in the
          new panel; every report_date equals a calendar quarter-end of
          Mar/Jun/Sep/Dec; asof_gap_days distribution (if the column exists).

  Plus: per-quarter row counts old vs new, and the programmatically derived
  weekend/holiday quarter-end list with coverage before and after.

THRESHOLD PROVENANCE (read before quoting a PASS)
  Spec-given numbers:  2022Q4 in [1.85, 1.95] T; 2022Q4 YoY -21% +/- 2pp;
                       2021Q4 approximately unchanged (~$2.40T).
  Operator-set here:   GATE1_MAX_GAP_PP, GATE1_MAX_ARTIFACT_PP,
                       GATE2_MAX_TILT_GAP_PP, GATE3_Q4_2021_*.
                       These encode "small and not systematically signed" as a
                       number so the script can print PASS/FAIL. They are
                       calibrated to the 2022-2023 measurements in the spec
                       (post-fix weekday 88.2-90.2%, weekend 84.2-90.4%), NOT
                       handed down by it. Change them only with a written
                       reason.

  GATE1_MAX_GAP_PP was 5.0pp and is now 3.0pp. 5.0 was ~4x looser than its own
  calibration data (the 2022-2023 post-fix gap is 1.14pp) and sat just above
  the pooled full-sample value of 4.18pp, i.e. it would have printed PASS with
  0.82pp of headroom while the era that matters failed by 6pp. 3.0pp is what
  the calibration data supports: it clears modern 2016-2023 (1.63pp) and mid
  2006-2015 (2.92pp) and correctly fails ramp-up 1999-2005 (11.26pp).

  DISCLOSED ERA-LEVEL FAILURE — 1999-2005. The ramp-up era's coverage gap does
  NOT close and cannot be closed anywhere in the W plateau (W=3: 11.18pp,
  W=10: 11.26pp, W=14: 11.29pp; only W=31 helps, at the cost of importing a
  whole extra month of genuinely stale mid-quarter reporters). Early
  LionShares funds report on scattered mid-quarter dates rather than on the
  prior business day, so the as-of rule is not the binding constraint there.
  This is reported as status DISCLOSED-FAIL: a named, expected, pre-measured
  limitation, printed at the top of the summary, NOT absorbed into a pooled
  PASS and NOT counted as a regression in the exit code.

USAGE
  python diag_p0_compare.py
  python diag_p0_compare.py --refresh-denominator   # re-scan raw FactSet
  python diag_p0_compare.py --skip-denominator      # gates 3/4 only, fast
"""

from __future__ import annotations

import argparse
import calendar
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "output"

EOM_NEW = OUT / "holdings_eom.parquet"
EOM_OLD = OUT / "holdings_eom_exactEOM_preP0.parquet"
ICT_NEW = OUT / "ownership_ict.parquet"
ICT_OLD = OUT / "ownership_ict_preP0.parquet"

RAW_DIR = Path(os.environ.get("DPN_RAW_PARQUET_DIR", r"E:\Data\Data\raw_parquet"))
RAW_GLOB = (RAW_DIR / "Factset_FundOwners_*.parquet").as_posix()

# v2: the cache now carries a per-quarter US-fund denominator (ISO_COUNTRY was
# absent from v1, which made the selection tilt uncomputable in this script).
# The filename is bumped so the country-less v1 cache is never silently reused.
DENOM_CACHE = OUT / "diag_p0_raw_fund_denominator_v2.csv"
DENOM_CACHE_V1 = OUT / "diag_p0_raw_fund_denominator.csv"

# Era split. The ramp-up era is INSIDE the estimation sample: 06_cartesian_grid.jl
# builds the grid over range(1999, 2024) / 1999-03-31 to 2023-12-31.
ERAS = [
    ("FULL 1999-2023", 1999, 2023),
    ("ramp-up 1999-2005", 1999, 2005),
    ("mid 2006-2015", 2006, 2015),
    ("modern 2016-2023", 2016, 2023),
    ("main-session window 2022-2023", 2022, 2023),
]
# Eras whose GATE 1 failure is a KNOWN, pre-measured, W-unfixable property of
# the raw feed rather than a regression in this rebuild (see docstring).
DISCLOSED_GATE1_FAIL_ERAS = {"ramp-up 1999-2005"}

# EU_COUNTRIES verbatim from 00_setup.jl (28 "European" countries; includes
# GB/CH/NO, hence European, not EU).
EU_COUNTRIES = ("GB", "DE", "FR", "NL", "CH", "IT", "ES", "SE", "DK", "NO",
                "FI", "BE", "AT", "IE", "LU", "PT", "PL", "CZ", "HU", "GR",
                "RO", "SK", "SI", "BG", "HR", "EE", "LV", "LT")
EU_SQL = "(" + ",".join(f"'{c}'" for c in EU_COUNTRIES) + ")"

# ---- thresholds (see THRESHOLD PROVENANCE above) --------------------------
# operator-set: |weekday_mean - weekend_mean| coverage, applied PER ERA.
# Tightened from 5.0 -> 3.0 (see docstring): clears modern (1.63pp) and mid
# (2.92pp), correctly fails the disclosed ramp-up era (11.26pp).
GATE1_MAX_GAP_PP = 3.0
# operator-set: trend-free calendar artifact (local deviation vs neighbouring
# quarters). 2.0pp is the plateau criterion already used by
# diag_p0_asof_window.py; measured at W=10 the full-sample artifact is 1.06pp.
GATE1_MAX_ARTIFACT_PP = 2.0
# operator-set: |weekday_mean - weekend_mean| of the SELECTION TILT, per era.
# Measured at W=10: FULL +0.013, ramp-up +0.386, mid +0.510, modern -0.387 —
# all comfortably inside 1.0. At W=0 (the broken rule) they are -3.409 / -1.648
# / -5.130 / -2.585, so 1.0pp separates fixed from broken by a wide margin.
GATE2_MAX_TILT_GAP_PP = 1.0
# The raw US-share weekday-vs-weekend diff is REPORTED per era but NOT gated:
# it trends, so it pools by sign cancellation between opposite-signed eras.
GATE3_Q4_2022_LO, GATE3_Q4_2022_HI = 1.85e12, 1.95e12          # spec
GATE3_YOY_LO, GATE3_YOY_HI = -0.23, -0.19                      # spec (-21% +/-2pp)
GATE3_Q4_2021_LO, GATE3_Q4_2021_HI = 2.35e12, 2.45e12          # operator-set band around spec's ~$2.40T
GATE3_Q4_2021_MAX_REL_CHANGE = 0.01                            # operator-set: "approximately unchanged"
# Measured 2026-08-04 on the live pre-P0 artifacts (see module docstring):
# EQ/AD totals run ~0.75% below RAW adj_mv totals. Used only to translate the
# RAW-defined spec band onto the EQ/AD series for the secondary report.
EQAD_WEDGE = 0.0075

# (name, status, detail). status is one of PASS / FAIL / DISCLOSED-FAIL.
# DISCLOSED-FAIL is a named, pre-measured, W-unfixable limitation (currently
# only the 1999-2005 GATE 1 coverage gap). It is printed prominently and never
# silently folded into a pooled PASS, but it does not set the exit code,
# because it is not a regression introduced by this rebuild.
results: list[tuple[str, str, str]] = []

PASS, FAIL, DISCLOSED = "PASS", "FAIL", "DISCLOSED-FAIL"


def gate(name: str, ok: bool, detail: str, disclosed: bool = False) -> None:
    status = PASS if ok else (DISCLOSED if disclosed else FAIL)
    results.append((name, status, detail))
    print(f"  [{status}] {name}: {detail}")


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    # C: is tight (~12 GB free). Spill to the data volume when it exists.
    for cand in (RAW_DIR.drive + "/duckdb_tmp", str(BASE / ".tmp")):
        try:
            Path(cand).mkdir(parents=True, exist_ok=True)
            con.execute(f"SET temp_directory='{Path(cand).as_posix()}'")
            break
        except Exception:
            continue
    return con


def has_column(con, path: Path, col: str) -> bool:
    cols = con.execute(
        f"SELECT * FROM read_parquet('{path.as_posix()}') LIMIT 0").df().columns
    return col in cols


def quarter_label(ts: pd.Timestamp) -> str:
    return f"{ts.year}Q{(ts.month - 1)//3 + 1}"


def calendar_qend(ts: pd.Timestamp) -> pd.Timestamp:
    """Calendar last day of the quarter containing ts."""
    qm = ((ts.month - 1) // 3 + 1) * 3
    return pd.Timestamp(ts.year, qm, calendar.monthrange(ts.year, qm)[1])


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------
def panel_by_quarter(con, path: Path, tag: str) -> pd.DataFrame:
    """Per report_date: rows, distinct funds, distinct US funds."""
    df = con.execute(f"""
        SELECT CAST(report_date AS DATE)                       AS report_date,
               COUNT(*)                                        AS n_rows,
               COUNT(DISTINCT fund_id)                         AS n_funds,
               COUNT(DISTINCT CASE WHEN investor_country = 'US'
                                   THEN fund_id END)           AS n_funds_us
        FROM read_parquet('{path.as_posix()}')
        GROUP BY 1 ORDER BY 1
    """).df()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.rename(columns={c: f"{c}_{tag}" for c in
                            ("n_rows", "n_funds", "n_funds_us")})
    return df


def raw_denominator(con, refresh: bool) -> pd.DataFrame:
    """Distinct funds reporting ANYWHERE in each calendar quarter (ADJ_MV>0),
    SPLIT BY INVESTOR COUNTRY.

    The US split is what makes the GATE 2 selection tilt computable here: tilt =
    (US share of funds the rule CAPTURED) - (US share of ALL funds reporting in
    the same quarter). v1 of this query selected DISTINCT (quarter_start,
    fund_id) with no country column, so n_us_funds_reporting did not exist and
    GATE 2 could only look at the raw captured share, which trends.
    """
    if DENOM_CACHE.is_file() and not refresh:
        print(f"[denominator] cache hit: {DENOM_CACHE}")
        d = pd.read_csv(DENOM_CACHE)
        d["quarter_start"] = pd.to_datetime(d["quarter_start"])
        if "n_us_funds_reporting" not in d.columns:
            raise RuntimeError(
                f"{DENOM_CACHE} predates the US split. Delete it and re-run "
                "with --refresh-denominator.")
        return d

    if DENOM_CACHE_V1.is_file():
        print(f"[denominator] NOTE: the v1 (country-less) cache {DENOM_CACHE_V1.name} "
              "exists and is deliberately NOT reused; rebuilding v2 with ISO_COUNTRY.")

    if not list(RAW_DIR.glob("Factset_FundOwners_*.parquet")):
        raise FileNotFoundError(
            f"raw FactSet parquet cache not found under {RAW_DIR}. "
            "Set DPN_RAW_PARQUET_DIR, or pass --skip-denominator "
            "(gates 1 and 2 will be unavailable).")

    print(f"[denominator] full scan of {RAW_GLOB} — this is the slow step "
          "(one pass over ~187M rows, 4 columns). Be patient.")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE fund_q AS
        SELECT DISTINCT
            CAST(DATE_TRUNC('quarter', CAST(REPORT_DATE AS DATE)) AS DATE) AS quarter_start,
            FACTSET_FUND_ID AS fund_id,
            ISO_COUNTRY      AS investor_country
        FROM read_parquet('{RAW_GLOB}')
        WHERE ADJ_MV IS NOT NULL AND ADJ_MV > 0
    """)
    # COUNT(DISTINCT CASE WHEN ... ) mirrors panel_by_quarter() exactly, so the
    # numerator and denominator treat a multi-country fund identically.
    d = con.execute("""
        SELECT quarter_start,
               COUNT(DISTINCT fund_id) AS n_funds_reporting,
               COUNT(DISTINCT CASE WHEN investor_country = 'US' THEN fund_id END)
                   AS n_us_funds_reporting
        FROM fund_q GROUP BY 1 ORDER BY 1
    """).df()
    d["quarter_start"] = pd.to_datetime(d["quarter_start"])
    d.to_csv(DENOM_CACHE, index=False)
    print(f"[denominator] wrote cache {DENOM_CACHE} ({len(d)} quarters)")
    return d


def dollars_ict(con, path: Path, tag: str) -> pd.DataFrame:
    """EQ/AD definition: SUM(I_ict) from ownership_ict."""
    df = con.execute(f"""
        SELECT CAST(report_date AS DATE)      AS report_date,
               SUM(I_ict)                     AS usd_eqad,
               COUNT(DISTINCT sec_entity_id)  AS n_firms
        FROM read_parquet('{path.as_posix()}')
        WHERE investor_country = 'US' AND sec_country IN {EU_SQL}
        GROUP BY 1 ORDER BY 1
    """).df()
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df.rename(columns={"usd_eqad": f"usd_eqad_{tag}",
                              "n_firms": f"n_firms_{tag}"})


def dollars_raw(con, path: Path, tag: str) -> pd.DataFrame:
    """RAW definition: SUM(adj_mv) from holdings_eom, no issue_type filter.
    This is the definition the spec's $1,457B / $2,397B / $1,892B figures use
    (verified to the dollar on the pre-P0 artifacts)."""
    df = con.execute(f"""
        SELECT CAST(report_date AS DATE) AS report_date,
               SUM(adj_mv)               AS usd_raw
        FROM read_parquet('{path.as_posix()}')
        WHERE investor_country = 'US' AND sec_country IN {EU_SQL}
        GROUP BY 1 ORDER BY 1
    """).df()
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df.rename(columns={"usd_raw": f"usd_raw_{tag}"})


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-denominator", action="store_true")
    ap.add_argument("--skip-denominator", action="store_true",
                    help="skip the raw scan; gates 1 and 2 become UNAVAILABLE")
    args = ap.parse_args()

    # holdings_eom (old + new) is the hard requirement: gates 1, 2, 4 and the RAW
    # half of gate 3 are all computable from it alone.
    missing = [p for p in (EOM_NEW, EOM_OLD) if not p.is_file()]
    if missing:
        print("FATAL: required inputs missing:")
        for p in missing:
            print(f"  {p}")
        print("\nExpected state: archive_preP0.py --apply has run (creating the")
        print("_preP0 twins) AND 03 has been rebuilt (creating holdings_eom.parquet).")
        print("Nothing was read.")
        return 2

    # ownership_ict is OPTIONAL: it only feeds the EQ/AD half of gate 3, which is
    # reported alongside the gated RAW series. 04 has not necessarily been re-run
    # yet when 03's gates are checked, and archive_preP0.py has by then renamed
    # the pre-P0 copy away — so requiring it would make gates 1/2/4 unreachable
    # after a 03-only rebuild, which is exactly when they matter most.
    have_ict = ICT_NEW.is_file() and ICT_OLD.is_file()
    if not have_ict:
        print("NOTE: ownership_ict.parquet (new and/or _preP0) is absent, so the")
        print("      EQ/AD half of GATE 3 is UNAVAILABLE. The gated RAW series")
        print("      (SUM(adj_mv) over holdings_eom) is the spec-comparable one and")
        print("      is unaffected. Re-run after 04 to get the EQ/AD column back.")
        for p in (ICT_NEW, ICT_OLD):
            print(f"        {'present' if p.is_file() else 'ABSENT '}  {p.name}")

    con = connect()

    # ---------------- structure of the new panel (GATE 4) -----------------
    print("\n" + "=" * 90)
    print("GATE 4 — structure of the rebuilt panel")
    print("=" * 90)
    dup = con.execute(f"""
        SELECT COUNT(*) AS n FROM (
            SELECT fund_id, fsym_id, report_date
            FROM read_parquet('{EOM_NEW.as_posix()}')
            GROUP BY 1,2,3 HAVING COUNT(*) > 1)
    """).df()["n"].iloc[0]
    gate("4a one row per (fund_id, fsym_id, quarter)", int(dup) == 0,
         f"{int(dup):,} duplicate keys")

    dates = con.execute(f"""
        SELECT DISTINCT CAST(report_date AS DATE) AS d
        FROM read_parquet('{EOM_NEW.as_posix()}') ORDER BY 1
    """).df()["d"]
    dates = pd.to_datetime(dates)
    bad = [d for d in dates
           if not (d.month in (3, 6, 9, 12) and d == calendar_qend(d))]
    gate("4b every report_date is a calendar quarter-end (Mar/Jun/Sep/Dec)",
         len(bad) == 0,
         "all clean" if not bad else f"{len(bad)} offenders, e.g. {bad[:5]}")

    if has_column(con, EOM_NEW, "asof_gap_days"):
        g = con.execute(f"""
            SELECT CAST(report_date AS DATE) AS report_date,
                   MIN(asof_gap_days) AS gap_min,
                   QUANTILE_CONT(asof_gap_days, 0.5) AS gap_p50,
                   QUANTILE_CONT(asof_gap_days, 0.95) AS gap_p95,
                   MAX(asof_gap_days) AS gap_max,
                   AVG(CASE WHEN asof_gap_days = 0 THEN 1.0 ELSE 0.0 END) AS frac_gap0
            FROM read_parquet('{EOM_NEW.as_posix()}')
            GROUP BY 1 ORDER BY 1
        """).df()
        g.to_csv(OUT / "diag_p0_asof_gap_by_quarter.csv", index=False)
        print(f"  asof_gap_days: overall max={g['gap_max'].max():.0f}, "
              f"median of per-quarter p95={g['gap_p95'].median():.1f}, "
              f"mean frac at gap=0 {g['frac_gap0'].mean():.3f}")
        print("  -> diag_p0_asof_gap_by_quarter.csv")
    else:
        print("  NOTE: no asof_gap_days column — the rebuild did not stamp it, "
              "or this is still the old panel.")

    # ---------------- per-quarter row / fund counts ------------------------
    print("\n" + "=" * 90)
    print("Per-quarter row counts and fund counts, old vs new")
    print("=" * 90)
    old = panel_by_quarter(con, EOM_OLD, "old")
    new = panel_by_quarter(con, EOM_NEW, "new")
    q = old.merge(new, on="report_date", how="outer").sort_values("report_date")
    q["quarter"] = q["report_date"].map(quarter_label)
    q["dow"] = q["report_date"].dt.day_name().str[:3]
    q["is_weekend"] = q["report_date"].dt.dayofweek >= 5
    for c in ("n_rows_old", "n_funds_old", "n_funds_us_old",
              "n_rows_new", "n_funds_new", "n_funds_us_new"):
        q[c] = q[c].fillna(0).astype("int64")
    q["rows_pct_change"] = (q["n_rows_new"] / q["n_rows_old"].replace(0, pd.NA) - 1) * 100

    # ---------------- denominator + coverage (GATES 1, 2) ------------------
    have_denom = False
    if not args.skip_denominator:
        try:
            d = raw_denominator(con, args.refresh_denominator)
            d["report_date"] = d["quarter_start"].map(calendar_qend)
            q = q.merge(d[["report_date", "n_funds_reporting",
                           "n_us_funds_reporting"]],
                        on="report_date", how="left")
            have_denom = True
        except Exception as exc:                                  # noqa: BLE001
            print(f"[denominator] UNAVAILABLE: {exc}")
    else:
        print("[denominator] skipped by flag — gates 1 and 2 unavailable")

    q["year"] = q["report_date"].dt.year
    if have_denom:
        q["cov_old_pct"] = q["n_funds_old"] / q["n_funds_reporting"] * 100
        q["cov_new_pct"] = q["n_funds_new"] / q["n_funds_reporting"] * 100
        q["cov_delta_pp"] = q["cov_new_pct"] - q["cov_old_pct"]
        # TREND-FREE version: each quarter's coverage minus the mean of its two
        # neighbours. Sample-wide coverage rises secularly with the LionShares
        # ramp-up and weekend quarter-ends are not uniformly spread over that
        # trend, so a raw weekday-minus-weekend mean confounds the calendar
        # artifact with the trend. The artifact is inherently LOCAL, so measure
        # it locally. (Attenuated where two weekend quarter-ends are adjacent,
        # since the neighbour is then also depressed — a LOWER bound on the old
        # rule's damage and a conservative test of the fix.)
        q = q.sort_values("report_date").reset_index(drop=True)
        for v in ("old", "new"):
            nb = (q[f"cov_{v}_pct"].shift(1) + q[f"cov_{v}_pct"].shift(-1)) / 2.0
            q[f"locdev_{v}_pp"] = q[f"cov_{v}_pct"] - nb
        # US share of ALL funds reporting that quarter (the trend baseline).
        q["us_share_reporting_pct"] = (q["n_us_funds_reporting"] /
                                       q["n_funds_reporting"].replace(0, pd.NA)) * 100
    q["us_share_old_pct"] = (q["n_funds_us_old"] /
                             q["n_funds_old"].replace(0, pd.NA)) * 100
    q["us_share_new_pct"] = (q["n_funds_us_new"] /
                             q["n_funds_new"].replace(0, pd.NA)) * 100
    if have_denom:
        # SELECTION TILT: how much the selection rule itself tilts investor
        # composition, net of the secular trend in the US share.
        q["tilt_old_pp"] = q["us_share_old_pct"] - q["us_share_reporting_pct"]
        q["tilt_new_pp"] = q["us_share_new_pct"] - q["us_share_reporting_pct"]

    cols = ["quarter", "report_date", "dow", "is_weekend",
            "n_rows_old", "n_rows_new", "rows_pct_change",
            "n_funds_old", "n_funds_new",
            "us_share_old_pct", "us_share_new_pct"]
    if have_denom:
        cols[7:7] = ["n_funds_reporting"]
        cols += ["us_share_reporting_pct", "tilt_old_pp", "tilt_new_pp",
                 "cov_old_pct", "cov_new_pct", "cov_delta_pp",
                 "locdev_old_pp", "locdev_new_pp"]
    q[cols].to_csv(OUT / "diag_p0_quarter_coverage.csv", index=False)
    print("  -> diag_p0_quarter_coverage.csv (every quarter, with weekday)")
    with pd.option_context("display.width", 200, "display.max_rows", 200,
                           "display.float_format", lambda v: f"{v:,.2f}"):
        print(q[cols].to_string(index=False))

    if have_denom:
        # ------------------------------------------------------------------
        # GATE 1 — coverage gap closure, PER ERA + trend-free
        # ------------------------------------------------------------------
        print("\n" + "=" * 90)
        print("GATE 1 — weekday-vs-weekend coverage gap closure (PER ERA, not pooled)")
        print("=" * 90)

        era_rows = []
        for lab, y0, y1 in ERAS:
            s = q[(q["year"] >= y0) & (q["year"] <= y1)]
            wd, we = s[~s["is_weekend"]], s[s["is_weekend"]]
            if len(wd) == 0 or len(we) == 0:
                continue
            era_rows.append(dict(
                era=lab, n_wd=len(wd), n_we=len(we),
                wd_old=wd["cov_old_pct"].mean(), we_old=we["cov_old_pct"].mean(),
                gap_old_pp=wd["cov_old_pct"].mean() - we["cov_old_pct"].mean(),
                wd_new=wd["cov_new_pct"].mean(), we_new=we["cov_new_pct"].mean(),
                gap_new_pp=wd["cov_new_pct"].mean() - we["cov_new_pct"].mean(),
                wd_drift_pp=wd["cov_delta_pp"].mean(),
                we_drift_pp=we["cov_delta_pp"].mean(),
                we_worst_new=we["cov_new_pct"].min(),
                artifact_old_pp=(wd["locdev_old_pp"].mean() -
                                 we["locdev_old_pp"].mean()),
                artifact_new_pp=(wd["locdev_new_pp"].mean() -
                                 we["locdev_new_pp"].mean()),
            ))
        g1 = pd.DataFrame(era_rows)
        g1.to_csv(OUT / "diag_p0_gate1_by_era.csv", index=False)
        with pd.option_context("display.width", 220,
                               "display.float_format", lambda v: f"{v:,.2f}"):
            print(g1.to_string(index=False))
        print("  -> diag_p0_gate1_by_era.csv")
        print("\n  gap_*_pp    = weekday-mean minus weekend-mean coverage (raw).")
        print("  artifact_*_pp = the SAME contrast on the trend-free local deviation")
        print("                  (quarter coverage minus the mean of its two neighbours).")
        print("  wd_drift_pp = weekday coverage change old->new. EXPECTED POSITIVE —")
        print("                early reporters are now included. NOT a failure.")

        for r in era_rows:
            lab = r["era"]
            if lab == "main-session window 2022-2023":
                continue          # a subset of 'modern'; reported, not gated twice
            disclosed = lab in DISCLOSED_GATE1_FAIL_ERAS
            ok = abs(r["gap_new_pp"]) <= GATE1_MAX_GAP_PP
            note = ""
            if disclosed and not ok:
                note = ("  [DISCLOSED: LionShares ramp-up funds report on scattered "
                        "mid-quarter dates, not the prior business day; no W in the "
                        "plateau closes this — W=3 11.18pp, W=10 11.26pp, W=14 11.29pp]")
            gate(f"1[{lab}] coverage gap closed", ok,
                 f"post-fix {r['gap_new_pp']:+.2f}pp vs pre-fix {r['gap_old_pp']:+.2f}pp "
                 f"(threshold {GATE1_MAX_GAP_PP}pp); trend-free artifact "
                 f"{r['artifact_new_pp']:+.2f}pp vs {r['artifact_old_pp']:+.2f}pp; "
                 f"weekday drift {r['wd_drift_pp']:+.2f}pp{note}",
                 disclosed=disclosed)

        full = next(r for r in era_rows if r["era"] == "FULL 1999-2023")
        gate("1t trend-free calendar artifact removed (full sample)",
             abs(full["artifact_new_pp"]) <= GATE1_MAX_ARTIFACT_PP,
             f"post-fix {full['artifact_new_pp']:+.2f}pp vs pre-fix "
             f"{full['artifact_old_pp']:+.2f}pp (threshold {GATE1_MAX_ARTIFACT_PP}pp)")

        we_all = q[q["is_weekend"]]
        print("\n  Weekend / non-business quarter-ends (derived programmatically):")
        we_tbl = we_all[["quarter", "report_date", "dow", "cov_old_pct",
                         "cov_new_pct", "cov_delta_pp", "n_rows_old", "n_rows_new"]]
        we_tbl.to_csv(OUT / "diag_p0_weekend_quarters.csv", index=False)
        with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
            print(we_tbl.to_string(index=False))
        print("  -> diag_p0_weekend_quarters.csv")

        # ------------------------------------------------------------------
        # GATE 2 — US-share calendar swing, gated on the SELECTION TILT, per era
        # ------------------------------------------------------------------
        print("\n" + "=" * 90)
        print("GATE 2 — US-share calendar swing (gated on the SELECTION TILT, per era)")
        print("=" * 90)
        print("  selection tilt = US share of CAPTURED funds  MINUS  US share of ALL")
        print("  funds reporting that quarter. It removes the secular trend in the US")
        print("  share, which makes the RAW weekday-minus-weekend diff pool to a PASS")
        print("  by cancellation between opposite-signed eras. raw_diff is reported")
        print("  but NOT gated.")
        g2_rows = []
        for lab, y0, y1 in ERAS:
            s = q[(q["year"] >= y0) & (q["year"] <= y1)]
            wd, we = s[~s["is_weekend"]], s[s["is_weekend"]]
            if len(wd) == 0 or len(we) == 0:
                continue
            g2_rows.append(dict(
                era=lab, n_wd=len(wd), n_we=len(we),
                us_share_reporting=s["us_share_reporting_pct"].mean(),
                raw_wd_old=wd["us_share_old_pct"].mean(),
                raw_we_old=we["us_share_old_pct"].mean(),
                raw_diff_old_pp=(wd["us_share_old_pct"].mean() -
                                 we["us_share_old_pct"].mean()),
                raw_wd_new=wd["us_share_new_pct"].mean(),
                raw_we_new=we["us_share_new_pct"].mean(),
                raw_diff_new_pp=(wd["us_share_new_pct"].mean() -
                                 we["us_share_new_pct"].mean()),
                tilt_wd_old=wd["tilt_old_pp"].mean(),
                tilt_we_old=we["tilt_old_pp"].mean(),
                tilt_gap_old_pp=(wd["tilt_old_pp"].mean() - we["tilt_old_pp"].mean()),
                tilt_wd_new=wd["tilt_new_pp"].mean(),
                tilt_we_new=we["tilt_new_pp"].mean(),
                tilt_gap_new_pp=(wd["tilt_new_pp"].mean() - we["tilt_new_pp"].mean()),
            ))
        g2 = pd.DataFrame(g2_rows)
        g2.to_csv(OUT / "diag_p0_gate2_by_era.csv", index=False)
        with pd.option_context("display.width", 250,
                               "display.float_format", lambda v: f"{v:,.2f}"):
            print(g2.to_string(index=False))
        print("  -> diag_p0_gate2_by_era.csv")

        for r in g2_rows:
            lab = r["era"]
            if lab == "main-session window 2022-2023":
                continue
            gate(f"2[{lab}] selection tilt flat across the calendar",
                 abs(r["tilt_gap_new_pp"]) <= GATE2_MAX_TILT_GAP_PP,
                 f"post-fix tilt gap {r['tilt_gap_new_pp']:+.2f}pp vs pre-fix "
                 f"{r['tilt_gap_old_pp']:+.2f}pp (threshold {GATE2_MAX_TILT_GAP_PP}pp); "
                 f"raw US-share diff {r['raw_diff_new_pp']:+.2f}pp vs "
                 f"{r['raw_diff_old_pp']:+.2f}pp (reported, not gated)")
    else:
        print("\nGATE 1 / GATE 2: UNAVAILABLE (no raw denominator). "
              "Re-run without --skip-denominator.")
        results.append(("1 coverage gap closed", FAIL, "UNAVAILABLE"))
        results.append(("2 US-share calendar swing removed", FAIL, "UNAVAILABLE"))

    # ---------------- GATE 3: dollar levels --------------------------------
    print("\n" + "=" * 90)
    print("GATE 3 — US x 28-European-country nominal totals")
    print("  RAW   = SUM(adj_mv) over holdings_eom      (spec-comparable, gated)")
    print("  EQ/AD = SUM(I_ict)  over ownership_ict     "
          + ("(reported alongside)" if have_ict else "(UNAVAILABLE — 04 not re-run)"))
    print("=" * 90)
    dd = (dollars_raw(con, EOM_OLD, "old")
          .merge(dollars_raw(con, EOM_NEW, "new"), on="report_date", how="outer"))
    if have_ict:
        dd = (dd.merge(dollars_ict(con, ICT_OLD, "old"), on="report_date", how="outer")
                .merge(dollars_ict(con, ICT_NEW, "new"), on="report_date", how="outer"))
    dd = dd.sort_values("report_date")
    dd["quarter"] = dd["report_date"].map(quarter_label)
    dd["dow"] = dd["report_date"].dt.day_name().str[:3]
    bases = ("raw", "eqad") if have_ict else ("raw",)
    for base in bases:
        dd[f"{base}_old_T"] = dd[f"usd_{base}_old"] / 1e12
        dd[f"{base}_new_T"] = dd[f"usd_{base}_new"] / 1e12
        dd[f"{base}_pct_change"] = (dd[f"usd_{base}_new"] /
                                    dd[f"usd_{base}_old"] - 1) * 100
        for v in ("old", "new"):
            dd[f"{base}_yoy_{v}_pct"] = (dd[f"usd_{base}_{v}"] /
                                         dd[f"usd_{base}_{v}"].shift(4) - 1) * 100

    keep = ["quarter", "report_date", "dow",
            "raw_old_T", "raw_new_T", "raw_pct_change",
            "raw_yoy_old_pct", "raw_yoy_new_pct"]
    if have_ict:
        dd["eqad_over_raw_new"] = dd["usd_eqad_new"] / dd["usd_raw_new"]
        keep += ["eqad_old_T", "eqad_new_T", "eqad_pct_change",
                 "eqad_yoy_old_pct", "eqad_yoy_new_pct",
                 "eqad_over_raw_new", "n_firms_old", "n_firms_new"]
    dd[keep].to_csv(OUT / "diag_p0_us_eu_totals.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200,
                           "display.float_format", lambda v: f"{v:,.3f}"):
        print(dd[keep].to_string(index=False))
    print("  -> diag_p0_us_eu_totals.csv")

    idx = dd.set_index("quarter")

    def val(qtr: str, col: str):
        return float(idx.loc[qtr, col]) if qtr in idx.index else float("nan")

    # NOTE: takes a CALLABLE, not a formatted string. Python evaluates call
    # arguments eagerly, so passing an f-string that reads usd_eqad_* would
    # KeyError before the have_ict check could suppress it.
    def eqad_note(fn) -> str:
        if not have_ict:
            return "; EQ/AD unavailable (04 not re-run)"
        return "; " + fn()

    t22 = val("2022Q4", "usd_raw_new")
    gate("3a 2022Q4 RAW total in $1.85-1.95T",
         GATE3_Q4_2022_LO <= t22 <= GATE3_Q4_2022_HI,
         f"RAW ${t22/1e12:.3f}T (was ${val('2022Q4','usd_raw_old')/1e12:.3f}T)"
         + eqad_note(lambda: f"EQ/AD ${val('2022Q4','usd_eqad_new')/1e12:.3f}T vs "
                             f"wedge-scaled band "
                             f"[{GATE3_Q4_2022_LO*(1-EQAD_WEDGE)/1e12:.3f}, "
                             f"{GATE3_Q4_2022_HI*(1-EQAD_WEDGE)/1e12:.3f}]T"))

    y22 = val("2022Q4", "raw_yoy_new_pct") / 100.0
    gate("3b 2022Q4 nominal YoY approximately -21% (+/-2pp)",
         GATE3_YOY_LO <= y22 <= GATE3_YOY_HI,
         f"RAW {y22*100:+.1f}% (was {val('2022Q4','raw_yoy_old_pct'):+.1f}%)"
         + eqad_note(lambda: f"EQ/AD {val('2022Q4','eqad_yoy_new_pct'):+.1f}% "
                             f"(was {val('2022Q4','eqad_yoy_old_pct'):+.1f}%)"))

    t21n, t21o = val("2021Q4", "usd_raw_new"), val("2021Q4", "usd_raw_old")
    rel21 = abs(t21n - t21o) / t21o if t21o else float("nan")
    gate("3c 2021Q4 RAW total approximately unchanged (~$2.40T)",
         (GATE3_Q4_2021_LO <= t21n <= GATE3_Q4_2021_HI)
         and rel21 <= GATE3_Q4_2021_MAX_REL_CHANGE,
         f"RAW ${t21n/1e12:.3f}T vs ${t21o/1e12:.3f}T, rel change {rel21*100:.2f}% "
         f"(operator threshold {GATE3_Q4_2021_MAX_REL_CHANGE*100:.0f}%)"
         + eqad_note(lambda: f"EQ/AD ${val('2021Q4','usd_eqad_new')/1e12:.3f}T vs "
                             f"${val('2021Q4','usd_eqad_old')/1e12:.3f}T"))

    # ---------------- summary ---------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    disclosed = [r for r in results if r[1] == DISCLOSED]
    if disclosed:
        print("\n  DISCLOSED ERA-LEVEL FAILURES (known, pre-measured, not a")
        print("  regression from this rebuild — must be carried into the write-up,")
        print("  NOT absorbed into a pooled PASS):")
        for name, _, detail in disclosed:
            print(f"    [{DISCLOSED}] {name} — {detail}")
        print("")

    for name, status, detail in results:
        print(f"  [{status}] {name} — {detail}")

    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_disc = len(disclosed)
    print(f"\n{n_pass}/{len(results)} gates PASS, {n_fail} FAIL, "
          f"{n_disc} DISCLOSED-FAIL.")
    if n_fail:
        print("A FAIL is a STOP: do not run the downstream chain on this panel.")
    print("Artifacts written (all new files; no pipeline artifact was modified):")
    for f in ("diag_p0_quarter_coverage.csv", "diag_p0_weekend_quarters.csv",
              "diag_p0_gate1_by_era.csv", "diag_p0_gate2_by_era.csv",
              "diag_p0_us_eu_totals.csv", "diag_p0_asof_gap_by_quarter.csv",
              DENOM_CACHE.name):
        p = OUT / f
        if p.is_file():
            print(f"  {p}")
    con.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
