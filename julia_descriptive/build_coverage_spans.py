"""
build_coverage_spans.py — TASK e4: security-coverage existence spans
(the structural-zeros fix for the c6 Cartesian grid).

INTENDED output: output/entity_coverage_spans.parquet with
(sec_entity_id, span_start, span_end) from LISTING/DELISTING dates in
Factset_Security_coverage.gz, replacing the outcome-derived holdings span
(in_span in audit_c6_panel.dta).

VERDICT (2026-08-10, inspection of the delivered extract): NOT BUILDABLE.
=========================================================================
Factset_Security_coverage.gz (own_sec_coverage, Ownership V5 feed) carries
EXACTLY these 13 columns — verified over the full file, 55,792 rows, one
row per FSYM_ID:

  FSYM_ID, SECURITY_NAME, ISO_COUNTRY, MIC_EXCHANGE_CODE, ISSUE_TYPE,
  CAP_GROUP, ADJDATE, VOTES, ACTIVE, UNIVERSE_TYPE,
  FDS_13F_FLAG, FDS_13F_CA_FLAG, FDS_UKSR_FLAG

There is NO listing date, NO delisting date, NO coverage start/end.
The two candidate fields fail on FactSet's own documentation:

  * ADJDATE — per FactSet_Standard_DataFeed_Ownership_V5_UserGuide p.16:
    "The Adjustment Date refers to the date through which corporate
    actions have been applied to ownership related data for a security."
    It is a split/corporate-actions vintage stamp, NOT a delisting date
    (e.g. FISERV, continuously listed, has ADJDATE 2018-03-20 = its last
    2-for-1 split), and there is nothing on the listing side at all.
  * ACTIVE — an UNDATED point-in-time flag (active vs terminated as of
    the 2023-12 feed snapshot). It says WHETHER a security terminated,
    never WHEN.

So this script REFUSES to fabricate entity_coverage_spans.parquet. What
it does instead:
  1. prints the schema + a sample (the loud documentation the task asks
     for when dates are unusable);
  2. runs the honest salvage: an entity-level cross-check of the UNDATED
     ACTIVE flag against the holdings-based span. If firms whose holdings
     end early are overwhelmingly flagged terminated, the holdings-based
     last_active is corroborated by a non-outcome field; firms still
     ACTIVE with early-ending holdings are candidate TRUE extensive-margin
     exits (exactly the observations in_span==1 would wrongly discard).
     It also tests, and reports, whether ADJDATE could proxy delisting
     (spoiler quantified below in the census output).
  3. writes the census to output/coverage_active_crosscheck.csv
     (entity level) + prints the summary block.

What WOULD unblock the real fix: FactSet Symbology sym_coverage
(fref_listing_termination / first & last trade dates), own_ent_coverage,
or Compustat Global secd / Datastream delisting dates, mapped to
sec_entity_id.

Left edge note: NOTHING in the delivered ownership bundle dates listings,
so pre-listing zeros cannot be identified from these files at all — the
left edge of the span can only come from an external symbology/pricing
source.
"""

import os
import sys
from pathlib import Path
import duckdb

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
_env_out = os.environ.get("DPN_OUT_DIR", "").strip()
OUT = Path(_env_out).resolve() if _env_out else PROJ / "output"
DATA_ROOT = Path(os.environ.get("DPN_DATA_ROOT", r"E:\Data\Data"))
COV = (DATA_ROOT / "Factset Ownership" / "Factset_Security_coverage.gz").as_posix()
EOM = (OUT / "holdings_eom.parquet").as_posix()
GRID = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
CENSUS = OUT / "coverage_active_crosscheck.csv"
SPANS_TARGET = OUT / "entity_coverage_spans.parquet"

EU = ("GB", "DE", "FR", "NL", "CH", "IT", "ES", "SE", "DK", "NO", "FI",
      "BE", "AT", "IE", "LU", "PT", "PL", "CZ", "HU", "GR", "RO", "SK",
      "SI", "BG", "HR", "EE", "LV", "LT")
EU_SQL = "(" + ",".join(f"'{c}'" for c in EU) + ")"

con = duckdb.connect()
con.execute("SET temp_directory='E:/duckdb_tmp'")
con.execute("SET memory_limit='8GB'")
con.execute("SET threads=4")

BANNER = "!" * 74
print(BANNER)
print("!! build_coverage_spans.py — COVERAGE FILE HAS NO LISTING/DELISTING DATES !!")
print("!! entity_coverage_spans.parquet is NOT emitted (refusing to fabricate).  !!")
print("!! ADJDATE = corporate-actions vintage (FactSet V5 guide p.16), ACTIVE =  !!")
print("!! undated snapshot flag. See module docstring for the unblock list.      !!")
print(BANNER)

print("\n[1] Coverage file schema + sample (own_sec_coverage):")
cov_rel = f"read_csv('{COV}', header=true)"
print(con.execute(f"DESCRIBE SELECT * FROM {cov_rel}").df().to_string(index=False))
print(con.execute(f"SELECT * FROM {cov_rel} LIMIT 10").df().to_string(index=False))
tot = con.execute(f"""
    SELECT COUNT(*) AS rows, COUNT(DISTINCT FSYM_ID) AS fsyms,
           SUM(CASE WHEN ACTIVE = 1 THEN 1 ELSE 0 END) AS n_active,
           MIN(ADJDATE) AS adj_min, MAX(ADJDATE) AS adj_max
    FROM {cov_rel}
""").df()
print(tot.to_string(index=False))

# --------------------------------------------------------------------------
# [2] Entity-level ACTIVE-flag cross-check against the holdings-based span.
# Bridge: sec_entity_id -> fsym_id pairs observed in holdings_eom (the same
# source the grid firms come from; for never-held securities no bridge to
# entities exists in the delivered bundle — one more reason the full C6
# universe fix needs a symbology delivery).
# --------------------------------------------------------------------------
print("\n[2] Entity-level cross-check: ACTIVE flag vs holdings span ...")
con.execute(f"""
CREATE TEMP TABLE span AS
SELECT sec_entity_id,
       MIN(CASE WHEN I_ict > 0 THEN report_date END) AS first_active,
       MAX(CASE WHEN I_ict > 0 THEN report_date END) AS last_active
FROM read_parquet('{GRID}')
GROUP BY 1
""")
qr = con.execute(f"""
    SELECT MIN(report_date) AS q_min, MAX(report_date) AS q_max,
           COUNT(DISTINCT report_date) AS n_q, COUNT(DISTINCT sec_entity_id) AS n_ent
    FROM read_parquet('{GRID}')
""").df()
print("  grid:", qr.to_string(index=False))
q_max = qr["q_max"].iloc[0]

con.execute(f"""
CREATE TEMP TABLE bridge AS
SELECT DISTINCT sec_entity_id, fsym_id, sec_country
FROM read_parquet('{EOM}')
WHERE sec_entity_id IN (SELECT sec_entity_id FROM span)
""")

con.execute(f"""
CREATE TEMP TABLE ent_cov AS
SELECT b.sec_entity_id,
       COUNT(DISTINCT b.fsym_id)                                    AS n_fsyms,
       COUNT(DISTINCT CASE WHEN b.sec_country IN {EU_SQL}
                           THEN b.fsym_id END)                      AS n_fsyms_eu,
       COUNT(DISTINCT c.FSYM_ID)                                    AS n_matched,
       MAX(CASE WHEN c.ACTIVE = 1 THEN 1 ELSE 0 END)                AS any_active,
       MAX(CASE WHEN b.sec_country IN {EU_SQL} AND c.ACTIVE = 1
                THEN 1 ELSE 0 END)                                  AS any_active_eu,
       MAX(c.ADJDATE)                                               AS max_adjdate
FROM bridge b
LEFT JOIN {cov_rel} c ON c.FSYM_ID = b.fsym_id
GROUP BY 1
""")

census = con.execute(f"""
SELECT s.sec_entity_id, s.first_active, s.last_active,
       e.n_fsyms, e.n_fsyms_eu, e.n_matched, e.any_active, e.any_active_eu,
       e.max_adjdate,
       CASE WHEN s.last_active < DATE '{q_max}' THEN 1 ELSE 0 END AS early_ender,
       DATEDIFF('day', s.last_active, e.max_adjdate)               AS adj_minus_last_days
FROM span s LEFT JOIN ent_cov e USING (sec_entity_id)
ORDER BY s.sec_entity_id
""").df()
census.to_csv(CENSUS, index=False)
print(f"  wrote {CENSUS}  ({len(census):,} entities)")

n = len(census)
matched = census["n_matched"].fillna(0) > 0
print(f"\n  entities in grid                 : {n:,}")
print(f"  matched to >=1 coverage row      : {matched.sum():,} ({matched.mean():.1%})")

ee = census[matched & (census["early_ender"] == 1)]
sv = census[matched & (census["early_ender"] == 0)]
print(f"\n  EARLY-ENDERS (last_active < {q_max}): {len(ee):,}")
print(f"    all fsyms terminated (ACTIVE=0)  : {(ee['any_active'] == 0).sum():,} "
      f"({(ee['any_active'] == 0).mean():.1%})  <- exit corroborated by termination")
print(f"    >=1 fsym still ACTIVE            : {(ee['any_active'] == 1).sum():,} "
      f"({(ee['any_active'] == 1).mean():.1%})  <- candidate TRUE extensive-margin exit"
      f" (in_span==1 restriction discards their post-exit zeros)")
print(f"  SURVIVORS-TO-END: {len(sv):,};  still ACTIVE: {(sv['any_active'] == 1).mean():.1%}")

# ADJDATE-as-delisting-proxy test, among terminated early-enders only.
sub = ee[(ee["any_active"] == 0)].dropna(subset=["adj_minus_last_days"])
if len(sub):
    q = sub["adj_minus_last_days"].quantile([.1, .25, .5, .75, .9])
    within = (sub["adj_minus_last_days"].abs() <= 370).mean()
    print(f"\n  ADJDATE-proxy test (terminated early-enders, n={len(sub):,}):")
    print("    ADJDATE - last_active, days: "
          + "  ".join(f"p{int(p * 100)}={v:,.0f}" for p, v in q.items()))
    print(f"    share within +/-370 days of last_active: {within:.1%}")
    print("    (read: if this is far from 100%, ADJDATE does NOT date the exit "
          "and cannot right-truncate spans)")

late = census[census["first_active"] > census["first_active"].min()]
print(f"\n  LEFT-EDGE (unfixable with this file): {len(late):,} entities "
      f"({len(late) / n:.1%}) first appear after the panel start; their "
      f"pre-first_active zeros cannot be split into pre-listing vs true "
      f"non-holding without an external listing-date source.")

assert not SPANS_TARGET.exists(), (
    f"{SPANS_TARGET} exists but this script never writes it - "
    "a stale/foreign artifact is sitting on the canonical name; investigate."
)
print("\n" + BANNER)
print("!! NOT WRITTEN: output/entity_coverage_spans.parquet - no usable dates.  !!")
print("!! run_headline_spans.do therefore runs samples (a) full grid and        !!")
print("!! (b) holdings-span only; its (c) coverage-span row is emitted as       !!")
print("!! blocked until a symbology delivery with listing/termination dates.    !!")
print(BANNER)
