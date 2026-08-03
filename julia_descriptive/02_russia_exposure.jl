# 02_russia_exposure.jl
# ISOMORPHIC copy of 02_china_exposure.jl for the Russia positive control
# (Second/Third external review round, R3-A #1 / P0 #3). ONLY the
# counterparty-region literal changes ('CN' -> 'RU', both WHERE clauses);
# every identifier is renamed china->russia / cn->ru so no column silently
# holds Russia counts under a China-labeled name. EU universe, as-of logic,
# quarter calendar, and the symmetric denominator (eu_any_edge) are
# byte-identical to the China run and reused (not re-persisted).

# 02_china_exposure.jl
# Section 5 — build Russia exposure measure from Revere supply chain data.
# Uses both directions (Path 1: EU=source, CN=target; Path 2: CN=source, EU=target).
# Runs in ~3-5 minutes on 8-10GB RAM machine.
#
# Audited 2026-06-01 — see AUDIT_2026_06_01_julia_descriptive.md.
# Critical-fix changes vs prior version:
#
# C5 (MOST_RECENT look-ahead): The previous version dedup'd Revere companies
#     to most-recent row per company_id, then stamped that row's home_region
#     / CUSIP / ISIN / SEDOL onto every quarter back to 2003. A firm that
#     redomiciled to the EU in 2020 entered the EU universe for 2005.
#     Fix: build a time-versioned (rev_co_asof_q) table that resolves each
#     company's attributes AS-OF every quarter-end, and use it for the EU
#     universe membership filter AND for edge SRC/TGT region classification.
#     Static rev_co (now using FIRST_VALUE IGNORE NULLS per field, with a
#     deterministic company_id tiebreaker) is preserved for non-time-sensitive
#     uses but is NEVER used for membership or region classification.
#
# C4 (asymmetric edge counting): The previous eu_any_edge denominator's
#     Path 2 had `WHERE src_region <> tgt_region`, which dropped legitimate
#     EU-EU edges from the target firm's denominator. Numerator (eu_russia_edge)
#     had no analogous filter, so russia_share was inflated for firms with
#     predominantly inbound EU links.
#     Fix: remove the `<>` filter. Each EU firm counts edges where IT is
#     either source or target; EU-EU edges contribute once to each endpoint's
#     denominator — symmetric with how every other edge is counted.
#
# Medium fixes also applied here:
#   - Removed firm_month_china_exposure.parquet alias (silent type-pun;
#     month_end column held quarter-end values). Downstream now reads
#     firm_quarter_russia_exposure.parquet directly.
#   - Added deterministic tiebreaker (, company_id ASC) to every ROW_NUMBER.
#   - Sentinel '4000-01-01' replaced with NULL at load time.
#   - dedup-edge diagnostic on rev_rel.
#   - Atomic writes via atomic_copy_to from 00_setup.

include("00_setup.jl")

# Source paths feeding this step (recorded in manifest)
const STEP_INPUTS = [REVERE_CO_PATH, REVERE_REL_PATH]

con = dbcon()

# ============================================================
# (1) Load Revere company master (RAW). Diagnostics on within-company variation.
# ============================================================
println("Loading Revere company master (raw, time-versioned)...")

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE rev_co_raw AS
    SELECT company_id,
           home_region,
           country,
           cusip,
           isin,
           sedol,
           covered,
           CAST(start_ AS DATE) AS start_d_raw,
           CAST(end_   AS DATE) AS end_d_raw,
           -- Normalise sentinel '4000-01-01' (and similar) to NULL at load time
           -- so downstream BETWEEN checks behave correctly.
           CAST(start_ AS DATE) AS start_d,
           CASE WHEN CAST(end_ AS DATE) >= DATE '4000-01-01' THEN NULL
                ELSE CAST(end_ AS DATE) END AS end_d
    FROM read_csv_auto('$(replace(REVERE_CO_PATH, "\\" => "/"))', sample_size=-1)
""")

n_raw = qdf(con, "SELECT COUNT(*) AS n FROM rev_co_raw").n[1]
n_unique = qdf(con, "SELECT COUNT(DISTINCT company_id) AS n FROM rev_co_raw").n[1]
println("  raw rows: $n_raw ; unique company_id: $n_unique ; avg rows per company: $(round(n_raw/n_unique, digits=2))")

# Sentinel diagnostic
sentinel_diag = qdf(con, """
    SELECT COUNT(*) FILTER (WHERE end_d_raw >= DATE '4000-01-01') AS n_sentinel_end,
           COUNT(*) FILTER (WHERE end_d IS NULL)                  AS n_null_end_post_norm
    FROM rev_co_raw
""")
println("  sentinel end_d normalised:")
println(sentinel_diag)

# Within-company-id variation diagnostic
println("\n--- Within-company-id variation diagnostic ---")
diag = qdf(con, """
    SELECT
        SUM(CASE WHEN n_regions > 1 THEN 1 ELSE 0 END) AS n_co_multi_region,
        SUM(CASE WHEN n_countries > 1 THEN 1 ELSE 0 END) AS n_co_multi_country,
        SUM(CASE WHEN n_cusips > 1 THEN 1 ELSE 0 END) AS n_co_multi_cusip,
        SUM(CASE WHEN n_isins > 1 THEN 1 ELSE 0 END) AS n_co_multi_isin,
        SUM(CASE WHEN n_sedols > 1 THEN 1 ELSE 0 END) AS n_co_multi_sedol,
        COUNT(*) AS n_co_total
    FROM (
        SELECT company_id,
               COUNT(DISTINCT home_region) AS n_regions,
               COUNT(DISTINCT country)     AS n_countries,
               COUNT(DISTINCT cusip)       AS n_cusips,
               COUNT(DISTINCT isin)        AS n_isins,
               COUNT(DISTINCT sedol)       AS n_sedols
        FROM rev_co_raw
        GROUP BY company_id
    )
""")
println(diag)
println("  -> % companies with >1 home_region: $(round(100*diag.n_co_multi_region[1]/diag.n_co_total[1], digits=2))%")
println("  -> % companies with >1 country:     $(round(100*diag.n_co_multi_country[1]/diag.n_co_total[1], digits=2))%")
println("  -> % companies with >1 cusip:       $(round(100*diag.n_co_multi_cusip[1]/diag.n_co_total[1], digits=2))%")
println("  -> % companies with >1 isin:        $(round(100*diag.n_co_multi_isin[1]/diag.n_co_total[1], digits=2))%")
println("  -> % companies with >1 sedol:       $(round(100*diag.n_co_multi_sedol[1]/diag.n_co_total[1], digits=2))%")
CSV.write(joinpath(OUT_DIR, "02_revere_co_variation_diag.csv"), diag)

# ============================================================
# (1a) Static dedup — FIRST_VALUE IGNORE NULLS per field, deterministic.
# Used for ID-based joins where time-invariance is acceptable. NEVER used
# for membership filters or for edge region classification — those use the
# time-versioned rev_co_asof_q built below.
# ============================================================
println("\n--- Static dedup (FIRST_VALUE IGNORE NULLS per field, company_id tiebreaker) ---")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE rev_co AS
    SELECT DISTINCT
        company_id,
        FIRST_VALUE(home_region IGNORE NULLS) OVER w AS home_region,
        FIRST_VALUE(country     IGNORE NULLS) OVER w AS country,
        FIRST_VALUE(cusip       IGNORE NULLS) OVER w AS cusip,
        FIRST_VALUE(isin        IGNORE NULLS) OVER w AS isin,
        FIRST_VALUE(sedol       IGNORE NULLS) OVER w AS sedol
    FROM rev_co_raw
    WINDOW w AS (
        PARTITION BY company_id
        ORDER BY end_d DESC NULLS FIRST, start_d DESC, company_id ASC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )
""")
DBInterface.execute(con, """
    ALTER TABLE rev_co ADD COLUMN ever_covered INTEGER DEFAULT 0
""")
DBInterface.execute(con, """
    UPDATE rev_co SET ever_covered = (
        SELECT MAX(CASE WHEN r.covered='Y' THEN 1 ELSE 0 END)
        FROM rev_co_raw r WHERE r.company_id = rev_co.company_id
    )
""")
n_co = qdf(con, "SELECT COUNT(*) AS n FROM rev_co").n[1]
println("  unique Revere companies after static dedup: $n_co")

# Country breakdown of STATIC home_region (for reference only — universe
# membership uses the time-versioned table below).
co_country = qdf(con, """
    SELECT home_region, COUNT(*) AS n
    FROM rev_co
    WHERE home_region IS NOT NULL
    GROUP BY home_region ORDER BY n DESC LIMIT 25
""")
CSV.write(joinpath(OUT_DIR, "02_revere_company_country_top25.csv"), co_country)

# ============================================================
# (1b) Quarter calendar — single source for time-versioning + edge filter.
# Spans 2003-Q1 (Revere coverage begin) through 2025-Q2.
# ============================================================
const PANEL_START_QUARTER_END = "2003-03-31"
const PANEL_END_QUARTER_END   = "2025-06-30"

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE quarters AS
    SELECT LAST_DAY(MAKE_DATE(y, m, 1)) AS qend
    FROM range(2003, 2026) y(y)
    CROSS JOIN (VALUES (3),(6),(9),(12)) AS month_tbl(m)
    WHERE LAST_DAY(MAKE_DATE(y, m, 1)) BETWEEN DATE '$PANEL_START_QUARTER_END'
                                           AND DATE '$PANEL_END_QUARTER_END'
    ORDER BY qend
""")

# ============================================================
# (1c) Time-versioned company attributes — AS-OF every quarter-end.
# For each (company_id, qend), pick the rev_co_raw row that was active at qend
# (start_d <= qend AND (end_d IS NULL OR end_d >= qend)). If multiple match,
# prefer the row with the latest start_d, then latest end_d, then company_id
# for a deterministic tiebreaker.
#
# This is the CANONICAL source of company attributes (esp. home_region) for
# membership filters and edge classification. NEVER use rev_co for those.
# ============================================================
println("\nBuilding time-versioned rev_co_asof_q (point-in-time attributes per quarter)...")

# (1c-pre) Deterministic dedup at the (company_id, start_d) grain. This is
# shared by rev_co_asof_q and the rev_rel src/tgt joins below. Pre-deduping
# the history avoids tiebreaker ambiguity inside ASOF JOIN — ASOF picks the
# row with the LATEST start_d <= probe; we resolve same-start_d collisions
# here so that ASOF's pick is unique.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE rev_co_dedup AS
    SELECT company_id, home_region, country, cusip, isin, sedol, start_d, end_d
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY company_id, start_d
                   -- NULLS FIRST (2026-07-21 fix, mirrors 02_china_exposure):
                   -- open segment (NULL end_d) beats same-start_d zero-length
                   -- closed row; NULLS LAST silently dropped live firms.
                   -- Downstream parquets NOT yet rebuilt under this fix.
                   ORDER BY end_d DESC NULLS FIRST, company_id ASC
               ) AS rn
        FROM rev_co_raw
    )
    WHERE rn = 1
""")
n_dedup = qdf(con, "SELECT COUNT(*) AS n FROM rev_co_dedup").n[1]
println("  rev_co_dedup rows (one per (company_id, start_d)): $n_dedup")

# (1c) ASOF JOIN version of rev_co_asof_q.
# For each (company_id, qend) probe, ASOF selects the latest history row with
# start_d <= qend. We then filter on end_d to enforce interval coverage —
# equivalent to the original BETWEEN-style join but linear in 'companies x
# quarters' instead of cross-product * ROW_NUMBER cost.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE rev_co_asof_q AS
    WITH co_ids AS (
        SELECT DISTINCT company_id FROM rev_co_dedup
    ),
    probe AS (
        SELECT c.company_id, q.qend
        FROM co_ids c
        CROSS JOIN quarters q
    )
    SELECT p.qend, p.company_id, h.home_region, h.country, h.cusip, h.isin, h.sedol
    FROM probe p
    ASOF LEFT JOIN rev_co_dedup h
      ON p.company_id = h.company_id
     AND p.qend >= h.start_d
    WHERE h.company_id IS NOT NULL
      AND (h.end_d IS NULL OR h.end_d >= p.qend)
""")

# Diagnostic: how often does the as-of attribute differ from static rev_co?
asof_drift = qdf(con, """
    SELECT
        COUNT(*) AS n_pairs,
        SUM(CASE WHEN a.home_region IS NULL THEN 1 ELSE 0 END)            AS n_asof_null_region,
        SUM(CASE WHEN c.home_region <> a.home_region THEN 1 ELSE 0 END)   AS n_region_drift,
        SUM(CASE WHEN c.cusip       <> a.cusip       THEN 1 ELSE 0 END)   AS n_cusip_drift,
        SUM(CASE WHEN c.isin        <> a.isin        THEN 1 ELSE 0 END)   AS n_isin_drift
    FROM rev_co_asof_q a
    JOIN rev_co c USING (company_id)
""")
println("As-of vs static (drift counts; non-zero = real look-ahead in old version):")
println(asof_drift)
CSV.write(joinpath(OUT_DIR, "02_rev_co_asof_drift_diag.csv"), asof_drift)

# ============================================================
# (1d) Supply-chain relationships, with src/tgt attributes joined AS-OF rel_start.
# Each edge classified by what its endpoints were AT THE TIME it began, not
# what they are today. Eliminates the look-ahead in edge classification.
# ============================================================
println("\nLoading supply chain relationships (5.5M rows) + AS-OF rel_start join...")

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE rev_rel_raw AS
    SELECT  r.source_company_id,
            r.target_company_id,
            r.rel_type,
            CAST(r.start_ AS DATE) AS rel_start,
            CASE WHEN CAST(r.end_ AS DATE) >= DATE '4000-01-01' THEN NULL
                 ELSE CAST(r.end_ AS DATE) END AS rel_end,
            r.revenue_percent
    FROM read_csv_auto('$(replace(REVERE_REL_PATH, "\\" => "/"))', sample_size=-1) r
""")

# Dedup-edge diagnostic — per audit
dup_edge_diag = qdf(con, """
    SELECT COUNT(*) AS n_duplicate_edges
    FROM (
        SELECT source_company_id, target_company_id, rel_type, rel_start, rel_end,
               COUNT(*) AS c
        FROM rev_rel_raw
        GROUP BY 1,2,3,4,5
        HAVING COUNT(*) > 1
    )
""")
println("Duplicate-edge diagnostic on rev_rel: $(dup_edge_diag.n_duplicate_edges[1]) duplicate keys")

# For AS-OF lookup at rel_start, use the pre-deduped rev_co_dedup history
# (deterministic tiebreaker). Two chained ASOF LEFT JOINs (src then tgt)
# replace the previous BETWEEN-style join + ROW_NUMBER. We then NULL-out the
# region/cusip/isin/sedol columns when the picked history row's end_d is
# strictly before rel_start (the edge fell into a gap in the company's
# coverage). This preserves the original semantics: only history rows whose
# [start_d, end_d] interval covers rel_start contribute a region tag.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE rev_rel AS
    WITH src_asof AS (
        SELECT r.source_company_id, r.target_company_id, r.rel_type, r.rel_start, r.rel_end,
               r.revenue_percent,
               s.home_region AS src_region_raw,
               s.cusip       AS src_cusip_raw,
               s.isin        AS src_isin_raw,
               s.sedol       AS src_sedol_raw,
               s.end_d       AS src_end_d
        FROM rev_rel_raw r
        ASOF LEFT JOIN rev_co_dedup s
          ON r.source_company_id = s.company_id
         AND r.rel_start >= s.start_d
    ),
    src_picked AS (
        SELECT source_company_id, target_company_id, rel_type, rel_start, rel_end,
               revenue_percent,
               CASE WHEN src_end_d IS NULL OR src_end_d >= rel_start THEN src_region_raw ELSE NULL END AS src_region,
               CASE WHEN src_end_d IS NULL OR src_end_d >= rel_start THEN src_cusip_raw  ELSE NULL END AS src_cusip,
               CASE WHEN src_end_d IS NULL OR src_end_d >= rel_start THEN src_isin_raw   ELSE NULL END AS src_isin,
               CASE WHEN src_end_d IS NULL OR src_end_d >= rel_start THEN src_sedol_raw  ELSE NULL END AS src_sedol
        FROM src_asof
    ),
    tgt_asof AS (
        SELECT sp.*,
               t.home_region AS tgt_region_raw,
               t.cusip       AS tgt_cusip_raw,
               t.isin        AS tgt_isin_raw,
               t.sedol       AS tgt_sedol_raw,
               t.end_d       AS tgt_end_d
        FROM src_picked sp
        ASOF LEFT JOIN rev_co_dedup t
          ON sp.target_company_id = t.company_id
         AND sp.rel_start >= t.start_d
    )
    SELECT source_company_id, target_company_id, rel_type, rel_start, rel_end,
           revenue_percent,
           src_region, src_cusip, src_isin, src_sedol,
           CASE WHEN tgt_end_d IS NULL OR tgt_end_d >= rel_start THEN tgt_region_raw ELSE NULL END AS tgt_region,
           CASE WHEN tgt_end_d IS NULL OR tgt_end_d >= rel_start THEN tgt_cusip_raw  ELSE NULL END AS tgt_cusip,
           CASE WHEN tgt_end_d IS NULL OR tgt_end_d >= rel_start THEN tgt_isin_raw   ELSE NULL END AS tgt_isin,
           CASE WHEN tgt_end_d IS NULL OR tgt_end_d >= rel_start THEN tgt_sedol_raw  ELSE NULL END AS tgt_sedol
    FROM tgt_asof
""")
n_rel = qdf(con, "SELECT COUNT(*) AS n FROM rev_rel").n[1]
println("  relationships (with AS-OF rel_start regions): $n_rel")

# ============================================================
# (2) Russia-edge view: i has a RU counterparty (either side). Region tags
# are AS-OF the edge's rel_start, so an EU firm that became EU in 2020 does
# NOT have its 2010 edges retroactively reclassified.
# ============================================================
println("\nBuilding Russia-edge view (Path 1 ∪ Path 2)...")

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE eu_russia_edge AS
    -- Path 1: EU company i is source, RU counterparty is target
    SELECT source_company_id AS eu_company_id,
           target_company_id AS ru_company_id,
           rel_type,
           rel_start, rel_end,
           src_cusip AS eu_cusip,
           src_isin AS eu_isin,
           src_sedol AS eu_sedol,
           'EU_SRC' AS path
    FROM rev_rel
    WHERE src_region IN $EU_SQL_TUPLE AND tgt_region = 'RU'

    UNION ALL

    -- Path 2: CN company is source, EU company i is target
    SELECT target_company_id AS eu_company_id,
           source_company_id AS ru_company_id,
           rel_type,
           rel_start, rel_end,
           tgt_cusip AS eu_cusip,
           tgt_isin AS eu_isin,
           tgt_sedol AS eu_sedol,
           'CN_SRC' AS path
    FROM rev_rel
    WHERE src_region = 'RU' AND tgt_region IN $EU_SQL_TUPLE
""")

stats = qdf(con, """
    SELECT
        path,
        COUNT(*) AS n_relations,
        COUNT(DISTINCT eu_company_id) AS n_eu,
        COUNT(DISTINCT ru_company_id) AS n_cn
    FROM eu_russia_edge GROUP BY path
""")
println("\nPath breakdown:")
println(stats)
CSV.write(joinpath(OUT_DIR, "02_russia_edge_path_summary.csv"), stats)

# Union counts
union_stats = qdf(con, """
    SELECT
        COUNT(*) AS total_relations,
        COUNT(DISTINCT eu_company_id) AS unique_eu_with_cn,
        COUNT(DISTINCT ru_company_id) AS unique_ru_with_eu
    FROM eu_russia_edge
""")
println("\nUnion totals:")
println(union_stats)

# ============================================================
# (3) EU "any-edge" view — SYMMETRIC fix (C4).
# Every relation where AT LEAST ONE side is an EU firm contributes once to
# each EU endpoint's denominator. EU-EU edges contribute once to BOTH
# endpoints' denominators (was previously dropped from the target endpoint's
# denominator via `src_region <> tgt_region`, inflating russia_share for
# inbound-link firms).
# ============================================================
println("\nBuilding EU any-edge view (SYMMETRIC denominator)...")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE eu_any_edge AS
    -- EU company i is source, any counterparty is target
    SELECT source_company_id AS eu_company_id,
           rel_type,
           rel_start, rel_end,
           'EU_SRC' AS path
    FROM rev_rel
    WHERE src_region IN $EU_SQL_TUPLE

    UNION ALL

    -- Any company is source, EU company i is target.
    -- (C4 FIX): no `src_region <> tgt_region` filter. EU-EU edges are
    -- counted once on each endpoint, symmetric with EU-RU edges.
    SELECT target_company_id AS eu_company_id,
           rel_type,
           rel_start, rel_end,
           'EU_TGT' AS path
    FROM rev_rel
    WHERE tgt_region IN $EU_SQL_TUPLE
""")
any_stats = qdf(con, """
    SELECT COUNT(*) AS n_relations,
           COUNT(DISTINCT eu_company_id) AS n_eu_companies
    FROM eu_any_edge
""")
println("EU any-edge totals (after C4 symmetric fix):")
println(any_stats)

# C4 verification: every EU-EU edge appears in BOTH endpoints' denominators.
c4_check = qdf(con, """
    WITH eu_eu_edges AS (
        SELECT source_company_id, target_company_id, rel_type, rel_start, rel_end
        FROM rev_rel
        WHERE src_region IN $EU_SQL_TUPLE AND tgt_region IN $EU_SQL_TUPLE
    ),
    src_seen AS (
        SELECT COUNT(*) AS n FROM eu_eu_edges e
        JOIN eu_any_edge a
          ON a.eu_company_id = e.source_company_id
         AND a.rel_type = e.rel_type
         AND a.rel_start = e.rel_start
         AND (a.rel_end = e.rel_end OR (a.rel_end IS NULL AND e.rel_end IS NULL))
         AND a.path = 'EU_SRC'
    ),
    tgt_seen AS (
        SELECT COUNT(*) AS n FROM eu_eu_edges e
        JOIN eu_any_edge a
          ON a.eu_company_id = e.target_company_id
         AND a.rel_type = e.rel_type
         AND a.rel_start = e.rel_start
         AND (a.rel_end = e.rel_end OR (a.rel_end IS NULL AND e.rel_end IS NULL))
         AND a.path = 'EU_TGT'
    ),
    eu_eu_total AS (SELECT COUNT(*) AS n FROM eu_eu_edges)
    SELECT (SELECT n FROM eu_eu_total)  AS n_eu_eu_edges,
           (SELECT n FROM src_seen)     AS n_eu_eu_in_src_path,
           (SELECT n FROM tgt_seen)     AS n_eu_eu_in_tgt_path
""")
println("C4 symmetric-counting verification:")
println(c4_check)

# ============================================================
# (4) EU Revere universe — time-versioned. For each quarter q, the set of
# Revere companies whose home_region AT q is in the EU country list.
# (C5 fix.)
# ============================================================
println("\nBuilding time-versioned EU Revere universe (per quarter)...")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE eu_revere_universe_qend AS
    SELECT a.company_id AS eu_company_id,
           a.qend,
           a.home_region AS eu_home_region,
           a.cusip       AS eu_cusip,
           a.isin        AS eu_isin,
           a.sedol       AS eu_sedol
    FROM rev_co_asof_q a
    WHERE a.home_region IN $EU_SQL_TUPLE
""")

eu_univ_qend_stats = qdf(con, """
    SELECT
        COUNT(*) AS n_company_quarters,
        COUNT(DISTINCT eu_company_id) AS n_eu_companies_ever,
        MIN(qend) AS first_qend,
        MAX(qend) AS last_qend
    FROM eu_revere_universe_qend
""")
println("EU Revere universe (time-versioned) stats:")
println(eu_univ_qend_stats)

# For downstream consumers that need a static snapshot (the SAMPLE END-DATE
# universe), we also emit a flat parquet. Note: this is for documentation
# only — sample-membership joins MUST use eu_revere_universe_qend AS-OF the
# joining quarter.
universe_static_path = test_suffix_path(joinpath(OUT_DIR, "eu_revere_universe.parquet"))
atomic_copy_to(con, """
    SELECT DISTINCT
        u.eu_company_id,
        FIRST_VALUE(u.eu_home_region) OVER w AS eu_home_region,
        FIRST_VALUE(u.eu_cusip)       OVER w AS eu_cusip,
        FIRST_VALUE(u.eu_isin)        OVER w AS eu_isin,
        FIRST_VALUE(u.eu_sedol)       OVER w AS eu_sedol
    FROM eu_revere_universe_qend u
    WINDOW w AS (PARTITION BY u.eu_company_id ORDER BY u.qend DESC
                 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
""", universe_static_path)
println("  -> latest-snapshot universe saved to $(basename(universe_static_path))")

# Also emit the time-versioned panel — this is what 05 MUST use.
universe_qend_path = test_suffix_path(joinpath(OUT_DIR, "eu_revere_universe_qend.parquet"))
# universe identical to the China run; not re-persisted here (Russia copy)
println("  -> time-versioned universe saved to $(basename(universe_qend_path))")
write_manifest("02_eu_revere_universe_qend", universe_qend_path;
               row_count=eu_univ_qend_stats.n_company_quarters[1],
               input_paths=STEP_INPUTS)

# ============================================================
# (5) Firm-QUARTER Russia exposure panel — uses time-versioned EU universe.
# A firm i appears in the panel for quarter q ONLY if it was an EU firm at q
# (home_region AS-OF q in EU). Pre-2003 quarters are not produced; downstream
# must preserve NULL for those.
# ============================================================
println("\nBuilding firm-quarter exposure panel (2003-Q1 to 2025-Q2, EU AS-OF q)...")

# CN counts per (firm, quarter) — restricted to EU firms AS-OF q.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE firm_quarter_cn AS
    SELECT
        e.eu_company_id,
        u.eu_cusip,
        u.eu_isin,
        u.eu_sedol,
        q.qend,
        SUM(CASE WHEN e.rel_type = 'CUSTOMER' THEN 1 ELSE 0 END) AS n_ru_customer,
        SUM(CASE WHEN e.rel_type = 'SUPPLIER' THEN 1 ELSE 0 END) AS n_ru_supplier,
        SUM(CASE WHEN e.rel_type = 'PARTNER-JVENTUR' THEN 1 ELSE 0 END) AS n_ru_jv,
        SUM(CASE WHEN e.rel_type = 'PARTNER-MANUFAC' THEN 1 ELSE 0 END) AS n_ru_manuf,
        SUM(CASE WHEN e.rel_type LIKE 'PARTNER%' THEN 1 ELSE 0 END) AS n_ru_partner_any,
        COUNT(*) AS n_ru_total
    FROM eu_russia_edge e
    JOIN quarters q
      ON q.qend >= e.rel_start
     AND (e.rel_end IS NULL OR q.qend <= e.rel_end)
    JOIN eu_revere_universe_qend u
      ON u.eu_company_id = e.eu_company_id
     AND u.qend          = q.qend
    GROUP BY e.eu_company_id, u.eu_cusip, u.eu_isin, u.eu_sedol, q.qend
""")

# Total link counts per (firm, quarter) — DENOMINATOR for share. SYMMETRIC.
# Also restricted to EU firms AS-OF q.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE firm_quarter_total AS
    SELECT
        a.eu_company_id,
        q.qend,
        COUNT(*) AS n_total_links,
        -- (B7 FIX, 2026-08-03) supply-chain-only denominator: CUSTOMER +
        -- SUPPLIER edges only, dropping COMPETITOR and all PARTNER-* types.
        -- Mirrors the China script's B7 fix (02_china_exposure.jl). Matches
        -- Figure 1's descriptive universe and the doc §4.2 estimand
        -- ("supply-chain dependence"). n_total_links kept as an
        -- all-relationship diagnostic.
        COUNT(*) FILTER (WHERE a.rel_type IN ('CUSTOMER','SUPPLIER'))
            AS n_supplychain_links
    FROM eu_any_edge a
    JOIN quarters q
      ON q.qend >= a.rel_start
     AND (a.rel_end IS NULL OR q.qend <= a.rel_end)
    JOIN eu_revere_universe_qend u
      ON u.eu_company_id = a.eu_company_id
     AND u.qend          = q.qend
    GROUP BY a.eu_company_id, q.qend
""")

# Join to compute share. Firms in the universe with NO supply-chain link
# (CN or otherwise) at q do not appear; downstream treats their russia_share
# as NULL (NOT zero), per pre-2003 convention.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE firm_quarter_exposure AS
    SELECT
        t.eu_company_id,
        u.eu_cusip,
        u.eu_isin,
        u.eu_sedol,
        t.qend AS quarter_end,
        COALESCE(c.n_ru_customer, 0) AS n_ru_customer,
        COALESCE(c.n_ru_supplier, 0) AS n_ru_supplier,
        COALESCE(c.n_ru_jv,       0) AS n_ru_jv,
        COALESCE(c.n_ru_manuf,    0) AS n_ru_manuf,
        COALESCE(c.n_ru_partner_any, 0) AS n_ru_partner_any,
        COALESCE(c.n_ru_total,    0) AS n_ru_total,
        t.n_total_links,
        t.n_supplychain_links,
        -- (B7 FIX, 2026-08-03) russia_share = RU supply-chain links / total
        -- supply-chain links (CUSTOMER + SUPPLIER, both endpoints). Excludes
        -- COMPETITOR and PARTNER-*. Mirrors the China script's B7 fix exactly.
        -- Firms with supply-chain links but none to Russia get 0; firms with
        -- only competitor/partner links get NULL (undefined supply-chain
        -- exposure), correctly dropped downstream.
        CAST(COALESCE(c.n_ru_customer, 0) + COALESCE(c.n_ru_supplier, 0) AS DOUBLE)
            / NULLIF(t.n_supplychain_links, 0) AS russia_share,
        -- legacy all-relationship-type share, retained for diagnostics only
        CAST(COALESCE(c.n_ru_total, 0) AS DOUBLE) / NULLIF(t.n_total_links, 0)
            AS russia_share_alltypes
    FROM firm_quarter_total t
    JOIN eu_revere_universe_qend u
      ON u.eu_company_id = t.eu_company_id AND u.qend = t.qend
    LEFT JOIN firm_quarter_cn c
        ON t.eu_company_id = c.eu_company_id AND t.qend = c.qend
""")

n_panel = qdf(con, "SELECT COUNT(*) AS n FROM firm_quarter_exposure").n[1]
n_firms = qdf(con, "SELECT COUNT(DISTINCT eu_company_id) AS n FROM firm_quarter_exposure").n[1]
n_exposed = qdf(con, "SELECT COUNT(*) AS n FROM firm_quarter_exposure WHERE n_ru_total > 0").n[1]
println("  panel rows: $n_panel ; unique EU firms with any supply-chain link: $n_firms ; rows with positive RU exposure: $n_exposed")

# Save panel as parquet for downstream use (atomic write)
firm_quarter_path = test_suffix_path(joinpath(OUT_DIR, "firm_quarter_russia_exposure.parquet"))
atomic_copy_to(con, "SELECT * FROM firm_quarter_exposure", firm_quarter_path)
println("  -> saved to $(basename(firm_quarter_path))")
write_manifest("02_firm_quarter_russia_exposure", firm_quarter_path;
               row_count=n_panel, input_paths=STEP_INPUTS)

# NOTE: previous version also emitted firm_month_china_exposure.parquet as an
# "alias" — but it contained QUARTER-END dates under a column named month_end,
# which silently misled downstream readers. The alias has been REMOVED per
# audit recommendation. 05 now reads firm_quarter_russia_exposure.parquet
# directly.

# ============================================================
# (6) Descriptive: distribution of share-based exposure at one snapshot
# ============================================================
const SNAP_QUARTER = "2018-12-31"
println("\n========== Descriptive snapshot: $SNAP_QUARTER ==========")

snap = qdf(con, """
    SELECT n_ru_total,
           COUNT(*) AS n_firms
    FROM firm_quarter_exposure
    WHERE quarter_end = DATE '$SNAP_QUARTER' AND n_ru_total > 0
    GROUP BY n_ru_total
    ORDER BY n_ru_total
""")
println("Distribution of CN relation count ($SNAP_QUARTER, firms with positive RU exposure only):")
println(first(snap, 30))
CSV.write(joinpath(OUT_DIR, "02_dist_ru_total_2018.csv"), snap)

# Summary percentiles for both the count and the new share measure
pct = qdf(con, """
    SELECT
        COUNT(*) AS n_firms,
        AVG(n_ru_total) AS mean_count,
        QUANTILE_CONT(n_ru_total, 0.5) AS count_p50,
        QUANTILE_CONT(n_ru_total, 0.75) AS count_p75,
        QUANTILE_CONT(n_ru_total, 0.90) AS count_p90,
        MAX(n_ru_total) AS count_max,
        AVG(russia_share) AS mean_share,
        QUANTILE_CONT(russia_share, 0.5) AS share_p50,
        QUANTILE_CONT(russia_share, 0.75) AS share_p75,
        QUANTILE_CONT(russia_share, 0.90) AS share_p90,
        MAX(russia_share) AS share_max
    FROM firm_quarter_exposure
    WHERE quarter_end = DATE '$SNAP_QUARTER' AND n_total_links > 0
""")
println("\nPercentiles ($SNAP_QUARTER):")
println(pct)

# Time series: per-quarter mean China share and mean CN relation count
ts = qdf(con, """
    SELECT quarter_end,
           COUNT(*) FILTER (WHERE n_ru_total > 0)            AS n_firms_with_cn,
           COUNT(*)                                          AS n_firms_with_any_link,
           AVG(n_ru_total) FILTER (WHERE n_ru_total > 0)     AS avg_cn_rels,
           AVG(n_ru_customer) FILTER (WHERE n_ru_total > 0)  AS avg_cn_customer,
           AVG(n_ru_supplier) FILTER (WHERE n_ru_total > 0)  AS avg_cn_supplier,
           AVG(n_ru_jv) FILTER (WHERE n_ru_total > 0)        AS avg_cn_jv,
           AVG(russia_share)                                  AS avg_russia_share,
           AVG(russia_share) FILTER (WHERE n_ru_total > 0)    AS avg_russia_share_among_exposed
    FROM firm_quarter_exposure
    WHERE n_total_links > 0
    GROUP BY quarter_end ORDER BY quarter_end
""")
CSV.write(joinpath(OUT_DIR, "02_russia_exposure_timeseries.csv"), ts)
println("\nTime series -> 02_russia_exposure_timeseries.csv")

try
    DBInterface.close!(con)
catch e
    @warn "DBInterface.close! failed" exception=e
end

println("\n========== DONE ==========")
println("Key outputs:")
println("  firm_quarter_russia_exposure.parquet  (firm-quarter panel)")
println("  eu_revere_universe.parquet           (latest snapshot — DOC only)")
println("  eu_revere_universe_qend.parquet      (time-versioned — required by 05)")
println("  02_russia_edge_path_summary.csv")
println("  02_russia_exposure_timeseries.csv")
println("  02_rev_co_asof_drift_diag.csv         (look-ahead drift diagnostic)")
println("\nNext: 03_eom_etl.jl  (heavy — runs ~30-60 min for full panel)")
