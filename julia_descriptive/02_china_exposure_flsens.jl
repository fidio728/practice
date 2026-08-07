# 02_china_exposure.jl
# Section 5 — build China exposure measure from Revere supply chain data.
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
#     EU-EU edges from the target firm's denominator. Numerator (eu_china_edge)
#     had no analogous filter, so china_share was inflated for firms with
#     predominantly inbound EU links.
#     Fix: remove the `<>` filter. Each EU firm counts edges where IT is
#     either source or target; EU-EU edges contribute once to each endpoint's
#     denominator — symmetric with how every other edge is counted.
#
# B7 (rel_type contamination, 2026-07-22): china_share previously counted
#     ALL relationship types (COMPETITOR = 14.85% of CN edges, PARTNER-* =
#     21%) in both numerator and denominator, while the doc §4.2 defined the
#     estimand as supply-chain dependence and Figure 1 used CUSTOMER+SUPPLIER
#     only. Fix (measure B): china_share = CN CUSTOMER+SUPPLIER links / total
#     CUSTOMER+SUPPLIER links. Drops COMPETITOR and PARTNER-* from both. The
#     all-type share is retained as china_share_alltypes for diagnostics; the
#     per-type breakdown columns (n_cn_customer/supplier/jv/manuf/partner_any/
#     total) and n_total_links are unchanged. Impact (numerator-only proxy on
#     the old parquet): ~18.5% of positive firm-quarters change HIGH/LOW.
#     Downstream (05 -> 06 -> regressions) MUST be rebuilt.
#
# EM-CHANGE-2 (zero/missing recode, Emanuele email 2026-08-05):
#     Previously a firm-quarter was emitted ONLY if the firm had at least one
#     ACTIVE Revere relationship of ANY type at q, and china_share was
#     NULLIF(n_supplychain_links, 0) — so BOTH of the following came out as
#     "missing" and dropped from every regression:
#       (i)  a firm whose only active links at q are COMPETITOR / PARTNER-*
#            (present in the panel, n_supplychain_links = 0 -> NULL);
#       (ii) a firm with no active link of any type at q (no row at all).
#     Emanuele's rule: both are GENUINE ZEROS as long as the firm is in the
#     Revere universe at q. They belong in the low-exposure arm, not in the
#     missing bin. Only a firm that is ABSENT from the Revere universe stays
#     NULL. Presence is POINT-IN-TIME (see (4b) below) — a firm is codable as
#     zero only from the quarter its Revere record begins, never before.
#     Implementation: firm_quarter_exposure is now built on the PIT-present EU
#     Revere universe (LEFT JOIN the link aggregates onto it) instead of on the
#     link aggregates. Numerator/denominator definitions are UNCHANGED (B7:
#     CUSTOMER + SUPPLIER only). New provenance columns:
#       revere_pit_present  always 1 (every emitted row is PIT-present)
#       zero_recode_flag    0 = ratio computed from a positive denominator
#                           1 = recoded zero, competitor/partner-only firm-qtr
#                           2 = recoded zero, no active link of any type
#       revere_coverage_start  first quarter-end the firm is PIT-present
#     n_supplychain_links (0 vs >0) therefore stays fully distinguishable, and
#     zero_recode_flag separates the two kinds of zero.
#     Downstream (05, 06) apply the SAME recode at their own china_share sites
#     — see the B7 lesson: china_share is computed at THREE places.
#
# Medium fixes also applied here:
#   - Removed firm_month_china_exposure.parquet alias (silent type-pun;
#     month_end column held quarter-end values). Downstream now reads
#     firm_quarter_china_exposure.parquet directly.
#   - Added deterministic tiebreaker (, company_id ASC) to every ROW_NUMBER.
#   - Sentinel '4000-01-01' replaced with NULL at load time.
#   - dedup-edge diagnostic on rev_rel.
#   - Atomic writes via atomic_copy_to from 00_setup.

include("00_setup.jl")

# ===========================================================================
# SIDE-RUN COPY (presence-rule sensitivity, 2026-08-06). DO NOT use outputs as
# canonical. Every OUT_DIR write is suffixed with DPN_SENS_SUFFIX (default
# "_flsens") so this run CANNOT clobber canonical artifacts. Default presence
# rule here is "first_link" (strictest honest lower bound); override with
# DPN_REVERE_PRESENCE_RULE + DPN_SENS_SUFFIX for other variants.
# ===========================================================================
const SENS_SUFFIX = get(ENV, "DPN_SENS_SUFFIX", "_flsens")
function sens_path(p::AbstractString)
    base, ext = splitext(p)
    return base * SENS_SUFFIX * ext
end
println("SIDE-RUN: output suffix = '$SENS_SUFFIX' (zero canonical writes)")

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
CSV.write(sens_path(joinpath(OUT_DIR, "02_revere_co_variation_diag.csv")), diag)

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
CSV.write(sens_path(joinpath(OUT_DIR, "02_revere_company_country_top25.csv")), co_country)

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
                   -- NULLS FIRST (2026-07-21 fix): NULL end_d = open segment =
                   -- +inf, must beat a same-start_d zero-length closed row
                   -- (s,s) — the WRDS same-day-correction pattern. NULLS LAST
                   -- kept the dead row, opened an artificial coverage gap and
                   -- silently dropped live firms from all later quarters
                   -- (independent audit, 11 firms, tail-quarter undercount
                   -- <=0.36%). Static dedup above (line ~130) already used
                   -- NULLS FIRST. Downstream parquets NOT yet rebuilt under
                   -- this fix — re-run 02→05→06 before next regression update.
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
CSV.write(sens_path(joinpath(OUT_DIR, "02_rev_co_asof_drift_diag.csv")), asof_drift)

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
# (2) China-edge view: i has a CN counterparty (either side). Region tags
# are AS-OF the edge's rel_start, so an EU firm that became EU in 2020 does
# NOT have its 2010 edges retroactively reclassified.
# ============================================================
println("\nBuilding China-edge view (Path 1 ∪ Path 2)...")

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE eu_china_edge AS
    -- Path 1: EU company i is source, CN counterparty is target
    SELECT source_company_id AS eu_company_id,
           target_company_id AS cn_company_id,
           rel_type,
           rel_start, rel_end,
           src_cusip AS eu_cusip,
           src_isin AS eu_isin,
           src_sedol AS eu_sedol,
           'EU_SRC' AS path
    FROM rev_rel
    WHERE src_region IN $EU_SQL_TUPLE AND tgt_region = 'CN'

    UNION ALL

    -- Path 2: CN company is source, EU company i is target
    SELECT target_company_id AS eu_company_id,
           source_company_id AS cn_company_id,
           rel_type,
           rel_start, rel_end,
           tgt_cusip AS eu_cusip,
           tgt_isin AS eu_isin,
           tgt_sedol AS eu_sedol,
           'CN_SRC' AS path
    FROM rev_rel
    WHERE src_region = 'CN' AND tgt_region IN $EU_SQL_TUPLE
""")

stats = qdf(con, """
    SELECT
        path,
        COUNT(*) AS n_relations,
        COUNT(DISTINCT eu_company_id) AS n_eu,
        COUNT(DISTINCT cn_company_id) AS n_cn
    FROM eu_china_edge GROUP BY path
""")
println("\nPath breakdown:")
println(stats)
CSV.write(sens_path(joinpath(OUT_DIR, "02_china_edge_path_summary.csv")), stats)

# Union counts
union_stats = qdf(con, """
    SELECT
        COUNT(*) AS total_relations,
        COUNT(DISTINCT eu_company_id) AS unique_eu_with_cn,
        COUNT(DISTINCT cn_company_id) AS unique_cn_with_eu
    FROM eu_china_edge
""")
println("\nUnion totals:")
println(union_stats)

# ============================================================
# (3) EU "any-edge" view — SYMMETRIC fix (C4).
# Every relation where AT LEAST ONE side is an EU firm contributes once to
# each EU endpoint's denominator. EU-EU edges contribute once to BOTH
# endpoints' denominators (was previously dropped from the target endpoint's
# denominator via `src_region <> tgt_region`, inflating china_share for
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
    -- counted once on each endpoint, symmetric with EU-CN edges.
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
universe_static_path = test_suffix_path(sens_path(joinpath(OUT_DIR, "eu_revere_universe.parquet")))
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
universe_qend_path = test_suffix_path(sens_path(joinpath(OUT_DIR, "eu_revere_universe_qend.parquet")))
atomic_copy_to(con, "SELECT * FROM eu_revere_universe_qend", universe_qend_path)
println("  -> time-versioned universe saved to $(basename(universe_qend_path))")
write_manifest("02_eu_revere_universe_qend", universe_qend_path;
               row_count=eu_univ_qend_stats.n_company_quarters[1],
               input_paths=STEP_INPUTS)

# ============================================================
# (4b) EM-CHANGE-2 — POINT-IN-TIME REVERE PRESENCE ("coverage start").
#
# WHAT THE SOURCE FILES OFFER AS A COVERAGE-START SIGNAL
# ------------------------------------------------------
# revere_company_wrds.csv is a slowly-changing-dimension (SCD-2) history:
# one row per (company_id, attribute-version) carrying
#     company_id, name, home_region, country, cusip, isin, sedol, covered,
#     start_, end_        (end_ = '4000-01-01' sentinel for an open segment)
# There is NO separate "Revere added this company on date X" column. The two
# candidate signals the file itself carries are
#     S1  the record-validity interval [start_, end_]  -> earliest record date
#         per company_id = MIN(start_). This is the file's own statement of
#         when it starts knowing about the firm, and — unlike a scalar add
#         date — it also encodes coverage EXITS and RE-ENTRIES (9.0% of
#         companies have a gap in their presence, see the diagnostic below).
#     S2  the `covered` Y/N flag, which is itself versioned by [start_, end_].
#         Only ever used in this pipeline as a static MAX() (`ever_covered`),
#         never as a filter, and its within-company time profile has never
#         been characterised.
# data_giorgio.csv (relationships) offers only
#     S3  the firm's earliest relationship start date (MIN over rel_start
#         where the firm is source or target, any rel_type).
#
# RULE CHOSEN — S1, the record-validity interval, EU-restricted.
#   present(i, q)  <=>  a revere_company_wrds row for i is valid at q
#                       (start_ <= q <= end_, sentinel-normalised)
#                       AND that row's home_region at q is in the EU list.
# That predicate is EXACTLY `eu_revere_universe_qend`, already built above by
# the C5 point-in-time fix, so the zero-recode inherits a construct this
# pipeline has already audited rather than introducing a second, parallel
# notion of "in Revere".
#
# WHY S1 AND NOT S2 / S3
#   * S3 is unusable as the PRIMARY rule: it is defined only for firms that
#     ever have a relationship record, so a firm that is genuinely zero-link
#     for its whole life would never become codable — which is precisely the
#     firm Emanuele's recode is meant to put into the low-exposure arm. It is
#     retained as an optional STRICTER variant (rule "first_link") and as a
#     diagnostic, because it is the honest test of the worry that start_ might
#     be a backfilled incorporation-style date rather than a Revere add date.
#     The diagnostic below reports the entry-to-first-link gap distribution.
#   * S2 is not adopted blind. `covered` is read and profiled below; if it
#     turns out to be informative the rule "record_interval_covered" switches
#     it on. It is NOT the default because a covered='N'-dominated file would
#     silently empty the universe, and that has never been checked.
#   * S1 additionally handles coverage gaps and exits, which a scalar
#     coverage-start date cannot.
#
# Rule is selectable via DPN_REVERE_PRESENCE_RULE for internal robustness:
#   "record_interval"          (DEFAULT) S1
#   "record_interval_covered"  S1 AND the as-of row has covered = 'Y'
#   "first_link"               S1 AND q >= the firm's first quarter with any
#                              active Revere relationship (S3 floor)
# ============================================================
const REVERE_PRESENCE_RULE = get(ENV, "DPN_REVERE_PRESENCE_RULE", "first_link")
const VALID_PRESENCE_RULES = ("record_interval", "record_interval_covered", "first_link")
REVERE_PRESENCE_RULE in VALID_PRESENCE_RULES ||
    error("DPN_REVERE_PRESENCE_RULE='$REVERE_PRESENCE_RULE' is not one of $VALID_PRESENCE_RULES")

"""
    build_revere_presence!(con; rule=REVERE_PRESENCE_RULE)

Materialise `revere_present_q(eu_company_id, qend, eu_cusip, eu_isin,
eu_sedol, revere_coverage_start)` — the set of (firm, quarter) cells at which
the firm counts as PRESENT in the Revere universe and is therefore codable as
a genuine zero when it has no supply-chain link.

This is the ONE place the coverage-start rule lives. 05 and 06 do not
re-derive it; they inherit it through row existence in the emitted parquet.
"""
function build_revere_presence!(con; rule::AbstractString=REVERE_PRESENCE_RULE)
    # S1 base: the EU-restricted record-validity interval (= eu_revere_universe_qend).
    DBInterface.execute(con, """
        CREATE OR REPLACE TABLE revere_present_base AS
        SELECT eu_company_id, qend, eu_cusip, eu_isin, eu_sedol
        FROM eu_revere_universe_qend
    """)

    if rule == "record_interval_covered"
        # S2 variant: additionally require the AS-OF row to be flagged covered.
        # rev_co_asof_q does not carry `covered`, so re-derive it as-of q from
        # rev_co_raw with the same latest-start_d-wins tiebreaker.
        DBInterface.execute(con, """
            CREATE OR REPLACE TABLE covered_asof_q AS
            WITH hist AS (
                SELECT company_id, start_d, end_d, covered
                FROM (
                    SELECT company_id, start_d, end_d, covered,
                           ROW_NUMBER() OVER (PARTITION BY company_id, start_d
                                              ORDER BY end_d DESC NULLS FIRST, company_id ASC) rn
                    FROM rev_co_raw
                ) WHERE rn = 1
            ),
            probe AS (
                SELECT DISTINCT eu_company_id AS company_id, qend FROM revere_present_base
            )
            SELECT p.company_id, p.qend, h.covered
            FROM probe p
            ASOF LEFT JOIN hist h
              ON p.company_id = h.company_id AND p.qend >= h.start_d
            WHERE h.company_id IS NOT NULL
              AND (h.end_d IS NULL OR h.end_d >= p.qend)
        """)
        DBInterface.execute(con, """
            CREATE OR REPLACE TABLE revere_present_filtered AS
            SELECT b.*
            FROM revere_present_base b
            JOIN covered_asof_q c
              ON c.company_id = b.eu_company_id AND c.qend = b.qend
            WHERE c.covered = 'Y'
        """)
    elseif rule == "first_link"
        # S3 floor: no zero before the firm's first observed relationship.
        DBInterface.execute(con, """
            CREATE OR REPLACE TABLE first_rel_q AS
            WITH firm_rel AS (
                SELECT source_company_id AS company_id, rel_start FROM rev_rel_raw
                UNION ALL
                SELECT target_company_id AS company_id, rel_start FROM rev_rel_raw
            )
            SELECT company_id, MIN(rel_start) AS first_rel_start
            FROM firm_rel GROUP BY company_id
        """)
        DBInterface.execute(con, """
            CREATE OR REPLACE TABLE revere_present_filtered AS
            SELECT b.*
            FROM revere_present_base b
            JOIN first_rel_q f ON f.company_id = b.eu_company_id
            WHERE b.qend >= f.first_rel_start
        """)
    else  # "record_interval" — the default
        DBInterface.execute(con, """
            CREATE OR REPLACE TABLE revere_present_filtered AS
            SELECT * FROM revere_present_base
        """)
    end

    # Stamp the firm's coverage start (first PIT-present quarter UNDER THE
    # ACTIVE RULE) onto every row, as provenance travelling with the panel.
    DBInterface.execute(con, """
        CREATE OR REPLACE TABLE revere_present_q AS
        SELECT f.eu_company_id, f.qend, f.eu_cusip, f.eu_isin, f.eu_sedol,
               s.revere_coverage_start
        FROM revere_present_filtered f
        JOIN (SELECT eu_company_id, MIN(qend) AS revere_coverage_start
              FROM revere_present_filtered GROUP BY eu_company_id) s
          ON s.eu_company_id = f.eu_company_id
    """)

    # One row per (firm, quarter) — the zero-recode grid must be a clean key.
    dupq = qdf(con, """
        SELECT COUNT(*) AS n FROM (
            SELECT eu_company_id, qend FROM revere_present_q
            GROUP BY 1,2 HAVING COUNT(*) > 1)
    """).n[1]
    @assert dupq == 0 "revere_present_q is not unique on (eu_company_id, qend): $dupq dup keys"
    return nothing
end

println("\n(4b) Building point-in-time Revere presence — rule = '$REVERE_PRESENCE_RULE' ...")
build_revere_presence!(con)

pres_stats = qdf(con, """
    SELECT COUNT(*) AS n_firm_quarters,
           COUNT(DISTINCT eu_company_id) AS n_firms,
           MIN(qend) AS first_qend, MAX(qend) AS last_qend
    FROM revere_present_q
""")
println("  PIT-present EU firm-quarters:")
println(pres_stats)

# --- coverage-start diagnostics (these are what justify the rule choice) ---
# (a) entry-year histogram. If start_ were a backfilled non-Revere date the
#     mass would pile up at the panel floor 2003-Q1; staggered entry is
#     evidence that start_ carries real add-date information.
entry_hist = qdf(con, """
    SELECT YEAR(revere_coverage_start) AS entry_year,
           COUNT(DISTINCT eu_company_id) AS n_firms
    FROM revere_present_q GROUP BY 1 ORDER BY 1
""")
CSV.write(sens_path(joinpath(OUT_DIR, "02_revere_coverage_start_hist.csv")), entry_hist)
println("  coverage-start (entry-year) histogram -> 02_revere_coverage_start_hist.csv")
println(first(entry_hist, 30))

# (b) presence contiguity — how often the record interval implies a GAP or an
#     exit, which a scalar coverage-start date could not represent.
contig = qdf(con, """
    WITH f AS (SELECT eu_company_id, MIN(qend) f_q, MAX(qend) l_q, COUNT(*) n_q
               FROM revere_present_q GROUP BY 1)
    SELECT COUNT(*) AS n_firms,
           COUNT(*) FILTER (WHERE n_q = DATEDIFF('quarter', f_q, l_q) + 1) AS n_contiguous,
           COUNT(*) FILTER (WHERE n_q < DATEDIFF('quarter', f_q, l_q) + 1) AS n_with_gaps
    FROM f
""")
println("  presence contiguity (gaps = coverage exits/re-entries):")
println(contig)

# (c) S1-vs-S3 divergence: how long after entering the universe does a firm
#     first show ANY relationship? A long right tail would mean the default
#     rule codes long leading zero runs that might really be pre-coverage.
s1_s3 = qdf(con, """
    WITH u AS (SELECT eu_company_id, MIN(qend) AS first_univ_q FROM revere_present_q GROUP BY 1),
         r AS (
            WITH firm_rel AS (
                SELECT source_company_id AS company_id, rel_start FROM rev_rel_raw
                UNION ALL
                SELECT target_company_id AS company_id, rel_start FROM rev_rel_raw)
            SELECT company_id, MIN(rel_start) AS first_rel_start FROM firm_rel GROUP BY 1),
         j AS (SELECT u.eu_company_id, u.first_univ_q, r.first_rel_start,
                      DATEDIFF('quarter', u.first_univ_q, r.first_rel_start) AS q_gap
               FROM u LEFT JOIN r ON r.company_id = u.eu_company_id)
    SELECT COUNT(*) AS n_firms,
           COUNT(*) FILTER (WHERE first_rel_start IS NULL) AS n_never_any_link,
           COUNT(*) FILTER (WHERE q_gap <= 0)              AS n_link_at_or_before_entry,
           COUNT(*) FILTER (WHERE q_gap > 0)               AS n_link_after_entry,
           MEDIAN(q_gap)                  AS med_q_gap,
           QUANTILE_CONT(q_gap, 0.90)     AS p90_q_gap,
           QUANTILE_CONT(q_gap, 0.99)     AS p99_q_gap
    FROM j
""")
println("  S1(record interval) vs S3(first relationship) divergence:")
println(s1_s3)
CSV.write(sens_path(joinpath(OUT_DIR, "02_revere_coverage_start_vs_firstlink.csv")), s1_s3)

# (d) profile of the `covered` flag — reported, never silently applied.
cov_prof = qdf(con, """
    SELECT covered, COUNT(*) AS n_rows, COUNT(DISTINCT company_id) AS n_firms
    FROM rev_co_raw GROUP BY 1 ORDER BY n_rows DESC
""")
println("  `covered` flag profile on rev_co_raw (S2 candidate, NOT applied unless rule='record_interval_covered'):")
println(cov_prof)
CSV.write(sens_path(joinpath(OUT_DIR, "02_revere_covered_flag_profile.csv")), cov_prof)

# ============================================================
# (5) Firm-QUARTER China exposure panel — uses time-versioned EU universe.
# A firm i appears in the panel for quarter q ONLY if it was an EU firm at q
# (home_region AS-OF q in EU). Pre-2003 quarters are not produced; downstream
# must preserve NULL for those.
#
# EM-CHANGE-2: the panel spine is now revere_present_q (every PIT-present EU
# firm-quarter), NOT firm_quarter_total (only firm-quarters with >=1 active
# link). Firms with no active link at all now appear, with zeros.
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
        SUM(CASE WHEN e.rel_type = 'CUSTOMER' THEN 1 ELSE 0 END) AS n_cn_customer,
        SUM(CASE WHEN e.rel_type = 'SUPPLIER' THEN 1 ELSE 0 END) AS n_cn_supplier,
        -- (DIRECTION FIX, 2026-08-02) n_cn_customer/n_cn_supplier above are raw
        -- rel_type label counts and MIX the two paths: the FactSet convention
        -- (methodology guide p.3) is source-perspective — CUSTOMER: source
        -- SELLS to target; SUPPLIER: source BUYS from target. From the EU
        -- firm's perspective the economic direction needs rel_type x path:
        --   sell-to-China = EU_SRC x CUSTOMER  +  CN_SRC x SUPPLIER
        --   buy-from-China = EU_SRC x SUPPLIER +  CN_SRC x CUSTOMER
        -- Row-level identity: n_cn_sell + n_cn_buy = n_cn_customer + n_cn_supplier
        -- (asserted below). These are LINK-COUNT proxies (revenue-exposure /
        -- input-dependence), NOT revenue or cost shares.
        SUM(CASE WHEN (e.rel_type = 'CUSTOMER' AND e.path = 'EU_SRC')
                   OR (e.rel_type = 'SUPPLIER' AND e.path = 'CN_SRC')
                 THEN 1 ELSE 0 END) AS n_cn_sell,
        SUM(CASE WHEN (e.rel_type = 'SUPPLIER' AND e.path = 'EU_SRC')
                   OR (e.rel_type = 'CUSTOMER' AND e.path = 'CN_SRC')
                 THEN 1 ELSE 0 END) AS n_cn_buy,
        -- unique CN counterparties per direction — DESCRIPTIVE-ONLY columns
        -- (not propagated to 05/06/regression panels). They quantify the
        -- reciprocal double-record issue (A->B CUSTOMER and B->A SUPPLIER are
        -- one economic relationship recorded from both sides; row counts take
        -- it twice, distinct-counterparty counts once). The double-record
        -- robustness that actually reaches the regression is the INDICATOR
        -- (any-sell/any-buy) spec in run_direction_split.do.
        COUNT(DISTINCT CASE WHEN (e.rel_type = 'CUSTOMER' AND e.path = 'EU_SRC')
                              OR (e.rel_type = 'SUPPLIER' AND e.path = 'CN_SRC')
                            THEN e.cn_company_id END) AS n_cn_cp_sell,
        COUNT(DISTINCT CASE WHEN (e.rel_type = 'SUPPLIER' AND e.path = 'EU_SRC')
                              OR (e.rel_type = 'CUSTOMER' AND e.path = 'CN_SRC')
                            THEN e.cn_company_id END) AS n_cn_cp_buy,
        SUM(CASE WHEN e.rel_type = 'PARTNER-JVENTUR' THEN 1 ELSE 0 END) AS n_cn_jv,
        SUM(CASE WHEN e.rel_type = 'PARTNER-MANUFAC' THEN 1 ELSE 0 END) AS n_cn_manuf,
        SUM(CASE WHEN e.rel_type LIKE 'PARTNER%' THEN 1 ELSE 0 END) AS n_cn_partner_any,
        COUNT(*) AS n_cn_total
    FROM eu_china_edge e
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
        -- (B7 FIX, 2026-07-22) supply-chain-only denominator: CUSTOMER +
        -- SUPPLIER edges only, dropping COMPETITOR (14.85% of CN edges) and
        -- all PARTNER-* types. Matches Figure 1's descriptive universe and
        -- the doc §4.2 estimand ("supply-chain dependence"). n_total_links
        -- kept as an all-relationship diagnostic.
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

# EM-CHANGE-2. Spine = revere_present_q (every PIT-present EU firm-quarter).
# The link aggregates are LEFT JOINed onto it, so:
#   * firm PIT-present with >0 supply-chain links  -> ratio, as before, bit
#     identical to the old panel (asserted against the _preEM archive below);
#   * firm PIT-present with 0 supply-chain links   -> china_share = 0
#       flag 1 if it has other-type links (competitor/partner-only firm),
#       flag 2 if it has no active link of any type (no row in either
#              aggregate — these firm-quarters did not exist in the old panel);
#   * firm NOT PIT-present at q                    -> no row -> NULL downstream.
# Numerator and denominator definitions are UNCHANGED (B7: CUSTOMER+SUPPLIER).
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE firm_quarter_exposure AS
    SELECT
        p.eu_company_id,
        p.eu_cusip,
        p.eu_isin,
        p.eu_sedol,
        p.qend AS quarter_end,
        COALESCE(c.n_cn_customer, 0) AS n_cn_customer,
        COALESCE(c.n_cn_supplier, 0) AS n_cn_supplier,
        COALESCE(c.n_cn_sell,     0) AS n_cn_sell,
        COALESCE(c.n_cn_buy,      0) AS n_cn_buy,
        COALESCE(c.n_cn_cp_sell,  0) AS n_cn_cp_sell,
        COALESCE(c.n_cn_cp_buy,   0) AS n_cn_cp_buy,
        COALESCE(c.n_cn_jv,       0) AS n_cn_jv,
        COALESCE(c.n_cn_manuf,    0) AS n_cn_manuf,
        COALESCE(c.n_cn_partner_any, 0) AS n_cn_partner_any,
        COALESCE(c.n_cn_total,    0) AS n_cn_total,
        COALESCE(t.n_total_links,       0) AS n_total_links,
        COALESCE(t.n_supplychain_links, 0) AS n_supplychain_links,
        -- EM-CHANGE-2 provenance. Keeps the two kinds of zero distinguishable
        -- from each other AND from a computed ratio that happens to be 0.
        1 AS revere_pit_present,
        CASE WHEN COALESCE(t.n_supplychain_links, 0) > 0 THEN 0
             WHEN COALESCE(t.n_total_links,       0) > 0 THEN 1
             ELSE 2 END AS zero_recode_flag,
        p.revere_coverage_start,
        -- (B7 FIX, 2026-07-22) china_share = CN supply-chain links / total
        -- supply-chain links (CUSTOMER + SUPPLIER, both endpoints). Excludes
        -- COMPETITOR and PARTNER-*.
        -- (EM-CHANGE-2, 2026-08-06) a PIT-present firm with a ZERO
        -- supply-chain denominator is a genuine zero, not a missing value —
        -- NULLIF(...) replaced by an explicit CASE. NULL is now reachable
        -- ONLY by absence of the row (firm not in Revere at q).
        CASE WHEN COALESCE(t.n_supplychain_links, 0) > 0
             THEN CAST(COALESCE(c.n_cn_customer, 0) + COALESCE(c.n_cn_supplier, 0) AS DOUBLE)
                  / t.n_supplychain_links
             ELSE 0.0 END AS china_share,
        -- (DIRECTION FIX, 2026-08-02) directional LINK-COUNT shares over the
        -- SAME supply-chain denominator, so that
        --   china_sell_link_share + china_buy_link_share = china_share
        -- holds row-wise (additive decomposition of the headline measure).
        -- Proxies for revenue exposure (sell) / input dependence (buy) — NOT
        -- revenue or cost shares. Same EM-CHANGE-2 zero treatment.
        CASE WHEN COALESCE(t.n_supplychain_links, 0) > 0
             THEN CAST(COALESCE(c.n_cn_sell, 0) AS DOUBLE) / t.n_supplychain_links
             ELSE 0.0 END AS china_sell_link_share,
        CASE WHEN COALESCE(t.n_supplychain_links, 0) > 0
             THEN CAST(COALESCE(c.n_cn_buy, 0) AS DOUBLE) / t.n_supplychain_links
             ELSE 0.0 END AS china_buy_link_share,
        -- legacy all-relationship-type share, retained for diagnostics only.
        -- Same recode for internal consistency: a PIT-present firm with no
        -- link of ANY type has an all-type China share of 0, not missing.
        CASE WHEN COALESCE(t.n_total_links, 0) > 0
             THEN CAST(COALESCE(c.n_cn_total, 0) AS DOUBLE) / t.n_total_links
             ELSE 0.0 END AS china_share_alltypes
    FROM revere_present_q p
    LEFT JOIN firm_quarter_total t
        ON t.eu_company_id = p.eu_company_id AND t.qend = p.qend
    LEFT JOIN firm_quarter_cn c
        ON c.eu_company_id = p.eu_company_id AND c.qend = p.qend
""")

n_panel = qdf(con, "SELECT COUNT(*) AS n FROM firm_quarter_exposure").n[1]
n_firms = qdf(con, "SELECT COUNT(DISTINCT eu_company_id) AS n FROM firm_quarter_exposure").n[1]
n_exposed = qdf(con, "SELECT COUNT(*) AS n FROM firm_quarter_exposure WHERE (n_cn_customer + n_cn_supplier) > 0").n[1]
n_exposed_all = qdf(con, "SELECT COUNT(*) AS n FROM firm_quarter_exposure WHERE n_cn_total > 0").n[1]
println("  panel rows (ALL PIT-present EU firm-quarters): $n_panel ; unique EU firms: $n_firms")
println("  rows with positive CN supply-chain (CUST+SUPP) exposure: $n_exposed ; (all rel-types, diagnostic): $n_exposed_all")

# --- EM-CHANGE-2 census: exactly which cells the recode creates ------------
recode_census = qdf(con, """
    SELECT zero_recode_flag,
           CASE zero_recode_flag
                WHEN 0 THEN 'ratio_from_positive_denominator'
                WHEN 1 THEN 'recoded_zero__competitor_or_partner_only'
                ELSE        'recoded_zero__no_active_link_of_any_type' END AS label,
           COUNT(*) AS n_firm_quarters,
           COUNT(DISTINCT eu_company_id) AS n_firms
    FROM firm_quarter_exposure GROUP BY 1,2 ORDER BY 1
""")
println("\nEM-CHANGE-2 recode census (firm-quarter level, Revere-company grain):")
println(recode_census)
CSV.write(sens_path(joinpath(OUT_DIR, "02_em_zero_recode_census.csv")), recode_census)

# HARD ASSERTS -------------------------------------------------------------
dup_key = qdf(con, """
    SELECT COUNT(*) AS n FROM (
        SELECT eu_company_id, quarter_end FROM firm_quarter_exposure
        GROUP BY 1,2 HAVING COUNT(*) > 1)
""").n[1]
@assert dup_key == 0 "firm_quarter_exposure has $dup_key duplicate (eu_company_id, quarter_end) keys"

n_null_share = qdf(con, "SELECT COUNT(*) AS n FROM firm_quarter_exposure WHERE china_share IS NULL").n[1]
@assert n_null_share == 0 "EM-CHANGE-2 invariant broken: $n_null_share emitted rows still carry NULL china_share; NULL must be reachable only by row ABSENCE"

oob02 = qdf(con, "SELECT COUNT(*) AS n FROM firm_quarter_exposure WHERE china_share < 0 OR china_share > 1").n[1]
@assert oob02 == 0 "firm_quarter_exposure has $oob02 rows with china_share ∉ [0,1]"

before_cov = qdf(con, "SELECT COUNT(*) AS n FROM firm_quarter_exposure WHERE quarter_end < revere_coverage_start").n[1]
@assert before_cov == 0 "$before_cov rows precede the firm's own Revere coverage start — point-in-time rule violated"
println("  asserts: unique key ✓ ; no NULL china_share on emitted rows ✓ ; share ∈ [0,1] ✓ ; no pre-coverage rows ✓")

# Orphan-link guard. firm_quarter_total is built off eu_revere_universe_qend
# (the S1 rule). Under a STRICTER presence rule the spine is a subset, so some
# firm-quarters that DO carry links would be dropped. Under the default rule
# this must be exactly 0; under the others it is reported, never silent.
orphan = qdf(con, """
    SELECT COUNT(*) AS n
    FROM firm_quarter_total t
    LEFT JOIN revere_present_q p
      ON p.eu_company_id = t.eu_company_id AND p.qend = t.qend
    WHERE p.eu_company_id IS NULL
""").n[1]
if REVERE_PRESENCE_RULE == "record_interval"
    @assert orphan == 0 "default presence rule dropped $orphan link-bearing firm-quarters — spine mismatch with firm_quarter_total"
    println("  orphan-link guard: 0 link-bearing firm-quarters dropped by the presence rule ✓")
else
    println("  [rule='$REVERE_PRESENCE_RULE'] $orphan link-bearing firm-quarters dropped by the stricter presence rule")
end

# (DIRECTION FIX) hard identity check: sell + buy must equal customer + supplier
# on every row (the direction split is a re-partition of the same records).
id_viol = qdf(con, """
    SELECT COUNT(*) AS n FROM firm_quarter_exposure
    WHERE (n_cn_sell + n_cn_buy) != (n_cn_customer + n_cn_supplier)
""").n[1]
@assert id_viol == 0 "direction split violates additive identity on $id_viol rows"
println("  direction identity: n_cn_sell + n_cn_buy == n_cn_customer + n_cn_supplier on every row ✓")

# reciprocal-record diagnostic: the same economic EU-CN relationship recorded
# from both sides (A->B CUSTOMER and B->A SUPPLIER) classifies consistently by
# direction but is counted twice in row counts. Quantify how common that is.
recip = qdf(con, """
    WITH sell_pairs AS (
        SELECT DISTINCT eu_company_id, cn_company_id, path
        FROM eu_china_edge
        WHERE (rel_type = 'CUSTOMER' AND path = 'EU_SRC')
           OR (rel_type = 'SUPPLIER' AND path = 'CN_SRC')
    )
    SELECT
        COUNT(*) AS n_pairs,
        COUNT(*) FILTER (WHERE n_paths = 2) AS n_both_sides
    FROM (
        SELECT eu_company_id, cn_company_id, COUNT(DISTINCT path) AS n_paths
        FROM sell_pairs GROUP BY 1, 2
    )
""")
println("  reciprocal-record diagnostic (sell direction): $(recip.n_both_sides[1]) of $(recip.n_pairs[1]) EU-CN pairs recorded from both sides (double-counted in link counts; distinct-counterparty columns n_cn_cp_sell/buy are immune)")

recip_buy = qdf(con, """
    WITH buy_pairs AS (
        SELECT DISTINCT eu_company_id, cn_company_id, path
        FROM eu_china_edge
        WHERE (rel_type = 'SUPPLIER' AND path = 'EU_SRC')
           OR (rel_type = 'CUSTOMER' AND path = 'CN_SRC')
    )
    SELECT
        COUNT(*) AS n_pairs,
        COUNT(*) FILTER (WHERE n_paths = 2) AS n_both_sides
    FROM (
        SELECT eu_company_id, cn_company_id, COUNT(DISTINCT path) AS n_paths
        FROM buy_pairs GROUP BY 1, 2
    )
""")
println("  reciprocal-record diagnostic (buy direction):  $(recip_buy.n_both_sides[1]) of $(recip_buy.n_pairs[1]) EU-CN pairs recorded from both sides")

# ============================================================
# (5b) EM-CHANGE-2 P0 GATE — "exposure changes ONLY where the recode applies".
# If the pre-change panel has been archived as
# firm_quarter_china_exposure_preEM.parquet, every (firm, quarter) cell that
# existed there WITH a positive supply-chain denominator must come out
# bit-identical here. Any drift means the recode leaked into cells it must not
# touch. Skipped (with a loud note) when the archive is absent.
# ============================================================
preem_path = joinpath(OUT_DIR, test_suffix_path("firm_quarter_china_exposure_preEM.parquet"))
if isfile(preem_path)
    preem_fwd = replace(preem_path, "\\" => "/")
    gate = qdf(con, """
        WITH old AS (
            SELECT eu_company_id, quarter_end, n_supplychain_links,
                   china_share, china_sell_link_share, china_buy_link_share
            FROM read_parquet('$preem_fwd')
            WHERE n_supplychain_links > 0
        )
        SELECT COUNT(*) AS n_old_positive_cells,
               COUNT(n.eu_company_id) AS n_matched_in_new,
               COUNT(*) FILTER (WHERE n.eu_company_id IS NULL) AS n_lost,
               COUNT(*) FILTER (WHERE n.n_supplychain_links IS DISTINCT FROM old.n_supplychain_links) AS n_denom_drift,
               COUNT(*) FILTER (WHERE n.china_share IS DISTINCT FROM old.china_share) AS n_share_drift,
               COUNT(*) FILTER (WHERE n.china_sell_link_share IS DISTINCT FROM old.china_sell_link_share) AS n_sell_drift,
               COUNT(*) FILTER (WHERE n.china_buy_link_share  IS DISTINCT FROM old.china_buy_link_share)  AS n_buy_drift
        FROM old
        LEFT JOIN firm_quarter_exposure n
          ON n.eu_company_id = old.eu_company_id AND n.quarter_end = old.quarter_end
    """)
    println("\nEM-CHANGE-2 P0 gate vs $(basename(preem_path)):")
    println(gate)
    CSV.write(sens_path(joinpath(OUT_DIR, "02_em_p0_gate_vs_preEM.csv")), gate)
    if REVERE_PRESENCE_RULE == "record_interval"
        @assert gate.n_lost[1] == 0 "P0 gate: $(gate.n_lost[1]) pre-change positive-denominator cells vanished"
        @assert gate.n_denom_drift[1] == 0 "P0 gate: denominator drifted on $(gate.n_denom_drift[1]) untouched cells"
        @assert gate.n_share_drift[1] == 0 "P0 gate: china_share drifted on $(gate.n_share_drift[1]) untouched cells"
        @assert gate.n_sell_drift[1] == 0 "P0 gate: china_sell_link_share drifted on $(gate.n_sell_drift[1]) cells"
        @assert gate.n_buy_drift[1] == 0  "P0 gate: china_buy_link_share drifted on $(gate.n_buy_drift[1]) cells"
        println("  P0 gate PASSED — untouched cells bit-identical ✓")
    else
        # SIDE-RUN: under a STRICTER presence rule losing cells relative to the
        # preEM archive is by design (the spine shrinks), and the IS DISTINCT
        # FROM drift counters include those lost cells (NULL vs value). Report
        # loudly instead of asserting; the drift-on-SURVIVING-cells check is the
        # one that must still hold and is computed separately here.
        surv_drift = qdf(con, """
            WITH old AS (
                SELECT eu_company_id, quarter_end, n_supplychain_links,
                       china_share, china_sell_link_share, china_buy_link_share
                FROM read_parquet('$preem_fwd')
                WHERE n_supplychain_links > 0
            )
            SELECT COUNT(*) AS n_surviving,
                   COUNT(*) FILTER (WHERE n.n_supplychain_links IS DISTINCT FROM old.n_supplychain_links
                                       OR n.china_share IS DISTINCT FROM old.china_share
                                       OR n.china_sell_link_share IS DISTINCT FROM old.china_sell_link_share
                                       OR n.china_buy_link_share  IS DISTINCT FROM old.china_buy_link_share) AS n_value_drift
            FROM old
            JOIN firm_quarter_exposure n
              ON n.eu_company_id = old.eu_company_id AND n.quarter_end = old.quarter_end
        """)
        println("  [rule='$REVERE_PRESENCE_RULE'] P0 gate relaxed: $(gate.n_lost[1]) positive-denominator cells dropped by the stricter spine (by design).")
        println("  surviving-cell value drift (MUST be 0): $(surv_drift.n_value_drift[1]) of $(surv_drift.n_surviving[1])")
        @assert surv_drift.n_value_drift[1] == 0 "side-run gate: values drifted on $(surv_drift.n_value_drift[1]) SURVIVING positive-denominator cells — rule must only remove rows, never change values"
    end
else
    # (EM-FIX-8, 2026-08-06) HARD ERROR, not a skip. This gate is the ONLY
    # automated proof that the zero recode touched exactly the cells it was
    # supposed to touch and left every positive-denominator cell bit-identical.
    # A silent skip means a missed archive step downgrades the run from "verified"
    # to "unverified" without anyone noticing in a 20-minute log. Opt out only
    # deliberately, and only when there genuinely is no pre-change vintage on this
    # machine (e.g. a first-ever build).
    if parse(Bool, lowercase(get(ENV, "DPN_EM_SKIP_P0_GATE", "false")))
        println("\n" * "!"^74)
        println("!! EM-CHANGE-2 P0 GATE SKIPPED — DPN_EM_SKIP_P0_GATE=true             !!")
        println("!! No archive at $(basename(preem_path))")
        println("!! This run is UNVERIFIED: nothing proves that positive-denominator   !!")
        println("!! cells came out unchanged. Say so wherever these numbers are used.  !!")
        println("!"^74 * "\n")
    else
        error("""
        EM-CHANGE-2 P0 GATE CANNOT RUN — refusing to emit an unverified panel.

          missing: $preem_path

        This gate proves that the zero/missing recode changed ONLY the cells it
        was meant to change: every (firm, quarter) cell that existed in the
        pre-change panel WITH n_supplychain_links > 0 must come out bit-identical
        (denominator, china_share, and both directional shares).

        Fix: archive the pre-change panel by RENAME before re-running 02, e.g.

          python archive_preP0.py --suffix preEM --apply

        which renames firm_quarter_china_exposure.parquet (and its .meta.json)
        to *_preEM.parquet. NEVER overwrite an existing _preEM archive.

        If there genuinely is no pre-change vintage on this machine, set
        DPN_EM_SKIP_P0_GATE=true and record in VINTAGE_PREEM.md that the run is
        unverified.
        """)
    end
end

# Save panel as parquet for downstream use (atomic write)
firm_quarter_path = test_suffix_path(sens_path(joinpath(OUT_DIR, "firm_quarter_china_exposure.parquet")))
atomic_copy_to(con, "SELECT * FROM firm_quarter_exposure", firm_quarter_path)
println("  -> saved to $(basename(firm_quarter_path))")
write_manifest("02_firm_quarter_china_exposure", firm_quarter_path;
               row_count=n_panel, input_paths=STEP_INPUTS)

# NOTE: previous version also emitted firm_month_china_exposure.parquet as an
# "alias" — but it contained QUARTER-END dates under a column named month_end,
# which silently misled downstream readers. The alias has been REMOVED per
# audit recommendation. 05 now reads firm_quarter_china_exposure.parquet
# directly.

# ============================================================
# (6) Descriptive: distribution of share-based exposure at one snapshot
# ============================================================
const SNAP_QUARTER = "2018-12-31"
println("\n========== Descriptive snapshot: $SNAP_QUARTER ==========")

snap = qdf(con, """
    SELECT n_cn_total,
           COUNT(*) AS n_firms
    FROM firm_quarter_exposure
    WHERE quarter_end = DATE '$SNAP_QUARTER' AND n_cn_total > 0
    GROUP BY n_cn_total
    ORDER BY n_cn_total
""")
println("Distribution of CN relation count ($SNAP_QUARTER, firms with positive CN exposure only):")
println(first(snap, 30))
CSV.write(sens_path(joinpath(OUT_DIR, "02_dist_cn_total_2018.csv")), snap)

# Summary percentiles for both the count and the new share measure.
# (EM-CHANGE-2) Reported on TWO universes, side by side and clearly labelled:
#   scope='pit_universe'   every PIT-present firm (recoded zeros INCLUDED) —
#                          this is the new estimation universe;
#   scope='legacy_positive_denominator'
#                          the cells the PRE-CHANGE vintage actually contributed
#                          to these statistics, so the before/after anchor is a
#                          like-for-like comparison.
#
# (EM-FIX-3, 2026-08-06) THE LEGACY ROW MUST BE CONDITIONED ON THE PRE-CHANGE
# CODABILITY RULE, NOT ON n_total_links > 0. Before EM-CHANGE-2, china_share was
#     (n_cn_customer + n_cn_supplier) / NULLIF(n_supplychain_links, 0)
# i.e. NULL whenever n_supplychain_links = 0, and AVG()/QUANTILE_CONT() skip
# NULLs silently. Competitor/partner-only firm-quarters (n_total_links > 0 but
# n_supplychain_links = 0) therefore contributed NOTHING to the old
# share_mean/p50/p75/p90. Under the recode they carry an explicit 0.0, so a row
# filtered on n_total_links > 0 would MIX them in and silently report a
# DIFFERENT statistic from the vintage it claims to reproduce — in the one
# artifact whose whole job is to be the before/after anchor. Filtering on
# n_supplychain_links > 0 reproduces the old cell set exactly.
# The COUNT columns are unaffected by this (a count over n_total_links > 0
# genuinely is unchanged), but they are reported on the same restricted set here
# so every column of the row describes one population; the n_total_links > 0
# head-count is still in the time series as n_firms_with_any_link.
pct = qdf(con, """
    SELECT 'pit_universe' AS scope,
        COUNT(*) AS n_firms,
        AVG(n_cn_total) AS mean_count,
        QUANTILE_CONT(n_cn_total, 0.5) AS count_p50,
        QUANTILE_CONT(n_cn_total, 0.75) AS count_p75,
        QUANTILE_CONT(n_cn_total, 0.90) AS count_p90,
        MAX(n_cn_total) AS count_max,
        AVG(china_share) AS mean_share,
        QUANTILE_CONT(china_share, 0.5) AS share_p50,
        QUANTILE_CONT(china_share, 0.75) AS share_p75,
        QUANTILE_CONT(china_share, 0.90) AS share_p90,
        MAX(china_share) AS share_max
    FROM firm_quarter_exposure
    WHERE quarter_end = DATE '$SNAP_QUARTER'
    UNION ALL
    SELECT 'legacy_positive_denominator' AS scope,
        COUNT(*), AVG(n_cn_total),
        QUANTILE_CONT(n_cn_total, 0.5), QUANTILE_CONT(n_cn_total, 0.75),
        QUANTILE_CONT(n_cn_total, 0.90), MAX(n_cn_total),
        AVG(china_share),
        QUANTILE_CONT(china_share, 0.5), QUANTILE_CONT(china_share, 0.75),
        QUANTILE_CONT(china_share, 0.90), MAX(china_share)
    FROM firm_quarter_exposure
    WHERE quarter_end = DATE '$SNAP_QUARTER' AND n_supplychain_links > 0
""")
println("\nPercentiles ($SNAP_QUARTER), PIT universe vs legacy positive-denominator cells:")
println(pct)
CSV.write(sens_path(joinpath(OUT_DIR, "02_china_exposure_percentiles_snapshot.csv")), pct)

# Time series: per-quarter mean China share and mean CN relation count.
# (EM-CHANGE-2) no longer filtered to n_total_links > 0 — the denominator is
# now the PIT-present universe. Both denominators are reported per quarter so
# the recode's effect on any time series is legible rather than silent.
ts = qdf(con, """
    SELECT quarter_end,
           COUNT(*) FILTER (WHERE n_cn_total > 0)            AS n_firms_with_cn,
           COUNT(*) FILTER (WHERE n_total_links > 0)         AS n_firms_with_any_link,
           COUNT(*)                                          AS n_firms_pit_present,
           COUNT(*) FILTER (WHERE zero_recode_flag = 1)      AS n_recoded_zero_competitor_partner_only,
           COUNT(*) FILTER (WHERE zero_recode_flag = 2)      AS n_recoded_zero_no_links,
           AVG(n_cn_total) FILTER (WHERE n_cn_total > 0)     AS avg_cn_rels,
           AVG(n_cn_customer) FILTER (WHERE n_cn_total > 0)  AS avg_cn_customer,
           AVG(n_cn_supplier) FILTER (WHERE n_cn_total > 0)  AS avg_cn_supplier,
           AVG(n_cn_jv) FILTER (WHERE n_cn_total > 0)        AS avg_cn_jv,
           AVG(china_share)                                  AS avg_china_share,
           -- (EM-FIX-3, 2026-08-06) the legacy comparator is conditioned on the
           -- PRE-CHANGE codability rule (n_supplychain_links > 0), not on
           -- n_total_links > 0. Before EM-CHANGE-2 a competitor/partner-only
           -- firm-quarter had china_share = NULL and was skipped by AVG; it now
           -- carries an explicit 0.0, so filtering on n_total_links > 0 would
           -- report a different statistic under a "legacy" label. Renamed so a
           -- reader of the CSV cannot mistake the two.
           AVG(china_share) FILTER (WHERE n_supplychain_links > 0)
               AS avg_china_share_legacy_positive_denominator,
           -- Same defect, same fix: n_cn_total > 0 alone admits China-COMPETITOR-
           -- only firms whose supply-chain denominator is 0 and which the old
           -- vintage never counted. Column name kept (plots/plot_all.jl:149 and
           -- plots/plots_python.py:166 read it by name).
           AVG(china_share) FILTER (WHERE n_cn_total > 0 AND n_supplychain_links > 0)
               AS avg_china_share_among_exposed
    FROM firm_quarter_exposure
    GROUP BY quarter_end ORDER BY quarter_end
""")
CSV.write(sens_path(joinpath(OUT_DIR, "02_china_exposure_timeseries.csv")), ts)
println("\nTime series -> 02_china_exposure_timeseries.csv")

try
    DBInterface.close!(con)
catch e
    @warn "DBInterface.close! failed" exception=e
end

println("\n========== DONE ==========")
println("Key outputs:")
println("  firm_quarter_china_exposure.parquet  (firm-quarter panel; EM-CHANGE-2:")
println("                                        now ALL PIT-present EU firm-quarters,")
println("                                        china_share NEVER NULL on an emitted row)")
println("  eu_revere_universe.parquet           (latest snapshot — DOC only)")
println("  eu_revere_universe_qend.parquet      (time-versioned — required by 05)")
println("  02_china_edge_path_summary.csv")
println("  02_china_exposure_timeseries.csv")
println("  02_china_exposure_percentiles_snapshot.csv")
println("  02_rev_co_asof_drift_diag.csv         (look-ahead drift diagnostic)")
println("  02_em_zero_recode_census.csv          (EM-CHANGE-2 flag census)")
println("  02_revere_coverage_start_hist.csv     (coverage-start entry years)")
println("  02_revere_coverage_start_vs_firstlink.csv (S1 vs S3 divergence)")
println("  02_revere_covered_flag_profile.csv    (S2 `covered` profile — not applied by default)")
println("\nEM-CHANGE-2 presence rule in force: '$REVERE_PRESENCE_RULE'")
println("  (override with DPN_REVERE_PRESENCE_RULE = record_interval |")
println("   record_interval_covered | first_link)")
println("\nNext: 03_eom_etl.jl  (heavy — runs ~30-60 min for full panel)")
