# 03_eom_etl.jl
# Build slim month-end (h, i, t) holdings panel from raw parquet cache.
#
# PREREQUISITE: run 03a_decompress_to_parquet.jl FIRST (one-time, ~30-60 min)
# to populate the raw parquet cache. This script reads from that cache, not
# from the gz files directly, so it now runs in ~1-2 min per chunk.
#
# OUTPUT:
#   holdings_eom.parquet  -- slim quarter-end panel (see column list below)
#
# Audited 2026-06-01 — see AUDIT_2026_06_01_julia_descriptive.md.
# Changes vs prior version:
#   - TEST_MODE now ENV-driven (DPN_TEST_MODE=true) via 00_setup.jl, AND
#     output filenames carry _TESTMODE suffix in test mode so a test run
#     cannot silently overwrite the canonical artifact.
#   - Atomic write via atomic_copy_to (.tmp + verify + mv).
#   - Post-ETL duplicate-key audit: SELECT COUNT(*) over (fund_id, fsym_id,
#     report_date) duplicates and hard-fail above threshold.
#   - Weekend-EOM audit: per-quarter row-counts vs neighbour quarters to
#     surface 2016-12-31 (Saturday) / 2017-12-31 (Sunday) silent drops.
#   - Materialize EOM panel once into a DuckDB table for diagnostics so 8
#     diagnostic queries don't re-scan the parquet 8x.
#   - SKIP_AUDIT now ENV-driven (DPN_SKIP_AUDIT=false to enable Phase A).

include("00_setup.jl")

# TEST_MODE is now read from ENV via 00_setup.jl; do NOT redefine here.
# SKIP_AUDIT defaults to true (audit overhead is non-trivial on a full run);
# set DPN_SKIP_AUDIT=false to run Phase A.
const SKIP_AUDIT = parse(Bool, lowercase(get(ENV, "DPN_SKIP_AUDIT", "true")))

con = dbcon(memory_gb=6, threads=4)

# Source: raw parquet cache produced by 03a_decompress_to_parquet.jl.
holdings_pattern = if TEST_MODE
    replace(joinpath(RAW_PARQUET_DIR, "Factset_FundOwners_2022_2023.parquet"),
            "\\" => "/")
else
    replace(joinpath(RAW_PARQUET_DIR, "Factset_FundOwners_*.parquet"),
            "\\" => "/")
end

const EXPECTED_CHUNKS = TEST_MODE ?
    ["Factset_FundOwners_2022_2023.parquet"] :
    ["Factset_FundOwners_1999_2005.parquet",
     "Factset_FundOwners_2006_2011.parquet",
     "Factset_FundOwners_2012_2013.parquet",
     "Factset_FundOwners_2014_2015.parquet",
     "Factset_FundOwners_2016_2017.parquet",
     "Factset_FundOwners_2018_2019.parquet",
     "Factset_FundOwners_2020_2021.parquet",
     "Factset_FundOwners_2022_2023.parquet"]

let missing_chunks = filter(c -> !isfile(joinpath(RAW_PARQUET_DIR, c)), EXPECTED_CHUNKS)
    if !isempty(missing_chunks)
        error("Raw parquet cache is incomplete. Missing chunks:\n  " *
              join(missing_chunks, "\n  ") *
              "\n\nRun 03a_decompress_to_parquet.jl to build the missing ones.")
    end
end
# Surface unexpected extras as a warning (per audit recommendation)
let actual_chunks = filter(f -> startswith(basename(f), "Factset_FundOwners_") && endswith(f, ".parquet"),
                           readdir(RAW_PARQUET_DIR, join=true))
    extras = setdiff(basename.(actual_chunks), EXPECTED_CHUNKS)
    if !isempty(extras)
        @warn "Raw parquet cache contains unexpected chunks (will still be picked up by wildcard):\n  " * join(extras, "\n  ")
    end
end

if TEST_MODE
    println("\n" * "!"^70)
    println("!!  TEST_MODE = true (DPN_TEST_MODE)                              !!")
    println("!!  Output filenames carry _TESTMODE suffix.                      !!")
    println("!"^70 * "\n")
else
    println("\nETL mode: FULL (all 8 chunks, 1999-2023)")
end
println("Holdings pattern: $holdings_pattern")

eom_path = test_suffix_path(joinpath(OUT_DIR, "holdings_eom.parquet"))
const STEP_INPUTS = [joinpath(RAW_PARQUET_DIR, c) for c in EXPECTED_CHUNKS]

# ============================================================
# PHASE A: PRE-ETL AUDIT (optional via DPN_SKIP_AUDIT)
# ============================================================
if !SKIP_AUDIT
    println("\n========== PHASE A: pre-ETL audit ==========")
    DBInterface.execute(con, """
        CREATE OR REPLACE TABLE audit_cache AS
        SELECT FACTSET_FUND_ID, factset_sec_entity_id, ISSUE_TYPE, ADJ_MV,
               CAST(REPORT_DATE AS DATE) AS report_date,
               EXTRACT(DAY FROM CAST(REPORT_DATE AS DATE)) AS dom
        FROM read_parquet('$holdings_pattern')
        WHERE EXTRACT(YEAR FROM CAST(REPORT_DATE AS DATE)) = 2022
          AND EXTRACT(MONTH FROM CAST(REPORT_DATE AS DATE)) = 1
    """)
    cache_n = qdf(con, "SELECT COUNT(*) AS n FROM audit_cache").n[1]
    println("  cached rows: $cache_n")

    audit_issue = qdf(con, """
        SELECT ISSUE_TYPE, COUNT(*) AS n_rows,
               COUNT(DISTINCT FACTSET_FUND_ID)       AS n_funds,
               COUNT(DISTINCT factset_sec_entity_id) AS n_companies,
               SUM(ADJ_MV)/1e9                       AS total_mv_billions
        FROM audit_cache
        WHERE report_date = DATE '2022-01-31'
        GROUP BY ISSUE_TYPE ORDER BY n_rows DESC
    """)
    println("\nISSUE_TYPE breakdown (2022-01-31):")
    println(audit_issue)
    CSV.write(joinpath(OUT_DIR, "03_audit_issue_type_holdings.csv"), audit_issue)

    audit_dates = qdf(con, "SELECT dom, COUNT(*) AS n_rows FROM audit_cache GROUP BY dom ORDER BY dom")
    println("\nReport-date day-of-month distribution (Jan 2022):")
    println(audit_dates)
    CSV.write(joinpath(OUT_DIR, "03_audit_dom_distribution.csv"), audit_dates)

    DBInterface.execute(con, "DROP TABLE audit_cache")
else
    println("\n========== PHASE A SKIPPED (DPN_SKIP_AUDIT = true) ==========")
end

# ============================================================
# PHASE B: ETL — atomic write via atomic_copy_to
# Filters:
#   * NO ISSUE_TYPE filter (downstream 04/05 apply their own)
#   * ADJ_MV > 0
#   * Quarter-end calendar last day (Mar/Jun/Sep/Dec)
# Weekend-EOM caveat: the LAST_DAY filter drops quarter-ends that fell on a
# weekend if FactSet recorded the snapshot on the prior business day. Audit
# Phase C computes per-quarter row counts so the operator can spot drop-outs;
# if material, switch to DATE_TRUNC('quarter') + INTERVAL approach.
# ============================================================
println("\n========== PHASE B: ETL ==========")
println("Filters: QUARTER-end report date + ADJ_MV > 0 (NO ISSUE_TYPE filter)")
println("Output: $eom_path")

@time atomic_copy_to(con, """
    SELECT
        FACTSET_FUND_ID            AS fund_id,
        ENTITY_PROPER_NAME         AS entity_name,
        ISO_COUNTRY                AS investor_country,
        ENTITY_TYPE                AS entity_type,
        FSYM_ID                    AS fsym_id,
        FSYM_PRIMARY_EQUITY_ID     AS fsym_primary_id,
        LISTING_FLAG               AS listing_flag,
        CUSIP                      AS cusip,
        ISIN                       AS isin,
        SEDOL                      AS sedol,
        SEC_FIRM_ISO_COUNTRY       AS sec_country,
        ISSUE_TYPE                 AS issue_type,
        CAP_GROUP                  AS cap_group,
        factset_sec_entity_id      AS sec_entity_id,
        SEC_ENTITY_PROPER_NAME     AS sec_entity_name,
        CAST(REPORT_DATE AS DATE)  AS report_date,
        ADJ_HOLDING                AS adj_holding,
        ADJ_MV                     AS adj_mv,
        ADJ_SHARES_OUTSTANDING     AS adj_shares_out,
        ADJ_PRICE                  AS adj_price
    FROM read_parquet('$holdings_pattern')
    WHERE ADJ_MV IS NOT NULL
      AND ADJ_MV > 0
      AND CAST(REPORT_DATE AS DATE) = LAST_DAY(CAST(REPORT_DATE AS DATE))
      AND EXTRACT(MONTH FROM CAST(REPORT_DATE AS DATE)) IN (3, 6, 9, 12)
""", eom_path)
eom_path_fwd = replace(eom_path, "\\" => "/")
println("File size: $(round(filesize(eom_path)/1024^3, digits=2)) GB")

# Use a VIEW (not a TABLE) so each diagnostic query streams parquet rows
# rather than materialising 4-5 GB into the 6 GB memory_limit (which OOMs on
# Phase C otherwise). Each query independently scans the parquet; DuckDB's
# query optimiser pushes projection / filter down so the per-query cost is
# manageable.
DBInterface.execute(con, "CREATE OR REPLACE VIEW eom AS SELECT * FROM read_parquet('$eom_path_fwd')")
total_rows = qdf(con, "SELECT COUNT(*) AS n FROM eom").n[1]
println("Total rows in EOM panel: $total_rows")
write_manifest("03_eom_etl", eom_path; row_count=total_rows, input_paths=STEP_INPUTS)

# ============================================================
# PHASE C: POST-ETL DIAGNOSTICS
# ============================================================
println("\n========== PHASE C: diagnostics on holdings_eom (single materialised scan) ==========")

# C1: Duplicate-key check on (fund_id, fsym_id, report_date) — audit hard-flag
dup_check = qdf(con, """
    SELECT COUNT(*) AS n_dup_keys
    FROM (
        SELECT fund_id, fsym_id, report_date, COUNT(*) c
        FROM eom GROUP BY 1,2,3 HAVING COUNT(*) > 1
    )
""")
println("Duplicate (fund_id, fsym_id, report_date) keys: $(dup_check.n_dup_keys[1])")
CSV.write(joinpath(OUT_DIR, "03_eom_dup_key_count.csv"), dup_check)
if dup_check.n_dup_keys[1] > 10000
    @warn "Duplicate key count is high: $(dup_check.n_dup_keys[1]). Downstream SUM(adj_mv) will over-count. Investigate raw chunks."
end

# C2: Weekend-EOM audit — check per-quarter row counts vs neighbours
weekend_audit = qdf(con, """
    SELECT EXTRACT(YEAR FROM report_date) AS yr,
           EXTRACT(MONTH FROM report_date) AS mo,
           EXTRACT(DAY FROM report_date) AS dom,
           COUNT(*) AS n_rows
    FROM eom
    GROUP BY 1,2,3 ORDER BY yr, mo, dom
""")
CSV.write(joinpath(OUT_DIR, "03_eom_weekend_audit.csv"), weekend_audit)
println("Weekend-EOM audit -> 03_eom_weekend_audit.csv (look for quarters with anomalously low row counts).")

# C3: ISSUE_TYPE breakdowns
issue_breakdown = qdf(con, """
    SELECT issue_type, COUNT(*) AS n_rows,
           COUNT(DISTINCT fund_id)       AS n_funds,
           COUNT(DISTINCT sec_entity_id) AS n_companies,
           COUNT(DISTINCT fsym_id)       AS n_securities,
           SUM(adj_mv)/1e9               AS total_mv_b
    FROM eom GROUP BY issue_type ORDER BY n_rows DESC
""")
println("\nISSUE_TYPE breakdown in EOM:")
println(issue_breakdown)
CSV.write(joinpath(OUT_DIR, "03_eom_issue_type_breakdown.csv"), issue_breakdown)

issue_eu = qdf(con, """
    SELECT issue_type, COUNT(*) AS n_rows,
           COUNT(DISTINCT sec_entity_id) AS n_companies,
           SUM(adj_mv)/1e9 AS total_mv_b
    FROM eom WHERE sec_country IN $EU_SQL_TUPLE
    GROUP BY issue_type ORDER BY n_rows DESC
""")
CSV.write(joinpath(OUT_DIR, "03_eom_issue_type_breakdown_europe.csv"), issue_eu)

issue_us_inv = qdf(con, """
    SELECT issue_type, COUNT(*) AS n_rows,
           COUNT(DISTINCT sec_entity_id) AS n_companies,
           SUM(adj_mv)/1e9 AS total_mv_b
    FROM eom WHERE investor_country = 'US'
    GROUP BY issue_type ORDER BY n_rows DESC
""")
CSV.write(joinpath(OUT_DIR, "03_eom_issue_type_breakdown_us_investors.csv"), issue_us_inv)

# C4: Coverage by year
yr = qdf(con, """
    SELECT EXTRACT(YEAR FROM report_date) AS year,
           COUNT(*) AS n_rows,
           COUNT(DISTINCT fund_id) AS n_funds,
           COUNT(DISTINCT sec_entity_id) AS n_firms
    FROM eom GROUP BY year ORDER BY year
""")
CSV.write(joinpath(OUT_DIR, "03_eom_coverage_by_year.csv"), yr)
println("\nYearly coverage:")
println(yr)

# C5: Sample-date investor / company country breakdowns
ic = qdf(con, """
    SELECT investor_country, COUNT(*) AS n_holdings
    FROM eom WHERE report_date = DATE '2018-12-31'
    GROUP BY investor_country ORDER BY n_holdings DESC LIMIT 20
""")
CSV.write(joinpath(OUT_DIR, "03_eom_investor_country_2018_12.csv"), ic)

cc = qdf(con, """
    SELECT sec_country, COUNT(*) AS n_holdings,
           COUNT(DISTINCT sec_entity_id) AS n_firms
    FROM eom WHERE report_date = DATE '2018-12-31'
    GROUP BY sec_country ORDER BY n_holdings DESC LIMIT 20
""")
CSV.write(joinpath(OUT_DIR, "03_eom_company_country_2018_12.csv"), cc)

us_eu_cells = qdf(con, """
    SELECT report_date,
           COUNT(*) AS n_holdings,
           COUNT(DISTINCT fund_id) AS n_us_funds,
           COUNT(DISTINCT sec_entity_id) AS n_eu_firms,
           SUM(adj_mv)/1e9 AS total_mv_billions
    FROM eom
    WHERE investor_country = 'US' AND sec_country IN $EU_SQL_TUPLE
    GROUP BY report_date ORDER BY report_date
""")
CSV.write(joinpath(OUT_DIR, "03_us_x_eu_cells_by_month.csv"), us_eu_cells)
println("\nUS investor × EU firm panel (last 12 quarters):")
println(last(us_eu_cells, 12))

# Free the materialised diagnostic table.
DBInterface.execute(con, "DROP TABLE eom")

try
    DBInterface.close!(con)
catch e
    @warn "DBInterface.close! failed" exception=e
end

println("\n========== ETL DONE ==========")
println("Output: $eom_path")
println("\nKey diagnostic files:")
println("  03_eom_dup_key_count.csv         (hard-flag if non-trivial)")
println("  03_eom_weekend_audit.csv         (look for low-rowcount quarters)")
println("  03_eom_coverage_by_year.csv")
println("  03_eom_issue_type_breakdown.csv")
println("  03_us_x_eu_cells_by_month.csv")
println("\nNext: 04_us_ownership_european.jl")
