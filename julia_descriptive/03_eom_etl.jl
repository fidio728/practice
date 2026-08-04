# 03_eom_etl.jl
# Build slim quarter-end (h, i, t) holdings panel from raw parquet cache.
#
# PREREQUISITE: run 03a_decompress_to_parquet.jl FIRST (one-time, ~30-60 min)
# to populate the raw parquet cache. This script reads from that cache, not
# from the gz files directly, so it now runs in ~1-2 min per chunk.
#
# OUTPUT:
#   holdings_eom.parquet  -- slim quarter-end panel (see column list below)
#
# ============================================================================
# P0 REBUILD 2026-08-04 — AS-OF QUARTER-END SNAPSHOT SELECTION
# ============================================================================
#
# ----------------------------------------------------------------------------
# ESTIMAND / UNIVERSE  (read this before using the panel for anything)
# ----------------------------------------------------------------------------
# WHAT A CELL IS. Each (fund, security, quarter) cell of holdings_eom.parquet is
# the fund's MOST RECENT REPORTED POSITION at, or within W = ASOF_WINDOW_DAYS
# calendar days BEFORE, the calendar quarter-end. It is NOT "the position on the
# quarter-end date".
#
# WHAT THE UNIVERSE IS. The panel's fund universe for quarter Q is
#     "funds with a valid ADJ_MV > 0 report inside [qend_Q - W, qend_Q]"
# NOT
#     "funds reporting exactly ON qend_Q"   <- the pre-P0 universe.
#
# THIS IS A UNIVERSE CHANGE ON EVERY QUARTER, NOT A WEEKEND-ONLY REPAIR.
# The rule admits EARLY REPORTERS — funds that stamped their snapshot a few days
# before quarter-end — which the exact-EOM rule dropped even when the quarter-end
# fell on a business day. Measured on the full 1999-2023 sample
# (diag_p0_asof_weekday_weekend_gap.csv, W = 10):
#     weekday quarter-end coverage  81.82%  ->  83.94%   (+2.12pp on the mean;
#                                                         +0.9 to +3.7pp per quarter)
#     weekend quarter-end coverage  59.49%  ->  79.76%
# The weekday rise is EXPECTED and is NOT a gate failure. Any comparison of a
# post-P0 level against a pre-P0 level is a comparison across two universes.
#
# WHY W = 10 (numbers recorded here so the choice is reproducible from the ETL
# alone; source: diag_p0_asof_window.py -> output/diag_p0_asof_*.csv, full
# 1999-2023 sample, 100 quarters):
#     W      mean coverage   rowshare(gap>14d)   share of fund-quarters
#                                                 reporting MORE THAN ONCE
#                                                 inside the window
#     0        75.12%            0.00%                  0.000
#     3        82.30%            0.00%                  0.0024
#     7        82.59%            0.00%                  0.0025
#    10        82.68%            0.00%                  0.0026
#    14        82.79%            0.00%                  0.0026
#    31        90.74%           28.39%                  0.3464
#    92       100.00%           46.34%                  0.4086   (degenerate anchor)
#   W = 3..14 is a FLAT PLATEAU: 82.30-82.79% mean coverage, a 0.49pp spread over
#   an 11-day widening. W = 31 jumps +7.95pp, but buys it by importing genuinely
#   stale mid-quarter reporting: 28.4% of captured rows then come from a gap > 14
#   days, and 34.6% of fund-quarters report more than once inside the window (vs
#   0.26% at W = 10), i.e. the per-(fund, fsym) rule starts silently unioning two
#   different portfolio snapshots. W = 10 sits mid-plateau and is CONFIRMED.
#   Trend-free calendar artifact (local deviation vs neighbouring quarters):
#   11.11pp at W = 0 -> 1.06pp at W = 10; W = 3/14 give 1.07/1.08pp, so the choice
#   inside the plateau is not load-bearing.
#
# DISCLOSED CAVEAT — THE 1999-2005 GAP DOES NOT CLOSE. The weekday-vs-weekend
# coverage gap closes in the mid (2006-2015) and modern (2016-2023) eras but NOT
# in the LionShares ramp-up era:
#     era                W = 0 gap    W = 10 gap
#     ramp-up 1999-2005   14.31pp      11.26pp    <- NOT CLOSED
#     mid     2006-2015   17.69pp       2.92pp
#     modern  2016-2023   34.00pp       1.63pp
#   No W inside the plateau helps (W = 3: 11.18pp, W = 14: 11.29pp). Early
#   LionShares funds report on scattered mid-quarter dates, not on the prior
#   business day, so the as-of rule is not the binding constraint there. The
#   ramp-up era IS inside the estimation sample (06_cartesian_grid.jl builds the
#   grid over 1999-2023), so this is a disclosed era-level residual, not an
#   out-of-sample footnote. Report it; do not widen W to hide it.
#
# WHAT CHANGED (this is the ONLY change in this package; 05's shock code is
# untouched):
#
#   OLD rule (Phase B):
#       CAST(REPORT_DATE AS DATE) = LAST_DAY(CAST(REPORT_DATE AS DATE))
#       AND EXTRACT(MONTH FROM REPORT_DATE) IN (3, 6, 9, 12)
#   i.e. an EXACT calendar quarter-end date match.
#
#   WHY IT WAS BROKEN: when a calendar quarter-end falls on a weekend, a large
#   part of the fund universe stamps its snapshot on the prior business day and
#   is therefore dropped wholesale. Fund coverage swings with the calendar:
#   ~87% on weekday quarter-ends vs 58.4% (2022-12-31, Sat) and 38.6%
#   (2023-09-30, Sat). Worse, the loss is NOT US/NONUS symmetric — the dropped
#   Friday batch is ~26% US vs ~21% US in the kept Saturday batch — so group x
#   quarter FE cannot absorb it and beta3 can be biased, not merely noisier.
#
#   NEW rule: per (fund_id, fsym_id, quarter), take the LATEST REPORT_DATE that
#   is <= the calendar quarter-end and no more than ASOF_WINDOW_DAYS before it.
#   The surviving row is STAMPED with report_date = calendar quarter-end, so
#   every downstream quarter join, merge and FE is unchanged. The true date is
#   preserved in report_date_actual, with asof_gap_days = qend - actual.
#
#   NEW COLUMNS: report_date_actual, asof_gap_days, n_asof_candidates.
#   (n_asof_candidates = how many raw rows shared this (fund, fsym, quarter)
#   key before the ROW_NUMBER dedup; it makes the dedup magnitude auditable
#   downstream without re-scanning 187M raw rows. It is ~1 for almost every row
#   and costs essentially nothing after zstd.)
#
#   W (ASOF_WINDOW_DAYS): default 10, override with DPN_ASOF_WINDOW_DAYS.
#   See diag_p0_asof_window.py for the coverage-vs-W curve that justifies it.
#
#   IMPLEMENTATION: per-chunk loop, NOT one global window function. A global
#   PARTITION BY (fund, fsym) over ~187M rows can OOM/spill under the 6 GB
#   memory_limit. Because raw chunks split on YEAR boundaries and a quarter
#   never spans a year, each chunk can be selected independently for W <= 89;
#   the parts are then merged with one COPY. W >= 90 is REFUSED (at W >= 90 a
#   snapshot can be carried into the NEXT quarter, which needs a quarter cross
#   join, not the per-row quarter assignment used here).
#
# Audited 2026-06-01 — see AUDIT_2026_06_01_julia_descriptive.md.
# Changes vs prior version:
#   - TEST_MODE now ENV-driven (DPN_TEST_MODE=true) via 00_setup.jl, AND
#     output filenames carry _TESTMODE suffix in test mode so a test run
#     cannot silently overwrite the canonical artifact.
#   - Atomic write via atomic_copy_to (.tmp + verify + mv).
#   - Post-ETL duplicate-key audit: now a HARD ASSERT (was a warn threshold).
#   - Weekend-EOM audit: replaced by the per-quarter old-rule-vs-new-rule
#     coverage table (Gate 1) and the US-share table (Gate 2).
#   - SKIP_AUDIT now ENV-driven (DPN_SKIP_AUDIT=false to enable Phase A).

include("00_setup.jl")

# TEST_MODE is now read from ENV via 00_setup.jl; do NOT redefine here.
# SKIP_AUDIT defaults to true (audit overhead is non-trivial on a full run);
# set DPN_SKIP_AUDIT=false to run Phase A.
const SKIP_AUDIT = parse(Bool, lowercase(get(ENV, "DPN_SKIP_AUDIT", "true")))

# ============================================================
# P0: AS-OF WINDOW CONSTANT
# ============================================================
const ASOF_WINDOW_DAYS = parse(Int, get(ENV, "DPN_ASOF_WINDOW_DAYS", "10"))
if ASOF_WINDOW_DAYS < 0 || ASOF_WINDOW_DAYS > 89
    error("""
    DPN_ASOF_WINDOW_DAYS = $ASOF_WINDOW_DAYS is out of the supported range [0, 89].

    W = 0  reproduces the OLD exact-quarter-end rule (kept for A/B checks).
    W >= 90 is REFUSED on purpose: at 90+ days a snapshot from the previous
    quarter can be the as-of pick for the NEXT quarter-end, so a row no longer
    maps to exactly one quarter. That needs a quarter cross join and a
    carry-forward flag, which this per-chunk implementation does not do; it
    would silently under-fill instead of erroring. Build W=92 as a separate
    robustness path if it is ever needed.
    """)
end

# Force a rebuild of the per-chunk parts even if they already exist.
const ASOF_FORCE_REBUILD = parse(Bool, lowercase(get(ENV, "DPN_ASOF_FORCE_REBUILD", "false")))
# Escape hatch for the pre-P0 archive guard (see PHASE B0).
const SKIP_ARCHIVE_GUARD = parse(Bool, lowercase(get(ENV, "DPN_P0_SKIP_ARCHIVE_GUARD", "false")))

con = dbcon(memory_gb=6, threads=4)

# ----------------------------------------------------------------------------
# SPILL LOCATION. dbcon() puts DuckDB's temp_directory under tempdir() (C:), but
# this step's aggregations do not fit in the 6 GB memory_limit: the Phase C
# duplicate-key GROUP BY over 208M rows spilled 4+ GB and killed a full run with
# "IO Error: Could not write file ... duckdb_temp_storage ... 磁盘空间不足"
# (disk full) on 2026-08-04, AFTER the 6.93 GB panel had already been written.
# C: holds the 6.93 GB new panel plus the 4.66 GB pre-P0 twin, so it has no room
# to spare. Spill to the raw-data volume instead, which has hundreds of GB.
# Override with DPN_DUCKDB_TEMP_DIR.
let spill = get(ENV, "DPN_DUCKDB_TEMP_DIR",
                joinpath(dirname(rstrip(RAW_PARQUET_DIR, ['\\', '/'])), "_duckdb_spill_03"))
    try
        isdir(spill) || mkpath(spill)
        DBInterface.execute(con, "SET temp_directory='$(replace(spill, "\\" => "/"))'")
        println("DuckDB spill dir: $spill")
    catch e
        @warn "could not set temp_directory; falling back to the dbcon() default on C:. " *
              "Phase C may fail with a disk-full IO Error." spill exception=e
    end
end

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
        @warn "Raw parquet cache contains unexpected chunks:\n  " * join(extras, "\n  ") *
              "\nThe P0 per-chunk loop iterates EXPECTED_CHUNKS only, so these are IGNORED " *
              "(the old wildcard scan would have picked them up). Add them to EXPECTED_CHUNKS if they are real."
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
println("AS-OF window W  : $ASOF_WINDOW_DAYS days (DPN_ASOF_WINDOW_DAYS)")

eom_path = test_suffix_path(joinpath(OUT_DIR, "holdings_eom.parquet"))
const STEP_INPUTS = [joinpath(RAW_PARQUET_DIR, c) for c in EXPECTED_CHUNKS]

# Per-chunk parts live OFF the OneDrive-synced OUT_DIR (~5 GB of churn).
const ASOF_PARTS_DIR = let d = get(ENV, "DPN_ASOF_PARTS_DIR", joinpath(RAW_PARQUET_DIR, "_p0_asof_parts"))
    TEST_MODE ? d * TEST_SUFFIX : d
end
isdir(ASOF_PARTS_DIR) || mkpath(ASOF_PARTS_DIR)

# ============================================================
# PHASE A: PRE-ETL AUDIT (optional via DPN_SKIP_AUDIT)
# Operates on RAW report dates (= report_date_actual downstream), which is
# exactly the distribution the as-of rule has to cope with.
# ============================================================
if !SKIP_AUDIT
    println("\n========== PHASE A: pre-ETL audit ==========")
    DBInterface.execute(con, """
        CREATE OR REPLACE TABLE audit_cache AS
        SELECT FACTSET_FUND_ID, factset_sec_entity_id, ISSUE_TYPE, ADJ_MV, ISO_COUNTRY,
               CAST(REPORT_DATE AS DATE) AS report_date_actual,
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
        WHERE report_date_actual = DATE '2022-01-31'
        GROUP BY ISSUE_TYPE ORDER BY n_rows DESC
    """)
    println("\nISSUE_TYPE breakdown (2022-01-31):")
    println(audit_issue)
    CSV.write(joinpath(OUT_DIR, "03_audit_issue_type_holdings.csv"), audit_issue)

    audit_dates = qdf(con, "SELECT dom, COUNT(*) AS n_rows FROM audit_cache GROUP BY dom ORDER BY dom")
    println("\nReport-date day-of-month distribution (Jan 2022, RAW dates):")
    println(audit_dates)
    CSV.write(joinpath(OUT_DIR, "03_audit_dom_distribution.csv"), audit_dates)

    DBInterface.execute(con, "DROP TABLE audit_cache")

    # P0 addition: the raw-date pile-up around the four 2022-2023 quarter-ends,
    # i.e. the exact pattern the as-of rule exists to absorb. Cheap (date-filtered).
    println("\nRAW report-date pile-up around recent quarter-ends (the P0 problem, raw data):")
    pileup = qdf(con, """
        SELECT CAST(REPORT_DATE AS DATE) AS report_date_actual,
               dayname(CAST(REPORT_DATE AS DATE)) AS dow,
               COUNT(DISTINCT FACTSET_FUND_ID)    AS n_funds,
               COUNT(*)                           AS n_rows
        FROM read_parquet('$holdings_pattern')
        WHERE ADJ_MV IS NOT NULL AND ADJ_MV > 0
          AND (CAST(REPORT_DATE AS DATE) BETWEEN DATE '2022-12-20' AND DATE '2022-12-31'
            OR CAST(REPORT_DATE AS DATE) BETWEEN DATE '2023-09-20' AND DATE '2023-09-30')
        GROUP BY 1,2 ORDER BY 1
    """)
    println(pileup)
    CSV.write(joinpath(OUT_DIR, "03_audit_qend_pileup.csv"), pileup)
else
    println("\n========== PHASE A SKIPPED (DPN_SKIP_AUDIT = true) ==========")
end

# ============================================================
# PHASE B0: PRE-P0 ARCHIVE GUARD
# The P0 spec requires archive-by-RENAME of every artifact this chain
# overwrites BEFORE the rebuild. Refuse to clobber an existing canonical
# holdings_eom.parquet that has no pre-P0 archive beside it.
# ============================================================
# NOTE the !TEST_MODE gate. In test mode eom_path already carries the _TESTMODE
# suffix, which is ITSELF the mechanism that stops a test run touching the
# canonical artifact — so the guard has nothing to protect and would instead
# demand a holdings_eom_TESTMODE_exactEOM_preP0.parquet that no workflow creates
# and archive_preP0.py does not know about. Worse, its printed remedy is
# DPN_P0_SKIP_ARCHIVE_GUARD=true, a sticky shell env var: a false trip in a
# harmless context trains the operator to disable the guard that matters, and if
# it is still set when the real full run starts, 03 silently overwrites an
# un-archived canonical panel.
let archive_path = replace(eom_path, ".parquet" => "_exactEOM_preP0.parquet")
    if TEST_MODE
        println("\n[PHASE B0] archive guard SKIPPED: TEST_MODE (output is $(basename(eom_path)), " *
                "the canonical artifact cannot be reached).")
    end
    if !TEST_MODE && isfile(eom_path) && !isfile(archive_path) && !SKIP_ARCHIVE_GUARD
        error("""
        REFUSING TO OVERWRITE the existing pre-P0 artifact.

          exists : $eom_path
          missing: $archive_path

        The P0 spec requires archive-by-RENAME (same volume, cheap) before the
        rebuild, so the exact-EOM vintage stays reproducible. Run:

          mv "$eom_path" "$archive_path"

        ...and archive the downstream artifacts this chain feeds
        (merged_us_eu_zero_filled.parquet, c6_panel.dta, audit_c6_panel.dta,
        audit_c6_panel.parquet, ownership_ict.parquet -> *_preP0.*) too.

        Set DPN_P0_SKIP_ARCHIVE_GUARD=true ONLY if the archive already exists
        elsewhere and you know what you are discarding.
        """)
    end
end

# ============================================================
# PHASE B: ETL — as-of quarter-end selection, per-chunk, then merge.
# Filters:
#   * NO ISSUE_TYPE filter (downstream 04/05 apply their own)
#   * ADJ_MV > 0                                     [UNCHANGED]
#   * as-of quarter-end within ASOF_WINDOW_DAYS      [WAS: exact LAST_DAY]
# ============================================================
println("\n========== PHASE B: ETL (as-of quarter-end, W = $ASOF_WINDOW_DAYS d) ==========")
println("Filters: ADJ_MV > 0 + as-of quarter-end snapshot (NO ISSUE_TYPE filter)")
println("Parts dir: $ASOF_PARTS_DIR")
println("Output   : $eom_path")

# Quarter-end of a row's own quarter. For W <= 89 this is the ONLY quarter-end a
# row can serve, so no quarter cross join is needed (see header note).
const QEND_SQL = "CAST(DATE_TRUNC('quarter', CAST(REPORT_DATE AS DATE)) + INTERVAL 3 MONTH - INTERVAL 1 DAY AS DATE)"

"""
    asof_select_sql(src_glob) -> String

Per (fund_id, fsym_id, quarter): keep the row with the LATEST report date inside
[qend - W, qend]. Ties (same fund, same security, same date — these exist in the
raw feed) are broken deterministically so the build is reproducible.
"""
function asof_select_sql(src_glob::AbstractString)
    return """
    SELECT
        fund_id, entity_name, investor_country, entity_type,
        fsym_id, fsym_primary_id, listing_flag,
        cusip, isin, sedol, sec_country, issue_type, cap_group,
        sec_entity_id, sec_entity_name,
        report_date,
        adj_holding, adj_mv, adj_shares_out, adj_price,
        report_date_actual, asof_gap_days, n_asof_candidates
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY fund_id, fsym_id, report_date
                   ORDER BY report_date_actual DESC,
                            adj_mv DESC,
                            adj_holding DESC,
                            COALESCE(isin, ''),
                            COALESCE(cusip, ''),
                            COALESCE(sedol, ''),
                            COALESCE(sec_entity_id, ''),
                            COALESCE(fsym_primary_id, ''),
                            COALESCE(issue_type, '')
               ) AS rn,
               COUNT(*) OVER (PARTITION BY fund_id, fsym_id, report_date) AS n_asof_candidates
        FROM (
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
                $QEND_SQL                  AS report_date,
                CAST(REPORT_DATE AS DATE)  AS report_date_actual,
                DATEDIFF('day', CAST(REPORT_DATE AS DATE), $QEND_SQL) AS asof_gap_days,
                ADJ_HOLDING                AS adj_holding,
                ADJ_MV                     AS adj_mv,
                ADJ_SHARES_OUTSTANDING     AS adj_shares_out,
                ADJ_PRICE                  AS adj_price
            FROM read_parquet('$src_glob')
            WHERE ADJ_MV IS NOT NULL
              AND ADJ_MV > 0
        )
        WHERE asof_gap_days >= 0
          AND asof_gap_days <= $ASOF_WINDOW_DAYS
    )
    WHERE rn = 1
    """
end

part_paths = String[]
for chunk in EXPECTED_CHUNKS
    src  = replace(joinpath(RAW_PARQUET_DIR, chunk), "\\" => "/")
    stem = replace(chunk, ".parquet" => "")
    # W is IN the filename: a part built under a different W can never be reused.
    part = replace(joinpath(ASOF_PARTS_DIR, "$(stem)_asofW$(ASOF_WINDOW_DAYS).parquet"), "\\" => "/")
    part_native = replace(part, "/" => "\\")
    push!(part_paths, part)

    if isfile(part_native) && filesize(part_native) > 0 && !ASOF_FORCE_REBUILD
        # HAZARD: reuse keys ONLY on (chunk name, W). It does NOT fingerprint the
        # raw input file or the SQL text, so a part built before an SQL edit, or
        # before 03a regenerated the raw cache, would be merged silently. PRE-MERGE
        # GATE M2 below is the check that makes that detectable; keep it.
        println("  [skip] $(basename(part)) already built ($(round(filesize(part_native)/1024^3, digits=2)) GB). " *
                "Reuse keys on (chunk, W) ONLY — no raw/SQL fingerprint. " *
                "DPN_ASOF_FORCE_REBUILD=true to redo.")
        continue
    end
    println("  [build] $chunk -> $(basename(part))")
    @time atomic_copy_to(con, asof_select_sql(src), part)
    println("          $(round(filesize(part_native)/1024^3, digits=2)) GB")
end

# Independence assumption: chunks split on YEAR boundaries and a quarter never
# spans a year, so no (fund, fsym, quarter) key can appear in two parts. Verify
# it rather than assume it — a chunk holding out-of-range years would silently
# split a quarter across parts and defeat the dedup.
println("\nPer-part year-range check (chunk name vs data):")
for (chunk, part) in zip(EXPECTED_CHUNKS, part_paths)
    m = match(r"Factset_FundOwners_(\d{4})_(\d{4})\.parquet", chunk)
    rng = qdf(con, """
        SELECT MIN(EXTRACT(YEAR FROM report_date)) AS y0,
               MAX(EXTRACT(YEAR FROM report_date)) AS y1,
               COUNT(*) AS n
        FROM read_parquet('$part')
    """)
    y0, y1, n = rng.y0[1], rng.y1[1], rng.n[1]
    println("  $chunk: quarters span $y0-$y1, $n rows")
    if m !== nothing && n > 0
        exp0, exp1 = parse(Int, m.captures[1]), parse(Int, m.captures[2])
        if y0 < exp0 || y1 > exp1
            error("Chunk $chunk contains quarter-ends in $y0-$y1, outside its declared range " *
                  "$exp0-$exp1. The per-chunk dedup assumption is violated — a (fund, fsym, quarter) " *
                  "key can now span two parts. Rebuild with a single global pass or re-split the chunks.")
        end
    end
end

parts_glob = replace(joinpath(ASOF_PARTS_DIR, "*_asofW$(ASOF_WINDOW_DAYS).parquet"), "\\" => "/")
n_parts = length(filter(f -> endswith(f, "_asofW$(ASOF_WINDOW_DAYS).parquet"), readdir(ASOF_PARTS_DIR)))
@assert n_parts == length(EXPECTED_CHUNKS) "Expected $(length(EXPECTED_CHUNKS)) parts at W=$ASOF_WINDOW_DAYS, found $n_parts in $ASOF_PARTS_DIR"

# ============================================================
# PRE-MERGE GATES — run against the PARTS, before anything touches the canonical
# path. These used to live after the merge + write_manifest, which meant a failed
# run left a corrupt holdings_eom.parquet on disk CERTIFIED VALID by its own
# manifest (row count + sha256), with the pre-P0 original already renamed away.
# The error text said "Do NOT use this panel" but nothing enforced it: 05, 06,
# build_desc_trend_china_links.py and build_desc_trend_us_holdings.py all read
# holdings_eom directly with no duplicate-key guard of their own (only 04 has
# one). Fail here instead, where the canonical path is still untouched.
# ============================================================
println("\n========== PRE-MERGE GATES (on parts; canonical path untouched) ==========")

# ---- M1: exactly one row per (fund_id, fsym_id, quarter) across all parts ----
let d = qdf(con, """
        SELECT COUNT(*) AS n_dup_keys, COALESCE(SUM(c - 1), 0) AS n_excess_rows
        FROM (
            SELECT fund_id, fsym_id, report_date, COUNT(*) c
            FROM read_parquet('$parts_glob') GROUP BY 1,2,3 HAVING COUNT(*) > 1
        )
    """)
    println("  M1 duplicate (fund_id, fsym_id, quarter) keys in parts: $(d.n_dup_keys[1]) " *
            "(excess rows: $(d.n_excess_rows[1]))")
    if d.n_dup_keys[1] > 0
        println(qdf(con, """
            SELECT fund_id, fsym_id, report_date, COUNT(*) c
            FROM read_parquet('$parts_glob') GROUP BY 1,2,3
            HAVING COUNT(*) > 1 ORDER BY c DESC LIMIT 10
        """))
        error("""
        P0 PRE-MERGE GATE M1 FAILED: $(d.n_dup_keys[1]) duplicate
        (fund_id, fsym_id, quarter) keys survived the ROW_NUMBER dedup.
        Downstream SUM(adj_mv) would over-count. NOTHING was written to
        $eom_path — the pre-P0 chain state is unchanged. Parts: $parts_glob
        """)
    end
    println("  [OK] M1 one-row-per-(fund, security, quarter).")
end

# ---- M2: the as-of rule must be a STRICT SUPERSET of the old exact-EOM rule ---
# The old rule (LAST_DAY + month IN (3,6,9,12)) is provably identical to
# asof_gap_days = 0, and a gap = 0 row is ALWAYS rn = 1 because the ORDER BY
# leads on report_date_actual DESC and gap = 0 is the maximum attainable actual
# date in the window. So every old-rule row must reappear, one-for-one, unless
# two raw rows shared a (fund_id, fsym_id, report_date_actual) key — and that
# count is ZERO sample-wide (verified 2026-08-04 on the live pre-P0 panel, all
# 25 years, 186,800,295 rows, 0 duplicate keys, 0 excess rows).
#
# WHY THIS GATE EXISTS. The part-reuse skip above keys only on (chunk name, W),
# with no fingerprint of the raw input or of the SQL text. A re-run after any
# SQL edit, or after 03a regenerates the raw cache, would silently merge STALE
# parts — and every other gate in this script would still pass. This is the only
# check that makes such a regression detectable.
const EXPECTED_OLD_ROWS = 186_800_295   # live pre-P0 holdings_eom.parquet, re-derived 2026-08-04
if TEST_MODE
    println("  M2 SKIPPED: TEST_MODE reads one chunk, so the full-sample old-rule " *
            "row count is not comparable.")
else
    n_old = qdf(con, "SELECT COUNT(*) AS n FROM read_parquet('$parts_glob') WHERE asof_gap_days = 0").n[1]
    println("  M2 old-rule rows recovered (asof_gap_days = 0): $n_old " *
            "(expected $EXPECTED_OLD_ROWS)")
    if n_old != EXPECTED_OLD_ROWS
        delta = n_old - EXPECTED_OLD_ROWS
        # Only pay for this scan on the error path. It separates the two causes:
        # a legitimate raw-vintage change shows up as raw within-date duplicate
        # keys and/or a changed raw row count; a stale-part bug does not.
        println("  M2 MISMATCH — scanning the RAW feed to attribute the delta " *
                "(this takes a few minutes) ...")
        raw = qdf(con, """
            SELECT COUNT(*)                               AS n_raw_qend_rows,
                   SUM(CASE WHEN c > 1 THEN 1 ELSE 0 END) AS n_raw_dup_keys,
                   SUM(c) - COUNT(*)                      AS n_raw_excess_rows
            FROM (
                SELECT FACTSET_FUND_ID, FSYM_ID, CAST(REPORT_DATE AS DATE) AS d, COUNT(*) AS c
                FROM read_parquet('$holdings_pattern')
                WHERE ADJ_MV IS NOT NULL AND ADJ_MV > 0
                  AND CAST(REPORT_DATE AS DATE) = LAST_DAY(CAST(REPORT_DATE AS DATE))
                  AND EXTRACT(MONTH FROM CAST(REPORT_DATE AS DATE)) IN (3, 6, 9, 12)
                GROUP BY 1,2,3
            )
        """)
        error("""
        P0 PRE-MERGE GATE M2 FAILED: the as-of rule is NOT a strict superset of
        the old exact-EOM rule.

          old-rule rows recovered at asof_gap_days = 0 : $n_old
          expected (live pre-P0 panel, 2026-08-04)     : $EXPECTED_OLD_ROWS
          delta                                        : $delta

        RAW feed attribution (exact-quarter-end rows, ADJ_MV > 0):
          raw rows at an exact quarter-end date        : $(raw.n_raw_qend_rows[1])
          raw within-date duplicate (fund, fsym, date) keys : $(raw.n_raw_dup_keys[1])
          raw excess rows on those keys                : $(raw.n_raw_excess_rows[1])

        HOW TO READ IT:
          * raw_dup_keys > 0 and delta == -raw_excess_rows
                -> LEGITIMATE: the raw vintage now contains within-date duplicate
                   keys that the new ROW_NUMBER dedup collapses. Update
                   EXPECTED_OLD_ROWS in this file (a committed edit, on purpose)
                   and note it in VINTAGE_P0.md.
          * raw_dup_keys == 0 (or the delta does not reconcile)
                -> STALE PARTS or an SQL regression. The per-chunk parts in
                   $ASOF_PARTS_DIR are keyed only on (chunk, W); they do not
                   fingerprint the raw input or the SQL. Delete them, or set
                   DPN_ASOF_FORCE_REBUILD=true, and re-run.

        NOTHING was written to $eom_path — the pre-P0 chain state is unchanged.
        """)
    end
    println("  [OK] M2 strict-superset: every old-rule row recovered one-for-one.")
end

println("\nMerging $n_parts parts -> $eom_path")
@time atomic_copy_to(con, "SELECT * FROM read_parquet('$parts_glob')", eom_path)
eom_path_fwd = replace(eom_path, "\\" => "/")
println("File size: $(round(filesize(eom_path)/1024^3, digits=2)) GB")

# Belt-and-braces demotion for the POST-merge asserts below. If one of them
# fires, the canonical artifact on disk is corrupt and must not be readable as
# holdings_eom.parquet by 04/05/06/the desc_trend scripts. Strip the manifest
# (so nothing certifies it) and rename it out of the way, THEN error.
function fail_canonical(msg::AbstractString)
    meta = replace(eom_path, "/" => "\\") * ".meta.json"
    isfile(meta) && rm(meta; force=true)
    failed = replace(eom_path, ".parquet" => "_FAILED.parquet")
    try
        isfile(eom_path) && mv(eom_path, failed; force=true)
    catch e
        @warn "could not rename the corrupt canonical artifact" exception=e
    end
    error(msg * "\n\nThe corrupt output was DEMOTED to:\n  $failed\n" *
          "and its .meta.json was deleted, so no downstream script can read it as\n" *
          "holdings_eom.parquet and nothing certifies it as valid.")
end

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
println("\n========== PHASE C: diagnostics on holdings_eom ==========")

# ---------------------------------------------------------------------------
# C1: HARD ASSERT — exactly one row per (fund_id, fsym_id, quarter).
# report_date is now the STAMPED quarter-end, so (fund_id, fsym_id, report_date)
# IS the (fund, security, quarter) key. Under the old rule this was a warn-only
# threshold; the as-of rule dedups by construction, so any survivor is a bug.
#
# BELT AND BRACES ONLY. The authoritative version of this check is PRE-MERGE
# GATE M1 above, which runs against the parts while the canonical path is still
# untouched. This copy catches a merge-level corruption; because it runs AFTER
# atomic_copy_to and write_manifest, its failure path must DEMOTE the artifact
# (fail_canonical) rather than leave a corrupt panel certified by its own
# manifest.
# ---------------------------------------------------------------------------
dup_check = qdf(con, """
    SELECT COUNT(*) AS n_dup_keys, COALESCE(SUM(c - 1), 0) AS n_excess_rows
    FROM (
        SELECT fund_id, fsym_id, report_date, COUNT(*) c
        FROM eom GROUP BY 1,2,3 HAVING COUNT(*) > 1
    )
""")
println("Duplicate (fund_id, fsym_id, quarter) keys: $(dup_check.n_dup_keys[1]) " *
        "(excess rows: $(dup_check.n_excess_rows[1]))")
CSV.write(joinpath(OUT_DIR, "03_eom_dup_key_count.csv"), dup_check)
if dup_check.n_dup_keys[1] > 0
    offenders = qdf(con, """
        SELECT fund_id, fsym_id, report_date, COUNT(*) c
        FROM eom GROUP BY 1,2,3 HAVING COUNT(*) > 1 ORDER BY c DESC LIMIT 10
    """)
    println(offenders)
    fail_canonical("P0 HARD ASSERT FAILED (post-merge C1): $(dup_check.n_dup_keys[1]) " *
                   "duplicate (fund_id, fsym_id, quarter) keys in the merged panel, " *
                   "even though PRE-MERGE GATE M1 passed on the parts. That means the " *
                   "merge itself duplicated rows (e.g. the parts glob matched a stray " *
                   "file). Downstream SUM(adj_mv) would over-count.")
end
println("  [OK] one-row-per-(fund, security, quarter) assert passed.")

# ---------------------------------------------------------------------------
# C1b: strict-superset re-check on the MERGED panel (belt and braces for M2).
# M2 ran on the parts glob; this confirms the merge preserved the count. Same
# demotion discipline on failure.
# ---------------------------------------------------------------------------
if TEST_MODE
    println("  C1b SKIPPED: TEST_MODE (see M2).")
else
    n_old_merged = qdf(con, "SELECT COUNT(*) AS n FROM eom WHERE asof_gap_days = 0").n[1]
    println("Old-rule rows in merged panel (asof_gap_days = 0): $n_old_merged " *
            "(expected $EXPECTED_OLD_ROWS)")
    if n_old_merged != EXPECTED_OLD_ROWS
        fail_canonical("P0 HARD ASSERT FAILED (post-merge C1b): merged panel has " *
                       "$n_old_merged rows at asof_gap_days = 0, expected " *
                       "$EXPECTED_OLD_ROWS. PRE-MERGE GATE M2 passed on the parts, so " *
                       "the merge dropped or duplicated old-rule rows.")
    end
    println("  [OK] strict-superset assert passed on the merged panel.")
end

# ---------------------------------------------------------------------------
# C2: GATE 1 + GATE 2 — per-quarter coverage and US share, OLD rule vs NEW rule.
# The OLD rule's selection is recoverable from the new panel: it kept exactly
# the rows whose actual report date IS the calendar quarter-end, i.e.
# asof_gap_days = 0. (Fund/security COUNTS are identical to the old build; ROW
# counts are the old rows AFTER the same dedup, which is what isolates the
# as-of change from the dedup change.)
# ---------------------------------------------------------------------------
oldnew = qdf(con, """
    SELECT report_date                                        AS qend,
           dayname(report_date)                               AS qend_dow,
           CASE WHEN dayname(report_date) IN ('Saturday','Sunday') THEN 1 ELSE 0 END AS qend_is_weekend,
           COUNT(*)                                           AS n_rows_new,
           SUM(CASE WHEN asof_gap_days = 0 THEN 1 ELSE 0 END) AS n_rows_old_dedup,
           COUNT(DISTINCT fund_id)                            AS n_funds_new,
           COUNT(DISTINCT CASE WHEN asof_gap_days = 0 THEN fund_id END) AS n_funds_old,
           COUNT(DISTINCT CASE WHEN investor_country = 'US' THEN fund_id END) AS n_us_funds_new,
           COUNT(DISTINCT CASE WHEN asof_gap_days = 0 AND investor_country = 'US' THEN fund_id END) AS n_us_funds_old,
           AVG(CAST(asof_gap_days AS DOUBLE))                 AS mean_gap_days,
           SUM(n_asof_candidates) - COUNT(*)                  AS n_rows_collapsed_by_dedup,
           SUM(adj_mv)/1e9                                    AS mv_new_b,
           SUM(CASE WHEN asof_gap_days = 0 THEN adj_mv ELSE 0 END)/1e9 AS mv_old_b
    FROM eom
    GROUP BY 1,2,3 ORDER BY 1
""")
oldnew.fund_ratio_new_over_old = oldnew.n_funds_new ./ max.(oldnew.n_funds_old, 1)
oldnew.us_share_old = oldnew.n_us_funds_old ./ max.(oldnew.n_funds_old, 1)
oldnew.us_share_new = oldnew.n_us_funds_new ./ max.(oldnew.n_funds_new, 1)
CSV.write(joinpath(OUT_DIR, "03_eom_asof_coverage_old_vs_new.csv"), oldnew)

println("\n--- GATE 1: per-quarter fund counts, OLD rule vs NEW rule ---")
println(oldnew[:, [:qend, :qend_dow, :n_funds_old, :n_funds_new, :fund_ratio_new_over_old,
                   :n_rows_old_dedup, :n_rows_new, :mean_gap_days]])

let wd = oldnew[oldnew.qend_is_weekend .== 0, :], we = oldnew[oldnew.qend_is_weekend .== 1, :]
    if nrow(we) > 0 && nrow(wd) > 0
        # Coverage here is relative to the NEW capture (the raw-data denominator
        # lives in diag_p0_asof_window.py); the gate is that the weekday-vs-
        # weekend GAP in the old/new ratio collapses.
        println("\nGATE 1 summary (quarter-ends: $(nrow(wd)) weekday, $(nrow(we)) weekend)")
        println("  OLD rule, funds captured as a share of NEW-rule funds:")
        println("    weekday mean = $(round(100*mean(wd.n_funds_old ./ wd.n_funds_new), digits=2))%")
        println("    weekend mean = $(round(100*mean(we.n_funds_old ./ we.n_funds_new), digits=2))%")
        println("    gap          = $(round(100*(mean(wd.n_funds_old ./ wd.n_funds_new) -
                                                 mean(we.n_funds_old ./ we.n_funds_new)), digits=2)) pp   <-- was the bug")
        println("\n  GATE 2: US share of captured funds")
        println("    OLD  weekday = $(round(100*mean(wd.us_share_old), digits=2))%   weekend = $(round(100*mean(we.us_share_old), digits=2))%   diff = $(round(100*(mean(wd.us_share_old)-mean(we.us_share_old)), digits=2)) pp")
        println("    NEW  weekday = $(round(100*mean(wd.us_share_new), digits=2))%   weekend = $(round(100*mean(we.us_share_new), digits=2))%   diff = $(round(100*(mean(wd.us_share_new)-mean(we.us_share_new)), digits=2)) pp")
    else
        println("\nGATE 1/2 summary skipped: sample has no weekend (or no weekday) quarter-ends.")
    end
end

# C2b: as-of gap distribution — how much of the panel is a same-day snapshot?
gap_dist = qdf(con, """
    SELECT asof_gap_days, COUNT(*) AS n_rows,
           COUNT(DISTINCT fund_id) AS n_funds,
           SUM(adj_mv)/1e9 AS mv_b
    FROM eom GROUP BY 1 ORDER BY 1
""")
println("\nAs-of gap distribution (0 = exact quarter-end snapshot):")
println(gap_dist)
CSV.write(joinpath(OUT_DIR, "03_eom_asof_gap_distribution.csv"), gap_dist)

# C2c: raw actual-date distribution (replaces the old weekend-EOM audit)
weekend_audit = qdf(con, """
    SELECT report_date AS qend,
           report_date_actual,
           dayname(report_date_actual) AS actual_dow,
           asof_gap_days,
           COUNT(*) AS n_rows,
           COUNT(DISTINCT fund_id) AS n_funds
    FROM eom
    GROUP BY 1,2,3,4 ORDER BY 1,2
""")
CSV.write(joinpath(OUT_DIR, "03_eom_weekend_audit.csv"), weekend_audit)
println("Actual-report-date audit -> 03_eom_weekend_audit.csv")

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
# (report_date is the STAMPED quarter-end, so these literals still resolve.)
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
           dayname(report_date) AS qend_dow,
           COUNT(*) AS n_holdings,
           COUNT(DISTINCT fund_id) AS n_us_funds,
           COUNT(DISTINCT sec_entity_id) AS n_eu_firms,
           SUM(adj_mv)/1e9 AS total_mv_billions,
           SUM(CASE WHEN asof_gap_days = 0 THEN adj_mv ELSE 0 END)/1e9 AS total_mv_billions_oldrule
    FROM eom
    WHERE investor_country = 'US' AND sec_country IN $EU_SQL_TUPLE
    GROUP BY 1,2 ORDER BY report_date
""")
CSV.write(joinpath(OUT_DIR, "03_us_x_eu_cells_by_month.csv"), us_eu_cells)
println("\nUS investor × EU firm panel (last 12 quarters, NEW vs OLD \$B):")
println(last(us_eu_cells, 12))

DBInterface.execute(con, "DROP VIEW eom")

try
    DBInterface.close!(con)
catch e
    @warn "DBInterface.close! failed" exception=e
end

println("\n========== ETL DONE ==========")
println("Output: $eom_path   (as-of W = $ASOF_WINDOW_DAYS days)")
println("\nKey diagnostic files:")
println("  03_eom_asof_coverage_old_vs_new.csv  (GATE 1 + GATE 2, per quarter)")
println("  03_eom_asof_gap_distribution.csv     (how stale the panel is)")
println("  03_eom_dup_key_count.csv             (hard assert, must be 0)")
println("  03_eom_weekend_audit.csv             (actual report dates behind each quarter)")
println("  03_eom_coverage_by_year.csv")
println("  03_eom_issue_type_breakdown.csv")
println("  03_us_x_eu_cells_by_month.csv        (NEW vs OLD-rule \$ totals)")
println("\nParts kept at: $ASOF_PARTS_DIR (delete once the merge is verified)")
println("\nNext: 04_us_ownership_european.jl")
