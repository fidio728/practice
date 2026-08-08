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
# SNAPSHOT GRAIN — PER-FUND (DEFAULT since 2026-08-08) vs PER-SECURITY (LEGACY)
# ============================================================================
#
# DPN_SNAPSHOT_GRAIN = fund      (DEFAULT) "last observation inside the
#     quarter" is the ADVISOR'S rule; applying it at FUND grain (last complete
#     report, rather than each security's own last positive row) is a
#     RESEARCHER DECISION, adjudicated by the reappearance diagnostic cited
#     below — the advisor directive did not specify the grain. Rule: per
#     (fund_id, quarter), d_use =
#     MAX(report_date_actual) among the fund's in-window ADJ_MV > 0 reports;
#     keep ALL rows of that single report. A security absent from the d_use
#     report was sold / is no longer held and contributes NOTHING to the
#     quarter — which is correct (see the RESOLUTION note in the chimera block
#     below). report_date stays stamped to the quarter-end; on every kept row
#     report_date_actual = d_use (uniform within the fund-quarter) and
#     asof_gap_days = qend - d_use.
# DPN_SNAPSHOT_GRAIN = security  legacy rule (P0 / EM vintages): per
#     (fund_id, fsym_id, quarter) keep the pair's own last in-window row. This
#     path is BIT-IDENTICAL to the pre-2026-08-08 code (same SQL text, same
#     part filenames, so existing security-grain parts still reuse) and is
#     retained for sensitivity / attribution runs only.
#
# WHY FUND GRAIN IS THE DEFAULT (diag_snapshot_reappearance.py, 2026-08-08,
# raw 2018-2023, 104.8M security-quarters):
#   * each (fund, report_date) IS a complete portfolio snapshot: 92.8% of
#     multi-date fund-quarters have a last report >= 90% of the quarter-max
#     report size, median ratio 1.0, only 0.56% below 50%;
#   * securities present earlier in the quarter but absent from the fund's
#     LAST in-quarter report reappear next quarter only 10.76% (count) /
#     12.34% (MV-weighted), vs a 95.23% / 82.70% baseline -> carried positions
#     are GENUINE EXITS, not reporting gaps;
#   * carried MV = 0.41% of 2018-2023 MV (full-sample C2d: 15.3%, concentrated
#     pre-2012).
#
# The grain is ORTHOGONAL to the window mode documented below:
# DPN_ASOF_WINDOW_DAYS still selects the CANDIDATE window ([quarter_start,
# qend] by default), the grain selects WHAT is kept from it (the fund's last
# report vs each pair's last row). Part filenames carry the grain in the rule
# tag (asofQTRFg / asofW10Fg vs asofQTR / asofW10), so a part built under one
# grain can never be reused under the other.
#
# ============================================================================
# SNAPSHOT SELECTION — "LAST REPORT INSIDE THE CALENDAR QUARTER"
# ADVISOR-DIRECTED, 2026-08-06 (supersedes the W = 10 as-of window of the
# 2026-08-04 P0 rebuild; the W window survives as a numeric override only)
# ============================================================================
#
# DIRECTIVE (Emanuele, 2026-08-04 advisor meeting, 24:17): take the LAST
# observation whose report_date falls INSIDE the calendar quarter
# [quarter_start_Q, qend_Q]; stamp report_date to the quarter-end; keep
# report_date_actual + asof_gap_days as provenance. The directive did NOT
# specify the grain. The 2026-08-06 implementation applied it per (fund_id,
# fsym_id) — a researcher choice inherited from the W=10 as-of design, later
# shown to stitch multi-date portfolios; since 2026-08-08 the default applies
# it per fund_id (last complete report), a researcher decision adjudicated by
# diag_snapshot_reappearance.py (see SNAPSHOT GRAIN block above). What is the
# advisor's: last-in-quarter + quarter-end stamping. What is ours: the grain.
#
# WHAT THIS CHANGES vs the W = 10 window shipped on 2026-08-04:
#   * the lower bound of the selection window is the QUARTER START, not
#     qend - 10. Intra-quarter reporters up to ~3 months stale are now IN.
#   * a (fund, security) pair with no report inside the quarter is still
#     ABSENT from that quarter. That is unchanged.
#   * ADJ_MV > 0 filter: unchanged. One-row-per-(fund, security, quarter)
#     hard assert: unchanged. Stamping to qend: unchanged. Column list:
#     unchanged (report_date_actual, asof_gap_days, n_asof_candidates were
#     already added by P0).
#
# WHY THE WINDOW CAN NEVER CROSS INTO THE PRIOR QUARTER. Every row is assigned
# to the quarter-end OF ITS OWN report date (QEND_SQL below is computed from
# REPORT_DATE, not from a quarter calendar joined on). So a row can only ever
# be a candidate for the quarter it already sits in, and the effective lower
# bound is exactly quarter_start. A FIXED 92-day window would NOT have this
# property (it would reach back across the quarter boundary on long quarters);
# this implementation is not a 92-day window and must not be described as one.
#
# ----------------------------------------------------------------------------
# ESTIMAND / UNIVERSE  (read this before using the panel for anything)
# ----------------------------------------------------------------------------
# WHAT A CELL IS. Each (fund, security, quarter) cell of holdings_eom.parquet is
# the fund's LAST REPORTED POSITION INSIDE that calendar quarter (default rule),
# or, under the numeric override, its most recent reported position within
# W = ASOF_WINDOW_DAYS calendar days BEFORE the quarter-end. It is NOT "the
# position on the quarter-end date".
#
# GRAIN NOTE (2026-08-08, DPN_SNAPSHOT_GRAIN=fund is the default): the cell is
# the fund's position in that security ON THE FUND'S d_use REPORT (the fund's
# last in-window snapshot). All cells of a fund-quarter then share ONE
# valuation date, and a security not on the d_use report has NO cell that
# quarter. Under DPN_SNAPSHOT_GRAIN=security the per-pair text above applies
# unchanged (each pair's own last report; valuation dates can differ within a
# fund-quarter — the chimera problem documented further down).
#
# WHAT THE UNIVERSE IS. The panel's fund universe for quarter Q is
#     "funds with a valid ADJ_MV > 0 report inside [quarter_start_Q, qend_Q]"
#         <- DEFAULT, advisor rule
#     "funds with a valid ADJ_MV > 0 report inside [qend_Q - W, qend_Q]"
#         <- numeric override DPN_ASOF_WINDOW_DAYS (P0 vintage, W = 10)
# NOT
#     "funds reporting exactly ON qend_Q"   <- the pre-P0 universe.
#
# THIS IS A UNIVERSE CHANGE ON EVERY QUARTER, NOT A WEEKEND-ONLY REPAIR.
# The rule admits EARLY REPORTERS — funds that stamped their snapshot before
# quarter-end — which the exact-EOM rule dropped even when the quarter-end fell
# on a business day. Under the quarter rule it additionally admits every
# intra-quarter reporter, so the fund universe expands FURTHER than the W = 10
# numbers below. Measured on the full 1999-2023 sample at W = 10
# (diag_p0_asof_weekday_weekend_gap.csv), for reference:
#     weekday quarter-end coverage  81.82%  ->  83.94%   (+2.12pp on the mean;
#                                                         +0.9 to +3.7pp per quarter)
#     weekend quarter-end coverage  59.49%  ->  79.76%
# The quarter rule takes both to ~100% by construction (any report in the
# quarter qualifies), i.e. the weekend coverage hole closes completely and the
# 1999-2005 residual documented below closes with it. The PRICE is staleness:
# report the realised gap distribution (share at gap 0, median, share > 14d,
# max) from 03_eom_asof_gap_summary.csv every time this panel is used. Do not
# present the expanded universe without that table.
# The weekday rise is EXPECTED and is NOT a gate failure. Any comparison of a
# level under this rule against a pre-P0 or a W = 10 level is a comparison
# across two universes.
#
# WHY W = 10 WAS THE P0 CHOICE — RETAINED AS AN INTERNAL ROBUSTNESS PATH ONLY.
# The block below is the evidence that justified W = 10 on 2026-08-04. It is NOT
# the live default any more; it documents what DPN_ASOF_WINDOW_DAYS=10 buys and,
# read the other way, exactly what the quarter rule now imports (the W = 31 and
# W = 92 rows are the honest preview of the staleness the default accepts).
# (numbers recorded here so the choice is reproducible from the ETL alone;
# source: diag_p0_asof_window.py -> output/diag_p0_asof_*.csv, full 1999-2023
# sample, 100 quarters):
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
#   0.26% at W = 10). W = 10 sits mid-plateau and WAS CONFIRMED for P0.
#
#   WHAT THE MULTI-REPORT SHARE ACTUALLY MEANS UNDER THE QUARTER RULE — READ THIS
#   BEFORE USING ANY CROSS-SECURITY SUM. Two different grains, two different
#   answers, and only the first one is deduplicated:
#     * at the (fund, fsym, quarter) grain: EXACTLY ONE row survives. The rule
#       takes the LAST report per pair inside the quarter. A pair that reported
#       three times contributes one row, its latest. No duplication, no union.
#     * at the FUND-PORTFOLIO grain: a fund's quarter book IS ASSEMBLED FROM
#       MULTIPLE REPORT DATES. Fund F reporting Jan-31 and Mar-31 contributes its
#       Mar-31 row for security A and its Jan-31 row for security B if B was sold
#       before quarter-end and therefore never appeared in the Mar-31 file. F's
#       Q1 "portfolio" is then stitched from up to three monthly snapshots up to
#       91 days apart, and it can carry positions the fund had already exited by
#       the quarter-end the row is stamped to.
#   This is FIRST ORDER, not a tail: the W = 92 row above says 34.6% (W = 31) to
#   40.9% (whole quarter) of fund-quarters report more than once inside the
#   window, versus 0.26% at W = 10 where the dispersion was bounded at 10 days.
#   EVERY CROSS-SECURITY SUM INHERITS IT. Specifically:
#     04_us_ownership_european.jl:141-153  I_ict = SUM(adj_mv) over funds x securities
#     04_us_ownership_european.jl:225-232  country_total_holdings_eu = SUM(I_ict)
#         -> the DENOMINATOR of portfolio_weight_eu, i.e. the outcome variable of
#            the headline regression and of Figure A;
#     04_us_ownership_european.jl:170-192  market_cap = AVG(price) x AVG(shares),
#            now averaging across valuation dates unless restricted (see the
#            MIN(asof_gap_days) restriction added there under EM-FIX-2).
#   Mitigation is measurement, not repair: the fund-level dispersion diagnostic in
#   PHASE C (C2d -> 03_eom_fund_snapshot_dispersion.csv) reports the share of
#   fund-quarters spanning more than one report date and the MV-weighted mean
#   spread in days. Report that table beside the coverage gain, always.
#   n_asof_candidates is the per-row audit of the per-pair candidate count and
#   will now commonly be 2 or 3 for monthly reporters (it was ~1 almost
#   everywhere at W = 10). It does NOT measure the fund-level dispersion above —
#   C2d does.
#
#   ------------------------------------------------------------------------
#   RESOLUTION (2026-08-08) — THE STITCHED FUND-PORTFOLIO ("CHIMERA") IS
#   RETIRED AS THE DEFAULT. The block above is kept as history; it now
#   describes DPN_SNAPSHOT_GRAIN=security only. The stitching problem it
#   documents was tested directly by diag_snapshot_reappearance.py
#   (2026-08-08, raw 2018-2023, 104.8M security-quarters):
#     * each (fund, report_date) IS a complete portfolio snapshot — 92.8% of
#       multi-date fund-quarters have a last report >= 90% of the quarter-max
#       report size, median ratio 1.0, only 0.56% below 50%;
#     * securities present earlier but absent from the fund's LAST in-quarter
#       report reappear next quarter only 10.76% count / 12.34% MV-weighted,
#       vs baseline 95.23% / 82.70% -> the positions the stitch "rescued" are
#       GENUINE EXITS, and the stitched book is wrong to include them;
#     * carried MV = 0.41% of 2018-2023 MV (full-sample C2d: 15.3%,
#       concentrated pre-2012).
#   The DEFAULT grain is therefore FUND (see the grain section at the top):
#   the fund's quarter book is its d_use report alone, no cross-date union, so
#   each FUND's contribution to every cross-security SUM in 04 (I_ict,
#   country_total_holdings_eu, market_cap) is single-valuation-date by
#   construction. CROSS-FUND date mixing REMAINS: those sums still aggregate
#   across funds whose d_use dates differ within the quarter (which is why
#   EM-FIX-2 exists — 04's market_cap restricts to MIN(asof_gap_days) rows;
#   I_ict / country totals carry the cross-fund dispersion, measured by C2d).
#   C2d must show ZERO multi-date fund-quarters WITHIN a fund (asserted there
#   under the fund grain).
#   ------------------------------------------------------------------------
#   Trend-free calendar artifact (local deviation vs neighbouring quarters):
#   11.11pp at W = 0 -> 1.06pp at W = 10; W = 3/14 give 1.07/1.08pp, so the choice
#   inside the plateau is not load-bearing.
#
# DISCLOSED CAVEAT (W-WINDOW PATH ONLY) — THE 1999-2005 GAP DOES NOT CLOSE. The
# text below applies to the numeric-W override. Under the default quarter rule
# the gap closes mechanically in every era, because scattered mid-quarter
# LionShares report dates are now INSIDE the window rather than outside it. That
# is a coverage fix, not a data fix: those early rows are genuinely stale and
# will show up in the tail of the gap distribution. The weekday-vs-weekend
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
#   P0 rule (2026-08-04, now the OVERRIDE): per (fund_id, fsym_id, quarter),
#   take the LATEST REPORT_DATE that is <= the calendar quarter-end and no more
#   than ASOF_WINDOW_DAYS before it.
#
#   DEFAULT rule (2026-08-06, advisor-directed): per (fund_id, fsym_id,
#   quarter), take the LATEST REPORT_DATE INSIDE [quarter_start, qend]. This is
#   the P0 rule with the lower bound moved from qend - W to the quarter start.
#
#   In both cases the surviving row is STAMPED with report_date = calendar
#   quarter-end, so every downstream quarter join, merge and FE is unchanged.
#   The true date is preserved in report_date_actual, with
#   asof_gap_days = qend - actual (0 .. 91 under the default, 0 .. W under the
#   override).
#
#   PROVENANCE COLUMNS (added by P0, unchanged here): report_date_actual,
#   asof_gap_days, n_asof_candidates.
#   (n_asof_candidates = how many raw rows shared this (fund, fsym, quarter)
#   key before the ROW_NUMBER dedup; it makes the dedup magnitude auditable
#   downstream without re-scanning 187M raw rows. At W = 10 it was ~1 for almost
#   every row; under the quarter rule it is 2-3 wherever a fund reports monthly,
#   which is expected and is the audit trail for the LAST-report tie-break.)
#
#   MODE SELECTION: DPN_ASOF_WINDOW_DAYS UNSET -> quarter rule (default).
#   DPN_ASOF_WINDOW_DAYS=<int in 0..89> -> window rule with lower bound
#   max(quarter_start, qend - W); W = 0 reproduces the pre-P0 exact-EOM rule and
#   W = 10 reproduces the 2026-08-04 P0 vintage. See diag_p0_asof_window.py for
#   the coverage-vs-W curve behind the W = 10 vintage.
#   DPN_SNAPSHOT_GRAIN (2026-08-08) selects the KEEP grain ON TOP of the window
#   mode: fund (default) keeps the fund's whole d_use report; security keeps
#   each pair's own last row (legacy, bit-identical to the pre-2026-08-08
#   code). See the grain section at the top of this file.
#
#   IMPLEMENTATION: per-chunk loop, NOT one global window function. A global
#   PARTITION BY (fund, fsym) over ~187M rows can OOM/spill under the 6 GB
#   memory_limit. Because raw chunks split on YEAR boundaries and a quarter
#   never spans a year, each chunk can be selected independently under BOTH
#   rules (each row serves only its own quarter, which lives entirely inside one
#   year); the parts are then merged with one COPY. W >= 90 is still REFUSED on
#   the override path: with the quarter clamp it is merely a slower spelling of
#   the default rule, so use the default instead of a 90+ window.
#
#   COST WARNING FOR THE DEFAULT RULE. Under the quarter rule the ROW_NUMBER /
#   COUNT windows see EVERY ADJ_MV > 0 row in the chunk, not just the rows within
#   10 days of quarter-end, so the per-chunk sort is several times larger than
#   the P0 build and WILL spill. Keep the spill directory on a volume with
#   >= 100 GB free (default: the raw-data volume; override DPN_DUCKDB_TEMP_DIR,
#   e.g. E:/duckdb_tmp). A spill onto C: is the documented failure mode.
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
# SNAPSHOT SELECTION MODE  (advisor-directed default + numeric override)
# ============================================================
# DPN_ASOF_WINDOW_DAYS UNSET / empty -> :quarter  (DEFAULT, advisor rule:
#     last report inside [quarter_start, qend])
# DPN_ASOF_WINDOW_DAYS = <int 0..89> -> :window   (P0 rule, lower bound
#     max(quarter_start, qend - W); W=10 = the 2026-08-04 vintage,
#     W=0 = the pre-P0 exact-EOM rule)
const ASOF_WINDOW_ENV = strip(get(ENV, "DPN_ASOF_WINDOW_DAYS", ""))
const ASOF_MODE = isempty(ASOF_WINDOW_ENV) ? :quarter : :window

const ASOF_WINDOW_DAYS = if ASOF_MODE == :window
    w = tryparse(Int, ASOF_WINDOW_ENV)
    w === nothing && error("""
    DPN_ASOF_WINDOW_DAYS = "$ASOF_WINDOW_ENV" is not an integer.
    Leave it UNSET for the default quarter rule, or set an integer in [0, 89]
    for the P0 window rule.
    """)
    w
else
    -1   # sentinel: no numeric window is in force
end

if ASOF_MODE == :window && (ASOF_WINDOW_DAYS < 0 || ASOF_WINDOW_DAYS > 89)
    error("""
    DPN_ASOF_WINDOW_DAYS = $ASOF_WINDOW_DAYS is out of the supported range [0, 89].

    W = 0  reproduces the OLD exact-quarter-end rule (kept for A/B checks).
    W = 10 reproduces the 2026-08-04 P0 vintage.
    W >= 90 is REFUSED on purpose. Under the quarter clamp a 90+ window cannot
    reach into the prior quarter (the row is assigned to its own quarter), so it
    is not unsafe — it is simply a slower, less legible spelling of the DEFAULT
    quarter rule. UNSET DPN_ASOF_WINDOW_DAYS instead of asking for W >= 90.
    """)
end

# ============================================================
# SNAPSHOT GRAIN  (2026-08-08; see the grain section in the file header)
# ============================================================
# DPN_SNAPSHOT_GRAIN = fund     -> per (fund, quarter) keep ALL rows of the
#                                  fund's LAST in-window report d_use (DEFAULT;
#                                  justified by diag_snapshot_reappearance.py)
# DPN_SNAPSHOT_GRAIN = security -> per (fund, fsym, quarter) keep the pair's
#                                  own last in-window row (legacy P0/EM
#                                  behaviour, bit-identical to the
#                                  pre-2026-08-08 code)
const SNAPSHOT_GRAIN = let g = lowercase(strip(get(ENV, "DPN_SNAPSHOT_GRAIN", "fund")))
    if !(g in ("fund", "security"))
        error("""
        DPN_SNAPSHOT_GRAIN = "$g" is not a supported snapshot grain.
        Use "fund" (DEFAULT: per (fund, quarter) keep ALL rows of the fund's
        last in-window report) or "security" (legacy: per (fund, fsym, quarter)
        keep the pair's own last row).
        """)
    end
    Symbol(g)
end

# Tag that identifies the rule in per-chunk part filenames and globs. A part
# built under one rule can therefore never be silently reused under another.
# NOTE the exact spelling: "W10" keeps the P0-vintage part names byte-identical
# so an existing W=10 parts directory still reuses. The FUND grain appends "Fg"
# (asofQTRFg / asofW10Fg): fund-grain parts live in their own namespace, the
# SECURITY-grain names stay byte-identical to the pre-2026-08-08 vintage (so
# existing security-grain parts still reuse), and a part built under one grain
# can never be merged under the other. (No glob overlap: "*_asofQTR.parquet"
# cannot match "..._asofQTRFg.parquet" and vice versa.)
const ASOF_TAG = (ASOF_MODE == :quarter ? "QTR" : "W$(ASOF_WINDOW_DAYS)") *
                 (SNAPSHOT_GRAIN == :fund ? "Fg" : "")
const ASOF_RULE_DESC = ASOF_MODE == :quarter ?
    "QUARTER rule (last report inside [quarter_start, qend]) — advisor default, 2026-08-06" :
    "WINDOW rule W = $ASOF_WINDOW_DAYS d (last report inside [max(quarter_start, qend-$ASOF_WINDOW_DAYS), qend]) — DPN_ASOF_WINDOW_DAYS override"
const GRAIN_DESC = SNAPSHOT_GRAIN == :fund ?
    "FUND grain (per (fund, quarter): keep ALL rows of the fund's last in-window report d_use) — DEFAULT, 2026-08-08" :
    "SECURITY grain (per (fund, fsym, quarter): keep the pair's own last in-window row) — legacy P0/EM behaviour"
# Fund-grain pathology diagnostic (PHASE B-DIAG below) rescans the raw chunks;
# DPN_SKIP_GRAIN_DIAG=true skips it on a re-run whose numbers are already on
# disk. Default: run it.
const SKIP_GRAIN_DIAG = parse(Bool, lowercase(get(ENV, "DPN_SKIP_GRAIN_DIAG", "false")))

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
println("Snapshot rule   : $ASOF_RULE_DESC")
println("Snapshot grain  : $GRAIN_DESC")
println("Parts tag       : asof$ASOF_TAG")

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
# Two archives are required, one per vintage this script has already shipped:
#   *_exactEOM_preP0.parquet -> the pre-P0 exact-EOM panel (2026-06-08 vintage)
#   *_preEM.parquet          -> the P0 as-of W=10 panel (2026-08-04 vintage),
#                               which the advisor-directed quarter rule now
#                               overwrites. Same archive-by-RENAME discipline.
let archive_p0  = replace(eom_path, ".parquet" => "_exactEOM_preP0.parquet"),
    archive_em  = replace(eom_path, ".parquet" => "_preEM.parquet"),
    archive_mm  = replace(eom_path, ".parquet" => "_mmv2.parquet")

    if TEST_MODE
        println("\n[PHASE B0] archive guard SKIPPED: TEST_MODE (output is $(basename(eom_path)), " *
                "the canonical artifact cannot be reached).")
    end
    missing_archives = String[]
    if !TEST_MODE && isfile(eom_path) && !SKIP_ARCHIVE_GUARD
        isfile(archive_p0) || push!(missing_archives, archive_p0)
        isfile(archive_em) || push!(missing_archives, archive_em)
        # (2026-08-08) The FUND-grain default overwrites the MM-FIX-v2
        # SECURITY-grain quarter panel: that vintage must be archived as *_mmv2
        # first (same archive-by-RENAME discipline). A security-grain rerun does
        # not require it, keeping the legacy path's behaviour unchanged.
        if SNAPSHOT_GRAIN == :fund
            isfile(archive_mm) || push!(missing_archives, archive_mm)
        end
    end
    if !isempty(missing_archives)
        error("""
        REFUSING TO OVERWRITE the existing canonical holdings panel.

          exists : $eom_path
          missing: $(join(missing_archives, "\n                   "))

        Archive-by-RENAME (same volume, cheap) is required before the rebuild so
        every shipped vintage stays reproducible:

          $(isfile(archive_p0) ? "[have]" : "[need]") $(basename(archive_p0))   pre-P0 exact-EOM panel
          $(isfile(archive_em) ? "[have]" : "[need]") $(basename(archive_em))   P0 as-of W=10 panel (overwritten by the quarter rule)
          $(SNAPSHOT_GRAIN != :fund ? "[n/a ]" : (isfile(archive_mm) ? "[have]" : "[need]")) $(basename(archive_mm))   MM-FIX-v2 security-grain quarter panel (overwritten by the FUND grain)

        Rename the panel currently on disk to the archive name that matches ITS
        OWN vintage — check the rule it was built under before you type this:

          mv "$eom_path" "$(missing_archives[end])"

        NEVER overwrite an archive that already exists — if the target is there,
        the vintage you are about to discard is NOT the one it holds; stop and
        work out which is which before doing anything else.

        If SEVERAL archives are listed as missing, only ONE of them can be
        produced by renaming this file (a panel has exactly one vintage).
        Produce that one, then set DPN_P0_SKIP_ARCHIVE_GUARD=true for the
        vintages that were never archived on this machine, and say so in
        VINTAGE_P0.md. Under DPN_SNAPSHOT_GRAIN=fund (default) the *_mmv2
        archive holds the MM-FIX-v2 SECURITY-grain quarter panel — if the panel
        on disk was built under the quarter rule at security grain, *_mmv2 is
        the name it renames to.

        Archive the downstream artifacts this chain feeds with the same suffix
        (merged_us_eu_matched.parquet, merged_us_eu_zero_filled.parquet,
        c6_panel.dta, audit_c6_panel.dta, audit_c6_panel.parquet,
        ownership_ict.parquet -> *_preEM.*) too.

        Set DPN_P0_SKIP_ARCHIVE_GUARD=true ONLY if the archive already exists
        elsewhere and you know what you are discarding.
        """)
    end
end

# ============================================================
# PHASE B: ETL — snapshot selection, per-chunk, then merge.
# Filters:
#   * NO ISSUE_TYPE filter (downstream 04/05 apply their own)
#   * ADJ_MV > 0                                     [UNCHANGED]
#   * last report inside [quarter_start, qend]       [DEFAULT, advisor rule]
#     or inside [max(quarter_start, qend-W), qend]   [numeric override]
#     (was: exact LAST_DAY, pre-P0)
# ============================================================
println("\n========== PHASE B: ETL ($ASOF_RULE_DESC) ==========")
println("Filters: ADJ_MV > 0 + snapshot selection (NO ISSUE_TYPE filter)")
println("Grain    : $GRAIN_DESC")
println("Parts dir: $ASOF_PARTS_DIR")
println("Output   : $eom_path")

# Quarter start / quarter end of a row's OWN quarter, both derived from the row's
# own REPORT_DATE. This is what makes the selection window incapable of reaching
# into the prior quarter under EITHER rule: a row is only ever a candidate for
# the quarter it already sits in, so no quarter cross join is needed and the
# per-chunk (year-split) loop stays valid. See the header note.
const QSTART_SQL = "CAST(DATE_TRUNC('quarter', CAST(REPORT_DATE AS DATE)) AS DATE)"
const QEND_SQL = "CAST(DATE_TRUNC('quarter', CAST(REPORT_DATE AS DATE)) + INTERVAL 3 MONTH - INTERVAL 1 DAY AS DATE)"

# Lower-bound predicate on the candidate set.
#   :quarter -> report_date_actual >= quarter_start. Given the own-quarter
#               assignment above this is a TAUTOLOGY, kept explicit so the
#               advisor's rule is legible in the SQL and so a future change to
#               the quarter assignment cannot silently widen the window.
#   :window  -> additionally asof_gap_days <= W, i.e. a lower bound of
#               max(quarter_start, qend - W). The max() is implicit for the same
#               reason: quarter_start already bounds the candidate set.
const ASOF_WINDOW_PREDICATE = ASOF_MODE == :quarter ?
    "AND report_date_actual >= quarter_start" :
    "AND report_date_actual >= quarter_start\n          AND asof_gap_days <= $ASOF_WINDOW_DAYS"

"""
    asof_select_sql(src_glob) -> String

DPN_SNAPSHOT_GRAIN = security (legacy): per (fund_id, fsym_id, quarter) keep the
row with the LATEST report date inside the selection window — [quarter_start,
qend] by default (advisor rule, Emanuele 2026-08-04 meeting), or
[max(quarter_start, qend - W), qend] under the numeric DPN_ASOF_WINDOW_DAYS
override. This branch is BYTE-IDENTICAL to the pre-2026-08-08 SQL.

DPN_SNAPSHOT_GRAIN = fund (DEFAULT, 2026-08-08): per (fund_id, quarter) compute
d_use = MAX(report_date_actual) over the fund's in-window ADJ_MV > 0 rows and
keep ALL rows of that single report — the fund's quarter book is one snapshot,
no cross-date stitching (diag_snapshot_reappearance.py; see the file header). A
security absent from the d_use report contributes nothing that quarter (a
genuine exit). Every kept row has report_date_actual = d_use and
asof_gap_days = qend - d_use.

Ties (same fund, same security, same date — these exist in the raw feed) are
broken deterministically UNDER BOTH GRAINS by the same ROW_NUMBER ORDER BY, so
the build is reproducible. n_asof_candidates keeps its P0 definition under both
grains: raw in-window rows sharing the (fund, fsym, quarter) key (2-3 for
monthly reporters), NOT just rows of the d_use report.
"""
function asof_select_sql(src_glob::AbstractString)
    if SNAPSHOT_GRAIN == :security
    # LEGACY SECURITY GRAIN — byte-identical to the pre-2026-08-08 SQL. Do not
    # edit this branch except in lockstep with an EXPECTED_OLD_ROWS review.
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
                $QSTART_SQL                AS quarter_start,
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
          $ASOF_WINDOW_PREDICATE
    )
    WHERE rn = 1
    """
    end
    # FUND GRAIN (DEFAULT). d_use is a fund x quarter window MAX; the outer
    # WHERE keeps only rows of the d_use report. rn is the SAME pair-level
    # tie-break as the security grain: within a (fund, fsym, quarter) partition
    # it orders report_date_actual DESC first, so rn = 1 always sits on the
    # pair's latest in-window date. Combined with report_date_actual = d_use it
    #   (a) drops entirely any pair absent from the d_use report (its rn = 1
    #       row has an earlier date and it has NO row at d_use), and
    #   (b) collapses same-day duplicate rows OF the d_use report with the
    #       existing deterministic ORDER BY (report_date_actual is constant
    #       there, so the tie-break falls through to adj_mv DESC etc. exactly
    #       as before).
    # CHUNK SAFETY of d_use: the PARTITION BY (fund_id, report_date) window is
    # computed per year-chunk. report_date is the qend of the row's OWN quarter
    # (QEND_SQL is derived from the row's REPORT_DATE), a calendar quarter never
    # spans a year, and the chunk files split on whole, disjoint year ranges
    # (EXPECTED_CHUNKS; asserted against the data by the per-part year-range
    # check after the build loop). Every (fund, quarter) partition therefore
    # lives entirely inside ONE chunk file and the within-chunk MAX equals the
    # global MAX.
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
               MAX(report_date_actual) OVER (
                   PARTITION BY fund_id, report_date
               ) AS d_use,
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
                $QSTART_SQL                AS quarter_start,
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
          $ASOF_WINDOW_PREDICATE
    )
    WHERE report_date_actual = d_use
      AND rn = 1
    """
end

part_paths = String[]
for chunk in EXPECTED_CHUNKS
    src  = replace(joinpath(RAW_PARQUET_DIR, chunk), "\\" => "/")
    stem = replace(chunk, ".parquet" => "")
    # The RULE TAG is IN the filename: a part built under a different rule (or a
    # different W) can never be reused. QTR = advisor quarter rule, W<n> = window.
    part = replace(joinpath(ASOF_PARTS_DIR, "$(stem)_asof$(ASOF_TAG).parquet"), "\\" => "/")
    part_native = replace(part, "/" => "\\")
    push!(part_paths, part)

    if isfile(part_native) && filesize(part_native) > 0 && !ASOF_FORCE_REBUILD
        # HAZARD: reuse keys ONLY on (chunk name, rule tag). It does NOT fingerprint the
        # raw input file or the SQL text, so a part built before an SQL edit, or
        # before 03a regenerated the raw cache, would be merged silently. PRE-MERGE
        # GATE M2 below is the check that makes that detectable; keep it.
        println("  [skip] $(basename(part)) already built ($(round(filesize(part_native)/1024^3, digits=2)) GB). " *
                "Reuse keys on (chunk, rule tag) ONLY — no raw/SQL fingerprint. " *
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
# split a quarter across parts and defeat the dedup. Under the FUND grain this
# check carries MORE weight: d_use = MAX(report_date_actual) is computed per
# (fund, quarter) WITHIN a chunk, and it equals the global MAX only because a
# (fund, quarter) never spans two chunk files (the declared chunk year ranges
# in EXPECTED_CHUNKS are disjoint; this loop asserts the data respects them —
# a violation would split a fund-quarter across parts and let each part pick
# its own "last" report).
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

# ============================================================
# PHASE B-DIAG (FUND GRAIN ONLY): pathology diagnostic on the RAW feed.
# The fund-grain parts keep ONLY the d_use report, so the information needed
# here — how the d_use report compares to the fund's OTHER in-window reports —
# is gone from the parts; this has to rescan the raw chunks. Per-chunk
# aggregation is exact because a (fund, quarter) never spans two chunk files
# (same argument as the d_use window; see asof_select_sql), so the pooled
# numbers are plain sums over chunks. Runs even when parts were reused.
# Reports, per chunk and pooled (03_eom_fund_grain_pathology_by_chunk.csv):
#   * PATHOLOGY: multi-date fund-quarters whose d_use report holds
#     n_secs < 0.5 x the quarter-max report size — a partial "last" report;
#     expected ~0.56% of multi-date fund-quarters
#     (diag_snapshot_reappearance.py, 2018-2023) — and their MV share;
#   * rows kept under the fund grain vs the security-grain (legacy) row count
#     (= distinct (fund, fsym, quarter) keys in the candidate set — exact, not
#     an approximation), i.e. what the grain switch drops.
# This is a DISCLOSURE, not a gate: a pathological fund-quarter still follows
# the advisor rule (its last report IS its snapshot); the count is printed so a
# blow-up vs the diagnostic's 0.56% is visible immediately.
# ============================================================
if SNAPSHOT_GRAIN == :fund && !SKIP_GRAIN_DIAG
    println("\n========== PHASE B-DIAG: fund-grain pathology (raw rescan, per chunk) ==========")
    diag_parts = DataFrame[]
    for chunk in EXPECTED_CHUNKS
        src = replace(joinpath(RAW_PARQUET_DIR, chunk), "\\" => "/")
        d = qdf(con, """
            WITH cand AS (
                SELECT FACTSET_FUND_ID           AS fund_id,
                       FSYM_ID                   AS fsym_id,
                       $QEND_SQL                 AS report_date,
                       $QSTART_SQL               AS quarter_start,
                       CAST(REPORT_DATE AS DATE) AS report_date_actual,
                       DATEDIFF('day', CAST(REPORT_DATE AS DATE), $QEND_SQL) AS asof_gap_days,
                       ADJ_MV                    AS adj_mv
                FROM read_parquet('$src')
                WHERE ADJ_MV IS NOT NULL AND ADJ_MV > 0
            ),
            cand_w AS (
                SELECT * FROM cand
                WHERE asof_gap_days >= 0
                  $ASOF_WINDOW_PREDICATE
            ),
            -- NOTE (disclosure-only): per_date sums RAW in-window rows,
            -- INCLUDING same-day duplicate (fund, fsym, date) rows that the
            -- rn = 1 tie-break collapses in the kept panel. mv_at_duse below
            -- therefore slightly overstates BOTH numerator and denominator of
            -- the MV-share print vs the kept panel. Ratio bias is second-order;
            -- labeled as raw-feed MV in the print, not kept-panel MV.
            per_date AS (
                SELECT fund_id, report_date, report_date_actual,
                       COUNT(DISTINCT fsym_id) AS n_secs,
                       SUM(adj_mv)             AS mv
                FROM cand_w GROUP BY 1, 2, 3
            ),
            per_fq AS (
                SELECT fund_id, report_date,
                       COUNT(*)                            AS n_dates,
                       MAX(n_secs)                         AS max_n_secs,
                       arg_max(n_secs, report_date_actual) AS n_secs_at_duse,
                       arg_max(mv,     report_date_actual) AS mv_at_duse
                FROM per_date GROUP BY 1, 2
            )
            SELECT COUNT(*)                                            AS n_fund_quarters,
                   COUNT(*) FILTER (WHERE n_dates > 1)                 AS n_fq_multi_date,
                   COUNT(*) FILTER (WHERE n_dates > 1
                        AND n_secs_at_duse < 0.5 * max_n_secs)         AS n_fq_pathological,
                   COALESCE(SUM(mv_at_duse), 0)                        AS mv_duse_total,
                   COALESCE(SUM(mv_at_duse) FILTER (WHERE n_dates > 1
                        AND n_secs_at_duse < 0.5 * max_n_secs), 0)     AS mv_duse_pathological,
                   COALESCE(SUM(n_secs_at_duse), 0)                    AS rows_kept_fund_grain,
                   (SELECT COUNT(*) FROM (
                        SELECT DISTINCT fund_id, fsym_id, report_date FROM cand_w
                    ))                                                 AS rows_security_grain
            FROM per_fq
        """)
        d.chunk = [chunk]
        push!(diag_parts, d)
        @printf("  %-42s fund-qtrs %10d | multi-date %9d | pathological %7d | rows: fund %11d vs security %11d\n",
                chunk, d.n_fund_quarters[1], d.n_fq_multi_date[1], d.n_fq_pathological[1],
                d.rows_kept_fund_grain[1], d.rows_security_grain[1])
    end
    diag_df = vcat(diag_parts...)
    CSV.write(joinpath(OUT_DIR, "03_eom_fund_grain_pathology_by_chunk.csv"), diag_df)
    let nfq   = sum(diag_df.n_fund_quarters),
        nmd   = sum(diag_df.n_fq_multi_date),
        npath = sum(diag_df.n_fq_pathological),
        mvtot = sum(diag_df.mv_duse_total),
        mvpat = sum(diag_df.mv_duse_pathological),
        rk    = sum(diag_df.rows_kept_fund_grain),
        rs    = sum(diag_df.rows_security_grain)
        println("\n--- FUND-GRAIN PATHOLOGY SUMMARY (pooled over chunks) ---")
        @printf("  fund-quarters: %d, of which multi-date: %d (%.2f%%)\n",
                nfq, nmd, 100 * nmd / max(nfq, 1))
        @printf("  PATHOLOGICAL (d_use report n_secs < 0.5 x quarter-max; multi-date only): %d = %.2f%% of multi-date fund-quarters\n",
                npath, 100 * npath / max(nmd, 1))
        println("    (expected ~0.56% per diag_snapshot_reappearance.py, raw 2018-2023)")
        @printf("  MV share of pathological fund-quarters (raw d_use-report MV / total raw d_use-report MV; raw rows incl. same-day dups that rn=1 collapses, so both sides slightly overstate the kept panel): %.4f%%\n",
                100 * mvpat / max(mvtot, 1e-12))
        @printf("  rows kept (fund grain): %d vs security-grain (legacy) rows: %d -> grain switch drops %d (%.2f%%)\n",
                rk, rs, rs - rk, 100 * (rs - rk) / max(rs, 1))
        println("  -> per-chunk table: 03_eom_fund_grain_pathology_by_chunk.csv")
    end
elseif SNAPSHOT_GRAIN == :fund
    println("\n[PHASE B-DIAG] SKIPPED (DPN_SKIP_GRAIN_DIAG = true).")
end

parts_glob = replace(joinpath(ASOF_PARTS_DIR, "*_asof$(ASOF_TAG).parquet"), "\\" => "/")
n_parts = length(filter(f -> endswith(f, "_asof$(ASOF_TAG).parquet"), readdir(ASOF_PARTS_DIR)))
@assert n_parts == length(EXPECTED_CHUNKS) "Expected $(length(EXPECTED_CHUNKS)) parts under rule tag asof$ASOF_TAG, found $n_parts in $ASOF_PARTS_DIR"

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
# WHY THIS GATE EXISTS. The part-reuse skip above keys only on (chunk name,
# rule tag), with no fingerprint of the raw input or of the SQL text. A re-run after any
# SQL edit, or after 03a regenerates the raw cache, would silently merge STALE
# parts — and every other gate in this script would still pass. This is the only
# check that makes such a regression detectable.
#
# FUND-GRAIN NOTE (2026-08-08). EXPECTED_OLD_ROWS is GRAIN-INVARIANT, so this
# gate is kept UNCHANGED under DPN_SNAPSHOT_GRAIN=fund. Why it still holds: a
# gap = 0 row can only come from a fund with an ADJ_MV > 0 report ON the
# calendar quarter-end; qend is the maximum attainable in-window date, so for
# such a fund d_use = qend and the fund keeps its ENTIRE quarter-end (exact-EOM)
# report. The kept gap = 0 rows are therefore exactly the distinct
# (fund, fsym, qend) keys carrying an exact quarter-end raw row — the same set
# the SECURITY grain keeps at gap = 0 (a pair with a qend row has that row as
# its own latest), and the same-day dedup is the identical ROW_NUMBER tie-break
# in both branches. Do NOT fork this constant by grain.
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
                   $ASOF_PARTS_DIR are keyed only on (chunk, rule tag); they do not
                   fingerprint the raw input or the SQL. Delete them, or set
                   DPN_ASOF_FORCE_REBUILD=true, and re-run.

        NOTHING was written to $eom_path — the pre-P0 chain state is unchanged.
        """)
    end
    println("  [OK] M2 strict-superset: every old-rule row recovered one-for-one.")
end

# ---- M3: every kept row must sit INSIDE the quarter it is stamped to ---------
# This is the gate on the advisor rule's one hard boundary: the selection window
# must NEVER cross into the prior quarter. Under the own-quarter assignment it
# cannot, so this check is expected to be trivially satisfied — which is exactly
# why it is worth asserting: if the quarter assignment is ever changed to a
# calendar join, or a fixed 92-day window is ever substituted, this is the check
# that fires instead of the panel silently double-counting a quarter.
#   report_date        = stamped quarter-end
#   report_date_actual = true report date
#   valid iff  DATE_TRUNC('quarter', report_date) <= report_date_actual <= report_date
# Equivalently 0 <= asof_gap_days <= (days in the stamped quarter) - 1, i.e. <= 91.
let m3 = qdf(con, """
        SELECT COUNT(*) AS n_out_of_quarter,
               COALESCE(MIN(asof_gap_days), 0) AS min_gap,
               COALESCE(MAX(asof_gap_days), 0) AS max_gap
        FROM read_parquet('$parts_glob')
        WHERE report_date_actual > report_date
           OR report_date_actual < CAST(DATE_TRUNC('quarter', report_date) AS DATE)
    """)
    rng = qdf(con, "SELECT MIN(asof_gap_days) AS lo, MAX(asof_gap_days) AS hi FROM read_parquet('$parts_glob')")
    println("  M3 rows outside their stamped quarter: $(m3.n_out_of_quarter[1]) " *
            "(observed gap range: $(rng.lo[1]) .. $(rng.hi[1]) days)")
    if m3.n_out_of_quarter[1] > 0
        error("""
        PRE-MERGE GATE M3 FAILED: $(m3.n_out_of_quarter[1]) rows have a
        report_date_actual outside [quarter_start, quarter_end] of the quarter
        they were stamped to. The selection window has crossed a quarter
        boundary, so a single raw report is serving two quarters and the panel
        would double-count. NOTHING was written to $eom_path.
        """)
    end
    if ASOF_MODE == :window && rng.hi[1] > ASOF_WINDOW_DAYS
        error("""
        PRE-MERGE GATE M3 FAILED: max asof_gap_days = $(rng.hi[1]) exceeds the
        requested window W = $ASOF_WINDOW_DAYS. The parts were almost certainly
        built under a DIFFERENT rule and reused. Delete $ASOF_PARTS_DIR or set
        DPN_ASOF_FORCE_REBUILD=true. NOTHING was written to $eom_path.
        """)
    end
    println("  [OK] M3 every row sits inside its own stamped quarter.")
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

# ---------------------------------------------------------------------------
# C2b: STALENESS — the gap distribution the quarter rule buys.
#
# The advisor rule expands the universe by admitting intra-quarter reporters, so
# the panel is MORE stale by construction. This block is the honest price tag and
# must be reported alongside any coverage or N gain:
#     share of rows at gap = 0   (a true quarter-end snapshot)
#     median gap                 (row-weighted, from the histogram)
#     share of rows at gap > 14d
#     max gap                    (<= 91 by construction; M3 asserts it)
# Reported pooled, per quarter, and per era, row-weighted and MV-weighted.
# Median convention: the SMALLEST gap whose cumulative row share reaches 0.5
# (lower median on a discrete histogram). No interpolation.
# ---------------------------------------------------------------------------
gap_dist = qdf(con, """
    SELECT asof_gap_days, COUNT(*) AS n_rows,
           COUNT(DISTINCT fund_id) AS n_funds,
           SUM(adj_mv)/1e9 AS mv_b
    FROM eom GROUP BY 1 ORDER BY 1
""")
println("\nAs-of gap distribution (0 = exact quarter-end snapshot):")
println(first(gap_dist, 25))
nrow(gap_dist) > 25 && println("  ... $(nrow(gap_dist) - 25) further gap values (see the CSV)")
CSV.write(joinpath(OUT_DIR, "03_eom_asof_gap_distribution.csv"), gap_dist)

"""
    gap_summary(gaps, w) -> NamedTuple

Four staleness statistics from a (gap value, weight) histogram, where w is a row
count or an MV total. Lower-median convention (see C2b header). Returns NaN/0 on
an empty histogram rather than throwing, so a quarter with no rows cannot kill
the diagnostic pass of a 20-minute build.
"""
function gap_summary(gaps::AbstractVector, w::AbstractVector)
    tot = sum(w)
    # -1 is an impossible gap, so it reads as "undefined" and cannot be mistaken
    # for a same-day snapshot in the CSV.
    tot <= 0 && return (share_gap0 = NaN, median_gap = -1, share_gap_gt14 = NaN,
                        max_gap = -1, mean_gap = NaN, total = tot)
    ord   = sortperm(collect(gaps))
    g     = collect(gaps)[ord]
    ww    = collect(w)[ord]
    cum   = cumsum(ww) ./ tot
    med_i = findfirst(>=(0.5), cum)
    return (share_gap0     = sum(ww[g .== 0]) / tot,
            median_gap     = med_i === nothing ? g[end] : g[med_i],
            share_gap_gt14 = sum(ww[g .> 14]) / tot,
            max_gap        = maximum(g),
            mean_gap       = sum(g .* ww) / tot,
            total          = tot)
end

let s_rows = gap_summary(gap_dist.asof_gap_days, gap_dist.n_rows),
    s_mv   = gap_summary(gap_dist.asof_gap_days, gap_dist.mv_b)
    println("\n--- STALENESS SUMMARY (pooled, $ASOF_RULE_DESC) ---")
    @printf("  row-weighted : share(gap=0) = %6.2f%%   median = %3d d   share(gap>14d) = %6.2f%%   max = %3d d   mean = %5.2f d\n",
            100*s_rows.share_gap0, s_rows.median_gap, 100*s_rows.share_gap_gt14,
            s_rows.max_gap, s_rows.mean_gap)
    @printf("  MV-weighted  : share(gap=0) = %6.2f%%   median = %3d d   share(gap>14d) = %6.2f%%   max = %3d d   mean = %5.2f d\n",
            100*s_mv.share_gap0, s_mv.median_gap, 100*s_mv.share_gap_gt14,
            s_mv.max_gap, s_mv.mean_gap)
end

# Per-quarter and per-era version, derived from a (quarter, gap) histogram so the
# medians are exact without materialising 300M+ rows for a MEDIAN() aggregate.
gap_hist_q = qdf(con, """
    SELECT report_date AS qend, asof_gap_days,
           COUNT(*) AS n_rows, SUM(adj_mv)/1e9 AS mv_b
    FROM eom GROUP BY 1,2 ORDER BY 1,2
""")
gap_by_q = combine(groupby(gap_hist_q, :qend)) do sdf
    r = gap_summary(sdf.asof_gap_days, sdf.n_rows)
    m = gap_summary(sdf.asof_gap_days, sdf.mv_b)
    (n_rows          = r.total,
     share_gap0      = r.share_gap0,
     median_gap      = r.median_gap,
     share_gap_gt14  = r.share_gap_gt14,
     max_gap         = r.max_gap,
     mean_gap        = r.mean_gap,
     mv_share_gap0     = m.share_gap0,
     mv_median_gap     = m.median_gap,
     mv_share_gap_gt14 = m.share_gap_gt14)
end
CSV.write(joinpath(OUT_DIR, "03_eom_asof_gap_summary.csv"), gap_by_q)
println("\nPer-quarter staleness -> 03_eom_asof_gap_summary.csv " *
        "($(nrow(gap_by_q)) quarters). Last 8:")
println(last(gap_by_q[:, [:qend, :n_rows, :share_gap0, :median_gap, :share_gap_gt14, :max_gap]], 8))

gap_hist_q.era = map(q -> year(q) <= 2005 ? "1999-2005 ramp-up" :
                          year(q) <= 2015 ? "2006-2015 mid" : "2016-2023 modern",
                     gap_hist_q.qend)
gap_by_era = combine(groupby(gap_hist_q, :era)) do sdf
    h = combine(groupby(sdf, :asof_gap_days), :n_rows => sum => :n_rows)
    r = gap_summary(h.asof_gap_days, h.n_rows)
    (n_rows = r.total, share_gap0 = r.share_gap0, median_gap = r.median_gap,
     share_gap_gt14 = r.share_gap_gt14, max_gap = r.max_gap, mean_gap = r.mean_gap)
end
sort!(gap_by_era, :era)
CSV.write(joinpath(OUT_DIR, "03_eom_asof_gap_summary_by_era.csv"), gap_by_era)
println("\nStaleness by era (the 1999-2005 ramp-up is where the quarter rule adds the most):")
println(gap_by_era)

# ---------------------------------------------------------------------------
# C2d: FUND-PORTFOLIO SNAPSHOT DISPERSION — the number the advisor needs to see
# next to the coverage gain. (EM-FIX-1, 2026-08-06.)
#
# C2b measures staleness ROW BY ROW. It cannot see the problem that actually
# bites downstream: under the quarter rule a single fund's quarter book is
# stitched together from several report dates (see the header note), so the
# cross-security SUM(adj_mv) that 04 turns into I_ict and into
# country_total_holdings_eu — the denominator of the headline outcome — mixes
# valuation dates and can include positions already exited by quarter-end.
#
# Per (fund_id, report_date = stamped quarter-end) this measures:
#     n_distinct_actual_dates  = COUNT(DISTINCT report_date_actual)
#     gap_spread_days          = MAX(asof_gap_days) - MIN(asof_gap_days)
#     mv_share_from_stale      = share of the fund-quarter's adj_mv carried by
#                                rows with asof_gap_days > 0
# and reports, pooled and per quarter:
#     share of fund-quarters spanning MORE THAN ONE report date
#     MV-weighted mean gap spread in days
# Under W = 10 this was bounded at 10 days and ~0.26% of fund-quarters; under the
# quarter rule it is expected to be first order. It is a DISCLOSURE, not a gate.
# FUND-GRAIN NOTE (2026-08-08): under DPN_SNAPSHOT_GRAIN=fund every kept row of
# a fund-quarter carries report_date_actual = d_use, so n_distinct_actual_dates
# MUST equal 1 everywhere and this block flips from disclosure to HARD
# INVARIANT (asserted after the pooled table below, with the fail_canonical
# demotion discipline — a violation means the grain SQL or the merge is
# broken). Under the SECURITY grain it remains a disclosure.
# ---------------------------------------------------------------------------
DBInterface.execute(con, """
    CREATE OR REPLACE TEMP TABLE fund_snap_disp AS
    SELECT fund_id,
           report_date AS qend,
           COUNT(DISTINCT report_date_actual)                       AS n_distinct_actual_dates,
           MAX(asof_gap_days) - MIN(asof_gap_days)                  AS gap_spread_days,
           MIN(asof_gap_days)                                       AS min_gap,
           MAX(asof_gap_days)                                       AS max_gap,
           SUM(adj_mv)                                              AS fq_mv,
           SUM(CASE WHEN asof_gap_days > 0 THEN adj_mv ELSE 0 END)  AS fq_mv_stale,
           COUNT(*)                                                 AS n_positions
    FROM eom
    GROUP BY 1, 2
""")

fund_disp_q = qdf(con, """
    SELECT qend,
           COUNT(*)                                                        AS n_fund_quarters,
           COUNT(*) FILTER (WHERE n_distinct_actual_dates > 1)             AS n_fq_multi_date,
           AVG(CASE WHEN n_distinct_actual_dates > 1 THEN 1.0 ELSE 0.0 END) AS share_fq_multi_date,
           AVG(n_distinct_actual_dates)                                    AS mean_n_dates,
           MAX(n_distinct_actual_dates)                                    AS max_n_dates,
           AVG(gap_spread_days)                                            AS mean_gap_spread_days,
           MAX(gap_spread_days)                                            AS max_gap_spread_days,
           SUM(gap_spread_days * fq_mv) / NULLIF(SUM(fq_mv), 0)            AS mv_wtd_mean_gap_spread_days,
           SUM(fq_mv_stale) / NULLIF(SUM(fq_mv), 0)                        AS mv_share_from_gap_gt0
    FROM fund_snap_disp
    GROUP BY 1 ORDER BY 1
""")
CSV.write(joinpath(OUT_DIR, "03_eom_fund_snapshot_dispersion.csv"), fund_disp_q)

fund_disp_hist = qdf(con, """
    SELECT n_distinct_actual_dates, COUNT(*) AS n_fund_quarters,
           SUM(fq_mv)/1e9 AS mv_b
    FROM fund_snap_disp GROUP BY 1 ORDER BY 1
""")
CSV.write(joinpath(OUT_DIR, "03_eom_fund_snapshot_dispersion_hist.csv"), fund_disp_hist)

fund_disp_pooled = qdf(con, """
    SELECT COUNT(*)                                                        AS n_fund_quarters,
           COUNT(*) FILTER (WHERE n_distinct_actual_dates > 1)             AS n_fq_multi_date,
           AVG(CASE WHEN n_distinct_actual_dates > 1 THEN 1.0 ELSE 0.0 END) AS share_fq_multi_date,
           AVG(gap_spread_days)                                            AS mean_gap_spread_days,
           MAX(gap_spread_days)                                            AS max_gap_spread_days,
           SUM(gap_spread_days * fq_mv) / NULLIF(SUM(fq_mv), 0)            AS mv_wtd_mean_gap_spread_days,
           SUM(fq_mv_stale) / NULLIF(SUM(fq_mv), 0)                        AS mv_share_from_gap_gt0
    FROM fund_snap_disp
""")
println("\n--- C2d FUND-PORTFOLIO SNAPSHOT DISPERSION (pooled, $ASOF_RULE_DESC, $GRAIN_DESC) ---")
println("  (a fund-quarter spanning >1 report date has its cross-security SUM(adj_mv)")
println("   stitched from several valuation dates — this is what 04's I_ict inherits)")
println(fund_disp_pooled)
println("\n  n_distinct_actual_dates histogram (fund-quarters):")
println(first(fund_disp_hist, 15))
println("\n  Per-quarter -> 03_eom_fund_snapshot_dispersion.csv ($(nrow(fund_disp_q)) quarters). Last 8:")
println(last(fund_disp_q[:, [:qend, :n_fund_quarters, :share_fq_multi_date,
                             :mv_wtd_mean_gap_spread_days, :mv_share_from_gap_gt0]], 8))
# FUND-GRAIN HARD INVARIANT (2026-08-08; see the C2d header note): every
# fund-quarter must be a single d_use snapshot. Post-merge check, so it must
# demote the artifact on failure like C1/C1b.
if SNAPSHOT_GRAIN == :fund && fund_disp_pooled.n_fq_multi_date[1] > 0
    fail_canonical("FUND-GRAIN INVARIANT FAILED (C2d): " *
                   "$(fund_disp_pooled.n_fq_multi_date[1]) fund-quarters span more than one " *
                   "report_date_actual in a panel built under DPN_SNAPSHOT_GRAIN=fund. " *
                   "Every fund-quarter must be exactly one d_use report; the fund-grain SQL, " *
                   "a stale part, or a mixed-grain merge is broken.")
end
SNAPSHOT_GRAIN == :fund &&
    println("  [OK] fund-grain invariant: every fund-quarter is a single d_use snapshot.")
DBInterface.execute(con, "DROP TABLE fund_snap_disp")

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
println("Output: $eom_path")
println("Rule  : $ASOF_RULE_DESC")
println("Grain : $GRAIN_DESC")
println("\nKey diagnostic files:")
println("  03_eom_asof_coverage_old_vs_new.csv  (GATE 1 + GATE 2, per quarter)")
println("  03_eom_asof_gap_distribution.csv     (full gap histogram)")
println("  03_eom_asof_gap_summary.csv          (per-quarter staleness: share gap=0, median, share>14d, max)")
println("  03_eom_asof_gap_summary_by_era.csv   (same, by era)")
println("  03_eom_dup_key_count.csv             (hard assert, must be 0)")
SNAPSHOT_GRAIN == :fund &&
    println("  03_eom_fund_grain_pathology_by_chunk.csv (fund grain: partial-d_use pathology + rows vs security grain)")
println("  03_eom_weekend_audit.csv             (actual report dates behind each quarter)")
println("  03_eom_coverage_by_year.csv")
println("  03_eom_issue_type_breakdown.csv")
println("  03_us_x_eu_cells_by_month.csv        (NEW vs OLD-rule \$ totals)")
println("\nParts kept at: $ASOF_PARTS_DIR (delete once the merge is verified)")
println("\nNext: 04_us_ownership_european.jl")
