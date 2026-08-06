# 04_us_ownership_european.jl
# Sections 2-4 descriptive analysis using the EOM panel from script 03.
# Builds:
#   - I_{i,c,t}                   (country-firm aggregation)
#   - w_{i,c,t}    [portfolio_weight_eu]  (EU-restricted; this is the REGRESSION input)
#   - w_{i,c,t}    [portfolio_weight_global]  (global; kept for diagnostic only)
#   - Ownership_{i,US,t}          (US ownership share of European firms)
# And computes distributions, percentiles, top firms.
#
# Audited 2026-06-01 — see AUDIT_2026_06_01_julia_descriptive.md.
# Critical-fix changes vs prior version:
#
# C1 (denominator): The previous portfolio_weight used a GLOBAL denominator —
#     sum of holder-country I across ALL sec_country, not just EU. US weights
#     were deflated ~6-7x relative to the regression spec. Fix: build a parallel
#     country_total_holdings_eu column (denominator restricted to EU sec_country)
#     and emit portfolio_weight_eu alongside portfolio_weight_global. The
#     EU-restricted one is what 05 must use; the global one is kept for the
#     fig2 "US-EU engagement" descriptive only.
#
# Market-cap MAX×MAX (high): The previous MAX(adj_shares_out) × MAX(adj_price)
#     silently selected the upper bound of intra-group dispersion (same firm-MAX
#     leak pattern as the sibling robots project). Fix: use AVG within
#     (sec_entity_id, fsym_id, report_date) — if rows are identical (the usual
#     case), AVG = MIN = MAX so the result is unambiguous. STDDEV per group is
#     captured as a dispersion diagnostic; downstream can flag (sec, fsym, t)
#     groups where stddev > 0.
#
# C6 (selection-on-outcome universe — PARTIAL): The EU firm universe is still
#     derived from FactSet ownership ("firms ever held by ≥1 institution"),
#     not from Factset_Security_coverage. This is selection-on-outcome and
#     remains a STRUCTURAL TODO requiring a fresh build from Factset_Security_coverage
#     + Cartesian (firm × country × quarter) grid + zero-fill. Marked at top
#     of file; do NOT treat the current panel as the regression panel until
#     this is rebuilt. See AUDIT file for the full fix recipe.
#
# Also applied:
#   - ownership_share > 1 cells: diagnostic emitted BEFORE silent drop.
#   - Atomic writes via atomic_copy_to.

include("00_setup.jl")

const EOM_PATH = replace(joinpath(OUT_DIR, test_suffix_path("holdings_eom.parquet")), "\\" => "/")
if !isfile(replace(EOM_PATH, "/" => "\\"))
    error("$EOM_PATH not found. Run 03_eom_etl.jl first.")
end

# Step 04 reads only one external artifact — the EOM panel from 03.
const STEP_INPUTS = [replace(EOM_PATH, "/" => "\\")]

# (kept legacy name for any in-file references; canonical list is in 00_setup.jl)
EU_str = EU_SQL_TUPLE

con = dbcon()

# Runtime banner mirroring 05's end-of-run notice. Printed at script start so
# anyone reading stdout (not just the source comments) sees the PROVISIONAL
# state of the EU firm universe.
println("\n" * "!"^70)
println("!! 04_us_ownership_european.jl — DESCRIPTIVE PIPELINE                !!")
println("!! EU firm universe is currently SELECTION-ON-OUTCOME (C6 deferred). !!")
println("!! Outputs are DESCRIPTIVE ONLY; downstream regression panel must    !!")
println("!! be rebuilt from Factset_Security_coverage + Cartesian grid +      !!")
println("!! zero-fill. See AUDIT_2026_06_01_julia_descriptive.md.             !!")
println("!"^70 * "\n")

# ============================================================
# STRUCTURAL TODO (C6) — flagged at the top so a reviewer cannot miss it.
# The EU firm universe used here is derived from "appears in FactSet ownership
# panel with EU sec_country" — i.e., firms ever held by at least one investor.
# That is selection-on-outcome for a regression whose dependent variable is
# institutional holdings. Extensive-margin exits (firms US investors fully
# exit) silently vanish from the panel. Fix requires:
#   1. Pull Factset_Security_coverage to enumerate ALL EU-listed equities.
#   2. Build a (firm × holder-country × quarter) Cartesian grid.
#   3. LEFT-JOIN ownership_ict onto the grid; zero-fill (i,c,t) cells where
#      the institution did not hold (legitimate zeros, not missingness).
#   4. Drop gap_months=6 hard filter in 05; require only that t-1 and t+1
#      exist on the grid (zeros allowed).
# Until done, the merged panel is descriptive-only; do NOT use as regression
# panel.
# ============================================================

# WARNING: if 03_eom_etl.jl was run with TEST_MODE=true, the EOM panel covers
# only a subset. Time series plots will reflect only those months.
year_range = qdf(con, """
    SELECT MIN(EXTRACT(YEAR FROM report_date)) AS ymin,
           MAX(EXTRACT(YEAR FROM report_date)) AS ymax,
           COUNT(DISTINCT EXTRACT(YEAR FROM report_date)) AS n_years
    FROM read_parquet('$EOM_PATH')
""")
if year_range.n_years[1] < 5
    println("\n" * "!"^70)
    println("WARNING: EOM panel covers only $(year_range.ymin[1])-$(year_range.ymax[1]) ($(year_range.n_years[1]) year(s)).")
    println("This looks like TEST_MODE output from 03_eom_etl.jl.")
    println("Trend plots will NOT represent the full 1999-2023 sample.")
    println("!"^70 * "\n")
end

# ============================================================
# (0) Pre-aggregation duplicate-key audit on holdings panel.
# If the EOM panel has multiple rows per (fund_id, fsym_id, report_date), a
# downstream SUM(adj_mv) silently double-counts dollars. We hard-fail on ANY
# duplicate (measured 0 on the current 186,800,295-row holdings_eom.parquet).
# ============================================================
println("Auditing holdings panel for duplicate keys...")
dup_check = qdf(con, """
    SELECT COUNT(*) AS n_dup_keys
    FROM (
        SELECT fund_id, fsym_id, report_date, COUNT(*) AS c
        FROM read_parquet('$EOM_PATH')
        GROUP BY 1,2,3
        HAVING COUNT(*) > 1
    )
""")
n_dup = dup_check.n_dup_keys[1]
println("  duplicate (fund_id, fsym_id, report_date) keys: $n_dup")
if n_dup > 0
    error("Holdings panel has $n_dup duplicate (fund_id, fsym_id, report_date) keys — " *
          "downstream SUM(adj_mv) would double-count dollars and distort I_ict / portfolio weights. " *
          "Fix the 03_eom_etl.jl dedup rule before proceeding. " *
          "(Neither the build_c6_panel dedup assert nor the weight-sum check catches raw fund-level double-counting.)")
end

# ============================================================
# (1) Build I_{i,c,t} — investor-country × firm × quarter aggregation
# ============================================================
# ISSUE_TYPE filter applied here (NOT in 03):
#   EQ = common equity of operating companies (the main object of analysis)
#   AD = ADR/GDR/depositary receipts (US investors' main path to hold foreign
#        firms — must include or US holdings of EU firms gets under-counted)
# Excluded: OE/ET/CE/UIT (funds), PF/CP (preferred), WT (warrants),
#           AI/EP/DR/BC (alt/private). See 03_eom_issue_type_breakdown.csv.
#
# Atomically write.
# ============================================================
println("Building I_{i,c,t} country-firm aggregation (ISSUE_TYPE in EQ, AD)...")

ict_path = test_suffix_path(joinpath(OUT_DIR, "I_ict_panel.parquet"))

@time atomic_copy_to(con, """
    SELECT
        sec_entity_id,
        sec_country,
        investor_country,
        report_date,
        SUM(adj_mv) AS I_ict
    FROM read_parquet('$EOM_PATH')
    WHERE sec_entity_id IS NOT NULL
      AND investor_country IS NOT NULL
      AND issue_type IN ('EQ', 'AD')
    GROUP BY sec_entity_id, sec_country, investor_country, report_date
""", ict_path)
ict_path_fwd = replace(ict_path, "\\" => "/")

n_ict = qdf(con, "SELECT COUNT(*) AS n FROM read_parquet('$ict_path_fwd')").n[1]
println("  rows in I_ict panel: $n_ict")
write_manifest("04_I_ict_panel", ict_path; row_count=n_ict, input_paths=STEP_INPUTS)

# ============================================================
# (2) MarketCap_{i,t} — built from PRIMARY EQUITY only, at the FRESHEST
#     available valuation date inside the quarter.
#
# Fix (high, 2026-06-01): the previous version used MAX × MAX which silently
# selected the upper bound of intra-group dispersion (firm-MAX leak pattern).
# Replaced by AVG within group.
#
# EM-FIX-2 (2026-08-06) — POINT-IN-TIME RESTRICTION. THIS IS AN EXPLICIT
# DECISION, NOT A SILENT INHERITANCE. Under the W = 10 as-of window every row
# contributing to a (sec_entity_id, sec_country, report_date, fsym_id) group was
# within 10 days of quarter-end, so AVG = MIN = MAX and the 2026-06-01
# justification ("if all rows are identical, AVG is exact") held. Under the
# advisor-directed QUARTER rule (03_eom_etl.jl, Emanuele 2026-08-04 24:17) the
# same group can contain a JANUARY price and a MARCH price for the same security,
# reported by two different funds. AVG would then be an unweighted average of
# prices up to ~91 days apart, and market_cap feeds:
#     ownership_share = I_ict / market_cap                    (line ~253 below)
#     the Figure-A scatter us_ownership_share                 (06_cartesian_grid.jl:516)
#
# RULE APPLIED: within each (sec_entity_id, sec_country, report_date, fsym_id)
# group, keep only the rows at MIN(asof_gap_days) — the freshest observation of
# that security in that quarter — then AVG within the tie (unchanged convention,
# so genuine same-date disagreement between funds is still averaged, not
# max-picked). The STDDEV dispersion diagnostic is computed on the RESTRICTED
# set, and the would-be UNRESTRICTED dispersion is printed beside it so the size
# of what the restriction removes is on the record rather than assumed away.
#
# NOTE ON SCOPE: the parent instruction forbade changing the WEIGHT construction
# (w is computed on the full European book, 04 section 3a/3b untouched). It did
# NOT cover the market-cap builder. This change touches ONLY market_cap.
# ============================================================
# Guard: the restriction needs the P0/EM provenance column. A pre-P0 panel has
# no asof_gap_days, and silently falling back to the unrestricted AVG is exactly
# the failure mode this fix exists to prevent.
let eom_cols = Set(qdf(con, "DESCRIBE SELECT * FROM read_parquet('$EOM_PATH')").column_name)
    "asof_gap_days" in eom_cols || error(
        "EM-FIX-2: holdings_eom.parquet has no 'asof_gap_days' column, so the " *
        "point-in-time market-cap restriction cannot be applied. This panel was " *
        "built by a pre-P0 03_eom_etl.jl. Re-run 03 before 04.")
end

mcap_path = test_suffix_path(joinpath(OUT_DIR, "marketcap_it.parquet"))

@time atomic_copy_to(con, """
    WITH src AS (
        SELECT sec_entity_id, sec_country, report_date, fsym_id,
               adj_shares_out, adj_price, asof_gap_days,
               MIN(asof_gap_days) OVER (
                   PARTITION BY sec_entity_id, sec_country, report_date, fsym_id
               ) AS min_gap_in_group
        FROM read_parquet('$EOM_PATH')
        WHERE fsym_id = fsym_primary_id
          AND issue_type = 'EQ'
          AND adj_shares_out IS NOT NULL AND adj_shares_out > 0
          AND adj_price IS NOT NULL AND adj_price > 0
    ),
    primary_only AS (
        SELECT sec_entity_id, sec_country, report_date, fsym_id,
               AVG(adj_shares_out) AS shares_out,
               AVG(adj_price)      AS price,
               STDDEV_SAMP(adj_shares_out) AS shares_out_stddev,
               STDDEV_SAMP(adj_price)      AS price_stddev,
               COUNT(*) AS n_rows_in_group,
               MIN(asof_gap_days) AS asof_gap_days_used
        FROM src
        WHERE asof_gap_days = min_gap_in_group      -- EM-FIX-2: freshest only
        GROUP BY sec_entity_id, sec_country, report_date, fsym_id
    )
    SELECT sec_entity_id, sec_country, report_date,
           SUM(shares_out * price) AS market_cap,
           COUNT(*) AS n_primary_classes,
           MAX(shares_out_stddev) AS max_class_shares_stddev,
           MAX(price_stddev)      AS max_class_price_stddev,
           -- EM-FIX-2 provenance: the valuation date actually used, in days
           -- before the stamped quarter-end. 0 = a true quarter-end price.
           MAX(asof_gap_days_used) AS max_asof_gap_days_used,
           MIN(asof_gap_days_used) AS min_asof_gap_days_used
    FROM primary_only
    GROUP BY sec_entity_id, sec_country, report_date
""", mcap_path)
mcap_path_fwd = replace(mcap_path, "\\" => "/")

# EM-FIX-2 disclosure: how much cross-date mixing the restriction removed, and
# how stale the surviving valuation dates are. Printed, and written to CSV so it
# can be carried as a caveat wherever market_cap-scaled outcomes are reported.
mcap_pit_diag = qdf(con, """
    WITH src AS (
        SELECT sec_entity_id, sec_country, report_date, fsym_id, asof_gap_days,
               MIN(asof_gap_days) OVER (
                   PARTITION BY sec_entity_id, sec_country, report_date, fsym_id
               ) AS min_gap_in_group
        FROM read_parquet('$EOM_PATH')
        WHERE fsym_id = fsym_primary_id
          AND issue_type = 'EQ'
          AND adj_shares_out IS NOT NULL AND adj_shares_out > 0
          AND adj_price IS NOT NULL AND adj_price > 0
    ),
    grp AS (
        SELECT sec_entity_id, sec_country, report_date, fsym_id,
               COUNT(*) AS n_rows_all,
               COUNT(*) FILTER (WHERE asof_gap_days = min_gap_in_group) AS n_rows_kept,
               COUNT(DISTINCT asof_gap_days) AS n_distinct_gaps,
               MAX(asof_gap_days) - MIN(asof_gap_days) AS gap_spread_days
        FROM src GROUP BY 1,2,3,4
    )
    SELECT COUNT(*) AS n_groups,
           COUNT(*) FILTER (WHERE n_distinct_gaps > 1) AS n_groups_multi_date,
           AVG(CASE WHEN n_distinct_gaps > 1 THEN 1.0 ELSE 0.0 END) AS share_groups_multi_date,
           AVG(gap_spread_days) AS mean_gap_spread_days,
           MAX(gap_spread_days) AS max_gap_spread_days,
           SUM(n_rows_all - n_rows_kept) AS n_rows_dropped_by_pit,
           SUM(n_rows_all) AS n_rows_all
    FROM grp
""")
println("\n  EM-FIX-2 point-in-time market-cap restriction (pre-restriction dispersion):")
println(mcap_pit_diag)
CSV.write(joinpath(OUT_DIR, "04_marketcap_pit_restriction_diag.csv"), mcap_pit_diag)

mcap_gap_used = qdf(con, """
    SELECT max_asof_gap_days_used AS asof_gap_days_used,
           COUNT(*) AS n_company_quarters
    FROM read_parquet('$mcap_path_fwd')
    GROUP BY 1 ORDER BY 1
""")
println("\n  Valuation-date staleness of the surviving market_cap (0 = true quarter-end price):")
println(first(mcap_gap_used, 15))
CSV.write(joinpath(OUT_DIR, "04_marketcap_gap_used_distribution.csv"), mcap_gap_used)

# Diagnostic: how many companies got a clean mcap? Any dispersion?
mcap_diag = qdf(con, """
    SELECT COUNT(*) AS n_company_quarters,
           COUNT(DISTINCT sec_entity_id) AS n_companies,
           AVG(n_primary_classes) AS avg_primary_classes,
           SUM(CASE WHEN max_class_shares_stddev > 0 THEN 1 ELSE 0 END) AS n_shares_dispersion,
           SUM(CASE WHEN max_class_price_stddev > 0 THEN 1 ELSE 0 END)  AS n_price_dispersion
    FROM read_parquet('$mcap_path_fwd')
""")
println("  mcap coverage + dispersion diagnostic:")
println(mcap_diag)
if mcap_diag.n_shares_dispersion[1] > 0 || mcap_diag.n_price_dispersion[1] > 0
    println("  -> $(mcap_diag.n_shares_dispersion[1]) groups had within-group shares dispersion;")
    println("     $(mcap_diag.n_price_dispersion[1]) groups had within-group price dispersion.")
    println("     AVG was used. Spot-check via marketcap_it.parquet if these counts are non-trivial.")
end
write_manifest("04_marketcap_it", mcap_path; row_count=mcap_diag.n_company_quarters[1], input_paths=STEP_INPUTS)

# ============================================================
# (3a) Country totals — TWO versions.
#
# country_total_holdings_GLOBAL  = sum_j I_{j,c,t} across ALL sec_country
# country_total_holdings_EU      = sum_{j in EU} I_{j,c,t}  (regression input)
#
# C1 FIX: the regression spec requires the EU-restricted denominator so US and
# NONUS series use the same scope. The global version is preserved for the
# fig2 descriptive ONLY.
# ============================================================
country_total_path = test_suffix_path(joinpath(OUT_DIR, "country_total_ct.parquet"))

@time atomic_copy_to(con, """
    SELECT
        investor_country,
        report_date,
        SUM(I_ict) AS country_total_holdings_global,
        SUM(CASE WHEN sec_country IN $EU_str THEN I_ict ELSE 0 END) AS country_total_holdings_eu
    FROM read_parquet('$ict_path_fwd')
    GROUP BY investor_country, report_date
""", country_total_path)
country_total_path_fwd = replace(country_total_path, "\\" => "/")
write_manifest("04_country_total_ct", country_total_path; input_paths=STEP_INPUTS)

# ============================================================
# (3b) Ownership_{i,c,t} AND portfolio weight w_{i,c,t}
# Emit BOTH portfolio_weight_eu (regression input) and portfolio_weight_global
# (diagnostic). Downstream / 05 must use portfolio_weight_eu.
# ============================================================
println("\nBuilding Ownership_{i,c,t} and portfolio weight w_{i,c,t} (EU + global)...")

own_path = test_suffix_path(joinpath(OUT_DIR, "ownership_ict.parquet"))

@time atomic_copy_to(con, """
    SELECT i.sec_entity_id,
           i.sec_country,
           i.investor_country,
           i.report_date,
           i.I_ict,
           m.market_cap,
           i.I_ict / NULLIF(m.market_cap, 0) AS ownership_share,
           -- C1-FIX regression input: denominator restricted to EU sec_country
           CASE WHEN i.sec_country IN $EU_str
                THEN i.I_ict / NULLIF(ct.country_total_holdings_eu, 0)
                ELSE NULL END AS portfolio_weight_eu,
           -- Diagnostic only: global denominator (do NOT use in regression)
           i.I_ict / NULLIF(ct.country_total_holdings_global, 0) AS portfolio_weight_global
    FROM read_parquet('$ict_path_fwd') i
    LEFT JOIN read_parquet('$mcap_path_fwd') m
        ON i.sec_entity_id = m.sec_entity_id
       AND i.report_date = m.report_date
    LEFT JOIN read_parquet('$country_total_path_fwd') ct
        ON i.investor_country = ct.investor_country
       AND i.report_date      = ct.report_date
""", own_path)
own_path_fwd = replace(own_path, "\\" => "/")
write_manifest("04_ownership_ict", own_path; input_paths=STEP_INPUTS)

# Sanity-check: sum of portfolio_weight_eu over (investor_country, report_date)
# should be ~1 for EU sec_country only.
pw_sanity = qdf(con, """
    SELECT investor_country, report_date,
           SUM(CASE WHEN sec_country IN $EU_str THEN portfolio_weight_eu ELSE 0 END) AS sum_w_eu
    FROM read_parquet('$own_path_fwd')
    WHERE investor_country = 'US' AND report_date IN (DATE '2018-12-31', DATE '2022-12-31')
    GROUP BY 1,2 ORDER BY 1,2
""")
println("Sanity (sum w_eu in EU per US-quarter; should be ~1):")
println(pw_sanity)

# ============================================================
# (4) Descriptive: US ownership of European firms
# ============================================================
println("\n========== US Ownership of European firms ==========")

# Pick snapshot date: prefer 2018-12-31 if in panel, else most recent month-end
snap_q = qdf(con, """
    SELECT MAX(report_date) AS d
    FROM read_parquet('$own_path_fwd')
    WHERE report_date <= DATE '2018-12-31'
""")
snap_date = (snap_q.d[1] === missing || snap_q.d[1] === nothing || nrow(snap_q) == 0) ?
    qdf(con, "SELECT MAX(report_date) AS d FROM read_parquet('$own_path_fwd')").d[1] :
    snap_q.d[1]
println("Snapshot date for cross-sectional descriptives: $snap_date")

# Diagnostic BEFORE the silent BETWEEN(0,1) drop: how many cells get filtered?
share_filter_diag = qdf(con, """
    SELECT
        COUNT(*) AS n_total,
        SUM(CASE WHEN ownership_share IS NULL THEN 1 ELSE 0 END) AS n_null,
        SUM(CASE WHEN ownership_share < 0  THEN 1 ELSE 0 END) AS n_negative,
        SUM(CASE WHEN ownership_share > 1  THEN 1 ELSE 0 END) AS n_above_one,
        SUM(CASE WHEN ownership_share BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS n_in_range
    FROM read_parquet('$own_path_fwd')
    WHERE investor_country = 'US'
      AND sec_country IN $EU_str
      AND report_date = DATE '$snap_date'
""")
println("Ownership-share filter diagnostic ($snap_date):")
println(share_filter_diag)
if share_filter_diag.n_above_one[1] > 0
    above_one_top = qdf(con, """
        SELECT sec_entity_id, sec_country, market_cap, I_ict, ownership_share
        FROM read_parquet('$own_path_fwd')
        WHERE investor_country = 'US'
          AND sec_country IN $EU_str
          AND report_date = DATE '$snap_date'
          AND ownership_share > 1
        ORDER BY ownership_share DESC LIMIT 5
    """)
    println("  Top-5 sec_entity_ids with ownership_share > 1 (typically dual-class / inversion):")
    println(above_one_top)
end

us_eu_snap = qdf(con, """
    SELECT sec_entity_id, sec_country, I_ict, market_cap, ownership_share
    FROM read_parquet('$own_path_fwd')
    WHERE investor_country = 'US'
      AND sec_country IN $EU_str
      AND report_date = DATE '$snap_date'
      AND market_cap > 0
""")
println("US ownership cells on $snap_date: $(nrow(us_eu_snap))")

# Drop pathological values (with the above_one_top diagnostic capturing what's lost)
clean = filter(:ownership_share => x -> !ismissing(x) && 0 <= x <= 1, us_eu_snap)
println("After filtering 0 <= ownership <= 1: $(nrow(clean))")

if nrow(clean) == 0
    try
        DBInterface.close!(con)
    catch
    end
    error("Empty snapshot — check EOM panel and ISSUE_TYPE filter in 04.")
end

# Percentiles
qs = quantile(skipmissing(clean.ownership_share), [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
println("\nOwnership share distribution (US holdings of EU firms, $snap_date):")
for (p, q) in zip([10,25,50,75,90,95,99], qs)
    @printf("  P%-3d: %.4f (%.2f%%)\n", p, q, 100q)
end
@printf("  mean: %.4f (%.2f%%)\n", mean(skipmissing(clean.ownership_share)),
        100*mean(skipmissing(clean.ownership_share)))
@printf("  max:  %.4f\n", maximum(skipmissing(clean.ownership_share)))

CSV.write(joinpath(OUT_DIR, "04_us_ownership_eu_snapshot.csv"), clean)

# By country
by_country = qdf(con, """
    SELECT sec_country,
           COUNT(*) AS n_firms,
           AVG(ownership_share) AS mean_us_own,
           QUANTILE_CONT(ownership_share, 0.5) AS median_us_own,
           SUM(I_ict)/1e9 AS total_us_holding_b
    FROM read_parquet('$own_path_fwd')
    WHERE investor_country = 'US'
      AND sec_country IN $EU_str
      AND report_date = DATE '$snap_date'
      AND ownership_share IS NOT NULL AND ownership_share BETWEEN 0 AND 1
    GROUP BY sec_country ORDER BY total_us_holding_b DESC
""")
println("\nUS ownership of EU firms by country ($snap_date):")
println(by_country)
CSV.write(joinpath(OUT_DIR, "04_us_own_by_eu_country_snapshot.csv"), by_country)

# ============================================================
# (5) Time series: US ownership of EU firms over time
# ============================================================
println("\n========== Time series of US ownership in EU ==========")
ts = qdf(con, """
    SELECT report_date,
           COUNT(DISTINCT sec_entity_id) AS n_eu_firms,
           AVG(ownership_share) AS mean_us_own,
           SUM(I_ict)/1e9 AS total_us_holding_b
    FROM read_parquet('$own_path_fwd')
    WHERE investor_country = 'US'
      AND sec_country IN $EU_str
      AND ownership_share IS NOT NULL AND ownership_share BETWEEN 0 AND 1
    GROUP BY report_date ORDER BY report_date
""")
CSV.write(joinpath(OUT_DIR, "04_us_ownership_eu_timeseries.csv"), ts)

# Top 30 EU firms by US ownership at snapshot date
top_us = qdf(con, """
    SELECT sec_entity_id, sec_country, market_cap/1e9 AS mcap_b,
           I_ict/1e9 AS us_holding_b, ownership_share
    FROM read_parquet('$own_path_fwd')
    WHERE investor_country = 'US'
      AND sec_country IN $EU_str
      AND report_date = DATE '$snap_date'
      AND ownership_share BETWEEN 0 AND 1
      AND market_cap > 1e9
    ORDER BY ownership_share DESC LIMIT 30
""")
CSV.write(joinpath(OUT_DIR, "04_top30_us_owned_eu_firms.csv"), top_us)
println("\nTop 30 EU firms by US ownership share at $snap_date (mcap > 1B) -> 04_top30_us_owned_eu_firms.csv")

try
    DBInterface.close!(con)
catch e
    @warn "DBInterface.close! failed" exception=e
end

println("\n========== DONE ==========")
println("Key outputs:")
println("  I_ict_panel.parquet")
println("  ownership_ict.parquet     (portfolio_weight_eu = regression input; portfolio_weight_global = diagnostic)")
println("  marketcap_it.parquet      (AVG-not-MAX; dispersion stddev columns)")
println("  country_total_ct.parquet  (both global and EU-restricted totals)")
println("  04_us_ownership_eu_*.csv")
println("\nNext: 05_combine_visualize.jl  (Sections 6-7 pre-regression checks)")
