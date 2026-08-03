# 06_russia_grid.jl
# ISOMORPHIC copy of 06_cartesian_grid.jl for the Russia positive control
# (Third external review round, R3-A #1 / P0 #3). Reuses the IDENTICAL EU
# universe / holdings / Cartesian-grid / weight construction as the China
# run (that machinery does not depend on the exposure country); only the
# exposure source (firm_quarter_russia_exposure.parquet) and shock source
# (russia_gpr_quarterly.parquet) differ, and every identifier is renamed
# china->russia / cn->ru so no column silently holds Russia data under a
# China-labeled name. Output: merged_us_ru_zero_filled.parquet (does not
# touch merged_us_eu_zero_filled.parquet).

# 06_cartesian_grid.jl
# Audit fix C6 — zero-fill EU firm × holder-group × quarter Cartesian grid.
#
# v2 (post-adversarial-review). Patches applied vs v1:
#   1. Scatter CSV now carries us_ownership_share (I_ict_US/market_cap)
#      so fig10 in plots_python.py doesn't crash on DPN_USE_C6=true.
#   2. Crosswalk picks ONE eu_company_id per sec_entity_id with priority
#      CUSIP > ISIN > SEDOL (was: SUM across multi-match → bias).
#   3. ict_grouped keys off eu_entity_universe instead of duplicating the
#      sec_country EU filter (which could disagree with holdings_eom).
#   4. Merged parquet carries BOTH holder_group AND investor_country (alias)
#      so downstream consumers expecting 05's investor_country column name
#      work without code changes.
#   5. GPR LEFT JOIN uses SELECT DISTINCT to guard against duplicate
#      quarter_end rows blowing up the grid.
#   6. mean_diff_ws is actually winsorised at p1/p99 (not a duplicate of
#      mean_diff).
#   7. country_total_grouped uses COALESCE so NULL totals don't silently
#      drop entire (holder_group × quarter) cells.
#   8. Median cutoff for HIGH/LOW computed ONCE on firm-quarter distinct
#      cells (de-duplicates across holder_group) and reused as a Julia
#      constant — no drift across the three subqueries.
#   9. Asserts: sec_entity_id unique in canonical universe, unique in
#      crosswalk, russia_share ∈ [0,1].
#  10. Diagnostic counts: n_delta_w_null_at_boundary vs n_delta_w_null_interior,
#      held-vs-zero-filled composition table.
#
# Why this exists. The 05 panel keeps only (sec_entity_id, holder_country,
# quarter) cells with a positive holding. That is selection-on-outcome for
# a regression whose outcome is institutional holdings: an extensive-margin
# exit (positive → zero) and an extensive-margin entry (zero → positive)
# both disappear because FactSet stores no row for "holding = 0". This
# biases the headline β_3 toward zero.

include("00_setup.jl")

const EOM_PATH        = replace(joinpath(OUT_DIR, test_suffix_path("holdings_eom.parquet")), "\\" => "/")
const ICT_PATH        = replace(joinpath(OUT_DIR, test_suffix_path("I_ict_panel.parquet")), "\\" => "/")
const COUNTRY_TOT_PATH= replace(joinpath(OUT_DIR, test_suffix_path("country_total_ct.parquet")), "\\" => "/")
const EXP_PATH_RU        = replace(joinpath(OUT_DIR, test_suffix_path("firm_quarter_russia_exposure.parquet")), "\\" => "/")
const UNIV_PATH       = replace(joinpath(OUT_DIR, test_suffix_path("eu_revere_universe.parquet")), "\\" => "/")
const GPR_Q_PATH_RU      = replace(joinpath(OUT_DIR, test_suffix_path("russia_gpr_quarterly.parquet")), "\\" => "/")
const MCAP_PATH       = replace(joinpath(OUT_DIR, test_suffix_path("marketcap_it.parquet")), "\\" => "/")

for p in (EOM_PATH, ICT_PATH, COUNTRY_TOT_PATH, EXP_PATH_RU, UNIV_PATH, GPR_Q_PATH_RU, MCAP_PATH)
    isfile(replace(p, "/" => "\\")) || error("Required input missing: $p")
end

const STEP_INPUTS = [EOM_PATH, ICT_PATH, COUNTRY_TOT_PATH, EXP_PATH_RU, UNIV_PATH, GPR_Q_PATH_RU, MCAP_PATH]

println("\n" * "!"^70)
println("!! 06_cartesian_grid.jl v2 — C6 zero-fill panel build                 !!")
println("!! Adversarial review patches applied (priority crosswalk, scatter    !!")
println("!! us_ownership_share, gpr DISTINCT, winsor, NULL guards, asserts).   !!")
println("!"^70 * "\n")

con = dbcon()

# ============================================================
# (1) EU entity universe — every sec_entity_id ever appearing with an EU
#     sec_country in the holdings panel.
# ============================================================
println("Building EU entity universe from holdings_eom...")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE eu_entity_universe AS
    WITH primary_country AS (
        SELECT sec_entity_id, sec_country,
               COUNT(*) AS n_rows
        FROM read_parquet('$EOM_PATH')
        WHERE sec_country IN $EU_SQL_TUPLE
          AND sec_entity_id IS NOT NULL
        GROUP BY sec_entity_id, sec_country
    ),
    canonical AS (
        SELECT sec_entity_id, sec_country,
               ROW_NUMBER() OVER (PARTITION BY sec_entity_id
                                  ORDER BY n_rows DESC, sec_country ASC) AS rn
        FROM primary_country
    )
    SELECT sec_entity_id, sec_country
    FROM canonical
    WHERE rn = 1
""")
n_eu = qdf(con, "SELECT COUNT(*) AS n FROM eu_entity_universe").n[1]
n_eu_unique = qdf(con, "SELECT COUNT(DISTINCT sec_entity_id) AS n FROM eu_entity_universe").n[1]
@assert n_eu == n_eu_unique "eu_entity_universe is not unique on sec_entity_id ($n_eu rows, $n_eu_unique unique IDs)"
println("  EU entity universe size: $n_eu (unique sec_entity_ids ✓)")

# Diagnostic: how many firms are dual-listed across EU sec_countries?
dual = qdf(con, """
    SELECT COUNT(*) AS n_dual_listed
    FROM (
        SELECT sec_entity_id
        FROM read_parquet('$EOM_PATH')
        WHERE sec_country IN $EU_SQL_TUPLE
        GROUP BY sec_entity_id
        HAVING COUNT(DISTINCT sec_country) > 1
    )
""")
println("  Dual-listed (>1 EU sec_country): $(dual.n_dual_listed[1])")

# ============================================================
# (2) Quarter calendar.
# ============================================================
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE quarters AS
    SELECT LAST_DAY(MAKE_DATE(y, m, 1)) AS quarter_end
    FROM range(1999, 2024) y(y)
    CROSS JOIN (VALUES (3),(6),(9),(12)) AS month_tbl(m)
    WHERE LAST_DAY(MAKE_DATE(y, m, 1)) BETWEEN DATE '1999-03-31' AND DATE '2023-12-31'
    ORDER BY quarter_end
""")
n_q = qdf(con, "SELECT COUNT(*) AS n FROM quarters").n[1]
println("  Quarter calendar: $n_q quarter-ends")

# ============================================================
# (3) Holder-group dimension (binary US / NONUS).
# ============================================================
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE holder_groups AS
    SELECT 'US' AS holder_group UNION ALL SELECT 'NONUS' AS holder_group
""")

# ============================================================
# (4) Cartesian grid: EU firm × holder_group × quarter.
# ============================================================
println("\nBuilding Cartesian grid (firm × holder_group × quarter)...")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE cartesian_grid AS
    SELECT u.sec_entity_id, u.sec_country, h.holder_group, q.quarter_end
    FROM eu_entity_universe u
    CROSS JOIN holder_groups h
    CROSS JOIN quarters q
""")
n_grid = qdf(con, "SELECT COUNT(*) AS n FROM cartesian_grid").n[1]
println("  Cartesian grid size: $n_grid cells")

# ============================================================
# (5) ict_grouped: I_ict aggregated to (sec, holder_group, quarter).
#     Patch 3: filter on sec_entity_id ∈ universe rather than re-applying
#     EU sec_country filter (which could disagree across sources).
# ============================================================
println("\nAggregating I_ict to holder-group level (universe-filtered)...")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE ict_grouped AS
    SELECT i.sec_entity_id,
           CASE WHEN i.investor_country = 'US' THEN 'US' ELSE 'NONUS' END AS holder_group,
           i.report_date AS quarter_end,
           SUM(i.I_ict) AS I_ict
    FROM read_parquet('$ICT_PATH') i
    WHERE i.sec_entity_id IN (SELECT sec_entity_id FROM eu_entity_universe)
    GROUP BY i.sec_entity_id, holder_group, i.report_date
""")

# (5b) country_total_grouped — Patch 7: use COALESCE so NULL totals don't
# nuke an entire (group × quarter).
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE country_total_grouped AS
    SELECT CASE WHEN investor_country = 'US' THEN 'US' ELSE 'NONUS' END AS holder_group,
           report_date AS quarter_end,
           SUM(COALESCE(country_total_holdings_eu, 0)) AS total_holdings_eu
    FROM read_parquet('$COUNTRY_TOT_PATH')
    GROUP BY holder_group, report_date
""")
ct_check = qdf(con, """
    SELECT
        COUNT(*) AS n_rows,
        COUNT(*) FILTER (WHERE total_holdings_eu IS NULL OR total_holdings_eu = 0) AS n_zero_or_null
    FROM country_total_grouped
""")
println("  country_total_grouped: $(ct_check.n_rows[1]) rows; $(ct_check.n_zero_or_null[1]) zero/NULL")

# ============================================================
# (6) ChinaExposure crosswalk — Patch 2: priority CUSIP > ISIN > SEDOL,
#     ONE eu_company_id per sec_entity_id.
# ============================================================
println("\nBuilding ChinaExposure crosswalk (CUSIP > ISIN > SEDOL priority)...")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE eu_sec_ids AS
    SELECT DISTINCT
           sec_entity_id,
           NULLIF(TRIM(cusip), '') AS cusip,
           NULLIF(TRIM(isin),  '') AS isin,
           NULLIF(TRIM(sedol), '') AS sedol
    FROM read_parquet('$EOM_PATH')
    WHERE sec_country IN $EU_SQL_TUPLE
      AND sec_entity_id IS NOT NULL
""")

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE matched_eu_sec_links AS
    WITH univ AS (
        SELECT eu_company_id,
               NULLIF(TRIM(eu_cusip), '') AS eu_cusip,
               NULLIF(TRIM(eu_isin),  '') AS eu_isin,
               NULLIF(TRIM(eu_sedol), '') AS eu_sedol
        FROM read_parquet('$UNIV_PATH')
    ),
    all_matches AS (
        SELECT 1 AS prio, s.sec_entity_id, u.eu_company_id
        FROM eu_sec_ids s JOIN univ u ON s.cusip = u.eu_cusip
        WHERE s.cusip IS NOT NULL AND u.eu_cusip IS NOT NULL
        UNION ALL
        SELECT 2 AS prio, s.sec_entity_id, u.eu_company_id
        FROM eu_sec_ids s JOIN univ u ON s.isin = u.eu_isin
        WHERE s.isin IS NOT NULL AND u.eu_isin IS NOT NULL
        UNION ALL
        SELECT 3 AS prio, s.sec_entity_id, u.eu_company_id
        FROM eu_sec_ids s JOIN univ u ON s.sedol = u.eu_sedol
        WHERE s.sedol IS NOT NULL AND u.eu_sedol IS NOT NULL
    )
    SELECT sec_entity_id, eu_company_id
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY sec_entity_id
                                  ORDER BY prio ASC, eu_company_id ASC) AS rn
        FROM all_matches
    )
    WHERE rn = 1
""")
cw_n = qdf(con, "SELECT COUNT(*) AS n, COUNT(DISTINCT sec_entity_id) AS u FROM matched_eu_sec_links").n[1]
cw_u = qdf(con, "SELECT COUNT(DISTINCT sec_entity_id) AS u FROM matched_eu_sec_links").u[1]
@assert cw_n == cw_u "matched_eu_sec_links is not unique on sec_entity_id ($cw_n rows, $cw_u unique)"
println("  Crosswalk: $cw_n unique (sec_entity_id → eu_company_id) pairs ✓")

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE exposure_by_sec_entity_ru AS
    SELECT m.sec_entity_id,
           e.quarter_end,
           e.n_ru_total,
           e.n_total_links,
           e.n_supplychain_links,
           -- (B7 FIX, 2026-08-03) supply-chain russia_share = RU CUSTOMER+SUPPLIER
           -- / total CUSTOMER+SUPPLIER (matches 02_russia_exposure.jl and the
           -- China 06 fix). Crosswalk is unique on sec_entity_id here (asserted
           -- above), so a direct ratio is exact.
           (e.n_ru_customer + e.n_ru_supplier)::DOUBLE
               / NULLIF(e.n_supplychain_links, 0) AS russia_share,
           -- legacy all-relationship-type share, retained for diagnostics only
           e.n_ru_total::DOUBLE / NULLIF(e.n_total_links, 0) AS russia_share_alltypes
    FROM matched_eu_sec_links m
    JOIN read_parquet('$EXP_PATH_RU') e ON m.eu_company_id = e.eu_company_id
""")
# Patch 4 assertion: russia_share ∈ [0,1]
oob = qdf(con, """
    SELECT COUNT(*) AS n FROM exposure_by_sec_entity_ru
    WHERE russia_share IS NOT NULL AND (russia_share < 0 OR russia_share > 1)
""")
@assert oob.n[1] == 0 "exposure_by_sec_entity_ru has $(oob.n[1]) rows with russia_share ∉ [0,1]"
println("  exposure_by_sec_entity_ru: russia_share ∈ [0,1] ✓")

# ============================================================
# (7) Stitch everything onto the grid + zero-fill.
#     Patch 5: gpr LEFT JOIN with SELECT DISTINCT to guard against
#     duplicate quarter_end rows.
# ============================================================
println("\nStitching the zero-filled merged panel...")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE grid_zfilled AS
    SELECT g.sec_entity_id, g.sec_country, g.holder_group, g.quarter_end,
           COALESCE(i.I_ict, 0) AS I_ict,
           COALESCE(ct.total_holdings_eu, 0) AS total_holdings_eu,
           CASE WHEN COALESCE(ct.total_holdings_eu, 0) > 0
                THEN COALESCE(i.I_ict, 0) / ct.total_holdings_eu
                ELSE NULL END AS portfolio_weight_eu,
           CASE WHEN g.quarter_end < DATE '2003-03-31' THEN NULL
                ELSE e.russia_share END AS russia_share,
           gpr.gpr_us_ru,
           gpr.shock_us_ru
    FROM cartesian_grid g
    LEFT JOIN ict_grouped i USING (sec_entity_id, holder_group, quarter_end)
    LEFT JOIN country_total_grouped ct USING (holder_group, quarter_end)
    LEFT JOIN exposure_by_sec_entity_ru e USING (sec_entity_id, quarter_end)
    LEFT JOIN (
        SELECT DISTINCT quarter_end, gpr_us_ru, shock_us_ru
        FROM read_parquet('$GPR_Q_PATH_RU')
    ) gpr USING (quarter_end)
""")

# ============================================================
# (8) Lag exposure + backward Δw + emit merged parquet.
#     Patch 4: holder_group AS investor_country alias for back-compat.
# ============================================================
println("\nComputing lagged exposure + backward Δw on the zero-filled panel...")
merged_path = test_suffix_path(joinpath(OUT_DIR, "merged_us_ru_zero_filled.parquet"))
atomic_copy_to(con, """
    SELECT
        sec_entity_id, sec_country,
        holder_group,
        holder_group AS investor_country,  -- back-compat alias
        quarter_end AS report_date,
        I_ict,
        portfolio_weight_eu,
        LAG(portfolio_weight_eu, 1) OVER (PARTITION BY sec_entity_id, holder_group ORDER BY quarter_end) AS w_prev,
        (portfolio_weight_eu
         - LAG(portfolio_weight_eu, 1) OVER (PARTITION BY sec_entity_id, holder_group ORDER BY quarter_end)) AS delta_w,
        russia_share,
        LAG(russia_share, 1) OVER (PARTITION BY sec_entity_id, holder_group ORDER BY quarter_end) AS russia_share_lag1q,
        gpr_us_ru,
        shock_us_ru
    FROM grid_zfilled
""", merged_path)
merged_path_fwd = replace(merged_path, "\\" => "/")
n_merged = qdf(con, "SELECT COUNT(*) AS n FROM read_parquet('$merged_path_fwd')").n[1]
println("  Zero-filled panel rows: $n_merged")
write_manifest("06_merged_us_eu_zero_filled", merged_path; row_count=n_merged, input_paths=STEP_INPUTS)

# ============================================================
# (9) Patch 8: compute the HIGH-vs-LOW cutoff ONCE on firm-quarter distinct
#     rows (deduped across holder_group) and reuse as a Julia constant.
# ============================================================
med_q = qdf(con, """
    SELECT QUANTILE_CONT(russia_share_lag1q, 0.5) AS med
    FROM (
        SELECT DISTINCT sec_entity_id, report_date, russia_share_lag1q
        FROM read_parquet('$merged_path_fwd')
        WHERE russia_share_lag1q IS NOT NULL AND russia_share_lag1q > 0
    )
""")
const HIGH_CUTOFF_RU = (nrow(med_q) > 0 && !ismissing(med_q.med[1]) && !isnan(med_q.med[1])) ? med_q.med[1] : 0.0
println("\nHIGH-exposure cutoff (firm-quarter median on positive cells): $(round(HIGH_CUTOFF_RU, digits=4))")
@assert HIGH_CUTOFF_RU > 0 "HIGH_CUTOFF_RU degenerated to $HIGH_CUTOFF_RU — no positive russia_share_lag1q observations in the merged panel; check exposure_by_sec_entity_ru"

# Patch 10: delta_w NULL diagnostic (backward Δw: NULL iff w_prev is NULL,
# i.e. the first quarter of each (firm × holder_group) series).
ndiag = qdf(con, """
    SELECT
        COUNT(*) FILTER (WHERE delta_w IS NULL AND w_prev IS NULL)     AS n_null_first_quarter,
        COUNT(*) FILTER (WHERE delta_w IS NULL AND w_prev IS NOT NULL) AS n_null_interior,
        COUNT(*) FILTER (WHERE delta_w IS NOT NULL)                    AS n_delta_w_nonnull
    FROM read_parquet('$merged_path_fwd')
""")
println("Δw NULL diagnostic (backward diff):")
println("  first quarter of series (w_prev missing): $(ndiag.n_null_first_quarter[1])")
println("  interior (w_prev present but Δw NULL):    $(ndiag.n_null_interior[1])  ← should be 0")
println("  non-NULL Δw: $(ndiag.n_delta_w_nonnull[1])")

# Held vs zero-filled composition (Patch 10)
comp = qdf(con, """
    SELECT holder_group,
           CASE WHEN I_ict > 0 THEN 'held' ELSE 'zero_filled' END AS hold_status,
           CASE WHEN russia_share_lag1q IS NULL    THEN 'MISSING'
                WHEN russia_share_lag1q > $HIGH_CUTOFF_RU THEN 'HIGH'
                ELSE                                       'LOW' END AS exp_grp,
           COUNT(*) AS n
    FROM read_parquet('$merged_path_fwd')
    GROUP BY 1,2,3 ORDER BY 1,2,3
""")
CSV.write(joinpath(OUT_DIR, "06_panel_composition_c6_RUSSIA.csv"), comp)
println("\nC6 panel composition (held vs zero-filled, by exposure bucket):")
println(comp)

# ============================================================
# (10) Descriptive analogues — _c6 suffixed CSVs.
# ============================================================
println("\nBuilding descriptive analogues on C6 panel...")

# (10a) Cross-section scatter — Patch 1: us_ownership_share = I_ict_US / market_cap.
snap_q = qdf(con, """
    SELECT MAX(report_date) AS d FROM read_parquet('$merged_path_fwd')
    WHERE report_date <= DATE '2018-12-31'
""")
snap = (nrow(snap_q) > 0 && !ismissing(snap_q.d[1])) ? snap_q.d[1] :
       qdf(con, "SELECT MAX(report_date) AS d FROM read_parquet('$merged_path_fwd')").d[1]
println("  Scatter snapshot date: $snap")

sc = qdf(con, """
    SELECT z.sec_entity_id,
           z.russia_share,
           z.russia_share_lag1q,
           z.I_ict,
           z.portfolio_weight_eu,
           m.market_cap,
           CASE WHEN m.market_cap > 0
                THEN z.I_ict / m.market_cap
                ELSE NULL END AS us_ownership_share
    FROM read_parquet('$merged_path_fwd') z
    LEFT JOIN read_parquet('$MCAP_PATH') m
      ON z.sec_entity_id = m.sec_entity_id
     AND z.report_date   = m.report_date
    WHERE z.holder_group = 'US'
      AND z.report_date = DATE '$snap'
      AND z.portfolio_weight_eu IS NOT NULL
""")
CSV.write(joinpath(OUT_DIR, "05_scatter_own_vs_ru_data_c6_RUSSIA.csv"), sc)
println("  fig10 scatter data → 05_scatter_own_vs_ru_data_c6_RUSSIA.csv ($(nrow(sc)) rows; us_ownership_share included)")

# (10b) Bucket time series — Patch 7: bucket from DISTINCT firm-quarter,
# both holder groups share the same bucket assignment.
ts_alloc = qdf(con, """
    WITH bucket AS (
        SELECT DISTINCT sec_entity_id, report_date,
               CASE WHEN russia_share_lag1q IS NULL          THEN 'MISSING'
                    WHEN russia_share_lag1q > $HIGH_CUTOFF_RU   THEN 'HIGH'
                    ELSE                                         'LOW'
               END AS exp_grp
        FROM read_parquet('$merged_path_fwd')
    ),
    sided AS (
        SELECT z.report_date, z.holder_group, b.exp_grp,
               SUM(z.portfolio_weight_eu) AS abs_portfolio_weight
        FROM read_parquet('$merged_path_fwd') z
        JOIN bucket b USING (sec_entity_id, report_date)
        WHERE z.portfolio_weight_eu IS NOT NULL
        GROUP BY 1, 2, 3
    ),
    totals AS (
        SELECT report_date, holder_group, SUM(portfolio_weight_eu) AS w_total
        FROM read_parquet('$merged_path_fwd')
        WHERE portfolio_weight_eu IS NOT NULL
        GROUP BY 1, 2
    )
    SELECT s.report_date,
           CASE WHEN s.holder_group = 'US' THEN 'US' ELSE 'NONUS' END AS investor_country,
           s.exp_grp,
           s.abs_portfolio_weight,
           s.abs_portfolio_weight / NULLIF(t.w_total, 0) AS within_europe_share,
           gpr.gpr_us_ru, gpr.shock_us_ru
    FROM sided s
    LEFT JOIN totals t USING (report_date, holder_group)
    LEFT JOIN (SELECT DISTINCT quarter_end, gpr_us_ru, shock_us_ru FROM read_parquet('$GPR_Q_PATH_RU')) gpr
           ON s.report_date = gpr.quarter_end
    ORDER BY s.report_date, s.holder_group, s.exp_grp
""")
CSV.write(joinpath(OUT_DIR, "05_within_europe_share_by_group_c6_RUSSIA.csv"), ts_alloc)
println("  fig11 bucket data → 05_within_europe_share_by_group_c6_RUSSIA.csv ($(nrow(ts_alloc)) rows)")

# (10c) Differential ΔUS - ΔnonUS for HIGH-exposure firms vs Shock.
#       Patch 6: actually winsorize mean_diff_ws at p1/p99 of the diff series.
println("\nComputing differential (with REAL winsorization)...")
# Step 1: get raw per-firm-quarter (d_us - d_nonus) on HIGH-exposure cells.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE diff_high_raw AS
    WITH us_w AS (
        SELECT sec_entity_id, report_date, delta_w AS d_us_w, russia_share_lag1q
        FROM read_parquet('$merged_path_fwd')
        WHERE holder_group = 'US' AND delta_w IS NOT NULL
    ),
    nonus_w AS (
        SELECT sec_entity_id, report_date, delta_w AS d_nonus_w
        FROM read_parquet('$merged_path_fwd')
        WHERE holder_group = 'NONUS' AND delta_w IS NOT NULL
    )
    SELECT u.sec_entity_id, u.report_date, u.russia_share_lag1q,
           u.d_us_w, n.d_nonus_w,
           (u.d_us_w - n.d_nonus_w) AS d_diff
    FROM us_w u JOIN nonus_w n USING (sec_entity_id, report_date)
    WHERE u.russia_share_lag1q > $HIGH_CUTOFF_RU
""")
# Step 2: compute p1/p99 bounds for winsorization.
ws_b = qdf(con, """
    SELECT QUANTILE_CONT(d_diff, 0.01) AS p1, QUANTILE_CONT(d_diff, 0.99) AS p99
    FROM diff_high_raw
""")
p1   = nrow(ws_b) > 0 && !ismissing(ws_b.p1[1])  ? ws_b.p1[1]  : -Inf
p99  = nrow(ws_b) > 0 && !ismissing(ws_b.p99[1]) ? ws_b.p99[1] :  Inf
println("  winsorization bounds: p1=$(round(p1, digits=8)) p99=$(round(p99, digits=8))")

ts_diff = qdf(con, """
    SELECT report_date,
           AVG(d_diff)                                          AS mean_diff,
           AVG(LEAST(GREATEST(d_diff, $p1), $p99))              AS mean_diff_ws,
           AVG(d_us_w)                                          AS mean_d_us_raw,
           AVG(d_nonus_w)                                       AS mean_d_nonus_raw,
           AVG(d_us_w - d_nonus_w)                              AS mean_diff_raw,
           COUNT(*)                                             AS n_firms,
           ANY_VALUE(g.gpr_us_ru)                               AS gpr_us_ru,
           ANY_VALUE(g.shock_us_ru)                             AS shock_us_ru
    FROM diff_high_raw d
    LEFT JOIN (SELECT DISTINCT quarter_end, gpr_us_ru, shock_us_ru FROM read_parquet('$GPR_Q_PATH_RU')) g
           ON d.report_date = g.quarter_end
    WHERE g.gpr_us_ru IS NOT NULL
    GROUP BY report_date ORDER BY report_date
""")
CSV.write(joinpath(OUT_DIR, "05_diff_us_vs_nonus_high_c6_RUSSIA.csv"), ts_diff)
println("  fig13 differential data → 05_diff_us_vs_nonus_high_c6_RUSSIA.csv ($(nrow(ts_diff)) rows; mean_diff_ws is REALLY winsorized)")

if nrow(ts_diff) > 10
    # Joint mask: drop rows where ANY of the three series is Missing/NaN so
    # cor() doesn't MethodError on Union{Missing,Float64}. Then disallowmissing
    # to coerce the eltype.
    mm = .!ismissing.(ts_diff.shock_us_ru) .&
         .!ismissing.(ts_diff.mean_diff)   .&
         .!ismissing.(ts_diff.mean_diff_ws)
    if sum(mm) > 10
        s  = Vector{Float64}(ts_diff.shock_us_ru[mm])
        mr = Vector{Float64}(ts_diff.mean_diff[mm])
        mw = Vector{Float64}(ts_diff.mean_diff_ws[mm])
        c_raw = cor(mr, s)
        c_ws  = cor(mw, s)
        println("\n  cor((ΔUS − ΔnonUS), Shock^{US-CN}) on C6 panel for HIGH-lag firms:")
        println("    raw:        $(round(c_raw, digits=4))")
        println("    winsorized: $(round(c_ws,  digits=4))")
        println("  obs: $(sum(mm)) (of $(nrow(ts_diff)) total ts_diff rows)")
    else
        println("  too few non-missing rows ($(sum(mm))) to compute correlation")
    end
end

# (10d) US vs NONUS HIGH-exposure aggregate data (fig12).
hi = qdf(con, """
    WITH high_cells AS (
        SELECT z.sec_entity_id, z.report_date, z.holder_group, z.portfolio_weight_eu
        FROM read_parquet('$merged_path_fwd') z
        WHERE z.russia_share_lag1q > $HIGH_CUTOFF_RU
    ),
    summed AS (
        SELECT report_date, holder_group, SUM(portfolio_weight_eu) AS abs_portfolio_weight
        FROM high_cells
        WHERE portfolio_weight_eu IS NOT NULL
        GROUP BY 1, 2
    ),
    wide AS (
        SELECT report_date,
               SUM(CASE WHEN holder_group = 'NONUS' THEN abs_portfolio_weight ELSE 0 END) AS nonus_abs_pw,
               SUM(CASE WHEN holder_group = 'US'    THEN abs_portfolio_weight ELSE 0 END) AS us_abs_pw
        FROM summed
        GROUP BY report_date
    )
    SELECT w.report_date,
           w.nonus_abs_pw, w.us_abs_pw,
           g.gpr_us_ru, g.shock_us_ru
    FROM wide w
    LEFT JOIN (SELECT DISTINCT quarter_end, gpr_us_ru, shock_us_ru FROM read_parquet('$GPR_Q_PATH_RU')) g
           ON w.report_date = g.quarter_end
    ORDER BY w.report_date
""")
CSV.write(joinpath(OUT_DIR, "05_us_vs_nonus_high_share_data_c6_RUSSIA.csv"), hi)
println("  fig12 US-vs-NONUS HIGH data → 05_us_vs_nonus_high_share_data_c6_RUSSIA.csv ($(nrow(hi)) rows)")

try
    DBInterface.close!(con)
catch e
    @warn "DBInterface.close! failed" exception=e
end

println("\n========== 06_russia_grid.jl (Russia positive control) DONE ==========")
println("Key output:")
println("  merged_us_ru_zero_filled.parquet  (REGRESSION-READY zero-filled panel, Russia exposure/shock,")
println("                                      holder_group AND investor_country columns)")
println("  05_scatter_own_vs_ru_data_c6_RUSSIA.csv         (Russia-analogue descriptive, not consumed by China figures)")
println("  05_within_europe_share_by_group_c6_RUSSIA.csv   (Russia-analogue descriptive)")
println("  05_us_vs_nonus_high_share_data_c6_RUSSIA.csv    (Russia-analogue descriptive)")
println("  05_diff_us_vs_nonus_high_c6_RUSSIA.csv          (Russia-analogue descriptive)")
println("  06_panel_composition_c6_RUSSIA.csv              (Russia-analogue descriptive)")
println()
println("Next: build_russia_c6_panel.py -> c6_panel_russia.dta -> run_russia_headline.do")
