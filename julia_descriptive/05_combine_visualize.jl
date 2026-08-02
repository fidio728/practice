# 05_combine_visualize.jl
# Sections 6-7 PRE-REGRESSION descriptive checks (audited revision).
#
# !!! IMPORTANT !!!
# This script produces merged_us_eu_matched.parquet, but per the 2026-06-01
# audit (AUDIT_2026_06_01_julia_descriptive.md) the EU firm universe is still
# selection-on-outcome (firms ever held by ≥1 institution in FactSet). DO NOT
# use the output as a REGRESSION PANEL. It is DESCRIPTIVE ONLY. The full fix
# requires rebuilding the universe from Factset_Security_coverage with a
# Cartesian (firm × holder-country × quarter) grid + zero-fill — see C6 in
# the audit file. Until that rebuild lands, treat all numerical claims here
# as PROVISIONAL.
#
# Critical-fix changes in THIS script (per audit):
#
# C1 (denominator): switches to portfolio_weight_eu from 04 for the US side
#     and builds a parallel nonus_aggregate_eu so US and NONUS series share
#     scope. The previous nonus_aggregate used a global denominator and was
#     not comparable to US portfolio_weight (which was also global, see 04
#     fix). Sanity check sum_w_eu ≈ 1 at the US-quarter level is printed.
#
# C2 (pre-2003 NULL): drops the silent COALESCE→0 for pre-Revere quarters.
#     Pre-2003 firm-quarters now carry NULL china_share / n_cn_* and are
#     excluded from the exposure bucket CTE rather than populating ZERO.
#
# C3 (lag): adds LAG(china_share) OVER (PARTITION BY sec_entity_id ORDER BY
#     report_date) as china_share_lag1q to the merged panel. The bucket CTE
#     and any regression interaction MUST use the lagged column.
#
# C5 propagation: switched the merged-panel firm filter from the static
#     matched_eu_sec table to a time-versioned (sec_entity_id, quarter_end)
#     filter that respects the eu_revere_universe_qend AS-OF table built
#     in 02.
#
# High (AR(1)): refit on MONTHLY GPR (the spec). Residuals at the quarter-end
#     month are taken as Shock^{US-CN}_t. Coefficients (a, b) and the full
#     monthly + quarterly shock series are persisted to disk so the regression
#     input is reproducible from the repo alone.
#
# High (multi-match aggregation): replace MAX(n_cn_total) / MAX(china_share)
#     with SUM(n_cn_total) / SUM(n_total_links) ratio — the only self-
#     consistent ratio when one FactSet sec_entity_id maps to multiple Revere
#     companies. DISTINCT applied to the exposure_share_by_sec_entity JOIN.
#
# Other: drop firm_month_china_exposure.parquet alias (now reads
#     firm_quarter_china_exposure.parquet directly); replace hard ±1% trim
#     with winsorization in the descriptive scatter; emit gap_months
#     diagnostic.

include("00_setup.jl")

const EOM_PATH      = replace(joinpath(OUT_DIR, test_suffix_path("holdings_eom.parquet")), "\\" => "/")
const OWN_PATH      = replace(joinpath(OUT_DIR, test_suffix_path("ownership_ict.parquet")), "\\" => "/")
const EXP_PATH      = replace(joinpath(OUT_DIR, test_suffix_path("firm_quarter_china_exposure.parquet")), "\\" => "/")
const UNIV_PATH     = replace(joinpath(OUT_DIR, test_suffix_path("eu_revere_universe.parquet")), "\\" => "/")
const UNIV_QEND_PATH= replace(joinpath(OUT_DIR, test_suffix_path("eu_revere_universe_qend.parquet")), "\\" => "/")
const ICT_PATH      = replace(joinpath(OUT_DIR, test_suffix_path("I_ict_panel.parquet")), "\\" => "/")

for p in (EOM_PATH, OWN_PATH, EXP_PATH, UNIV_PATH, UNIV_QEND_PATH, ICT_PATH)
    if !isfile(replace(p, "/" => "\\"))
        error("$(basename(p)) not found. Run 02 (China exposure) and 03/04 (ETL) first.")
    end
end

# Note: EU_SQL_TUPLE is defined in 00_setup.jl as the single source of truth.
const EUROPE_str = EU_SQL_TUPLE

const STEP_INPUTS = [
    replace(EOM_PATH, "/" => "\\"),
    replace(OWN_PATH, "/" => "\\"),
    replace(EXP_PATH, "/" => "\\"),
    replace(UNIV_QEND_PATH, "/" => "\\"),
    GPR_PATH,
]

con = dbcon()

# Sample-period warning
yr = qdf(con, """
    SELECT MIN(EXTRACT(YEAR FROM report_date)) AS ymin,
           MAX(EXTRACT(YEAR FROM report_date)) AS ymax
    FROM read_parquet('$EOM_PATH')
""")
if yr.ymax[1] - yr.ymin[1] < 5
    println("\n" * "!"^70)
    println("WARNING: EOM panel covers only $(yr.ymin[1])-$(yr.ymax[1]).")
    println("This looks like TEST_MODE output. All time series below are restricted.")
    println("!"^70 * "\n")
end

# ============================================================
# (0) GeoPressure series — AR(1) decomposition on MONTHLY data (audit fix).
# Spec calls for monthly bilateral AI-GPR. The previous version fit AR(1) on
# the quarterly mean, which (a) does not match the spec, (b) attenuates the
# innovation magnitude, and (c) leaves the persistence coefficient on a
# different time scale than the data are documented at. We now fit AR(1) on
# monthly, then take the residual at the quarter-end month as Shock^{US-CN}_t.
# Coefficients and the full series are persisted to disk so the regression
# input is reproducible from the repo alone.
# ============================================================
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE gpr_monthly AS
    SELECT CAST(Date AS DATE)               AS month_first,
           LAST_DAY(CAST(Date AS DATE))     AS month_end,
           "USA|China"                      AS gpr_us_cn,
           GPR_AI                           AS gpr_global
    FROM read_csv_auto('$(replace(GPR_PATH, "\\" => "/"))', sample_size=-1)
    ORDER BY month_first
""")

# AR(1) on monthly series.
gpr_m = qdf(con, "SELECT * FROM gpr_monthly ORDER BY month_end")
println("GPR monthly: $(nrow(gpr_m)) months, $(gpr_m.month_end[1]) → $(gpr_m.month_end[end])")

a_hat_m = 0.0; b_hat_m = 0.0
let g = collect(skipmissing(gpr_m.gpr_us_cn))
    y  = g[2:end]
    yL = g[1:end-1]
    yL_mean = mean(yL); y_mean = mean(y)
    b_hat_m = sum((yL .- yL_mean) .* (y .- y_mean)) / sum((yL .- yL_mean).^2)
    a_hat_m = y_mean - b_hat_m * yL_mean
    println("AR(1) on MONTHLY USA|China GPR: a = $(round(a_hat_m, digits=4)), b = $(round(b_hat_m, digits=4))")
    # Monthly innovation = y_t - (a + b·y_{t-1}). First obs has no lag → missing.
    shock_m = vcat([missing], y .- (a_hat_m .+ b_hat_m .* yL))
    gpr_m.shock_us_cn_monthly = shock_m
end

# Persist monthly coefficients + series.
ar1_coef_csv = joinpath(OUT_DIR, "gpr_ar1_coefficients.csv")
open(ar1_coef_csv, "w") do io
    write(io, "param,value,frequency\n")
    write(io, "a,$(a_hat_m),monthly\n")
    write(io, "b,$(b_hat_m),monthly\n")
    write(io, "n_months,$(nrow(gpr_m)),monthly\n")
    write(io, "sample_first,$(gpr_m.month_end[1]),monthly\n")
    write(io, "sample_last,$(gpr_m.month_end[end]),monthly\n")
end
println("  AR(1) coefficients persisted -> $(basename(ar1_coef_csv))")

DuckDB.register_data_frame(con, gpr_m, "gpr_monthly_with_shock")

gpr_monthly_path = test_suffix_path(joinpath(OUT_DIR, "gpr_monthly_with_shock.parquet"))
atomic_copy_to(con, "SELECT * FROM gpr_monthly_with_shock", gpr_monthly_path)
write_manifest("05_gpr_monthly_with_shock", gpr_monthly_path; row_count=nrow(gpr_m), input_paths=[GPR_PATH])

# For joins to quarterly holdings, take the quarter-end-month innovation as
# Shock^{US-CN}_t. (Robustness via SUM of three monthly innovations is a
# documented next step.)
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE gpr_ts AS
    SELECT month_end AS quarter_end,
           gpr_us_cn,
           gpr_global,
           shock_us_cn_monthly AS shock_us_cn
    FROM gpr_monthly_with_shock
    WHERE EXTRACT(MONTH FROM month_end) IN (3, 6, 9, 12)
""")
gpr_quarterly_path = test_suffix_path(joinpath(OUT_DIR, "gpr_quarterly_with_shock.parquet"))
atomic_copy_to(con, "SELECT * FROM gpr_ts", gpr_quarterly_path)
write_manifest("05_gpr_quarterly_with_shock", gpr_quarterly_path; input_paths=[GPR_PATH])
println("GPR (monthly + quarter-end-month) persisted and registered as gpr_ts.")

# ============================================================
# (1) BUILD MATCHED-EUROPEAN-FIRM CROSSWALK (CUSIP-first).
# FactSet Ownership stores identifiers for European firms mostly in CUSIP.
# Use CUSIP first, keep ISIN/SEDOL as secondary diagnostics.
# ============================================================
println("\nBuilding CUSIP-first crosswalk: FactSet sec_entity_id → Revere company_id")

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE eu_sec_ids AS
    SELECT DISTINCT
           sec_entity_id,
           NULLIF(TRIM(cusip), '') AS cusip,
           NULLIF(TRIM(isin),  '') AS isin,
           NULLIF(TRIM(sedol), '') AS sedol
    FROM read_parquet('$EOM_PATH')
    WHERE sec_country IN $EUROPE_str
      AND sec_entity_id IS NOT NULL
""")

id_cov = qdf(con, """
    SELECT
        COUNT(DISTINCT sec_entity_id) AS n_eu_sec_total,
        COUNT(DISTINCT CASE WHEN cusip IS NOT NULL THEN sec_entity_id END) AS n_eu_sec_with_cusip,
        COUNT(DISTINCT CASE WHEN isin  IS NOT NULL THEN sec_entity_id END) AS n_eu_sec_with_isin,
        COUNT(DISTINCT CASE WHEN sedol IS NOT NULL THEN sec_entity_id END) AS n_eu_sec_with_sedol,
        COUNT(*) AS n_sec_id_rows
    FROM eu_sec_ids
""")
println("  EU identifier coverage in holdings:")
println(id_cov)

# Crosswalk uses the latest-snapshot universe IDs (CUSIP/ISIN/SEDOL).
# Per audit C5: ID stability over the panel is assumed for these joiners
# (the time-versioned EU MEMBERSHIP filter below is what actually closes the
# look-ahead loop).
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE matched_eu_sec_links AS
    WITH univ AS (
        SELECT eu_company_id,
               NULLIF(TRIM(eu_cusip), '') AS eu_cusip,
               NULLIF(TRIM(eu_isin),  '') AS eu_isin,
               NULLIF(TRIM(eu_sedol), '') AS eu_sedol
        FROM read_parquet('$UNIV_PATH')
    )
    SELECT DISTINCT s.sec_entity_id, u.eu_company_id,
           'CUSIP' AS match_type, s.cusip AS matched_id
    FROM eu_sec_ids s
    JOIN univ u ON s.cusip = u.eu_cusip
    WHERE s.cusip IS NOT NULL AND u.eu_cusip IS NOT NULL

    UNION ALL

    SELECT DISTINCT s.sec_entity_id, u.eu_company_id,
           'ISIN_EXACT' AS match_type, s.isin AS matched_id
    FROM eu_sec_ids s
    JOIN univ u ON s.isin = u.eu_isin
    WHERE s.isin IS NOT NULL AND u.eu_isin IS NOT NULL

    UNION ALL

    SELECT DISTINCT s.sec_entity_id, u.eu_company_id,
           'SEDOL' AS match_type, s.sedol AS matched_id
    FROM eu_sec_ids s
    JOIN univ u ON s.sedol = u.eu_sedol
    WHERE s.sedol IS NOT NULL AND u.eu_sedol IS NOT NULL
""")

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE matched_eu_sec AS
    SELECT DISTINCT sec_entity_id FROM matched_eu_sec_links
""")

n_matched = qdf(con, "SELECT COUNT(*) AS n FROM matched_eu_sec").n[1]
n_eu_sec_total = id_cov.n_eu_sec_total[1]
println("  Matched to Revere European universe: $n_matched / $n_eu_sec_total = $(round(100*n_matched/n_eu_sec_total, digits=1))% of EU sec_entity_ids")

# Profile unmatched per audit: characterize the dropped firms.
unmatched_profile = qdf(con, """
    WITH all_eu AS (SELECT DISTINCT sec_entity_id, sec_country FROM read_parquet('$EOM_PATH') WHERE sec_country IN $EUROPE_str)
    SELECT a.sec_country,
           COUNT(*) AS n_all,
           COUNT(*) FILTER (WHERE m.sec_entity_id IS NULL) AS n_unmatched,
           ROUND(100.0 * COUNT(*) FILTER (WHERE m.sec_entity_id IS NULL) / COUNT(*), 2) AS pct_unmatched
    FROM all_eu a
    LEFT JOIN matched_eu_sec m USING (sec_entity_id)
    GROUP BY a.sec_country
    ORDER BY n_unmatched DESC
""")
CSV.write(joinpath(OUT_DIR, "05_unmatched_profile_by_country.csv"), unmatched_profile)
println("  Unmatched profile by country -> 05_unmatched_profile_by_country.csv")

match_type = qdf(con, """
    SELECT match_type,
           COUNT(DISTINCT sec_entity_id) AS n_sec_entities,
           COUNT(*) AS n_links
    FROM matched_eu_sec_links
    GROUP BY match_type ORDER BY n_sec_entities DESC
""")
CSV.write(joinpath(OUT_DIR, "05_match_type_distribution.csv"), match_type)

multi_match = qdf(con, """
    SELECT n_revere_companies, COUNT(*) AS n_sec_entities
    FROM (
        SELECT sec_entity_id, COUNT(DISTINCT eu_company_id) AS n_revere_companies
        FROM matched_eu_sec_links GROUP BY sec_entity_id
    )
    GROUP BY n_revere_companies ORDER BY n_revere_companies
""")
CSV.write(joinpath(OUT_DIR, "05_multi_match_per_sec_entity.csv"), multi_match)

# ============================================================
# (1a) PRE-AGGREGATE exposure to (sec_entity_id × quarter_end) using
# SELF-CONSISTENT ratio aggregation: SUM(n_cn_total)/SUM(n_total_links).
# Previous version used MAX which (1) is the upper bound, not "conservative",
# and (2) takes numerator and denominator from potentially different sub-entities.
# DISTINCT applied to crosswalk subquery (mirrors exposure_by_sec_entity).
# ============================================================
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE exposure_by_sec_entity AS
    WITH crosswalk AS (
        SELECT DISTINCT sec_entity_id, eu_company_id FROM matched_eu_sec_links
    )
    SELECT m.sec_entity_id,
           e.quarter_end,
           SUM(e.n_cn_total)         AS n_cn_total,
           SUM(e.n_cn_customer)      AS n_cn_customer,
           SUM(e.n_cn_supplier)      AS n_cn_supplier,
           SUM(e.n_cn_sell)          AS n_cn_sell,
           SUM(e.n_cn_buy)           AS n_cn_buy,
           SUM(e.n_cn_jv)            AS n_cn_jv,
           SUM(e.n_total_links)      AS n_total_links,
           SUM(e.n_supplychain_links) AS n_supplychain_links,
           -- (B7 FIX, 2026-07-22) supply-chain china_share: CN CUSTOMER+SUPPLIER
           -- links / total CUSTOMER+SUPPLIER links, SUM-aggregated across
           -- multi-matched Revere companies (self-consistent ratio, matches 02).
           -- The old all-rel-type ratio is kept as china_share_alltypes.
           (SUM(e.n_cn_customer) + SUM(e.n_cn_supplier))::DOUBLE
               / NULLIF(SUM(e.n_supplychain_links), 0) AS china_share,
           -- (DIRECTION FIX, 2026-08-02) directional link-count shares over the
           -- same denominator; sell + buy = china_share row-wise (see 02).
           -- NOTE: descriptive-parity only in 05 — the REGRESSION path for the
           -- direction split is 02(parquet) -> 06 -> build_c6; 06 rebuilds its
           -- own exposure_by_sec_entity and does not read this table.
           SUM(e.n_cn_sell)::DOUBLE / NULLIF(SUM(e.n_supplychain_links), 0)
               AS china_sell_link_share,
           SUM(e.n_cn_buy)::DOUBLE / NULLIF(SUM(e.n_supplychain_links), 0)
               AS china_buy_link_share,
           SUM(e.n_cn_total)::DOUBLE / NULLIF(SUM(e.n_total_links), 0)
               AS china_share_alltypes
    FROM crosswalk m
    JOIN read_parquet('$EXP_PATH') e ON m.eu_company_id = e.eu_company_id
    GROUP BY m.sec_entity_id, e.quarter_end
""")
n_exp_se = qdf(con, "SELECT COUNT(*) AS n FROM exposure_by_sec_entity").n[1]
println("  Pre-aggregated exposure rows (sec_entity_id × quarter, SUM-based): $n_exp_se")

# Time-versioned EU membership at sec_entity level. A sec_entity is included
# in the merged panel for quarter q ONLY if at least one matched Revere
# company was an EU firm AS-OF q. This propagates the C5 fix from 02.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE matched_eu_sec_qend AS
    SELECT DISTINCT m.sec_entity_id, u.qend AS quarter_end
    FROM matched_eu_sec_links m
    JOIN read_parquet('$UNIV_QEND_PATH') u
      ON m.eu_company_id = u.eu_company_id
""")
n_se_qend = qdf(con, "SELECT COUNT(*) AS n FROM matched_eu_sec_qend").n[1]
println("  Time-versioned (sec_entity × quarter) membership rows: $n_se_qend")

# Coverage cascade
coverage = qdf(con, """
    WITH all_eu_sec AS (
        SELECT DISTINCT sec_entity_id FROM read_parquet('$EOM_PATH')
        WHERE sec_country IN $EUROPE_str
    )
    SELECT
        (SELECT COUNT(*) FROM all_eu_sec) AS n_eu_sec_total,
        (SELECT COUNT(DISTINCT CASE WHEN cusip IS NOT NULL THEN sec_entity_id END) FROM eu_sec_ids) AS n_eu_sec_with_cusip,
        (SELECT COUNT(DISTINCT CASE WHEN isin IS NOT NULL THEN sec_entity_id END) FROM eu_sec_ids) AS n_eu_sec_with_isin,
        (SELECT COUNT(DISTINCT CASE WHEN sedol IS NOT NULL THEN sec_entity_id END) FROM eu_sec_ids) AS n_eu_sec_with_sedol,
        (SELECT COUNT(DISTINCT sec_entity_id) FROM matched_eu_sec) AS n_eu_sec_matched
""")
println("\nCoverage cascade:")
println(coverage)
CSV.write(joinpath(OUT_DIR, "05_coverage_cascade.csv"), coverage)

# ============================================================
# (2) BUILD MERGED PANEL
# Time-versioned EU membership filter (matched_eu_sec_qend) closes C5 from 02.
# Exposure joined via the SUM-aggregated table.
# C2 fix: pre-2003 quarters carry NULL china_share / n_cn_* (NOT zero).
# C3 fix: LAG(china_share) over (sec_entity_id, report_date) → china_share_lag1q.
# C1 fix: regression weight is portfolio_weight_eu (from 04), not _global.
# ============================================================
println("\nBuilding merged panel (US-investor × matched-EU-AS-OF-q × quarter)...")

merged_path = test_suffix_path(joinpath(OUT_DIR, "merged_us_eu_matched.parquet"))

atomic_copy_to(con, """
    WITH ownership_matched AS (
        SELECT o.sec_entity_id, o.sec_country, o.investor_country, o.report_date,
               o.I_ict, o.market_cap, o.ownership_share,
               o.portfolio_weight_eu      AS portfolio_weight_eu,
               o.portfolio_weight_global  AS portfolio_weight_global
        FROM read_parquet('$OWN_PATH') o
        JOIN matched_eu_sec_qend mq
          ON mq.sec_entity_id = o.sec_entity_id
         AND mq.quarter_end   = o.report_date
        WHERE o.sec_country IN $EUROPE_str
    ),
    base AS (
        SELECT
            om.sec_entity_id,
            om.sec_country,
            om.investor_country,
            om.report_date,
            om.I_ict,
            om.market_cap,
            om.ownership_share,
            om.portfolio_weight_eu,
            om.portfolio_weight_global,
            -- C2 fix: pre-2003-Q1 firm-quarters carry NULL exposure (NOT zero).
            CASE WHEN om.report_date < DATE '2003-03-31' THEN NULL
                 ELSE e.n_cn_total END         AS n_cn_total_raw,
            CASE WHEN om.report_date < DATE '2003-03-31' THEN NULL
                 ELSE e.n_cn_customer END      AS n_cn_customer_raw,
            CASE WHEN om.report_date < DATE '2003-03-31' THEN NULL
                 ELSE e.n_cn_supplier END      AS n_cn_supplier_raw,
            CASE WHEN om.report_date < DATE '2003-03-31' THEN NULL
                 ELSE e.n_cn_jv END            AS n_cn_jv_raw,
            CASE WHEN om.report_date < DATE '2003-03-31' THEN NULL
                 ELSE e.n_total_links END      AS n_total_links_raw,
            CASE WHEN om.report_date < DATE '2003-03-31' THEN NULL
                 ELSE e.china_share END        AS china_share_raw,
            (om.report_date >= DATE '2003-03-31'
             AND e.quarter_end IS NOT NULL)    AS in_revere_coverage,
            g.gpr_us_cn,
            g.gpr_global,
            g.shock_us_cn
        FROM ownership_matched om
        LEFT JOIN exposure_by_sec_entity e
               ON om.sec_entity_id = e.sec_entity_id
              AND om.report_date   = e.quarter_end
        LEFT JOIN gpr_ts g ON om.report_date = g.quarter_end
    )
    SELECT
        sec_entity_id,
        sec_country,
        investor_country,
        report_date,
        I_ict,
        market_cap,
        ownership_share,
        portfolio_weight_eu,
        portfolio_weight_global,
        n_cn_total_raw     AS n_cn_total,
        n_cn_customer_raw  AS n_cn_customer,
        n_cn_supplier_raw  AS n_cn_supplier,
        n_cn_jv_raw        AS n_cn_jv,
        n_total_links_raw  AS n_total_links,
        china_share_raw    AS china_share,
        -- C3 FIX: regression-spec ChinaExposure_{i,t-1} as a lagged column.
        LAG(china_share_raw)   OVER (PARTITION BY sec_entity_id, investor_country
                                      ORDER BY report_date) AS china_share_lag1q,
        LAG(n_cn_total_raw)    OVER (PARTITION BY sec_entity_id, investor_country
                                      ORDER BY report_date) AS n_cn_total_lag1q,
        (n_cn_total_raw > 0)   AS has_cn_exposure,
        in_revere_coverage,
        gpr_us_cn,
        gpr_global,
        shock_us_cn
    FROM base
""", merged_path)
merged_path_fwd = replace(merged_path, "\\" => "/")

n_merged = qdf(con, "SELECT COUNT(*) AS n FROM read_parquet('$merged_path_fwd')").n[1]
println("  merged panel rows (matched, EU-AS-OF-q): $n_merged")
write_manifest("05_merged_us_eu_matched", merged_path; row_count=n_merged, input_paths=STEP_INPUTS)

# Composition diagnostics
comp = qdf(con, """
    SELECT investor_country = 'US' AS is_us,
           in_revere_coverage,
           has_cn_exposure,
           COUNT(*) AS n
    FROM read_parquet('$merged_path_fwd')
    GROUP BY 1,2,3 ORDER BY 1 DESC, 2 DESC, 3 DESC
""")
println("\nMerged panel composition (US / in-coverage / has-cn):")
println(comp)
CSV.write(joinpath(OUT_DIR, "05_merged_panel_composition.csv"), comp)

# ============================================================
# (2b) NONUS AGGREGATE — EU-restricted AND matched-only (C1 fix v2).
# Old version used a global denominator (sum of all non-US holdings worldwide).
# v1 of the fix restricted to EU sec_country but kept the denominator at
# all-EU; verifier flagged that the bucket numerator was matched-only while
# the denominator was all-EU, breaking apples-to-apples comparability with the
# matched-only US numerator. Fix v2: BOTH numerator and denominator restricted
# to the matched_eu_sec_qend universe — the SAME scope as the US side uses
# via merged_path_fwd. Result: bucket within_europe_share now sums to 1 on
# both sides per quarter, and the cross-holder comparison is clean.
# ============================================================
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE nonus_aggregate_eu AS
    WITH matched AS (
        SELECT DISTINCT sec_entity_id, quarter_end FROM matched_eu_sec_qend
    ),
    per_firm AS (
        SELECT i.sec_entity_id, i.report_date,
               SUM(i.I_ict) AS nonus_I
        FROM read_parquet('$ICT_PATH') i
        JOIN matched m
          ON m.sec_entity_id = i.sec_entity_id
         AND m.quarter_end   = i.report_date
        WHERE i.investor_country != 'US' AND i.investor_country IS NOT NULL
          AND i.sec_country IN $EUROPE_str
        GROUP BY i.sec_entity_id, i.report_date
    ),
    per_quarter AS (
        SELECT i.report_date,
               SUM(i.I_ict) AS nonus_total_eu
        FROM read_parquet('$ICT_PATH') i
        JOIN matched m
          ON m.sec_entity_id = i.sec_entity_id
         AND m.quarter_end   = i.report_date
        WHERE i.investor_country != 'US' AND i.investor_country IS NOT NULL
          AND i.sec_country IN $EUROPE_str
        GROUP BY i.report_date
    )
    SELECT p.sec_entity_id, p.report_date,
           p.nonus_I,
           q.nonus_total_eu,
           p.nonus_I / NULLIF(q.nonus_total_eu, 0) AS nonus_portfolio_weight_eu
    FROM per_firm p
    JOIN per_quarter q USING (report_date)
""")
nonus_diag = qdf(con, """
    SELECT
        COUNT(*) AS n_quarters,
        MIN(sum_w_eu) AS min_sum,
        MAX(sum_w_eu) AS max_sum,
        MAX(ABS(sum_w_eu - 1.0)) AS max_abs_dev_from_1
    FROM (
        SELECT report_date, SUM(nonus_portfolio_weight_eu) AS sum_w_eu
        FROM nonus_aggregate_eu
        GROUP BY report_date
    )
""")
println("\nNONUS-EU aggregate sanity check (sum_w_eu should be ≈ 1 per quarter, ALL quarters):")
println(nonus_diag)
if nonus_diag.max_abs_dev_from_1[1] > 1e-6
    @warn "nonus_aggregate_eu does not sum to 1 within tolerance" max_abs_dev=nonus_diag.max_abs_dev_from_1[1]
end

# ============================================================
# (A) SCATTER: US ownership vs ChinaExposure (snapshot)
# Uses CHINA_SHARE_LAG1Q for the descriptive analogue of the regression
# specification (ChinaExposure_{i,t-1}).
# ============================================================
println("\n(A) Scatter: US ownership vs CN exposure (snapshot, LAGGED china_share)")

snap_date_q = qdf(con, """
    SELECT MAX(report_date) AS d
    FROM read_parquet('$merged_path_fwd')
    WHERE report_date <= DATE '2018-12-31'
""")
snap_date = snap_date_q.d[1]
if snap_date === missing || snap_date === nothing
    snap_date_q = qdf(con, "SELECT MAX(report_date) AS d FROM read_parquet('$merged_path_fwd')")
    snap_date = snap_date_q.d[1]
end
println("  using snapshot date: $snap_date")

scatter_df = qdf(con, """
    SELECT us_ownership_share, n_cn_total, china_share, china_share_lag1q, has_cn_exposure
    FROM (
        SELECT investor_country, sec_entity_id,
               n_cn_total, china_share, china_share_lag1q, has_cn_exposure,
               ownership_share AS us_ownership_share, market_cap
        FROM read_parquet('$merged_path_fwd')
        WHERE investor_country = 'US'
          AND report_date = DATE '$snap_date'
          AND market_cap > 0
          AND ownership_share IS NOT NULL
          AND ownership_share BETWEEN 0 AND 1
    )
""")
println("  cells: $(nrow(scatter_df))")
if nrow(scatter_df) > 10
    if hasproperty(scatter_df, :china_share_lag1q)
        keep = .!ismissing.(scatter_df.china_share_lag1q)
        if sum(keep) > 10
            println("  Correlation(US ownership, china_share_lag1q): ",
                    round(cor(scatter_df.us_ownership_share[keep],
                              identity.(scatter_df.china_share_lag1q[keep])), digits=4))
        end
    end
end
CSV.write(joinpath(OUT_DIR, "05_scatter_own_vs_cn_data.csv"), scatter_df)

# ============================================================
# (B) TIME SERIES — US vs NONUS allocation by exposure group (lagged).
# Bucket built from china_share_lag1q (NOT contemporaneous) — descriptive
# analogue of ChinaExposure_{i,t-1} in the regression.
# Pre-2003 quarters drop out automatically because china_share_lag1q is NULL.
# Bucket is a firm-quarter attribute (not investor-conditional).
# ============================================================
println("\n(B) Within-Europe US allocation by exposure group (china_share_lag1q)")

ts_alloc = qdf(con, """
    WITH median_cutoff AS (
        -- Median of china_share_lag1q across firm-quarter cells WITH positive
        -- exposure. Data-driven cutoff replaces the earlier arbitrary 0.20
        -- threshold. Cells with zero or NULL exposure are excluded from the
        -- median computation but classified separately below.
        SELECT QUANTILE_CONT(china_share_lag1q, 0.5) AS med
        FROM read_parquet('$merged_path_fwd')
        WHERE china_share_lag1q > 0
    ),
    bucket AS (
        SELECT DISTINCT sec_entity_id, report_date, china_share_lag1q,
               CASE
                   WHEN china_share_lag1q IS NULL                                           THEN 'MISSING'
                   WHEN china_share_lag1q > (SELECT med FROM median_cutoff)                 THEN 'HIGH'
                   ELSE                                                                          'LOW'
               END AS exp_grp
        FROM read_parquet('$merged_path_fwd')
    ),
    us_side AS (
        SELECT m.report_date, b.exp_grp, SUM(m.portfolio_weight_eu) AS w_in_grp
        FROM read_parquet('$merged_path_fwd') m
        JOIN bucket b ON m.sec_entity_id = b.sec_entity_id AND m.report_date = b.report_date
        WHERE m.investor_country = 'US' AND m.portfolio_weight_eu IS NOT NULL
        GROUP BY m.report_date, b.exp_grp
    ),
    us_total AS (
        SELECT m.report_date, SUM(m.portfolio_weight_eu) AS w_total
        FROM read_parquet('$merged_path_fwd') m
        WHERE m.investor_country = 'US' AND m.portfolio_weight_eu IS NOT NULL
        GROUP BY m.report_date
    ),
    nonus_side AS (
        SELECT n.report_date, b.exp_grp, SUM(n.nonus_portfolio_weight_eu) AS w_in_grp
        FROM nonus_aggregate_eu n
        JOIN bucket b ON n.sec_entity_id = b.sec_entity_id AND n.report_date = b.report_date
        GROUP BY n.report_date, b.exp_grp
    ),
    nonus_total AS (
        SELECT report_date, SUM(nonus_portfolio_weight_eu) AS w_total
        FROM nonus_aggregate_eu GROUP BY report_date
    )
    SELECT u.report_date, 'US' AS investor_country, u.exp_grp,
           u.w_in_grp                              AS abs_portfolio_weight,
           u.w_in_grp / NULLIF(ut.w_total, 0)      AS within_europe_share,
           g.gpr_us_cn, g.shock_us_cn
    FROM us_side u
    LEFT JOIN us_total ut USING (report_date)
    LEFT JOIN gpr_ts g ON u.report_date = g.quarter_end
    UNION ALL
    SELECT n.report_date, 'NONUS' AS investor_country, n.exp_grp,
           n.w_in_grp                              AS abs_portfolio_weight,
           n.w_in_grp / NULLIF(nt.w_total, 0)      AS within_europe_share,
           g.gpr_us_cn, g.shock_us_cn
    FROM nonus_side n
    LEFT JOIN nonus_total nt USING (report_date)
    LEFT JOIN gpr_ts g ON n.report_date = g.quarter_end
    ORDER BY report_date, investor_country, exp_grp
""")
CSV.write(joinpath(OUT_DIR, "05_within_europe_share_by_group.csv"), ts_alloc)

# US-vs-non-US HIGH-only
hi_nonus_agg = let
    sub = filter(r -> r.investor_country == "NONUS" && r.exp_grp == "HIGH", ts_alloc)
    DataFrame(
        report_date           = sub.report_date,
        nonus_within_eu_share = sub.within_europe_share,
        nonus_abs_pw          = sub.abs_portfolio_weight,
        gpr_us_cn             = sub.gpr_us_cn,
        shock_us_cn           = sub.shock_us_cn,
    )
end
CSV.write(joinpath(OUT_DIR, "05_us_vs_nonus_high_share_data.csv"), hi_nonus_agg)

# ============================================================
# (C) DIFFERENTIAL — Δw vs GPR AR(1) shock, HIGH-LAGGED group.
# Uses portfolio_weight_eu (US) and nonus_portfolio_weight_eu (NONUS).
# Winsorize Δw at 1st/99th percentile rather than hard-trim at ±1%, so
# extreme moves (the behavior of interest) are not silently dropped.
# Bucket built from LAGGED china_share.
# ============================================================
println("\n(C) Differential (descriptive) — Δw vs GPR shock, HIGH-lag")

diff_panel_path = test_suffix_path(joinpath(OUT_DIR, "us_vs_nonus_diff.parquet"))

atomic_copy_to(con, """
    WITH us_w_t AS (
        SELECT sec_entity_id, report_date, china_share_lag1q,
               SUM(portfolio_weight_eu) AS us_w
        FROM read_parquet('$merged_path_fwd')
        WHERE investor_country = 'US' AND portfolio_weight_eu IS NOT NULL
        GROUP BY sec_entity_id, report_date, china_share_lag1q
    ),
    nonus_w_t AS (
        SELECT sec_entity_id, report_date,
               nonus_portfolio_weight_eu AS nonus_w
        FROM nonus_aggregate_eu
    ),
    firm_q AS (
        SELECT u.sec_entity_id, u.report_date, u.china_share_lag1q, u.us_w,
               COALESCE(n.nonus_w, 0) AS nonus_w
        FROM us_w_t u
        LEFT JOIN nonus_w_t n
               ON u.sec_entity_id = n.sec_entity_id
              AND u.report_date   = n.report_date
    ),
    windowed AS (
        SELECT sec_entity_id, report_date, china_share_lag1q,
               LAG(us_w,    1) OVER (PARTITION BY sec_entity_id ORDER BY report_date) AS us_w_prev,
               LEAD(us_w,   1) OVER (PARTITION BY sec_entity_id ORDER BY report_date) AS us_w_next,
               LAG(nonus_w, 1) OVER (PARTITION BY sec_entity_id ORDER BY report_date) AS nonus_w_prev,
               LEAD(nonus_w,1) OVER (PARTITION BY sec_entity_id ORDER BY report_date) AS nonus_w_next,
               DATEDIFF('month',
                        LAG(report_date,  1) OVER (PARTITION BY sec_entity_id ORDER BY report_date),
                        LEAD(report_date, 1) OVER (PARTITION BY sec_entity_id ORDER BY report_date)) AS gap_months
        FROM firm_q
    )
    SELECT sec_entity_id, report_date, china_share_lag1q, gap_months,
           (us_w_next    - us_w_prev)    AS d_us_w,
           (nonus_w_next - nonus_w_prev) AS d_nonus_w
    FROM windowed
    WHERE us_w_prev IS NOT NULL AND us_w_next IS NOT NULL
      AND nonus_w_prev IS NOT NULL AND nonus_w_next IS NOT NULL
      AND gap_months = 6
""", diff_panel_path)
diff_panel_path_fwd = replace(diff_panel_path, "\\" => "/")

# Gap-months diagnostic (per audit medium fix)
gap_diag = qdf(con, """
    WITH all_pairs AS (
        SELECT sec_entity_id, report_date,
               LAG(report_date,  1) OVER (PARTITION BY sec_entity_id ORDER BY report_date) AS prev,
               LEAD(report_date, 1) OVER (PARTITION BY sec_entity_id ORDER BY report_date) AS next,
               DATEDIFF('month',
                        LAG(report_date, 1) OVER (PARTITION BY sec_entity_id ORDER BY report_date),
                        LEAD(report_date,1) OVER (PARTITION BY sec_entity_id ORDER BY report_date)) AS gap
        FROM read_parquet('$merged_path_fwd')
        WHERE investor_country = 'US'
    )
    SELECT
        SUM(CASE WHEN prev IS NULL THEN 1 ELSE 0 END) AS n_missing_prev,
        SUM(CASE WHEN next IS NULL THEN 1 ELSE 0 END) AS n_missing_next,
        SUM(CASE WHEN gap IS NOT NULL AND gap <> 6 THEN 1 ELSE 0 END) AS n_gap_not_6,
        SUM(CASE WHEN gap = 6 THEN 1 ELSE 0 END) AS n_gap_6
    FROM all_pairs
""")
println("Δw gap_months diagnostic (firm-quarters dropped by reason):")
println(gap_diag)
CSV.write(joinpath(OUT_DIR, "05_gap_months_diagnostic.csv"), gap_diag)

# HIGH-exposure threshold: median of china_share_lag1q across firm-quarter
# cells with positive exposure. Data-driven; replaces the earlier arbitrary
# 0.20 cutoff. We compute the median from the differential panel itself so
# the cutoff is consistent with the panel used for Δw.
med_pos_q = qdf(con, """
    SELECT QUANTILE_CONT(china_share_lag1q, 0.5) AS med
    FROM read_parquet('$diff_panel_path_fwd')
    WHERE china_share_lag1q > 0
""")
high_cutoff = (nrow(med_pos_q) > 0 && !ismissing(med_pos_q.med[1])) ? med_pos_q.med[1] : 0.0
println("HIGH-exposure cutoff (median of positive china_share_lag1q): $(round(high_cutoff, digits=4))")

# Winsorize at p1/p99 instead of hard ±1% trim.
ws_bounds_q = qdf(con, """
    SELECT
        QUANTILE_CONT(d_us_w,    0.01) AS d_us_p1,
        QUANTILE_CONT(d_us_w,    0.99) AS d_us_p99,
        QUANTILE_CONT(d_nonus_w, 0.01) AS d_nonus_p1,
        QUANTILE_CONT(d_nonus_w, 0.99) AS d_nonus_p99
    FROM read_parquet('$diff_panel_path_fwd')
    WHERE china_share_lag1q > $high_cutoff
""")
d_us_p1    = ws_bounds_q.d_us_p1[1];    d_us_p99    = ws_bounds_q.d_us_p99[1]
d_nonus_p1 = ws_bounds_q.d_nonus_p1[1]; d_nonus_p99 = ws_bounds_q.d_nonus_p99[1]

ts_diff = qdf(con, """
    SELECT report_date,
           AVG(LEAST(GREATEST(d_us_w, $d_us_p1), $d_us_p99))       AS mean_d_us_ws,
           AVG(LEAST(GREATEST(d_nonus_w, $d_nonus_p1), $d_nonus_p99)) AS mean_d_nonus_ws,
           AVG(LEAST(GREATEST(d_us_w, $d_us_p1), $d_us_p99)
              - LEAST(GREATEST(d_nonus_w, $d_nonus_p1), $d_nonus_p99)) AS mean_diff_ws,
           AVG(d_us_w)                 AS mean_d_us_raw,
           AVG(d_nonus_w)              AS mean_d_nonus_raw,
           AVG(d_us_w - d_nonus_w)     AS mean_diff_raw,
           COUNT(*)                    AS n_firms,
           ANY_VALUE(g.gpr_us_cn)      AS gpr_us_cn,
           ANY_VALUE(g.shock_us_cn)    AS shock_us_cn
    FROM read_parquet('$diff_panel_path_fwd') d
    LEFT JOIN gpr_ts g ON d.report_date = g.quarter_end
    WHERE china_share_lag1q > $high_cutoff
      AND g.gpr_us_cn IS NOT NULL
    GROUP BY report_date ORDER BY report_date
""")
CSV.write(joinpath(OUT_DIR, "05_diff_us_vs_nonus_high.csv"), ts_diff)

if nrow(ts_diff) > 10
    # Use winsorized series for headline reporting; raw is also in the CSV.
    mask_ws = .!ismissing.(ts_diff.shock_us_cn)
    if sum(mask_ws) > 10
        c_shock_ws = cor(ts_diff.mean_diff_ws[mask_ws], identity.(ts_diff.shock_us_cn[mask_ws]))
        println("  cor((ΔUS − ΔnonUS)_winsor, Shock^{US-CN}) for HIGH-lag firms: $(round(c_shock_ws, digits=4))")
        println("  obs: $(nrow(ts_diff)) (winsorized at p1/p99)")
    end
end

try
    DBInterface.close!(con)
catch e
    @warn "DBInterface.close! failed" exception=e
end

println("\n========== DONE ==========")
println("Files written (CSV data; figures generated separately by plots/plot_all.jl):")
println("  merged_us_eu_matched.parquet            (DESCRIPTIVE ONLY — see C6 banner at top)")
println("  gpr_ar1_coefficients.csv                (monthly a, b — regression-ready)")
println("  gpr_monthly_with_shock.parquet")
println("  gpr_quarterly_with_shock.parquet")
println("  05_coverage_cascade.csv                 (matched vs unmatched cascade)")
println("  05_unmatched_profile_by_country.csv     (composition of dropped firms)")
println("  05_match_type_distribution.csv          (CUSIP / ISIN / SEDOL contribution)")
println("  05_multi_match_per_sec_entity.csv       (# Revere companies per sec_entity)")
println("  05_merged_panel_composition.csv         (US / coverage / has-CN breakdown)")
println("  05_within_europe_share_by_group.csv     (4-group within-Europe share, lagged bucket)")
println("  05_us_vs_nonus_high_share_data.csv      (non-US side for compare plot)")
println("  05_scatter_own_vs_cn_data.csv           (A: scatter data, lagged exposure)")
println("  05_diff_us_vs_nonus_high.csv            (C: Δw differential, HIGH-lag, winsorized)")
println("  05_gap_months_diagnostic.csv            (Δw drop reasons)")
println()
println("Interpretation notes:")
println("  * All figures are DESCRIPTIVE. Visual correlation is suggestive, not causal.")
println("  * portfolio_weight_eu (regression input) restricts denominator to EU sec_country.")
println("  * NONUS aggregate now also EU-restricted (nonus_aggregate_eu) — fair comparison.")
println("  * Pre-2003 quarters carry NULL china_share (NOT zero).")
println("  * Bucket uses china_share_LAG1Q to match the regression spec.")
println("  * Δw winsorized at p1/p99; raw mean_diff_raw also in CSV for comparison.")
println("  * C6 (selection-on-outcome universe) is NOT fixed here. The merged panel is")
println("    NOT a regression panel. See AUDIT_2026_06_01_julia_descriptive.md.")
