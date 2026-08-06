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
#      crosswalk, china_share ∈ [0,1].
#  10. Diagnostic counts: n_delta_w_null_at_boundary vs n_delta_w_null_interior,
#      held-vs-zero-filled composition table.
#
# EM-CHANGE-2 (zero/missing recode, Emanuele email 2026-08-05) — applied here
# at the china_share site in (6) and propagated through (7)/(8):
#   * a firm POINT-IN-TIME present in the Revere universe at q with zero active
#     customer+supplier links gets china_share = 0 (genuine zero, joins the
#     low-exposure arm) — this includes firms whose only active links are
#     competitor/partner types;
#   * a firm absent from Revere at q (never matched, or before its Revere
#     coverage start, or after a coverage exit) keeps NULL and drops.
# The coverage-start rule itself lives in 02_china_exposure.jl (4b); this file
# inherits it through row existence in firm_quarter_china_exposure.parquet and
# hard-fails if that parquet predates the change.
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
const EXP_PATH        = replace(joinpath(OUT_DIR, test_suffix_path("firm_quarter_china_exposure.parquet")), "\\" => "/")
const UNIV_PATH       = replace(joinpath(OUT_DIR, test_suffix_path("eu_revere_universe.parquet")), "\\" => "/")
const GPR_Q_PATH      = replace(joinpath(OUT_DIR, test_suffix_path("gpr_quarterly_with_shock.parquet")), "\\" => "/")
const MCAP_PATH       = replace(joinpath(OUT_DIR, test_suffix_path("marketcap_it.parquet")), "\\" => "/")

for p in (EOM_PATH, ICT_PATH, COUNTRY_TOT_PATH, EXP_PATH, UNIV_PATH, GPR_Q_PATH, MCAP_PATH)
    isfile(replace(p, "/" => "\\")) || error("Required input missing: $p")
end

const STEP_INPUTS = [EOM_PATH, ICT_PATH, COUNTRY_TOT_PATH, EXP_PATH, UNIV_PATH, GPR_Q_PATH, MCAP_PATH]

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

# ============================================================
# (EM-FIX-9, 2026-08-06) PERSIST THE CROSSWALK.
#
# matched_eu_sec_links was a DuckDB table that died with the process. It was
# never written to disk, and neither merged_us_eu_zero_filled.parquet nor
# merged_us_eu_matched.parquet carries eu_company_id — so the ONLY bridge
# between the FactSet grain (sec_entity_id: holdings, market cap) and the Revere
# grain (eu_company_id: China link counts) evaporated at the end of every run.
#
# Two NAMED advisor deliverables need it:
#   M2 (market-cap-weighted country exposure ratio) joins firm-level CN at Revere
#      grain to market_cap at FactSet grain (marketcap_it.parquet is keyed on
#      sec_entity_id);
#   FIGURE B joins China link COUNTS (only in firm_quarter_china_exposure.parquet,
#      Revere grain) to US-investor status (only in the holdings panels, FactSet
#      grain).
# Without this file the next phase re-implements the CUSIP > ISIN > SEDOL
# priority join for the FOURTH time — the exact B7 failure mode that already
# produced three divergent china_share definitions in this codebase.
#
# Written AFTER the uniqueness assert above, so the persisted file is guaranteed
# to be 1:1 on sec_entity_id.
# ============================================================
crosswalk_path = test_suffix_path(joinpath(OUT_DIR, "crosswalk_sec_entity_revere.parquet"))
atomic_copy_to(con, "SELECT sec_entity_id, eu_company_id FROM matched_eu_sec_links",
               crosswalk_path)
write_manifest("06_crosswalk_sec_entity_revere", crosswalk_path;
               row_count=cw_n, input_paths=STEP_INPUTS)
println("  Crosswalk persisted -> $(basename(crosswalk_path)) " *
        "(1:1 on sec_entity_id; join key for M2 and Figure B)")

# (EM-CHANGE-2, 2026-08-06) The exposure parquet emitted by 02 now contains a
# row for EVERY point-in-time-present EU Revere firm-quarter, including firms
# whose only active links are competitor/partner (n_supplychain_links = 0) and
# firms with no active link at all (n_total_links = 0). Those are GENUINE
# ZEROS, not missings. So the NULLIF(n_supplychain_links, 0) guard that used
# to manufacture a NULL is replaced by an explicit CASE that returns 0.
#
# The B7 lesson applies literally here: china_share is computed at THREE
# sites (02 firm_quarter_exposure, 05 exposure_by_sec_entity, 06
# exposure_by_sec_entity). All three carry the same CASE. This site is the one
# the REGRESSION path uses (02 parquet -> 06 -> build_c6_panel.py -> Stata).
#
# After this change the ONLY way a grid cell gets china_share = NULL is:
#   (a) sec_entity_id never matched into Revere at all (no crosswalk row), or
#   (b) the matched Revere company is not PIT-present at that quarter, so 02
#       emitted no row and the LEFT JOIN in (7) yields NULL, or
#   (c) the quarter is pre-2003 (explicit NULL in (7)).
#
# Backward compatibility: a 02 parquet built BEFORE this change has no
# zero_recode_flag column. Detected here and hard-failed rather than silently
# producing a half-recoded panel.
exp_cols = Set(qdf(con, "DESCRIBE SELECT * FROM read_parquet('$EXP_PATH')").column_name)
for req in ("zero_recode_flag", "revere_pit_present", "revere_coverage_start")
    req in exp_cols || error(
        "EM-CHANGE-2: $(basename(EXP_PATH)) is missing column '$req'. " *
        "It was built by a pre-EM-CHANGE-2 version of 02_china_exposure.jl. " *
        "Re-run 02 before 06 — otherwise the zero-recode would be applied at " *
        "06 only and the panel would disagree with the parquet.")
end

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE exposure_by_sec_entity AS
    SELECT m.sec_entity_id,
           e.quarter_end,
           e.n_cn_total,
           e.n_total_links,
           e.n_supplychain_links,
           e.zero_recode_flag,
           e.revere_pit_present,
           -- (EM-FIX-9, 2026-08-06) RAW China supply-chain link COUNTS carried
           -- through to the merged panel. FIGURE B's outcome is the GROWTH in the
           -- count of China customer+supplier links; without these columns it
           -- would have to be back-derived as round(china_share *
           -- n_supplychain_links) from a float ratio, which is exactly the kind
           -- of reconstruction that silently disagrees with the source.
           e.n_cn_customer,
           e.n_cn_supplier,
           (e.n_cn_customer + e.n_cn_supplier) AS n_cn_supplychain,
           -- (B7 FIX, 2026-07-22) supply-chain china_share = CN CUSTOMER+SUPPLIER
           -- / total CUSTOMER+SUPPLIER (matches 02 and 05). Crosswalk is unique
           -- on sec_entity_id here (asserted above), so a direct ratio is exact.
           -- (EM-CHANGE-2) zero denominator on a PIT-present firm -> 0, not NULL.
           CASE WHEN e.n_supplychain_links > 0
                THEN (e.n_cn_customer + e.n_cn_supplier)::DOUBLE / e.n_supplychain_links
                ELSE 0.0 END AS china_share,
           -- (DIRECTION FIX, 2026-08-02) directional link-count shares, same
           -- denominator: sell + buy = china_share row-wise (see 02).
           CASE WHEN e.n_supplychain_links > 0
                THEN e.n_cn_sell::DOUBLE / e.n_supplychain_links
                ELSE 0.0 END AS china_sell_link_share,
           CASE WHEN e.n_supplychain_links > 0
                THEN e.n_cn_buy::DOUBLE / e.n_supplychain_links
                ELSE 0.0 END AS china_buy_link_share,
           CASE WHEN e.n_total_links > 0
                THEN e.n_cn_total::DOUBLE / e.n_total_links
                ELSE 0.0 END AS china_share_alltypes
    FROM matched_eu_sec_links m
    JOIN read_parquet('$EXP_PATH') e ON m.eu_company_id = e.eu_company_id
""")
# Patch 4 assertion: china_share ∈ [0,1]
oob = qdf(con, """
    SELECT COUNT(*) AS n FROM exposure_by_sec_entity
    WHERE china_share IS NOT NULL AND (china_share < 0 OR china_share > 1)
""")
@assert oob.n[1] == 0 "exposure_by_sec_entity has $(oob.n[1]) rows with china_share ∉ [0,1]"
println("  exposure_by_sec_entity: china_share ∈ [0,1] ✓")

# (EM-CHANGE-2) invariant: every row that survives the crosswalk join is
# PIT-present by construction in 02, so china_share must be non-NULL here.
# NULL may only be introduced later, by the LEFT JOIN onto the grid.
n_null_exp = qdf(con, "SELECT COUNT(*) AS n FROM exposure_by_sec_entity WHERE china_share IS NULL").n[1]
@assert n_null_exp == 0 "EM-CHANGE-2: $n_null_exp exposure_by_sec_entity rows carry NULL china_share; NULL must come only from grid non-match"

exp_census = qdf(con, """
    SELECT zero_recode_flag,
           CASE zero_recode_flag
                WHEN 0 THEN 'ratio_from_positive_denominator'
                WHEN 1 THEN 'recoded_zero__competitor_or_partner_only'
                ELSE        'recoded_zero__no_active_link_of_any_type' END AS label,
           COUNT(*) AS n_sec_entity_quarters,
           COUNT(DISTINCT sec_entity_id) AS n_sec_entities
    FROM exposure_by_sec_entity GROUP BY 1,2 ORDER BY 1
""")
println("  EM-CHANGE-2 recode census at sec_entity × quarter:")
println(exp_census)
CSV.write(joinpath(OUT_DIR, "06_em_zero_recode_census.csv"), exp_census)

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
                ELSE e.china_share END AS china_share,
           CASE WHEN g.quarter_end < DATE '2003-03-31' THEN NULL
                ELSE e.china_sell_link_share END AS china_sell_link_share,
           CASE WHEN g.quarter_end < DATE '2003-03-31' THEN NULL
                ELSE e.china_buy_link_share END AS china_buy_link_share,
           -- (EM-CHANGE-2) provenance carried to the regression panel so the
           -- zero arm stays auditable in Stata:
           --   n_supplychain_links  0 vs >0 (kept distinguishable, per spec)
           --   zero_recode_flag     0 ratio / 1 competitor-partner-only zero
           --                        / 2 no-link zero / NULL = not codable
           --   revere_pit_present   1 = firm was in Revere at q; NULL = absent
           CASE WHEN g.quarter_end < DATE '2003-03-31' THEN NULL
                ELSE e.n_supplychain_links END AS n_supplychain_links,
           CASE WHEN g.quarter_end < DATE '2003-03-31' THEN NULL
                ELSE e.zero_recode_flag END AS zero_recode_flag,
           CASE WHEN g.quarter_end < DATE '2003-03-31' THEN NULL
                ELSE e.revere_pit_present END AS revere_pit_present,
           -- (EM-FIX-9) raw China supply-chain link counts for Figure B.
           CASE WHEN g.quarter_end < DATE '2003-03-31' THEN NULL
                ELSE e.n_cn_customer END AS n_cn_customer,
           CASE WHEN g.quarter_end < DATE '2003-03-31' THEN NULL
                ELSE e.n_cn_supplier END AS n_cn_supplier,
           CASE WHEN g.quarter_end < DATE '2003-03-31' THEN NULL
                ELSE e.n_cn_supplychain END AS n_cn_supplychain,
           gpr.gpr_us_cn,
           gpr.shock_us_cn
    FROM cartesian_grid g
    LEFT JOIN ict_grouped i USING (sec_entity_id, holder_group, quarter_end)
    LEFT JOIN country_total_grouped ct USING (holder_group, quarter_end)
    LEFT JOIN exposure_by_sec_entity e USING (sec_entity_id, quarter_end)
    LEFT JOIN (
        SELECT DISTINCT quarter_end, gpr_us_cn, shock_us_cn
        FROM read_parquet('$GPR_Q_PATH')
    ) gpr USING (quarter_end)
""")

# ============================================================
# (8) Lag exposure + backward Δw + emit merged parquet.
#     Patch 4: holder_group AS investor_country alias for back-compat.
# ============================================================
println("\nComputing lagged exposure + backward Δw on the zero-filled panel...")
merged_path = test_suffix_path(joinpath(OUT_DIR, "merged_us_eu_zero_filled.parquet"))
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
        china_share,
        LAG(china_share, 1) OVER (PARTITION BY sec_entity_id, holder_group ORDER BY quarter_end) AS china_share_lag1q,
        china_sell_link_share,
        china_buy_link_share,
        LAG(china_sell_link_share, 1) OVER (PARTITION BY sec_entity_id, holder_group ORDER BY quarter_end) AS sell_share_lag1q,
        LAG(china_buy_link_share, 1)  OVER (PARTITION BY sec_entity_id, holder_group ORDER BY quarter_end) AS buy_share_lag1q,
        -- (EM-CHANGE-2) provenance, both contemporaneous and lagged. The
        -- REGRESSOR is china_share_lag1q, so the flag that describes it is the
        -- LAGGED one — use zero_recode_flag_lag1q to split the estimation
        -- sample into "computed ratio" vs "recoded zero" arms.
        n_supplychain_links,
        zero_recode_flag,
        revere_pit_present,
        LAG(n_supplychain_links, 1) OVER (PARTITION BY sec_entity_id, holder_group ORDER BY quarter_end) AS n_supplychain_links_lag1q,
        LAG(zero_recode_flag, 1)    OVER (PARTITION BY sec_entity_id, holder_group ORDER BY quarter_end) AS zero_recode_flag_lag1q,
        LAG(revere_pit_present, 1)  OVER (PARTITION BY sec_entity_id, holder_group ORDER BY quarter_end) AS revere_pit_present_lag1q,
        -- (EM-FIX-9) raw China supply-chain link counts + t-1 twins. FIGURE B
        -- classifies on ties at t-1 and measures link growth at t (Emanuele
        -- 35:49), and this panel IS a contiguous quarterly grid (verified: every
        -- predecessor pair is exactly 3 months apart), so these LAGs are TRUE
        -- t-1 — unlike the *_lag1q columns of merged_us_eu_matched.parquet.
        n_cn_customer,
        n_cn_supplier,
        n_cn_supplychain,
        LAG(n_cn_supplychain, 1) OVER (PARTITION BY sec_entity_id, holder_group ORDER BY quarter_end) AS n_cn_supplychain_lag1q,
        LAG(n_cn_customer, 1)    OVER (PARTITION BY sec_entity_id, holder_group ORDER BY quarter_end) AS n_cn_customer_lag1q,
        LAG(n_cn_supplier, 1)    OVER (PARTITION BY sec_entity_id, holder_group ORDER BY quarter_end) AS n_cn_supplier_lag1q,
        gpr_us_cn,
        shock_us_cn
    FROM grid_zfilled
""", merged_path)
merged_path_fwd = replace(merged_path, "\\" => "/")
n_merged = qdf(con, "SELECT COUNT(*) AS n FROM read_parquet('$merged_path_fwd')").n[1]
println("  Zero-filled panel rows: $n_merged")
write_manifest("06_merged_us_eu_zero_filled", merged_path; row_count=n_merged, input_paths=STEP_INPUTS)

# ============================================================
# (8b) EM-CHANGE-2 attribution census on the emitted panel.
# The estimation sample for the headline is
#     delta_w NOT NULL AND china_share_lag1q NOT NULL AND shock NOT NULL
# (build_c6_panel.py). This table decomposes that sample into the cells that
# existed before the recode (zero_recode_flag_lag1q = 0) and the cells the
# recode adds (= 1 or 2), so the sample-expansion effect is separable from the
# snapshot effect in the written update. Prior P0 headline N was 347,690.
# ============================================================
em_attr = qdf(con, """
    SELECT COALESCE(CAST(zero_recode_flag_lag1q AS VARCHAR), 'NULL_not_codable') AS lag_flag,
           COUNT(*) AS n_panel_rows,
           COUNT(*) FILTER (WHERE china_share_lag1q IS NOT NULL) AS n_cn_lag_nonnull,
           COUNT(*) FILTER (WHERE delta_w IS NOT NULL
                              AND china_share_lag1q IS NOT NULL
                              AND shock_us_cn IS NOT NULL) AS n_estimation_sample,
           COUNT(DISTINCT sec_entity_id) AS n_firms
    FROM read_parquet('$merged_path_fwd')
    GROUP BY 1 ORDER BY 1
""")
println("\nEM-CHANGE-2 attribution census on merged_us_eu_zero_filled:")
println(em_attr)
CSV.write(joinpath(OUT_DIR, "06_em_attribution_census.csv"), em_attr)

em_tot = qdf(con, """
    SELECT COUNT(*) FILTER (WHERE delta_w IS NOT NULL
                              AND china_share_lag1q IS NOT NULL
                              AND shock_us_cn IS NOT NULL) AS n_est_total,
           COUNT(*) FILTER (WHERE delta_w IS NOT NULL
                              AND china_share_lag1q IS NOT NULL
                              AND shock_us_cn IS NOT NULL
                              AND zero_recode_flag_lag1q = 0) AS n_est_prechange_cells,
           COUNT(DISTINCT CASE WHEN china_share_lag1q IS NOT NULL THEN sec_entity_id END) AS n_firms_codable
    FROM read_parquet('$merged_path_fwd')
""")
println("  estimation-sample rows total / of which pre-change cells / codable firms:")
println(em_tot)

# ============================================================
# (9) Patch 8: compute the HIGH-vs-LOW cutoff ONCE on firm-quarter distinct
#     rows (deduped across holder_group) and reuse as a Julia constant.
# ============================================================
med_q = qdf(con, """
    SELECT QUANTILE_CONT(china_share_lag1q, 0.5) AS med
    FROM (
        SELECT DISTINCT sec_entity_id, report_date, china_share_lag1q
        FROM read_parquet('$merged_path_fwd')
        WHERE china_share_lag1q IS NOT NULL AND china_share_lag1q > 0
    )
""")
const HIGH_CUTOFF = (nrow(med_q) > 0 && !ismissing(med_q.med[1]) && !isnan(med_q.med[1])) ? med_q.med[1] : 0.0
println("\nHIGH-exposure cutoff (firm-quarter median on positive cells): $(round(HIGH_CUTOFF, digits=4))")
@assert HIGH_CUTOFF > 0 "HIGH_CUTOFF degenerated to $HIGH_CUTOFF — no positive china_share_lag1q observations in the merged panel; check exposure_by_sec_entity"

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
           CASE WHEN china_share_lag1q IS NULL    THEN 'MISSING'
                WHEN china_share_lag1q > $HIGH_CUTOFF THEN 'HIGH'
                ELSE                                       'LOW' END AS exp_grp,
           COUNT(*) AS n
    FROM read_parquet('$merged_path_fwd')
    GROUP BY 1,2,3 ORDER BY 1,2,3
""")
CSV.write(joinpath(OUT_DIR, "06_panel_composition_c6.csv"), comp)
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
           z.china_share,
           z.china_share_lag1q,
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
CSV.write(joinpath(OUT_DIR, "05_scatter_own_vs_cn_data_c6.csv"), sc)
println("  fig10 scatter data → 05_scatter_own_vs_cn_data_c6.csv ($(nrow(sc)) rows; us_ownership_share included)")

# (10b) Bucket time series — Patch 7: bucket from DISTINCT firm-quarter,
# both holder groups share the same bucket assignment.
ts_alloc = qdf(con, """
    WITH bucket AS (
        SELECT DISTINCT sec_entity_id, report_date,
               CASE WHEN china_share_lag1q IS NULL          THEN 'MISSING'
                    WHEN china_share_lag1q > $HIGH_CUTOFF   THEN 'HIGH'
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
           gpr.gpr_us_cn, gpr.shock_us_cn
    FROM sided s
    LEFT JOIN totals t USING (report_date, holder_group)
    LEFT JOIN (SELECT DISTINCT quarter_end, gpr_us_cn, shock_us_cn FROM read_parquet('$GPR_Q_PATH')) gpr
           ON s.report_date = gpr.quarter_end
    ORDER BY s.report_date, s.holder_group, s.exp_grp
""")
CSV.write(joinpath(OUT_DIR, "05_within_europe_share_by_group_c6.csv"), ts_alloc)
println("  fig11 bucket data → 05_within_europe_share_by_group_c6.csv ($(nrow(ts_alloc)) rows)")

# (10c) Differential ΔUS - ΔnonUS for HIGH-exposure firms vs Shock.
#       Patch 6: actually winsorize mean_diff_ws at p1/p99 of the diff series.
println("\nComputing differential (with REAL winsorization)...")
# Step 1: get raw per-firm-quarter (d_us - d_nonus) on HIGH-exposure cells.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE diff_high_raw AS
    WITH us_w AS (
        SELECT sec_entity_id, report_date, delta_w AS d_us_w, china_share_lag1q
        FROM read_parquet('$merged_path_fwd')
        WHERE holder_group = 'US' AND delta_w IS NOT NULL
    ),
    nonus_w AS (
        SELECT sec_entity_id, report_date, delta_w AS d_nonus_w
        FROM read_parquet('$merged_path_fwd')
        WHERE holder_group = 'NONUS' AND delta_w IS NOT NULL
    )
    SELECT u.sec_entity_id, u.report_date, u.china_share_lag1q,
           u.d_us_w, n.d_nonus_w,
           (u.d_us_w - n.d_nonus_w) AS d_diff
    FROM us_w u JOIN nonus_w n USING (sec_entity_id, report_date)
    WHERE u.china_share_lag1q > $HIGH_CUTOFF
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
           ANY_VALUE(g.gpr_us_cn)                               AS gpr_us_cn,
           ANY_VALUE(g.shock_us_cn)                             AS shock_us_cn
    FROM diff_high_raw d
    LEFT JOIN (SELECT DISTINCT quarter_end, gpr_us_cn, shock_us_cn FROM read_parquet('$GPR_Q_PATH')) g
           ON d.report_date = g.quarter_end
    WHERE g.gpr_us_cn IS NOT NULL
    GROUP BY report_date ORDER BY report_date
""")
CSV.write(joinpath(OUT_DIR, "05_diff_us_vs_nonus_high_c6.csv"), ts_diff)
println("  fig13 differential data → 05_diff_us_vs_nonus_high_c6.csv ($(nrow(ts_diff)) rows; mean_diff_ws is REALLY winsorized)")

if nrow(ts_diff) > 10
    # Joint mask: drop rows where ANY of the three series is Missing/NaN so
    # cor() doesn't MethodError on Union{Missing,Float64}. Then disallowmissing
    # to coerce the eltype.
    mm = .!ismissing.(ts_diff.shock_us_cn) .&
         .!ismissing.(ts_diff.mean_diff)   .&
         .!ismissing.(ts_diff.mean_diff_ws)
    if sum(mm) > 10
        s  = Vector{Float64}(ts_diff.shock_us_cn[mm])
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
        WHERE z.china_share_lag1q > $HIGH_CUTOFF
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
           g.gpr_us_cn, g.shock_us_cn
    FROM wide w
    LEFT JOIN (SELECT DISTINCT quarter_end, gpr_us_cn, shock_us_cn FROM read_parquet('$GPR_Q_PATH')) g
           ON w.report_date = g.quarter_end
    ORDER BY w.report_date
""")
CSV.write(joinpath(OUT_DIR, "05_us_vs_nonus_high_share_data_c6.csv"), hi)
println("  fig12 US-vs-NONUS HIGH data → 05_us_vs_nonus_high_share_data_c6.csv ($(nrow(hi)) rows)")

try
    DBInterface.close!(con)
catch e
    @warn "DBInterface.close! failed" exception=e
end

println("\n========== 06 v2 DONE ==========")
println("Key output:")
println("  merged_us_eu_zero_filled.parquet  (REGRESSION-READY zero-filled panel,")
println("                                      holder_group AND investor_country columns)")
println("  05_scatter_own_vs_cn_data_c6.csv         (fig10 input incl us_ownership_share)")
println("  05_within_europe_share_by_group_c6.csv   (fig11 input)")
println("  05_us_vs_nonus_high_share_data_c6.csv    (fig12 input)")
println("  05_diff_us_vs_nonus_high_c6.csv          (fig13 input, mean_diff_ws REALLY winsorized)")
println("  06_panel_composition_c6.csv              (held vs zero-filled diagnostic)")
println()
println("To re-generate figures with the C6 panel:")
println("  set DPN_USE_C6=true && python plots_python.py")
