# 01_master_files.jl
# Descriptive stats for all master files (small, runs in <1 minute).
# Covers:
#   - Section 1 sample prep (security coverage)
#   - Section 2 country mapping (institutions, funds)
#   - Section 9 heterogeneity audit (pension fund candidates)

include("00_setup.jl")

con = dbcon()

# ============================================================
# (1) INSTITUTIONS MASTER
# ============================================================
println("\n========== Factset_LionShares_Institutions ==========")

insts = qdf(con, """
    SELECT * FROM read_csv_auto('$(replace(INSTITUTIONS_PATH, "\\" => "/"))',
                                 compression='gzip')
""")
println("rows: $(nrow(insts)), unique entity_id: $(length(unique(insts.FACTSET_ENTITY_ID)))")

# Country distribution
country_dist = qdf(con, """
    SELECT ISO_COUNTRY, COUNT(*) AS n
    FROM read_csv_auto('$(replace(INSTITUTIONS_PATH, "\\" => "/"))', compression='gzip')
    GROUP BY ISO_COUNTRY ORDER BY n DESC LIMIT 20
""")
CSV.write(joinpath(OUT_DIR, "01_inst_country_top20.csv"), country_dist)
println("Top investor countries -> 01_inst_country_top20.csv")
println(first(country_dist, 10))

# Manager style breakdown
style_dist = qdf(con, """
    SELECT MANAGER_STYLE, COUNT(*) AS n
    FROM read_csv_auto('$(replace(INSTITUTIONS_PATH, "\\" => "/"))', compression='gzip')
    GROUP BY MANAGER_STYLE ORDER BY n DESC
""")
CSV.write(joinpath(OUT_DIR, "01_inst_manager_style.csv"), style_dist)
println("\nManager style -> 01_inst_manager_style.csv")
println(style_dist)

# ============================================================
# (2) HETEROGENEITY AUDIT: PENSION / STATE / CONSTRAINED CANDIDATES
# ============================================================
# Section 9 — find candidate "constrained" US institutions by name pattern.
# Patterns based on US public pension fund nomenclature.
println("\n========== Section 9 Heterogeneity Audit ==========")

const PENSION_PATTERNS = [
    "PENSION", "RETIREMENT", "RETIREES",
    "CALPERS", "CALSTRS", "TEACHERS",
    "PUBLIC EMPLOYEES", "STATE EMPLOYEES", "MUNICIPAL",
    "FIREFIGHTERS", "POLICE", "JUDICIAL",
    "ENDOWMENT", "FOUNDATION"
]

# Build SQL LIKE pattern
pension_like = join(["UPPER(ENTITY_PROPER_NAME) LIKE '%$p%'" for p in PENSION_PATTERNS], " OR ")

constrained = qdf(con, """
    SELECT FACTSET_ENTITY_ID, ENTITY_PROPER_NAME, ISO_COUNTRY,
           MANAGER_STYLE, TOTAL_AUM
    FROM read_csv_auto('$(replace(INSTITUTIONS_PATH, "\\" => "/"))', compression='gzip')
    WHERE ISO_COUNTRY = 'US'
      AND ($pension_like)
    ORDER BY TOTAL_AUM DESC NULLS LAST
""")
CSV.write(joinpath(OUT_DIR, "01_us_constrained_candidates.csv"), constrained)
println("US constrained candidates: $(nrow(constrained))")
println("  -> 01_us_constrained_candidates.csv")
println("Top 20 by AUM:")
println(first(constrained, 20))

# US hedge fund candidates (unconstrained)
hedge = qdf(con, """
    SELECT FACTSET_ENTITY_ID, ENTITY_PROPER_NAME, ISO_COUNTRY,
           MANAGER_STYLE, TOTAL_AUM
    FROM read_csv_auto('$(replace(INSTITUTIONS_PATH, "\\" => "/"))', compression='gzip')
    WHERE ISO_COUNTRY = 'US'
      AND MANAGER_STYLE = 'Hedge Fund'
    ORDER BY TOTAL_AUM DESC NULLS LAST
""")
CSV.write(joinpath(OUT_DIR, "01_us_hedge_candidates.csv"), hedge)
println("\nUS hedge fund (unconstrained candidates): $(nrow(hedge))")

# ============================================================
# (3) FUNDS MASTER
# ============================================================
println("\n========== Factset_LionShares_Funds ==========")

fund_type = qdf(con, """
    SELECT FUND_TYPE, COUNT(*) AS n,
           SUM(PORTFOLIO_VALUE)/1e9 AS total_pv_billions
    FROM read_csv_auto('$(replace(FUNDS_PATH, "\\" => "/"))', compression='gzip')
    GROUP BY FUND_TYPE ORDER BY n DESC
""")
CSV.write(joinpath(OUT_DIR, "01_fund_type.csv"), fund_type)
println("Fund type breakdown -> 01_fund_type.csv")
println(fund_type)

# Pension plan funds (FUND_TYPE = PLP) — alternate constrained candidate signal
plp = qdf(con, """
    SELECT f.FACTSET_FUND_ID, f.FACTSET_ENTITY_ID, f.ENTITY_PROPER_NAME,
           f.ISO_COUNTRY, f.PORTFOLIO_VALUE
    FROM read_csv_auto('$(replace(FUNDS_PATH, "\\" => "/"))', compression='gzip') f
    WHERE f.FUND_TYPE = 'PLP' AND f.ISO_COUNTRY = 'US'
    ORDER BY f.PORTFOLIO_VALUE DESC NULLS LAST
""")
CSV.write(joinpath(OUT_DIR, "01_us_plp_funds.csv"), plp)
println("\nUS PLP (pension plan) funds: $(nrow(plp))")

# ============================================================
# (4) SECURITY COVERAGE — for Section 1 sample selection
# ============================================================
# Purpose: report how many European securities are in FactSet's master
# universe (upper bound on potential sample). NOT the same as 03's
# diagnostic, which counts securities ACTUALLY HELD by institutions.
#
# All filters are made EXPLICIT here (no silent EQ+AD or ACTIVE=1) so the
# user can see the cascade and decide where to cut.
println("\n========== Factset_Security_coverage ==========")

EU_countries = "('GB','DE','FR','NL','CH','IT','ES','SE','DK','NO','FI','BE','AT','IE','LU','PT','PL','CZ','HU','GR','RO','SK','SI','BG','HR','EE','LV','LT')"

# Full breakdown: ISSUE_TYPE × ACTIVE — see what each filter drops
eu_full = qdf(con, """
    SELECT ISSUE_TYPE, ACTIVE, COUNT(*) AS n
    FROM read_csv_auto('$(replace(SEC_COVERAGE_PATH, "\\" => "/"))', compression='gzip')
    WHERE ISO_COUNTRY IN $EU_countries
    GROUP BY ISSUE_TYPE, ACTIVE
    ORDER BY ISSUE_TYPE, ACTIVE
""")
CSV.write(joinpath(OUT_DIR, "01_eu_universe_breakdown.csv"), eu_full)
println("\nEuropean securities by ISSUE_TYPE x ACTIVE:")
println(eu_full)

# Cascade: show how the sample shrinks at each filter step
cascade = qdf(con, """
    SELECT
        COUNT(*) AS total_european,
        SUM(CASE WHEN ACTIVE = 1 THEN 1 ELSE 0 END) AS active_only,
        SUM(CASE WHEN ACTIVE = 1 AND ISSUE_TYPE IN ('EQ','AD') THEN 1 ELSE 0 END) AS active_and_eq_ad,
        SUM(CASE WHEN ACTIVE = 1 AND ISSUE_TYPE = 'EQ' THEN 1 ELSE 0 END) AS active_and_eq_only,
        SUM(CASE WHEN ACTIVE = 1 AND ISSUE_TYPE IN ('EQ','AD')
                  AND CAP_GROUP IN ('MEGA','LARGE','MID') THEN 1 ELSE 0 END) AS active_eq_ad_midplus,
        SUM(CASE WHEN ACTIVE = 1 AND ISSUE_TYPE IN ('EQ','AD')
                  AND CAP_GROUP IN ('MEGA','LARGE') THEN 1 ELSE 0 END) AS active_eq_ad_largeplus,
        SUM(CASE WHEN ACTIVE = 1 AND ISSUE_TYPE IN ('EQ','AD')
                  AND CAP_GROUP = 'MEGA' THEN 1 ELSE 0 END) AS active_eq_ad_mega
    FROM read_csv_auto('$(replace(SEC_COVERAGE_PATH, "\\" => "/"))', compression='gzip')
    WHERE ISO_COUNTRY IN $EU_countries
""")
CSV.write(joinpath(OUT_DIR, "01_eu_universe_cascade.csv"), cascade)
println("\nEuropean security-universe cascade (each col is one further filter):")
println(cascade)

# Country × cap, no ISSUE_TYPE filter, but ACTIVE=1 (most actionable cut)
eu_by_country_cap = qdf(con, """
    SELECT ISO_COUNTRY, CAP_GROUP, COUNT(*) AS n
    FROM read_csv_auto('$(replace(SEC_COVERAGE_PATH, "\\" => "/"))', compression='gzip')
    WHERE ISO_COUNTRY IN $EU_countries AND ACTIVE = 1
    GROUP BY ISO_COUNTRY, CAP_GROUP
    ORDER BY ISO_COUNTRY, CAP_GROUP
""")
CSV.write(joinpath(OUT_DIR, "01_eu_universe_by_country_cap.csv"), eu_by_country_cap)
println("\nEuropean ACTIVE securities by country x cap (no ISSUE_TYPE filter):")
println("  -> 01_eu_universe_by_country_cap.csv")
show(stdout, eu_by_country_cap; allrows=true)
println()

DBInterface.close!(con)

println("\n========== DONE ==========")
println("Outputs in $OUT_DIR")
println("Files written: 01_*.csv (7 files)")
println("\nNext: run 02_china_exposure.jl")
