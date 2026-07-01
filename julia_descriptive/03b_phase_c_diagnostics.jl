# 03b_phase_c_diagnostics.jl
# Runs Phase C of 03 (diagnostics on holdings_eom.parquet) standalone.
# Use when Phase B already wrote the parquet but Phase C didn't run
# (e.g., when 03 was killed by SIGPIPE from head/tail).

include("00_setup.jl")
con = dbcon()
eom_path = replace(joinpath(OUT_DIR, "holdings_eom.parquet"), "\\" => "/")

EU_str = "('GB','DE','FR','NL','CH','IT','ES','SE','DK','NO','FI','BE','AT','IE','LU','PT','PL','CZ','HU','GR','RO','SK','SI','BG','HR','EE','LV','LT')"

total_rows = qdf(con, "SELECT COUNT(*) AS n FROM read_parquet('$eom_path')").n[1]
println("EOM parquet rows: $total_rows")

println("\n=== Coverage by year ===")
yr = qdf(con, """
    SELECT EXTRACT(YEAR FROM report_date) AS year,
           COUNT(*) AS n_rows,
           COUNT(DISTINCT fund_id) AS n_funds,
           COUNT(DISTINCT sec_entity_id) AS n_firms
    FROM read_parquet('$eom_path')
    GROUP BY year ORDER BY year
""")
println(yr)
CSV.write(joinpath(OUT_DIR, "03_eom_coverage_by_year.csv"), yr)

println("\n=== ISSUE_TYPE — European-listed securities only ===")
eu = qdf(con, """
    SELECT issue_type, COUNT(*) AS n_rows,
           COUNT(DISTINCT sec_entity_id) AS n_companies,
           SUM(adj_mv)/1e9 AS total_mv_b
    FROM read_parquet('$eom_path')
    WHERE sec_country IN $EU_str
    GROUP BY issue_type ORDER BY n_rows DESC
""")
println(eu)
CSV.write(joinpath(OUT_DIR, "03_eom_issue_type_breakdown_europe.csv"), eu)

println("\n=== ISSUE_TYPE — US investors only (what they hold) ===")
us_inv = qdf(con, """
    SELECT issue_type, COUNT(*) AS n_rows,
           COUNT(DISTINCT sec_entity_id) AS n_companies,
           SUM(adj_mv)/1e9 AS total_mv_b
    FROM read_parquet('$eom_path')
    WHERE investor_country = 'US'
    GROUP BY issue_type ORDER BY n_rows DESC
""")
println(us_inv)
CSV.write(joinpath(OUT_DIR, "03_eom_issue_type_breakdown_us_investors.csv"), us_inv)

println("\n=== US x EU cells by month (key sample size) ===")
us_eu = qdf(con, """
    SELECT report_date,
           COUNT(*) AS n_holdings,
           COUNT(DISTINCT fund_id) AS n_us_funds,
           COUNT(DISTINCT sec_entity_id) AS n_eu_firms,
           SUM(adj_mv)/1e9 AS total_mv_b
    FROM read_parquet('$eom_path')
    WHERE investor_country = 'US' AND sec_country IN $EU_str
    GROUP BY report_date ORDER BY report_date
""")
println(us_eu)
CSV.write(joinpath(OUT_DIR, "03_us_x_eu_cells_by_month.csv"), us_eu)

DBInterface.close!(con)
println("\nDone.")
