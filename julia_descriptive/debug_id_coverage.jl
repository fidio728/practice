# debug_id_coverage.jl
# Diagnose why so few EU sec_entity_ids match Revere via ISIN.

include("00_setup.jl")
con = dbcon()

EU_str = "('GB','DE','FR','NL','CH','IT','ES','SE','DK','NO','FI','BE','AT','IE','LU','PT','PL','CZ','HU','GR','RO')"

println("=== (1) ID coverage in SOURCE holdings (2022-01-31, EU companies) ===")
r1 = qdf(con, """
    SELECT
        COUNT(*) AS n_rows,
        COUNT(DISTINCT factset_sec_entity_id) AS n_unique_companies,
        SUM(CASE WHEN ISIN  IS NOT NULL THEN 1 ELSE 0 END) AS n_with_isin,
        SUM(CASE WHEN CUSIP IS NOT NULL THEN 1 ELSE 0 END) AS n_with_cusip,
        SUM(CASE WHEN SEDOL IS NOT NULL THEN 1 ELSE 0 END) AS n_with_sedol
    FROM read_csv_auto('E:/Data/Data/Factset Ownership/Factset_FundOwners_2022_2023.gz',
                       compression='gzip', sample_size=20000)
    WHERE SEC_FIRM_ISO_COUNTRY IN $EU_str
      AND CAST(REPORT_DATE AS DATE) = DATE '2022-01-31'
""")
println(r1)

println("\n=== (2) ID coverage at unique-company level (distinct only) ===")
r2 = qdf(con, """
    WITH eu_co AS (
        SELECT DISTINCT factset_sec_entity_id, ISIN, CUSIP, SEDOL
        FROM read_csv_auto('E:/Data/Data/Factset Ownership/Factset_FundOwners_2022_2023.gz',
                           compression='gzip', sample_size=20000)
        WHERE SEC_FIRM_ISO_COUNTRY IN $EU_str
          AND CAST(REPORT_DATE AS DATE) = DATE '2022-01-31'
    )
    SELECT
        COUNT(DISTINCT factset_sec_entity_id) AS n_companies,
        COUNT(DISTINCT CASE WHEN ISIN  IS NOT NULL THEN factset_sec_entity_id END) AS n_with_any_isin,
        COUNT(DISTINCT CASE WHEN CUSIP IS NOT NULL THEN factset_sec_entity_id END) AS n_with_any_cusip,
        COUNT(DISTINCT CASE WHEN SEDOL IS NOT NULL THEN factset_sec_entity_id END) AS n_with_any_sedol
    FROM eu_co
""")
println(r2)

println("\n=== (3) Revere ISIN sample (head 10) ===")
r3 = qdf(con, """
    SELECT isin, company_id, name FROM read_csv_auto('E:/Data/Data/Factset Revere/revere_company_wrds.csv', sample_size=-1)
    WHERE isin IS NOT NULL
    LIMIT 10
""")
println(r3)

println("\n=== (4) Sample EU FactSet ISINs from EOM panel ===")
EOM_PATH = replace(joinpath(OUT_DIR, "holdings_eom.parquet"), "\\" => "/")
r4 = qdf(con, """
    SELECT DISTINCT isin, sec_entity_id, sec_entity_name, sec_country
    FROM read_parquet('$EOM_PATH')
    WHERE sec_country IN $EU_str
      AND isin IS NOT NULL
    LIMIT 10
""")
println(r4)
