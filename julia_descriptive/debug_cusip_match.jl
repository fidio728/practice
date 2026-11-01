# debug_cusip_match.jl
# Test whether matching on CUSIP instead of ISIN gives better Revere coverage.

include("00_setup.jl")
con = dbcon()

EU_str = "('GB','DE','FR','NL','CH','IT','ES','SE','DK','NO','FI','BE','AT','IE','LU','PT','PL','CZ','HU','GR','RO')"

# Pull EU (sec_entity_id, CUSIP) directly from source holdings
println("Pulling EU sec_entity_id + CUSIP from source holdings...")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE eu_sec_cusips AS
    SELECT DISTINCT factset_sec_entity_id AS sec_entity_id, CUSIP AS cusip
    FROM read_csv_auto('E:/Data/Data/Factset Ownership/Factset_FundOwners_2022_2023.gz',
                       compression='gzip', sample_size=20000)
    WHERE SEC_FIRM_ISO_COUNTRY IN $EU_str
      AND CUSIP IS NOT NULL
""")
n = qdf(con, "SELECT COUNT(*) AS n_pairs, COUNT(DISTINCT sec_entity_id) AS n_sec FROM eu_sec_cusips")
println("EU (sec_entity_id, CUSIP) pairs: $(n.n_pairs[1]) ; distinct sec_entity_ids: $(n.n_sec[1])")

# Revere CUSIP coverage for EU companies
println("\nRevere CUSIP coverage for EU companies (home_region IN $EU_str)...")
r1 = qdf(con, """
    SELECT
        COUNT(DISTINCT company_id) AS n_eu_companies,
        COUNT(DISTINCT CASE WHEN cusip IS NOT NULL THEN company_id END) AS n_with_cusip,
        COUNT(DISTINCT CASE WHEN isin IS NOT NULL THEN company_id END) AS n_with_isin
    FROM read_csv_auto('E:/Data/Data/Factset Revere/revere_company_wrds.csv', sample_size=-1)
    WHERE home_region IN $EU_str
""")
println(r1)

# Match on CUSIP
println("\nMatching FactSet EU CUSIPs to Revere CUSIPs...")
match = qdf(con, """
    SELECT
        COUNT(DISTINCT s.sec_entity_id) AS n_matched_sec_entities
    FROM eu_sec_cusips s
    JOIN (
        SELECT DISTINCT cusip
        FROM read_csv_auto('E:/Data/Data/Factset Revere/revere_company_wrds.csv', sample_size=-1)
        WHERE cusip IS NOT NULL
    ) r ON s.cusip = r.cusip
""")
println("EU sec_entity_ids matched to Revere via CUSIP: $(match.n_matched_sec_entities[1])")

# Compare with ISIN match
println("\nFor comparison: matching ISIN[3:11] (9-char CUSIP-portion) to Revere CUSIP...")
match2 = qdf(con, """
    WITH eu_sec_isins AS (
        SELECT DISTINCT factset_sec_entity_id AS sec_entity_id,
                        SUBSTRING(ISIN, 3, 9) AS cusip_from_isin
        FROM read_csv_auto('E:/Data/Data/Factset Ownership/Factset_FundOwners_2022_2023.gz',
                           compression='gzip', sample_size=20000)
        WHERE SEC_FIRM_ISO_COUNTRY IN $EU_str
          AND ISIN IS NOT NULL
    )
    SELECT COUNT(DISTINCT s.sec_entity_id) AS n_matched
    FROM eu_sec_isins s
    JOIN (
        SELECT DISTINCT cusip
        FROM read_csv_auto('E:/Data/Data/Factset Revere/revere_company_wrds.csv', sample_size=-1)
        WHERE cusip IS NOT NULL
    ) r ON s.cusip_from_isin = r.cusip
""")
println("Matched via ISIN→CUSIP extract: $(match2.n_matched[1])")

# Sample matched cases
println("\nSample matched cases:")
sample = qdf(con, """
    SELECT s.sec_entity_id, s.cusip, r.name, r.home_region
    FROM eu_sec_cusips s
    JOIN (
        SELECT DISTINCT cusip, ANY_VALUE(name) AS name, ANY_VALUE(home_region) AS home_region
        FROM read_csv_auto('E:/Data/Data/Factset Revere/revere_company_wrds.csv', sample_size=-1)
        WHERE cusip IS NOT NULL
        GROUP BY cusip
    ) r ON s.cusip = r.cusip
    LIMIT 10
""")
println(sample)
