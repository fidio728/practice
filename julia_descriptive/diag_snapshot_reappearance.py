# Snapshot-grain adjudication diagnostic: are (fund, report_date) rows COMPLETE
# portfolio snapshots, and are securities carried past the fund's last in-quarter
# report genuine exits?
#
#   reappear at ~baseline rate -> last report incomplete -> per-security carry defensible
#   reappear rarely            -> genuine exits          -> fund-grain snapshot correct
#
# First run (2026-08-08, window 2018-2023): last report >=90% of quarter-max in
# 92.8% of multi-date fund-quarters (median ratio 1.0, 0.56% below 50%); carried
# securities reappear next quarter 10.76% count / 12.34% MV vs baseline 95.23% /
# 82.70% -> fund grain adopted as 03 default (DPN_SNAPSHOT_GRAIN=fund).
#
# 2026-08-09 revision (external re-review):
#   * results now WRITTEN to output/diag_snapshot_reappearance_<label>.csv
#     (one row per metric), not just printed;
#   * same-day duplicate securities deduped per (fund, fsym, date) BEFORE any
#     count — matches the production asof_select_sql tie-break grain (report
#     sizes are COUNT of DISTINCT securities, mv per pair-date = MAX(adj_mv),
#     aligning with the production ORDER BY adj_mv DESC keep);
#   * window parameterized so the early sample can be tested directly:
#       python diag_snapshot_reappearance.py 2018_2023   (default)
#       python diag_snapshot_reappearance.py 2006_2011
#       python diag_snapshot_reappearance.py 1999_2005
#     The pre-2012 window matters most: carried MV is concentrated there
#     (full-sample C2d 15.3% vs 0.41% in 2018-2023), so the "genuine exits"
#     conclusion must be shown, not extrapolated, on the early feed.
import duckdb, sys, os

LABEL = sys.argv[1] if len(sys.argv) > 1 else "2018_2023"
RAW_DIR = os.environ.get("DPN_RAW_PARQUET_DIR", "E:/Data/Data/raw_parquet")
CHUNKS = {
    "1999_2005": ["Factset_FundOwners_1999_2005.parquet"],
    "2006_2011": ["Factset_FundOwners_2006_2011.parquet"],
    "2012_2013": ["Factset_FundOwners_2012_2013.parquet"],
    "2018_2023": ["Factset_FundOwners_2018_2019.parquet",
                  "Factset_FundOwners_2020_2021.parquet",
                  "Factset_FundOwners_2022_2023.parquet"],
}
files = [f"{RAW_DIR}/{c}" for c in CHUNKS[LABEL]]
OUT_DIR = os.environ.get("DPN_OUT_DIR",
    r"C:\Users\xl\OneDrive - Universitat Ramón Llull\git\practice\julia_descriptive\output")
out_csv = os.path.join(OUT_DIR, f"diag_snapshot_reappearance_{LABEL}.csv")

con = duckdb.connect()
con.execute("SET temp_directory='E:/duckdb_diag_tmp'")
con.execute("SET max_temp_directory_size='100GB'")
con.execute("SET memory_limit='12GB'")
con.execute("SET threads=4")

# Same-day dedup FIRST: one row per (fund, security, date), mv = MAX(adj_mv)
# (production keeps the adj_mv DESC row among same-day duplicates).
con.execute(f"""
CREATE TEMP TABLE rep AS
SELECT FACTSET_FUND_ID AS fund, FSYM_ID AS sec,
       CAST(REPORT_DATE AS DATE) AS d,
       CAST(DATE_TRUNC('quarter', CAST(REPORT_DATE AS DATE)) AS DATE) AS q,
       MAX(ADJ_MV) AS mv
FROM read_parquet({files})
WHERE ADJ_MV > 0
GROUP BY 1,2,3,4
""")

con.execute("CREATE TEMP TABLE repsize AS SELECT fund, q, d, COUNT(*) AS n_secs, SUM(mv) AS mv FROM rep GROUP BY 1,2,3")
con.execute("""
CREATE TEMP TABLE fq AS
SELECT fund, q, MAX(d) AS d_max, COUNT(*) AS n_dates, MAX(n_secs) AS max_n, SUM(mv) AS mv_all
FROM repsize GROUP BY 1,2
""")
con.execute("""
CREATE TEMP TABLE fq2 AS
SELECT f.*, r.n_secs AS n_at_dmax, r.mv AS mv_at_dmax
FROM fq f JOIN repsize r ON r.fund=f.fund AND r.q=f.q AND r.d=f.d_max
""")

rows = []
def emit(section, metric, value):
    rows.append((section, metric, value))
    print(f"{section:12s} {metric:38s} {value}")

a = con.execute("""
SELECT COUNT(*), SUM(CASE WHEN n_dates>1 THEN 1 ELSE 0 END) FROM fq2""").fetchone()
emit("A_counts", "fund_quarters", a[0])
emit("A_counts", "multi_date_fund_quarters", a[1])
emit("A_counts", "pct_multi_date", round(100.0*a[1]/a[0], 2))

b = con.execute("""
SELECT 100.0*SUM(CASE WHEN n_at_dmax >= 0.9*max_n THEN 1 ELSE 0 END)/COUNT(*),
       100.0*SUM(CASE WHEN n_at_dmax <  0.5*max_n THEN 1 ELSE 0 END)/COUNT(*),
       100.0*SUM(CASE WHEN n_at_dmax <  0.1*max_n THEN 1 ELSE 0 END)/COUNT(*),
       MEDIAN(1.0*n_at_dmax/max_n)
FROM fq2 WHERE n_dates>1""").fetchone()
emit("B_lastsize", "pct_last_ge_90pct_of_max", round(b[0], 2))
emit("B_lastsize", "pct_last_lt_50pct_of_max", round(b[1], 2))
emit("B_lastsize", "pct_last_lt_10pct_of_max", round(b[2], 2))
emit("B_lastsize", "median_ratio_last_over_max", round(b[3], 4))

con.execute("""
CREATE TEMP TABLE secq AS
SELECT r.fund, r.q, r.sec, MAX(r.d) AS d_sec_last, ARG_MAX(r.mv, r.d) AS mv_last
FROM rep r GROUP BY 1,2,3
""")
con.execute("""
CREATE TEMP TABLE secq2 AS
SELECT s.*, f.d_max, CASE WHEN s.d_sec_last < f.d_max THEN 1 ELSE 0 END AS carried
FROM secq s JOIN fq2 f ON f.fund=s.fund AND f.q=s.q
""")

c = con.execute("""
SELECT SUM(carried), COUNT(*),
       100.0*SUM(CASE WHEN carried=1 THEN mv_last ELSE 0 END)/SUM(mv_last)
FROM secq2""").fetchone()
emit("C_carried", "carried_security_quarters", c[0])
emit("C_carried", "all_security_quarters", c[1])
emit("C_carried", "pct_carried_rows", round(100.0*c[0]/c[1], 2))
emit("C_carried", "pct_carried_mv", round(c[2], 2))

con.execute("CREATE TEMP TABLE nextq AS SELECT DISTINCT fund, q FROM rep")
con.execute("""
CREATE TEMP TABLE secq3 AS
SELECT s.*,
       EXISTS (SELECT 1 FROM nextq n WHERE n.fund=s.fund AND n.q = s.q + INTERVAL 3 MONTH) AS fund_reports_next,
       EXISTS (SELECT 1 FROM secq t WHERE t.fund=s.fund AND t.sec=s.sec AND t.q = s.q + INTERVAL 3 MONTH) AS sec_in_next
FROM secq2 s
""")

for carried, n, pct in con.execute("""
SELECT carried, COUNT(*), 100.0*AVG(CASE WHEN sec_in_next THEN 1.0 ELSE 0.0 END)
FROM secq3 WHERE fund_reports_next GROUP BY carried ORDER BY carried""").fetchall():
    tag = "carried" if carried == 1 else "baseline"
    emit("D_reappear", f"{tag}_n_security_quarters", n)
    emit("D_reappear", f"{tag}_pct_reappear_next_q", round(pct, 2))

for carried, pct in con.execute("""
SELECT carried, 100.0*SUM(CASE WHEN sec_in_next THEN mv_last ELSE 0 END)/SUM(mv_last)
FROM secq3 WHERE fund_reports_next GROUP BY carried ORDER BY carried""").fetchall():
    tag = "carried" if carried == 1 else "baseline"
    emit("E_reapp_mv", f"{tag}_pct_mv_reappear_next_q", round(pct, 2))

for bucket, n, pct in con.execute("""
SELECT CASE WHEN DATE_DIFF('day', d_sec_last, d_max) <= 31 THEN '01-31d'
            WHEN DATE_DIFF('day', d_sec_last, d_max) <= 62 THEN '32-62d'
            ELSE '63-91d' END, COUNT(*),
       100.0*AVG(CASE WHEN sec_in_next THEN 1.0 ELSE 0.0 END)
FROM secq3 WHERE fund_reports_next AND carried=1 GROUP BY 1 ORDER BY 1""").fetchall():
    emit("F_gapbucket", f"reappear_pct_{bucket}", round(pct, 2))
    emit("F_gapbucket", f"n_{bucket}", n)

import csv
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["section", "metric", "value", "window", "dedup", "generated_by"])
    for s, m, v in rows:
        w.writerow([s, m, v, LABEL, "per (fund,fsym,date) MAX(adj_mv)", "diag_snapshot_reappearance.py"])
print(f"\nwrote {out_csv}")
