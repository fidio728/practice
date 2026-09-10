"""eom_snapshot_dod_analysis.py — for each row with has_snapshot==1 in
monthly_snapshots.csv + monthly_snapshots_new.csv, parse the YYYYMMDD prefix
of snapshot_timestamp and compute days-until-end-of-month. Output a
frequency table.

Also compute: how many (host, year_month) cells have more than one row
with has_snapshot==1 (i.e. multiple within-month snapshots), and the
distribution of the count.

The script runs locally; no Wayback calls.
"""
import csv
import calendar
from collections import Counter, defaultdict
from pathlib import Path

CSVS = [
    Path(r"D:\anoconda\bot_paper_data\monthly_snapshots.csv"),
    Path(r"D:\anoconda\bot_paper_data\monthly_snapshots_new.csv"),
]


def days_until_eom(ym, day):
    """ym = 'YYYY-MM' (string), day = int. Returns days_until_eom: 0 if last
    day, 1 if 2nd-to-last, etc."""
    yy, mm = int(ym[:4]), int(ym[5:7])
    last = calendar.monthrange(yy, mm)[1]
    return last - day


def main():
    n_total = 0
    n_with_snapshot = 0
    n_bad_ts = 0
    dom_counter = Counter()       # day-of-month
    until_eom_counter = Counter()  # days from eom (0 = eom)
    cell_count = defaultdict(int)  # (host, year_month) -> count of has_snapshot==1 rows

    for csv_path in CSVS:
        if not csv_path.exists():
            print(f"MISSING: {csv_path}")
            continue
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_total += 1
                if row.get("has_snapshot", "0") != "1":
                    continue
                n_with_snapshot += 1
                ts = row.get("snapshot_timestamp", "") or ""
                if len(ts) < 8 or not ts[:8].isdigit():
                    n_bad_ts += 1
                    continue
                yy, mm, dd = int(ts[0:4]), int(ts[4:6]), int(ts[6:8])
                ym = row.get("year_month", "")
                if not ym or len(ym) < 7:
                    n_bad_ts += 1
                    continue
                # Sanity: month should match
                try:
                    last = calendar.monthrange(yy, mm)[1]
                except (ValueError, calendar.IllegalMonthError):
                    n_bad_ts += 1
                    continue
                if dd < 1 or dd > last:
                    n_bad_ts += 1
                    continue
                dom_counter[dd] += 1
                until_eom_counter[last - dd] += 1
                cell_count[(row.get("host", ""), ym)] += 1

    print(f"=== monthly_snapshots inputs (.csv + _new.csv) ===")
    print(f"  total rows:                  {n_total:,}")
    print(f"  has_snapshot==1:             {n_with_snapshot:,}")
    print(f"  unparseable timestamp:       {n_bad_ts:,}")
    print(f"  unique (host, year_month) cells with snapshot: {len(cell_count):,}")
    print()

    print("=== Days from end-of-month (0 = last day, 1 = 2nd-to-last, ...) ===")
    print("  d_until_eom    n        pct of has_snapshot")
    cum = 0
    for d in sorted(until_eom_counter.keys()):
        n = until_eom_counter[d]
        cum += n
        pct = 100.0 * n / max(1, n_with_snapshot)
        cum_pct = 100.0 * cum / max(1, n_with_snapshot)
        print(f"  {d:>3}    {n:>10,}    {pct:>6.2f}%   (cum {cum_pct:>6.2f}%)")
    print()

    print("=== Day-of-month distribution (1 = first of month) ===")
    print("  dom    n        pct")
    for d in sorted(dom_counter.keys()):
        n = dom_counter[d]
        pct = 100.0 * n / max(1, n_with_snapshot)
        print(f"  {d:>3}    {n:>10,}    {pct:>5.2f}%")
    print()

    print("=== Multiple snapshots per (host, year_month) cell ===")
    rows_per_cell = Counter(cell_count.values())
    cells_with_multi = sum(c for k, c in rows_per_cell.items() if k > 1)
    extra_rows_from_multi = sum((k - 1) * c for k, c in rows_per_cell.items() if k > 1)
    print(f"  cells with snapshot:                {len(cell_count):,}")
    print(f"  cells with >1 row:                  {cells_with_multi:,} "
          f"({100*cells_with_multi/max(1,len(cell_count)):.2f}%)")
    print(f"  extra rows beyond 1-per-cell:       {extra_rows_from_multi:,}")
    print()
    print("  rows_per_cell    n_cells")
    for k in sorted(rows_per_cell.keys()):
        print(f"  {k:>2}             {rows_per_cell[k]:,}")


if __name__ == "__main__":
    main()
