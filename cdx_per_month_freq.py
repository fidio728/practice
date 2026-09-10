"""cdx_per_month_freq.py — for each (host, year_month) in
cdx_snapshots.jsonl + cdx_snapshots_new.jsonl, count how many CDX
snapshots Wayback returned for that host-month. Output a frequency
table: how many cells have 1 snapshot, 2 snapshots, ... and the
distribution.

This counts ALL CDX candidates from the API (before any picker
filters: status, digest dedup, etc.).
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

INPUTS = [
    Path(r"D:\anoconda\bot_paper_data\cdx_snapshots.jsonl"),
    Path(r"D:\anoconda\bot_paper_data\cdx_snapshots_new.jsonl"),
]


def main():
    n_hosts = 0
    n_total_snaps = 0
    cell_count = defaultdict(int)            # (host, ym) -> raw cdx snapshot count
    cell_count_status_filtered = defaultdict(int)  # only 200-class? optional aux
    bad_ts = 0

    for path in INPUTS:
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                host = r.get("host", "")
                snaps = r.get("snapshots", []) or []
                if not host:
                    continue
                n_hosts += 1
                for s in snaps:
                    if not s or len(s) < 1:
                        continue
                    ts = s[0] or ""
                    if len(ts) < 6 or not ts[:6].isdigit():
                        bad_ts += 1
                        continue
                    ym = f"{ts[0:4]}-{ts[4:6]}"
                    cell_count[(host, ym)] += 1
                    n_total_snaps += 1

    print(f"=== CDX raw input ===")
    print(f"  hosts (lines):                         {n_hosts:,}")
    print(f"  total raw snapshots:                   {n_total_snaps:,}")
    print(f"  bad timestamps skipped:                {bad_ts:,}")
    print(f"  unique (host, year_month) cells:       {len(cell_count):,}")
    print()

    rpc = Counter(cell_count.values())
    total_cells = len(cell_count)
    multi_cells = sum(c for k, c in rpc.items() if k > 1)
    extra = sum((k - 1) * c for k, c in rpc.items() if k > 1)

    print(f"=== Cells with >1 raw CDX snapshot ===")
    print(f"  cells with 1 snapshot:                 {rpc.get(1, 0):,} "
          f"({100*rpc.get(1,0)/max(1,total_cells):.2f}%)")
    print(f"  cells with >1 snapshot:                {multi_cells:,} "
          f"({100*multi_cells/max(1,total_cells):.2f}%)")
    print(f"  extra rows beyond 1-per-cell:          {extra:,}")
    print()

    print("=== Snapshots-per-cell frequency ===")
    print("  k     n_cells    pct_of_cells")
    cum = 0
    for k in sorted(rpc.keys()):
        n = rpc[k]
        cum += n
        pct = 100.0 * n / max(1, total_cells)
        cum_pct = 100.0 * cum / max(1, total_cells)
        if k <= 20 or k % 5 == 0 or k == max(rpc.keys()):
            print(f"  {k:>3}    {n:>10,}    {pct:>6.2f}%   (cum {cum_pct:>6.2f}%)")

    # bucketed view
    buckets = [(1,1),(2,2),(3,3),(4,5),(6,10),(11,20),(21,50),(51,100),(101,10**9)]
    print()
    print("=== Bucketed view ===")
    print("  range          n_cells    pct_of_cells   total_snaps_in_bucket")
    for lo, hi in buckets:
        n_c = sum(c for k, c in rpc.items() if lo <= k <= hi)
        n_s = sum(k * c for k, c in rpc.items() if lo <= k <= hi)
        label = f"{lo}" if lo == hi else (f"{lo}+" if hi >= 10**9 else f"{lo}-{hi}")
        pct = 100.0 * n_c / max(1, total_cells)
        print(f"  {label:<14} {n_c:>10,}    {pct:>6.2f}%       {n_s:>12,}")

    # max
    if rpc:
        kmax = max(rpc.keys())
        print()
        print(f"  Max snapshots in a single (host, ym): {kmax:,}")
        # show top examples
        big = sorted(((c, hy) for hy, c in cell_count.items() if c >= max(20, kmax // 2)),
                     reverse=True)[:10]
        if big:
            print("  Top 10 host-months by candidate count:")
            for c, (h, y) in big:
                print(f"    {c:>6,}   {h:<40}  {y}")


if __name__ == "__main__":
    main()
