"""dedup_digest_bot_policy.py — collapse digest_bot_policy.csv to one row
per digest, keeping the best parse_status seen.

Multiple parser runs (after MemoryError-induced restarts) appended duplicate
rows for some digests. This script reads the full CSV, ranks rows by
parse_status quality (ok > too_large > uninterpretable > memory_error >
parse_error_*), keeps the best per digest, and rewrites the CSV in place
(via a sibling .dedup.csv then atomic replace).

Output: E:/bot_paper_data/digest_bot_policy.csv  (replaced; 99,745 rows expected)
"""

import csv
import os
from collections import Counter
from pathlib import Path

CSV_PATH = Path("E:/bot_paper_data/digest_bot_policy.csv")
TMP_PATH = CSV_PATH.with_suffix(".dedup.csv")

RANK = {"ok": 5, "too_large": 4, "uninterpretable": 3,
        "memory_error": 2, "parse_error": 1}


def rank_of(status):
    if status in RANK:
        return RANK[status]
    if status and status.startswith("parse_error"):
        return RANK["parse_error"]
    return 0


def main():
    print(f"Reading {CSV_PATH}")
    best_idx = {}      # digest -> best row dict
    n_rows = 0
    status_counts = Counter()
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            n_rows += 1
            d = r.get("digest")
            if not d:
                continue
            cur_rank = rank_of(r.get("parse_status", ""))
            if d in best_idx:
                prev_rank = rank_of(best_idx[d].get("parse_status", ""))
                if cur_rank > prev_rank:
                    best_idx[d] = r
            else:
                best_idx[d] = r
    for r in best_idx.values():
        status_counts[r.get("parse_status", "")] += 1
    print(f"  rows read:         {n_rows:,}")
    print(f"  unique digests:    {len(best_idx):,}")
    print(f"  duplicates dropped:{n_rows - len(best_idx):,}")

    print(f"\nWriting deduplicated CSV: {TMP_PATH}")
    with open(TMP_PATH, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for d in sorted(best_idx):
            writer.writerow(best_idx[d])

    print(f"Replacing {CSV_PATH} <- {TMP_PATH}")
    os.replace(TMP_PATH, CSV_PATH)

    print(f"\nDedup parse_status breakdown:")
    for st, c in status_counts.most_common():
        pct = 100.0 * c / max(1, len(best_idx))
        print(f"  {st:<20} {c:>10,}  ({pct:>5.2f}%)")


if __name__ == "__main__":
    main()
