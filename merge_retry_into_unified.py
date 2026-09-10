"""merge_retry_into_unified.py — merge retry_output_r1_m{N}.jsonl files into
the existing unified Step 6 jsonl by rank-precedence dedup
(clean > deterministic_partial > transient_error).

A retry result REPLACES the unified entry only if its rank is strictly
higher than the existing one. So an unsuccessful retry (still transient)
does NOT overwrite an already-clean/det row.

Strips the `_retry_round` field before writing (so the unified file's row
shape stays identical to original Step 6 output, consumed downstream by
Step 6.5 / Step 7 unchanged).

Backs up the existing unified file before overwriting.
"""
import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

DETERMINISTIC_ERRORS = {"depth_limit_exceeded", "xml_parse_error", "gzip_error",
                        "archived_403", "archived_404", "archived_410",
                        "archived_5xx", "http_404", "http_410"}
RANK = {"clean": 3, "deterministic_partial": 2, "transient_error": 1}


def is_det(err):
    if not err:
        return False
    if err in DETERMINISTIC_ERRORS:
        return True
    return err.startswith("unknown_root_tag_")


def classify(r):
    e = r.get("error", "") or ""
    if e:
        if is_det(e):
            return "deterministic_partial"
        return "transient_error"
    if r.get("had_child_errors"):
        return "transient_error"
    if r.get("child_truncated") or r.get("url_cap_hit"):
        return "deterministic_partial"
    return "clean"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="D:/anoconda/bot_paper_data")
    p.add_argument("--unified", default="sitemap_panel_results_eom_primary.jsonl")
    p.add_argument("--retry-prefix", default="retry_output_r1_m")
    p.add_argument("--retry-shards", default="11,12,13,14,15")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    unified_path = data_dir / args.unified
    if not unified_path.exists():
        raise SystemExit(f"unified missing: {unified_path}")

    # Pass 1: load unified
    best = {}  # key -> (rank, line_str)
    with open(unified_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                k = (r["norm_key"], r["year_month"], r["digest"], r["timestamp"])
            except KeyError:
                continue
            best[k] = (RANK[classify(r)], line)

    n_unified = len(best)
    print(f"Loaded {n_unified:,} keys from {unified_path.name}")

    pre_counts = Counter()
    for _, line in best.values():
        try:
            pre_counts[classify(json.loads(line))] += 1
        except json.JSONDecodeError:
            pass
    print(f"  Pre-merge: clean={pre_counts['clean']:,}  "
          f"det={pre_counts['deterministic_partial']:,}  "
          f"tr={pre_counts['transient_error']:,}")
    print()

    # Pass 2: walk all retry shard outputs, replace by rank
    n_retry_lines = 0
    n_retry_keys = set()
    n_replaced = 0
    n_kept_unchanged = 0

    for sfx in [s.strip() for s in args.retry_shards.split(",") if s.strip()]:
        rpath = data_dir / f"{args.retry_prefix}{sfx}.jsonl"
        if not rpath.exists():
            print(f"WARN: missing retry file: {rpath}")
            continue
        with open(rpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_retry_lines += 1
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    k = (r["norm_key"], r["year_month"],
                         r["digest"], r["timestamp"])
                except KeyError:
                    continue
                # Drop the per-row diagnostic field so the unified file's
                # schema stays identical to the original Step 6 contract.
                r.pop("_retry_round", None)
                stripped_line = json.dumps(r)

                rank = RANK[classify(r)]
                n_retry_keys.add(k)
                prev = best.get(k)
                if prev is None:
                    # New key (shouldn't happen — retry input came from unified)
                    best[k] = (rank, stripped_line)
                    n_replaced += 1
                elif rank > prev[0]:
                    best[k] = (rank, stripped_line)
                    n_replaced += 1
                else:
                    n_kept_unchanged += 1
        print(f"  scanned {rpath.name}")

    print()
    print(f"Retry totals:")
    print(f"  total retry attempts:    {n_retry_lines:,}")
    print(f"  unique keys touched:     {len(n_retry_keys):,}")
    print(f"  unified entries upgraded: {n_replaced:,}")
    print(f"  retry attempts kept lower-rank (no change): {n_kept_unchanged:,}")
    print()

    post_counts = Counter()
    for _, line in best.values():
        try:
            post_counts[classify(json.loads(line))] += 1
        except json.JSONDecodeError:
            pass
    print(f"Post-merge: clean={post_counts['clean']:,}  "
          f"det={post_counts['deterministic_partial']:,}  "
          f"tr={post_counts['transient_error']:,}")
    print(f"Δ:          clean=+{post_counts['clean']-pre_counts['clean']:,}  "
          f"det=+{post_counts['deterministic_partial']-pre_counts['deterministic_partial']:,}  "
          f"tr={post_counts['transient_error']-pre_counts['transient_error']:+,}")
    print()

    # Backup unified
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = unified_path.with_name(unified_path.stem + f".pre_retry_r1_backup_{ts}.jsonl")
    shutil.copy(unified_path, bak)
    print(f"Backed up unified -> {bak.name}")

    # Atomic write
    tmp = unified_path.with_suffix(unified_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for k in sorted(best.keys()):
            fh.write(best[k][1] + "\n")
    os.replace(tmp, unified_path)
    print(f"OK. Updated {unified_path.name} ({sum(post_counts.values()):,} rows, "
          f"{unified_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
