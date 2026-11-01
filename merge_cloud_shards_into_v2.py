"""
merge_cloud_shards_into_v2.py — Merge 5 cloud shards into a diagnostic-only
legacy file. Does NOT touch the EOM-primary v2 file used by Step 6.5/7.

Rationale (decision 2026-04-28, option A):
  The 5 cloud shards were produced against the OLD day-15 v1 file
  (sitemap_panel_results.jsonl). Their rows carry day-15 timestamps. Under
  the new 4-tuple key contract (norm_key, year_month, digest, timestamp),
  EOM and day-15 picks differ in roughly a third of host-month cells, so
  almost none of the shards' rows would match on the 4-tuple anyway.
  Reusing them in the EOM-primary pipeline would either silently mix
  day-15 paths into an EOM panel (correctness risk) or require a complex
  coincident-match filter (B) that needs Step 2 to have run first.

  Option A: write everything to a separate legacy file. The EOM-primary
  v2 (sitemap_url_paths_v2_eom_primary.jsonl) is left untouched (created
  empty by Step 6.5 on first run). Step 6.5 refetches everything under
  the new key contract from scratch. The cost is ~5,007 clean rows of
  cloud work that we don't reuse; the gain is no day-15 contamination.

Inputs:
  D:/anoconda/bot_paper_data/shards_backup/sitemap_url_paths_v2_m{0..4}.jsonl
  (existing EOM-primary v2, if present, is read-only here for the optional
   pre-merge sanity scan)

Output:
  D:/anoconda/bot_paper_data/sitemap_url_paths_v2_legacy_shards.jsonl
        (concatenation of all valid shard lines; diagnostic only)

Strategy:
  1. Refuse to run if the legacy output already exists (idempotency guard).
     Delete it manually to force a re-merge.
  2. Stream-read each shard, filter malformed JSON lines, write valid lines
     to a tmp file (sitemap_url_paths_v2_legacy_shards_tmp.jsonl).
  3. Post-scan the tmp file: count lines, malformed, distinct 4-tuples,
     terminal-clean-with-paths counts. Compare to expected pre-scan totals.
  4. If post-scan looks sane (>= 80% of shard lines retained, all keys
     parse), atomically rename tmp -> final. Else abort and leave tmp for
     inspection.

Status reporting: per-shard line/malformed/clean counts plus the union
4-tuple count. Warns prominently that this file is NOT consumed by the
EOM-primary pipeline.
"""

import json
import os
from pathlib import Path

DATA_DIR = Path("D:/anoconda/bot_paper_data")
SHARD_DIR = DATA_DIR / "shards_backup"
SHARD_FILES = [SHARD_DIR / f"sitemap_url_paths_v2_m{i}.jsonl" for i in range(5)]

LEGACY_OUT = DATA_DIR / "sitemap_url_paths_v2_legacy_shards.jsonl"
LEGACY_TMP = DATA_DIR / "sitemap_url_paths_v2_legacy_shards_tmp.jsonl"

# Sanity reference (read-only; not modified)
EOM_PRIMARY_V2 = DATA_DIR / "sitemap_url_paths_v2_eom_primary.jsonl"


def is_terminal_clean_with_paths(row):
    return (
        not row.get("error")
        and not row.get("had_child_errors")
        and not row.get("url_cap_hit")
        and not row.get("child_truncated")
        and row.get("n_paths", row.get("n_urls", 0)) > 0
    )


def is_terminal_clean_empty(row):
    return (
        not row.get("error")
        and not row.get("had_child_errors")
        and not row.get("url_cap_hit")
        and not row.get("child_truncated")
        and row.get("n_paths", row.get("n_urls", 0)) == 0
    )


def stream_shard_to_tmp(shard_path, tmp_handle):
    """Stream a shard, append valid JSON lines to tmp_handle. Return per-shard
    stats: (n_read, n_malformed, n_appended, distinct_4tuples_set,
    n_clean_with_paths, n_clean_empty)."""
    n_read = n_bad = n_appended = 0
    keys = set()
    n_clean_paths = 0
    n_clean_empty = 0
    with open(shard_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if not stripped.strip():
                continue
            n_read += 1
            try:
                r = json.loads(stripped)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            try:
                k = (r["norm_key"], r["year_month"], r["digest"], r["timestamp"])
            except KeyError:
                n_bad += 1
                continue
            tmp_handle.write(stripped + "\n")
            n_appended += 1
            keys.add(k)
            if is_terminal_clean_with_paths(r):
                n_clean_paths += 1
            elif is_terminal_clean_empty(r):
                n_clean_empty += 1
    return n_read, n_bad, n_appended, keys, n_clean_paths, n_clean_empty


def post_scan(path):
    """Re-scan the tmp file. Returns (n_lines, n_bad, distinct_keys_set,
    n_clean_with_paths)."""
    n_lines = n_bad = n_clean_paths = 0
    keys = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            n_lines += 1
            try:
                r = json.loads(stripped)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            try:
                k = (r["norm_key"], r["year_month"], r["digest"], r["timestamp"])
            except KeyError:
                n_bad += 1
                continue
            keys.add(k)
            if is_terminal_clean_with_paths(r):
                n_clean_paths += 1
    return n_lines, n_bad, keys, n_clean_paths


def main():
    missing = [p for p in SHARD_FILES if not p.exists()]
    if missing:
        print("ERROR: missing shard files:")
        for p in missing:
            print(f"  {p}")
        return 1

    if LEGACY_OUT.exists():
        print(f"REFUSING to re-run: legacy output already exists at")
        print(f"  {LEGACY_OUT}")
        print(f"This means the merge has already been performed. To force a")
        print(f"re-merge, delete the legacy file manually first.")
        return 2

    if LEGACY_TMP.exists():
        print(f"WARN: stale tmp file found at {LEGACY_TMP}")
        print(f"  Removing it before this run.")
        LEGACY_TMP.unlink()

    print("=== Merging 5 cloud shards into LEGACY diagnostic file ===")
    print(f"  Output (diagnostic only): {LEGACY_OUT}")
    print(f"  This file is NOT consumed by Step 6.5 or Step 7.")
    print(f"  EOM-primary v2: {EOM_PRIMARY_V2}")
    if EOM_PRIMARY_V2.exists():
        n_eom = sum(1 for _ in open(EOM_PRIMARY_V2, "r", encoding="utf-8"))
        print(f"    (already exists, {n_eom:,} lines — left untouched)")
    else:
        print(f"    (does not exist yet — Step 6.5 will create it on first run)")
    print()

    # Stream shards into tmp file
    pre_total_lines = 0
    pre_total_bad = 0
    pre_total_appended = 0
    pre_total_clean_paths = 0
    pre_total_clean_empty = 0
    union_keys = set()
    with open(LEGACY_TMP, "w", encoding="utf-8") as tmp_handle:
        for shard in SHARD_FILES:
            n_read, n_bad, n_app, sk, n_cp, n_ce = stream_shard_to_tmp(
                shard, tmp_handle)
            pre_total_lines += n_read
            pre_total_bad += n_bad
            pre_total_appended += n_app
            pre_total_clean_paths += n_cp
            pre_total_clean_empty += n_ce
            union_keys |= sk
            print(f"  {shard.name}: read={n_read:,}  malformed={n_bad:,}  "
                  f"appended={n_app:,}  4tuples={len(sk):,}  "
                  f"clean_with_paths={n_cp:,}  clean_empty={n_ce:,}")

    print()
    print(f"=== Pre-scan totals (during stream) ===")
    print(f"  shard lines read:        {pre_total_lines:,}")
    print(f"  malformed (skipped):     {pre_total_bad:,}")
    print(f"  appended to tmp:         {pre_total_appended:,}")
    print(f"  union of 4-tuples:       {len(union_keys):,}")
    print(f"  clean_with_paths rows:   {pre_total_clean_paths:,}")
    print(f"  clean_empty rows:        {pre_total_clean_empty:,}")
    print()

    # Post-scan: re-read tmp file independently
    print(f"=== Post-scan (independent verification) ===")
    post_lines, post_bad, post_keys, post_clean_paths = post_scan(LEGACY_TMP)
    print(f"  tmp file lines:          {post_lines:,}")
    print(f"  malformed in tmp:        {post_bad:,}")
    print(f"  distinct 4-tuples:       {len(post_keys):,}")
    print(f"  clean_with_paths rows:   {post_clean_paths:,}")
    print()

    # Sanity: tmp lines must equal appended count, and >=80% of shard lines
    # should have made it through.
    if post_lines != pre_total_appended:
        print(f"ABORT: tmp line count {post_lines:,} != appended {pre_total_appended:,}")
        print(f"  tmp file left at {LEGACY_TMP} for inspection.")
        return 3
    if post_bad != 0:
        print(f"ABORT: post-scan found {post_bad} malformed rows in tmp.")
        print(f"  tmp file left at {LEGACY_TMP} for inspection.")
        return 4
    if pre_total_lines > 0 and post_lines / pre_total_lines < 0.80:
        ratio = post_lines / pre_total_lines
        print(f"ABORT: only {100*ratio:.1f}% of shard lines retained "
              f"(threshold 80%).")
        print(f"  tmp file left at {LEGACY_TMP} for inspection.")
        return 5

    # Atomic replace
    os.replace(LEGACY_TMP, LEGACY_OUT)
    print(f"OK. Atomic replace complete.")
    print(f"  Final file: {LEGACY_OUT}")
    print(f"  Size: {LEGACY_OUT.stat().st_size:,} bytes")
    print()
    print(f"REMINDER: this file is DIAGNOSTIC ONLY and is NOT consumed by")
    print(f"Step 6.5 or Step 7. The EOM-primary pipeline rebuilds")
    print(f"sitemap_url_paths_v2_eom_primary.jsonl from scratch under the")
    print(f"4-tuple key contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
