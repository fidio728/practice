"""
split_step6_partial_by_shard.py — Partition the existing unsharded Step 6
output (`sitemap_panel_results_eom_primary.jsonl`) into N shard files so
each cloud machine can resume against its own shard's checkpoint without
seeing rows that belong to other shards.

Why this is necessary:
  The shard key in 06_collect_sitemaps_for_panel.py is
    md5(f"{norm_key}|{year_month}|{digest}|{timestamp}") % num_shards
  If you naively renamed the existing unsharded file to `_m0.jsonl`, then:
    - m0 would correctly skip all clean/det_partial rows for shard 0,
    - m1..m4 would start from empty checkpoints and re-fetch every clean
      row that happens to belong to their shard (massive redundant work),
    - the final post-merge file could end up with duplicate 4-tuples.
  Splitting by the SAME hash function keeps every shard's checkpoint
  honest.

CLI:
  --data-dir PATH             defaults to D:/anoconda/bot_paper_data
  --num-shards M              defaults to 5
  --input  FILENAME           defaults to sitemap_panel_results_eom_primary.jsonl
  --output-prefix FILENAME    defaults to sitemap_panel_results_eom_primary
                              (output files are <prefix>_m{0..M-1}.jsonl)

Strategy:
  1. Refuse if any output shard file already exists (idempotency guard).
  2. Stream-read input; for each valid JSON line with a 4-tuple, compute
     shard_for(...) and append the line VERBATIM to the corresponding
     <prefix>_m{shard}_tmp.jsonl file. Malformed lines counted and skipped.
  3. Post-scan each tmp file: count lines, distinct 4-tuples.
  4. If pre/post line counts reconcile, atomically rename tmp -> final
     (one shard at a time). Else abort, leave tmp files for inspection.

Does not delete the original input file. Caller may move it after the
split (e.g., to `sitemap_panel_results_eom_primary_pre_split_backup.jsonl`).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


def shard_for(norm_key, year_month, digest, timestamp, num_shards):
    """Must be byte-identical to 06_collect_sitemaps_for_panel.py:shard_for."""
    s = f"{norm_key}|{year_month}|{digest}|{timestamp}"
    h = int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)
    return h % num_shards


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="D:/anoconda/bot_paper_data")
    p.add_argument("--num-shards", type=int, default=5)
    p.add_argument("--input", default="sitemap_panel_results_eom_primary.jsonl")
    p.add_argument("--output-prefix", default="sitemap_panel_results_eom_primary")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"data-dir does not exist: {data_dir}")
    if args.num_shards < 2:
        raise SystemExit(f"--num-shards must be >= 2 (got {args.num_shards})")

    input_path = data_dir / args.input
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")

    final_paths = [data_dir / f"{args.output_prefix}_m{i}.jsonl"
                   for i in range(args.num_shards)]
    tmp_paths = [data_dir / f"{args.output_prefix}_m{i}_tmp.jsonl"
                 for i in range(args.num_shards)]

    # Idempotency guard
    existing = [p for p in final_paths if p.exists()]
    if existing:
        print("REFUSING to run: shard output files already exist:")
        for p in existing:
            print(f"  {p}")
        print("Delete them manually if you really want to re-split.")
        return 2

    # Wipe stale tmp files (e.g., from a prior aborted split)
    for p in tmp_paths:
        if p.exists():
            print(f"WARN: stale tmp file removed: {p}")
            p.unlink()

    print(f"=== split_step6_partial_by_shard ===")
    print(f"  input:     {input_path}")
    print(f"  num shards: {args.num_shards}")
    for p in final_paths:
        print(f"  -> {p}")
    print()

    # Pass 1: stream + dispatch lines to per-shard tmp files
    handles = [open(p, "w", encoding="utf-8") for p in tmp_paths]
    n_input_lines = 0
    n_blank = 0
    n_malformed = 0
    n_missing_keys = 0
    per_shard_lines = [0] * args.num_shards
    try:
        with open(input_path, "r", encoding="utf-8") as fin:
            for line in fin:
                stripped = line.rstrip("\n")
                if not stripped.strip():
                    n_blank += 1
                    continue
                n_input_lines += 1
                try:
                    r = json.loads(stripped)
                except json.JSONDecodeError:
                    n_malformed += 1
                    continue
                try:
                    nk = r["norm_key"]
                    ym = r["year_month"]
                    dg = r["digest"]
                    ts = r["timestamp"]
                except KeyError:
                    n_missing_keys += 1
                    continue
                s = shard_for(nk, ym, dg, ts, args.num_shards)
                handles[s].write(stripped + "\n")
                per_shard_lines[s] += 1
    finally:
        for h in handles:
            h.close()

    print(f"=== Pre-scan totals ===")
    print(f"  input lines (non-blank):     {n_input_lines:,}")
    print(f"  blank lines (skipped):       {n_blank:,}")
    print(f"  malformed JSON (skipped):    {n_malformed:,}")
    print(f"  missing 4-tuple keys (skip): {n_missing_keys:,}")
    print()
    for i, n in enumerate(per_shard_lines):
        print(f"  m{i}: {n:,} lines")
    print()

    # Sanity: routed count must equal kept input count
    kept = n_input_lines - n_malformed - n_missing_keys
    routed = sum(per_shard_lines)
    if routed != kept:
        print(f"ABORT: routed lines {routed:,} != kept lines {kept:,}")
        print(f"  tmp files left at:")
        for p in tmp_paths:
            print(f"    {p}")
        return 3

    # Pass 2: post-scan (independent re-read) for distinct 4-tuples and bad lines
    print(f"=== Post-scan (independent verification) ===")
    all_ok = True
    for i, tmp in enumerate(tmp_paths):
        n_lines = 0
        n_bad = 0
        keys = set()
        with open(tmp, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                n_lines += 1
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                try:
                    keys.add((r["norm_key"], r["year_month"],
                              r["digest"], r["timestamp"]))
                except KeyError:
                    n_bad += 1
        print(f"  m{i}: {n_lines:,} lines, {n_bad} bad, {len(keys):,} distinct 4-tuples")
        if n_bad:
            all_ok = False
        if n_lines != per_shard_lines[i]:
            print(f"    ABORT: post-scan lines != pre-scan lines for m{i}")
            all_ok = False
    print()

    if not all_ok:
        print("ABORT: post-scan inconsistencies. Tmp files left for inspection.")
        return 4

    # Atomic rename
    for tmp, final in zip(tmp_paths, final_paths):
        os.replace(tmp, final)
        print(f"  renamed {tmp.name} -> {final.name}")
    print()
    print(f"OK. {args.num_shards} shard files written under {data_dir}")
    print(f"Original input ({input_path.name}) left in place. Move/rename it")
    print(f"manually if you want to keep a pre-split backup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
