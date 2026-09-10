"""
merge_step6_shards.py — Combine the M shard outputs from cloud Step 6 runs
into the unified `sitemap_panel_results_eom_primary.jsonl` consumed by
Step 6.5 / Step 7.

Naive concat is unsafe: each shard's file is append-only and may contain
multiple lines for the same 4-tuple (a transient retry followed by a
later clean). The unified file should have ONE row per 4-tuple, with
status precedence:
    clean > deterministic_partial > transient_error

Mirrors the rank semantics in 06_5's load_v2_status() and 07's strict
ingest contract.

CLI:
  --data-dir PATH             defaults to D:/anoconda/bot_paper_data
  --num-shards M              defaults to 5
  --input-prefix FILENAME     defaults to sitemap_panel_results_eom_primary
                              (reads <prefix>_m{0..M-1}.jsonl)
  --output FILENAME           defaults to sitemap_panel_results_eom_primary.jsonl

Strategy:
  1. Refuse if output already exists (idempotency).
  2. Pass 1: stream every shard, classify each row, keep best-rank row per
     4-tuple in memory.
  3. Pass 2: write winners to <output>_tmp.jsonl.
  4. Post-scan (line count, distinct 4-tuples).
  5. Atomic rename tmp -> output.
"""

import argparse
import json
import os
from pathlib import Path
from collections import Counter

DETERMINISTIC_ERRORS = {"depth_limit_exceeded", "xml_parse_error", "gzip_error",
                        "archived_403", "archived_404", "archived_410", "archived_5xx",
                        "http_404", "http_410"}

RANK = {"clean": 3, "deterministic_partial": 2, "transient_error": 1}


def is_deterministic_error(err):
    if not err:
        return False
    if err in DETERMINISTIC_ERRORS:
        return True
    if err.startswith("unknown_root_tag_"):
        return True
    return False


def classify_result_status(result):
    err = result.get("error", "") or ""
    if err:
        if is_deterministic_error(err):
            return "deterministic_partial"
        return "transient_error"
    if result.get("had_child_errors"):
        return "transient_error"
    if result.get("child_truncated") or result.get("url_cap_hit"):
        return "deterministic_partial"
    return "clean"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="D:/anoconda/bot_paper_data")
    p.add_argument("--num-shards", type=int, default=5)
    p.add_argument("--input-prefix", default="sitemap_panel_results_eom_primary")
    p.add_argument("--output", default="sitemap_panel_results_eom_primary.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"data-dir does not exist: {data_dir}")
    if args.num_shards < 2:
        raise SystemExit(f"--num-shards must be >= 2 (got {args.num_shards})")

    shard_paths = [data_dir / f"{args.input_prefix}_m{i}.jsonl"
                   for i in range(args.num_shards)]
    missing = [p for p in shard_paths if not p.exists()]
    if missing:
        print("ERROR: missing shard files:")
        for p in missing:
            print(f"  {p}")
        return 2

    output_path = data_dir / args.output
    tmp_path = data_dir / f"{args.output}.tmp"

    if output_path.exists():
        print(f"REFUSING to run: output already exists at {output_path}")
        print(f"Delete it manually if you really want to re-merge.")
        return 3
    if tmp_path.exists():
        print(f"WARN: stale tmp file removed: {tmp_path}")
        tmp_path.unlink()

    print(f"=== merge_step6_shards ===")
    print(f"  data_dir:    {data_dir}")
    print(f"  num shards:  {args.num_shards}")
    print(f"  output:      {output_path}")
    print()

    # Pass 1: scan all shards, keep best-rank row per 4-tuple
    best = {}  # 4-tuple -> (rank, line_str)
    n_input_total = 0
    n_malformed = 0
    n_missing_keys = 0
    per_shard_input = [0] * args.num_shards
    status_counts = Counter()

    for i, shard in enumerate(shard_paths):
        with open(shard, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.rstrip("\n")
                if not stripped.strip():
                    continue
                n_input_total += 1
                per_shard_input[i] += 1
                try:
                    r = json.loads(stripped)
                except json.JSONDecodeError:
                    n_malformed += 1
                    continue
                try:
                    key = (r["norm_key"], r["year_month"],
                           r["digest"], r["timestamp"])
                except KeyError:
                    n_missing_keys += 1
                    continue
                status = classify_result_status(r)
                rank = RANK.get(status, 0)
                prev = best.get(key)
                if prev is None or rank > prev[0]:
                    best[key] = (rank, stripped)
                # else: keep existing winner

        print(f"  m{i}: read {per_shard_input[i]:,} lines from {shard.name}")

    # Re-classify final winners for status breakdown
    for _, (rank, line_str) in best.items():
        try:
            r = json.loads(line_str)
        except json.JSONDecodeError:
            continue
        status_counts[classify_result_status(r)] += 1

    print(f"\n=== Pass 1 totals ===")
    print(f"  input lines across all shards: {n_input_total:,}")
    print(f"  malformed JSON skipped:        {n_malformed:,}")
    print(f"  missing 4-tuple keys skipped:  {n_missing_keys:,}")
    print(f"  unique 4-tuples after dedup:   {len(best):,}")
    print(f"  winners by status:")
    for s in ("clean", "deterministic_partial", "transient_error"):
        print(f"    {s:<24} {status_counts[s]:,}")
    print()

    # Pass 2: write winners to tmp, sorted by key for stable output
    with open(tmp_path, "w", encoding="utf-8") as fout:
        for key in sorted(best.keys()):
            fout.write(best[key][1] + "\n")

    # Post-scan
    n_out_lines = 0
    n_out_bad = 0
    out_keys = set()
    with open(tmp_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            n_out_lines += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                n_out_bad += 1
                continue
            try:
                out_keys.add((r["norm_key"], r["year_month"],
                              r["digest"], r["timestamp"]))
            except KeyError:
                n_out_bad += 1

    print(f"=== Post-scan ===")
    print(f"  output lines:         {n_out_lines:,}")
    print(f"  malformed in output:  {n_out_bad}")
    print(f"  distinct 4-tuples:    {len(out_keys):,}")
    print()

    if n_out_bad != 0 or n_out_lines != len(best) or len(out_keys) != len(best):
        print(f"ABORT: output integrity check failed.")
        print(f"  tmp file left at {tmp_path} for inspection.")
        return 4

    os.replace(tmp_path, output_path)
    print(f"OK. Atomic replace complete.")
    print(f"  Final file: {output_path}")
    print(f"  Size: {output_path.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
