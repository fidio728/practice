"""dry_run_step6_5.py — verify Step 6.5 queue construction without
launching any Wayback fetches.

Reads the unified Step 6 jsonl, applies the SAME selection rules as
06_5_collect_sitemap_paths_monthly.py:build_work_queue, and reports:
  - total v1 rows by category
  - eligible queue size unsharded
  - eligible queue size per shard for --num-shards 16
  - confirms sum-of-shards == unsharded
"""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

SCRIPT = Path(r"C:\Users\xl\OneDrive - Universitat Ramón Llull\bot_paper\panel_construction\06_5_collect_sitemap_paths_monthly.py")
spec = importlib.util.spec_from_file_location("step65", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

INPUT = Path(r"D:\anoconda\bot_paper_data\sitemap_panel_results_eom_primary.jsonl")

categories = Counter()
shard_eligible_unshard = 0
shard_eligible_16 = Counter()
n_total = 0

with open(INPUT, encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_total += 1
        err = r.get("error", "") or ""
        n_urls = r.get("n_urls", 0)
        had_child_errors = bool(r.get("had_child_errors"))
        url_cap_hit = bool(r.get("url_cap_hit"))
        child_truncated = bool(r.get("child_truncated"))

        is_strict_clean = (
            err == ""
            and not had_child_errors
            and not url_cap_hit
            and not child_truncated
            and n_urls > 0
        )
        is_transient_retryable = bool(err) and not m.is_deterministic_error(err)

        if is_strict_clean:
            categories["strict_clean"] += 1
            eligible = True
        elif is_transient_retryable:
            categories["transient_retryable"] += 1
            eligible = True
        elif had_child_errors or url_cap_hit or child_truncated:
            categories["partial_skipped"] += 1
            eligible = False
        elif m.is_deterministic_error(err):
            categories["det_fail_skipped"] += 1
            eligible = False
        elif err == "" and n_urls == 0:
            categories["clean_empty_skipped"] += 1
            eligible = False
        else:
            categories["other_skipped"] += 1
            eligible = False

        if not eligible:
            continue

        try:
            key = (r["norm_key"], r["year_month"], r["digest"], r["timestamp"])
        except KeyError:
            continue

        shard_eligible_unshard += 1
        s = m.shard_for(*key, 16)
        shard_eligible_16[s] += 1

print(f"INPUT: {INPUT}")
print(f"v1 total rows: {n_total:,}\n")
print("Categories:")
for k, v in categories.most_common():
    print(f"  {k:<24} {v:>10,}")

print()
print(f"Eligible (unsharded queue):       {shard_eligible_unshard:,}")
print(f"Sum of 16-shard splits:           {sum(shard_eligible_16.values()):,}")
print(f"Match: {shard_eligible_unshard == sum(shard_eligible_16.values())}")
print()
print("Per-shard distribution (--num-shards 16):")
for s in range(16):
    print(f"  m{s:<2}: {shard_eligible_16[s]:,}")
mn = min(shard_eligible_16.values()); mx = max(shard_eligible_16.values())
print(f"  min={mn:,}  max={mx:,}  spread={mx-mn:,} ({100*(mx-mn)/mn:.2f}% of min)")
