"""partition_m8_to_5.py — partition m8 v2 jsonl by shard_for(key) % 80
into 5 files corresponding to mod-80 shards 8, 24, 40, 56, 72 (each is a
1/5 slice of mod-16=8). Output goes to /tmp/migrate/m{8,24,40,56,72}.jsonl.

Why these shard ids: 8, 24, 40, 56, 72 are the mod-80 buckets that satisfy
%16 == 8 (since 80 = 5×16). Their union covers exactly m8's items.
"""
import hashlib
import json
import os
from pathlib import Path

INPUT = Path(r"D:\anoconda\bot_paper_data\step6_5_v2_phase_done\m8.jsonl")
OUTDIR = Path("/tmp/migrate")
OUTDIR.mkdir(parents=True, exist_ok=True)

SUB_SHARDS = [8, 24, 40, 56, 72]
NUM = 80


def shard_for(norm_key, year_month, digest, timestamp, num_shards):
    s = f"{norm_key}|{year_month}|{digest}|{timestamp}"
    h = int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)
    return h % num_shards


# Open output files
fhs = {sid: open(OUTDIR / f"m{sid}_seed.jsonl", "w", encoding="utf-8") for sid in SUB_SHARDS}
counts = {sid: 0 for sid in SUB_SHARDS}
mismatch = 0
total = 0

with open(INPUT, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        try:
            key = (r["norm_key"], r["year_month"], r["digest"], r["timestamp"])
        except KeyError:
            continue
        # Sanity: should be mod-16 == 8
        s16 = shard_for(*key, 16)
        if s16 != 8:
            mismatch += 1
            continue
        s80 = shard_for(*key, NUM)
        if s80 not in fhs:
            mismatch += 1
            continue
        fhs[s80].write(line + "\n")
        counts[s80] += 1

for fh in fhs.values():
    fh.close()

print(f"INPUT: {INPUT}")
print(f"total rows: {total:,}")
print(f"mismatch (unexpected): {mismatch:,}")
print(f"")
print(f"Partition by shard_for(key)%80 (mod-16=8 ⊂ {sorted(SUB_SHARDS)}):")
for sid in SUB_SHARDS:
    p = OUTDIR / f"m{sid}_seed.jsonl"
    print(f"  m{sid:>2}  ({counts[sid]:>5,} rows)  {p}")
print()
sum_c = sum(counts.values())
print(f"sum across 5 partitions: {sum_c:,} (expected ≈ {total - mismatch:,})")
