"""merge_digest_robots_cache.py — merge 16 shard outputs into one
deduplicated digest_robots_cache.jsonl, classify, and emit a retry queue.

Why dedup:
  Shard-0 was launched twice early on (the broken sequential launch
  partially started shards 0-3 before being killed; the parallel launch
  then started a second process for the same shards). On shards where
  this happened, the same digest can appear multiple times in the shard's
  output. Within a single python process duplicates are also possible if
  the queue contained the same digest twice (it doesn't — input is
  unique-by-digest), so duplicates are 100% from double-launch.

Best-rank rule:
  per digest, prefer:
    1) ok          — error is None AND http_status==200 AND content_len>0
    2) det_fail    — error in DETERMINISTIC set (fetch is final, no retry)
    3) transient   — any other error (round-2 retry candidate)
    4) empty       — http_status==200 but content_len==0 (rare; treat as
                     terminal, no retry — body genuinely empty in archive)
  Within the same rank, keep the first observed row (sorted by digest for
  determinism).

Outputs (all in E:/bot_paper_data):
  digest_robots_cache.jsonl              — deduplicated unified cache
  digest_robots_cache_retry_queue.jsonl  — digests still needing fetch
                                            (transient errors + missing
                                            digests not yet in any shard)
  digest_robots_cache_summary.txt        — class counts, top errors,
                                            sanity checks
"""

import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path("E:/bot_paper_data")
INPUT_INDEX = DATA_DIR / "cdx_digest_to_fetch.jsonl"
OUTPUT_CACHE = DATA_DIR / "digest_robots_cache.jsonl"
OUTPUT_RETRY = DATA_DIR / "digest_robots_cache_retry_queue.jsonl"
OUTPUT_SUMMARY = DATA_DIR / "digest_robots_cache_summary.txt"
NUM_SHARDS = 16
NUM_ROUND2_SHARDS = 7

# Extra retry rounds beyond round-2. Each entry is (suffix, num_shards).
# When round-N is finished and pulled to E:, append it here and re-run.
EXTRA_ROUNDS = [
    ("round3", 7),
    ("round4", 7),
    ("round5", 7),
]

DETERMINISTIC = {
    "archived_403", "archived_404", "archived_410", "archived_5xx",
    "http_404", "http_410", "gzip_error", "decode_error",
}

RANK = {"ok": 4, "det_fail": 3, "empty": 2, "transient": 1, None: 0}


def classify(r):
    e = r.get("error")
    if e is None and r.get("http_status") == 200:
        if (r.get("content_len") or 0) > 0:
            return "ok"
        return "empty"
    if e in DETERMINISTIC:
        return "det_fail"
    return "transient"


def main():
    print(f"Loading input index for sanity: {INPUT_INDEX.name}")
    expected = {}
    with open(INPUT_INDEX, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            expected[r["digest"]] = (r["sample_ts"], r["sample_url"])
    print(f"  expected unique digests: {len(expected):,}")

    best = {}             # digest -> (rank, row, cls)
    n_rows_total = 0
    rows_per_shard = []

    def ingest(path, label):
        nonlocal n_rows_total
        n_rows = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d = r.get("digest")
                if not d:
                    continue
                cls = classify(r)
                rk = RANK[cls]
                cur = best.get(d)
                if cur is None or rk > cur[0]:
                    best[d] = (rk, r, cls)
                n_rows += 1
                n_rows_total += 1
        print(f"  {label}: rows={n_rows:>6,}  unique-so-far={len(best):>6,}")
        return n_rows

    for s in range(NUM_SHARDS):
        path = DATA_DIR / f"digest_robots_cache_m{s}.jsonl"
        if not path.exists():
            print(f"  MISSING shard {s}: {path}")
            rows_per_shard.append(0)
            continue
        rows_per_shard.append(ingest(path, f"shard {s:>2}"))

    rows_per_round2 = []
    for s in range(NUM_ROUND2_SHARDS):
        path = DATA_DIR / f"digest_robots_cache_round2_m{s}.jsonl"
        if not path.exists():
            rows_per_round2.append(0)
            continue
        rows_per_round2.append(ingest(path, f"round2 m{s}"))

    extra_rounds_summary = []
    for suffix, n_shards in EXTRA_ROUNDS:
        round_rows = []
        for s in range(n_shards):
            path = DATA_DIR / f"digest_robots_cache_{suffix}_m{s}.jsonl"
            if not path.exists():
                round_rows.append(0)
                continue
            round_rows.append(ingest(path, f"{suffix} m{s}"))
        extra_rounds_summary.append((suffix, round_rows))

    print(f"\nTotal rows read across shards: {n_rows_total:,}")
    print(f"Unique digests after dedup:    {len(best):,}")
    duplicates = n_rows_total - len(best)
    print(f"Duplicate rows dropped:        {duplicates:,}")

    counts = Counter(v[2] for v in best.values())
    err_hist = Counter()
    for _, r, cls in best.values():
        if cls in {"det_fail", "transient"}:
            err_hist[r.get("error") or "<none>"] += 1

    n_in_index = len(expected)
    n_fetched = len(best)
    missing = set(expected) - set(best)
    n_missing = len(missing)

    # Write deduplicated cache
    print(f"\nWriting dedup cache: {OUTPUT_CACHE}")
    with open(OUTPUT_CACHE, "w", encoding="utf-8", newline="\n") as out:
        for d in sorted(best):
            out.write(json.dumps(best[d][1], ensure_ascii=False) + "\n")

    # Write retry queue: transient errors + missing-from-index
    print(f"Writing retry queue: {OUTPUT_RETRY}")
    n_retry = 0
    with open(OUTPUT_RETRY, "w", encoding="utf-8", newline="\n") as out:
        for d, (_rk, r, cls) in best.items():
            if cls == "transient":
                ts, url = expected.get(d, (r.get("sample_ts"), r.get("sample_url")))
                out.write(json.dumps(
                    {"digest": d, "sample_ts": ts, "sample_url": url},
                    ensure_ascii=False) + "\n")
                n_retry += 1
        for d in sorted(missing):
            ts, url = expected[d]
            out.write(json.dumps(
                {"digest": d, "sample_ts": ts, "sample_url": url},
                ensure_ascii=False) + "\n")
            n_retry += 1
    print(f"  retry queue size: {n_retry:,}")

    # Summary
    summary = []
    summary.append(f"merge_digest_robots_cache.py — summary")
    summary.append(f"")
    summary.append(f"Input index:     {INPUT_INDEX} ({n_in_index:,} digests)")
    summary.append(f"Output cache:    {OUTPUT_CACHE} ({n_fetched:,} digests)")
    summary.append(f"Output retry:    {OUTPUT_RETRY} ({n_retry:,} digests)")
    summary.append(f"")
    summary.append(f"Round-1 per-shard rows: " + ", ".join(
        f"m{i}={c}" for i, c in enumerate(rows_per_shard)))
    summary.append(f"Round-2 per-shard rows: " + ", ".join(
        f"m{i}={c}" for i, c in enumerate(rows_per_round2)))
    for suffix, rs in extra_rounds_summary:
        summary.append(f"{suffix} per-shard rows: " + ", ".join(
            f"m{i}={c}" for i, c in enumerate(rs)))
    summary.append(f"Total rows read across shards:  {n_rows_total:,}")
    summary.append(f"Unique digests after dedup:     {n_fetched:,}")
    summary.append(f"Duplicate rows dropped:         {duplicates:,}")
    summary.append(f"")
    summary.append(f"Class breakdown (per unique digest):")
    for cls in ("ok", "empty", "det_fail", "transient"):
        c = counts.get(cls, 0)
        pct = 100.0 * c / max(1, n_fetched)
        summary.append(f"  {cls:<10} {c:>10,}  ({pct:>5.2f}%)")
    summary.append(f"")
    summary.append(f"Index coverage:")
    summary.append(f"  digests in index:    {n_in_index:,}")
    summary.append(f"  digests fetched:     {n_fetched:,} "
                   f"({100.0*n_fetched/n_in_index:.2f}%)")
    summary.append(f"  digests missing:     {n_missing:,} "
                   f"({100.0*n_missing/n_in_index:.2f}%)")
    summary.append(f"")
    summary.append(f"Top non-OK error labels:")
    for e, c in err_hist.most_common(15):
        summary.append(f"  {e:<25} {c:>8,}")

    out_text = "\n".join(summary) + "\n"
    print("\n" + out_text)
    OUTPUT_SUMMARY.write_text(out_text, encoding="utf-8")
    print(f"Summary written to {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
