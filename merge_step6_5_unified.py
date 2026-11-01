"""merge_step6_5_unified.py — consolidate v2 jsonl shards into unified
v2 by 4-tuple, picking BEST row per key by rank.

Inputs (36 files after round 2):
  step6_5_v2_phase_done/m{0..7,9..15}.jsonl              (15 round-1 root, NOT m8 root)
  step6_5_v2_phase_done/m8_subs/m{8,24,40,56,72}.jsonl   (5 round-1 m8 sub-shards)
  step6_5_v2_phase_done/round2/m{0..15}.jsonl            (16 round-2 retry outputs)

Output:
  sitemap_url_paths_v2_eom_primary.jsonl  (unified, dedup by 4-tuple)

Rank (highest wins):
  final_clean         (err="", no partial flag, n_paths>0)
  final_clean_empty   (err="", no partial flag, n_paths==0)
  final_det_fail      (err in DET set, no partial flag)
  retryable           (everything else: NET, partial flag, http_429, etc.)

Tiebreak within same rank: prefer larger n_paths, then last-seen (which is
guaranteed to be from round 2 since it's last in the input list).
"""
import json
from collections import Counter
from pathlib import Path

INPUT_FILES = [
    Path(f"D:/anoconda/bot_paper_data/step6_5_v2_phase_done/m{i}.jsonl")
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
] + [
    Path(f"D:/anoconda/bot_paper_data/step6_5_v2_phase_done/m8_subs/m{i}.jsonl")
    for i in [8, 24, 40, 56, 72]
] + [
    Path(f"D:/anoconda/bot_paper_data/step6_5_v2_phase_done/round2/m{i}.jsonl")
    for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
]

OUTPUT = Path(r"E:\bot_paper_data\sitemap_url_paths_v2_eom_primary.jsonl")

DET_ERRORS = {
    "depth_limit_exceeded", "xml_parse_error", "gzip_error",
    "archived_403", "archived_404", "archived_410", "archived_5xx",
    "http_404", "http_410",
}
DET_PREFIX = ("unknown_root_tag_",)

RANK = {"final_clean": 3, "final_clean_empty": 2, "final_det_fail": 1, "retryable": 0}
INV_RANK = {v: k for k, v in RANK.items()}


def classify(r):
    err = r.get("error", "") or ""
    had_child = bool(r.get("had_child_errors"))
    cap_hit = bool(r.get("url_cap_hit"))
    child_trunc = bool(r.get("child_truncated"))
    n_paths = r.get("n_paths", 0) or 0
    has_partial = had_child or cap_hit or child_trunc
    is_det = (err in DET_ERRORS) or any(err.startswith(p) for p in DET_PREFIX)
    if err == "" and not has_partial:
        return ("final_clean" if n_paths > 0 else "final_clean_empty"), n_paths
    if err and is_det and not has_partial:
        return "final_det_fail", n_paths
    return "retryable", n_paths


def main():
    # PASS 1: scan all files, track winners per key
    print(f"PASS 1: scanning {len(INPUT_FILES)} files...")
    best = {}  # key -> (rank, n_paths, file_idx, line_idx)
    bad_json = 0
    missing_key = 0
    total_lines = 0

    for fidx, fpath in enumerate(INPUT_FILES):
        n = 0
        with open(fpath, encoding="utf-8") as f:
            for lidx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                n += 1
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    bad_json += 1
                    continue
                try:
                    key = (r["norm_key"], r["year_month"], r["digest"], r["timestamp"])
                except KeyError:
                    missing_key += 1
                    continue
                status, n_paths = classify(r)
                rank = RANK[status]
                cur = best.get(key)
                # rank ascending desired; if equal, larger n_paths wins; if still equal,
                # later occurrence wins (we always replace on >= for that)
                if cur is None or (rank, n_paths) > (cur[0], cur[1]):
                    best[key] = (rank, n_paths, fidx, lidx)
                elif (rank, n_paths) == (cur[0], cur[1]):
                    # tiebreak: keep last (this overwrites)
                    best[key] = (rank, n_paths, fidx, lidx)
        total_lines += n
        print(f"  [{fidx+1:>2}/{len(INPUT_FILES)}] {fpath.name:<20} {n:>7,} lines  cum_keys={len(best):>7,}")

    print()
    print(f"PASS 1 done.")
    print(f"  total lines:    {total_lines:>10,}")
    print(f"  unique keys:    {len(best):>10,}")
    print(f"  bad JSON:       {bad_json:>10,}")
    print(f"  missing key:    {missing_key:>10,}")

    # Status histogram (best-rank)
    status_counts = Counter()
    n_paths_total = 0
    for rank, n_paths, _, _ in best.values():
        status_counts[INV_RANK[rank]] += 1
        if INV_RANK[rank] == "final_clean":
            n_paths_total += n_paths

    print()
    print("Best-rank status counts (after dedupe):")
    for s in ["final_clean", "final_clean_empty", "final_det_fail", "retryable"]:
        print(f"  {s:<22} {status_counts[s]:>10,}")
    print(f"  sum n_paths in final_clean: {n_paths_total:,}")

    # Build set of (file_idx, line_idx) to emit
    winners = set()
    for rank, n_paths, fidx, lidx in best.values():
        winners.add((fidx, lidx))
    print(f"  winner lines:   {len(winners):>10,}")

    # PASS 2: re-stream and emit only winner lines
    print()
    print(f"PASS 2: writing {OUTPUT}...")
    written = 0
    with open(OUTPUT, "w", encoding="utf-8", newline="\n") as out:
        for fidx, fpath in enumerate(INPUT_FILES):
            with open(fpath, encoding="utf-8") as f:
                for lidx, line in enumerate(f):
                    if (fidx, lidx) in winners:
                        line = line.rstrip("\r\n")
                        out.write(line + "\n")
                        written += 1

    print(f"PASS 2 done. Wrote {written:,} lines.")
    print()
    print(f"OUTPUT: {OUTPUT}")
    print(f"  size: {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
