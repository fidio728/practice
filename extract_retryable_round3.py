"""extract_retryable_round2.py — read unified v2 jsonl, output only the
remaining ~6k retryable rows into a separate jsonl. Round 2 retry will only fetch
these 4-tuples.

Output:
  sitemap_url_paths_v2_eom_primary_retryable_round3.jsonl

This is the queue for the next retry round. Each row's 4-tuple
(norm_key, year_month, digest, timestamp) maps back to the v1 input.
"""
import json
from collections import Counter
from pathlib import Path

INPUT = Path(r"E:\cc_step6_shards\sitemap_url_paths_v2_eom_primary_r2merged.jsonl")
OUTPUT = Path(r"E:\cc_step6_shards\sitemap_url_paths_v2_eom_primary_retryable_round3.jsonl")

DET_ERRORS = {
    "depth_limit_exceeded", "xml_parse_error", "gzip_error",
    "archived_403", "archived_404", "archived_410", "archived_5xx",
    "http_404", "http_410",
}
DET_PREFIX = ("unknown_root_tag_",)


def is_retryable(r):
    err = r.get("error", "") or ""
    had_child = bool(r.get("had_child_errors"))
    cap_hit = bool(r.get("url_cap_hit"))
    child_trunc = bool(r.get("child_truncated"))
    n_paths = r.get("n_paths", 0) or 0
    has_partial = had_child or cap_hit or child_trunc
    is_det = (err in DET_ERRORS) or any(err.startswith(p) for p in DET_PREFIX)
    if err == "" and not has_partial:
        return False  # final_clean or final_clean_empty
    if err and is_det:
        return False  # final_det_fail (regardless of partial flags)
    return True  # everything else


def main():
    err_dist = Counter()
    n_total = 0
    n_retry = 0
    with open(INPUT, encoding="utf-8") as fin, \
         open(OUTPUT, "w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            line = line.rstrip("\r\n")
            if not line:
                continue
            n_total += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if is_retryable(r):
                n_retry += 1
                err = r.get("error", "") or ""
                if not err:
                    if r.get("had_child_errors"): err = "<had_child_errors>"
                    elif r.get("url_cap_hit"): err = "<url_cap_hit>"
                    elif r.get("child_truncated"): err = "<child_truncated>"
                    else: err = "<unknown>"
                err_dist[err] += 1
                fout.write(line + "\n")

    print(f"INPUT:  {INPUT}")
    print(f"  total rows:     {n_total:>10,}")
    print(f"OUTPUT: {OUTPUT}")
    print(f"  retryable rows: {n_retry:>10,}")
    print(f"  size: {OUTPUT.stat().st_size:,} bytes")
    print()
    print("Retryable error breakdown:")
    for err, cnt in err_dist.most_common(15):
        pct = 100.0 * cnt / max(1, n_retry)
        print(f"  {err:<35} {cnt:>8,}  ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
