"""extract_transient_residuals.py — pull all transient_error rows from the
unified Step 6 output into a retry-input file, splitting into N round-robin
batches for parallel cloud retry.

Each output line contains exactly the fields retry_transients_round.py needs:
  raw_url, norm_key, year_month, digest, timestamp, host

Idempotency: this script ALWAYS overwrites its outputs; it does not modify the
unified file or any per-shard partial. Re-running is safe.
"""
import argparse
import json
from pathlib import Path

DETERMINISTIC_ERRORS = {"depth_limit_exceeded", "xml_parse_error", "gzip_error",
                        "archived_403", "archived_404", "archived_410",
                        "archived_5xx", "http_404", "http_410"}


def is_det(err):
    if not err:
        return False
    if err in DETERMINISTIC_ERRORS:
        return True
    if err.startswith("unknown_root_tag_"):
        return True
    return False


def classify(r):
    err = r.get("error", "") or ""
    if err:
        if is_det(err):
            return "deterministic_partial"
        return "transient_error"
    if r.get("had_child_errors"):
        return "transient_error"
    if r.get("child_truncated") or r.get("url_cap_hit"):
        return "deterministic_partial"
    return "clean"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",
                   default="D:/anoconda/bot_paper_data/sitemap_panel_results_eom_primary.jsonl")
    p.add_argument("--output-prefix",
                   default="D:/anoconda/bot_paper_data/retry_input_r1")
    p.add_argument("--batch-shards", default="11,12,13,14,15",
                   help="comma-separated suffixes for the batch files")
    return p.parse_args()


def main():
    args = parse_args()
    transients = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if classify(r) == "transient_error":
                transients.append({
                    "raw_url": r.get("sitemap_url", ""),
                    "norm_key": r.get("norm_key", ""),
                    "year_month": r.get("year_month", ""),
                    "digest": r.get("digest", ""),
                    "timestamp": r.get("timestamp", ""),
                    "host": r.get("host", ""),
                })
    print(f"Found {len(transients):,} transient obs in {args.input}")

    suffixes = [s.strip() for s in args.batch_shards.split(",") if s.strip()]
    n = len(suffixes)
    if n < 1:
        raise SystemExit("--batch-shards must list at least one suffix")

    transients.sort(key=lambda r: (r["norm_key"], r["year_month"],
                                   r["digest"], r["timestamp"]))

    batches = [[] for _ in range(n)]
    for i, obs in enumerate(transients):
        batches[i % n].append(obs)

    for sfx, batch in zip(suffixes, batches):
        outpath = Path(f"{args.output_prefix}_m{sfx}.jsonl")
        with open(outpath, "w", encoding="utf-8") as fh:
            for obs in batch:
                fh.write(json.dumps(obs) + "\n")
        print(f"  -> {outpath.name} ({len(batch):,} obs)")

    outall = Path(f"{args.output_prefix}.jsonl")
    with open(outall, "w", encoding="utf-8") as fh:
        for obs in transients:
            fh.write(json.dumps(obs) + "\n")
    print(f"  -> {outall.name} ({len(transients):,} obs total, unsplit)")


if __name__ == "__main__":
    main()
