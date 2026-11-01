"""retry_transients_round.py — re-fetch a list of residual transient_error
obs (4-tuples) up to MAX_ROUNDS times, with a sleep between rounds so the
same obs is retried at different absolute times (avoids momentary Wayback
outages).

Imports fetch_sitemap_urls + helpers from 06_collect_sitemaps_for_panel.py
via importlib (filename starts with a digit so a normal `import` won't work).

Stops retrying a key as soon as its result is clean or deterministic_partial.
Writes ONE jsonl line per attempt to the output file (so each round's
attempt for each key is preserved for audit). Each line has the standard
Step 6 result shape plus a `_retry_round` field.

Output is APPEND-friendly: caller passes a fresh path; the script overwrites
on first open and appends thereafter for that run.
"""
import argparse
import importlib.util
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


def load_main_module(script_path):
    """Load 06_collect_sitemaps_for_panel.py as a module."""
    spec = importlib.util.spec_from_file_location("collect06", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--main-script", default="06_collect_sitemaps_for_panel.py",
                   help="path to 06_collect_sitemaps_for_panel.py for fetch_sitemap_urls")
    p.add_argument("--input", required=True,
                   help="JSONL of obs to retry (raw_url, norm_key, year_month, digest, timestamp, host)")
    p.add_argument("--output", required=True,
                   help="JSONL output for retry attempts")
    p.add_argument("--max-rounds", type=int, default=5)
    p.add_argument("--round-delay-sec", type=int, default=60)
    p.add_argument("--rate-limit", type=float, default=0.7)
    p.add_argument("--workers", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    main_mod = load_main_module(args.main_script)
    main_mod.RATE_LIMIT = args.rate_limit  # override the module-level throttle gate

    fetch_sitemap_urls = main_mod.fetch_sitemap_urls
    classify = main_mod.classify_result_status
    compute_urlset_hash = main_mod.compute_urlset_hash

    obs_list = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obs_list.append(json.loads(line))

    print(f"=== retry_transients_round ===", flush=True)
    print(f"  input:        {args.input}", flush=True)
    print(f"  output:       {args.output}", flush=True)
    print(f"  obs to retry: {len(obs_list):,}", flush=True)
    print(f"  max rounds:   {args.max_rounds}", flush=True)
    print(f"  round delay:  {args.round_delay_sec}s", flush=True)
    print(f"  rate limit:   {args.rate_limit}s", flush=True)
    print(f"  workers:      {args.workers}", flush=True)
    print(flush=True)

    # Truncate the output file at start (single-source-of-truth for this run)
    Path(args.output).write_text("", encoding="utf-8")
    output_handle = open(args.output, "a", encoding="utf-8")
    write_lock = threading.Lock()
    resolved = set()

    thread_local = threading.local()

    def get_session():
        if not hasattr(thread_local, "session"):
            thread_local.session = requests.Session()
            thread_local.session.headers["User-Agent"] = USER_AGENT
        return thread_local.session

    def key_of(o):
        return (o["norm_key"], o["year_month"], o["digest"], o["timestamp"])

    def attempt(obs, round_num):
        session = get_session()
        url_paths, child_truncated, url_cap_hit, had_child_errors, error = (
            fetch_sitemap_urls(obs["raw_url"], obs["timestamp"], session)
        )
        urlset_hash = compute_urlset_hash(url_paths) if url_paths else ""
        result = {
            "sitemap_url": obs["raw_url"],
            "norm_key": obs["norm_key"],
            "year_month": obs["year_month"],
            "timestamp": obs["timestamp"],
            "digest": obs["digest"],
            "host": obs["host"],
            "urlset_hash": urlset_hash,
            "n_urls": len(url_paths),
            "child_truncated": child_truncated,
            "url_cap_hit": url_cap_hit,
            "had_child_errors": had_child_errors,
            "error": error,
            "_retry_round": round_num,
        }
        with write_lock:
            output_handle.write(json.dumps(result) + "\n")
            output_handle.flush()
        status = classify(result)
        if status in ("clean", "deterministic_partial"):
            resolved.add(key_of(obs))
        return status

    for round_num in range(1, args.max_rounds + 1):
        pending = [o for o in obs_list if key_of(o) not in resolved]
        if not pending:
            print(f"All obs resolved before round {round_num}. Done.", flush=True)
            break
        print(f"=== Round {round_num}: processing {len(pending):,} obs ===", flush=True)
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(lambda o: attempt(o, round_num), pending))
        n_resolved = sum(1 for o in obs_list if key_of(o) in resolved)
        dt = time.monotonic() - t0
        print(f"  round {round_num} took {dt:.1f}s; resolved cumulative "
              f"{n_resolved}/{len(obs_list)}", flush=True)
        if round_num < args.max_rounds and n_resolved < len(obs_list):
            print(f"  sleeping {args.round_delay_sec}s before next round...",
                  flush=True)
            time.sleep(args.round_delay_sec)

    output_handle.close()

    n_resolved = sum(1 for o in obs_list if key_of(o) in resolved)
    print(f"\n=== Final ===", flush=True)
    print(f"  total input obs:   {len(obs_list):,}", flush=True)
    print(f"  resolved:          {n_resolved:,}", flush=True)
    print(f"  still transient:   {len(obs_list) - n_resolved:,}", flush=True)


if __name__ == "__main__":
    main()
