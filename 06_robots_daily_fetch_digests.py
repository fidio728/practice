"""06_robots_daily_fetch_digests.py — fetch robots.txt body once per
unique Wayback digest, for the daily-level transition pipeline.

Design (per DAILY_ROBOTS_TRANSITION_PROJECT.md, 2026-05-03):
  Wayback's `digest` is a content hash of the body. Two captures with the
  same digest serve the same robots.txt body. So the fetch unit is the
  unique digest, NOT (host, ts, digest) triples. The CDX raw stream has
  ~25.9M captures → 425,329 unique digests → only 99,745 with status==200.
  Only those 99,745 need a body fetch; the rest are interpretable from
  status alone via the Step 2 mapping (404/410=allow-all; 3xx/403/5xx=
  uninterpretable; 200=parse body).

This script is the fetch stage. NO sitemap recursion. NO XML parsing.
Just GET the Wayback `id_/` URL and save the raw body text.

Sharding:
  --shard N --num-shards M splits the 99,745 queue across M machines.
  Shard key = md5(digest) mod M (deterministic, stable across runs).

Resume:
  On rerun, any digest already present in this shard's output file is
  skipped. To retry transient failures, extract retryable rows (where
  error is set and not in the deterministic set) and feed them in via
  --input-digest-jsonl pointing to that retry file (round-2 mode).

Round-2 retry mode:
  --input-digest-jsonl PATH switches the queue source from the default
  cdx_digest_to_fetch.jsonl to a pre-extracted retryable jsonl. Output
  goes to <prefix>_round2_m{shard}.jsonl, never overwrites round-1.

Throttle / sessions:
  - global throttle (process-wide lock) on Wayback fetch rate
  - thread-local persistent requests.Session per worker — critical lesson
    from Step 6.5 round 2: per-fetch Session()+close() paid full TCP+TLS
    handshake on every request and was a major contributor to
    ConnectionError storms.
  - workers > 4-6 is wasted unless rate-limit is much lower; the throttle
    is global, so nworkers x rate-limit is the actual ceiling.

CLI:
  --shard N                    this machine's shard index (default 0)
  --num-shards M               total shards in deployment (default 1)
  --data-dir PATH              where to read cdx_digest_to_fetch.jsonl
                               and write digest_robots_cache_m{shard}.jsonl
                               (default E:/bot_paper_data; cloud machines
                               without E: must pass their local path)
  --workers N                  worker thread count (default 4)
  --rate-limit S               global min seconds between fetches
                               (default 0.7; 0.5 is the aggressive test
                               value, 0.7 is safer)
  --input-digest-jsonl PATH    override the default queue file. Use this
                               for round-2 retry mode against an extracted
                               retryable subset.

Output schema (one row per digest):
  digest         : canonical digest from input
  sample_ts      : 14-digit Wayback ts used for the fetch
  sample_url     : the original url (always /robots.txt for this corpus)
  http_status    : actual HTTP status code from Wayback (int) or null
  content_sha1   : sha1 hex of returned body (sanity check)
  content_len    : bytes of returned body
  content_text   : decoded body (utf-8 with errors='replace') or null
                    on fetch failure
  error          : null if ok, otherwise an error label
                    (ConnectionError / Timeout / archived_403 /
                     archived_404 / archived_410 / archived_5xx /
                     http_404 / http_410 / http_<other> / gzip_error /
                     decode_error / exc_<ExceptionName>)

Deterministic-failure labels (will not be retried by round-2 extraction):
  archived_403, archived_404, archived_410, archived_5xx,
  http_404, http_410, gzip_error, decode_error
All other error labels are treated as transient retryable.
"""

import argparse
import functools
import gzip
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

print = functools.partial(print, flush=True)

# Defaults; overridden by CLI in main().
DATA_DIR = Path("E:/bot_paper_data")
N_WORKERS = 4
RATE_LIMIT = 0.7                # global throttle, seconds. Per-machine ceiling is
                                # ~1/RATE_LIMIT req/s. 16 machines * 1/0.7 ≈ 23
                                # req/s globally — well under what Wayback was
                                # serving sustained during Step 6.5.
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 30

# Filenames set in main() once CLI is parsed.
INPUT_JSONL = None              # cdx_digest_to_fetch.jsonl (or override)
OUTPUT_JSONL = None             # digest_robots_cache_m{shard}.jsonl

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

DETERMINISTIC_ERRORS_EXACT = {
    "archived_403",
    "archived_404",
    "archived_410",
    "archived_5xx",
    "http_404",
    "http_410",
    "gzip_error",
    "decode_error",
}

_rate_lock = threading.Lock()
_last_request_at = [0.0]
_write_lock = threading.Lock()
_thread_local = threading.local()


def get_session():
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = USER_AGENT
        _thread_local.session = s
    return s


def throttle():
    with _rate_lock:
        dt = time.monotonic() - _last_request_at[0]
        if dt < RATE_LIMIT:
            time.sleep(RATE_LIMIT - dt)
        _last_request_at[0] = time.monotonic()


def is_deterministic_error(err):
    return bool(err) and err in DETERMINISTIC_ERRORS_EXACT


def shard_for(digest, num_shards):
    """md5(digest) mod num_shards. Stable across Python invocations."""
    h = int(hashlib.md5(digest.encode("utf-8")).hexdigest()[:8], 16)
    return h % num_shards


def wayback_url(timestamp, url):
    return f"https://web.archive.org/web/{timestamp}id_/{url}"


# ---------------------- HTTP layer ----------------------

def fetch_robots_body(session, ts, url):
    """Returns (http_status, content_bytes, error). http_status is None
    on transport failure (ConnectionError / Timeout / etc.)."""
    throttle()
    wb = wayback_url(ts, url)
    try:
        resp = session.get(wb, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except requests.exceptions.RequestException as e:
        return None, None, type(e).__name__

    code = resp.status_code
    if code != 200:
        is_archived = any(k.lower().startswith("x-archive-orig-") for k in resp.headers)
        if is_archived:
            if code == 403:
                return code, None, "archived_403"
            if code == 404:
                return code, None, "archived_404"
            if code == 410:
                return code, None, "archived_410"
            if 500 <= code < 600:
                return code, None, "archived_5xx"
            return code, None, f"archived_{code}"
        if code == 404:
            return code, None, "http_404"
        if code == 410:
            return code, None, "http_410"
        return code, None, f"http_{code}"

    body = resp.content
    if url.lower().endswith(".gz") or body[:2] == b"\x1f\x8b":
        try:
            body = gzip.decompress(body)
        except Exception:
            return code, None, "gzip_error"
    return code, body, None


# ---------------------- per-digest worker ----------------------

def _row(item, http_status, body, error):
    if body is None:
        sha1 = None
        clen = 0
        text = None
    else:
        sha1 = hashlib.sha1(body).hexdigest()
        clen = len(body)
        try:
            text = body.decode("utf-8", errors="replace")
        except Exception:
            text = None
            if not error:
                error = "decode_error"
    return {
        "digest":        item["digest"],
        "sample_ts":     item["sample_ts"],
        "sample_url":    item["sample_url"],
        "http_status":   http_status,
        "content_sha1":  sha1,
        "content_len":   clen,
        "content_text":  text,
        "error":         error,
    }


def fetch_one_digest(item):
    session = get_session()
    try:
        http_status, body, err = fetch_robots_body(
            session, item["sample_ts"], item["sample_url"]
        )
        return _row(item, http_status, body, err)
    except Exception as e:
        return _row(item, None, None, f"exc_{type(e).__name__}")


# ---------------------- queue + resume ----------------------

def load_done_digests():
    """Read this shard's output file, return set of digests already
    written (regardless of success/failure). Round-2 mode is invoked by
    pointing --input-digest-jsonl at a retryable subset; the output file
    in that mode is round2-suffixed and starts empty on a fresh round."""
    done = set()
    if not OUTPUT_JSONL.exists():
        return done
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = r.get("digest")
            if d:
                done.add(d)
    return done


def build_queue(shard, num_shards):
    if not INPUT_JSONL.exists():
        print(f"ERROR: {INPUT_JSONL} not found.")
        raise SystemExit(1)
    print(f"Reading queue: {INPUT_JSONL.name}")
    done = load_done_digests()
    print(f"  output file already has {len(done):,} digests done (will skip)")

    queue = []
    n_total = 0
    n_other_shard = 0
    n_already_done = 0
    n_missing_field = 0
    n_non_200 = 0          # only relevant if user passes a non-pre-filtered file

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1

            digest = r.get("digest")
            sample_ts = r.get("sample_ts")
            sample_url = r.get("sample_url")
            if not digest or not sample_ts or not sample_url:
                n_missing_field += 1
                continue

            # Defensive: in round-1 mode the input is pre-filtered to
            # sample_status==200. If a caller passes the unfiltered index
            # by mistake, drop non-200 rows here so we don't waste fetches
            # on rows whose body is determined purely by status.
            sample_status = r.get("sample_status")
            if sample_status is not None and str(sample_status) != "200":
                n_non_200 += 1
                continue

            if num_shards > 1 and shard_for(digest, num_shards) != shard:
                n_other_shard += 1
                continue

            if digest in done:
                n_already_done += 1
                continue

            queue.append({
                "digest":      digest,
                "sample_ts":   sample_ts,
                "sample_url":  sample_url,
            })

    print(f"  input rows total:                 {n_total:,}")
    if n_non_200:
        print(f"    non-200 sample_status (skipped):{n_non_200:,}")
    if num_shards > 1:
        print(f"    other shards (mod-{num_shards}):           {n_other_shard:,}")
    print(f"    output already has digest:      {n_already_done:,}")
    print(f"    missing field (bad row):        {n_missing_field:,}")
    print(f"  remaining to fetch this run:      {len(queue):,}")
    return queue


def append_result(row):
    with _write_lock:
        with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------- main ----------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=0,
                   help="this machine's shard index (0..num_shards-1)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="total shards across the deployment "
                        "(1 = unsharded local run)")
    p.add_argument("--data-dir", type=str, default="E:/bot_paper_data",
                   help="data directory; cloud machines without E: must "
                        "pass their local path here (e.g., "
                        "/home/fidio728/bot_paper_data)")
    p.add_argument("--workers", type=int, default=N_WORKERS,
                   help="number of worker threads for parallel Wayback fetch")
    p.add_argument("--rate-limit", type=float, default=RATE_LIMIT,
                   help="global throttle: minimum seconds between Wayback "
                        "fetches across all workers")
    p.add_argument("--input-digest-jsonl", type=str, default=None,
                   help="override input queue path. Default is "
                        "<data-dir>/cdx_digest_to_fetch.jsonl. Use this for "
                        "round-2 retry mode against an extracted retryable "
                        "subset.")
    p.add_argument("--round2", action="store_true",
                   help="round-2 mode: write to digest_robots_cache_round2_m"
                        "{shard}.jsonl instead of the round-1 main output. "
                        "Implied when --input-digest-jsonl is set; pass this "
                        "explicitly only if you also want round-2 naming.")
    p.add_argument("--output-suffix", type=str, default=None,
                   help="custom suffix for output filename. When set, output "
                        "is digest_robots_cache_<suffix>_m{shard}.jsonl. "
                        "Use this for round-3+ to avoid colliding with "
                        "round-2's output. Implies retry mode.")
    return p.parse_args()


def main():
    global DATA_DIR, N_WORKERS, RATE_LIMIT
    global INPUT_JSONL, OUTPUT_JSONL

    args = parse_args()
    shard = args.shard
    num_shards = args.num_shards
    if num_shards < 1:
        raise SystemExit(f"--num-shards must be >= 1 (got {num_shards})")
    if not (0 <= shard < num_shards):
        raise SystemExit(f"--shard must be in [0, {num_shards-1}] (got {shard})")
    if args.workers < 1:
        raise SystemExit(f"--workers must be >= 1 (got {args.workers})")
    if args.rate_limit < 0:
        raise SystemExit(f"--rate-limit must be >= 0 (got {args.rate_limit})")

    DATA_DIR = Path(args.data_dir)
    N_WORKERS = args.workers
    RATE_LIMIT = args.rate_limit

    if not DATA_DIR.exists():
        raise SystemExit(f"--data-dir does not exist: {DATA_DIR}")

    retry_mode = (args.round2 or args.input_digest_jsonl is not None
                  or args.output_suffix is not None)

    if args.input_digest_jsonl is not None:
        INPUT_JSONL = Path(args.input_digest_jsonl)
    else:
        INPUT_JSONL = DATA_DIR / "cdx_digest_to_fetch.jsonl"

    if args.output_suffix is not None:
        suf = args.output_suffix
        if num_shards > 1:
            OUTPUT_JSONL = DATA_DIR / f"digest_robots_cache_{suf}_m{shard}.jsonl"
        else:
            OUTPUT_JSONL = DATA_DIR / f"digest_robots_cache_{suf}.jsonl"
    elif retry_mode:
        if num_shards > 1:
            OUTPUT_JSONL = DATA_DIR / f"digest_robots_cache_round2_m{shard}.jsonl"
        else:
            OUTPUT_JSONL = DATA_DIR / "digest_robots_cache_round2.jsonl"
    else:
        if num_shards > 1:
            OUTPUT_JSONL = DATA_DIR / f"digest_robots_cache_m{shard}.jsonl"
        else:
            OUTPUT_JSONL = DATA_DIR / "digest_robots_cache.jsonl"

    print(f"=== robots daily fetch by digest ===")
    print(f"  mode:         {'retry (suffix='+args.output_suffix+')' if args.output_suffix else ('round-2 retry' if retry_mode else 'round-1 main')}")
    print(f"  data_dir:     {DATA_DIR}")
    print(f"  shard:        {shard} of {num_shards}")
    print(f"  input:        {INPUT_JSONL}")
    print(f"  output:       {OUTPUT_JSONL}")
    print(f"  workers:      {N_WORKERS}")
    print(f"  rate limit:   {RATE_LIMIT}s (global, shared across workers)")
    print()

    queue = build_queue(shard, num_shards)
    if not queue:
        print("Nothing to do.")
        return

    print(f"\nStarting parallel fetch ({N_WORKERS} workers, "
          f"RATE_LIMIT={RATE_LIMIT}s global)")
    total = len(queue)
    done_n = ok_n = empty_n = err_n = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(fetch_one_digest, t) for t in queue]
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception as e:
                print(f"worker exception: {e}")
                continue
            append_result(res)
            done_n += 1
            if res["error"] is None and res["http_status"] == 200:
                if (res["content_len"] or 0) > 0:
                    ok_n += 1
                else:
                    empty_n += 1
            else:
                err_n += 1
            if done_n % 200 == 0:
                dt = time.monotonic() - t0
                rate = done_n / max(1.0, dt)
                eta_min = (total - done_n) / max(0.01, rate) / 60.0
                print(f"  [{done_n}/{total}] ok={ok_n}, empty={empty_n}, "
                      f"err={err_n} | {rate:.2f} req/s | ETA {eta_min:.1f} min")

    dt = time.monotonic() - t0
    print(f"\nDone. {done_n} processed in {dt/60.0:.1f} min "
          f"({done_n/max(1.0, dt):.2f} req/s).")
    print(f"  ok (200, body):    {ok_n}")
    print(f"  empty (200, 0 B):  {empty_n}")
    print(f"  errors:            {err_n}")
    print(f"Re-run this script to resume; round-2 mode handles transient retries.")


if __name__ == "__main__":
    main()
