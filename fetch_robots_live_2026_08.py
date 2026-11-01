"""Live robots.txt scan, August 2026 vintage.

Purpose
-------
Go/no-go probe for the AI-crawler-specific identification question: in 2021-2025
88% of restricting firms used a bare `User-agent: *` wildcard, which is why the
paper cannot separate AI crawlers from traditional ones. This scan measures how
much AI-specific naming exists NOW, on the same host universe as the
2026-03-27 scan, so the change over five months is also measurable.

Design notes (differs from `gpt answer/scan_robots.py`)
------------------------------------------------------
* Checkpointing is append-only JSONL, flushed after every record. Resume reads
  the file back and skips hosts already present. The old scanner rewrote the
  whole progress JSON per host (O(n^2)) and kept no raw text.
* The RAW robots.txt body is stored. Re-parsing under a different crawler
  taxonomy then costs nothing; no refetch.
* Concurrent across hosts. Each host receives exactly one or two requests, so
  concurrency here is not a per-site rate-limit question.
* Fetch semantics mirror `fetch_utils.fetch_robots` exactly (https, ssl_error
  -> http fallback, 403 -> browser UA retry) so this vintage stays comparable
  with the March scan.

Usage
-----
    python fetch_robots_live_2026_08.py                  # fetch / resume
    python fetch_robots_live_2026_08.py --retry-errors   # re-fetch failed rows only
    python fetch_robots_live_2026_08.py --workers 8      # gentler
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
import requests

PROJECT_DIR = Path(r"c:\Users\xl\OneDrive - Universitat Ramón Llull\git\practice")
DEFAULT_INPUT = PROJECT_DIR / "gpt answer" / "compustat_2025_clean.csv"
DEFAULT_OUTDIR = Path(r"E:\robots_live_2026_08")

UA_RESEARCH = "Mozilla/5.0 (compatible; AcademicResearchBot/1.0)"
UA_BROWSER = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
MAX_CONTENT_BYTES = 1_000_000
MAX_STORE_CHARS = 300_000

_write_lock = threading.Lock()
_counter_lock = threading.Lock()
_counters = {"done": 0, "ok": 0, "err": 0}


# ---------------------------------------------------------------- fetch layer


def _decode_text(response: requests.Response) -> str:
    encoding = response.encoding or "utf-8"
    try:
        return response.content[:MAX_CONTENT_BYTES].decode(encoding, errors="replace")
    except LookupError:
        return response.content[:MAX_CONTENT_BYTES].decode("utf-8", errors="replace")


def _single_request(url: str, timeout: int, user_agent: str) -> Dict[str, Any]:
    try:
        response = requests.get(
            url,
            headers={"User-Agent": user_agent},
            timeout=timeout,
            allow_redirects=True,
        )
    except requests.exceptions.Timeout:
        return _err(url, "timeout", user_agent)
    except requests.exceptions.SSLError:
        return _err(url, "ssl_error", user_agent)
    except requests.exceptions.TooManyRedirects:
        return _err(url, "too_many_redirects", user_agent)
    except requests.exceptions.ConnectionError:
        return _err(url, "connection_error", user_agent)
    except requests.RequestException as exc:
        return _err(url, f"request_exception:{type(exc).__name__}", user_agent)
    except Exception as exc:  # nothing may escape and kill the worker
        return _err(url, f"unexpected:{type(exc).__name__}", user_agent)

    return {
        "status_code": response.status_code,
        "content": _decode_text(response),
        "final_url": response.url,
        "fetch_error": "",
        "ua_used": user_agent,
    }


def _err(url: str, message: str, user_agent: str) -> Dict[str, Any]:
    return {
        "status_code": None,
        "content": None,
        "final_url": url,
        "fetch_error": message,
        "ua_used": user_agent,
    }


def fetch_url(url: str, timeout: int, ua: str = UA_RESEARCH) -> Dict[str, Any]:
    first = _single_request(url, timeout=timeout, user_agent=ua)
    if first["status_code"] == 403 and ua != UA_BROWSER:
        retry = _single_request(url, timeout=timeout, user_agent=UA_BROWSER)
        if retry["status_code"] is not None or retry["fetch_error"]:
            return retry
    return first


def fetch_robots(host: str, timeout: int = 20) -> Dict[str, Any]:
    """Mirrors gpt answer/fetch_utils.fetch_robots so vintages stay comparable."""
    result = fetch_url(f"https://{host}/robots.txt", timeout=timeout, ua=UA_RESEARCH)
    result["http_fallback"] = False

    if result.get("fetch_error") == "ssl_error":
        result = fetch_url(f"http://{host}/robots.txt", timeout=timeout, ua=UA_RESEARCH)
        result["http_fallback"] = True

    return result


def classify_status_type(fetch_result: Dict[str, Any]) -> str:
    status_code = fetch_result.get("status_code")
    if status_code == 200:
        return "ok_200"
    if status_code == 404:
        return "missing_404"
    if isinstance(status_code, int) and 400 <= status_code < 500:
        return "http_error_4xx"
    if isinstance(status_code, int) and 500 <= status_code < 600:
        return "http_error_5xx"

    fetch_error = str(fetch_result.get("fetch_error") or "")
    for prefix in ("timeout", "ssl_error", "connection_error", "too_many_redirects"):
        if fetch_error.startswith(prefix):
            return prefix
    return "request_error"


def scan_host(host: str, timeout: int) -> Dict[str, Any]:
    stamp = datetime.now(timezone.utc)
    result = fetch_robots(host, timeout=timeout)
    content = result.get("content")
    truncated = False
    if isinstance(content, str) and len(content) > MAX_STORE_CHARS:
        content = content[:MAX_STORE_CHARS]
        truncated = True

    return {
        "scan_utc": stamp.isoformat().replace("+00:00", "Z"),
        "scan_date": stamp.date().isoformat(),
        "host": host,
        "status_code": result.get("status_code"),
        "robots_status_type": classify_status_type(result),
        "fetch_error": result.get("fetch_error") or "",
        "robots_final_url": result.get("final_url") or "",
        "http_fallback": int(bool(result.get("http_fallback"))),
        "ua_used": result.get("ua_used") or "",
        "content_truncated": int(truncated),
        "content": content,
    }


# ----------------------------------------------------------- checkpoint layer


OK_STATUS = {"ok_200", "missing_404"}


def load_done(path: Path, retry_errors: bool) -> Set[str]:
    """Hosts already recorded. With --retry-errors, failed rows do not count."""
    if not path.exists():
        return set()

    done: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from a hard kill
            host = row.get("host")
            if not host:
                continue
            if retry_errors and row.get("robots_status_type") not in OK_STATUS:
                continue
            done.add(host)
    return done


def append_record(path: Path, record: Dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with _write_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def log(logfile: Path, message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    with _write_lock:
        with logfile.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
    print(line, flush=True)


# -------------------------------------------------------------------- driver


def load_hosts(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, dtype={"gvkey": str, "cik": str, "tic": str}, low_memory=False)
    valid = df.loc[
        (~df["weburl_missing"].astype(bool))
        & (~df["url_parse_error"].astype(bool))
        & df["host_clean"].notna()
    ].copy()
    valid["host_clean"] = valid["host_clean"].astype(str)
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Live robots.txt scan, 2026-08 vintage.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-fetch only hosts whose stored row is not ok_200/missing_404.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    raw_path = args.outdir / "raw_robots.jsonl"
    log_path = args.outdir / "fetch.log"
    host_list_path = args.outdir / "host_universe.csv"

    valid = load_hosts(args.input)
    hosts: List[str] = list(dict.fromkeys(valid["host_clean"].tolist()))
    if args.limit is not None:
        hosts = hosts[: args.limit]

    # Persist the gvkey <-> host crosswalk alongside the scan so the parse step
    # never has to re-derive it from a file that may move.
    valid.loc[
        valid["host_clean"].isin(set(hosts)),
        ["gvkey", "conm", "tic", "cik", "weburl", "host_clean", "shared_host_flag", "host_duplicate_count"],
    ].to_csv(host_list_path, index=False)

    done = load_done(raw_path, args.retry_errors)
    todo = [h for h in hosts if h not in done]

    log(log_path, f"universe={len(hosts)} hosts | already_done={len(done)} | todo={len(todo)}")
    if args.retry_errors:
        log(log_path, "retry-errors mode: failed rows will be appended again (parse takes the last row per host)")
    if not todo:
        log(log_path, "nothing to do")
        return

    started = time.time()
    total = len(todo)

    def work(host: str) -> None:
        try:
            record = scan_host(host, timeout=args.timeout)
        except Exception as exc:  # a dead worker must never stall the pool
            record = {
                "scan_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "scan_date": datetime.now(timezone.utc).date().isoformat(),
                "host": host,
                "status_code": None,
                "robots_status_type": "request_error",
                "fetch_error": f"worker_exception:{type(exc).__name__}:{exc}",
                "robots_final_url": "",
                "http_fallback": 0,
                "ua_used": "",
                "content_truncated": 0,
                "content": None,
            }
        append_record(raw_path, record)

        with _counter_lock:
            _counters["done"] += 1
            if record["robots_status_type"] in OK_STATUS:
                _counters["ok"] += 1
            else:
                _counters["err"] += 1
            n = _counters["done"]

        if n % 100 == 0 or n == total:
            elapsed = time.time() - started
            rate = n / elapsed if elapsed else 0.0
            eta = (total - n) / rate if rate else 0.0
            log(
                log_path,
                f"{n}/{total} done | ok={_counters['ok']} err={_counters['err']} "
                f"| {rate:.1f} host/s | eta {eta/60:.1f} min",
            )

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, host) for host in todo]
            for future in as_completed(futures):
                future.result()
    except KeyboardInterrupt:
        log(log_path, "interrupted; every completed host is already on disk, rerun to resume")
        sys.exit(130)

    elapsed = time.time() - started
    log(
        log_path,
        f"FINISHED {_counters['done']}/{total} in {elapsed/60:.1f} min "
        f"| ok={_counters['ok']} err={_counters['err']} | raw -> {raw_path}",
    )


if __name__ == "__main__":
    main()
