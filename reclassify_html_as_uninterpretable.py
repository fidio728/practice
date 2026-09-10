"""reclassify_html_as_uninterpretable.py — fix Step B's body-validation
hole.

Problem (per user audit 2026-05-03):
  parse_status=ok in digest_bot_policy.csv currently includes 41,741
  digests with n_groups=0. Of those, 40,324 are HTML challenge / error /
  Cloudflare-style pages (the server returned HTML for /robots.txt, not
  a real robots file). The Step B parser ran parse_robots() on them and
  got groups=[], then assigned root_block=0 to every bot — i.e., it
  silently classified those captures as "explicitly unblocked" allow-all.
  In Step C this would inject fake transition windows for every (host,
  bot) pair whose timeline crosses one of those HTML captures.

  The other ~838 n_groups=0 cases (Sitemap-only, comments-only,
  Allow-only-without-User-agent) are RFC 9309 valid allow-all and should
  STAY as ok.

Fix:
  Pass 1 — scan digest_robots_cache.jsonl with the same binary
  streaming-drain reader the parser uses (so an 8 MB challenge HTML
  doesn't OOM the host on tight RAM). For each digest with body present,
  classify the body two ways:
    has_html_marker       — case-insensitive substring search for
                            <html, <!doctype, <head, <body, <title,
                            <meta in the body's first 4 KB
    has_user_agent_line   — any line whose stripped lowercase prefix is
                            "user-agent:"
  If has_html_marker AND not has_user_agent_line → add digest to the
  to_reclassify set. Both conditions are intentional: a malformed page
  that injects a "User-agent:" line into HTML still has parseable robots
  semantics in parse_robots, so we keep it as ok.

  Pass 2 — rewrite digest_bot_policy.csv in place: each row whose digest
  is in to_reclassify becomes parse_status="uninterpretable", with
  n_groups, n_disallow_rules, and all 69 × 4 per-bot columns blank.
  Other rows pass through unchanged.

  Pass 3 — regenerate digest_bot_policy_summary.txt from the corrected
  CSV (stats only; no per-bot detail since most-frequently-blocked
  rankings would shift trivially after the fix and the user's downstream
  analysis re-derives them anyway).

Inputs:
  E:/bot_paper_data/digest_robots_cache.jsonl
  E:/bot_paper_data/digest_bot_policy.csv

Outputs (in place):
  E:/bot_paper_data/digest_bot_policy.csv          — fixed
  E:/bot_paper_data/digest_bot_policy_summary.txt  — regenerated
  E:/bot_paper_data/reclassify_html_audit.jsonl    — one line per
      reclassified digest with body diagnostics, for spot-checking
"""

import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

DATA_DIR = Path("E:/bot_paper_data")
INPUT_CACHE = DATA_DIR / "digest_robots_cache.jsonl"
CSV_PATH = DATA_DIR / "digest_bot_policy.csv"
TMP_CSV = CSV_PATH.with_suffix(".reclassify.csv")
SUMMARY_PATH = DATA_DIR / "digest_bot_policy_summary.txt"
AUDIT_JSONL = DATA_DIR / "reclassify_html_audit.jsonl"

_DIGEST_RE = re.compile(rb'"digest"\s*:\s*"([^"]+)"')

# HTML detection: substrings we expect in any HTML response served as
# /robots.txt. Lowercase; we lowercase the body sample too.
HTML_MARKERS = (b"<html", b"<!doctype", b"<head", b"<body", b"<title", b"<meta ")

# User-agent detection: any line whose stripped lowercase prefix is
# exactly this. Use binary so we can run the check before utf-8 decoding
# huge-but-not-huge-enough-to-skip lines.
_USER_AGENT_LINE_RE = re.compile(
    rb'(?im)^[ \t]*user-agent[ \t]*:'
)

# How many leading bytes of the body to scan for HTML markers. Cloudflare
# challenge pages start their <html> tag in the first few hundred bytes;
# 4 KB is a safe ceiling.
HTML_SCAN_HEAD_BYTES = 4096


def is_html_body(body_bytes):
    head = body_bytes[:HTML_SCAN_HEAD_BYTES].lower()
    return any(m in head for m in HTML_MARKERS)


def has_user_agent_line(body_bytes):
    return _USER_AGENT_LINE_RE.search(body_bytes) is not None


def scan_cache_for_reclassify():
    """Pass 1. Return (to_reclassify_set, audit_records, scan_stats)."""
    READLINE_MAX = 500_000
    to_reclassify = set()
    audit = []
    n_lines = 0
    n_with_body = 0
    n_html_markers = 0
    n_user_agent = 0
    n_oversize = 0
    t0 = time.monotonic()

    with open(INPUT_CACHE, "rb") as fin:
        while True:
            line_b = fin.readline(READLINE_MAX)
            if not line_b:
                break
            n_lines += 1

            # Oversize: regex-extract digest, drain rest of line streaming.
            if not line_b.endswith(b"\n"):
                n_oversize += 1
                head = line_b
                # Continue reading without keeping; if drained len < 64MB we
                # take the body sample for HTML check from `head` only.
                # That's enough: HTML markers are always at the start.
                while True:
                    extra = fin.readline(READLINE_MAX)
                    if not extra:
                        break
                    if extra.endswith(b"\n"):
                        break
                m = _DIGEST_RE.search(head)
                if not m:
                    head = None
                    continue
                digest = m.group(1).decode("utf-8", errors="replace")
                # Extract content_text from `head` if we can — this only
                # works if "content_text":"... fits in the first 500K of
                # the line. For oversize lines this is true (the giant
                # tail is the body itself, and we only need the head for
                # HTML detection anyway).
                # Find the start of content_text value in head:
                ct_start = head.find(b'"content_text":"')
                if ct_start == -1:
                    # No content available for sniff — skip
                    head = None
                    continue
                body_sample = head[ct_start + len(b'"content_text":"'):
                                   ct_start + len(b'"content_text":"') + HTML_SCAN_HEAD_BYTES * 4]
                # Body sample is JSON-escaped — for HTML markers like <html
                # the json escape doesn't change them (< and ! and / are
                # all literal), so substring search still works.
                if is_html_body(body_sample):
                    n_html_markers += 1
                    # Scan up to ~64 KB of head for User-agent line
                    ua_present = has_user_agent_line(head[:65536])
                    if ua_present:
                        n_user_agent += 1
                    else:
                        to_reclassify.add(digest)
                        audit.append({
                            "digest": digest,
                            "reason": "html_no_user_agent",
                            "oversize_line": True,
                            "head_len": len(head),
                        })
                head = None
                body_sample = None
                continue

            # Normal-size line
            try:
                r = json.loads(line_b)
            except json.JSONDecodeError:
                continue
            digest = r.get("digest")
            if not digest:
                continue
            content = r.get("content_text")
            if not content:
                continue   # already uninterpretable from Step A
            n_with_body += 1
            # Re-encode to bytes once for the marker checks
            cb = content.encode("utf-8", errors="replace")
            if is_html_body(cb):
                n_html_markers += 1
                if has_user_agent_line(cb):
                    n_user_agent += 1
                else:
                    to_reclassify.add(digest)
                    audit.append({
                        "digest": digest,
                        "reason": "html_no_user_agent",
                        "oversize_line": False,
                        "body_len": len(content),
                        "preview": content[:120].replace("\n", " "),
                    })
            cb = None
            content = None
            r = None
            line_b = None

            if n_lines % 10000 == 0:
                dt = time.monotonic() - t0
                print(f"  [{n_lines:,}] reclassify-set={len(to_reclassify):,} "
                      f"html_total={n_html_markers:,} html+ua={n_user_agent:,} "
                      f"oversize={n_oversize:,}  {n_lines/max(0.01,dt):.0f}/s",
                      flush=True)

    stats = {
        "lines_total": n_lines,
        "with_body": n_with_body,
        "html_marker_total": n_html_markers,
        "html_with_user_agent_kept_ok": n_user_agent,
        "oversize_lines": n_oversize,
        "to_reclassify": len(to_reclassify),
        "scan_seconds": time.monotonic() - t0,
    }
    return to_reclassify, audit, stats


def rewrite_csv(to_reclassify):
    """Pass 2. Returns (n_total, n_replaced, post_status_counts)."""
    print(f"\nRewriting {CSV_PATH} (reclassify {len(to_reclassify):,} digests)...")
    n_total = 0
    n_replaced = 0
    post_status = Counter()
    fieldnames = None
    with open(CSV_PATH, encoding="utf-8", newline="") as fin, \
         open(TMP_CSV, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for r in reader:
            n_total += 1
            d = r["digest"]
            if d in to_reclassify:
                # Wipe everything except digest; mark uninterpretable.
                blanked = {fn: "" for fn in fieldnames}
                blanked["digest"] = d
                blanked["parse_status"] = "uninterpretable"
                writer.writerow(blanked)
                n_replaced += 1
                post_status["uninterpretable"] += 1
            else:
                writer.writerow(r)
                post_status[r.get("parse_status", "")] += 1
    os.replace(TMP_CSV, CSV_PATH)
    return n_total, n_replaced, post_status


def write_summary(stats, n_total, n_replaced, post_status):
    lines = []
    lines.append("digest_bot_policy.csv — summary (post-reclassify)")
    lines.append("")
    lines.append(f"Input cache:  {INPUT_CACHE}")
    lines.append(f"Output CSV:   {CSV_PATH}  ({n_total:,} rows)")
    lines.append("")
    lines.append("Reclassification scan stats:")
    lines.append(f"  cache rows scanned:               {stats['lines_total']:,}")
    lines.append(f"  rows with body present:           {stats['with_body']:,}")
    lines.append(f"  rows with HTML markers in body:   {stats['html_marker_total']:,}")
    lines.append(f"    of those, also has User-agent (kept ok): {stats['html_with_user_agent_kept_ok']:,}")
    lines.append(f"  rows reclassified ok -> uninterpretable:    {stats['to_reclassify']:,}")
    lines.append(f"  oversize lines (drained, head sniffed only):{stats['oversize_lines']:,}")
    lines.append(f"  scan time:                                  {stats['scan_seconds']:.1f}s")
    lines.append("")
    lines.append(f"CSV rows replaced this run:           {n_replaced:,}")
    lines.append("")
    lines.append("Final parse_status breakdown (per unique digest):")
    for st in ("ok", "uninterpretable", "too_large", "parse_error"):
        c = post_status.get(st, 0)
        if c:
            pct = 100.0 * c / max(1, n_total)
            lines.append(f"  {st:<20} {c:>10,}  ({pct:>5.2f}%)")
    # any other statuses
    for st, c in post_status.most_common():
        if st in ("ok", "uninterpretable", "too_large", "parse_error"):
            continue
        if c:
            pct = 100.0 * c / max(1, n_total)
            lines.append(f"  {st:<20} {c:>10,}  ({pct:>5.2f}%)")

    text = "\n".join(lines) + "\n"
    print("\n" + text)
    SUMMARY_PATH.write_text(text, encoding="utf-8")
    print(f"Summary written to {SUMMARY_PATH}")


def main():
    if not INPUT_CACHE.exists():
        raise SystemExit(f"missing {INPUT_CACHE}")
    if not CSV_PATH.exists():
        raise SystemExit(f"missing {CSV_PATH}")

    print(f"=== reclassify HTML-as-robots digests as uninterpretable ===")
    print(f"  cache: {INPUT_CACHE}")
    print(f"  csv:   {CSV_PATH}")
    print()
    print("Pass 1: scanning cache for HTML-without-User-agent bodies...")
    to_reclassify, audit, stats = scan_cache_for_reclassify()
    print(f"\n  -> {len(to_reclassify):,} digests will be reclassified")

    # Persist audit so user can spot-check the criterion
    print(f"\nWriting audit: {AUDIT_JSONL}")
    with open(AUDIT_JSONL, "w", encoding="utf-8", newline="\n") as f:
        for rec in audit:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\nPass 2: rewriting digest_bot_policy.csv...")
    n_total, n_replaced, post_status = rewrite_csv(to_reclassify)
    print(f"  rows total:    {n_total:,}")
    print(f"  rows replaced: {n_replaced:,}")

    print("\nPass 3: writing summary...")
    write_summary(stats, n_total, n_replaced, post_status)


if __name__ == "__main__":
    main()
