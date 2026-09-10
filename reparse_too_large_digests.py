"""reparse_too_large_digests.py — full-rigor pass over digests that the
main parser skipped due to the 500 KB body cap.

Approach (memory-conscious one-at-a-time):
  1. Read digest_bot_policy.csv → set of digests marked "too_large".
  2. Single pass over digest_robots_cache.jsonl. For each line, do a cheap
     bytes-level digest extraction first; only when the digest is in the
     too_large set do we decode + json.loads the full line. This keeps the
     hot path (99k+ skip) cheap.
  3. For each matching digest, run the FULL parser/classifier with no
     size cap, then immediately drop the body and run gc.collect() before
     processing the next.
  4. Build new row for the digest with parse_status="ok" (or
     "parse_error_*" if parse_robots raises).
  5. Rewrite digest_bot_policy.csv with the updated rows merged in.

Why single-pass vs offset-seek: simpler, no index file, and at 262
too-large digests / 99,745 total = 0.26% hit rate, the cost of regex-only
on the 99,483 misses is negligible (microseconds each).
"""

import csv
import gc
import json
import os
import re
import sys
import time
from importlib import util as _import_util
from pathlib import Path

DATA_DIR = Path("E:/bot_paper_data")
INPUT_CACHE = DATA_DIR / "digest_robots_cache.jsonl"
CSV_PATH = DATA_DIR / "digest_bot_policy.csv"
TMP_CSV = CSV_PATH.with_suffix(".reparse.csv")
SUMMARY_PATH = DATA_DIR / "digest_bot_policy_reparse_summary.txt"

BOT_PAPER_ROOT = Path(r"c:\Users\xl\OneDrive - Universitat Ramón Llull\bot_paper")
_DIGEST_RE = re.compile(rb'"digest"\s*:\s*"([^"]+)"')


def _import_main_line_modules():
    sys.path.insert(0, str(BOT_PAPER_ROOT))
    sys.path.insert(0, str(BOT_PAPER_ROOT / "panel_construction"))
    import robots_parser as rp
    panel04 = BOT_PAPER_ROOT / "panel_construction" / "04_parse_panel.py"
    spec = _import_util.spec_from_file_location("step04", panel04)
    step04 = _import_util.module_from_spec(spec)
    spec.loader.exec_module(step04)
    return rp, step04.BOT_TAXONOMY


def main():
    rp, BOT_TAXONOMY = _import_main_line_modules()
    bots = list(BOT_TAXONOMY.keys())
    print(f"Loaded {len(bots)} bots")

    # 1. Identify too_large digests from current CSV
    print(f"\nReading {CSV_PATH}")
    too_large = set()
    fieldnames = None
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            if r.get("parse_status") == "too_large":
                too_large.add(r["digest"])
    print(f"  too_large digests to reparse: {len(too_large):,}")
    if not too_large:
        print("Nothing to do.")
        return

    # 2. Single pass over jsonl, full-parse only matching digests
    print(f"\nScanning {INPUT_CACHE} for matching bodies...")
    new_rows = {}     # digest -> updated CSV row dict
    n_lines = n_full = n_parsed = n_perr = 0
    t0 = time.monotonic()

    # Use binary readline with a generous max so a single 8 MB line fits
    # whole. With 262 too-large digests handled one at a time + gc.collect
    # between, peak memory stays well under 100 MB even on tight hosts.
    READLINE_MAX = 50_000_000   # 50 MB ceiling; current max in cache is 8 MB
    with open(INPUT_CACHE, "rb") as fin:
        while True:
            line_b = fin.readline(READLINE_MAX)
            if not line_b:
                break
            n_lines += 1
            # Drain residual bytes if line was longer than READLINE_MAX
            # (shouldn't happen given the 50 MB ceiling, but be safe).
            while not line_b.endswith(b"\n") and len(line_b) >= READLINE_MAX:
                extra = fin.readline(READLINE_MAX)
                if not extra:
                    break
                line_b = line_b + extra

            m = _DIGEST_RE.search(line_b)
            if not m:
                continue
            digest = m.group(1).decode("utf-8", errors="replace")
            if digest not in too_large:
                continue
            # Full path
            n_full += 1
            try:
                line = line_b.decode("utf-8")
                r = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                row = {"digest": digest,
                       "parse_status": f"parse_error_{type(e).__name__}",
                       "n_groups": "", "n_disallow_rules": ""}
                new_rows[digest] = row
                continue
            content = r.get("content_text") or ""
            if not content:
                row = {"digest": digest, "parse_status": "uninterpretable",
                       "n_groups": "", "n_disallow_rules": ""}
                new_rows[digest] = row
                continue
            try:
                groups = rp.parse_robots(content)
                n_dis = sum(1 for g in groups for d, _ in g["rules"]
                            if d == "disallow")
                row = {"digest": digest, "parse_status": "ok",
                       "n_groups": len(groups), "n_disallow_rules": n_dis}
                for bot_name in bots:
                    aliases = BOT_TAXONOMY[bot_name]["aliases"]
                    cls = rp.classify_bot(groups, bot_name, aliases)
                    row[f"{bot_name}_root_block"] = cls["root_block"]
                    row[f"{bot_name}_has_nonroot_disallow"] = cls["has_nonroot_disallow"]
                    row[f"{bot_name}_specific_rule"] = cls["specific_rule"]
                    row[f"{bot_name}_effective_source"] = cls["effective_source"]
                new_rows[digest] = row
                n_parsed += 1
                print(f"  [{n_full:>3}] {digest}  body={len(content):,}B "
                      f"groups={len(groups)} disallow={n_dis}", flush=True)
            except Exception as e:
                row = {"digest": digest,
                       "parse_status": f"parse_error_{type(e).__name__}",
                       "n_groups": "", "n_disallow_rules": ""}
                new_rows[digest] = row
                n_perr += 1
                print(f"  [{n_full:>3}] {digest}  parse_error: {type(e).__name__}",
                      flush=True)
            finally:
                # Aggressively release the big content + r dict before next
                # too_large digest hits us.
                content = None
                r = None
                line = None
                gc.collect()

            line_b = None

    dt = time.monotonic() - t0
    print(f"\nScan done. lines={n_lines:,} matched={n_full:,} "
          f"parsed_ok={n_parsed:,} parse_errors={n_perr:,} "
          f"in {dt:.1f}s")

    # 3. Rewrite CSV with updated rows
    print(f"\nRewriting {CSV_PATH} (merging {len(new_rows):,} updated rows)...")
    n_replaced = 0
    n_total = 0
    with open(CSV_PATH, encoding="utf-8", newline="") as fin, \
         open(TMP_CSV, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for r in reader:
            n_total += 1
            d = r["digest"]
            if d in new_rows:
                merged = {fn: "" for fn in fieldnames}
                merged.update(new_rows[d])
                # Make sure digest survives merge correctly
                merged["digest"] = d
                writer.writerow(merged)
                n_replaced += 1
            else:
                writer.writerow(r)
    os.replace(TMP_CSV, CSV_PATH)
    print(f"  rows in CSV:    {n_total:,}")
    print(f"  rows replaced:  {n_replaced:,}")

    # Summary
    lines = []
    lines.append("reparse_too_large_digests.py — summary")
    lines.append("")
    lines.append(f"too_large digests in CSV before reparse: {len(too_large):,}")
    lines.append(f"matched in cache:                        {n_full:,}")
    lines.append(f"parsed_ok:                                {n_parsed:,}")
    lines.append(f"parse_errors:                             {n_perr:,}")
    lines.append(f"rows in CSV after reparse:                {n_total:,}")
    lines.append(f"rows replaced:                            {n_replaced:,}")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
