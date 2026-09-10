"""parse_digest_robots_to_bot_policy.py — Step B of the daily robots
transition pipeline.

For each digest in digest_robots_cache.jsonl, parse the robots.txt body and
emit a per-digest, per-bot policy table. Local script, no network.

Reuses the main-line parser at bot_paper/robots_parser.py and the canonical
69-bot BOT_TAXONOMY from bot_paper/panel_construction/04_parse_panel.py
unchanged. The daily timeline must agree with the monthly EOM panel on bot
classification, so we MUST share these two modules — do NOT reimplement.

Inputs:
  E:/bot_paper_data/digest_robots_cache.jsonl       (~99,745 rows)
  bot_paper/robots_parser.py                         (parse_robots, classify_bot)
  bot_paper/panel_construction/04_parse_panel.py     (BOT_TAXONOMY)

Outputs (E:/bot_paper_data):
  digest_bot_policy.csv         wide CSV, one row per digest:
                                  digest, parse_status, n_groups, n_disallow_rules,
                                  <bot>_root_block, <bot>_has_nonroot_disallow,
                                  <bot>_specific_rule, <bot>_effective_source
                                for each of 69 bots in BOT_TAXONOMY.
                                For digests with no parseable body
                                (transient / det_fail / empty), parse_status
                                is "uninterpretable" and per-bot columns are
                                empty (treated as NA in Step C).
  digest_bot_policy_summary.txt counts of digests by parse_status, per-bot
                                blocked-count summary.

parse_status values:
  ok                   body parsed, classify_bot ran successfully
  uninterpretable      no body (Step A error / empty / det_fail)
  parse_error          body present but parse_robots raised; should be 0
                       in practice (parse_robots is permissive)

Note on `no_robots` semantics:
  This script does NOT emit the "no robots policy = allow-all" Step 2 row.
  That happens in Step C, when we walk the per-host CDX timeline and see a
  capture with status 404/410: we treat the bot as ALLOWED at that ts,
  regardless of digest. Step B is purely "parse the bodies we did fetch."
"""

import csv
import gc
import json
import re
import sys
import time
from collections import Counter
from importlib import util as _import_util
from pathlib import Path

# Robots.txt bodies above this size are almost always not real robots files
# (HTML error pages, log dumps, etc.). Parsing them is slow and the result
# is junk anyway — emit "too_large" and let Step C treat them as
# uninterpretable. This also bounds per-iteration memory pressure that was
# previously triggering MemoryError around digest #36k.
MAX_CONTENT_BYTES = 500_000

# A jsonl line larger than this is decoded only enough to extract the
# digest field via regex; the body is never loaded into Python as a string.
# This protects the parser when the user's machine is under memory pressure
# (we hit MemoryError on a 8MB jsonl line on a host with 880MB free RAM).
MAX_LINE_BYTES_FOR_FULL_PARSE = 800_000

_DIGEST_RE = re.compile(rb'"digest"\s*:\s*"([^"]+)"')

DATA_DIR = Path("E:/bot_paper_data")
INPUT_CACHE = DATA_DIR / "digest_robots_cache.jsonl"
OUTPUT_CSV = DATA_DIR / "digest_bot_policy.csv"
OUTPUT_SUMMARY = DATA_DIR / "digest_bot_policy_summary.txt"

# bot_paper/robots_parser.py is at this path on the user's machine. Adjust
# if the repo moves.
BOT_PAPER_ROOT = Path(r"c:\Users\xl\OneDrive - Universitat Ramón Llull\bot_paper")


def _import_main_line_modules():
    """Import the main-line robots_parser + BOT_TAXONOMY from bot_paper.
    Done dynamically because bot_paper isn't an installed package."""
    if not BOT_PAPER_ROOT.exists():
        sys.exit(f"BOT_PAPER_ROOT not found: {BOT_PAPER_ROOT}")
    sys.path.insert(0, str(BOT_PAPER_ROOT))
    sys.path.insert(0, str(BOT_PAPER_ROOT / "panel_construction"))
    import robots_parser as rp  # noqa: E402

    # 04_parse_panel.py starts with a digit; can't import via `import`.
    panel04_path = BOT_PAPER_ROOT / "panel_construction" / "04_parse_panel.py"
    spec = _import_util.spec_from_file_location("step04_parse_panel", panel04_path)
    step04 = _import_util.module_from_spec(spec)
    spec.loader.exec_module(step04)
    return rp, step04.BOT_TAXONOMY


def main():
    rp, BOT_TAXONOMY = _import_main_line_modules()
    bots = list(BOT_TAXONOMY.keys())
    print(f"Loaded {len(bots)} bots from BOT_TAXONOMY")
    print(f"  sample: {bots[:5]} ... {bots[-3:]}")

    if not INPUT_CACHE.exists():
        sys.exit(f"input not found: {INPUT_CACHE}")

    fieldnames = ["digest", "parse_status", "n_groups", "n_disallow_rules"]
    for b in bots:
        fieldnames.extend([
            f"{b}_root_block",
            f"{b}_has_nonroot_disallow",
            f"{b}_specific_rule",
            f"{b}_effective_source",
        ])

    counts_status = Counter()
    bot_blocked_count = Counter()
    bot_specific_count = Counter()
    n_total = 0
    t0 = time.monotonic()

    # Resume strategy: input is sorted by digest (merge_digest_robots_cache.py
    # writes `for d in sorted(best)`). The CSV is appended in input order, so
    # to resume we just need the last successfully written digest — we don't
    # need to materialize a 99k-element done set in memory.
    last_done_digest = None
    write_mode = "w"
    if OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0:
        with open(OUTPUT_CSV, "rb") as fexist:
            fexist.seek(0, 2)
            size = fexist.tell()
            tail_bytes = 8192
            fexist.seek(max(0, size - tail_bytes))
            tail = fexist.read()
        # Last non-empty line that isn't the header
        for line_b in reversed(tail.splitlines()):
            if not line_b.strip():
                continue
            txt = line_b.decode("utf-8", errors="replace")
            if txt.startswith("digest,"):
                continue
            last_done_digest = txt.split(",", 1)[0]
            break
        if last_done_digest:
            print(f"\nResuming after last digest: {last_done_digest}")
            write_mode = "a"

    print(f"\nWriting {OUTPUT_CSV} (mode={write_mode})")
    # Open input in BINARY mode and read with explicit readline(MAX_BYTES)
    # so Python's line iterator never allocates more than MAX_BYTES per
    # call. On low-memory hosts the default `for line in fin:` was OS-
    # killed silently when it tried to buffer an 8 MB jsonl line.
    READLINE_MAX = 500_000
    with open(INPUT_CACHE, "rb") as fin, \
         open(OUTPUT_CSV, write_mode, encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        if write_mode == "w":
            writer.writeheader()
        while True:
            line_b = fin.readline(READLINE_MAX)
            if not line_b:
                break
            # If this chunk does not end with \n, the underlying jsonl line
            # was longer than READLINE_MAX. Drain the rest of the line in
            # fixed-size pieces WITHOUT keeping them; we already have enough
            # bytes from the head to regex-extract the digest, since
            # jsonl rows put "digest":"..." very early in the row.
            if not line_b.endswith(b"\n"):
                head = line_b
                drained = 0
                while True:
                    extra = fin.readline(READLINE_MAX)
                    if not extra:
                        break
                    drained += len(extra)
                    if extra.endswith(b"\n"):
                        break
                m = _DIGEST_RE.search(head)
                if m:
                    digest = m.group(1).decode("utf-8", errors="replace")
                    if last_done_digest is None or digest > last_done_digest:
                        n_total += 1
                        writer.writerow({
                            "digest": digest, "parse_status": "too_large",
                            "n_groups": "", "n_disallow_rules": "",
                        })
                        counts_status["too_large"] += 1
                head = None
                continue
            try:
                if not line_b.strip():
                    continue

                # By construction line_b is <= READLINE_MAX (500 KB). The
                # oversize-line branch above already handled bigger inputs
                # via streaming drain.
                try:
                    line = line_b.decode("utf-8")
                    r = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                digest = r.get("digest")
                if not digest:
                    continue
                if last_done_digest is not None and digest <= last_done_digest:
                    continue
                n_total += 1
                content = r.get("content_text")
                row = {"digest": digest}
                if content is None or len(content) == 0:
                    row["parse_status"] = "uninterpretable"
                    row["n_groups"] = ""
                    row["n_disallow_rules"] = ""
                    writer.writerow(row)
                    counts_status["uninterpretable"] += 1
                    continue
                if len(content) > MAX_CONTENT_BYTES:
                    row["parse_status"] = "too_large"
                    row["n_groups"] = ""
                    row["n_disallow_rules"] = ""
                    writer.writerow(row)
                    counts_status["too_large"] += 1
                    # release the big string before next iteration
                    content = None
                    r = None
                    line = None
                    continue
                try:
                    groups = rp.parse_robots(content)
                except Exception as e:
                    row["parse_status"] = f"parse_error_{type(e).__name__}"
                    row["n_groups"] = ""
                    row["n_disallow_rules"] = ""
                    writer.writerow(row)
                    counts_status["parse_error"] += 1
                    continue

                n_dis = sum(1 for g in groups for d, _ in g["rules"] if d == "disallow")
                row["parse_status"] = "ok"
                row["n_groups"] = len(groups)
                row["n_disallow_rules"] = n_dis

                for bot_name in bots:
                    aliases = BOT_TAXONOMY[bot_name]["aliases"]
                    cls = rp.classify_bot(groups, bot_name, aliases)
                    row[f"{bot_name}_root_block"] = cls["root_block"]
                    row[f"{bot_name}_has_nonroot_disallow"] = cls["has_nonroot_disallow"]
                    row[f"{bot_name}_specific_rule"] = cls["specific_rule"]
                    row[f"{bot_name}_effective_source"] = cls["effective_source"]
                    if cls["root_block"]:
                        bot_blocked_count[bot_name] += 1
                    if cls["specific_rule"]:
                        bot_specific_count[bot_name] += 1

                writer.writerow(row)
                counts_status["ok"] += 1
            except MemoryError:
                # Emit a marker row so the digest is not retried on resume,
                # and force a GC sweep to release whatever held the line.
                row_err = {"digest": r.get("digest", "<unknown>") if isinstance(r, dict) else "<unknown>",
                           "parse_status": "memory_error",
                           "n_groups": "", "n_disallow_rules": ""}
                writer.writerow(row_err)
                counts_status["memory_error"] += 1
                content = None
                r = None
                line = None
                gc.collect()
                continue

            if n_total % 5000 == 0:
                fout.flush()
                gc.collect()
            if n_total % 10000 == 0:
                dt = time.monotonic() - t0
                rate = n_total / max(0.01, dt)
                print(f"  [{n_total:,}]  ok={counts_status['ok']:,} "
                      f"uninterp={counts_status['uninterpretable']:,} "
                      f"too_large={counts_status['too_large']:,}  "
                      f"{rate:,.0f} digest/s", flush=True)

    dt = time.monotonic() - t0
    print(f"\nProcessed {n_total:,} digests in {dt:.1f}s "
          f"({n_total/max(0.01,dt):,.0f} digest/s)")

    # Summary
    lines = []
    lines.append(f"parse_digest_robots_to_bot_policy.py — summary")
    lines.append(f"")
    lines.append(f"Input:    {INPUT_CACHE} ({n_total:,} digests)")
    lines.append(f"Output:   {OUTPUT_CSV}")
    lines.append(f"")
    lines.append(f"Parse-status breakdown:")
    for st in ("ok", "uninterpretable", "parse_error"):
        c = counts_status.get(st, 0)
        pct = 100.0 * c / max(1, n_total)
        lines.append(f"  {st:<20} {c:>10,}  ({pct:>5.2f}%)")
    lines.append(f"")
    lines.append(f"Top blocked-bot counts (digests where root_block==1):")
    for bot, c in bot_blocked_count.most_common(25):
        pct = 100.0 * c / max(1, counts_status['ok'])
        lines.append(f"  {bot:<35} {c:>8,}  ({pct:>5.2f}% of ok digests)")
    lines.append(f"")
    lines.append(f"Top specific-rule bot counts (digests where specific_rule==1):")
    for bot, c in bot_specific_count.most_common(25):
        pct = 100.0 * c / max(1, counts_status['ok'])
        lines.append(f"  {bot:<35} {c:>8,}  ({pct:>5.2f}% of ok digests)")

    out_text = "\n".join(lines) + "\n"
    print("\n" + out_text)
    OUTPUT_SUMMARY.write_text(out_text, encoding="utf-8")
    print(f"Summary written to {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
