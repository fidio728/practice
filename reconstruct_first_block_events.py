"""reconstruct_first_block_events.py — Step C of the daily robots
transition pipeline.

For each (host, bot) pair, walk the host's full CDX timeline and find
the first interpretable capture where the bot is blocked AND the most
recent earlier interpretable capture had the bot allowed. Emit one row
per (host, bot) pair with timestamp + interval-censoring metadata.

Inputs:
  E:/bot_paper_data/digest_bot_policy.csv          (Step B output, post-reclassify)
  D:/anoconda/bot_paper_data/cdx_snapshots.jsonl
  D:/anoconda/bot_paper_data/cdx_snapshots_new.jsonl

Output:
  E:/bot_paper_data/daily_first_block_events.csv

Per-capture interpretation rules (must match Step 2 EOM main-line semantics):
  status==200 AND digest has parse_status=="ok"
      → bot_state = root_block per parsed policy (0=allowed, 1=blocked)
  status==404 OR status==410
      → bot_state = allowed (no robots policy ⇒ allow-all per RFC 9309)
  everything else (3xx, 403, 5xx, 429, 200+uninterpretable_body,
                   200+digest missing from cache, etc.)
      → uninterpretable; skip when looking for transition

Transition definition:
  first_observed_blocked_ts    = ts of first interpretable capture with bot blocked,
                                  AND latest earlier interpretable capture had bot
                                  allowed
  last_unblocked_ts            = ts of that latest earlier allowed capture
  interval_days                = (first_blocked - last_unblocked) in days

  If a host's first interpretable capture is already blocked (no earlier
  allowed observation), first_observed_blocked_ts is left blank — we
  cannot verify a transition from this archive. ever_blocked is still
  set in that case.

Schema:
  host, bot, bot_group,
  first_observed_blocked_ts, last_unblocked_ts, interval_days,
  ever_blocked, ever_allowed,
  n_captures_total, n_interpretable_total,
  n_blocked_for_bot, n_allowed_for_bot

Rows are emitted only when the host has ≥1 interpretable capture for
this bot (i.e., ≥1 status==200+ok-digest OR ≥1 404/410). Pairs with all
uninterpretable captures yield no row.

Memory plan:
  policy_ok[digest] = bytes(69) where bytes[i] is 1 if bot i is
                      root-blocked. ~99k digests * 69 bytes ≈ 7 MB.
  policy_other      = set of digests with parse_status != "ok"
                      (they exist in the cache but body is unusable).
                      Used to distinguish "200 + bad body" from
                      "200 + digest missing from cache" (edge case).
  CDX is streamed line-by-line; per-host memory is just one snapshot list.

Output is large (~few million rows). Stream-write CSV row by row.
"""

import csv
import gc
import json
import sys
import time
from datetime import datetime
from importlib import util as _import_util
from pathlib import Path

DATA_DIR = Path("E:/bot_paper_data")
POLICY_CSV = DATA_DIR / "digest_bot_policy.csv"
CDX_FILES = [
    Path(r"D:\anoconda\bot_paper_data\cdx_snapshots.jsonl"),
    Path(r"D:\anoconda\bot_paper_data\cdx_snapshots_new.jsonl"),
]
OUTPUT_CSV = DATA_DIR / "daily_first_block_events.csv"
OUTPUT_SUMMARY = DATA_DIR / "daily_first_block_events_summary.txt"

BOT_PAPER_ROOT = Path(r"c:\Users\xl\OneDrive - Universitat Ramón Llull\bot_paper")


def _load_bot_taxonomy():
    """Reuse the canonical 69-bot taxonomy from 04_parse_panel.py."""
    panel04 = BOT_PAPER_ROOT / "panel_construction" / "04_parse_panel.py"
    spec = _import_util.spec_from_file_location("step04", panel04)
    step04 = _import_util.module_from_spec(spec)
    spec.loader.exec_module(step04)
    return step04.BOT_TAXONOMY


def load_policy(bots):
    """Read digest_bot_policy.csv. Returns:
        policy_ok       : dict[digest] -> bytes(N_BOTS), bit per bot
        policy_other    : set[digest] (parse_status != "ok")
    """
    print(f"Loading {POLICY_CSV}")
    policy_ok = {}
    policy_other = set()
    n_ok = n_other = 0
    bot_cols = [f"{b}_root_block" for b in bots]
    t0 = time.monotonic()
    with open(POLICY_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            digest = r["digest"]
            st = r["parse_status"]
            if st == "ok":
                bv = bytearray(len(bots))
                for i, col in enumerate(bot_cols):
                    if r.get(col) == "1":
                        bv[i] = 1
                policy_ok[digest] = bytes(bv)
                n_ok += 1
            else:
                policy_other.add(digest)
                n_other += 1
    dt = time.monotonic() - t0
    print(f"  loaded {n_ok:,} ok-policy + {n_other:,} non-ok in {dt:.1f}s")
    return policy_ok, policy_other


def ts_to_date(ts14):
    return datetime.strptime(ts14[:8], "%Y%m%d")


def days_between(allowed_ts, blocked_ts):
    return (ts_to_date(blocked_ts) - ts_to_date(allowed_ts)).days


def main():
    BOT_TAXONOMY = _load_bot_taxonomy()
    BOTS = list(BOT_TAXONOMY.keys())
    BOT_GROUP = {b: BOT_TAXONOMY[b]["group"] for b in BOTS}
    N_BOTS = len(BOTS)
    print(f"Loaded {N_BOTS} bots from BOT_TAXONOMY")

    policy_ok, policy_other = load_policy(BOTS)

    ALLOW_ALL_VEC = bytes(N_BOTS)  # all zeros = nobody blocked

    fieldnames = [
        "host", "bot", "bot_group",
        "first_observed_blocked_ts", "last_unblocked_ts", "interval_days",
        "ever_blocked", "ever_allowed",
        "n_captures_total", "n_interpretable_total",
        "n_blocked_for_bot", "n_allowed_for_bot",
    ]

    print(f"\nWriting {OUTPUT_CSV}")
    n_hosts = 0
    n_rows_written = 0
    n_pairs_with_transition = 0
    n_pairs_ever_blocked = 0
    n_captures_total = 0
    t0 = time.monotonic()

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for cdx_path in CDX_FILES:
            if not cdx_path.exists():
                print(f"  MISSING: {cdx_path}")
                continue
            print(f"\nReading {cdx_path}")
            with open(cdx_path, encoding="utf-8") as fin:
                for line in fin:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    host = rec.get("host", "")
                    snaps = rec.get("snapshots", []) or []
                    if not host or not snaps:
                        continue
                    n_hosts += 1

                    # Sort by ts (CDX is usually already chronological,
                    # but be defensive).
                    snaps.sort(key=lambda s: s[0] if s else "")
                    n_total = len(snaps)
                    n_captures_total += n_total

                    # Pre-compute per-snapshot policy vector or None.
                    states = []
                    n_interp = 0
                    for s in snaps:
                        if not s or len(s) < 3:
                            states.append((None, None))
                            continue
                        ts, status, digest = s[0], s[1], s[2]
                        if status == "200":
                            vec = policy_ok.get(digest)
                            if vec is not None:
                                states.append((ts, vec))
                                n_interp += 1
                            else:
                                # 200 + digest body bad / missing → uninterpretable
                                states.append((ts, None))
                        elif status == "404" or status == "410":
                            states.append((ts, ALLOW_ALL_VEC))
                            n_interp += 1
                        else:
                            states.append((ts, None))

                    if n_interp == 0:
                        continue

                    # Per-bot accumulators.
                    ever_allowed = [False] * N_BOTS
                    ever_blocked = [False] * N_BOTS
                    n_allowed = [0] * N_BOTS
                    n_blocked = [0] * N_BOTS
                    latest_allowed_ts = [None] * N_BOTS
                    first_blocked_ts = [None] * N_BOTS
                    last_unblocked_ts = [None] * N_BOTS

                    for ts, vec in states:
                        if vec is None:
                            continue
                        # Inline 69-bot fold over the bytes vector.
                        for i in range(N_BOTS):
                            bit = vec[i]
                            if bit == 0:
                                ever_allowed[i] = True
                                n_allowed[i] += 1
                                latest_allowed_ts[i] = ts
                            else:
                                ever_blocked[i] = True
                                n_blocked[i] += 1
                                if (first_blocked_ts[i] is None
                                        and latest_allowed_ts[i] is not None):
                                    first_blocked_ts[i] = ts
                                    last_unblocked_ts[i] = latest_allowed_ts[i]

                    # Emit one row per bot with ≥1 interpretable observation.
                    for i, bot in enumerate(BOTS):
                        if not (ever_allowed[i] or ever_blocked[i]):
                            continue
                        fbt = first_blocked_ts[i]
                        lut = last_unblocked_ts[i]
                        interval = ""
                        if fbt and lut:
                            try:
                                interval = days_between(lut, fbt)
                            except Exception:
                                interval = ""
                        writer.writerow({
                            "host": host,
                            "bot": bot,
                            "bot_group": BOT_GROUP[bot],
                            "first_observed_blocked_ts": fbt or "",
                            "last_unblocked_ts": lut or "",
                            "interval_days": interval,
                            "ever_blocked": int(ever_blocked[i]),
                            "ever_allowed": int(ever_allowed[i]),
                            "n_captures_total": n_total,
                            "n_interpretable_total": n_interp,
                            "n_blocked_for_bot": n_blocked[i],
                            "n_allowed_for_bot": n_allowed[i],
                        })
                        n_rows_written += 1
                        if fbt:
                            n_pairs_with_transition += 1
                        if ever_blocked[i]:
                            n_pairs_ever_blocked += 1

                    if n_hosts % 5000 == 0:
                        dt = time.monotonic() - t0
                        rate = n_hosts / max(0.01, dt)
                        print(f"  [{n_hosts:,} hosts] rows={n_rows_written:,} "
                              f"trans={n_pairs_with_transition:,} "
                              f"ever_blk={n_pairs_ever_blocked:,}  "
                              f"{rate:,.0f} hosts/s", flush=True)

                    # Release per-host scratch.
                    states = None
                    if n_hosts % 20000 == 0:
                        gc.collect()

    dt = time.monotonic() - t0
    print(f"\nDone. {n_hosts:,} hosts in {dt:.1f}s ({n_hosts/max(0.01,dt):,.0f} hosts/s)")
    print(f"  rows written:                {n_rows_written:,}")
    print(f"  (host,bot) pairs ever_blocked:    {n_pairs_ever_blocked:,}")
    print(f"  (host,bot) pairs with transition: {n_pairs_with_transition:,}")

    # Summary
    lines = []
    lines.append("daily_first_block_events.csv — summary")
    lines.append("")
    lines.append(f"Inputs:  {POLICY_CSV}")
    for p in CDX_FILES:
        lines.append(f"         {p}")
    lines.append(f"Output:  {OUTPUT_CSV}")
    lines.append("")
    lines.append(f"Hosts processed:                       {n_hosts:,}")
    lines.append(f"Total CDX captures:                    {n_captures_total:,}")
    lines.append(f"Rows in events CSV:                    {n_rows_written:,}")
    lines.append(f"  (host,bot) pairs ever_blocked:       {n_pairs_ever_blocked:,}")
    lines.append(f"  (host,bot) pairs with transition:    {n_pairs_with_transition:,}")
    lines.append("")
    lines.append("Pairs with transition / ever_blocked is the share of blocked")
    lines.append("(host,bot) pairs where we observed an unblocked->blocked")
    lines.append("transition (vs. archive starting already-blocked).")
    lines.append("")
    lines.append(f"Wall time:                             {dt:.1f}s")
    OUTPUT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
