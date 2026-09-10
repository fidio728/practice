"""dedup_first_block_events.py — produce two clean derivatives of
daily_first_block_events.csv.

Problem (user audit 2026-05-03):
  Step C reads CDX line-by-line. The CDX input file lists the same host
  more than once for ~1,665 hosts (a host can map to multiple firms /
  gvkeys, and the upstream CDX query loop emitted one record per
  firm-host pair). Step C therefore wrote per-(host, bot) rows once for
  each appearance, producing 736,851 rows over 621,966 unique (host,
  bot) pairs (114,885 perfect duplicates).

  This is a presentation-layer issue, not a correctness issue: the
  computed (first_observed_blocked_ts, last_unblocked_ts, …) values are
  identical across the duplicates (same host → same CDX timeline → same
  derived values). But downstream code that joins by (host, bot) gets
  silently inflated counts, and code that wants firm-level analysis
  needs a gvkey column that Step C did not carry through.

This script produces two derivatives:

1. daily_first_block_events_hostbot_dedup.csv
     - One row per (host, bot). Drops perfect duplicates.
     - Rows are unchanged otherwise.
     - Use for host-level descriptive stats and the first-cut event
       study.

2. daily_first_block_events_firmhostbot.csv
     - Inner-join with the firm→host mapping
       (bot_paper/compustat_all_host_mapping.csv) to attach gvkey.
     - One row per (gvkey, host, bot). When a host maps to multiple
       gvkeys, the row is duplicated per gvkey (this is intentional —
       it is the input for firm-level regressions / merges with the
       monthly panel, where the unit of analysis is the firm-month).
     - Hosts in events but missing from the mapping are dropped from
       this output; they're still in the dedup output.

The original daily_first_block_events.csv is left untouched.
"""

import csv
from collections import Counter
from pathlib import Path

DATA_DIR = Path("E:/bot_paper_data")
EVENTS_CSV = DATA_DIR / "daily_first_block_events.csv"
MAPPING_CSV = Path(r"c:\Users\xl\OneDrive - Universitat Ramón Llull\bot_paper\compustat_all_host_mapping.csv")

DEDUP_OUT  = DATA_DIR / "daily_first_block_events_hostbot_dedup.csv"
FIRM_OUT   = DATA_DIR / "daily_first_block_events_firmhostbot.csv"
SUMMARY    = DATA_DIR / "daily_first_block_events_dedup_summary.txt"


def load_firm_host_mapping():
    """Return dict[host] -> list[gvkey]. Multiple gvkeys per host are
    preserved (corporate hierarchies, M&A)."""
    h2g = {}
    n_rows = 0
    with open(MAPPING_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            n_rows += 1
            host = (r.get("host_clean") or "").strip()
            gv = (r.get("gvkey") or "").strip()
            if not host or not gv:
                continue
            h2g.setdefault(host, []).append(gv)
    print(f"  mapping: {n_rows:,} rows, {len(h2g):,} unique hosts")
    n_multi = sum(1 for v in h2g.values() if len(v) > 1)
    print(f"  hosts with >1 gvkey: {n_multi:,}")
    return h2g


def main():
    print(f"Reading {EVENTS_CSV}")

    seen = {}    # (host, bot) -> row dict (first occurrence wins)
    n_rows_total = 0
    n_dups_dropped = 0
    n_dup_conflicts = 0
    fieldnames = None

    with open(EVENTS_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for r in reader:
            n_rows_total += 1
            key = (r["host"], r["bot"])
            prev = seen.get(key)
            if prev is None:
                seen[key] = r
            else:
                n_dups_dropped += 1
                # Sanity check: dup rows should be identical
                if (prev.get("first_observed_blocked_ts") != r.get("first_observed_blocked_ts")
                        or prev.get("last_unblocked_ts") != r.get("last_unblocked_ts")
                        or prev.get("ever_blocked") != r.get("ever_blocked")):
                    n_dup_conflicts += 1

    print(f"  rows in input:          {n_rows_total:,}")
    print(f"  unique (host, bot):     {len(seen):,}")
    print(f"  perfect dup rows dropped:{n_dups_dropped:,}")
    print(f"  duplicate-key conflicts:{n_dup_conflicts:,} "
          f"({'OK' if n_dup_conflicts == 0 else 'WARN: divergent rows for same key'})")

    # 1. (host, bot) dedup output
    print(f"\nWriting {DEDUP_OUT}")
    with open(DEDUP_OUT, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for (host, bot), row in sorted(seen.items()):
            writer.writerow(row)
    print(f"  rows written: {len(seen):,}")

    # 2. firm-host-bot output (inner-join with mapping)
    print(f"\nReading firm-host mapping: {MAPPING_CSV}")
    h2g = load_firm_host_mapping()

    firm_fieldnames = ["gvkey"] + fieldnames
    n_firm_rows = 0
    n_hosts_with_mapping = 0
    n_hosts_no_mapping = 0
    n_pairs_missing_mapping = 0
    hosts_in_events = set(host for host, _ in seen.keys())

    print(f"\nWriting {FIRM_OUT}")
    with open(FIRM_OUT, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=firm_fieldnames)
        writer.writeheader()
        for (host, bot), row in sorted(seen.items()):
            gvkeys = h2g.get(host)
            if not gvkeys:
                n_pairs_missing_mapping += 1
                continue
            for gv in gvkeys:
                out_row = {"gvkey": gv, **row}
                writer.writerow(out_row)
                n_firm_rows += 1
    n_hosts_with_mapping = sum(1 for h in hosts_in_events if h in h2g)
    n_hosts_no_mapping = len(hosts_in_events) - n_hosts_with_mapping
    print(f"  rows written:                          {n_firm_rows:,}")
    print(f"  unique events-hosts:                   {len(hosts_in_events):,}")
    print(f"  events-hosts in mapping:               {n_hosts_with_mapping:,}")
    print(f"  events-hosts missing from mapping:     {n_hosts_no_mapping:,}")
    print(f"  (host,bot) pairs dropped (no mapping): {n_pairs_missing_mapping:,}")

    # Distribution of #gvkeys per events-host
    gvkey_count = Counter(len(h2g[h]) for h in hosts_in_events if h in h2g)
    print(f"\n  #gvkeys per events-host distribution (top):")
    for k in sorted(gvkey_count)[:8]:
        print(f"    {k} gvkey(s): {gvkey_count[k]:,} hosts")

    # Summary file
    lines = []
    lines.append("daily_first_block_events dedup + firm-merge summary")
    lines.append("")
    lines.append(f"Source: {EVENTS_CSV} ({n_rows_total:,} rows)")
    lines.append(f"")
    lines.append(f"=== (host, bot) DEDUP ===")
    lines.append(f"Output:                            {DEDUP_OUT}")
    lines.append(f"Unique (host, bot) pairs:          {len(seen):,}")
    lines.append(f"Perfect dup rows dropped:          {n_dups_dropped:,}")
    lines.append(f"Duplicate-key conflicts:           {n_dup_conflicts:,}")
    lines.append("")
    lines.append(f"=== FIRM-HOST-BOT (gvkey merge) ===")
    lines.append(f"Mapping table:                     {MAPPING_CSV}")
    lines.append(f"Output:                            {FIRM_OUT}")
    lines.append(f"Rows written:                      {n_firm_rows:,}")
    lines.append(f"Unique events-hosts:               {len(hosts_in_events):,}")
    lines.append(f"Events-hosts in mapping:           {n_hosts_with_mapping:,}")
    lines.append(f"Events-hosts missing from mapping: {n_hosts_no_mapping:,}")
    lines.append(f"(host,bot) pairs dropped:          {n_pairs_missing_mapping:,}")
    lines.append("")
    lines.append("Distribution of #gvkeys per events-host:")
    for k in sorted(gvkey_count):
        lines.append(f"  {k} gvkey(s): {gvkey_count[k]:,} hosts")
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary: {SUMMARY}")


if __name__ == "__main__":
    main()
