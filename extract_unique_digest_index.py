"""extract_unique_digest_index.py — scan CDX snapshots, build unique-digest
index for robots.txt content fetching.

KEY INSIGHT (per 2026-05-03 design discussion):
  Wayback's digest is a content hash (SHA-1 of body). Two snapshots with
  the same digest have the SAME robots.txt body. So fetching by digest is
  enough — we DO NOT need to fetch every (host, digest) pair, and we DO
  NOT need to fetch every (host, timestamp, digest) capture. ~25.9M raw
  CDX captures usually compress to ~50k-150k unique digests.

Output schema (unique-digest level, one row per digest):
  digest          : the Wayback digest (32-char SHA-1 base32)
  sample_host     : one host that captured this digest (for fetcher)
  sample_url      : the original URL we'll request (robots.txt)
  sample_ts       : a Wayback timestamp where this digest was 200/served
                    (use 200/revisit-resolved if available; else any 200;
                    else fall back to most-common-status)
  n_captures      : how many (host, ts) captures map to this digest globally
  hosts           : count of distinct hosts where this digest appears

Inputs:
  D:/anoconda/bot_paper_data/cdx_snapshots.jsonl
  D:/anoconda/bot_paper_data/cdx_snapshots_new.jsonl

Output:
  E:/bot_paper_data/cdx_unique_digest_index.jsonl

This index drives the daily-level robots.txt fetcher. The fetcher reads
each digest, picks sample_url + sample_ts, fetches once via Wayback, parses
bot-block status, writes (digest, parsed_bot_policy) row. Joining back to
the full panel of (host, timestamp, digest) captures reconstructs the
daily-level bot-block timeline.

Status filtering for sample_ts selection:
  Wayback CDX rows have a status code at fetch time. We pick the sample to
  serve as the canonical fetch:
    1) Prefer status=='200' (or 200-equivalent revisit). The robots.txt
       content was actually served then; parser will work on it directly.
    2) If no 200 ever for this digest, fall back to the modal status — but
       these rare digests probably represent error pages cached by Wayback
       and parsing will produce unknown_root_tag_html or similar. Mark with
       sample_status so downstream can decide whether to fetch.

NOTE on Step 2 tier semantics: this index is JUST for sample selection.
The downstream timeline reconstruction must apply the full Step 2 status
mapping (200/revisit→parse; 404/410→allow-all; 3xx/403/5xx→uninterpretable;
no capture→gap). That logic lives in the timeline-reconstruction script,
not here.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

INPUTS = [
    Path(r"D:\anoconda\bot_paper_data\cdx_snapshots.jsonl"),
    Path(r"D:\anoconda\bot_paper_data\cdx_snapshots_new.jsonl"),
]
OUTPUT = Path(r"E:\bot_paper_data\cdx_unique_digest_index.jsonl")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    # digest -> {sample fields, capture stats}
    # We keep one canonical (host, ts, status, url) sample per digest, plus
    # global counts.
    digest_info = {}
    n_hosts = 0
    n_captures = 0

    print(f"Reading CDX snapshots from {len(INPUTS)} files...")
    for in_path in INPUTS:
        if not in_path.exists():
            print(f"  MISSING: {in_path}")
            continue
        with open(in_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                host = r.get("host", "")
                snaps = r.get("snapshots", []) or []
                if not host or not snaps:
                    continue
                n_hosts += 1
                for s in snaps:
                    if not s or len(s) < 5:
                        continue
                    ts, status, digest, length, url = s[0], s[1], s[2], s[3], s[4]
                    if not digest:
                        continue
                    n_captures += 1
                    cur = digest_info.get(digest)
                    if cur is None:
                        cur = {
                            "digest": digest,
                            "sample_host": host,
                            "sample_ts": ts,
                            "sample_status": status,
                            "sample_url": url,
                            "n_captures": 0,
                            "hosts_set": set(),
                        }
                        digest_info[digest] = cur
                    cur["n_captures"] += 1
                    cur["hosts_set"].add(host)
                    # Prefer 200 sample; if current sample isn't 200 and this row is, replace
                    if cur["sample_status"] != "200" and status == "200":
                        cur["sample_host"] = host
                        cur["sample_ts"] = ts
                        cur["sample_status"] = "200"
                        cur["sample_url"] = url

    n_digests = len(digest_info)
    print(f"\nScan done.")
    print(f"  hosts (jsonl rows):        {n_hosts:>10,}")
    print(f"  raw CDX captures:          {n_captures:>10,}")
    print(f"  UNIQUE digests:            {n_digests:>10,}")
    print()

    # status histogram for selected samples
    status_hist = Counter(d["sample_status"] for d in digest_info.values())
    print("Sample status distribution (one sample per digest):")
    for st, cnt in status_hist.most_common(15):
        pct = 100.0 * cnt / max(1, n_digests)
        print(f"  {st:<20} {cnt:>10,}  ({pct:>5.1f}%)")
    print()

    # cross-host sharing histogram
    host_count_hist = Counter()
    for d in digest_info.values():
        host_count_hist[len(d["hosts_set"])] += 1
    print("How many hosts share each digest:")
    bucket_total = 0
    for k in [1, 2, 3, 5, 10, 50, 100, 1000]:
        c = sum(v for n, v in host_count_hist.items() if (k <= n < k * 2 if k < 1000 else n >= 1000))
        bucket_total += c
        print(f"  digest in [{k}{'-'+str(k*2-1) if k < 1000 else '+'} hosts]:  {c:>8,}")
    print()

    # capture-count histogram
    n_cap_hist = Counter()
    for d in digest_info.values():
        c = d["n_captures"]
        n_cap_hist[c] += 1
    print("How many captures per digest (N → captures-with-this-N):")
    for k in [1, 2, 5, 10, 100, 1000, 10000]:
        c = sum(v for n, v in n_cap_hist.items() if (k <= n < k * 2 if k < 10000 else n >= 10000))
        print(f"  captures in [{k}{'-'+str(k*2-1) if k < 10000 else '+'}]:  {c:>8,}")
    print()

    # write output
    print(f"Writing {OUTPUT}")
    with open(OUTPUT, "w", encoding="utf-8", newline="\n") as out:
        for d in digest_info.values():
            row = {
                "digest":         d["digest"],
                "sample_host":    d["sample_host"],
                "sample_ts":      d["sample_ts"],
                "sample_status":  d["sample_status"],
                "sample_url":     d["sample_url"],
                "n_captures":     d["n_captures"],
                "n_hosts":        len(d["hosts_set"]),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. {n_digests:,} unique digests written.")
    print(f"OUTPUT size: {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
