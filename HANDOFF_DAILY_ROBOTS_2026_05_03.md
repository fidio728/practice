# Handoff — Daily robots transition pipeline, paused 2026-05-03 PM

Companion to `DAILY_ROBOTS_TRANSITION_PROJECT.md` (the pipeline design
doc). This file captures **what got built so far + the open semantic bug
+ the planned fix**, so the next session resumes cold.

## State summary

| Stage | Status | Output | Notes |
|---|---|---|---|
| Step A: digest fetch | ✅ done, 5 rounds | `E:\bot_paper_data\digest_robots_cache.jsonl` (99,745 rows, 99.50% with body) | 16-machine round 1 + 7-machine rounds 2-5 |
| Step B: per-bot policy | ⚠ done but BUGGY | `E:\bot_paper_data\digest_bot_policy.csv` (99,745 rows) | 40,324 HTML bodies misclassified as ok+allow-all |
| Step C: timeline → first-block | ⏸ not started | `daily_first_block_events.csv` | unblocked-by-only-interpretable rule pending |

## The bug to fix first

`parse_status=ok` rows in the current CSV include 41,741 with `n_groups=0`,
breakdown by content audit:

| content type | count | RFC 9309 verdict |
|---|---|---|
| HTML challenge / error page (Cloudflare, "Please wait while your request is being verified", server 5xx HTML) | **40,324** | NOT a valid robots.txt — should be NA |
| Sitemap-only (just `Sitemap:` lines) | 587 | valid allow-all |
| Comments-only or short non-robots text | 231 | valid allow-all |
| Allow-only without User-agent | 20 | valid allow-all |

`parse_robots()` from `bot_paper/robots_parser.py` is permissive — HTML
goes in, `groups=[]` comes out, every bot gets `root_block=0`. Step C
would treat those 40,324 captures as "explicitly unblocked" and inject
fake transition windows.

## Fix plan (write `reclassify_html_as_uninterpretable.py`)

```python
# Pass 1: scan digest_robots_cache.jsonl with binary streaming-drain
# reader (same memory pattern as parse_digest_robots_to_bot_policy.py).
# For each digest with content_text:
#   has_html = any HTML marker in body (case-insensitive):
#              <html, <!doctype, <head>, <body>, <title>, <meta
#   has_user_agent = any line whose stripped lowercase prefix is
#                    "user-agent:"
#   if has_html AND not has_user_agent:
#       to_reclassify.add(digest)
#
# Pass 2: rewrite digest_bot_policy.csv:
#   for each row in CSV:
#       if row.digest in to_reclassify:
#           parse_status = "uninterpretable"
#           n_groups = ""
#           n_disallow_rules = ""
#           all 69 × 4 per-bot columns = ""
#       writer.writerow(row)
#   atomic replace.
#
# Pass 3: regenerate digest_bot_policy_summary.txt from the corrected CSV.
```

Expected post-fix counts:
- ok: 99,246 → ~58,922
- uninterpretable: 499 → ~40,823

## Step C plan (after fix)

`reconstruct_first_block_events.py`:

1. Load `digest_bot_policy.csv` (corrected) keyed by digest.
2. Load CDX raw timeline from `D:\anoconda\bot_paper_data\cdx_snapshots.jsonl` + `cdx_snapshots_new.jsonl`.
3. For each (host, bot) pair, walk captures by timestamp:
   - **interpretable**:
     - status=200 + digest has `parse_status=ok` → blocked = `<bot>_root_block`
     - status=404 or 410 → allowed (no robots = allow-all)
   - **uninterpretable**: everything else (3xx, 403, 5xx, 429, parse_status=uninterpretable, missing digest in policy)
4. Find first interpretable capture where bot is blocked AND most recent earlier interpretable capture had bot allowed. Emit:
   - `first_observed_blocked_ts`, `last_unblocked_ts`, `interval_days`, `ever_blocked`, `n_captures_total`, `n_interpretable_total`
5. Save `E:\bot_paper_data\daily_first_block_events.csv`. User filters
   `interval_days <= 14` for event-study sub-sample.

## Files inventory (E:\bot_paper_data\)

```
cdx_unique_digest_index.jsonl              — 425,329 digests
cdx_digest_to_fetch.jsonl                  — 99,745 (sample_status==200 subset)
digest_robots_cache_m{0..15}.jsonl         — round-1 raw (16 shards)
digest_robots_cache_round2_m{0..6}.jsonl   — round-2 (rate=0.7)
digest_robots_cache_round3_m{0..6}.jsonl   — round-3 (rate=1.5)
digest_robots_cache_round4_m{0..6}.jsonl   — round-4 (rate=2.5)
digest_robots_cache_round5_m{0..6}.jsonl   — round-5 (rate=3.5)
digest_robots_cache.jsonl                  — UNIFIED: 99,745 dedup, 99.50% ok body
digest_robots_cache_retry_queue.jsonl      — 465 still-failed digests (NOT retried further)
digest_robots_cache_summary.txt            — Step A summary (post-round-5)
digest_bot_policy.csv                      — Step B output (BUGGY, fix pending)
digest_bot_policy_summary.txt              — STALE (claims 74,115); regenerate after fix
digest_bot_policy_reparse_summary.txt      — 262 too-large reparse verdict (clean)
```

## Files inventory (practice/)

```
extract_unique_digest_index.py             — Step A pre-processor
06_robots_daily_fetch_digests.py           — Step A fetcher
deploy_06_robots_daily.sh                  — push/launch/count/tail/pull for 16 machines
merge_digest_robots_cache.py               — Step A round merger; EXTRA_ROUNDS list now includes 3,4,5
parse_digest_robots_to_bot_policy.py       — Step B parser (binary reader + 500K size cap; resume-able)
dedup_digest_bot_policy.py                 — collapse Step B CSV duplicates
reparse_too_large_digests.py               — full-rigor pass for too_large bodies (already run)
DAILY_ROBOTS_TRANSITION_PROJECT.md         — design doc (start of project)
HANDOFF_DAILY_ROBOTS_2026_05_03.md         — this file
```

## Cloud VM hygiene

All 16 round-1 + 7 round-2..5 machines: fetch processes exited, but VMs
are likely still running and billing. Roster in
`DAILY_ROBOTS_TRANSITION_PROJECT.md` §"Available infrastructure".

User should release them in respective consoles (GCP / DigitalOcean) when
no longer needed for any future round.
