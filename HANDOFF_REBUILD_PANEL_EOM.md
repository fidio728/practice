# Handoff: Rebuild Wayback Panel under End-of-Month (EOM) Snapshot Rule

> **For a fresh AI session.** Self-contained context to take this from "Step 2 needs to be rewritten" to "Step 6.5 ready to restart." Read top to bottom.

**User:** Xueyang Liu, first-year PhD, ESADE Business School (Universitat Ramon Llull)
**Date:** 2026-04-28
**OS:** Windows 11; Git Bash + Stata MCP; Anaconda Python on D:\

---

## 1. The big picture (1 paragraph)

This project studies how firms' `robots.txt` configurations affect their visibility in AI-powered search systems (ChatGPT Search, Perplexity, Claude). The main treatment is **`search_sitemap_block`** = 1 if at least one URL in the firm's sitemap is blocked by at least one of three consumer AI search crawlers (OAI-SearchBot, PerplexityBot, Claude-SearchBot). A monthly panel for 2021–2025 is being built from Wayback Machine snapshots to support a staggered DiD identification. Today the supervisor flagged that the panel's monthly snapshot selection should align with month-end (not month-mid), and we audited a 26.8% work-unit waste fraction under the current day-15 selection. We are stopping the in-flight cloud computation, rebuilding Step 2 with EOM logic, then re-running Steps 3 → 4 → 6 → 6.5 → 7.

---

## 2. Today's decision and why we're rebuilding

Supervisor's panel-construction feedback (4 points; #1 and #3 are the operative ones for this rebuild):

1. **Snapshot timing should be month-end, not month-mid.** Outcomes (CRSP monthly, Compustat quarterly, I/B/E/S monthly) are reported at month-end; the treatment state must align.
2. Annual Compustat controls should be lagged or replaced by quarterly with `rdq`. (Out of scope for this handoff — it's a regression-stage change, not data-construction.)
3. Within-month variation: main spec uses month-end last snapshot; day-weighted exposure as robustness only; majority-vote rejected as a snapshot-frequency artifact.
4. CRSP filters (`EXCHD ∈ {1,2,3}`, `SHRCD ∈ {10,11}`, `L.PRC ≥ 1`, financials sometimes excluded). (Out of scope here.)

### The audit that drove the stop

I ran an audit comparing day-15 selection vs strict EOM-latest-timestamp selection across all `(host, year_month)` cells:

| Metric | Cells | % of cells | % of jobs |
|---|---|---|---|
| `single_snapshot` (no ambiguity) | 79,573 | 33.2% | — |
| `multi_same_digest` (multiple snaps, same content) | 128,243 | 53.4% | — |
| `multi_diff_digest` (EOM picks **different** digest) | 32,139 | 13.4% | — |
| **v1 cell universe (`host × ym`)** | 56,961 | — | — |
| **v1 job universe (`norm_key × ym × digest`)** | 121,366 | — | — |
| EOM differs at v1 cell level | 6,644 | **11.66%** | — |
| **EOM differs weighted by 6.5 jobs** | — | — | **~26.8%** |

The cell vs job distinction matters because cells with digest changes correlate with sitemap-rich (i.e., complex) websites, so the per-job waste fraction is more than 2× the per-cell fraction.

A second audit checked whether Step 7 join keys depend on `timestamp` (which would push the affected fraction higher). They do not — `07_build_search_sitemap_block_panel_v3.py` line 250 joins on `(norm_key, year_month, digest)`. So the "real" waste is 26.8%, not 53.1%.

### Decision

At ~4% progress on 6.5 (5,007 / 121,366 jobs done across 5 cloud machines), 26.8% future waste is large enough to justify stopping. **Stop. Rebuild Step 2 with EOM. Re-run downstream.** The previously-fetched 5,007 clean rows are preserved as 6.5 dedup material; about 73% of them stay valid under EOM and will be skipped on rerun.

---

## 3. Current state of the world (as of handoff time)

### What's running / stopped
- **5 cloud machines (AWS m0 + GCP m1/m2/m3/m4) have been stopped.** SSH'd in, `kill <PID>` (graceful, not -9), confirmed exit, downloaded the 5 shard jsonl files to local, then `aws ec2 stop-instances` / `gcloud compute instances stop`.
- **5 shard files now at:** `D:/anoconda/bot_paper_data/shards_backup/sitemap_url_paths_v2_m{0..4}.jsonl` — combined ~5,007 clean rows + some retryable / partial.
- **Original local v2 file:** `D:/anoconda/bot_paper_data/sitemap_url_paths_v2.jsonl` — pre-cloud-shard-era output (from earlier in-house runs). Same schema.

### Key files on local disk

| Path | What it is | Status |
|---|---|---|
| `D:/anoconda/bot_paper_data/cdx_snapshots.jsonl` | Raw CDX, all snapshots per host (4,427 hosts, original cohort) | ✅ Done |
| `D:/anoconda/bot_paper_data/cdx_snapshots_new.jsonl` | Same, supplementary cohort (6,669 hosts, survivorship-corrected adds) | ✅ Done |
| `D:/anoconda/bot_paper_data/monthly_snapshots.csv` | Step 2 output (current: day-15 logic) | 🟥 **Will be regenerated under EOM** |
| `D:/anoconda/bot_paper_data/monthly_snapshots_new.csv` | Same for new cohort | 🟥 **Same** |
| `D:/anoconda/bot_paper_data/robots_content.json` | Step 3 output (downloaded robots.txt by digest) | ⚠️ partial reuse — only the digests EOM picks are needed; others remain on disk harmlessly |
| `D:/anoconda/bot_paper_data/robots_content_new.json` | Same for new cohort | ⚠️ Same |
| `D:/anoconda/bot_paper_data/sitemap_panel_results.jsonl` | Step 6 v1 output (sitemap_url declarations + n_paths per (host, ym, digest)) | ⚠️ keyed by day-15 digest; need to add EOM-only digest rows |
| `D:/anoconda/bot_paper_data/sitemap_url_paths_v2.jsonl` | Step 6.5 output (URL paths per (norm_key, ym, digest)) | ⚠️ same — partial reuse via dedup |
| `D:/anoconda/bot_paper_data/shards_backup/sitemap_url_paths_v2_m{0..4}.jsonl` | 5 cloud shards just downloaded | ⚠️ merge into above before restarting 6.5 |

### Pipeline scripts
All in `C:/Users/xl/OneDrive - Universitat Ramón Llull/bot_paper/panel_construction/`:

| Script | Status / change needed |
|---|---|
| `01_query_cdx.py` | ✅ Done. No change. CDX data is already in `cdx_snapshots.jsonl` + `_new.jsonl`. |
| **`02_select_monthly.py`** | 🟥 **REWRITE: change selection rule from day-15 to EOM-latest-timestamp.** See §4. |
| `03_download_content.py` | Re-run only on the **new digests** that EOM introduces (~6,400 cells). Existing `robots_content.json` is keyed by digest → already-downloaded digests need not be re-fetched. |
| `04_parse_panel.py` | Re-run on the new full `monthly_snapshots.csv`. CPU only, fast. |
| `05_merge_firm_panel.py` | Re-run after Step 4. CPU only. |
| `06_collect_sitemaps_for_panel.py` | Re-run only on `(host, ym, digest)` cells not already in `sitemap_panel_results.jsonl`. |
| `06_5_collect_sitemap_paths_monthly.py` | After merging the 5 cloud shards into local v2, restart with new v1 input. Dedup will skip the cells that survived. |
| `07_build_search_sitemap_block_panel_v3.py` | No change. Join key is already `(norm_key, year_month, digest)`. |

---

## 4. The exact change to Step 2

### Current logic (in `02_select_monthly.py`)

For each `(host, year_month)`:
- Find all snapshots in that calendar month
- Pick the snapshot whose timestamp's **day-of-month is closest to 15**
- If multiple snapshots tie on day, the choice is order-of-iteration (effectively first-seen)

### New logic (EOM, strict latest-in-month)

For each `(host, year_month)`:
- Find all snapshots in that calendar month
- Pick the snapshot with the **maximum full timestamp** (YYYYMMDDHHMMSS) within the month — i.e., the literally last snapshot in the month, regardless of which day
- This is **strictly later** than "closest to last day" because it accounts for hour/minute/second when there are multiple snapshots on the same day

```python
# Pseudo-code:
def pick_eom_snapshot(snapshots_in_month):
    """snapshots_in_month: list of dicts/tuples with key 'timestamp' (YYYYMMDDHHMMSS string).
       Returns the one with the maximum timestamp (lexicographic order works because
       timestamps are zero-padded ISO-like)."""
    return max(snapshots_in_month, key=lambda s: s['timestamp'])
```

### Why "max timestamp" not "closest to month-end day"
Earlier audit code (in `audit_eom_gap.py`) used `pick_closest_to_day(snaps, eom_day)` which only compared `ts.day`. When a host has multiple snapshots on the same day, this returns the first one iterated, not the latest. The supervisor's intent is the actual most-recent state, which requires comparing the full timestamp.

### Files affected
- `02_select_monthly.py` — rewrite the selection lambda
- Re-run produces new `monthly_snapshots.csv` + `monthly_snapshots_new.csv`. The schema does NOT change. Columns expected:
  ```
  host, year_month, snapshot_timestamp, statuscode, digest, original_url, has_snapshot, content_changed
  ```
  `content_changed` is computed by comparing the new snapshot's digest to the previous month's chosen digest (carry the comparison through; this is unchanged logic just with new picks).

### Sanity checks after rerun
1. Total `(host, ym)` rows in `monthly_snapshots.csv` should be unchanged (same host coverage, same month coverage).
2. `digest` column will differ from current file in ~11.66% of `(host, ym)` cells.
3. `snapshot_timestamp` column will differ in ~35% of cells (because even when digest is unchanged, the picked snapshot's timestamp can be different).
4. Spot-check 5 hosts manually: open `cdx_snapshots.jsonl`, find the host, pick a multi-snapshot month, verify `monthly_snapshots.csv` has the latest-timestamp one.

---

## 5. The 5 cloud shard files — merge before restarting 6.5

The 5 shards (`sitemap_url_paths_v2_m{0..4}.jsonl` in `shards_backup/`) need to be unioned into the main local v2 file. Each shard is line-delimited JSON. Schema:

```json
{
  "norm_key": "...",
  "year_month": "2024-03",
  "digest": "ABC...XYZ",
  "host": "...",
  "timestamp": "20240331123456",
  "sitemap_url": "...",
  "n_paths": 142,
  "paths": ["/about", "/products/..."],
  "error": "",
  "had_child_errors": false,
  "url_cap_hit": false,
  "child_truncated": false
}
```

Merge strategy:
1. Read existing `D:/anoconda/bot_paper_data/sitemap_url_paths_v2.jsonl` into a `(norm_key, ym, digest) → row` map, keeping the latest line per key.
2. For each shard m0–m4, read line by line, and:
   - Skip malformed lines (a few exist from prior killed runs — `try: json.loads(line)`).
   - For each valid row, only **upsert** if the existing row (if any) is in a non-terminal state (`error != ""` or has any partial flag) and the new row is terminal-clean.
   - Definition of "terminal-clean": `error == ""` AND `not had_child_errors` AND `not url_cap_hit` AND `not child_truncated`. Reference: `06_5_collect_sitemap_paths_monthly.py` `load_v2_status()`.
3. Write the merged result back to `sitemap_url_paths_v2.jsonl` (or to a new file `sitemap_url_paths_v2_merged.jsonl` — safer, then rename).

Expected outcome: ~5,000 net new clean rows added to local v2.

---

## 6. Pipeline to execute (in order)

### Step A — Audit / verify the shard merge
- Backup current `sitemap_url_paths_v2.jsonl` to `sitemap_url_paths_v2_pre_eom_backup.jsonl`.
- Merge 5 cloud shards as in §5.
- After merge, run a quick scan: count distinct `(norm_key, ym, digest)` keys, count terminal-clean keys, compare to pre-merge.

### Step B — Rewrite and run Step 2 under EOM
- Edit `02_select_monthly.py` per §4.
- Run on both cohorts:
  ```
  python 02_select_monthly.py  # original cohort, writes monthly_snapshots.csv
  python 02_select_monthly.py --new   # if there's a flag for the new cohort
  ```
  Or whatever the existing flag pattern is — read the script first.
- Verify the four sanity checks in §4.

### Step C — Step 3 (download_content) for new digests only
- The script should already be incremental (skip digests already in `robots_content.json`). Run it; expect ~6,400 new HTTP fetches.
- This step has rate-limited Wayback fetches; run politely (1 req/sec). If many transient 5xx errors, retry after a few hours.

### Step D — Step 4 (parse_panel) and Step 5 (merge_firm_panel)
- Pure CPU; fast.
- Re-run completely; outputs are `robots_panel_monthly.csv` and `firm_panel_monthly.csv`.

### Step E — Step 6 (sitemap declarations) for new digests only
- Re-run; the script should be incremental on `(host, ym, digest)`. Expect ~9,600 new sitemap_url rows.
- Some of these will fail (sitemap not present, fetch error) — that's fine, downstream handles NA.

### Step F — Restart Step 6.5 (URL paths)
- Now you have a new v1 input from Step E.
- Restart 6.5 either locally (slow but free) or back on cloud machines (fast but $).
- Dedup in `load_v2_status()` will skip the ~3,665 rows that survived from the day-15 era.
- Expect ~117,000 jobs to fetch (the ~32,557 newly-introduced EOM digests + ~84,400 unchanged-digest cells that were never started).

  *Note: this number assumes none of the 5,007 currently-clean rows came from now-stranded day-15-only digests. In reality some will be stranded; merge step in §A doesn't filter them out, but they will simply not match any `(nk, ym, dg)` in the new input and will sit harmlessly in the file.*

### Step G — Step 7 (build search_sitemap_block panel)
- Run `07_build_search_sitemap_block_panel_v3.py` once 6.5 is reasonably complete.
- Output is the panel version of `search_sitemap_block`, which feeds the staggered-DiD analysis.

---

## 7. Key invariants to NOT break

1. **Step 7 join key is `(norm_key, year_month, digest)`.** Do not introduce timestamp into any downstream join. If you add new variables, key them on this tuple. Reference: `07_build_search_sitemap_block_panel_v3.py:250`.
2. **`norm_key` ≠ `host`.** `norm_key` is a normalized key over `(host, sitemap_url)` because one host can declare multiple sitemaps in its `robots.txt`. Don't conflate them.
3. **Digest semantics.** A digest in this pipeline is the Wayback `digest` field for the `robots.txt` snapshot at the chosen timestamp. Same digest = same `robots.txt` content. Different digest = different content (even if visually similar).
4. **`error == "" AND no partial flags`** is the definition of terminal-clean for 6.5. Anything else is retryable. Don't change this without updating `load_v2_status()`.
5. **`statuscode == "200"` filter** in audit / Step 2 is required — Wayback also archives 4xx/5xx responses, which are not parseable robots.txt content.
6. **The 5,007 already-clean rows are valuable**, even if some end up "stranded" (their `(nk, ym, dg)` not in the new EOM input). Don't delete them. Disk is cheap; compute is not.

---

## 8. The treatment variable hierarchy (for context if asked)

| Tier | Variable | Definition | Approx prevalence |
|---|---|---|---|
| 0 | `any_block` | Any of 62 AI bots restricted | 47% (non-fund obs.) |
| 1 | `search_block` | Any of 3 consumer AI search bots restricted | 45% |
| **2** | **`search_sitemap_block`** | Any sitemap URL blocked by any of 3 search bots | **5%** (137 / 2,796 evaluable non-fund firms) |
| 3 | `search_block_specific` | A search bot is blocked via explicit `User-agent` rule (not wildcard) | 1% |

Tier 2 is the main treatment. Cross-section is computed in `bot_paper/build_analysis_sample.py`. Panel is being built (this rebuild) by Steps 1–7.

Three core search bots (the `core: True` ones in `BOT_TAXONOMY` in `04_parse_panel.py`):
- OAI-SearchBot (OpenAI / ChatGPT Search)
- PerplexityBot (Perplexity AI)
- Claude-SearchBot (Anthropic / Claude)

Reasoning for these three only: they are the only AI search crawlers with double-digit webmaster awareness AND a public consumer product whose citation behavior was tested experimentally in the pilot.

---

## 9. The 26.8% number — where it came from (so you don't have to re-derive)

```
v1 jobs total          = 121,366
EOM differs at job lvl = 32,557  (~26.8%)
EOM differs at cell lvl = 6,644  (~11.66%)

Cell-vs-job ratio: 121,366 / 56,961 = 2.13 (avg sitemaps per host-month robots.txt)
The EOM-affected cells happen to be more sitemap-rich than average,
which is why job-fraction (26.8%) > cell-fraction (11.7%) × 2.13 / 2.13.
```

The audit script is at `c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/audit_eom_gap.py` if you want to reproduce or sanity-check after running new Step 2.

---

## 10. What I want from you (the new AI)

Concrete, in order:

1. **Read this whole document.** Confirm understanding of the rebuild rationale before touching any code.
2. **Help me write the corrected `02_select_monthly.py`** under EOM-latest-timestamp logic per §4. Use `max(snapshots, key=timestamp)` not `closest_to_day`.
3. **Write a small merge script** for the 5 cloud shards per §5. Output to `sitemap_url_paths_v2_merged.jsonl`, then I'll rename.
4. **Run the rebuild pipeline (B → C → D → E → F → G)** sequentially with status checks between each step. Don't skip ahead.
5. **Don't introduce timestamp into any join key downstream.** Step 7 v3 is correct as-is.
6. **Push back if any of my assumptions in this doc are wrong.** Especially anything about the existing scripts' behavior — you should verify against the actual code, not just my description here.

---

## 11. Quick reference: file paths

| What | Where |
|---|---|
| Pipeline scripts | `C:/Users/xl/OneDrive - Universitat Ramón Llull/bot_paper/panel_construction/` |
| Data dir | `D:/anoconda/bot_paper_data/` |
| Cloud shards (just downloaded) | `D:/anoconda/bot_paper_data/shards_backup/sitemap_url_paths_v2_m{0..4}.jsonl` |
| Audit script (for ref) | `c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/audit_eom_gap.py` |
| This handoff doc | `c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/HANDOFF_REBUILD_PANEL_EOM.md` |
| Project-level briefing | `c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/PROJECT_BRIEFING_FOR_NEW_AI.md` |
| Cross-section research design | `C:/Users/xl/OneDrive - Universitat Ramón Llull/bot_paper/CLAUDE.md` |
| Stata coding standards | `C:/Users/xl/OneDrive - Universitat Ramón Llull/bot_paper/stata/CLAUDE.md` |

---

## 12. Things outside this rebuild's scope (don't touch)

- Cross-section `analysis_sample.csv` and Stata Table 1 — these are fine
- The 1-page ETH summer school Research Statement — already submitted/draft
- The `recover_delisted_urls/` recovery work — done, 142 URLs recovered, integrated
- Pilot test data (ChatGPT/Claude/Perplexity API audit) — done
- The 4 supervisor feedback points #2 (lagged controls) and #4 (CRSP filters) — these are regression-stage decisions, applied later in Stata. Out of scope for this rebuild.
