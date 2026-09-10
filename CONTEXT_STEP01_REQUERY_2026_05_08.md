# Context for new window: Step 01 CDX re-query for Tier 1 late-2025 backfill

> **For the new agent reading this:** Everything you need is in this file.
> No need to ask the user for project background. Files referenced below
> already exist on disk; the user has 10 cloud VMs ready to deploy.

---

## 1. What this is about

The user is finalizing a corporate-finance paper on AI-search bot
blocking. The empirical pipeline builds firm-month panels from Wayback
Machine snapshots of robots.txt. The advisor's "first analysis" is a
determinants-of-blocking regression in
`bot_paper/stata/stata-mcp-folder/stata-mcp-dofile/13_determinants_of_blocking.do`,
which already ran on 6 LHS variants. Two `tier1_*` LHS variants were
deferred because the Tier 1 panel
(`bot_paper/panel_construction/gvkey_month_tier1_reg.dta`) has near-zero
`search_block` coverage in 2025-09 / 10 / 11 / 12.

The user diagnosed the root cause on 2026-05-08: the gap is **upstream
of Step 03 (download)**, which has an empty queue. Step 02
(`monthly_snapshots.csv`) itself has only 33 / 203 / 27 / 756
`has_snapshot` rows for 2025-09 / 10 / 11 / 12 in the primary cohort
(of ~4,350 hosts), and 15 / 68 / 18 in the new cohort (of ~5,058 hosts).
That means **Wayback CDX did not return many snapshots for those months
the last time Step 01 was queried** (April 2026). Internet Archive
indexing lags real-time crawling; a fresh re-query of CDX may surface
new late-2025 snapshots that have since been indexed.

Goal of this work item: re-query Wayback CDX across both cohorts,
re-run Step 02, and **decide based on whether `has_snapshot` for
2025-09/10/11/12 materially improved** whether to continue Step 03 →
Step 04 → Step 05 → `11a_merge_tier1_panel.do`.

**Do NOT run Step 03 onward until Step 02 coverage check passes.** If
`has_snapshot` for those months has not grown after the re-query,
stop and report — that means Wayback simply has not indexed those
months yet, and continuing the pipeline is wasted effort.

---

## 2. What is already done (no work needed here)

These are committed and ready:

- **`bot_paper/panel_construction/01_query_cdx.py`** — modified to
  support sharding via `--shard / --num-shards / --data-dir /
  --rate-limit / --new-hosts / --force-requery`. Shard hash matches
  Step 6: `int(md5(host)[:8], 16) % num_shards == shard`. Backwards
  compatible: running with no flags still does single-machine mode.
  Sharded output filenames embed `num_shards` (e.g.
  `cdx_snapshots_s10_m3.jsonl`) so a 5-shard run's outputs cannot
  silently collide with a 10-shard run's outputs.

- **`bot_paper/panel_construction/merge_cdx_shards.py`** — new file.
  Reads `cdx_snapshots_s{N}_m{0..N-1}.jsonl` (or the `_new_` variant
  with `--new-hosts`) and writes a merged jsonl. **Strict by default:**
  aborts on any cross-shard duplicate host. Default output goes to
  `cdx_snapshots_merged.jsonl` so the existing master is not
  overwritten without an explicit `Move-Item`.

- Both files use UTF-8 without BOM.

---

## 3. What you (new agent) need to do

### 3.1 Deploy to 10 VMs

Each VM needs three files:

| Source on Windows host | Upload to VM |
|---|---|
| `bot_paper/panel_construction/01_query_cdx.py` | `~/01_query_cdx.py` |
| `bot_paper/compustat_2025_clean.csv` | `~/compustat_2025_clean.csv` |
| `bot_paper/panel_construction/new_hosts_to_scan.csv` | `~/new_hosts_to_scan.csv` |

The script accepts `--input-csv` and `--new-hosts-csv` so flat layout
is fine — pass paths explicitly (see §3.2).

### 3.2 Per-VM command

On each VM (`<N>` = 0..9), run **two passes sequentially** under
`nohup` so an SSH disconnect does not kill the run:

```bash
mkdir -p ~/bot_paper_data
cd ~

# Primary cohort (4,350 hosts, ~435 per shard)
nohup python3 -u 01_query_cdx.py \
    --shard <N> --num-shards 10 \
    --data-dir ~/bot_paper_data \
    --rate-limit 2.0 \
    --input-csv ~/compustat_2025_clean.csv \
    > ~/step1_primary_m<N>.log 2>&1

# New cohort (5,058 hosts, ~506 per shard)
nohup python3 -u 01_query_cdx.py \
    --new-hosts \
    --shard <N> --num-shards 10 \
    --data-dir ~/bot_paper_data \
    --rate-limit 2.0 \
    --new-hosts-csv ~/new_hosts_to_scan.csv \
    > ~/step1_new_m<N>.log 2>&1
```

Replace `<N>` with the VM index (0..9). The two passes run sequentially
on each VM (primary then new). Total expected per VM: ~941 hosts ×
2.0 sec ≈ 31 min.

**Rate limit guidance:** 2.0 sec/req × 10 VMs ≈ 5 rps total to Wayback,
which is conservative. Do NOT lower below 1.5 sec/req without checking
with the user — Wayback is sensitive about CDX traffic.

Per-VM output (under `~/bot_paper_data/`):
- `cdx_snapshots_s10_m<N>.jsonl`     primary cohort, this shard
- `cdx_snapshots_new_s10_m<N>.jsonl` new cohort, this shard

The `s10` tag in the filename pins the partitioning (10 shards). If
you ever rerun with a different `--num-shards`, the new files will
have a different tag and cannot collide with these.

### 3.3 Monitor and collect

Watch each VM's `step1_primary_m<N>.log` and `step1_new_m<N>.log` for:
- "Done. Total hosts saved: XXX" — completion line
- "Failed hosts (will retry next run): N" — should be small (<5%)

Once all 10 VMs finish both passes, pull the 20 shard files (10
primary + 10 new) back to the Windows host, into:

```
D:/anoconda/bot_paper_data/cdx_snapshots_s10_m{0..9}.jsonl
D:/anoconda/bot_paper_data/cdx_snapshots_new_s10_m{0..9}.jsonl
```

### 3.4 Merge

```powershell
cd "C:/Users/xl/OneDrive - Universitat Ramón Llull/bot_paper/panel_construction"
python merge_cdx_shards.py --num-shards 10 --data-dir D:/anoconda/bot_paper_data
python merge_cdx_shards.py --num-shards 10 --data-dir D:/anoconda/bot_paper_data --new-hosts
```

Merge is **strict by default** — it will abort if any host appears in
two shards (would indicate hash drift or stale shard files). On a
clean run there should be zero cross-shard duplicates.

Outputs:
- `D:/anoconda/bot_paper_data/cdx_snapshots_merged.jsonl`
- `D:/anoconda/bot_paper_data/cdx_snapshots_new_merged.jsonl`

The existing master files
(`cdx_snapshots.jsonl` from 2026-04-13, `cdx_snapshots_new.jsonl` from
2026-04-25) are NOT overwritten. The merge script prints exact
PowerShell `Move-Item` commands at the end for the swap; review host
counts first, then run those commands.

### 3.5 Run Step 02 only (do NOT run 03 yet)

```powershell
cd "C:/Users/xl/OneDrive - Universitat Ramón Llull/bot_paper/panel_construction"
python 02_select_monthly.py
```

Only Step 02 — do NOT chain into Step 03 download. Coverage check is
the gate.

### 3.6 Coverage check (the critical decision point)

```python
import pandas as pd
fp = r"C:/Users/xl/OneDrive - Universitat Ramón Llull/bot_paper/panel_construction/monthly_snapshots.csv"
df = pd.read_csv(fp, dtype={"year_month": str})
months = ["2025-09", "2025-10", "2025-11", "2025-12"]
print("primary has_snapshot by month:")
print(df[df.year_month.isin(months)].groupby("year_month")["has_snapshot"].sum().to_string())
```

Repeat for the new-cohort `monthly_snapshots` if it has its own file.

**Baseline (before re-query, 2026-05-08):**
- primary: 2025-09=33, 2025-10=203, 2025-11=27, 2025-12=756
- new:     2025-09=15, 2025-10=68,  2025-11=18

**Decision:**

- If `has_snapshot` for 2025-09 / 2025-10 / 2025-11 grew by **≥ 5×**
  in EITHER cohort → continue Step 03 → 04 → 05 → 11a.
- If growth is small (e.g., <2×) → STOP. Report to user.
  Update memory note
  `~/.claude/projects/.../memory/project_tier1_late_2025_coverage_gap_2026_05_08.md`
  saying Wayback CDX still has not indexed these months as of <date>.

### 3.7 If Step 02 looks good, continue (single-machine, sequential)

```powershell
cd "C:/Users/xl/OneDrive - Universitat Ramón Llull/bot_paper/panel_construction"
python 03_download_content.py
python 04_parse_panel.py
python 05_merge_firm_panel.py
```

Check each script's CLI for `--new-hosts` flags before running. Then
in Stata MCP, re-run `11a_merge_tier1_panel.do` to rebuild
`gvkey_month_tier1_reg.dta`.

After `11a` completes, edit
`bot_paper/stata/stata-mcp-folder/stata-mcp-dofile/13_determinants_of_blocking.do`
to add `tier1_any_block` and `tier1_search_block` back into the
`lhs_list` (and re-add the merge of `gvkey_month_tier1_reg.dta` that
was removed on 2026-05-08), then re-run that do-file.

---

## 4. Things to be careful of

1. **Rate limit.** Do NOT lower below 1.5 sec/req per VM without
   user permission. 10 VMs at 2.0 sec/req = 5 rps total to Wayback.

2. **Stata MCP encoding.** Stata do-files run via Stata MCP must NOT
   have UTF-8 BOM (Stata interprets it as a command and fails r(199)).
   The existing `13_determinants_of_blocking.do` has no BOM.

3. **PowerShell + non-ASCII paths.** Project root has `Ramón`. Use
   double quotes, not single, for paths containing this character.
   PS1 scripts that read this path need a UTF-8 BOM (opposite of
   do-files; this is well-attested in project memory).

4. **Don't auto-launch downstream Stata after Step 02.** Coverage
   check is the gate. Pause for the user before running Step 03.

5. **VMs are billable.** Stop or release them in the cloud console as
   soon as Step 01 finishes — don't leave them idling.

6. **Merge --allow-dupes is the safety valve, not the default.** If
   merge aborts with cross-shard duplicates, do NOT just slap on
   `--allow-dupes`. Investigate first: most likely cause is stale
   shard files from a previous `--num-shards` run still in the data
   directory. Check the `s{N}` tags in the filenames.

---

## 5. File quick reference

| Path | Purpose |
|---|---|
| `bot_paper/panel_construction/01_query_cdx.py` | Step 01, sharded |
| `bot_paper/panel_construction/merge_cdx_shards.py` | merge 10 shards back, strict by default |
| `bot_paper/panel_construction/02_select_monthly.py` | Step 02, run after merge |
| `bot_paper/panel_construction/monthly_snapshots.csv` | Step 02 output, decision-gate file |
| `bot_paper/compustat_2025_clean.csv` | primary cohort hosts (4,350) |
| `bot_paper/panel_construction/new_hosts_to_scan.csv` | new cohort hosts (5,058) |
| `D:/anoconda/bot_paper_data/cdx_snapshots.jsonl` | existing master, primary, 1.65 GB, 2026-04-13 |
| `D:/anoconda/bot_paper_data/cdx_snapshots_new.jsonl` | existing master, new, 1.1 GB, 2026-04-25 |
| `bot_paper/stata/.../13_determinants_of_blocking.do` | downstream consumer, Tier1 deferred |

---

## 6. End-state success criterion

A run is successful when ONE of these is true:

(a) Step 02 has_snapshot counts for 2025-09/10/11 grew ≥ 5× in at
    least one cohort, AND Step 03 → 04 → 05 → 11a re-ran cleanly,
    AND the rebuilt Tier 1 panel covers 2025-09/10/11 with reasonable
    `search_block` non-missing rates (target: ≥70% of in-sample rows).

(b) Step 02 coverage did NOT grow, the team documented the
    Wayback-CDX-coverage limitation in the memory note, and the user
    decided how to proceed (window Tier 1 to 2021-01..2025-08, or
    wait for Wayback to backfill).

Either is a valid stopping point. Do NOT push past (a) into rebuilding
the full Tier 1 panel without verifying (a)'s coverage condition.
