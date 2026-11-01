# Monitor brief: Step 6 sharded cloud run (5 machines, EOM rebuild 2026-04-29)

This file is a self-contained briefing for a fresh Claude Code session whose
job is **only to monitor** 5 cloud machines running `Step 6` of the EOM
panel rebuild. The user opens that session, drops this file in context,
and types "**开始**" — that is the trigger to start monitoring.

> **NOTE on filename**: this file is for **Step 6** (`06_collect_sitemaps_for_panel.py`),
> which fetches sitemap-XML observations. Don't confuse with the older
> `MONITOR_06_5_SHARDS.md` in the same folder — that one was authored for
> the future **Step 6.5** monitor session and is not yet relevant.

---

## 0. What "开始" means

When the user says **开始**:

1. Read the **Machine roster** (§3) below.
2. SSH each machine and dump the last 25 lines of `~/step6_run.log` so the user sees the full state of all 5 once.
3. Compute and report a one-line per-machine summary (clean / det_partial / transient counts and rough ETA).
4. Set up a **persistent Monitor** that polls every 5 minutes:
   - For each of the 5 machines, ssh and read the latest `[N/M] clean=X, det_partial=Y, transient=Z` line + last error if any.
   - Emit one consolidated 5-line report per cycle.
5. Send a push notification when any of these happen:
   - A machine's `transient` jumps by ≥1000 in a single 5-min cycle (Wayback throttle storm).
   - A machine's process exits (PID gone) before queue is empty.
   - A machine's clean count is flat AND transient is rising for ≥2 cycles (stuck on rate-limit).
   - SSH to a machine fails 2 cycles in a row.
   - Any machine reaches `Phase 1 Fetch Summary` (it's done).

Do NOT spawn the run itself — the user starts each machine separately. This
session **only watches**.

---

## 1. Project context (1 paragraph)

Corporate finance × AI search visibility paper. Tier 2 panel construction
under EOM (latest-interpretable robots state per host-month). Steps 2-5
finished locally on 2026-04-29. Step 6 fetches sitemap XML from Wayback
across 394k 4-tuple observations (norm_key, year_month, digest, timestamp);
that's the long step. Local run hit Wayback per-IP throttle storms, so it
was sharded into 5 cloud machines via md5 hash of the 4-tuple. Each machine
processes ~80k obs of its shard. After all 5 finish, user runs
`merge_step6_shards.py` locally to produce
`sitemap_panel_results_eom_primary.jsonl` for Step 6.5 / Step 7.

---

## 2. Pipeline file register

| stage | local file | status |
|---|---|---|
| Step 2 v3 | `D:/anoconda/bot_paper_data/monthly_snapshots[_new].csv` | done |
| Step 3 | `D:/anoconda/bot_paper_data/robots_content[_new].json` | done (36,777 cached digests, 6 exhausted) |
| Step 4 | `D:/anoconda/bot_paper_data/robots_panel_monthly[_new].csv` | done |
| Step 5 | `bot_paper/panel_construction/firm_panel_monthly.csv` | done (932k firm-months, 6,066 search switchers) |
| **Step 6 (this)** | `D:/anoconda/bot_paper_data/sitemap_panel_results_eom_primary_m{0..4}.jsonl` | **5 cloud machines, in progress** |
| Step 6 merge | `git/practice/merge_step6_shards.py` | run after all 5 finish |
| Step 6.5 | `D:/anoconda/bot_paper_data/sitemap_url_paths_v2_eom_primary.jsonl` | not started |
| Step 7 | `bot_paper/panel_construction/search_sitemap_block_panel.csv` | not started |

---

## 3. Machine roster

The user fills this in as they bring each machine up. Each row needs:
- shard id (0–4)
- provider + region
- public DNS or IP
- absolute path to .pem (or GCP key file) on the user's local Windows
- ssh user (`ec2-user` for AWS Amazon Linux 2023; `<your-user>` for GCP)

**Roster (filled 2026-04-29):**

| shard | provider | region | DNS / IP | ssh user | key path (Windows) |
|---|---|---|---|---|---|
| 0 | AWS | us-east-1 (Virginia) | `ec2-98-86-229-102.compute-1.amazonaws.com` | `ec2-user` | `C:\Users\xl\OneDrive - Universitat Ramón Llull\keys\virginia.pem` |
| 1 | AWS | eu-north-1 (Stockholm) | `ec2-13-50-107-35.eu-north-1.compute.amazonaws.com` | `ec2-user` | `C:\Users\xl\Downloads\fidio.pem` |
| 2 | GCP | us-west1-b | `34.168.16.24` | `fidio728` | `C:\Users\xl\Downloads\gce_key` |
| 3 | GCP | us-south1-c | `34.174.42.206` | `fidio728` | `C:\Users\xl\Downloads\gce_key` |
| 4 | GCP | us-central1-f | `136.114.165.116` | `fidio728` | `C:\Users\xl\Downloads\gce_key` |

All 5 machines passed connectivity test on 2026-04-29.

---

## 4. SSH command template per machine

For shard `S` with `KEY=<key path>`, `DNS=<dns>`, `USER=<ssh user>`:

```bash
ssh -i "$KEY" ${USER}@${DNS} "tail -25 ~/step6_run.log"
```

Process alive check:

```bash
ssh -i "$KEY" ${USER}@${DNS} "pgrep -f 06_collect_sitemaps && echo ALIVE || echo DEAD"
```

End-of-run summary (after the run is done):

```bash
ssh -i "$KEY" ${USER}@${DNS} "grep -A 10 'Phase 1 Fetch Summary' ~/step6_run.log || echo not done yet"
```

---

## 5. Per-cycle report format

Every 5 minutes, for each machine, emit one line. Example shape:

```
HH:MM | m0 (Virginia)     [12345/82451] clean=4823 det=2104 transient=5418  Δ60s clean=+45 transient=+12  PID alive
HH:MM | m1 (Stockholm)    [9876/80123]  clean=3210 det=1876 transient=4790  Δ60s clean=+22 transient=+0   PID alive
HH:MM | m2 (Frankfurt)    ...
HH:MM | m3 (Tokyo)        ...
HH:MM | m4 (us-central1)  ...
```

The Δ values are computed against the previous cycle's snapshot. First
cycle prints `Δ=N/A`. If SSH fails: `m1 (Stockholm) | SSH FAILED (n consecutive)`.

---

## 6. Health thresholds (when to escalate)

Per machine, per 5-min cycle:

| signal | severity | action |
|---|---|---|
| clean Δ > 0 AND transient Δ < 100 | healthy | no escalation |
| clean Δ > 0 AND transient Δ between 100-1000 | mild throttle | note in cycle line, don't escalate |
| clean Δ == 0 AND transient Δ ≥ 500 | hard throttle pulse | flag in cycle line; if 2+ consecutive cycles → push notification |
| clean Δ == 0 AND transient Δ ≥ 5000 in one cycle | throttle storm | **immediate push notification** + recommend killing that machine to preserve retry budget |
| PID DEAD with `Phase 1 Fetch Summary` line | finished cleanly | push notification: "shard N done, X clean / Y det / Z transient" |
| PID DEAD without summary | crashed | **immediate push notification** with last 50 lines of log |
| SSH timeout 2 cycles | machine unreachable | **push notification** + recommend AWS/GCP console check |

---

## 7. When all 5 shards are done

After each machine's log shows `=== Phase 1 Fetch Summary ===`:

1. Tell the user "All 5 shards done."
2. Provide the scp commands to download each shard's output back to local. The user runs them from their Windows Git Bash:

```bash
# Per-shard download (user fills in $KEY $DNS $USER for each)
KEY=<path>
DNS=<dns>
USER=<user>
S=<shard id>
scp -i "$KEY" "${USER}@${DNS}:~/bot_paper_data/sitemap_panel_results_eom_primary_m${S}.jsonl" \
    "/d/anoconda/bot_paper_data/"
```

3. After all 5 are downloaded, instruct the user to run `merge_step6_shards.py` locally:

```bash
cd "/c/Users/xl/OneDrive - Universitat Ramón Llull/git/practice"
python merge_step6_shards.py --num-shards 5
```

4. Verify the unified `D:/anoconda/bot_paper_data/sitemap_panel_results_eom_primary.jsonl` exists with sane row count (should be ~330k+ obs after dedup), and tell the user they can move on to Step 6.5 in the main Claude Code session.

---

## 8. Things this session must NOT do

- Do not start, restart, or kill any cloud Python process unless the user explicitly says so. The user starts/stops each machine themselves.
- Do not modify any code in `bot_paper/panel_construction/` or `git/practice/`.
- Do not run Step 6.5 / Step 7. That happens in the main session.
- Do not retry SSH more than 1 extra time per cycle. If 2 cycles fail, escalate.

---

## 9. Output cadence

- 5-min Monitor cycle is the default. If user asks for tighter (e.g. "每 2 分钟"), use that instead.
- Single push notification per distinct event — don't spam.
- After each cycle, do not narrate "still running" — only emit the 5-line cycle report and any flagged signals.

---

## 10. Stopping the monitor

If user says **停止监控** or similar, TaskStop the Monitor task and confirm.
The cloud machines themselves keep running (nohup) — stopping monitor only
stops *this session's* observation.

---

## 11. Key facts the new session needs

- The 5 shard files were partitioned by `md5(f"{nk}|{ym}|{dg}|{ts}")[:8] % 5` — same hash function in `06_collect_sitemaps_for_panel.py:shard_for()` and in `split_step6_partial_by_shard.py`.
- Each shard had ~13,300 rows of pre-existing checkpoint (loaded from the local partial that was split by shard before deployment). New work per shard is ~70k obs.
- `--workers 6 --rate-limit 1.0` is the negotiated config. Each machine sends at most 1 req/sec sustained (global throttle); ~50-100 obs/min throughput after recursive sitemapindex expansion.
- ETA per shard at ~3 obs/sec average = ~6 hours; aggregate wall time across 5 machines is similar (they run in parallel).
- Wayback's per-IP throttle was the reason for sharding. If any machine's transient column explodes while others stay healthy, it's that machine's IP getting blocked, not a global Wayback outage.
- Code knowns:
  - `http_404`, `http_410`, `archived_403/404/410/5xx` are **deterministic** — never retried.
  - `http_5xx`, timeouts, connection resets, named exceptions like `ConnectionError` are **transient** — retried up to 10× across runs.
  - Child sitemap deterministic errors are caught and treated as `child_truncated` — they do NOT flip parent to transient retry. Only transient child errors flip `had_child_errors`.

---

## 12. Quick sanity check before saying 开始

Before responding to "开始", verify the machine roster (§3) has no TBD rows. If TBD remains, ask the user:

> 请补全 §3 第 N 行的 DNS / user / key path 再开始监控。

Once roster is complete, proceed with steps 1-5 of §0.

---

## 13. After Step 6 is fully merged — handoff back to main session

Tell the user (in plain Chinese):

> Step 6 已完成并合并。`sitemap_panel_results_eom_primary.jsonl` 已就绪。回到主 Claude Code 会话继续 Step 6.5 / Step 7 / 最终回归。本监控会话可以关闭。

---

## 14. Deployment record — 2026-04-29 14:40 (this run)

This run was performed inside the **main** Claude Code session (not a separate
monitor session — user explicitly chose "就在这"). Captured here so a future
session can reconstruct what was actually shipped.

### 14.1 Files deployed per machine (~484 MB total)

Uploaded under `~/bot_paper_data/` (script under `~/`):

| file | size | source |
|---|---|---|
| `robots_content.json` | 213 MB | `D:/anoconda/bot_paper_data/` |
| `robots_content_new.json` | 182 MB | `D:/anoconda/bot_paper_data/` |
| `monthly_snapshots.csv` | 41 MB | `D:/anoconda/bot_paper_data/` |
| `monthly_snapshots_new.csv` | 43 MB | `D:/anoconda/bot_paper_data/` |
| `sitemap_panel_results_eom_primary_m{N}.jsonl` | ~5 MB | per-shard partial checkpoint |
| `06_collect_sitemaps_for_panel.py` | 32 KB | `bot_paper/panel_construction/` |

scp parallel upload: 30-59s per machine, all 5 finished within 60s.
Python 3.9 (AWS) / 3.11 (GCP) pre-installed; `requests` already present.

### 14.2 Launch command per shard N (shard ∈ {0..4})

```bash
cd ~ && nohup python3 ~/06_collect_sitemaps_for_panel.py \
  --shard N --num-shards 5 \
  --data-dir ~/bot_paper_data \
  --workers 6 --rate-limit 1.0 \
  > ~/step6_run.log 2>&1 &
```

### 14.3 Initial state per shard (from first log emission)

| shard | total_obs | already_done (clean+det+exhausted) | retryable_transient | remaining_to_fetch |
|---|---|---|---|---|
| m0 (Virginia) | 79,217 | 3,517 | 9,737 | 75,700 |
| m1 (Stockholm) | 78,904 | 3,496 | 9,657 | 75,408 |
| m2 (us-west1) | 78,971 | 3,509 | 9,820 | 75,462 |
| m3 (us-south1) | 78,354 | 3,497 | 9,798 | 74,857 |
| m4 (us-central1) | 79,065 | 3,515 | 9,852 | 75,550 |
| **total** | **394,511** | **17,534** | **48,864** | **376,977** |

### 14.4 Monitor wiring (in this main session)

- **Cycle loop** (writes 5-min progress + ALERT lines to its output file):
  - PowerShell script: `C:\Users\xl\AppData\Local\Temp\step6_monitor.ps1`
  - Bash background task ID (current): `btlguqimm` (changes if relaunched)
  - Output file: `C:\Users\xl\AppData\Local\Temp\claude\<project-hash>\<session-id>\tasks\<task-id>.output`
- **Alert tail** (Monitor tool, persistent=true, regex `\[ALERT\]`):
  - Task ID (current): `be6yutffq`
  - Surfaces: STORM (transient Δ ≥ 1000), STUCK (clean Δ=0 ∧ tr Δ ≥ 500 for ≥2 cycles), DEAD (PID gone, not done), DONE (Phase 1 Fetch Summary), SSH_FAIL (≥2 consecutive)

**⚠️ Encoding gotcha (debugged 2026-04-29 14:55).** The shard 0 key path
contains `ó` (`Ramón Llull`). When the .ps1 file is written WITHOUT a UTF-8
BOM, PowerShell 5.1 invoked via `-File` falls back to system ANSI codepage
and decodes the UTF-8 byte sequence `C3 B3` as `Ã³`, producing a wrong path
and `Permission denied (publickey)` failures only on shard 0 (m1-m4 have no
special chars). **Fix:** ensure the script is saved with UTF-8 BOM:

```powershell
$content = [IO.File]::ReadAllText($path, [Text.Encoding]::UTF8)
[IO.File]::WriteAllText($path, $content, (New-Object Text.UTF8Encoding($true)))
```

If the session restarts, both task IDs change. Re-launch by:
1. Running `step6_monitor.ps1` (verify it starts with bytes `EF BB BF`) via
   `powershell.exe -NoProfile -ExecutionPolicy Bypass -File <path>` in
   background.
2. Re-arming the Monitor tool tailing the new task's `.output` file with
   regex `\[ALERT\]`.

### 14.5 Post-run download recipe (when all 5 PHASE_1_DONE arrive)

For each shard N, from local Windows PowerShell:

```powershell
$key = "<key path>"; $userhost = "<user>@<dns>"
scp -i $key "${userhost}:bot_paper_data/sitemap_panel_results_eom_primary_m${N}.jsonl" `
  "D:\anoconda\bot_paper_data\"
```

Then run `merge_step6_shards.py --num-shards 5` locally.
