# Daily Robots-Block Transition Pipeline — Research Documentation

**Author:** Xueyang Liu (URL/ESADE PhD)
**Last updated:** 2026-05-03
**Related main-line panel:** `bot_paper/panel_construction/07_build_search_sitemap_block_panel_v4.py` (monthly EOM, Step 7 v4)

---

## 1. Motivation and research question

### 1.1 Why a day-level pipeline at all

The main empirical analysis in this project is a monthly difference-in-
differences panel keyed on `(gvkey, year_month)`, in which the treatment
indicator `search_sitemap_block` is constructed at end-of-month
granularity from a firm's archived `robots.txt` plus its sitemap-listed
URL set (Step 6.5 → Step 7 main line). That panel is appropriate for
estimating quarterly-to-annual outcomes — quarterly Compustat
fundamentals, IBES forecast revisions, CRSP returns aggregated to
month/quarter — but it is too coarse for two empirically relevant
questions:

1. **Short-horizon attention dynamics around individual AI-bot launches.**
   The AI-search ecosystem expanded discontinuously: OpenAI published
   the `GPTBot` user-agent on 2023-08-07; Google introduced
   `Google-Extended` on 2023-09-28; Apple introduced `Applebot-Extended`
   in mid-2024. Whether firms reacted within days, weeks, or quarters
   is a first-order question for understanding the disclosure-channel
   mechanism, and a monthly panel pools any reaction within a single
   `year_month` cell.

2. **Measurement of treatment-timing precision.** A firm's true policy-
   change date is unobservable — Wayback Machine shows us only that a
   change had taken effect *by some capture timestamp*. The width of
   the bracketing interval (last observed-allowed capture to first
   observed-blocked capture) is the natural measure of how confident
   we can be in a transition date. This information is destroyed by
   coarsening to month-level.

The daily pipeline addresses both. Its primary deliverable is, for
each `(gvkey, AI-bot-group)` pair, the earliest interpretable Wayback
capture at which the firm transitioned from "bot allowed" to "bot
disallowed", together with the calendar gap to the most recent prior
allowed capture (interval-censoring metadata).

### 1.2 Key terminological convention

Throughout this document and the produced artifacts, the variable is
named **`first_observed_blocked_ts`**, *not* "the day the firm decided
to block the bot." The latter is unobservable. Wayback only tells us
that the disallow rule was already in effect by some capture; we
quantify that uncertainty with `interval_days`. Any event-study
inference that requires a precise calendar event date should restrict
to `interval_days <= K` for a stated cutoff `K` and report results at
multiple `K`.

---

## 2. Design constraints

The following constraints were fixed before implementation, on the
basis of a prior design discussion (2026-05-03) and lessons from Step
6.5 main-line construction (the sitemap-fetcher pipeline).

### 2.1 Fetch by unique digest, never by capture or `(host, digest)` pair

The Wayback Machine assigns each unique response body a SHA-1 content
digest. A captured `robots.txt` body served on `firmA.com` at time `t1`
and again on `firmB.com` at time `t2` will share a digest if and only
if the two responses are byte-identical. The CDX raw stream over our
universe contains 25,903,651 captures; these collapse to 425,329
unique digests; of those, only 99,745 had `sample_status == 200` and
therefore needed a body fetch. Fetching at the digest level rather
than the capture level reduces the workload by a factor of ~260 and
eliminates redundant network traffic.

### 2.2 Status mapping must agree with Step 2 main-line

The same status mapping that Step 2 uses to assign tier eligibility in
the monthly panel must apply here:

| HTTP status (replay) | Daily-pipeline interpretation               |
|---------------------|----------------------------------------------|
| `200` (or revisit-resolved 200) | parse body for bot policy           |
| `404` / `410`        | "no robots policy" ⇒ allow-all per RFC 9309  |
| `3xx` / `403` / `5xx` / `429` / `400` / `202` / unresolved | **uninterpretable**            |
| no CDX capture in window | **gap** (skip in transition detection)    |

Letting the daily timeline drift from this mapping would produce
spurious gaps between monthly and daily inference even when both
pipelines see identical underlying captures.

### 2.3 Interval censoring is mandatory in the output

The output schema must carry `last_unblocked_ts`, `first_observed_blocked_ts`,
and `interval_days = (first_blocked − last_unblocked)` measured in
calendar days. Any event-study sub-sample is then a defensible
interval-censored cohort, not a pseudo-precise event series.

### 2.4 Thread-local persistent `requests.Session` per worker

A documented operational lesson from the Step 6.5 sitemap-fetcher
deployment was that per-fetch `requests.Session() + session.close()`
patterns paid the full TCP + TLS handshake cost on every call and
correlated strongly with `ConnectionError` storms (50% network-error
rate in round-1; 1% under thread-local persistent sessions in
round-2). The daily fetcher reuses the persistent-session helper
unchanged.

### 2.5 Process-global throttle, not per-worker

The fetcher's throttle is a process-wide lock, not a per-worker delay.
Therefore the actual ceiling on requests per second per machine is
`1 / RATE_LIMIT`, irrespective of `--workers`. This was confirmed
empirically in Step 6.5 round-2.

---

## 3. Pipeline structure

The pipeline consists of three stages (A, B, C) plus a post-processing
phase that builds the regression-ready panel. Stage names follow the
hand-off briefing `DAILY_ROBOTS_TRANSITION_PROJECT.md`.

### 3.1 Stage A — Distributed digest-level fetch

**Script:** `06_robots_daily_fetch_digests.py`
**Input:** `cdx_digest_to_fetch.jsonl` (99,745 digests with `sample_status == 200`)
**Output:** `digest_robots_cache.jsonl` (one JSON line per digest with
the raw body text, an `error` label if the fetch failed, the actual
HTTP replay status, and a SHA-1 sanity hash of the body)

Five sequential rounds were run with progressively conservative
throttles, each time feeding the still-failed retry queue back into
the next round. Round 1 was distributed across 16 cloud machines (8
GCP + 8 mixed GCP/DigitalOcean); Rounds 2–5 used the seven machines
with the lowest round-1 transient-error rate. After each round the
shard outputs were merged with deterministic best-rank de-duplication
(success > deterministic-failure > transient-failure).

| Round | Machines | Throttle | Queue size | New `ok` | Cumulative `ok` | Coverage |
|------:|---------:|---------:|-----------:|---------:|----------------:|---------:|
| 1     | 16       | 0.7 s    | 99,745     | 96,754   | 96,754          | 97.41%   |
| 2     | 7        | 0.7 s    | 2,958      | +1,302   | 98,056          | 98.31%   |
| 3     | 7        | 1.5 s    | 1,655      | +755     | 98,811          | 99.06%   |
| 4     | 7        | 2.5 s    | 900        | +315     | 99,126          | 99.38%   |
| 5     | 7        | 3.5 s    | 585        | +120     | 99,246          | 99.50%   |

The remaining 0.50% (499 digests, 487 transient + 12 zero-byte) is
treated as `uninterpretable` in Stage C and contributes to interval
censoring rather than to spurious transitions.

### 3.2 Stage B — Per-digest, per-bot policy classification

**Script:** `parse_digest_robots_to_bot_policy.py` (subsequently
patched by `reclassify_html_as_uninterpretable.py` — see §4.1).
**Reused parser:** `bot_paper/robots_parser.py` (RFC 9309 compliant;
the same module the monthly main line uses).
**Reused taxonomy:** `bot_paper/panel_construction/04_parse_panel.py:BOT_TAXONOMY`
(69 bots in 4 groups: 16 `search`, 29 `training`, 17 `assistant`,
7 `control`). The choice to share these two modules verbatim is
deliberate: the daily pipeline must not introduce parsing semantics
that the monthly panel does not also apply, otherwise daily and
monthly results cannot be cross-validated.

**Output:** `digest_bot_policy.csv` — one row per unique digest, 69 ×
4 = 276 per-bot columns plus a `parse_status` field. Per bot the
recorded fields are `<bot>_root_block`, `<bot>_has_nonroot_disallow`,
`<bot>_specific_rule`, `<bot>_effective_source`. The downstream
transition detector uses only `root_block` (i.e., `Disallow: /`
semantics on the root path), consistent with the briefing's RFC
9309-grounded definition of "blocked."

### 3.3 Stage C — Per-host timeline reconstruction

**Script:** `reconstruct_first_block_events.py`
**Inputs:** the corrected `digest_bot_policy.csv` plus the raw CDX
streams `cdx_snapshots.jsonl` and `cdx_snapshots_new.jsonl`
(combined size 2.75 GB, 25,903,651 captures across 10,895 unique host
records).

For each host the algorithm walks captures in chronological order,
maps each capture to one of three states for each of the 69 bots —
`allowed` (status 404/410, or status 200 with `<bot>_root_block == 0`
on a parseable digest); `blocked` (status 200 with `<bot>_root_block ==
1` on a parseable digest); `uninterpretable` (anything else, including
status 200 with a digest that has `parse_status != "ok"`) — and emits,
per `(host, bot)` pair, the first interpretable capture at which the
bot is `blocked` *and* the most recent earlier interpretable capture
had the bot `allowed`. If no such transition is observed (host's first
interpretable capture is already blocked, or host is never blocked),
the `first_observed_blocked_ts` field is left empty while
`ever_blocked` is still recorded.

The implementation precomputes a per-snapshot 69-bit bot-state vector
once per host and then folds that vector into 69 parallel timelines in
a single inner loop, reducing the wall-clock cost from O(captures ×
bots) to roughly O(captures) per host. End-to-end runtime over 10,895
hosts × 25.9M captures: 77.8 seconds.

### 3.4 Post-processing — Firm merge and aggregation

**Scripts:** `dedup_first_block_events.py` (host-level dedup +
firm-host-bot fan-out) and `gvkey_first_block_aggregate.py` (gvkey-
level aggregation + monthly adoption curve).

The CDX raw stream contains 1,665 hosts that appear in more than one
record (because the upstream CDX query loop emitted one record per
firm-host pair, and a host can map to multiple firms via M&A or
holding-company structures). Stage C therefore wrote 736,851 rows
spanning 621,966 unique `(host, bot)` pairs (114,885 perfect
duplicates with byte-identical fields, confirmed by an audit pass with
zero key-conflicts). The dedup-and-merge script produces:

- `daily_first_block_events_hostbot_dedup.csv` — one row per
  `(host, bot)`, used for host-level descriptive statistics.

- `daily_first_block_events_firmhostbot.csv` — left-join with the
  canonical firm-host mapping (`bot_paper/compustat_all_host_mapping.csv`,
  14,808 rows over 9,408 unique hosts; 389 hosts map to multiple
  gvkeys), producing 977,523 rows. This is the firm-level analysis
  unit and the source for all subsequent aggregation.

The gvkey aggregation script collapses these to one row per gvkey
with `first_<group>_block_ts`, `interval_days_<group>`, and a timing-
precision flag for each of the four bot groups, plus an AI aggregate
(`first_any_ai_block_ts = min` over `search ∪ training ∪ assistant`).
A parallel monthly cumulative-adoption curve is also produced.

---

## 4. Diagnostics, corrections, and validity checks

The pipeline went through three substantive corrections during build,
each motivated by an audit finding rather than a code error.

### 4.1 HTML-as-`robots.txt` semantic bug (corrected)

**Symptom:** a post-Stage-B audit revealed that 41,741 of the digests
with `parse_status == "ok"` had `n_groups == 0` (no `User-agent:` line
parsed). Manual inspection of the underlying bodies showed:

| Body type                                              | Count | RFC 9309 valid allow-all? |
|--------------------------------------------------------|------:|--------------------------:|
| HTML challenge / error pages (Cloudflare, "Please wait while your request is being verified", server error pages) | 40,324 | **No** — body is not robots-formatted |
| Sitemap-only files (just `Sitemap:` lines, no group)   | 587   | Yes                        |
| Comments-only or short non-robots text                  | 231   | Yes                        |
| `Allow:` lines without preceding `User-agent:`          | 20    | Yes (rules without group are ignored) |

The standalone parser at `bot_paper/robots_parser.py` is intentionally
permissive — when no `User-agent:` line is present it returns
`groups == []`, which `classify_bot()` interprets as "no rule applies
⇒ root_block = 0 ⇒ bot allowed." For the 40,324 HTML-page bodies this
is incorrect: the server did not actually serve a `robots.txt`; it
served a challenge or error page. Treating those captures as
"explicitly bot-allowed" would inject false unblocked → blocked
transitions in Stage C.

**Correction:** `reclassify_html_as_uninterpretable.py` re-scans the
fetch cache. A digest is reclassified to `parse_status =
"uninterpretable"` if and only if the body contains an HTML marker
(case-insensitive substring search for `<html`, `<!doctype`, `<head>`,
`<body>`, `<title>`, `<meta `) **and** does not contain any
`User-agent:` directive line. The double condition is intentional: an
HTML page that nonetheless contains a parseable `User-agent:` line
(39 cases out of 40,254 HTML-marker hits) is preserved as `ok`, on
the conservative principle that we only override the parser when the
body unambiguously is not robots-formatted.

**Outcome:** 40,215 `ok` rows reclassified to `uninterpretable`,
yielding the final breakdown 59,031 `ok` (59.18%) and 40,714
`uninterpretable` (40.82%). The 838 `n_groups == 0` cases that are
RFC-9309-valid allow-all (Sitemap-only, comments-only, Allow-only-
without-User-agent) remain as `ok`, as they should.

### 4.2 Duplicate `(host, bot)` rows in Stage C output (corrected)

**Symptom:** the raw Stage C output had 736,851 rows but only 621,966
unique `(host, bot)` keys, because the CDX raw stream lists the same
host multiple times (one record per firm-host pair from the upstream
query loop). All 114,885 duplicate rows were byte-identical (audit
pass with zero key-conflicts), so this is a presentation-layer issue,
not a correctness issue. It was nonetheless corrected at the dedup
step to prevent inflated counts in downstream code.

### 4.3 Strict-vs-broad treatment definition (introduced as robustness)

**Issue:** an "AI-search blocker" in the broad definition can be either
(a) a firm that actively chose to opt out of AI search engines while
leaving traditional crawlers unrestricted, or (b) a firm that revised
its `robots.txt` to block all crawlers en masse — a "blanket-block"
edit. Distinguishing the two is essential for any inference about
AI-targeted opt-out behavior.

**Definition.** Following the briefing convention, we define

> A gvkey is a **strict** AI-search adopter (`first_search_block_strict`)
> if `first_search_block_ts` is observed AND
> `first_search_block_ts < first_control_block_ts`
> (or `first_control_block_ts` was never observed).

Operationally, a gvkey is strict if its earliest observed disallow on
any search-group bot strictly precedes its earliest observed disallow
on any control-group bot (`Googlebot`, `bingbot`, `DuckDuckBot`,
`YandexBot`, `Baiduspider`, `Yeti`, `SeznamBot`).

This is reported alongside the broad definition in the regression
panel and has its own monthly cumulative adoption curve.

---

## 5. Results

### 5.1 Coverage and counts

| Layer                                                          | Count       |
|----------------------------------------------------------------|------------:|
| Unique digests fetched (Stage A)                               | 99,745      |
| Digests with parseable body (`parse_status == "ok"`)            | 59,031      |
| Digests classified `uninterpretable` (Stage A failure or HTML body) | 40,714  |
| Hosts in CDX raw stream                                        | 10,895      |
| Total CDX captures                                             | 25,903,651  |
| `(host, bot)` rows after dedup                                 | 621,966     |
| `(gvkey, host, bot)` rows after firm merge                     | 977,523     |
| At-risk gvkeys (universe denominator)                          | 14,156      |
| `(gvkey, bot)` pairs with observable transition                | 30,516      |
| gvkeys with `first_search_block_ts`                            | 1,206       |
| gvkeys with `first_search_block_strict_ts`                     | 669         |

### 5.2 Distribution of `interval_days`

The interval `interval_days = (first_observed_blocked_ts −
last_unblocked_ts)` measures how tightly we can date a transition.
Across the 30,516 firm-bot transitions:

| Statistic | Value |
|-----------|------:|
| Minimum   | 0     |
| p25       | 5     |
| Median    | 8     |
| p75       | 42    |
| p90       | 130   |
| p95       | 220   |
| p99       | 533   |
| Maximum   | 1,124 |
| Mean      | 50.4  |

The right-skew is severe (mean ≫ median) because Wayback capture
density varies dramatically across hosts: high-capture hosts (typically
larger firms with more frequent crawler attention) yield interval =
1–8 days, while sparse-capture hosts can have interval = 100+ days
even when the underlying behavior change is sharp.

Cumulative coverage at common cutoffs:

| Cutoff (days) | Cumulative count | Share of 30,516 |
|--------------:|-----------------:|----------------:|
| ≤ 1           | 4,134            | 13.5%           |
| ≤ 3           | 6,618            | 21.7%           |
| **≤ 7**       | **14,426**       | **47.3%**       |
| **≤ 14**      | **18,141**       | **59.4%**       |
| **≤ 30**      | **21,160**       | **69.3%**       |
| ≤ 60          | 24,168           | 79.2%           |
| ≤ 90          | 25,629           | 84.0%           |
| ≤ 180         | 28,636           | 93.8%           |
| ≤ 365         | 29,563           | 96.9%           |

### 5.3 Group-level cumulative adoption

The figure below shows cumulative `(gvkey, group)` adoption shares,
defined as the running count of unique gvkeys for which
`first_<group>_block_ts <= calendar month`, divided by the at-risk
denominator (14,156 gvkeys present in the firmhostbot panel). Vertical
dashed lines mark four AI-bot launch dates that are plausibly relevant
to the timing of corporate `robots.txt` revisions.

![Cumulative bot-block adoption by group](E:/bot_paper_data/adoption_curves_by_group.png)

End-of-period cumulative shares (March 2026):

| Group / aggregate                                              | gvkeys | Share of 14,156 |
|----------------------------------------------------------------|-------:|---------------:|
| `first_search_block` (broad — main treatment)                  | 1,206  | 8.52%          |
| `first_training_block`                                          | 771    | 5.45%          |
| `first_assistant_block`                                         | 457    | 3.23%          |
| `first_control_block` (placebo / sanity)                        | 724    | 5.11%          |
| `first_any_ai_block` (search ∪ training ∪ assistant; appendix)  | 1,474  | 10.41%         |

Two empirical observations are worth recording explicitly:

1. **`search > training`.** The search-bot group is the most-blocked
   group in the at-risk panel (8.52% versus 5.45% for training),
   despite training bots (`GPTBot`, `ClaudeBot`, `CCBot`,
   `Google-Extended`, `Bytespider`) receiving most of the attention in
   the public discussion of AI opt-out. This is consistent with the
   monthly main-line treatment definition (`search_sitemap_block`)
   capturing the dominant adoption channel.

2. **`control` is not negligible (5.11%).** A non-trivial fraction of
   firms blocks traditional search engines (`Googlebot`, `bingbot`)
   roughly in line with their training-bot blocking. A naive
   "AI-search > control" inference would understate the role of
   blanket-block behavior, motivating the strict-version analysis
   below.

### 5.4 Strict definition: "AI-search blocked before traditional search"

The figure below shows three series: the broad `search` cumulative
share (red solid), the broad `control` cumulative share (grey dashed),
and the **strict** `search_strict` cumulative share (red dash-dotted),
defined as gvkeys whose first search-group transition strictly
precedes their first control-group transition.

![Broad vs strict AI-search adoption](E:/bot_paper_data/adoption_curves_strict.png)

Below, the cumulative percentage-point gap between `search_strict` and
`control` over time. Positive values mean strict AI-search blockers
exceed control blockers; the shaded band fills negative gaps in grey.

![Strict AI-search minus control adoption gap](E:/bot_paper_data/adoption_gap_strict_minus_control.png)

End-of-period values:

| Series                      | gvkeys | Share  |
|----------------------------|-------:|-------:|
| `search` (broad)            | 1,206  | 8.52%  |
| `search_strict`             | 669    | 4.73%  |
| `control`                   | 724    | 5.11%  |
| `search_strict − control`   | −55    | −0.38 pp |

#### 5.4.1 Strict-share as a function of timing precision

| Cutoff on `interval_days_search` | `first_search_block` | `first_search_block_strict` | strict / broad |
|--------------------------------:|--------------------:|---------------------------:|---------------:|
| all (no cutoff)                  | 1,206               | 669                        | **0.5547**     |
| ≤ 30                             | 885                 | 483                        | 0.5458         |
| ≤ 14                             | 528                 | 163                        | 0.3087         |
| ≤ 7                              | 345                 | 140                        | 0.4058         |

#### 5.4.2 Interpretation

Across the broad sample, **55.5%** of search-group adopters acted on
at least one AI-search bot strictly before they touched any traditional
search crawler. This is the headline statistic for the AI-targeted
robustness exercise. Two further observations:

- The strict ratio drops sharply at intermediate timing cutoffs (≤14
  days: 30.9%) and then recovers somewhat at the tightest cutoff (≤7
  days: 40.6%). The mechanical reason is bundled `robots.txt` edits.
  When a firm revises its `robots.txt` once and adds disallow rules
  for several bot families simultaneously, the search and control
  groups receive their first disallow at the *same* Wayback capture
  timestamp, and the strict definition (which requires a strict
  inequality) classifies the firm as non-strict. Since "same capture"
  is much more common at intermediate windows (where a single
  fortnight contains multiple bot-family edits) than at very tight
  windows (where the per-capture co-occurrence of any specific bot is
  smaller), the ratio dips and then partially recovers.

- The strict definition is **asymmetric** with respect to the broad
  control group: a firm that has no `first_search_block_ts` but has
  `first_control_block_ts` (a "control-only" adopter — blocks
  Googlebot but not GPTBot) is not in either the strict or broad
  search counts, but is in the control count. The negative
  `search_strict − control` end-of-period gap (−55 firms) therefore
  does not falsify AI-targeted behavior; it reflects the existence of
  a small population of firms that block traditional crawlers without
  blocking AI search crawlers, which is itself a research-relevant
  fact (those firms are presumably motivated by traditional anti-
  scraping concerns rather than AI-specific opt-out).

The recommended phrasing in a manuscript is:

> AI-search blocking is often part of broader crawler-policy revisions,
> but even under a strict definition that requires search-group
> restrictions to precede the firm's control-group restrictions, 55.5%
> of search adopters remain. The strict ratio declines at intermediate
> timing-precision cutoffs because bundled `robots.txt` edits — single
> revisions that disallow several bot families simultaneously —
> register as non-strict by construction.

### 5.5 Per-bot adoption (descriptive only, appendix-bound)

Among the four headline AI training crawlers, end-of-period gvkey
counts and shares of the at-risk universe are:

| Bot              | gvkeys | Share of 14,156 |
|------------------|-------:|---------------:|
| `GPTBot`         | 628    | 4.44%          |
| `Bytespider`     | 532    | 3.76%          |
| `ClaudeBot`      | 530    | 3.74%          |
| `Google-Extended`| 509    | 3.60%          |

Median `interval_days` and within-7-day shares (descriptive only):

| Bot              | Median interval | ≤7d share | ≤14d share | ≤30d share |
|------------------|---------------:|----------:|-----------:|-----------:|
| `GPTBot`          | 7              | 50.5%     | 59.1%      | 67.4%      |
| `Bytespider`      | 8              | 48.1%     | 58.3%      | 66.2%      |
| `ClaudeBot`       | 8              | 49.1%     | 58.3%      | 66.4%      |
| `Google-Extended` | 11             | 44.8%     | 54.2%      | 63.1%      |

Per-bot adoption-speed comparisons are deferred to the post-baseline
analyses; the daily pipeline's role is to produce the timing variables,
not to estimate the cross-bot heterogeneity that those variables
support.

---

## 6. Final regression-ready panel

**File:** `E:\bot_paper_data\gvkey_month_panel_for_regression.csv`
**Shape:** 891,828 rows = 14,156 at-risk gvkeys × 63 calendar months
(2021-01 to 2026-03 inclusive)
**Schema:**

| Column                              | Type   | Description                                                       |
|-------------------------------------|--------|-------------------------------------------------------------------|
| `gvkey`                             | string | Compustat firm identifier                                         |
| `month_yyyymm`                      | string | calendar month, 6-digit                                           |
| `first_search_block_month`          | string | gvkey-level constant; YYYYMM of broad search adoption (or empty)  |
| `first_search_block_strict_month`   | string | YYYYMM of strict adoption (search precedes control), or empty     |
| `first_control_block_month`         | string | YYYYMM of first control-group adoption, or empty                  |
| `interval_days_search`              | int    | timing-precision of the broad search transition (days)            |
| `timing_precision_search`           | string | one of `le7`, `le14`, `le30`, `wide`, or empty                    |
| `event_time_search`                 | int    | `month_yyyymm − first_search_block_month` in months; `0` at event |
| `event_time_search_strict`          | int    | analogous for strict                                              |
| `event_time_control`                | int    | analogous for control (placebo)                                   |
| `post_search`                       | 0/1    | indicator: `event_time_search >= 0`                               |
| `post_search_strict`                | 0/1    | indicator: `event_time_search_strict >= 0`                        |
| `post_control`                      | 0/1    | indicator: `event_time_control >= 0`                              |

The panel is shaped for direct use with `reghdfe`, `csdid`,
`did_imputation`, `did2s`, or any standard staggered DiD / event-study
implementation. Recommended primary specifications:

1. **Broad treatment.** `Y ~ post_search + γ_gvkey + δ_month`, with
   `post_search_strict` as a robustness arm reported alongside.

2. **Event study.** `Y ~ Σ_τ 1{event_time_search == τ} × γ_gvkey + δ_month`
   with τ binned at edges as is standard.

3. **Timing-precision robustness.** Re-estimate the broad treatment
   restricting to `timing_precision_search ∈ {le7}`, `{le7, le14}`,
   `{le7, le14, le30}`. The stability of the coefficient across these
   subsamples is the primary measurement-error robustness check.

---

## 7. File inventory

### 7.1 Code (in `practice/`)

| File                                         | Purpose                                                      |
|----------------------------------------------|--------------------------------------------------------------|
| `extract_unique_digest_index.py`             | Stage A pre-processor: CDX → unique-digest index             |
| `06_robots_daily_fetch_digests.py`           | Stage A fetcher (cloud-deployable)                            |
| `deploy_06_robots_daily.sh`                  | push/launch/count/tail/pull driver for the 16 cloud machines |
| `merge_digest_robots_cache.py`               | Stage A round merger (best-rank dedup across rounds 1–5)     |
| `parse_digest_robots_to_bot_policy.py`       | Stage B per-digest parser                                    |
| `dedup_digest_bot_policy.py`                 | Stage B deduplication after multiple resumes                  |
| `reparse_too_large_digests.py`               | Stage B full-rigor pass for `> 500 KB` bodies                 |
| `reclassify_html_as_uninterpretable.py`      | Stage B HTML-body validation correction                       |
| `reconstruct_first_block_events.py`          | Stage C timeline reconstruction                              |
| `dedup_first_block_events.py`                | (host, bot) dedup + firmhostbot fan-out                       |
| `gvkey_first_block_aggregate.py`             | gvkey-level aggregation + monthly adoption curve             |
| `daily_adoption_plots_and_strict.py`         | broad-version adoption plots + strict-timing table           |
| `daily_strict_curve_and_regression_panel.py` | strict-version adoption plots + regression panel             |

### 7.2 Data products (in `E:\bot_paper_data\`)

| File                                              | Rows         | Purpose                                                      |
|---------------------------------------------------|-------------:|--------------------------------------------------------------|
| `cdx_unique_digest_index.jsonl`                   | 425,329      | Unique-digest index from CDX raw                              |
| `cdx_digest_to_fetch.jsonl`                       | 99,745       | Subset with `sample_status == 200` — Stage A fetch queue      |
| `digest_robots_cache.jsonl`                       | 99,745       | Stage A unified cache (rounds 1–5, dedup-merged)              |
| `digest_bot_policy.csv`                           | 99,745       | Stage B output, post HTML-reclassify                          |
| `digest_bot_policy_summary.txt`                   | —            | Class breakdown                                              |
| `daily_first_block_events.csv`                    | 736,851      | Stage C raw output                                           |
| `daily_first_block_events_hostbot_dedup.csv`      | 621,966      | Stage C dedup, host-level                                    |
| `daily_first_block_events_firmhostbot.csv`        | 977,523      | Stage C dedup + gvkey merge                                  |
| `gvkey_first_block_dates.csv`                     | 14,156       | One row per gvkey: per-group first-block timing               |
| `gvkey_adoption_curve_monthly.csv`                | 63           | Broad cumulative adoption curve                              |
| `gvkey_adoption_curve_monthly_strict.csv`         | 63           | Strict cumulative adoption curve                             |
| `gvkey_month_panel_for_regression.csv`            | 891,828      | **Final regression panel** (gvkey × month)                   |
| `adoption_strict_timing_table.csv`                | 4            | Strict ratio by timing precision                             |
| `adoption_curves_by_group.png`                    | —            | Figure: 5 cumulative curves with launch-date annotations      |
| `adoption_gap_search_minus_control.png`           | —            | Figure: broad gap                                            |
| `adoption_curves_strict.png`                      | —            | Figure: broad vs strict vs control                            |
| `adoption_gap_strict_minus_control.png`           | —            | Figure: strict gap                                           |

---

## 8. Limitations

1. **Wayback capture density heterogeneity.** Sparse-capture hosts
   (typically smaller firms or hosts at the long tail of crawler
   attention) have right-skewed `interval_days`. Restricting to a
   tight precision cutoff (≤ 7 days) preserves 47% of transitions but
   does so non-randomly with respect to firm size, web-architecture
   complexity, and geographic location of the firm's web presence. The
   recommended mitigation is to report results at multiple cutoffs
   (`all`, `≤ 30`, `≤ 14`, `≤ 7`) rather than to pre-select a single
   cutoff.

2. **Step A residual `transient` digests (0.47%).** 465 digests
   remained un-fetchable after five retry rounds at progressively
   conservative throttles. The dominant error label is
   `ConnectionError` (90%), which after five geographically-dispersed
   attempts is best interpreted as Wayback-side replay unavailability
   rather than client-side network instability. These captures are
   classified `uninterpretable` in Stage C and contribute to interval
   censoring rather than to false transitions.

3. **HTML-body classifier is a heuristic.** The HTML-marker substring
   search (§4.1) is not a full HTML parser; it can in principle miss
   non-Latin-1 HTML preambles or overcount bodies whose first 4 KB
   happens to contain `<head>` literally inside a comment block. Spot
   checks of the 40,215 reclassified digests via the audit log show
   the false-positive rate to be negligible in our sample, but the
   heuristic should be re-validated if the script is run on a
   substantially different host distribution.

4. **Strict definition's path-dependence.** The strict ratio
   (55.5% across the broad sample) is computed against a simultaneous
   `first_control_block_ts`. A firm whose first control-group disallow
   is itself missing from the archive (uninterpretable captures) will
   appear strict by construction. We do not currently flag such cases
   separately; the share is small (most control-group disallows are
   on Googlebot, which has high Wayback capture density), but a
   sensitivity analysis that excludes ambiguous-control gvkeys would
   be appropriate at the manuscript stage.

5. **Cloud infrastructure costs.** The Stage A fetch consumed roughly
   3 hours of wall-clock time across 16 machines for round 1 plus
   five sequential single-shot rounds on 7 machines. The cloud
   instances should be released in the GCP/DigitalOcean console after
   each pipeline run; idle billing is the dominant marginal cost of
   re-running the fetch.

---

## 9. Reproducibility

The pipeline is deterministic given fixed inputs:

- Sharding uses `md5(...)[:8] mod num_shards`, which is stable across
  Python invocations (unlike the built-in `hash()`, which is
  PYTHONHASHSEED-randomized).
- The dedup-merge step uses an explicit best-rank ordering (success >
  deterministic-failure > transient-failure) so multiple-round runs
  always produce the same `digest_robots_cache.jsonl`.
- The HTML-reclassify pass is idempotent: re-running it on a corrected
  CSV finds no new rows to flip, because reclassified rows have
  `parse_status == "uninterpretable"` rather than `"ok"`, and the
  scan only flips `ok` rows.
- The gvkey aggregation uses `min` over per-bot first-block timestamps
  as the per-group reduction; ties are broken deterministically by
  underlying bot order (which is BOT_TAXONOMY insertion order in
  `04_parse_panel.py`).

To reproduce from scratch:

1. Confirm input availability:
   - `D:/anoconda/bot_paper_data/cdx_snapshots.jsonl`
   - `D:/anoconda/bot_paper_data/cdx_snapshots_new.jsonl`
   - `bot_paper/robots_parser.py`
   - `bot_paper/panel_construction/04_parse_panel.py`
   - `bot_paper/compustat_all_host_mapping.csv`
2. Run `extract_unique_digest_index.py` → `cdx_unique_digest_index.jsonl`.
3. Filter to `sample_status == "200"` → `cdx_digest_to_fetch.jsonl`.
4. Deploy and run `06_robots_daily_fetch_digests.py` over rounds 1–5
   per `deploy_06_robots_daily.sh`.
5. Run `merge_digest_robots_cache.py` (with `EXTRA_ROUNDS` set to
   `[("round3", 7), ("round4", 7), ("round5", 7)]`).
6. Run `parse_digest_robots_to_bot_policy.py`, then
   `dedup_digest_bot_policy.py`, then `reparse_too_large_digests.py`,
   then `reclassify_html_as_uninterpretable.py` (in this order).
7. Run `reconstruct_first_block_events.py`.
8. Run `dedup_first_block_events.py`.
9. Run `gvkey_first_block_aggregate.py`.
10. Run `daily_adoption_plots_and_strict.py` and
    `daily_strict_curve_and_regression_panel.py`.

The deterministic outputs at each step are stable; reruns produce
byte-identical files modulo float-printing formatting in summary
text files.
