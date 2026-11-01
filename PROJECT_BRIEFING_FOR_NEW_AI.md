# Project Briefing — Website Access Rules and AI Search Visibility

> Self-contained context for handing this project to a new AI assistant in a fresh chat window. Read top-to-bottom; everything you need to give useful feedback on supervisor comments is here.

**Author:** Xueyang Liu, first-year PhD, ESADE Business School (Universitat Ramon Llull)
**Supervisor:** Dr. Emanuele Rizzo (GREF research group)
**Date:** April 28, 2026

---

## 1. The research question

Do firms whose corporate websites restrict AI search crawlers (such as ChatGPT Search, Perplexity, Claude) become systematically less visible in AI-generated answers, and does this reduced visibility raise the cost of investor information acquisition and worsen the firm's information environment (analyst forecast dispersion, bid-ask spread, illiquidity)?

The story positions itself as the mirror image of Gao & Huang (RFS 2020) on EDGAR: where EDGAR lowered information acquisition costs and improved outsider information production, AI search restrictions raise costs and may degrade the information environment.

---

## 2. Treatment variable design (four tiers)

The paper defines a four-tier hierarchy of "blocking AI" measures, narrowing from broad rule existence to a specific overlap.

| Tier | Variable | Definition | Prevalence (non-fund obs.) |
|---|---|---|---|
| 0 | `any_block` | Any of 8 (now 62 in extended taxonomy) AI bots has any restriction in `robots.txt`, root or non-root | 46.8% |
| 1 | `search_block` | Any of 3 consumer AI search bots has any restriction | 45.4% |
| **2** | **`search_sitemap_block`** ⭐ | At least one URL in the firm's own sitemap is blocked by at least one of the 3 search bots — "sitemap–robots overlap" | **4.9%** (137 / 2,796 evaluable non-fund firms) |
| 3 | `search_block_specific` | Any of 3 search bots is blocked via an explicit `User-agent: <bot>` rule (not wildcard) | 1.2% (40 firms) |

**Tier 2 is the main treatment.** Rationale: 99% of Tier 0/1 "blocking" comes from generic `User-agent: *` wildcard rules that webmasters did not write with AI in mind (incidental). Tier 2 requires that the restricted URLs be ones the firm itself listed in its sitemap (declared discoverable content) — this is a directly observable conflict that is harder to dismiss as accidental.

**Three core search bots:**
- **OAI-SearchBot** (OpenAI / ChatGPT Search) — 12% blocking rate among top-1000 websites (Known Agents data)
- **PerplexityBot** (Perplexity AI) — 15%
- **Claude-SearchBot** (Anthropic / Claude) — 8%

These are the only AI search crawlers with double-digit webmaster awareness AND a downstream consumer product whose citation behavior we can audit experimentally. The remaining 13 documented AI search crawlers (Applebot, PetalBot, AzureAI-SearchBot, etc.) are excluded from the main treatment because of either low awareness, enterprise/regional positioning, or absence of a testable outcome (no public API or no structured citations).

---

## 3. Three layers of evidence

| Layer | Question | Status |
|---|---|---|
| **0 Descriptive** | Selection on observables, distribution of treatment | ✅ Done. Treated firms are larger (log_at norm-diff 0.57), more profitable (ROA 0.10), more intangible-intensive (intangibles ND 0.38), lower R&D (-0.26). Reported with Imbens-Wooldridge normalized differences in addition to t-tests. |
| **1 Mechanism (pilot)** | Does Tier-2 blocking actually reduce AI search citation rates? | ✅ Mostly done. 30 matched pairs × 5 queries × 3 platforms (ChatGPT manual, Claude via artifact, Perplexity API). Treated firms cited 4-11 pp less often in inline citations, consistent across platforms. |
| **2 Capital market consequences** | Does reduced AI visibility raise information acquisition costs and worsen analyst/liquidity outcomes? | 🟡 In progress. Outcome data merged (CRSP, I/B/E/S, Compustat). Identification: staggered DiD on robots.txt policy changes, 2021-2025 panel from Wayback Machine. Main shock candidate: GPTBot launch August 2023. |

---

## 4. Data construction (cross-section + panel)

### Cross-section (March 2026)
- ~9,500 US public firms from Compustat, 4,350 unique corporate hosts after dedup
- For each host: fetch `robots.txt` per RFC 9309, parse rules per the 62-bot taxonomy, fetch declared sitemaps, evaluate each sitemap URL × each bot rule for overlap
- Output: `analysis_sample.csv` (109 columns)

### Panel (2021-2025, monthly)
**Pipeline (in `bot_paper/panel_construction/`):**
| Step | Script | Status | Notes |
|---|---|---|---|
| 1 | `01_query_cdx.py` | ✅ | Wayback CDX API for all robots.txt snapshots per host |
| 2 | `02_select_monthly.py` | ✅ but **need redesign** | **Currently picks snapshot closest to day 15 of each month**. Supervisor says should use month-end. |
| 3 | `03_download_content.py` | ✅ | Downloads robots.txt content per selected snapshot |
| 4 | `04_parse_panel.py` | ✅ | Parses robots.txt under 62-bot taxonomy → `firm_panel_monthly` Tier 1 variables |
| 5 | `05_merge_firm_panel.py` | ✅ | Aggregates host × month → firm × month panel |
| 6 | `06_collect_sitemaps_for_panel.py` | ✅ | Fetches per-snapshot sitemap (returns sitemap_url, n_paths, but NOT the actual URL paths) |
| 6.5 | `06_5_collect_sitemap_paths_monthly.py` | 🟡 **running now on 5 cloud machines** | Re-fetches sitemap URL **paths** keyed by `(norm_key, year_month, digest)` so Step 7 can do per-snapshot URL-level Tier 2 matching |
| 7 | `07_build_search_sitemap_block_panel.py` | 🟡 partial | URL-level overlap matching → panel version of Tier 2 |
| 8 | `08_build_bot_change_events.py` | 🟡 | Identifies "switcher" firms (ever changed AI access policy) for staggered DiD |
| 9 | `09_merge_outcomes.py` | 🟡 | Merges CRSP / I/B/E/S / Compustat onto firm-month panel |

**Where 6.5 is right now:** 5 machines (1 AWS, 4 GCP) are running in parallel, sharded by stable hash. Each machine fetches ~24k of the 122k unique `(host, year_month, digest)` triples that need sitemap URL paths. About 9% complete. Dedup is on `(norm_key, year_month, digest)`.

---

## 5. Today's supervisor feedback (4 points)

The supervisor reviewed the panel design today. Four issues:

### 5.1. Snapshot timing — pick month-end, not month-mid
**Current:** Step 2 picks the snapshot closest to day 15 of each calendar month.
**Supervisor's view:** Should pick month-end (closest to last day). Reasoning: outcomes (CRSP monthly, Compustat quarterly, I/B/E/S monthly) are reported at month-end, so treatment state should be aligned to the same timestamp.
**Additional:** Use lagged month-end state to explain current-month outcomes (`L.search_block_eom` predicting `Y_t`) to avoid simultaneity (1 January state should not "explain" the entire January outcome).

### 5.2. Annual Compustat controls — switch to as-of (lagged annual or quarterly)
**Current:** Match annual Compustat fundamentals (fyear) onto every month using contemporaneous fyear.
**Supervisor's view:** This introduces look-ahead. For a month $t$, control variables should be the most recent value publicly available *as of* month $t$. Two implementations:
- **Lagged annual:** For all months in fiscal year $T+1$ before the 10-K is filed (typically $\sim$3 months after fiscal year end), use fyear $T-1$. After 10-K filing, use fyear $T$.
- **Quarterly Compustat (preferred if available):** Use Compustat Quarterly with `rdq` (report date of quarterly earnings). For month $t$, use the most recent quarter where `rdq ≤ end-of-month $t$`.

### 5.3. Multiple snapshots per month — within-month variation
**Issue:** A single calendar month can have multiple Wayback snapshots, but Step 2 picks only one. Information about within-month switches is lost.
**Supervisor's brainstorm:** majority vote (>50% of snapshots blocked → block); or day-weighted (block_share = days under "blocked" state / days in month).
**My (and one external AI's) recommendation:**
- **Main spec:** end-of-month last snapshot (transparent, aligns with outcomes)
- **Robustness:** day-weighted exposure under piecewise-constant policy assumption — but report this as a robustness check with explicit assumption disclosure, NOT as the main definition
- **Conservative bracket:** `any_blocked_in_month` (upper bound) and `all_blocked_in_month` (lower bound) — both are assumption-free
- **Reject majority vote:** Wayback snapshot frequency is not random sampling, so snapshot-count majority is a noisy aggregator (e.g., 5 snapshots with 3 blocked ≠ 60% of days blocked)

### 5.4. CRSP filters — standard market microstructure hygiene
Apply when defining the regression sample / outcome construction:
- `EXCHD ∈ {1, 2, 3}` — NYSE / AMEX / NASDAQ only
- `SHRCD ∈ {10, 11}` — common stock only (excludes ADRs, preferred, REITs, units, warrants, closed-end funds)
- `L.PRC ≥ 1` (or `≥ 5`) — exclude penny stocks, **using lagged price** to avoid using outcome-period price as a filter
- Exclude SIC 6000-6999 (financials) — supervisor's standard view; my CLAUDE.md note pushes back on this for the baseline (financials' websites have investor-relevant content) but I will run as appendix robustness

---

## 6. Specific question to the new AI: should I stop 6.5 right now?

### The audit I ran today
Compared current Step 2 (day-15) selection against EOM rule across all 239,955 (host, year_month) cells in the CDX panel:

```
single_snapshot (no ambiguity):              79,573  (33.2%)
multi_same_digest (different snaps, same digest): 128,243  (53.4%)
multi_diff_digest (EOM picks DIFFERENT digest):   32,139  (13.4%)

v1 (currently in panel) cell universe:       56,961
  EOM picks SAME digest as v1:               50,552  (88.7%)
  EOM picks DIFFERENT digest:                 6,409  (11.3%)  ← waste fraction
```

**11.3% of the cells 6.5 is currently fetching would need a different digest under EOM rule.** This means 6.5's work for ~88.7% of cells is reusable; for the remaining 11.3%, the sitemap paths fetched will not match what the EOM-aligned Step 7 needs.

### My current decision
**Continue 6.5, do not stop.** Reasoning:
1. 11.3% is in the "judgment call" zone (between 10-25% threshold)
2. 06_5 has dedup on `(norm_key, year_month, digest)` — incremental top-up after Step 2 redesign is cheap
3. Stopping now wastes the ~5,000 cells already cleanly fetched + 5 hours of cloud compute
4. After 6.5 finishes, change Step 2 → EOM, rerun Step 6 v1 only for the 6,400 affected cells, rerun 6.5 (dedup skips the 50k unchanged cells)

### What I want a second opinion on
1. Is the "continue 6.5, top up later" decision correct given 11.3% waste?
2. Is the "main = month-end last snapshot" specification right, with day-weighted as robustness only?
3. Am I missing any Step 6/7 dependency on the snapshot timestamp itself (not just the digest)? Even if EOM picks the same digest as day-15, the timestamp differs, and downstream merges may be timestamp-keyed.
4. Is there a cleaner approach to the within-month variation problem that I haven't considered?

---

## 7. Surrounding constraints

- **Submission deadlines:** ESADE PhD Research Proposal must be presented to research group by early June 2026, submitted June 19, defended July 4-10. Plus an ETH Zurich AI & Applied Economics Summer School application (deadline April 30) where this paper is the writing sample.
- **Data infrastructure:** Local Windows machine + cloud (AWS EC2 + 4 GCP) for Wayback fetching. Stata for analysis. Python for data construction.
- **Survivorship-bias correction:** Beyond the March 2026 cross-section (4,350 hosts), the panel adds ~4,920 host-firms that existed during 2021-2024 but exited before March 2026 (mostly via delisting / M&A / bankruptcy). Among 195 of these with EDGAR CIKs, I recovered 142 historical website URLs from EDGAR filings. The remaining 915 no-CIK firms are 96% ETFs (SIC 6722) without corporate websites.

---

## 8. Variable definitions you may need

```
search_sitemap_block (Tier 2, MAIN)
  = 1 if any sitemap-listed URL is blocked by any of {OAI-SearchBot, PerplexityBot, Claude-SearchBot}
  = 0 if sitemap URLs are observed and none are blocked
  = NA if no sitemap data or fetch error

ai_specific_intent (heterogeneity variable)
  = 1 if any AI bot is blocked AND no traditional search engine bot
       (Googlebot, Bingbot, Baiduspider, etc.) is blocked
  = 0 otherwise
  Captures "deliberately blocked AI but kept traditional search" — distinguishes
  AI-aware webmasters from blanket-block defaults.

L.search_block_eom (proposed main panel treatment after redesign)
  = month-end search_block status from previous month, used to explain Y_t
```

---

## 9. References (key papers)

- **Gao & Huang (RFS 2020)** — EDGAR digitalization → outsider information production. The closest analog and main motivating reference.
- **Goldstein, Yang, Zuo (JAR 2023)** — EDGAR's real effects on investment and financing.
- **Boulland, Bourveau, Breuer (JAR 2025)** — Corporate websites as a disclosure channel; this paper extends by showing that *configuration*, not just content, matters.
- **Da, Engelberg, Gao (JF 2011)** — In Search of Attention. Investor information acquisition via Google search.
- **Drake, Roulstone, Thornock (JAR 2012)** — Investor information demand around earnings via Google search.
- **Imbens & Wooldridge (JEL 2009)** — Normalized differences for balance assessment in observational studies.
- **Callaway & Sant'Anna (JoE 2021)** — Heterogeneous-effect estimator for staggered DiD.
- **Longpre et al. (NeurIPS 2024)** — Consent in Crisis: rapid decline of AI data commons (28% of Common Crawl now restricted).
- **Cui et al. (CCS 2025)** — robots.txt governance for LLM crawlers.

---

## 10. What you can helpfully give feedback on

- The supervisor's 4 points: any concerns or refinements?
- The 11.3% "continue 6.5" decision: do you agree, or would you stop now?
- The treatment variable hierarchy (Tier 0-3): is the framing convincing? Are there obvious robustness specifications I'm missing?
- The identification strategy (staggered DiD around GPTBot launch / firm-level robots.txt policy changes): what threats to identification should I worry about beyond what's listed?
- The three-platform pilot (ChatGPT, Perplexity, Claude): does the consistent -4 to -11 pp citation effect translate cleanly into a Layer-2 capital market story, or are there reasons to expect the cross-sectional pilot effect not to scale to the panel?
- The exclusion of Gemini from the pilot (because Gemini grounds via Googlebot, not a dedicated AI search bot): how would you frame this in the paper?
