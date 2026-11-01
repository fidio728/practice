# Context: Bot taxonomy (measurement) + post-event-UA design (identification)

> **For new agent / future-self reading this:** Self-contained plan that
> separates two distinct purposes — measurement and identification — and
> assigns a different bot set to each. Don't ask the user for project
> background; read this file then execute § 9 (Implementation) when given
> the go.
>
> **Naming disambiguation**: this document uses
> - **Layer A / B / C** for paper structure (measurement / identification / control)
> - **CF-Tier** for the underlying Cloudflare bot directory category
>   (CF-Tier 1 = AI Search + AI Crawler; CF-Tier 2 = AI Assistant; CF-Tier 3 = Search Engine Crawler)
>
> These are DIFFERENT from "Tier 1 panel / Tier 2 panel" (data construction
> coarseness — Tier 1 = robots-level any-block, Tier 2 = sitemap-conditioned)
> and "Tier 3 specificity ladder" (UA-naming specificity dimension). Don't
> conflate.

---

## TL;DR (one-liner)

**AI Crawler 算，而且在 measurement 上是主角;但在 clean DID 设计里,
不是所有 AI Crawler 都该包括,只有那些"事件前结构上不可能出现、
事件后才被 vendor 引入"的 UA 才该进主 battery。**

The paper has two distinct uses for "AI bot" classification:

1. **Measurement (broad)** — CF AI Search + AI Crawler (~41 bots) → prevalence,
   taxonomy table, determinants regression.
2. **Identification (narrow)** — 4 post-event-defined UAs → clean LP-DID
   event studies with structural pre-period $D = 0$.

Use the broad set for Sections 2–3 of the paper (descriptive + determinants).
Use the narrow set for Section 4 onwards (causal claims).

---

## 1. Two purposes, two bot sets

### Layer A — Measurement (broad)

**Bot set**: CF-Tier 1 = AI Search (7) + AI Crawler (34) ≈ **41 bots**.
Plus user's existing 69-bot list as the curated narrower comparison.

**Use cases**:
- Prevalence statistics: what fraction of CRSP-active firms block any
  AI bot in robots.txt over time.
- Determinants regression (the advisor's "first analysis"): which firm
  characteristics predict block rates? (already implemented in
  `13_determinants_of_blocking.do`).
- Taxonomy table for paper (appendix): list of 41 CF AI bots + 28 user-only
  bots not in CF directory (CCBot, Bytespider, DeepSeekBot, etc).
- Overall robots.txt-level "AI block intensity" measures.

**Why not used for causal claims**: most of these bots existed before
2024. Pre-existing block decisions are selected on firm characteristics
(size, web presence, sector). Strategy 1 (`12_bot_launch_did.do`)
established this decisively: pretrend rejected at $p < 10^{-5}$ in 30/30
(definition × outcome) combinations, and `sic2 × ym` FE doesn't rescue.
See `project_oai_launch_did_failed_2026_05_07.md` memory note.

### Layer B — Identification (narrow event-study battery)

**Bot set**: **4 UAs** for main battery, **7 UAs** for appendix robustness.

#### Main battery (4)

| UA | Operator | Introduction date | Group |
|---|---|---|---|
| **GPTBot** | OpenAI | **2023-08-08** | Training |
| **Google-Extended** | Google | **2023-09-28** | Training |
| **Applebot-Extended** | Apple | **2024-06-10** (WWDC) | Training |
| **OAI-SearchBot** | OpenAI | **2024-07-25** (SearchGPT) | Search |

#### Appendix expansion (+ 3)

| UA | Operator | Introduction date | Group |
|---|---|---|---|
| ClaudeBot | Anthropic | ~2023-09 | Training |
| meta-externalagent | Meta | ~2024-05 | Training |
| Claude-SearchBot | Anthropic | ~2024-10 | Search |

**Three criteria a UA must satisfy to enter Layer B**:

1. **Public, dated introduction event** (vendor blog post / spec
   announcement).
2. **Structurally cannot appear in robots.txt before that date** (the
   string did not exist in any standard reference; firms could not have
   pre-existing rules naming it).
3. **Sufficient post-event adoption** (≥ 50–100 firms in the in-sample
   panel write the UA explicitly within the panel window).

The first two conditions are the source of the design's identification
power: pre-event $D = 0$ for all firms by construction, eliminating
selection on unobserved trends.

**Use cases**:
- LP-DID event studies (Girardi canonical, absorb-time-only).
- BJS imputation for average ATT.
- dCDH for staggered treatment.
- Vendor-decomposition spec (run each UA separately, report cross-UA
  consistency).
- Pooled-4 spec (firm treated at first month adopting any of the 4).

**Why narrower than Layer A**: most CF AI Crawlers are pre-event UAs
(Applebot, Amazonbot, PetalBot, GoogleOther, etc.) that existed before
2023 and would inherit the Strategy-1 selection problem. Restricting to
post-event UAs is what gives the design its clean structural pre-period.

### Layer C — Controls / placebo

**Bot set**: CF-Tier 3 = Search Engine Crawler (~50 bots), and Layer-B
counterpart per-UA controls.

**Use cases**:
- Placebo regression: blocking traditional search crawlers (Googlebot,
  Bingbot, etc.) should yield null effect on outcomes if the paper's
  channel is AI-specific.
- Per-Layer-B control: never-treated firms within the panel window for
  the LP-DID counterfactual.

**Caveat**: Googlebot is not a clean placebo because Google feeds Gemini
and AI Overviews from the Search index. Run Layer C with and without
Googlebot.

### CF-Tier 2 (AI Assistant ~50 bots) — robustness, NOT main

User-triggered (`Perplexity-User`, `ChatGPT-User`), agentic
(`Amazon Buy For Me`, `Manus Bot`, `Devin`), and signed-agent
(`Anchor Browser`, `Browserbase`, etc). Mostly post-2024-Q3.

**Why not Layer A or B**:
- Most are Signed Agents using HTTP message signatures, not
  robots.txt-controllable.
- User-triggered nature blurs the consent model.
- Adoption window too short (< 12 months in panel).

**Use case**: optional "AI Assistant exposure" robustness check; not main
treatment.

---

## 2. Treatment variable naming convention

Stata 32-character limit respected throughout.

### Per-UA flags (host-month → firm-month aggregation: `max` over hosts)

| Variable | Meaning | Type |
|---|---|---|
| `mention_gpt` | firm-month robots.txt has any directive (Allow OR Disallow) addressing GPTBot | binary |
| `mention_gext` | same, Google-Extended | binary |
| `mention_aext` | same, Applebot-Extended | binary |
| `mention_oai` | same, OAI-SearchBot | binary |
| `disallow_gpt` | firm-month robots.txt has Disallow rule addressing GPTBot | binary |
| `disallow_gext` | same | binary |
| `disallow_aext` | same | binary |
| `disallow_oai` | same | binary |

### First-event indicators (absorbing)

| Variable | Meaning |
|---|---|
| `first_mention_gpt` | 1 in firm's first month with `mention_gpt = 1`, else 0 |
| `first_mention_gext` | same |
| `first_mention_aext` | same |
| `first_mention_oai` | same |
| `first_disallow_gpt` | 1 in firm's first month with `disallow_gpt = 1`, else 0 |
| `first_disallow_gext` | same |
| `first_disallow_aext` | same |
| `first_disallow_oai` | same |

### Cohort timing (firm-level constants)

| Variable | Meaning |
|---|---|
| `cohort_mention_gpt` | year_month string ("2023-08", ..., or `.` if never) |
| `cohort_mention_gext` | same |
| `cohort_mention_aext` | same |
| `cohort_mention_oai` | same |
| `cohort_disallow_gpt` | same for disallow |
| `cohort_disallow_gext` | same |
| `cohort_disallow_aext` | same |
| `cohort_disallow_oai` | same |

### Pooled aggregates

| Variable | Meaning |
|---|---|
| `mention_any4` | max of `mention_gpt mention_gext mention_aext mention_oai` |
| `disallow_any4` | max of `disallow_*` |
| `first_mention_any4` | 1 in firm's earliest first-mention month across the 4 UAs |
| `first_disallow_any4` | same |
| `cohort_mention_any4` | min of the 4 `cohort_mention_*` (earliest engaged UA) |

Appendix battery uses `_any7` suffix:
ClaudeBot (`cbot`), meta-externalagent (`meta`), Claude-SearchBot (`csearch`).

### Layer A aggregate (broad measurement)

| Variable | Meaning |
|---|---|
| `cf_tier1_specific` | max(`specific_rule = 1`) for any host of firm's any CF-Tier-1 bot in t |
| `cf_tier1_blocked` | max(`blocked_any = 1`) for any host of firm's any CF-Tier-1 bot in t (incl wildcard) |
| `cf_tier3_specific` | same for CF Search Engine Crawler (placebo) |
| `cf_tier3_blocked` | same |
| `cf_tier2_specific` | same for CF AI Assistant (robustness only) |
| `cf_tier2_blocked` | same |

### Constants stored as macros + sidecar CSV

```stata
* Public introduction dates (Stata monthly date format)
local d_gpt   = ym(2023, 8)   // 2023-08-08
local d_gext  = ym(2023, 9)   // 2023-09-28
local d_aext  = ym(2024, 6)   // 2024-06-10
local d_oai   = ym(2024, 7)   // 2024-07-25
local d_cbot  = ym(2023, 9)
local d_meta  = ym(2024, 5)
local d_csearch = ym(2024, 10)
```

Sidecar `bot_paper/panel_construction/post_event_ua_dates.csv`:

```csv
ua,operator,intro_date,group,in_main4
GPTBot,OpenAI,2023-08-08,training,1
Google-Extended,Google,2023-09-28,training,1
Applebot-Extended,Apple,2024-06-10,training,1
OAI-SearchBot,OpenAI,2024-07-25,search,1
ClaudeBot,Anthropic,2023-09-15,training,0
meta-externalagent,Meta,2024-05-15,training,0
Claude-SearchBot,Anthropic,2024-10-15,search,0
```

The `in_main4` flag lets Stata `merge` switch between main and appendix
battery in one line.

---

## 3. Caveat: Allow vs Disallow distribution (must diagnose before
   running main)

Most robots.txt files contain `Disallow:` directives almost exclusively.
Pure `Allow:` entries are rare. When `Allow` does appear, it's typically
in a conditional pattern:

```
User-agent: GPTBot
Disallow: /
Allow: /public/blog/
```

This means "block GPTBot in most places, but allow in /public/blog" — a
**conditional restriction**, not "we welcome GPTBot".

So `mention_<ua> = 1` actually contains three semantic types:

1. Pure `Disallow` (full block) — majority
2. Pure `Allow` (full welcome) — rare
3. Mixed `Disallow + Allow` (granular policy) — moderate

**Required diagnostic before main regression**: compute, for each of the 4
main UAs, the fraction of `mention = 1` cells that have `disallow = 1`.

| UA | `mention = 1` count | `disallow = 1` count | ratio |
|---|---|---|---|
| GPTBot | ? | ? | ? |
| Google-Extended | ? | ? | ? |
| Applebot-Extended | ? | ? | ? |
| OAI-SearchBot | ? | ? | ? |

Decision rule:
- If ratio ≥ 95% for all 4 UAs → `mention` and `disallow` are nearly
  identical; `mention` is just a slightly broader `disallow`. The
  "governance adoption vs access restriction" framing has limited
  empirical bite. Use `disallow` as primary, drop the framing.
- If ratio is 70–90% → real distinction. Both `mention` and `disallow`
  give different signal; "governance adoption" framing is justified.
- If ratio < 70% → many firms are using `mention` for granular `Allow`
  rules. The interpretation gets complex. Investigate sub-cases.

---

## 4. Estimation strategy

### Primary: LP-DID (Local Projection DID, Girardi canonical)

For each (UA, outcome, horizon $h$) cell:

$$
\Delta_h y_{i,t} \equiv y_{i,t+h} - y_{i,t-1} = \beta_h^{ua} \cdot \text{first\_mention}_{i,t}^{ua} + X_{i,t-1}'\gamma + \delta_t + \varepsilon_{i,t,h}
$$

- $h \in \{-3, -2, 0, 1, 2, 3, 6, 9, 12\}$ (months relative to first event)
- $X_{i,t-1}$ = M2 lagged controls (size, leverage, ROA, M/B, tangibility,
  R&D, dividend payer, log_urls, log_hosts)
- $\delta_t$ = year-month FE only (canonical Girardi specification — does
  NOT include firm FE; firm FE absorbs treatment timing variation in
  absorbing-event setting)
- Cluster: two-way (gvkey_n, ym)
- Pre-trend test: joint nullity of $\beta_{-3}$ and $\beta_{-2}$

Stress-test variant: same with `absorb(gvkey_n ym)` (firm + time FE) — expect
attenuation; report as appendix.

### Robustness 1: BJS imputation

`did_imputation` Stata package. Provides single average-ATT scalar per
(UA, outcome) cell + dynamic profile.

### Robustness 2: dCDH

`did_multiplegt_dyn`. Allows reversible treatment (firms can un-mention
later). Provides Av_tot_eff + pre-trend joint test.

### Per-UA outcomes

Two main outcomes:
- $\ln(\text{Amihud})$ — illiquidity
- $\ln(\text{mktcap})$ — valuation

Two appendix outcomes:
- $\ln(\text{qspread})$ — bid-ask
- $\ln(\text{turnover})$ — trading intensity

Plus IBES information-environment outcomes (numest, disp, stdev) in
robustness.

---

## 5. Paper-ready empirical design paragraph (LaTeX-ready)

```latex
\subsection{Identification: post-event-defined user-agent natural experiments}
\label{sec:identification}

Our identification strategy exploits a series of public events at which AI
vendors specified new user-agent (UA) strings for their crawlers, naming
each UA in robots.txt-readable documentation on a precise public date. Prior
to that date, the UA string did not exist in any standard reference and
could not appear in firms' robots.txt files; pre-event treatment status is
therefore a structural property of the data, not the result of unobserved
firm selection. This is the principal advantage of the design over
alternatives based on pre-existing latent restrictions (e.g., wildcard
\texttt{Disallow:~/} rules or sitemap directives), in which the
treatment-control composition reflects unobserved firm characteristics
correlated with outcome trends---a problem that, as we document in
Appendix~\ref{app:strategy1-failed}, defeats parallel-trends identification
across all combinations of treatment definition and outcome we attempted.

We focus on four post-event UAs spanning two AI use cases (training and
search) and three vendors (OpenAI, Google, Apple), with introduction dates
in chronological order:
%
\begin{enumerate}
    \item \texttt{GPTBot} (OpenAI): announced 2023-08-08 as OpenAI's
          training crawler with explicit robots.txt opt-out support.
    \item \texttt{Google-Extended} (Google): announced 2023-09-28 as the
          training opt-out signal for Bard / Gemini.
    \item \texttt{Applebot-Extended} (Apple): announced 2024-06-10 at WWDC
          alongside Apple Intelligence, as the training opt-out for
          Apple's foundation models.
    \item \texttt{OAI-SearchBot} (OpenAI): announced 2024-07-25 with the
          SearchGPT prototype, as OpenAI's user-facing search crawler.
\end{enumerate}
%
For each UA \emph{ua} and firm $i$ in month $t$, we define two parallel
treatment indicators capturing different layers of firm response:
%
\begin{align}
    M_{i,t}^{ua} &= \mathbf{1}\{\text{firm } i\text{'s robots.txt contains any directive (Allow or Disallow) addressing UA } ua\text{ in month } t\}\\
    B_{i,t}^{ua} &= \mathbf{1}\{\text{firm } i\text{'s robots.txt contains a } \texttt{Disallow}\text{ directive addressing UA } ua\text{ in month } t\}
\end{align}
%
The mention indicator $M$ captures firm engagement with AI bot
governance---an explicit policy stance, regardless of direction---while
the block indicator $B$ restricts to access restriction. We treat $M$ as
the primary outcome measure of \emph{AI governance adoption} and $B$ as
the secondary measure of \emph{AI access restriction}. Each indicator
admits an absorbing first-event variant: $M_{i,t}^{\star,ua} = 1$ in the
first month firm $i$ has $M = 1$ and zero before; analogously
$B_{i,t}^{\star,ua}$. Because each UA was specified at a public date $d^{ua}$,
$M_{i,t}^{ua} = B_{i,t}^{ua} = 0$ for all $i$ at all $t < d^{ua}$ by
construction; the pre-period contains no treated firms.

We estimate dynamic effects via local-projection difference-in-differences
\citep{dube2023local}. For each UA and outcome $y \in \{\ln(\text{Amihud}),
\ln(\text{mktcap}), \dots\}$, and each event-time horizon
$h \in \{-3, -2, 0, 1, 2, 3, 6, 9, 12\}$:
%
\begin{equation}
\Delta_{h} y_{i,t} \equiv y_{i,t+h} - y_{i,t-1} = \beta_h^{ua} \cdot M_{i,t}^{\star,ua} + X_{i,t-1}'\gamma + \delta_t + \varepsilon_{i,t,h}
\end{equation}
%
where $\Delta_h$ is the change in outcome from one month before
treatment to $h$ months after, $X_{i,t-1}$ is a vector of lagged firm
characteristics (size, leverage, ROA, market-to-book, tangibility, R\&D,
dividend status, web exposure), $\delta_t$ are calendar-month fixed
effects, and standard errors are two-way clustered on firm and month
\citep{cameron2011robust}. The specification follows the canonical
\citet{girardi2024lpdid} form, absorbing time only and not firm fixed
effects, since the latter would absorb part of the treatment timing
variation in this absorbing-event setting. We separately report a
firm-fixed-effect variant as a stress test.

Pre-trend is tested via the joint nullity of $\beta^{ua}_{-3}$ and
$\beta^{ua}_{-2}$. Because the pre-period structurally cannot contain
treated observations, this test is principally a check on
\emph{control-side} dynamics rather than treatment-side selection.
We supplement LP-DID with the imputation estimator of
\citet{borusyak2024revisiting} (BJS), which provides a single
average-ATT scalar for each (UA, outcome) cell, and the
\citet{deCh2020twoway} package for staggered settings with reversible
treatment as a robustness check.

We aggregate across the four UAs in two ways. The \emph{pooled-4} spec
treats firm $i$ as treated in the first month it adopts $M^{\star,ua} = 1$
for any $ua \in \{$GPTBot, Google-Extended, Applebot-Extended,
OAI-SearchBot$\}$, with cohort-specific first-event indicators. The
\emph{vendor-decomposition} spec runs each UA separately and reports
cross-UA consistency. Section~\ref{sec:appendix-7ua} extends the
battery to ClaudeBot, meta-externalagent, and Claude-SearchBot for
robustness.
```

Bib placeholders to replace: `dube2023local`, `borusyak2024revisiting`,
`girardi2024lpdid`, `deCh2020twoway`, `cameron2011robust`.

---

## 6. Files / state inventory

### 6.1 Existing infrastructure (anchor — don't break)

| Path | Role |
|---|---|
| `bot_paper/panel_construction/04_parse_panel.py:48-200` | `BOT_TAXONOMY` dict — user's 69-bot list with aliases. Both `Applebot` and `Applebot-Extended` already coded as separate UAs (lines 56, 75). |
| `bot_paper/panel_construction/robots_panel_monthly.csv` | LONG: (host, ym, bot, specific_rule, blocked_any, root_block, has_nonroot_disallow). 1.18M rows for Applebot+Extended alone. |
| `bot_paper/panel_construction/firm_panel_monthly.csv` | gvkey-host map for aggregation. |
| `bot_paper/panel_construction/treatment_anatomy_firmmonth.csv` | WIDE per (gvkey, ym): existing flags `search16_specific`, `core3_specific`, `training29_specific`, `all69_specific`. Will coexist with new flags. |
| `bot_paper/panel_construction/gvkey_month_tier2_reg.dta` | Main panel (in_sample, controls, outcomes). |
| `bot_paper/stata/.../12_main_v2.do` | Headline LP-DID (legacy, narrow user-list). |
| `bot_paper/stata/.../13_determinants_of_blocking.do` | First-analysis (advisor's main). |
| `bot_paper/draft_2026_05_06/table_strategy_tour.tex` | 6-panel master comparison (will be partially superseded). |

### 6.2 New files this work item creates

| Path | Role |
|---|---|
| `bot_paper/panel_construction/cf_bot_taxonomy.py` | NEW — Cloudflare 3-tier dictionary (CF-Tier 1 ~41, CF-Tier 2 ~50, CF-Tier 3 ~50). For Layer A measurement. |
| `bot_paper/panel_construction/post_event_ua_dates.csv` | NEW — sidecar table of 7 post-event UAs + introduction dates + main4-flag. For Layer B identification. |
| `bot_paper/panel_construction/build_cf_tier_panel.py` | NEW — produces `cf_tier_panel.csv` with `cf_tier1_specific/blocked`, `cf_tier2_*`, `cf_tier3_*` flags per (gvkey, ym). |
| `bot_paper/panel_construction/build_post_event_ua_panel.py` | NEW — produces `post_event_ua_panel.csv` with `mention_<ua>` and `disallow_<ua>` flags + first-event + cohort columns for each of 7 UAs + pooled `_any4`/`_any7` aggregates. |
| `bot_paper/panel_construction/cf_tier_panel.csv` | NEW — output, keyed (gvkey, year_month). |
| `bot_paper/panel_construction/post_event_ua_panel.csv` | NEW — output, keyed (gvkey, year_month). |
| `bot_paper/panel_construction/compare_taxonomy.py` | NEW — set-diff user's 69 vs Cloudflare ~141, output `taxonomy_crosswalk.csv`. |
| `bot_paper/panel_construction/taxonomy_crosswalk.csv` | NEW — bot-level mapping table. |
| `bot_paper/stata/.../11c_merge_cf_tier_panel.do` | NEW — merge `cf_tier_panel.csv` into main panel. |
| `bot_paper/stata/.../11d_merge_post_event_ua_panel.do` | NEW — merge `post_event_ua_panel.csv` into main panel. |
| `bot_paper/stata/.../14_layerA_measurement.do` | NEW — Layer A determinants regression on `cf_tier1_specific` + placebo on `cf_tier3_specific`. |
| `bot_paper/stata/.../15_layerB_post_event_uas_did.do` | NEW — single-file + loop driving 4 UAs × 2 treatments × 2 outcomes × 9 horizons (~144 cells, ~5 min Stata). LP-DID baseline. |
| `bot_paper/stata/.../15_layerB_pooled4_did.do` | NEW — pooled-4 spec (firm treated at first-of-any-4-UA month). |
| `bot_paper/stata/.../15_layerB_appendix_7ua_did.do` | NEW — appendix expansion to 7 UAs. |
| `bot_paper/stata/.../15_layerB_bjs.do` | NEW — BJS imputation robustness for the 4 UAs. |
| `bot_paper/draft_2026_05_06/section_post_event_ua_design.tex` | NEW — paper section text (the LaTeX in § 5 of this doc). |

---

## 7. Implementation plan

### Step 7.1 Diagnostics first (do not skip)

Before writing any new Stata, run a Python diagnostic on the existing
`robots_panel_monthly.csv` to verify the design is feasible:

For each of the 4 main UAs:

1. **First-appearance month verification**: confirm
   `min(year_month where bot=<ua> AND specific_rule=1)` matches the
   public introduction date (within ±1 month). If a UA shows
   `specific_rule = 1` *before* its public date, flag immediately.
2. **In-sample firm count**: count unique gvkeys with
   `mention_<ua> = 1` in any month, restricted to `in_sample = 1`. Need
   ≥ 50 per UA for adequate power. If < 50, flag for the user — that
   UA may need to be dropped from main battery.
3. **Allow-vs-Disallow ratio**: per § 3 caveat, compute fraction of
   `mention = 1` cells with `disallow = 1`. Decide whether `mention` is
   meaningfully different from `disallow`.

Output: `post_event_ua_diagnostic.txt` with these three statistics per
UA. Show user before continuing.

Required gate: all 4 UAs pass criteria (1) + (2) + (3-decision-made)
before proceeding to Step 7.2.

### Step 7.2 Build `cf_bot_taxonomy.py` (Layer A) and `post_event_ua_dates.csv` (Layer B)

#### `cf_bot_taxonomy.py`

Mirror `BOT_TAXONOMY` structure in `04_parse_panel.py`. Encode all
~141 Cloudflare AI bots (Tier 1 + Tier 2 + Tier 3) with canonical UA
names, aliases, operator, CF category, and CF-Tier assignment.

Schema:
```python
CF_BOT_TAXONOMY = {
    "Applebot": {
        "cf_tier": 1, "cf_cat": "AI Search", "operator": "Apple",
        "aliases": ["applebot"]
    },
    # ... ~141 entries total
}
```

#### `post_event_ua_dates.csv`

7 rows, one per post-event UA. Schema as in § 2.

### Step 7.3 Run `compare_taxonomy.py`

Set-diff user's 69 vs Cloudflare ~141. Output `taxonomy_crosswalk.csv`
columns: `bot_name`, `in_user_69`, `user_group`, `in_cf_directory`,
`cf_tier`, `cf_category`, `operator`, `respect_robots_txt` (from
ai-robots-txt's robots.json).

Stop here, present crosswalk to user.

### Step 7.4 Build firm-month flag panels

#### `build_cf_tier_panel.py` → `cf_tier_panel.csv`
6 columns: `cf_tier1_specific`, `cf_tier1_blocked`, `cf_tier2_*`,
`cf_tier3_*`. Aggregate from `robots_panel_monthly.csv` long format,
joined to gvkey via `firm_panel_monthly.csv`.

#### `build_post_event_ua_panel.py` → `post_event_ua_panel.csv`
For each of 7 UAs: `mention_<stub>`, `disallow_<stub>`,
`first_mention_<stub>`, `first_disallow_<stub>`, `cohort_mention_<stub>`,
`cohort_disallow_<stub>`. Plus `_any4`/`_any7` aggregates. Total:
6 cols × 7 UAs + 5 pooled = ~47 columns.

### Step 7.5 Stata merges

`11c_merge_cf_tier_panel.do` and `11d_merge_post_event_ua_panel.do`.
Mirror `11b_merge_tier2_panel.do` structure. Output sidecar dta files;
do NOT modify the main `gvkey_month_tier2_reg.dta`.

### Step 7.6 Layer A regressions

`14_layerA_measurement.do`:

- Determinants spec: 6 LHS = {`cf_tier1_specific`, `cf_tier1_blocked`,
  `cf_tier2_specific`, `cf_tier2_blocked`, `cf_tier3_specific`,
  `cf_tier3_blocked`}, RHS = M2 lagged controls, absorb(sic2 ym),
  cluster(gvkey_n ym).
- Output: `14_layerA_measurement_summary.csv` + 1 LaTeX table.

### Step 7.7 Layer B event studies (the meat)

`15_layerB_post_event_uas_did.do`:
- Outer loop over 4 UAs × 2 treatments (`first_mention`, `first_disallow`)
  × 2 outcomes (ln_amihud, ln_mktcap) × 9 horizons (-3, -2, 0, 1, 2, 3,
  6, 9, 12) = 144 cells.
- Inner: `reghdfe Δh_y first_<treat>_<ua> X_lag if sample, absorb(ym)
  cluster(gvkey_n ym)`.
- Write to `15_layerB_post_event_uas_did_summary.csv`.
- Also report joint pre-trend $F$ for $\beta_{-3}, \beta_{-2}$.

`15_layerB_pooled4_did.do`:
- Same structure but treatment = `first_mention_any4` /
  `first_disallow_any4`.

`15_layerB_appendix_7ua_did.do`:
- Same structure but adds `cbot`, `meta`, `csearch`. Use `_any7`
  pooled spec.

`15_layerB_bjs.do`:
- For each of the 4 UAs, run `did_imputation` for both treatment
  definitions and 2 outcomes. Output average ATT and event-time
  profile.

### Step 7.8 Plots and tables

- 8 event-study plots (4 UAs × 2 outcomes), with mention treatment as
  primary line and disallow treatment as secondary line.
- 1 summary table (UA × outcome × {β₀, β₃, β₆, β₁₂, pre-trend p, n_treated})
- 1 pooled-4 plot (single line, primary outcome ln_amihud).
- 1 cross-UA consistency table (each UA's $\beta_6$ and CI).
- Update `table_strategy_tour.tex` to integrate the new event studies in
  Panel B (replacing Strategy 2 stacked-DID Table B headline).

---

## 8. Acceptance criteria

A successful run produces:

1. **Diagnostic file (Step 7.1) passes** for all 4 main UAs:
   - First-appearance month within ±1 month of public date
   - ≥ 50 in-sample treated firms per UA
   - Mention vs Disallow ratio decision documented
2. **`taxonomy_crosswalk.csv`** covers ≥ 95% of bots in either user's
   69 or CF's ~141.
3. **`cf_tier_panel.csv`** keyed (gvkey, year_month), 6 + diagnostic
   columns. CF-Tier 1 first non-zero in 2023-08 (GPTBot intro).
4. **`post_event_ua_panel.csv`** keyed (gvkey, year_month), ~47 columns.
   Each UA's `mention_<ua>` first non-zero in its public-intro month
   (sanity: not before).
5. **`15_layerB_post_event_uas_did_summary.csv`** has 144 rows, no
   `FAILED` cells.
6. **At least 2 of the 4 UAs have meaningful pre-trend pass** ($p > 0.10$
   on joint $F$ for $\beta_{-3}, \beta_{-2}$). If 0/4 pass, the
   identification design is not as clean as theorized — flag for the
   user, do not push the paper.
7. **CF-Tier 3 placebo on consequences is null** (no significant β on
   ln_amihud or ln_mktcap when treatment is `cf_tier3_specific`).
   If significant, the channel is not AI-specific — flag.
8. **Paper section LaTeX** at
   `bot_paper/draft_2026_05_06/section_post_event_ua_design.tex` is
   ready to `\input{}` into master draft.

---

## 9. What NOT to do

- **Don't** delete or rename user's 69-bot taxonomy in `04_parse_panel.py`.
  CF-Tier flags coexist alongside `search16_specific`, `core3_specific`,
  etc.
- **Don't** re-parse `robots_panel_monthly.csv` from scratch. It already
  has per-UA `specific_rule` / `blocked_any` flags. Just aggregate.
- **Don't** break `12_main_v2.do` / `13_determinants_of_blocking.do`. They
  stay using user's 69-bot definitions. New `14_*` and `15_*` files use
  CF-Tier and post-event-UA respectively.
- **Don't** change Stata MCP encoding (do-files stay BOM-less; that's
  what Stata MCP needs).
- **Don't** put AI Assistant (CF-Tier 2 / Signed Agent bots) into Layer A
  main treatment. Cite reasons: signed-agent paradigm bypasses
  robots.txt; user-triggered nature blurs consent; adoption window short.
- **Don't** include legacy AI Crawlers (Applebot, PetalBot, Amazonbot,
  GoogleOther, Bingbot, Googlebot) in Layer B event-study battery — they
  fail criterion (2): pre-event existence violates structural-zero
  pre-period.
- **Don't** mix Layer A and Layer B regression outputs in the same paper
  table. They answer different questions (prevalence vs causal effect).

---

## 10. Decision points (open questions for user before execution)

1. **Apple two-UA accounting**: Apple has both `Applebot` (legacy,
   pre-LLM) and `Applebot-Extended` (post-2024-06, AI training opt-out).
   Cloudflare puts both in AI Search. Decision: include `Applebot-Extended`
   in Layer B (clean post-event UA, 2024-06 introduction); include
   `Applebot` in Layer A (CF taxonomy) but exclude from Layer B (pre-event,
   selection-biased). Confirm.
2. **Googlebot in Layer C placebo**: Googlebot is in CF Search Engine
   Crawler but feeds Gemini/AI Overviews. Run Layer C placebo with and
   without Googlebot; expect similar nulls.
3. **Allow vs Disallow framing**: depends on Step 7.1 diagnostic. If
   ratio ≥ 95%, drop `mention` framing entirely; use `disallow` only.
   Tentatively: keep both until diagnostic comes back.
4. **Appendix UAs (`cbot`, `meta`, `csearch`)**: introduction dates are
   approximate, not as well-documented as the main 4. Risk: if
   `specific_rule = 1` shows up before approximate intro date, the UA
   isn't actually post-event-defined. Diagnostic Step 7.1 covers this.
5. **CCBot**: Common Crawl's bot, in user's training29 but not in
   Cloudflare's AI Crawler directory. Pre-LLM era (since ~2010), heavily
   blocked since 2023. Keep in Layer A taxonomy with annotation; exclude
   from Layer B (pre-event).
6. **Bytespider**: not in Cloudflare directory because Cloudflare blocks
   it. In ai-robots-txt. Mark `respect_robots_txt = No`. Consider as
   Layer A robustness "non-compliant AI bot" subset.
7. **Pooled-4 timing**: with 4 UAs introduced over 11 months
   (2023-08 to 2024-07), the pooled-4 first-event for early-adopting
   firms could be GPTBot (2023-08), but for later-adopting firms could
   be OAI-SearchBot (2024-07). Common time-window for full
   identification is post-2024-07. If pooled-4 effect is significant,
   verify it's not driven by one UA (decompose by `cohort_mention_any4`
   value).

---

## 11. Where to start

Start with **Step 7.1** (diagnostic). Need:

1. Read the long-format `robots_panel_monthly.csv` (large file: ~30M
   rows). Filter to `bot ∈ {GPTBot, Google-Extended, Applebot-Extended,
   OAI-SearchBot, ClaudeBot, meta-externalagent, Claude-SearchBot}` and
   `specific_rule = 1`.
2. Group by `bot`, compute:
   - `min(year_month)` (first appearance) — verify against public dates
   - count of unique gvkeys with `mention = 1` (after gvkey-host join
     and `in_sample = 1` filter from main panel)
   - `disallow / mention` ratio
3. Write `post_event_ua_diagnostic.txt`. Show user.

Stop after Step 7.1 and present the diagnostic. Don't proceed to Step 7.2
without user acknowledgement that the 4 UAs pass criteria.

If the diagnostic surfaces issues (e.g., `Applebot-Extended` shows up in
some pre-2024 month, or `OAI-SearchBot` has only 30 in-sample firms),
the design needs adjustment before any code is written.

---

## 12. Phased rollout for paper

For paper writing, recommend this order:

**Phase 1 (sections 2–3, descriptive + determinants)**: Layer A only.
Cite Cloudflare directory, present taxonomy table, run determinants on
`cf_tier1_specific`, `cf_tier1_blocked`. These inherit the user's
existing `13_determinants_of_blocking.do` results almost verbatim, just
with one extra column for CF-Tier alignment.

**Phase 2 (section 4, identification)**: Layer B 4-UA event studies.
This is where the paper makes its causal claim. Lead with the
vendor-decomposition spec (4 separate event studies) showing
cross-vendor consistency, then pooled-4 for headline number, then BJS
average ATT for robustness.

**Phase 3 (section 5, mechanism / heterogeneity)**: Layer B 7-UA
appendix expansion + sub-decomposition of the main 4 (training vs
search; OpenAI vs Google vs Apple).

**Phase 4 (section 6, placebo / robustness)**: Layer C placebo + AI
Assistant robustness. Demonstrate:
- Layer C (Search Engine Crawler) placebo: null effect.
- AI Assistant robustness: weaker / null effect (cleaner controls
  for paper's main claim).

**Phase 5 (appendix)**: Strategy 1 failed pretrend, mention vs
disallow comparison, ai-robots-txt `respect` field heterogeneity
robustness.

This phased rollout naturally produces a paper that opens with broad
descriptive measurement, narrows to clean causal identification, then
fans out to robustness. It matches the advisor's "first analysis =
determinants, then consequences" steer.
