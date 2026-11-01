# Bot Paper Next-Window Context

Last updated: 2026-05-06

Purpose: use this file as handoff context for a new Codex window. In the new window, say: "Read `BOT_PAPER_NEXT_WINDOW_CONTEXT.md` and start from the Tier 3 specificity ladder."

## Project Paths

Main project:

```text
C:\Users\xl\OneDrive - Universitat Ramon Llull\bot_paper
```

Note: the real folder name contains the accented character in "Ramon/Ramon". If a command fails because of encoding, use PowerShell tab completion or the exact path shown by `Get-ChildItem`.

Important subfolders:

```text
bot_paper\panel_construction
bot_paper\stata\stata-mcp-folder\stata-mcp-dofile
bot_paper\stata\stata-mcp-folder\stata-mcp-log
bot_paper\stata\stata-mcp-folder\stata-mcp-result
bot_paper\draft_2026_05_06
E:\bot_paper_data
```

Current draft folder:

```text
C:\Users\xl\OneDrive - Universitat Ramon Llull\bot_paper\draft_2026_05_06
```

## Stata MCP / Codex Setup

Codex config was updated:

```text
C:\Users\xl\.codex\config.toml
```

Added MCP server:

```toml
[mcp_servers.stata-mcp]
command = "D:\\anoconda\\Scripts\\stata-mcp.exe"
args = []

[mcp_servers.stata-mcp.env]
STATA_MCP__CWD = "C:\\Users\\xl\\OneDrive - Universitat Ramon Llull\\bot_paper\\stata"
STATA_MCP__LOGGING_ON = "false"
```

May require restarting Codex CLI/new session before `stata_do` tools appear.

## Core Panel/Data Files

Main firm-month panel:

```text
panel_construction\gvkey_month_tier2_reg.dta
```

Category-long panel:

```text
panel_construction\gvkey_month_category_reg_long.dta
```

Tier 1 panel:

```text
panel_construction\gvkey_month_tier1_reg.dta
```

Important E-drive files from last data build:

```text
E:\bot_paper_data\digest_bot_policy.csv
E:\bot_paper_data\digest_bot_policy_summary.txt
E:\bot_paper_data\daily_first_block_events.csv
E:\bot_paper_data\daily_first_block_events_summary.txt
E:\bot_paper_data\daily_first_block_events_dedup_summary.txt
E:\bot_paper_data\gvkey_first_block_dates.csv
E:\bot_paper_data\gvkey_adoption_curve_monthly.csv
E:\bot_paper_data\gvkey_adoption_curve_monthly_strict.csv
E:\bot_paper_data\adoption_strict_timing_table.csv
E:\bot_paper_data\step7_v4_buckets
```

Useful construction scripts:

```text
panel_construction\06_5_collect_sitemap_paths_monthly.py
panel_construction\06_collect_sitemaps_for_panel.py
panel_construction\06b_collect_sitemap_url_paths.py
panel_construction\07_build_search_sitemap_block_panel.py
panel_construction\07_build_search_sitemap_block_panel_v3.py
panel_construction\07_build_search_sitemap_block_panel_v4.py
panel_construction\08_build_bot_change_events.py
panel_construction\09_merge_outcomes.py
```

## Main Treatment / Controls

Headline treatment:

```stata
L.search_sitemap_block
```

Meaning: lagged firm-month search-bot sitemap blocking state.

Current main controls after dropping `log_hosts`:

```stata
log_at_q leverage_q roa_ttm_q mb_q tangibility_q log_urls
```

Reason `log_hosts` dropped: 99%+ firm-months are single-host; max host count is 2, so it has negligible variation and adds interpretation burden.

Always be precise in writing:

```text
firm FE and calendar-month FE
```

Do not write "firm-month FE"; that would absorb treatment.

## Key Empirical Results Already Done

### Main TWFE State Regression

Main equation:

```stata
reghdfe Y L.search_sitemap_block controls, absorb(gvkey_n ym) cluster(gvkey_n ym)
```

Headline M2/no-host result approximately:

```text
ln_amihud: +0.132** (t about 2.4)
ln_mktcap: -0.076** (t about -2.1)
qspread/turnover weaker
```

Interpretation should be conditional reduced-form / blocked-state association, not strong causal claim.

### Control-Set Diagnostics

Done in:

```text
12_diagnostic_controls.do
```

Result: adding website complexity controls barely moved coefficients. With new control set, update paper text from "M2 = M1 + log_urls + log_hosts" to "M2 = M1 + log_urls". Mention `log_hosts` omitted because almost all firms are single-host.

### Intensity / Extensive Margin

Done in:

```text
12_intensity_specs.do
```

Tables:

```text
table_i1_full.tex
table_i2_full.tex
table_i3_full.tex
```

Findings:

- Continuous blocked share is null across outcomes.
- Two-part decomposition shows the extensive margin (`any blocking`) recovers headline.
- Intensive margin within blocked state is null.
- Partial vs full: partial blockers drive headline; full blockers same-signed but imprecise.

Writing recommendation: keep continuous + two-part in main text; move partial-vs-full to appendix or one short paragraph. It is useful only to say result is not solely site-wide `Disallow: /`.

### Horse Race: Search vs Control Bots

Done in:

```text
12_horse_race.do
```

Table:

```text
table_hr_full.tex
```

Key result:

```text
ln_amihud H3 joint:
search beta = +0.290** (t=2.27)
control beta = -0.191 (ns)
p(search = control) = 0.063

ln_mktcap H3 weaker:
p(search = control) = 0.281
```

Discordant cells in Amihud regression sample:

```text
search=1, control=0: 469 firm-months
search=0, control=1: 672 firm-months
total discordant: 1,141 firm-months from 85 firms
```

Strict-firmwide H4:

```text
ln_amihud strict search beta = +0.251, t=1.28
same sign, lower power
```

Writing: directionally consistent specificity evidence with marginal Wald rejection; not definitive causal isolation.

### Category Mechanism

Done in:

```text
12_category_mechanism.do
```

Finding: IR is not the unique driver. Significant per-category Amihud effects appear for HR, IR, PMC, and Other. Story should be broad AI discoverability, not IR-specific investor-relations channel.

Within-Boulland descriptive:

```text
PMC and IR dominate classified blocked paths.
IR is under-blocked relative to its sitemap share.
Small narrative categories (CSR/diversity/HR) have high lift but tiny mass.
```

Wide 8-way category regression is unstable because categories are non-exclusive; appendix only.

`other` is author-constructed residual, not a Boulland economic category. It should be decomposed or at least manually sampled.

### Robust Design / Controls

Done in:

```text
12_robust_controls.do
12_robust_design.do
```

Results:

```text
D4 n_switches <= 2: Amihud stays positive/significant.
D5 sic2 x ym FE: Amihud attenuates about 16-18% but remains significant.
D6 alternative cluster(gvkey, sic2): similar.
Expanded controls with cash/R&D/div attenuate but do not kill Amihud.
Bad controls (price/lag vol/lag return) collapse estimates, as expected.
```

### Switching Endogeneity Diagnostic

Done in:

```text
12_endo_diagnostics.do
12_endo_E1_summary.csv
```

Diagnostic: regress switching hazard on lags of recent outcome changes.

Primary DV: `switch_in` entry hazard, risk set restricted to `L.D == 0`.

Key joint p-values:

```text
switch_in:
Delta ln(Amihud): 0.958
Delta ln(qspread): 0.779
Delta ln(turnover): 0.435
Delta ln(dspread): 0.656
Delta ln(mktcap): 0.062
crsp_ret_cumul: 0.095
crsp_ret_vol: 0.024
```

Interpretation:

- Amihud headline is relatively clean: past Amihud changes do not predict entry.
- mktcap has timing concern: past market-cap changes weakly predict switching.
- volatility predicts entry, but volatility outcome is null in main specs; do not interpret volatility as affected outcome.
- Exit timing is not predicted by recent observable outcomes.

### TWFE Weight Diagnostic

Done in:

```text
12_twfe_weights.do
12_twfe_weights_summary.csv
12_twfe_weights_ln_amihud_cells.dta
12_twfe_weights_ln_mktcap_cells.dta
```

Findings:

```text
Cell-level ATTs: 4,201
positive-weight cells: 3,445 (82.0%)
negative-weight cells: 756 (18.0%)
sum positive weights: +1.0074
sum negative weights: -0.0074
most negative weight: -5.76e-05
```

Sigma robustness:

```text
ln_amihud beta_fe = +0.1322
sigma for true-zero average ATT = 0.1144
sigma for all-ATT opposite sign = 5.2225

ln_mktcap beta_fe = -0.0763
sigma for true-zero average ATT = 0.0660
sigma for all-ATT opposite sign = 3.0125
```

Interpretation:

- Negative-weight bias is unlikely to be the primary driver of TWFE.
- dCDH/TWFE discrepancy is estimand difference, not a negative-weight artifact.
- This diagnostic does not solve switching endogeneity.

### dCDH / First-Switch Dynamic DID

Do-files:

```text
12_dCDH.do
12_dCDH_v4.do
12_parallel_trends.do
```

Main conceptual point:

- dCDH is a first-switch dynamic estimator, not validation of TWFE state estimand.
- It tracks outcomes around first observed treatment change.
- Subsequent reversals enter post-first-switch path but do not create new event-zero observations.

Important results:

- `ln_amihud` V1 no-control switchers-in was negative around `-0.170`, marginal, opposite sign from TWFE.
- With controls, effect shrinks toward zero.
- mktcap dCDH results are mostly insignificant.
- Joint placebo p-values from parallel-trends figures:

```text
switchers(in):
ln_amihud 0.108
ln_mktcap 0.918
ln_qspread 0.489
ln_turnover 0.126

switchers(out):
ln_amihud 0.720
ln_mktcap 0.158
ln_qspread 0.851
ln_turnover 0.857
```

Writing:

- No joint evidence against pretrends, but Amihud entry has a marginal one-period placebo; interpret cautiously.
- Do not say dCDH confirms TWFE.

## Current Writing Framing

Best current headline:

```text
AI-search blocking is associated with higher Amihud illiquidity in a conditional reduced-form blocked-state design.
The effect is mostly extensive-margin: any blocking matters, not blocked share.
Horse race and controls support AI-search specificity for Amihud, but evidence is marginal and relies on discordant cells.
Mktcap is directionally consistent but more vulnerable to switching-endogeneity.
```

Do not claim:

```text
AI blocking causally raises illiquidity by 13%.
IR pages are the main mechanism.
dCDH verifies TWFE.
```

## Next Tasks To Start

### Task 1: Tier 3 Specificity Ladder

Goal: prove current Tier 2 result is about economically meaningful AI-mediated discoverability, not one stray blocked sitemap URL or generic crawler governance.

Construct a ladder:

```text
Tier2A: current broad sitemap conflict
Tier2B: core AI-search bots only
        OAI-SearchBot, PerplexityBot, Claude-SearchBot
Tier2C: core AI-search blocked AND Googlebot/Bingbot allowed on the same blocked paths
Tier2D: same as C, but only material/content-sensitive paths
        candidates: IR, press/news, annual/earnings/presentations,
        SEC/filings, governance, investor presentations
Tier2E: same as D, but rule source is AI-specific, not inherited from User-agent: *
```

Needed data:

```text
E:\bot_paper_data\digest_bot_policy.csv
panel_construction\search_sitemap_block_panel.csv
panel_construction\tier2_path_category_metrics.csv
panel_construction\gvkey_month_tier2_reg.dta
panel_construction\gvkey_month_category_reg_long.dta
```

Need to inspect whether same-path Googlebot/Bingbot allowed is already in data:

```text
search_sitemap_block_panel.csv likely has per-bot n_blocked / frac / any fields.
digest_bot_policy.csv has per-bot root_block, has_nonroot_disallow, specific_rule, effective_source.
```

Output desired:

```text
12_tier3_ladder.do
12_tier3_ladder_summary.csv
table_tier3_ladder.tex
```

Run same headline outcomes:

```text
ln_qspread ln_amihud ln_turnover ln_mktcap ln_dspread
```

Main interpretation:

- If Amihud same-sign/stronger as ladder tightens, specificity improves.
- If strict tiers underpowered but same direction, report as lower-powered construct-validity check.

### Task 2: Treatment Anatomy Table

Goal: describe what the broad Tier 2 treatment actually consists of.

Variables to build/check:

```text
single_url_block
low_intensity_block: blocked share < 1%
sitewide_block: root Disallow /
granular_block: nonroot/path-level disallow
ai_specific_rule
wildcard_inherited_rule
same_path_google_allowed
same_path_bing_allowed
same_path_all_controls_allowed
persistent_blocker
transient_blocker
entry
exit
spell_age
```

Most important:

```text
ai_specific_rule vs wildcard_inherited_rule
same-path AI blocked while Google/Bing allowed
sitewide vs granular
```

Output desired:

```text
12_treatment_anatomy.do
12_treatment_anatomy_summary.csv
table_treatment_anatomy.tex
```

This should be descriptive first. It may also feed the Tier 3 ladder.

### Task 3: Stacked Clean-Spell DID

Goal: better identification around clean entry/exit spells without pretending treatment is absorbing.

Basic design:

```text
Define entry spells: 0 -> 1
Define exit spells: 1 -> 0
Require no other switch within event window, e.g. [-3,+3] or [-6,+6]
Use matched controls in same SIC2, size bins, web complexity bins, and pre-trend bins.
Stack event windows.
Estimate treated x post or dynamic event-time coefficients with stack FE.
```

Important:

- Do not rely only on first switch.
- Because treatment is reversible, use clean spells and no nearby reversals.
- This is a robustness/alternative design, not replacement unless it works very well.

Candidate do-file:

```text
12_stacked_clean_spells.do
```

Possible controls/matching:

```text
same sic2
log_at_q bins
log_urls bins
pre-Amihud trend bins
never-treated or not-yet-treated controls within event window
```

Key outputs:

```text
event-study coefficients [-3,+3]
joint pretrend test
0-to-3 average post effect
separate entry and exit panels
```

### Task 4: Sitemap Freshness Robustness

Goal: address staleness of Wayback sitemap snapshots.

Potential variables:

```text
sitemap snapshot captured within same calendar month
days between sitemap snapshot and month-end
whether sitemap has lastmod tags
median/max lastmod age if tags exist
timing_precision / interval_days_search from gvkey_first_block_dates.csv
```

Use as:

```text
restrict to fresh snapshots
interact D with fresh_sitemap
compare stale vs fresh subsamples
```

Output desired:

```text
12_sitemap_freshness.do
12_sitemap_freshness_summary.csv
table_sitemap_freshness.tex
```

Warning:

- `lastmod` may be sparse/unreliable.
- More robust first check is snapshot freshness / timing precision rather than `lastmod`.

## Practical Start Command For New Window

In a new Codex window:

```text
Read BOT_PAPER_NEXT_WINDOW_CONTEXT.md. Start with Task 1: Tier 3 specificity ladder. First inspect the available per-bot/path/rule-source variables in E:\bot_paper_data and panel_construction. Do not edit existing pipeline until you know which variables already exist.
```

## Cautions

- Do not replace main treatment with IR-only; IR is not the unique driver.
- Do not call bot-launch immediate adopters a clean causal design; adopter selection remains endogenous.
- Do not overclaim dCDH; it estimates first-switch dynamics, not TWFE state effect.
- Do not call `other` a Boulland category; it is an author residual.
- Do not say "firm-month FE"; use "firm FE and calendar-month FE".
- Keep causal language conservative: conditional reduced-form, specificity evidence, construct-validity checks.
