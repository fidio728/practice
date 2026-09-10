# This-week executable plan: prove Layer B identification first

> **Compressed scope** of the broader plan in
> `CONTEXT_CLOUDFLARE_TAXONOMY_2026_05_10.md`. Reference that file for the
> Layer A taxonomy work and § 12 phased rollout.
>
> **This week's question to answer**: do the 4 post-event UAs produce
> clean, interpretable DID effects on firm value, using **baseline
> estimators only** (2×2 + TWFE event study)? If yes → Phase 2 brings
> in modern staggered estimators (LP-DID, BJS, dCDH) and CF taxonomy.
> If no → stop pushing this design.

---

## TL;DR

- **Baseline DID first, modern staggered later.** This week is Layer 0 +
  Layer 1 + Layer 2 (all baseline / single-event-date). LP-DID staggered
  with firm-specific first-event timing is deferred to Phase 2.
- **5 files, 4 outputs, 2 gates**
- **Per-UA only**; pooled-4 deferred
- **One treatment definition** in main — pick `mention` or `disallow`
  based on diagnostic ratio
- **NO CF taxonomy work** this week
- Total Stata runtime estimate: ~3 min for the whole battery

---

## Why baseline first

Layer 1 (2×2 around launch date) and Layer 2 (TWFE event study around
launch date) **don't have the staggered-DID problem** — event time 0 is
a fixed calendar date for all firms, not firm-specific. So Goodman-Bacon's
"forbidden comparison" issue doesn't apply at this layer. The estimators
are simple, intuitive, and referee-friendly.

Modern staggered estimators (LP-DID with firm-specific first-event;
BJS imputation; dCDH) become necessary only when treatment timing varies
across firms in a way the analyst wants to exploit. That's Phase 2.

If Layer 1+2 already shows the effect cleanly, Layer 3 is confirmation.
If Layer 1+2 shows nothing, Layer 3 won't rescue it (it's stricter, not
more permissive).

---

## Three layers in this week's scope

### Layer 0 — Descriptive pre/post means

For each (UA, outcome) cell, a simple table of raw means:

| UA | event date | n_treated | n_control | pre 12m mean (T) | post 12m mean (T) | Δ_T | pre 12m mean (C) | post 12m mean (C) | Δ_C | Δ_T − Δ_C |
|---|---|---|---|---|---|---|---|---|---|---|

`treated` = firm ever wrote the UA in panel. `control` = firm never wrote
any of the 4 UAs. No regression — just summary stats.

### Layer 1 — Per-UA 2×2 DID at launch date

For each UA, a single regression with one DID coefficient. Sample
restricted to a symmetric window around the launch date (default 12m
each side):

```stata
* For GPTBot, around 2023-08-08:
gen byte post_gpt    = (ym >= ym(2023, 8))
gen byte ever_gpt    = !missing(cohort_<treat>_gpt)   // ever wrote it in panel
gen byte treated_post_gpt = post_gpt * ever_gpt

reghdfe Y treated_post_gpt `m2_lag' if ///
    (ym >= ym(2022, 8) & ym <= ym(2024, 7)) & in_sample == 1, ///
    absorb(gvkey_n ym) cluster(gvkey_n ym)
```

`treated_post_gpt` coefficient = standard 2×2 DID. The firm + time FE
absorb the main effects. ~16 specs total (4 UAs × 2 outcomes × 2 window
choices: 6m and 12m).

### Layer 2 — Per-UA TWFE event study at launch date

Dynamic version of Layer 1 with calendar-locked event time:

```stata
gen long et_gpt = ym - ym(2023, 8)    // event time relative to launch
forvalues k = -6/12 {
    gen byte d_gpt_`k' = (et_gpt == `k') * ever_gpt
}
* omit d_gpt_-1 as reference

reghdfe Y d_gpt_-6 d_gpt_-5 ... d_gpt_-2 d_gpt_0 d_gpt_1 ... d_gpt_12 `m2_lag' if ///
    (et_gpt >= -6 & et_gpt <= 12) & in_sample == 1, ///
    absorb(gvkey_n ym) cluster(gvkey_n ym)
```

Output: 19 coefficients per (UA, outcome) cell → 8 event-study plots
(4 UAs × 2 outcomes). Pre-trend test: joint nullity of β_-6..β_-2.

### Layer 3 — LP-DID staggered (Phase 2, NOT this week)

Firm-specific first-event timing. Goodman-Bacon weighting matters here.
Deferred. Reference plan: `15_layerB_lpdid_staggered.do` in Phase 2.

---

## Files this week

### IN

| File | Layer | Purpose |
|---|---|---|
| `bot_paper/panel_construction/diagnostic_post_event_uas.py` | — | Run pre-build sanity (Gate 1) |
| `bot_paper/panel_construction/build_post_event_ua_panel.py` | — | Build firm-month flags for 4 UAs |
| `bot_paper/panel_construction/post_event_ua_panel.csv` | — | Output: keyed (gvkey, year_month) |
| `bot_paper/stata/.../11d_merge_post_event_ua_panel.do` | — | Merge into main panel |
| `bot_paper/stata/.../15_layerB_layer0_descriptive.do` | 0 | Pre/post mean tables |
| `bot_paper/stata/.../15_layerB_layer1_2x2.do` | 1 | Per-UA 2×2 DID |
| `bot_paper/stata/.../15_layerB_layer2_eventstudy.do` | 2 | Per-UA TWFE event study (leads/lags) |

### Outputs (4)

| File | Used for |
|---|---|
| `post_event_ua_diagnostic.txt` | **Gate 1** — UA admission check + treatment definition decision |
| `15_layerB_layer0_descriptive.csv` | Paper Table 1 (descriptive) |
| `15_layerB_layer1_2x2_summary.csv` | **Gate 2** — single-coefficient DID per UA |
| `15_layerB_layer2_eventstudy_summary.csv` | **Gate 2** — event-study profile + pre-trend joint F |

### OUT (Phase 2, explicitly NOT this week)

- All CF taxonomy work (`cf_bot_taxonomy.csv`, `compare_taxonomy.py`,
  `build_cf_tier_panel.py`, `cf_tier_panel.csv`,
  `11c_merge_cf_tier_panel.do`, `14_layerA_measurement.do`)
- Layer 3 modern staggered (`15_layerB_lpdid_staggered.do`,
  `15_layerB_bjs.do`, dCDH)
- Pooled-4 spec (`15_layerB_pooled4_did.do`)
- 7-UA appendix expansion
- Layer C placebo (CF Search Engine Crawler)
- AI Assistant robustness
- Section LaTeX (`section_post_event_ua_design.tex`)

These all live behind Gate 2.

---

## Execution order with gates

### Step 1: Diagnostic — `diagnostic_post_event_uas.py`

#### Inputs (3 files, in load order)

| # | Path | Purpose | Columns used |
|---|---|---|---|
| 1 | `D:/anoconda/bot_paper_data/robots_panel_monthly.csv` | Primary long: per-(host, year_month, bot) directives | `host`, `year_month`, `bot`, `specific_rule`, `blocked_any` |
| 2 | `bot_paper/panel_construction/firm_panel_monthly.csv` | Host→gvkey bridge | `host`, `gvkey` (drop duplicates) |
| 3 | `bot_paper/panel_construction/gvkey_month_tier2_reg.dta` | In-sample mask (CRSP-active main regression sample) | `gvkey`, `year_month`, `in_sample` |

**Note on input #3 vs #2 staleness**: input #3 was last built before
the most recent firm_panel_monthly refresh. That's fine for this
diagnostic — we use input #3's `in_sample` flag as a sample mask, not
as the source of truth for which gvkeys exist. **Don't rebuild input #3
just to run the diagnostic** — sidecar approach is correct (Step 3
below).

#### Logic (6 steps)

1. Read input #1, filter to `bot ∈ {GPTBot, Google-Extended,
   Applebot-Extended, OAI-SearchBot}` AND `specific_rule == 1`.
2. **Compute `first_appearance_any_host_month` per UA** = `min(year_month)`
   over the full filtered long file, **NO in_sample filter applied**.
   This is the structural pre-period check (Layer B admission criterion 2):
   if a UA shows up before its public introduction date in any
   host-month, the structural-zero claim fails — design contamination.
3. Read input #2, drop duplicates on `(host, gvkey)`. Inner-join to
   the filtered long file on `host`.
4. Aggregate to `(gvkey, year_month, bot)` with `max(specific_rule)` —
   the firm-month UA-mention indicator.
5. Read input #3, keep only `(gvkey, year_month, in_sample)`. Left-join
   to step-4 output on `(gvkey, year_month)`.
6. Compute in-sample-restricted statistics where `in_sample == 1`.

**Why dual-perspective**: criterion 2 (structural pre-period clean) is a
question about the **whole data universe** — any contamination anywhere
breaks the design. Power / feasibility (criterion 2's ≥ 50 + criterion 3's
ratio decision) are questions about the **regression sample** — only
`in_sample == 1` firms enter the DID.

#### Output: `post_event_ua_diagnostic.txt`

Per UA, 6 fields with both global and in-sample views:

```
=== GPTBot (public intro 2023-08-08) ===
  [Structural pre-period check — global, no in_sample filter]
  first_appearance_any_host_month:    2023-08
  vs public intro date:               ✓ same month (no contamination)

  [In-sample power check — restricted to in_sample == 1]
  first_appearance_in_sample:         2023-09
  n_gvkeys_ever_mention_in_sample:    421
  n_gvkeys_ever_disallow_in_sample:   409
  disallow_share_among_mentions:      97.1%

  [Admission criteria]
  pass C1 (date match within ±1 mo):  ✓
  pass C2 (≥ 50 in-sample treated):   ✓
  C3 ratio decision:                  use disallow only (≥ 95% overlap)
```

(Repeat for Google-Extended, Applebot-Extended, OAI-SearchBot.)

Plus a final decision block:

```
=== Overall decision matrix ===
  All 4 UAs pass C1 + C2:               [yes/no]
  C3 mention-vs-disallow consistency:    [≥ 95% across all 4 / mixed / different per UA]
  Recommended primary treatment:         [disallow only / mention only / both]
  Any UA with date contamination?        [list / none]
```

### Gate 1: User reviews diagnostic

Approve before Step 2. If any UA fails:

- **C1 (date contamination)**: `first_appearance_any_host_month` >
  ±1 month before public date → investigate. Could be a robots.txt
  parser bug or a UA-aliasing issue. Drop UA from main if confirmed
  contamination.
- **C2 (in-sample treated < 50)**: underpowered. Drop UA from main, or
  demote to appendix.
- **C3 (mention/disallow ratio decision)**:
  - ≥ 95% across all 4 UAs → use `disallow` only as primary; drop
    `mention` from main spec entirely
  - 70–90% → real distinction, keep both ("governance adoption"
    framing survives)
  - mixed (some UAs ≥ 95%, others < 90%) → use per-UA primary;
    document in paper

### Step 2: Build firm-month panel — `build_post_event_ua_panel.py`

For 4 UAs only. Per (gvkey, ym, ua):
- `mention_<stub>`, `disallow_<stub>` (binary, max over hosts)
- `cohort_mention_<stub>`, `cohort_disallow_<stub>` (year_month string of
  first event, or empty string if never)

Stubs: `gpt`, `gext`, `aext`, `oai`. ~16 columns total. **No `_any4`
pooled, no `first_*` indicators** (those are for Layer 3, deferred).

Output `post_event_ua_panel.csv` keyed (gvkey, year_month).

### Step 3: Stata merge — `11d_merge_post_event_ua_panel.do`

Mirror `11b_merge_tier2_panel.do`. Output sidecar
`gvkey_month_tier2_reg_layerB.dta`.

### Step 4a: Layer 0 descriptive — `15_layerB_layer0_descriptive.do`

For each (UA, outcome) ∈ {4 × 2 = 8 cells}:

```stata
foreach ua in gpt gext aext oai {
    * Compute pre 12m mean and post 12m mean for treated and control
    foreach y in ln_amihud ln_mktcap {
        sum `y' if ever_`ua' == 1 & ym >= launch_`ua' - 12 & ym < launch_`ua' & in_sample
        local pre_T = r(mean)
        sum `y' if ever_`ua' == 1 & ym >= launch_`ua' & ym < launch_`ua' + 12 & in_sample
        local post_T = r(mean)
        * same for control (ever_any4 == 0)
        * write row to csv
    }
}
```

Output: `15_layerB_layer0_descriptive.csv` with 8 rows × {n_T, n_C,
pre_T, post_T, Δ_T, pre_C, post_C, Δ_C, DiD = Δ_T − Δ_C}.

### Step 4b: Layer 1 per-UA 2×2 — `15_layerB_layer1_2x2.do`

```stata
local outcomes ln_amihud ln_mktcap
local windows  6 12

foreach ua in gpt gext aext oai {
    foreach y of local outcomes {
        foreach w of local windows {
            * Define treated_post_<ua>, post_<ua>, ever_<ua> for this UA
            reghdfe `y' treated_post_`ua' `m2_lag' if window_`w' == 1, ///
                absorb(gvkey_n ym) cluster(gvkey_n ym)
            * Write β, SE, p, N, n_treated to summary csv
        }
    }
}
```

4 UAs × 2 outcomes × 2 windows = **16 specs**. Output:
`15_layerB_layer1_2x2_summary.csv` with one row per cell.

### Step 4c: Layer 2 event study — `15_layerB_layer2_eventstudy.do`

```stata
foreach ua in gpt gext aext oai {
    foreach y in ln_amihud ln_mktcap {
        * Build leads/lags d_<ua>_-6 to d_<ua>_+12, omitting -1
        reghdfe `y' d_`ua'_neg6 ... d_`ua'_pos12 `m2_lag' if window, ///
            absorb(gvkey_n ym) cluster(gvkey_n ym)
        * Write each β_h to summary csv
        * Joint pre-trend test: F on (β_-6, β_-5, ..., β_-2) = 0
    }
}
```

4 UAs × 2 outcomes × 19 horizons (-6..+12) = **152 coefficient rows** +
8 pretrend rows. Output:
`15_layerB_layer2_eventstudy_summary.csv`.

### Gate 2: Stop / go decision

After Layer 0+1+2 complete, review:

| Layer 0 (raw Δ_T − Δ_C) | Layer 1 (2×2 β) | Layer 2 (event study profile) | Action |
|---|---|---|---|
| ≥ 2 of 4 UAs show consistent direction on either outcome | ≥ 2 of 4 UAs significant at 10% | ≥ 2 of 4 UAs have clean β_-6..β_-2 ≈ 0 + post-event divergence | **GO Phase 2** |
| 1 of 4 (likely Applebot-Extended) | 1 of 4 | 1 of 4 | **CAUTION**: investigate why others null |
| 0 of 4 | 0 of 4 | 0 of 4 | **STOP**: retreat to Layer A determinants only |

The three layers should all point the same direction. If Layer 0 shows
big Δ but Layer 1 shows null, that's a controls / sample selection issue
(investigate). If Layer 1 shows big β but Layer 2 has bad pre-trend,
identification is shaky (red flag — even calendar-locked event date
isn't immune if treated/control are systematically diverging pre-event).

---

## What this week DOES and DOES NOT tell us

### DOES tell us
- Whether ever-treated firms diverge from never-treated firms in raw
  outcomes around each UA's launch date (Layer 0)
- Whether that divergence survives firm + time FE controls in a
  baseline 2×2 (Layer 1)
- Whether the divergence has a clean pre-event flat profile and
  post-event dynamic (Layer 2)
- Whether 4 different UAs (different vendors, different launch dates,
  different intent) produce consistent signal

### DOES NOT tell us (Phase 2 scope)
- Whether the effect survives **modern staggered estimators with
  firm-specific first-event timing** (Layer 3 / LP-DID, BJS, dCDH)
- Whether the broader CF AI bot taxonomy effect is real (Layer A
  determinants on `cf_tier1_specific`)
- Whether Search Engine Crawler placebo is null (Layer C)
- Pooled cross-vendor average effect (pooled-4 spec)
- Mechanism heterogeneity (training vs search, vendor decomposition,
  sector heterogeneity)
- AI Assistant exposure effects

---

## Estimated time

| Step | Tool | Time |
|---|---|---|
| 1. Diagnostic | Python | ~5 min (one-shot scan of 30M rows) |
| 2. Build panel | Python | ~5 min |
| 3. Stata merge | Stata MCP | ~30 sec |
| 4a. Layer 0 descriptive | Stata MCP | ~1 min |
| 4b. Layer 1 2×2 (16 specs) | Stata MCP | ~1 min |
| 4c. Layer 2 event study (8 specs × 19 coefs) | Stata MCP | ~2 min |
| **Total** | | **~15 min** of compute, plus 2 user-gate reviews |

Realistic calendar: half-day with 2 user-in-the-loop checkpoints.

---

## Decisions baked in

1. **Baseline DID first, modern staggered second.** This week's Layer
   0+1+2 are all calendar-locked (event time 0 = fixed launch date).
   Goodman-Bacon staggered-DID concerns don't apply at this layer.
   LP-DID with firm-specific first-event timing is Phase 2.
2. **CSV-driven taxonomy when we get there**: Phase 2 uses
   `cf_bot_taxonomy.csv` + Python loader, not hand-coded `.py` dict.
3. **`mention` vs `disallow` is a single-line decision**: pick one based
   on Step 1 ratio. Don't run both in main.
4. **Per-UA only in main**: pooled-4 deferred. Tables/figures show 4
   separate UAs in main.
5. **Tier C placebo deferred**: validates a Phase-2 finding, not a
   Phase-1 gate.

---

## Where to start

Run **Step 1 only**. Show user the diagnostic. Wait for approval before
running Step 2.

Files to read first:
- `bot_paper/panel_construction/04_parse_panel.py` lines 48-100
  (BOT_TAXONOMY for canonical UA names + aliases)
- `D:/anoconda/bot_paper_data/robots_panel_monthly.csv` schema (host,
  year_month, bot, specific_rule, blocked_any)
- `bot_paper/panel_construction/firm_panel_monthly.csv` for gvkey-host
  map

Reference plan for Phase-2 context:
- `git/practice/CONTEXT_CLOUDFLARE_TAXONOMY_2026_05_10.md` — full
  Phase 1+2 vision (CF taxonomy, Layer A, LP-DID staggered, etc.)

---

## Outside this week's scope but worth flagging

- **`Allow:` semantic**: per § 3 caveat in the larger md, `Allow:` lines
  in robots.txt are mostly conditional exceptions inside a Disallow
  block. Step 1 ratio diagnostic settles whether `mention` ≈ `disallow`.
- **File encoding**: Chinese characters in the larger md may render as
  mojibake in some viewers — file is UTF-8 on disk (verified at write
  time); resave with explicit UTF-8-without-BOM if a viewer mis-renders.
  Not blocking execution.
- **Layer 3 (modern staggered) timing**: the natural next step after
  Gate 2 GO is `15_layerB_lpdid_staggered.do` mirroring
  `12_apple_petal_cohorts_lpdid.do` structure but for 4 UAs and
  first-event timing. Estimated ~5 min Stata for 4 UAs × 2 outcomes ×
  9 horizons = 72 cells. Then BJS imputation in
  `15_layerB_bjs.do` (~10 min Stata).
