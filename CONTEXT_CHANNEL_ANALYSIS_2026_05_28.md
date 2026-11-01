# Context handoff — Channel analysis (robots.txt → filing-window illiquidity)

**Written:** 2026-05-28. For a fresh window doing **channel / mechanism analysis**.
This doc is self-contained: a new session with no memory of the prior
conversation can start from here.

---

## 1. The paper in one paragraph

Corporate-finance study. When firms restrict **search / discovery crawler
access** via `robots.txt`, their **filing-window stock illiquidity (Amihud
price-impact measure) rises**. The story is now framed around **search-crawler
restrictions**, NOT "AI". The headline result: abnormal Amihud over the
`[0,+5]` trading-day filing window is **+0.037, p = 0.005** under the main
treatment, with firm + filing-month fixed effects and firm-clustered SEs.
The locked mechanism interpretation is a **price-impact / depth / Kyle-λ
channel**, established by (a) a positive Amihud effect plus strong fragility
heterogeneity, and (b) a battery of adjacent channels that all come back
**null** (so the effect is not analyst-diffusion, not spread-widening, not a
trading-volume artifact, not EDGAR substitution).

**The channel work's job now:** move from "we ruled out the other channels"
to (ideally) **positively characterizing the price-impact/depth channel** —
e.g. decomposing the Amihud rise into its numerator (|return| / volatility)
and denominator (dollar volume / depth) components.

---

## 2. Treatment definitions (current, search-framed — do NOT say "AI")

Treatment is built directly from archived `/robots.txt` rules (Wayback +
Common Crawl). A firm "becomes treated" the first month a directive of the
specified type appears. **X-strict cohort**: a firm enters the sample iff it
is either *ever-treated* (first block month observed, non-censored) or
*clean-never* (never blocks, same archival coverage). Excluded:
left-censored, future-treated, status-only, status-before-timestamp. The
post indicator is **strictly after** the first block month (the block month
itself is dropped from filing-window regressions).

| Treatment | What it is | Role |
|---|---|---|
| **broad23 / `broad_discovery`** | 23-bot search-related family (search engines + content-discovery + aggregators) | **MAIN / headline** |
| **search16** | 16-bot search-only subset of the 23 — a strict subset of the treated cohort (only 98 of 316 broad-treated firms ever block a search-only bot) | Stricter robustness; same sign/magnitude, underpowered alone |
| `robots_any` | any `Disallow` rule on any tracked bot | broadest; conflates targeted with generic crawler hygiene |
| `robots_wild` | wildcard `User-agent: *` Disallow | broad site policy, not targeted |
| `robots_root` | sitewide root `Disallow: /` | maximally restrictive |
| `robots_sb_spec` | bot-specific Disallow naming a pure search bot | narrowest directive-by-bot |
| `robots_spec` | bot-specific Disallow on the broader 16-bot list | legacy bot-specific |

**Treatment timing source:** `firm_discovery_treatment.csv`
(`first_discovery_block_month` "YYYY-MM" is the broad23 canonical timing;
`canon_broad_treated_strict`, `clean_never_broad`, `status_only_broad`,
`status_before_timestamp_broad`, `future_treated_broad` are the cohort
flags). The robots directive-type cohorts live in
`firm_robots_treatment.csv` (153 cols; prefixes `robots_any_*`,
`robots_spec_*`, `robots_wild_*`, `robots_root_*`, `robots_sb_spec_*`, etc.,
each with `_first_block / _canon_treated / _clean_never / _left_censored /
_future_treated`).

---

## 3. Locked mechanism: price-impact / depth / Kyle-λ

### Positive evidence (the effect that needs a channel)
- **Amihud rises** in the filing window: `[0,+5]` +0.037 (p=0.005),
  `[0,+1]` +0.034, `[-5,+5]` +0.034; delayed `[+6,+30]` is a clean
  **placebo null** (timing localizes to the disclosure window).
- **Fragility heterogeneity is the strongest pattern:** the effect
  concentrates in ex-ante fragile firms — low return-volatility firms
  (≈ +3×) and high-pre-filing-Amihud firms (≈ +20×; interaction
  +0.85, p < 1e-6). This is what a depth/price-impact story predicts:
  firms already thin on liquidity get hit hardest.

### Channels RULED OUT (all null / wrong-sign under broad23)
| Channel | Verdict | Evidence file |
|---|---|---|
| **EDGAR substitution** | NEGATIVE — investors spend *less* time on treated firms' filings, not more; no compensating-disclosure channel | `run_edgar_10kq8k.do` |
| **Analyst intermediation (IBES)** | null — Detail + Summary + channel-map; two BJS-only marginal positives at immediate window, not robust | `run_filing_window_non_edgar_channel_map_broad23.do` (Panel A) |
| **Posted spread** | null — CRSP quoted spread + 4 daily proxies (Corwin-Schultz, Abdi-Ranaldo CHL, Roll, HL range), 5,099 firms | `run_filing_window_non_edgar_channel_map_broad23.do` (Panel C) + `run_spread_proxy_robustness.do` |
| **Trading volume** | null — ln(share volume) all windows β ∈ [−0.07,−0.01], p ≥ 0.32 | `run_filing_window_non_edgar_channel_map_broad23.do` (Panel B) |
| **PEAD / analyst-information-diffusion** | null — HLT 2009, DVP 2009, Martineau 2022 all null on the surprise × treatment interaction (see §5) | PEAD do-files (§5) |

**Interpretation locked:** POSITIVE Amihud + fragility, with NEGATIVE EDGAR
substitution + IBES + spread + volume + PEAD ⇒ **price-impact / depth /
Kyle-λ**. Do NOT write "slower information diffusion" (that channel is the
null). Do NOT write "AI-discovery-specific" (search framing, and the anatomy
differential is only directionally suggestive, p≈0.20).

---

## 4. The natural next channel analysis: **Amihud numerator/denominator decomposition**

Amihud illiquidity = `|return| / dollar_volume`. A rise can come from:
- **numerator up** (larger absolute returns / higher volatility per unit
  trade → price-impact / adverse-selection story), or
- **denominator down** (lower dollar volume / thinner depth → depth story).

**Both components are ALREADY in the filing event panel**
(`edgar_filing_event_panel.csv`, 437 cols, 94,491 rows). Confirmed columns
across the standard windows (`0_1, 0_3, 0_5, m5_5, 6_30, 0_30`, plus
pre-filing benchmark windows `m30_m16, m15_m1`):
- **Amihud:** `amihud_mean_<win>` (42 cols)
- **Numerator side:** `abs_ret_sum_<win>`, `abs_ret_max_<win>`,
  `cum_ret_<win>`, `cum_logret_<win>` (81 ret cols total)
- **Denominator side:** `vol_sum_<win>` (51 vol cols)
- **Spread:** `qspread_mean_<win>` + `n_qspread_days_<win>` (55 cols)
- **EDGAR access:** `abn_edgar_{all,periodic,10k,10q,8k}_<win>` (40 cols)

The channel-map already ran **abnormal** versions: `abn_abs_ret_sum_<win>`
and `ln_vol_sum_<win>` (Panel A/B/C of `non_edgar_channel_map_results.csv`).
So the decomposition is buildable immediately:
- Run the same TWFE filing-window DID on `abn_abs_ret_sum_<win>` (numerator)
  and `ln_vol_sum_<win>` (denominator) under broad23.
- Volume came back **null** already — so if Amihud rises and volume doesn't
  fall, the rise must be **numerator-driven** (price-impact / volatility per
  trade), which is the cleanest positive support for the Kyle-λ reading.
- Cross-check with `abs_ret_max` (jump risk) and the fragility split.

**Data gap to be aware of:** the panel has **NO** order-imbalance, signed
volume, intraday depth, turnover, or a directly-estimated Kyle-λ column. A
positive Kyle-λ confirmation beyond the Amihud decomposition would need a new
data source (TAQ is unsubscribed — already established; see spread-proxy
memory). So the realistic positive-channel story is the **numerator vs
denominator decomposition** of the Amihud measure itself, not a fresh λ
estimate.

---

## 5. PEAD / drift channel (this session's work — NULL, locked)

Tested whether the treatments that raise filing-window Amihud also amplify
post-earnings-announcement drift (the slower-diffusion channel). **All null.**

- **DVP-faithful** (`run_pead_dvp_robustness.do` →
  `pead_dvp_robustness_results.csv`): 14 specs, BHAR_2_61 γ negative & ns in
  both surprise forms.
- **Martineau baseline power-check** (`run_pead_martineau_power_check.do` →
  `pead_martineau_power_check_results.csv`): baseline PEAD survives ONLY in
  microcap × random-walk × short windows (Martineau 2022 "rest in peace").
  All-but-microcap drift dead.
- **D2 treatment-sensitivity** (`run_pead_martineau_d2_treatment_sensitivity.do`
  → `pead_martineau_d2_treatment_sensitivity_results.csv`): 4 treatments × 5
  baseline-alive cells × 2 VCE = 40 specs, **zero reach p<0.05**, including
  broad23. One marginal cell (robots_any × SUE × micro × [2,5], p=0.059) is
  economically tiny.
- **HLT/DVP 4-treatment sensitivity:** a unified runner
  `run_pead_three_protocol_treatment_sensitivity.do` (in
  `git/practice/`) is written by the other AI but **NOT yet run** — it runs
  HLT/DVP/Martineau each in its own native window grid + surprise forms.
  Deferred; not needed for channel analysis unless drift comes back up.

PEAD framing rule (locked): frame as **NULL on amplification**, never as
"robots reduces PEAD" (the wrong-sign cells are opposite-direction noise,
not a finding).

---

## 6. Conventions (carry into channel work)

- **Clustering:** new specs use two-way `cluster(gvkey_n ym)`; when the
  two-way VCV is singular (our panel has only 19 year-quarters), fall back
  to **firm-only** `vce(cluster gvkey_id)` and flag the cell, never
  synthesize an SE. (PEAD D2 hit 8/40 singular two-way cells.)
- **Estimators:** TWFE (`reghdfe ... , absorb(gvkey_id filing_ym_n)
  vce(cluster gvkey_id)`) as companion; BJS `did_imputation ...
  autosample cluster(gvkey_id) pretrends(12)` as the modern-DID anchor.
  Monthly Amihud is a **diagnostic exhibit only** (monthly BJS rejects
  parallel trends); the **filing-window abnormal Amihud event design is the
  primary causal anchor**.
- **p-values:** `reghdfe` + `lincom` for cluster-adjusted t (use `e(df_r)`).
- **Winsor:** outcomes winsorized 1/99 within the analysis sample before
  estimation.
- **Pre-filing benchmark:** main `[-30,-6]`; robustness `[-45,-6]`,
  `[-60,-6]`, `[-60,-16]`. Right edge at −6 prevents overlap with the widest
  `[-5,+5]` disclosure window. NOT cited to any paper — derived from
  overlap-prevention logic.
- **Stata gotchas:** 32-char var-name truncation on `import delimited`
  (rename long discovery columns right after import); `!missing(...)` or
  lines starting with `!` trip the Stata-MCP shell-escape guard → rewrite as
  `(x < .)`; `duplicates drop ..., force` silently drops rows → use `isid`.

---

## 7. Key files (all in `bot_paper/panel_construction/` unless noted)

**Treatment build**
- `firm_discovery_treatment.csv` — broad23 + search16 cohort + timing
- `firm_discovery_treatment_narrow.csv` — narrow17 conservative subset
- `firm_robots_treatment.csv` — all robots directive-type cohorts (153 cols)
- `build_firm_discovery_treatment.py` / `_narrow.py`,
  `build_firm_robots_treatment.py`

**Outcome panel**
- `edgar_filing_event_panel.csv` — 94,491 filings × 437 cols. Carries
  amihud / abs_ret / vol / qspread / abn_edgar across all windows + pre-filing
  benchmark windows. **This is the panel for channel analysis.**
- `build_edgar_filing_event_panel.py` — its builder (has `ALT_PRE_WINDOWS`
  for benchmark robustness)

**Amihud headline + treatments**
- `run_amihud_main_broad_discovery.do` → `amihud_main_broad_discovery_results.csv`
  (headline + fragility heterogeneity)
- `run_amihud_filing_window_robots_broad23.do` →
  `amihud_robots_filing_window_results.csv` (all robots directive-type
  treatments, TWFE + BJS, 6 windows)
- `run_amihud_treatment_r2_augment.do` → `amihud_treatment_r2_augment_results.csv`
  (6-treatment [0,+5] table with R² — this session)

**Channel battery (the nulls)**
- `run_filing_window_non_edgar_channel_map_broad23.do` →
  `non_edgar_channel_map_results.csv` + `table_non_edgar_channel_map.tex`
  (Panel A IBES, Panel B ln_vol_sum, Panel C abn_qspread_mean,
  + abn_abs_ret_sum) ← **the decomposition starting point**
- `run_edgar_10kq8k.do` — form-specific EDGAR access (substitution ruled out)
- `run_spread_proxy_robustness.do` → `spread_proxy_robustness_results.csv`;
  `build_filing_daily_spread_proxies.py` → `firm_filing_daily_spread_proxies.csv`
  (4 daily spread proxies)

**PEAD (this session, all null)**
- `run_pead_dvp_robustness.do`, `run_pead_martineau_power_check.do`,
  `run_pead_martineau_d2_treatment_sensitivity.do`,
  `run_pead_three_protocol_treatment_sensitivity.do` (unrun)

**LaTeX**
- `git/practice/robots_txt_illiquidity_drift_standalone.tex` — the
  user-preferred standalone (treatment defs + Amihud table + drift), uses
  `Amihud2002`, `BorusyakJaravelSpiess2024`, `DellaVignaPollet2009`,
  `HirshleiferLimTeoh2009`, `Martineau2022` (note: Martineau cited as 2022).
- `bot_paper/draft_2026_05_06/section_appendix_pead_analyst_channel.tex` —
  PEAD null appendix.

---

## 8. Suggested first moves in the new window

1. **Open `non_edgar_channel_map_results.csv`** and pull the broad23 rows for
   `abn_abs_ret_sum_<win>` (numerator) vs `ln_vol_sum_<win>` (denominator).
   See whether the Amihud rise is numerator-driven (it should be — volume is
   null).
2. **Write a focused decomposition do-file** on `edgar_filing_event_panel.csv`:
   TWFE filing-window DID on the Amihud numerator and denominator separately,
   under broad23, with the same fragility splits (low-vol, high-pre-Amihud).
   Mirror `run_amihud_main_broad_discovery.do`'s sample construction
   (X-strict, drop first-block month, winsor 1/99, firm + filing-month FE,
   firm cluster).
3. **Decide the positive-channel claim:** if numerator (|ret|/volatility per
   unit volume) drives the Amihud rise while volume and spread stay flat, that
   is the cleanest available positive support for a price-impact / adverse-
   selection (Kyle-λ) reading — state it as such, and acknowledge no
   intraday/signed-volume data to push further (TAQ unsubscribed).

---

*End of handoff. The new window should re-read the relevant `do`/`csv` before
quoting any number, per the "verify before recommending" rule.*
