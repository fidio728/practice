# P0 Full-Battery Rerun — Pre-P0 vs Post-P0 Ledger

**Date:** 2026-08-04
**Scope:** Essay 2 (China-exposure DDD, US vs NONUS holdings). Ten families rerun on the P0 holdings rebuild (as-of W=10 rule). Baseline shock only.
**Panel:** `julia_descriptive/output/c6_panel.dta` = 348,156 rows / 6,867 firms / 82 quarters (2003Q3–2023Q4).
**Frozen inputs:** exposure and shock sides are bit-frozen. Verified two ways: `sd(shock)=2.57789`, `mean=0.16695` over 82 quarters (pre-P0 2.5779, unchanged), and `cmp` proves `country_pair_shocks_monthly.csv` / `country_pair_shocks_coefficients.csv` byte-identical to their pre-P0 backups. Everything that moved, moved because of the holdings side.
**Independent verification:** an adversarial verifier re-derived β₃ for seven families plus the pooled headline using its own singleton-dropper + alternating-projection FE absorption + plain OLS (algorithmically independent of the two-group collapse used by the `run_ri_*.py` scripts). Every re-derived coefficient matched to 4–12 significant figures. Verification scripts: `indep_rederive.py`, `indep_rederive2.py`, `indep3.py` in the session scratchpad.

---

## 0. Canonical headline

| Spec | Pre-P0 | Post-P0 (CANONICAL) |
|---|---|---|
| 3-pairwise (fq gq ig) β₃ | +2.746e-06 (se 1.70e-06) | **−5.279916e-07** (se 1.743166e-06) |
| 3-pairwise p (CRVE) | ~0.106 | **0.762748** |
| 3-pairwise N | 347,490 | **347,690** (466 singletons dropped) |
| 3-pairwise RI free / circ | 0.3083 / — | **0.8154 / 0.8293** |
| it+gt (fq gq) β₃ | +2.081e-06 (se 1.44e-06) | **−1.072712e-06** (se 1.819055e-06) |
| it+gt p | 0.151 | **0.557028** |
| it+gt N | 347,952 | **348,156** |
| it+gt RI | 0.412 | **0.5984** |

Source of truth: `output/headline_3pairwise_canonical.csv` (12:09:29) and `output/audit_ri_3pairwise.csv` (12:24:14), both post-P0. The gate value was independently re-derived at −5.2799160827e-07.

---

## 1. TOP SUMMARY

### 1.1 Conclusions that CHANGED

1. **The headline sign flipped and the whole positive pattern dissolved.** Pre-P0 the pooled DDD was +2.75e-06 (3pw) / +2.08e-06 (it+gt) with p in the 0.11–0.15 band. Post-P0 both are negative and flatly null (p=0.76 / 0.56). The old positive-sign pattern was a calendar artifact of the pre-P0 holdings stamping.
2. **The direction split reverses its narrative (§7.7).** Pre-P0 the *buy* arm carried the suggestive signal (+5.50e-06, RI p=0.136). Post-P0 the buy arm collapses to +1.01e-06 (p=0.716, RI 0.670) and the *sell* arm becomes the larger-|t| one at −1.96e-06 (p=0.157, RI 0.474). Any text asserting a buy-side directional tilt must be rewritten, not renumbered.
3. **The flow/ownership headline is DEAD (§7.6).** Pre-P0 flow 3pw was +8.90e-04 with CRVE p=0.0001. Post-P0 the identical spec gives −3.6143e-02, p=0.2366. Every CRVE spec in the family is now null and every sign flips negative. The raw-flow point estimate is also demonstrably 100% fat-tail: winsorized at p1/p99 it becomes +7.5e-05 (p=0.83), a sign flip and a ~480× magnitude collapse. Raw flow kurtosis is 198,985 with max 1724.9 against p99 = 0.1387.
4. **The four-group MAIN full-period near-significance is gone (§7.8).** +2.789e-06 (CRVE p=0.069, RI 0.289) → −7.065e-07 (p=0.684). This was the only near-significant column in that family.
5. **The country-pair b₁ is dead (§7.5).** +2.22e-05 (firm-quarter p=0.086) → −1.36e-06 (p=0.910 / 0.943 / 0.947 across the three clustering levels). Magnitude fell ~16× and the sign flipped. The whole point of the F4 clustering audit — that firm-quarter clustering might overstate precision relative to the honest 3-country level — is now moot: the point estimate is identical at all three levels and all three p-values exceed 0.91.
6. **The F1b local-projection significance is gone (§7.1 audit).** Pre-P0 LP h=1 p=0.017, h=2 p=0.003, h=4 p=0.035. Post-P0 all three are null: h=1 p=0.599, h=2 p=0.849, h=4 p=0.105, with RI 0.526 / 0.865 / 0.296.
7. **The cum4 rejection no longer survives even the free permutation (§7.1/§15).** Pre-P0 free RI p=0.0388. Post-P0 free 0.0664, circular 0.1220, moving-block 0.1074/0.1076, within-family max-|t| FWER 0.0558/0.0618. The claim "cum4 is the only RI-significant cell" no longer holds. **But** the score-based quarter-cluster WCB now reads p=0.0038 (t=+2.666) — MORE significant than pre-P0. The two inference routes point opposite ways at cum4. This is the single live inferential tension in the battery and needs an explicit adjudication sentence in the doc.
8. **The risk-set family loses its "directionally supportive" character (§7.4/07f/07g).** All five β₃ flip positive→negative (07g R1 +3.26e-06 → −1.73e-06; 07g R2 +3.51e-06 → −1.71e-06; F8 +3.35e-06 → −1.77e-06; 07f S1 +4.38e-06 → −2.62e-06; 07f S2 +4.32e-06 → −2.37e-06) and every p moves from 0.14–0.45 to 0.54–0.59. Nothing was significant before or after, so no conclusion is overturned, but the consistently-positive pre-P0 pattern that made the risk set look corroborative no longer exists.
9. **The Russia positive control weakens further (§7.9).** −4.21e-06/p=0.035/RI 0.243 → −3.842e-06/p=0.0189/RI 0.208 (it+gt); −4.28e-06/p=0.036/RI 0.246 → −3.869e-06/p=0.0178/RI 0.220 (3pw). CRVE got *more* significant, RI stayed null. The event-window placebo got worse (0.457 → 0.4815). The qualitative sentence — "right sign but does not reach significance under valid design-based inference; cannot be cited as passing design validation" — SURVIVES and is if anything more strongly supported.
10. **Motivating descriptive gap widened (§7.10).** US-vs-NONUS zero-holder gap on the full grid: +11.30pp → **+12.23pp**. On the estimable subset (cn_lag non-missing) it is **+15.18pp**. The paper must say which denominator it quotes; they now differ by ~3pp.
11. **The sagg magnitude grew while staying null (§7.3c).** 3pw h=0 −1.12e-06/p=0.411 → −1.66e-06/p=0.0939; it+gt −1.06e-06/p=0.348 → −1.55e-06/p=0.0974 with RI free 0.512→0.235, circ 0.463→0.220. The lowest p in the whole battery outside Russia, still not below 0.05, and the 3pw cell has no RI arbiter.
12. **The `dos` comparison spec is no longer VCE-degenerate.** Pre-P0 it was the pre-registered `se_valid=0` case. Post-P0 all 16 rows across the three ownership/flow diag files are `se_valid=1`. The vintage-dependence warnings baked into the `.do` comments now point at the wrong spec.

### 1.2 Conclusions that HELD

1. **Every family is null.** Across all ten families, the only cells clearing 0.05 under any inference route are: Russia CRVE (RI-null), the extensive-margin exit it+gt companion (p=0.0437, wrong-sign vs its own 3pw primary, no RI arbiter), and the cum4 score-WCB (contradicted by all four RI variants). No conclusion of "an effect" is licensed anywhere.
2. **Pooling remains licensed (§7.7).** The direction-split pooling-validity test (β₂ₛ=β₂ᵦ AND β₃ₛ=β₃ᵦ) is not rejected: F(2,81)=0.77, p=0.4651 (pre-P0 p=0.424). The pooled headline is still the right object.
3. **The split is informative, not collinear (§7.7).** corr(sell_lag, buy_lag | cn_lag>0) = −0.1353 on 24,346 exposed firm-quarters; cells sell-only 11,429 / buy-only 8,533 / both 4,384 / neither 0. Independently re-derived EXACT.
4. **Tercile assignment is unchanged (§7.3).** Cutpoints p33.33 = −0.6383, p66.67 = +0.0116; bins 28/27/27, identical to pre-P0, because the shock side is frozen. Independently re-derived EXACT.
5. **σ_S invariant holds.** sd = 2.57789 (pre-P0 2.5779). Shock side confirmed bit-frozen.
6. **The US-vs-NONUS firm-level ever-held gap is unchanged (§7.10).** US 56.59% vs NONUS 92.11% ≈ −35.5pp (pre-P0 ~56.7% / ~92.1%).
7. **The US 2021Q4 passive-share anchor holds (§7.8).** Documented 39.78% → recomputed 39.75% on the identical pooled denominator. The fund-universe change did not move it. (Note: this anchor is NOT coded as a check inside `build_fourgroup_panel.py`; it was recomputed independently.)
8. **Panel integrity is clean.** Four-group recon-A max_abs_dev = 0.000e+00 over 199 (side,quarter) cells; recon-B max|Σw−1| = 1.55e-15 over 395 books; risk-set pairing asserts 449,283/449,283 with 0 unpaired; sagg consistency assert max dev 7.1e-15; Russia weight-sum integrity < 1e-9; extensive-margin structural asserts (a)–(h) all OK.
9. **All families run on post-P0 inputs.** Every estimation panel post-dates the holdings rebuild (`holdings_eom.parquet` 2026-08-04 07:56:08). The ONE exception is `output/firm_ladder_panel.dta` (2026-07-22), feeding the DDD Stage-C bilateral control — disclosed below.
10. **B9 convention: exactly ONE `se_valid=0` row exists** across all ten `*_vce_diag.csv` files — `russia_headline_vce_diag.csv` row r3/us_ru_post — and it is the pre-registered expected one.

### 1.3 What is NOT DONE (blocking a complete refresh)

Three RI jobs were reported by their family agents as "still running, do not relaunch." **They are dead.** Verified at report time: only one `python.exe` exists (PID 880, the stata-mcp helper); all three logs are 446 bytes containing nothing but the two pandas import warnings; the three output CSVs still carry pre-P0 mtimes.

| Script | Log | Output CSV mtime | Status |
|---|---|---|---|
| `run_ri_flow.py` | `_rerun_ri_flow.log` 14:31:50, 446 B | `ri_flow_results.csv` 2026-08-03 11:09 | DEAD, produced nothing — RELAUNCH |
| `run_ri_extensive.py` | `_rerun_extmargin_ri.log` 14:41:56, 446 B | `ri_extensive_results.csv` 2026-08-04 00:39 (pre-07:56 rebuild) | DEAD, produced nothing — RELAUNCH |
| `run_ri_fourgroup.py` | `_rerun_fourgroup_ri.log` 14:49:27, 446 B | `ri_fourgroup_results.csv` 2026-08-03 00:02 | DEAD, produced nothing — RELAUNCH |

Downstream of `run_ri_flow.py`, two more steps were never launched: `run_flow_decomposition.py` (step 7) and `run_flow_decomp_step3.do` (step 8). The entire §7.6 decomposition is owed.

By contrast, **f1's RI is NOT pending** — `ri_direction_results.csv` has mtime 2026-08-04 14:20:22 and holds the finished P0 result. The f1 agent misread the mtime. Its numbers are folded into §7.7 below.

---

## 2. DOC REFRESH FEED — every section touched

### §0 — Headline / abstract-level numbers
Replace the pooled DDD everywhere. New: 3pw β₃ = **−5.279916e-07**, se 1.743166e-06, p = **0.762748**, N = **347,690**, RI free **0.8154** / circ **0.8293**. it+gt β₃ = **−1.072712e-06**, se 1.819055e-06, p = **0.557028**, N = **348,156**, RI **0.5984**. Panel = **348,156 rows / 6,867 firms / 82 quarters**. Any sentence containing "+2.7e-06", "+2.08e-06", "347,490", "347,952", or "6,854 firms" is stale.

### §5.5 — Sample construction / panel dimensions
- c6 panel: 348,156 / 6,867 / 82 (was 347,952 or 347,490 depending on FE set; 6,854 firms).
- Firm universe: **12,791** sec_entity_id (was 12,743).
- 3-pairwise singleton drop: **466** obs (was 462 — see the tercile footnote defect below), leaving 6,634 firm clusters.
- Ownership/flow universe: **248,024 rows / 5,191 firms** — i.e. ~76% of the main panel's firms, because the primary-EQ float restriction plus the BAD firm-quarter rule bind harder. Re-check whether the coverage gap moved with P0 before writing the sample-comparison sentence.
- Fraction of holdings MV on primary EQ: US **91.2144%** (was 91.08%), NONUS **96.7794%** (was 96.68%).

### §7.1 — Main DDD + audit consumers
New headline as §0. Audit cells: F1a lead −3.18e-07/p=0.791/N=338,076/RI 0.8775 (was +4.85e-07/p=0.774). F2 in-span −1.57e-06/p=0.579/N=272,140/RI 0.6093. F7 GPR: us_cn_gpr −2.20e-06 p=0.126, us_cn_gprlag +9.51e-07 p=0.350 (was 0.63/0.88). **F1b LP loses all significance**: h0 −1.07e-06/0.557/RI 0.5984; h1 −1.25e-06/0.599/RI 0.5260 (was p=0.017); h2 +3.85e-07/0.849/RI 0.8647 (was p=0.003); h3 +9.28e-08/0.968/RI 0.9711; h4 +3.10e-06/0.105/RI 0.2962 (was p=0.035). 07d three-spec: spec0 no-FE −9.29e-08/0.854; spec1 headline −1.07e-06/0.557/R²=0.590; spec4 weak-FE +5.24e-07/0.453. 07e: A0 −1.07e-06/0.557; A1 firm×group FE −5.28e-07/0.763/RI 0.8014; tail B1 k=1.645 −2.715e-05/0.129; B2 k=2.0 −2.302e-05/0.320; B3 k=3.0 −6.051e-05/0.0696.

### §7.3 — Shock-tercile dose menu
3pw: T3 **+2.77e-06** (se 1.87e-05, p 0.882, RI **0.848**) — was +2.05e-05 / p 0.264 / RI 0.233. T1 +7.02e-07 (p 0.972). Level −8.66e-07 (p 0.955). Contrast T3−T1 **+2.07e-06** (se 1.03e-05, p 0.841, RI **0.874**) — was +6.76e-06 / p 0.623 / RI 0.669. N = 347,690, R² 0.5908. it+gt column: T3 +4.55e-07 (p 0.980), T1 +1.01e-06 (p 0.958), level −1.79e-06 (p 0.908), N 348,156, R² 0.5902. Cutpoints and bins unchanged. **Fix the footnote**: `tercile_results.csv` addnote says "MAIN drops 462 singletons"; the true count is **466**. Confirmed independently.

### §7.3b — Shock lag
S_{t−1} it+gt **−1.07e-06** (se 1.18e-06, p **0.3696**, RI **0.6018**), N 348,156 — was −5.90e-07 / p 0.750 / RI 0.814. S_{t−1} 3pw −7.30e-07 (se 1.24e-06, p 0.5589), N 347,690, no RI. Per 1-SD: −2.75e-06 (it+gt) / −1.88e-06 (3pw), using sd(shock_{t−1}) = 2.5770. Cross-engine S_t checks reproduce the canonical headline exactly.

### §7.3c — Aggregated within-quarter shock (sagg)
3pw D1 h=0 **−1.66e-06** (se 9.80e-07, p **0.0939**) — was −1.12e-06 / p 0.411. it+gt D1 h=0 **−1.55e-06** (se 9.23e-07, p **0.0974**, RI free **0.2353** / circ **0.2195**) — was −1.06e-06 / p 0.348 / RI 0.512 / 0.463. D2 h=1: +7.36e-08 (3pw, p 0.925) / −4.10e-08 (it+gt, p 0.953). D3 distributed-lag 3pw: h0 −1.67e-06 (p 0.0906), h1 +1.90e-07 (p 0.793), joint F(2,81)=1.545 p 0.2194, cumulative −1.48e-06 (p 0.242, CI [−3.99e-06, 1.02e-06]). D3 it+gt: h0 −1.62e-06 (p 0.0953), h1 +3.11e-07 (p 0.632), joint F=1.426 p 0.2464, cumulative −1.31e-06 (p 0.198). σ(s_agg) = 4.0846, σ(stamped) = 2.5779, corr = 0.425, corr(s_agg, s_agg_l1) = 0.357. Panel 348,156 → 348,156 (zero dropped), 264 quarters with all 3 monthly residuals.
**Caveat to write in:** the per-SD sagg effect is −6.78e-06 versus the headline stamped-shock −1.36e-06, ~5× larger, more than measurement-error attenuation alone naturally produces, and it coexists with a null p. Two readings are consistent with the data (genuine de-attenuation from aligning the information window, or a low-power estimate wandering) and the battery cannot separate them. Do not present the magnitude growth as corroboration without noting the test does not reject zero.

### §7.4 / §7.5 — Risk set, country-pair, spell boundary
- 07g R1 riskset fq gq: **−1.73e-06** (se 2.84e-06, p **0.5441**), N **270,796**, R² 0.5902 — was +3.26e-06 / p 0.1505 / N 268,282.
- 07g R2 + firm×group: **−1.71e-06** (se 3.00e-06, p 0.5709), N 270,386 (410 singletons) — was +3.51e-06 / p 0.1432 / N 267,898.
- F8 lag-only (no look-ahead): **−1.77e-06** (se 2.90e-06, p 0.5443), N **268,800**, 5,613 firms — was +3.35e-06 / p 0.153. Stability vs the with-lead risk set still HOLDS (−1.77e-06 vs −1.73e-06): lead-conditioning is not driving the estimate.
- F4 country-pair b₁ (US×S_c, PRIMARY): **−1.36e-06** at every clustering level; p **0.9103** (firm-quarter) / **0.9432** (3 country clusters) / **0.9467** (country×quarter). Was +2.22e-05 / p 0.086 / p 0.246. b₃ (us_cn_sc) −8.13e-06, p 0.764 / 0.521 / 0.408. N **172,920** of a 1,315,200-row panel (cn_lag missing on 1,142,280 rows — worth a footnote).
- 07f S1 spell boundary: **−2.62e-06** (se 4.33e-06, p 0.5466), N **209,056** (30,870 singletons dropped) — was +4.38e-06 / p 0.4414 / N 250,918. S2 + firm×group: −2.37e-06 (se 4.37e-06, p 0.5899), N 208,874.
- **Power note owed:** the spell-boundary estimation N fell 16.7% (250,918 → 209,056) and firm clusters to 3,939, while the risk-set samples GREW. State this explicitly if 07f is cited as a robustness check.
- **No RI anywhere in this family.** The three do-files contain no ritest/permute block. This is not a missing output; supplying an RI arbiter here would require new code.

### §7.6 — Ownership share and flow
- r1 FLOW it+gt: **−3.3293e-02** (se 2.6793e-02, p **0.2176**), N 248,024.
- r2 FLOW 3pw (= old headline spec): **−3.6143e-02** (se 3.0312e-02, p **0.2366**), N 247,582 — was **+8.90e-04, p=0.0001**. Sign flip, ~40× magnitude, significance gone.
- r3 dos comparison: −8.59e-05 (se 3.4349e-04, p 0.8032) — and **no longer VCE-degenerate** (se_valid=1, was the pre-registered se_valid=0 case).
- Held-only (os>0, re-paired): fq+gq −5.4410e-02 (p 0.2031, N 196,050); 3pw −6.0805e-02 (p 0.2234, N 195,742) — was +1.35e-03.
- **Winsorized p1/p99**: fq+gq +7.5197e-05 (p 0.796); 3pw +7.4816e-05 (p 0.8265). The raw-vs-winsor gap proves the raw point estimate is entirely a handful of tiny-lagged-denominator outliers. Report the winsorized column as the informative one.
- Fat-tail diagnostics: kurtosis **198,985** (pre-P0 ~5,000), sd 3.674, max 1724.9, p99 0.1387, p1 −0.1364.
- Ownership observed panel: 582,463 firm-group-quarter cells, 11,732 firms, 100 quarters, 1999Q1–2023Q4; shrout missing 0.1835% of cells; os>1 impossible in 31 cells (0.0053%); os median 0.0325 / p90 0.1713 / p99 0.3800.
- **NOT DELIVERED:** RI for flow (raw + winsor) and the full 6-cell flow decomposition (flow_diff / flow_common / flow_r × raw/winsor) plus its CRVE companion.

### §7.7 — Direction split (sell vs buy)
| Cell | Pre-P0 | Post-P0 |
|---|---|---|
| b₃ SELL (link-share) | +2.527e-07, p 0.618, RI 0.931 | **−1.9639e-06**, se 1.3749e-06, p **0.1570**, RI **0.4735** |
| b₃ BUY (link-share) | +5.501e-06, p 0.147, RI 0.136 | **+1.0131e-06**, se 2.7750e-06, p **0.7160**, RI **0.6701** |
| Contrast sell−buy | −5.248e-06, RI 0.2046 | **−2.977e-06**, se 2.58e-06, p 0.253, RI **0.3171** |
| Joint b₃ₛ=b₃ᵦ=0 | not reported | F(2,81)=1.51, p 0.2281 |
| Pooling validity | p 0.424 | F(2,81)=0.77, p **0.4651** — still not rejected |
| Indicator b₃ any-SELL | +6.515e-07, p 0.609 | −1.2825e-06, se 1.1067e-06, p 0.2499 |
| Indicator b₃ any-BUY | +2.407e-06, p 0.536 | −9.241e-07, se 2.8035e-06, p 0.7425 |
N = 347,690, R² 0.5908 (was 347,490 / 0.6244). RI: N_PERM=5000, n_fq=174,078. Descriptives as §1.2.

### §7.8 — Four-group active/passive
| Column | Pre-P0 β₃ | Post-P0 β₃ | se | p |
|---|---|---|---|---|
| MAIN US-ACT vs NONUS-ACT, full | +2.7894e-06 (p 0.069, RI 0.289) | **−7.0647e-07** | 1.7270e-06 | **0.6836** |
| MAIN post-2018 (PRIMARY) | +1.1598e-06 (p 0.376, RI 0.561) | **+5.0479e-07** | 1.7146e-06 | **0.7713** |
| PASSIVE benchmark, full | +1.03e-06 | +3.3374e-07 | 1.3136e-06 | 0.8001 |
| PASSIVE benchmark, post-2018 | −0.14e-06 | **+1.5258e-06** | 1.0913e-06 | 0.1766 |
| US-internal ACT vs PAS, full | +1.56e-06 | +1.6021e-06 | 1.0290e-06 | 0.1234 |
| US-internal, post-2018 | −0.19e-06 | −2.0162e-07 | 9.3370e-07 | 0.8311 |
N 347,690 (full) / 183,880 (post-2018, 622 singletons, 22 quarters). Panel 696,312 rows / 6,867 firms / 82 quarters; funds master 106,965 unique fund_id (ACTIVE 39,429 / PASSIVE 9,353 / UNKNOWN 58,183). Matched-share decay US 0.995 (1999) → 0.917 (2023); NONUS 0.845 (2023).
**Framing change owed:** the "passive is the inert arm" sentence is weaker post-P0. The passive post-2018 benchmark flipped sign and is now the LARGEST |β₃| of the six columns (though still ns at p=0.177). Restate from the new numbers.
**Dilution-benchmark comment in `run_fourgroup.do` line 42 still cites the pre-P0 pooled +2.746e-06** — the live value is −5.279916e-07. Doc-pass fix, not a code fix.
**RI NOT DELIVERED.** Pre-P0 RI captured verbatim before overwrite for the record: full b_obs +2.789445742549961e-06, ri_p 0.28854229154169164, n_fq 173,976; post2018 b_obs +1.159757444844631e-06, ri_p 0.5612877424515097, n_fq 92,168.

### §7.9 — Russia positive control
| Cell | Pre-P0 | Post-P0 |
|---|---|---|
| R1 it+gt β₃ | −4.21e-06, se 1.97e-06, p 0.035, RI 0.243, N 347,952 | **−3.84187e-06**, se 1.60434e-06, p **0.0189**, RI **0.2085**, N **348,156** |
| R2 3pw β₃ | −4.28e-06, se 2.01e-06, p 0.036, RI 0.246, N 347,490 | **−3.86876e-06**, se 1.59921e-06, p **0.0178**, RI **0.2198**, N **347,690** |
| R3 event-window 2022Q1–Q2 | −3.44e-05, se degenerate, placebo 0.457 | **−2.8858e-05**, se = 0 (DEGENERATE, se_valid=0), placebo **0.4815** |
| LP cumulative h=0..7 | 8/8 negative, perm_p 0.218–0.413, n 4,289 | **8/8 negative**, β −2.53e-05 … −4.38e-05, perm_p **0.2703–0.4826**, n **4,294** |
LP t-stats range −0.634 to −1.086; none |t|>1.09; min perm_p 0.2703 at h=1. R3 CRVE p is `.` and must not be read as inference; the event-window placebo is the valid design-based read. Qualitative verdict UNCHANGED.

### §7.10 — Extensive margin
| DV (3pw PRIMARY) | Pre-P0 β₃ / RI | Post-P0 β₃ | se | CRVE p | N |
|---|---|---|---|---|---|
| d_breadth | +2.42e-05 / RI 0.768 | **+2.197184e-05** | 3.834406e-05 | 0.568220 | 347,690 |
| d_nh | −1.17 / RI 0.549 | **−1.453061** | 1.041273 | 0.166689 | 347,690 |
| exit (both-held) | +1.73e-03 / RI 0.695 | **+1.415490e-03** | 3.715630e-03 | 0.704233 | 201,634 |
| init (neither-held) | −1.11e-03 / RI 0.805 | **−6.622296e-04** | 2.361000e-03 | 0.779819 | 85,136 |
it+gt companions: d_breadth +1.4756e-05 (p 0.684); d_nh −1.3584 (p 0.182); **exit −5.0551e-03 (se 2.4673e-03, p 0.0437)** — the WATCH ITEM, was p=0.006; init −2.1752e-03 (p 0.433). Risk-set variants: d_breadth +2.468393e-05 (p 0.660, N 270,386); d_nh −2.053915 (p 0.178); exit **bit-identical** to the full-sample row (the risk-set restriction is mechanically non-binding for exit — do not present it as an independent robustness result); init N=7,824 / 8,518, underpowered, label uninformative.
Descriptives: zero-holder gap **+12.23pp** full grid (US 0.82757 vs NONUS 0.70527 over 1,279,100 cells) and **+15.18pp** on the estimable subset (US 0.4138 vs NONUS 0.2620 over 174,078). Firm-level ever-held US 0.5659 vs NONUS 0.9211 (−35.52pp). ≥1 zero-holder quarter: US 12,646/12,791 = 0.9887, NONUS 1.0000. Four-cell held_lag: both 206,809 (16.33%), US-only 11,003 (0.87%), NONUS-only 166,137 (13.12%), neither 882,360 (69.68%) of a 1,266,309 domain (was 204,314 / 12,356 / 155,352).
**RI arbiter NOT DELIVERED** for any DV. The family verdict is provisional on CRVE only.

### §13 — Summary tables
Every row that quotes a pre-P0 β₃/p/RI/N must be replaced from the tables above. Specifically flagged as stale by the family agents: the §7.9 table at lines 500–508 and the §13 summary rows at line 680 and line 796 of `Essay2_methodology_full.md`, which still carry −4.21e-06/0.035/0.243, −4.28e-06/0.036/0.246, −3.44e-05, N 347,952/347,490, placebo 0.457, LP perm_p 0.218–0.413, n=4,289. All eight of those figures moved.

### §15 — Inference hardening (cum4)
New gates (verified against `headline_3pairwise_canonical.csv` and `audit_ri_3pairwise.csv` BEFORE editing): h0 −5.2799161e-07, cum1 −1.5346165e-07, cum4 +6.6006275e-06; P_FREE_ANCHORS 0.8014 / 0.9368 / 0.0636. All three gates PASS at relerr ≤ 1.3e-08.

| Cell | β₃ | RI free | RI circ | RI move-block L5 / L8 | maxT FWER free / mb | score-WCB |
|---|---|---|---|---|---|---|
| h0 | −5.2799e-07 | 0.8154 | 0.8293 | 0.7856 / 0.7766 | 0.9990 / 0.9990 | 0.8032 |
| cum1 | −1.5346e-07 | 0.9376 | 0.9634 | 0.9400 / 0.9434 | 1.0000 / 1.0000 | 0.9463 |
| cum2 | +2.2694e-06 | 0.3675 | 0.4268 | 0.4069 / 0.3919 | 0.6181 / 0.5957 | 0.1774 |
| cum3 | +2.8071e-06 | 0.3619 | 0.3537 | 0.3997 / 0.4173 | 0.5069 / 0.4869 | 0.1160 |
| **cum4** | **+6.6006e-06** | **0.0664** | **0.1220** | **0.1074 / 0.1076** | **0.0558 / 0.0618** | **0.0038** |
Pre-P0 free RI anchors, for the record: h0 0.3083 (β +2.745538e-06), cum1 0.1052 (β +4.093621e-06), cum4 0.0388 (β +8.825974e-06).
Serial diagnostics on the shock: acf₁ = +0.270747, acf₂ = +0.163492; Ljung-Box Q(1) = 6.2336 p = 0.0125, Q(4) = 11.3632 p = 0.0228 over 82 quarters. Anchor cross-check drift +0.0140 / +0.0008 / +0.0028, all inside the 0.02 MC tolerance, no DRIFT flag.
**Write the adjudication explicitly:** four design-based routes put cum4 above 0.05 and one bootstrap route puts it at 0.0038. Given the pre-registered arbiter hierarchy (RI is the anchor when CRVE is CGM-repaired or when clusters are few), cum4 should be reported as NOT surviving, with the WCB disagreement disclosed rather than buried.

---

## 3. FAMILY-BY-FAMILY LEDGER

### f1 — Direction split (§7.7)
**STATUS: null-held, narrative-reversed. New issue: family agent's "RI pending / file stale" claim was FALSE.**

| Spec | Pre-P0 β₃ / p / RI | Post-P0 β₃ / se / p / RI | N |
|---|---|---|---|
| d_main us_sell_shock | +2.52673e-07 / 0.6181 / 0.9312 | **−1.96387e-06** / 1.37491e-06 / **0.1570** / **0.4735** | 347,690 |
| d_main us_buy_shock | +5.50084e-06 / 0.1473 / 0.1364 | **+1.01309e-06** / 2.77497e-06 / **0.7160** / **0.6701** | 347,690 |
| d_main us_sell (lower order) | +2.25e-06 / 0.7937 | +5.4292e-06 / 8.8654e-06 / 0.5420 | 347,690 |
| d_main us_buy (lower order) | +2.83e-06 / 0.7995 | −2.1217e-06 / 8.2310e-06 / 0.7972 | 347,690 |
| Contrast sell−buy | −5.24816e-06 / ~0.20 / 0.2046 | **−2.977e-06** / 2.58e-06 / 0.253 / **0.3171** | 347,690 |
| Joint b₃ₛ=b₃ᵦ=0 | — | F(2,81)=1.51, p 0.2281 | 347,690 |
| Pooling b₂ₛ=b₂ᵦ ∧ b₃ₛ=b₃ᵦ | p 0.424 | F(2,81)=0.77, **p 0.4651** | 347,690 |
| d_ind us_dsell_shock | +6.51546e-07 / 0.6089 | −1.28250e-06 / 1.10668e-06 / 0.2499 | 347,690 |
| d_ind us_dbuy_shock | +2.40717e-06 / 0.5363 | −9.24134e-07 / 2.80355e-06 / 0.7425 | 347,690 |
| d_ind us_dsell (lower) | −1.35e-06 / 0.7550 | −6.1699e-08 / 5.2836e-06 / 0.9907 | 347,690 |
| d_ind us_dbuy (lower) | −9.38e-06 / 0.4504 | −7.6830e-06 / 1.0708e-05 / 0.4751 | 347,690 |
| d_ind contrast | null | −3.58e-07 / 1.94e-06 / 0.854 | 347,690 |

Model fit: MAIN F(4,81)=0.81 p 0.5198; INDICATOR F(4,81)=0.90 p 0.4696; 466 singletons; 6,634 firm clusters; 82 month clusters; fq 173,845 / gq 164 / ig 13,268. R² 0.5908 (was 0.6244). RI config N_PERM=5000, n_fq 174,078.
B9: `direction_vce_diag.csv`, 4 rows, all se_valid=1.
**VERIFY VERDICT: DISCREPANT (status only, all numbers CONFIRMED).** β₃_sell re-derived at −1.9638664156e-06 (8 sig figs), β₃_buy +1.0130922505e-06, contrast −2.9769586661e-06; N, singleton count and all three FE-group counts match. `ri_direction_results.csv` mtime is 2026-08-04 14:20:22 and the log ends with `wrote .../ri_direction_results.csv`. The RI b_obs values reproduce the Stata β₃ to 11 digits, so the file is genuinely the P0 run. f1 is finishable now with no re-run.
**Open issues:** (a) no design-based RI for the INDICATOR spec, which is exactly the spec whose CRVE needed the Cameron-Gelbach-Miller non-PSD repair; (b) the RI script's hardcoded drift anchors (lines 111–113) are pre-P0 and print a large apparent mismatch — that mismatch is the result, not an error.

---

### f2 — Shock-tercile dose menu (§7.3)
**STATUS: null-held, decisively (RI p rose from 0.23 to 0.85). One real footnote defect.**

| Spec | Pre-P0 β₃ / p / RI | Post-P0 β₃ / se / p / RI | N |
|---|---|---|---|
| 3pw us_cn_t3 | +2.04654e-05 / 0.2640 / 0.2330 | **+2.77393e-06** / 1.86716e-05 / **0.8823** / **0.8478** | 347,690 |
| 3pw T3−T1 contrast | +6.75627e-06 / 0.623 / 0.6695 | **+2.07205e-06** / 1.03e-05 / **0.841** / **0.8742** | 347,690 |
| 3pw us_cn_t1 | +1.37092e-05 / 0.5347 | +7.01885e-07 / 1.99927e-05 / 0.9721 | 347,690 |
| 3pw us_cn level | −1.02e-05 / 0.5698 | −8.660e-07 / 1.5340e-05 / 0.9551 | 347,690 |
| it+gt us_cn_t3 | +1.74195e-05 / 0.3327 (N 347,952) | +4.55140e-07 / 1.77502e-05 / 0.9796 | 348,156 |
| it+gt us_cn_t1 | +1.32111e-05 / 0.5399 | +1.01129e-06 / 1.89277e-05 / 0.9575 | 348,156 |
| it+gt us_cn level | −1.14e-05 / 0.4883 | −1.7925e-06 / 1.5390e-05 / 0.9076 | 348,156 |

Cutpoints −0.6383 / +0.0116; bins 28/27/27 (identical to pre-P0). R² 0.5908 / 0.5902 (was 0.6244 / 0.6235). RI N_PERM=5000, DEMEAN_ITERS=30, seed 20260702, n_fq 174,078.
B9: `tercile_vce_diag.csv`, 4 rows, zero se_valid=0. reghdfe printed a non-PSD/CGM warning on MAIN; per the do-file header the CRVE is fragile there and RI is the anchor. Both agree on a decisive null, so nothing is load-bearing on that choice.
**VERIFY VERDICT: CONFIRMED.** Cutpoints and bins independently re-derived EXACT.
**Open issue:** `tercile_results.csv` addnote hardcodes "MAIN drops 462 singletons"; the true count is **466**. Fix before quoting the CSV.

---

### f3 — sagg, aggregated within-quarter shock (§7.3c)
**STATUS: moved materially (|β₃| up ~48%, p down ~4×), still null. Lowest non-Russia p in the battery.**

| Spec | Pre-P0 | Post-P0 β₃ / se / p | N |
|---|---|---|---|
| 3pw D1 h=0 | −1.12e-06 / p 0.411 | **−1.66e-06** / 9.80e-07 / **0.0939** | 347,690 |
| 3pw D2 h=1 | — | +7.36e-08 / 7.82e-07 / 0.9253 | 347,690 |
| 3pw D3 h=0 | — | −1.67e-06 / 9.78e-07 / 0.0906 | 347,690 |
| 3pw D3 h=1 | — | +1.90e-07 / 7.22e-07 / 0.7928 | 347,690 |
| 3pw D3 joint Wald | — | F(2,81)=1.545, p 0.2194 | 347,690 |
| 3pw D3 cumulative | — | −1.48e-06 / 1.26e-06 / 0.2421, CI [−3.99e-06, 1.02e-06] | 347,690 |
| it+gt D1 h=0 | −1.06e-06 / p 0.348 / RI free 0.512 circ 0.463 | **−1.55e-06** / 9.23e-07 / **0.0974** / RI free **0.2353** circ **0.2195** | 348,156 |
| it+gt D2 h=1 | — | −4.10e-08 / 6.87e-07 / 0.9526 | 348,156 |
| it+gt D3 h=0 | — | −1.62e-06 / 9.59e-07 / 0.0953 | 348,156 |
| it+gt D3 h=1 | — | +3.11e-07 / 6.47e-07 / 0.6316 | 348,156 |
| it+gt D3 joint | — | F(2,81)=1.426, p 0.2464 | 348,156 |
| it+gt D3 cumulative | — | −1.31e-06 / 1.01e-06 / 0.1982, CI [−3.31e-06, 6.97e-07] | 348,156 |

Build: 348,156 → 348,156 rows (zero dropped), 264 quarters with all 3 monthly residuals, consistency assert max dev 7.1e-15. σ(s_agg)=4.0846, σ(stamped)=2.5779, corr 0.425, corr(s_agg, s_agg_l1)=0.357. RI observed β₃ −1.5486616802969437e-06 reproduces the Stata it+gt estimate.
**VERIFY VERDICT: CONFIRMED.** All four point estimates re-derived independently from `sagg_panel.parquet`; the it+gt value matches `sagg_ri_check.csv` to 11 digits.
**Open issues:**
1. **Prose-vs-data contradiction.** `build_sagg_panel.py`'s docstring asserts the AR(1) innovations are "serially uncorrelated by construction, so the quarter's total news surprise is their sum," while the script's own diagnostic prints corr = 0.357 with the inline comment "(should be ~0)". Unchanged from pre-P0 (shock side frozen) and already documented as a caveat in `run_ri_sagg.py`, but the docstring claim would not survive external review as written.
2. **RI coverage is partial.** RI permutes only h=0 on the it+gt collapse. The 3pw h=0 cell — the lowest p in the family at 0.0939 — has NO permutation arbiter. Its 2-way analog inflated from CRVE 0.0974 to RI 0.2353, so the 3pw p should be presumed similarly non-robust; that is an inference, not a measurement.
3. **No `sagg_vce_diag.csv` exists.** B9 cannot be reported for this family. This matters here specifically because both D3 regressions emitted the non-PSD/CGM warning — exactly the condition a vce_diag file would flag.
4. Circular-shift resolution floor is 1/82 = 0.0122; not binding at p=0.2195 but the test can never deliver a small p with precision.
5. Neither the SEs, the joint Wald, nor the cumulative figures persist in any results CSV — they lived only in the transient Stata log.

---

### f4 — Shock lag (§7.3b)
**STATUS: null-held, magnitude roughly doubled.**

| Spec | Pre-P0 β₃ / p / RI | Post-P0 β₃ / se / p / RI | N |
|---|---|---|---|
| S_t it+gt (cross-check) | — | −1.07e-06 / 1.82e-06 / 0.5570 | 348,156 |
| S_t 3pw (cross-check) | +2.75e-06 / 0.110 | **−5.28e-07** / 1.74e-06 / **0.7627** | 347,690 |
| S_{t−1} it+gt (advisor spec) | −5.90e-07 / 0.750 / 0.814 | **−1.06687e-06** / 1.18e-06 / **0.3696** / **0.6018** | 348,156 |
| S_{t−1} 3pw | — | −7.30e-07 / 1.24e-06 / 0.5589 | 347,690 |
| S_{t−1} it+gt per 1-SD | — | −2.75e-06 (rescale by 2.5770; t/p unchanged) | 348,156 |
| S_{t−1} 3pw per 1-SD | — | −1.88e-06 | 347,690 |

Invariant: σ(shock_t) = 2.5779, σ(shock_{t−1}) = 2.5770; shock_t mean 0.1669468, var 6.645516, skew 1.667615, kurt 8.458486 over 82 quarters. **Pre-P0 σ_S = 2.5779 → UNCHANGED. Hard invariant HOLDS.** Grid 2,558,200 → 348,156 after requiring dw/cn_lag/shock_t/shock_{t−1}. RI 200,000 perms, seed 20260702.
**VERIFY VERDICT: CONFIRMED.** Both cross-engine S_t checks re-derived from `c6_panel.dta` and match the canonical headline exactly. σ_S re-derived at 2.57789.
**Open issues:** (a) the S_{t−1} 3pw spec triggered the non-PSD/CGM repair; its se/p are post-adjustment. (b) No `shocklag_vce_diag.csv`; B9 not applicable. (c) Process note: the f4 agent's read of `shocklag_ri_results.csv` landed after the rerun had already overwritten it, so the pre-P0 RI figure (0.814) quoted here comes from the task brief, not from disk. No estimation output affected.

---

### f5 — Risk set + country-pair + spell boundary (§7.4/§7.5/07f/07g)
**STATUS: family-wide sign flip, all still null. The family's only borderline number (F4 b₁) is dead.**

| Spec | Pre-P0 β₃ / p / N | Post-P0 β₃ / se / p / N |
|---|---|---|
| 07g R1 riskset fq gq | +3.26e-06 / 2.24e-06 / 0.1505 / 268,282 | **−1.7300e-06** / 2.8399e-06 / **0.5441** / **270,796** |
| 07g R2 + firm×group | +3.51e-06 / 2.38e-06 / 0.1432 / 267,898 | **−1.7063e-06** / 2.9983e-06 / 0.5709 / 270,386 |
| F8 lag-only (no look-ahead) | +3.35e-06 / 0.153 | **−1.77e-06** / 2.90e-06 / 0.5443 / **268,800** |
| F4a country-pair b₁, cluster firm×quarter | +2.22e-05 / 0.086 | **−1.36e-06** / 1.20e-05 / **0.9103** / 172,920 |
| F4b b₁, cluster country (3 clusters) | 0.246 | −1.36e-06 / 1.69e-05 / **0.9432** (df_r=2) |
| F4c b₁, cluster country×quarter | — | −1.36e-06 / 1.80e-05 / **0.9467** |
| F4 b₃ us_cn_sc | −1.21e-06, null | −8.13e-06; p 0.764 / 0.521 / 0.408 |
| 07f S1 spell boundary fq gq | +4.38e-06 / 5.66e-06 / 0.4414 / 250,918 | **−2.6226e-06** / 4.3314e-06 / 0.5466 / **209,056** |
| 07f S2 + firm×group | +4.32e-06 / 5.71e-06 / 0.4517 / 250,750 | **−2.3654e-06** / 4.3717e-06 / 0.5899 / 208,874 |

Lower-order β₂ (us_cn): 07g R1 −1.2244e-06 (p 0.8437, was +6.11e-07 p 0.9384); 07g R2 −1.8065e-06 (p 0.8154); F8 −1.22e-06 (p 0.846); F4 +8.42e-08 (p 0.984); 07f S1 −5.0172e-06 (p 0.5611); 07f S2 −1.6904e-06 (p 0.8721). R²: 0.5902 / 0.5910 / 0.5902 / 0.6392 / 0.6119 / 0.6171.
Builds: spell_riskset 898,566 pre-drop → 270,796 estimation, balanced US 135,398 / NONUS 135,398, 5,625 firms, all pairing asserts passed (449,283/449,283 exactly-2-row firm-quarters, 0 unpaired). riskset_lagonly 268,800 rows / 5,613 firms. spell_boundary 684,542 → 239,926, **uneven by design** NONUS 134,690 / US 105,236. country_pair 1,315,200 rows (GB 799,400 / DE 260,200 / FR 255,600), 795 shock rows, 0 unmatched. AR(1): US a=0.693556 b=0.657211 n=796; GB a=0.040330 b=0.234908; DE a=0.010026 b=0.049298; FR a=0.013404 b=0.469693.
**VERIFY VERDICT: CONFIRMED.** F8 (−1.7673711154e-06) and F4 (b₁ −1.3614366484e-06, b₃ −8.1297201356e-06, us_cn +8.4164044135e-08) re-derived independently, all matching. `cmp` proves both country-pair shock CSVs byte-identical to their pre-P0 backups.
**Open issues:**
1. **No RI anywhere in this family** — 07g, 07f and `run_audit_f4f8.do` contain no permutation block. Eight specs with no design-based arbiter.
2. **No `*_vce_diag.csv`** produced by any script here; B9 is N/A, not "pass."
3. **Spell-boundary power fell 16.7%** (N 250,918 → 209,056, firm clusters 3,939). State this if 07f is cited.
4. **87% of the country-pair panel is unestimable** (172,920 of 1,315,200; cn_lag missing on 1,142,280 rows). Pre-existing, but the gap between panel size and reported N deserves a footnote.
5. **Singleton drops are large in 07f** (30,870 / 31,052, ~13%) vs 410 in 07g R2 — the mechanical consequence of the unpaired per-(firm,group) selection that `build_spell_riskset.py` was written to fix. Do not quote the 239,926 build count and the 209,056 regression N interchangeably.
6. Small-cluster warnings in F4b/F4c ("missing F statistic … too few clusters", df_r=2) plus a CGM repair in F4c. Pre-existing design limits of 3 country units, not P0 artifacts.
7. F4a/F4b/F4c and F8 SEs and p-values persist in no artifact — transient Stata log only.
8. Stale header comments left in place per the no-edit rule in `07g_spell_riskset.do` (268,282 / 5,564 / "+2.081 full grid"), `07f_spell_boundary.do` (296,590 / 6,355 — already inconsistent with its own pre-P0 output of 250,918), `build_riskset_lagonly.py` line 74, and `run_audit_f4f8.do` line 24.

---

### f6 — Ownership share + flow (§7.6)
**STATUS: HEADLINE REVERSED (the +8.90e-04, p=0.0001 result is dead). Chain INCOMPLETE — steps 6/7/8 of 8 not delivered.**

| Spec | Pre-P0 β₃ / p / RI | Post-P0 β₃ / se / p | N |
|---|---|---|---|
| r1 FLOW it+gt | — | **−3.32934e-02** / 2.67930e-02 / **0.2176** | 248,024 |
| r2 FLOW 3pw (old headline) | **+8.90e-04 / 0.0001** / RI 0.3691 | **−3.61428e-02** / 3.03117e-02 / **0.2366** | 247,582 |
| r3 dos comparison | +5.49e-04, se_valid=**0** | −8.58877e-05 / 3.43486e-04 / 0.8032, se_valid=**1** | 248,024 |
| held-only fq+gq | — | −5.44103e-02 / 4.24065e-02 / 0.2031 | 196,050 |
| held-only 3pw | +1.35e-03 | **−6.08051e-02** / 4.95564e-02 / 0.2234 | 195,742 |
| winsor fq+gq | RI 0.384 | **+7.51968e-05** / 2.90052e-04 / 0.796 | 248,024 |
| winsor 3pw | RI 0.384 | **+7.48156e-05** / 3.40256e-04 / 0.8265 | 247,582 |
| RI flow raw | b 8.899634e-04, ri_p 0.36913, n_fq 120,123 | **NOT PRODUCED** | — |
| RI flow winsor | b 6.485671e-04, ri_p 0.38412 | **NOT PRODUCED** | — |
| Flow decomposition (6 cells) | flow_diff winsor 6.6515e-04/0.4135; flow_common winsor 3.3140e-05/0.9682; flow_r winsor 3.8059e-04/0.6883; flow_diff raw 8.8996e-04/0.3691; flow_common raw 1.1351e-04/0.9022; flow_r raw −6.1419e-05/0.9894 | **NOT PRODUCED** | — |
| `run_flow_decomp_step3.do` CRVE companion | — | **NOT RUN** | — |

β₂ (us_cn): r1 −1.4336e-02 (p 0.3671); r2 +1.2669e-03 (p 0.91); r3 −6.8353e-04 (p 0.5175); held-only fq+gq −2.4232e-02 (p 0.333); held-only 3pw −3.7339e-03 (p 0.823); winsor fq+gq −2.9582e-04 (p 0.757); winsor 3pw −2.2734e-04 (p 0.747). R²: 0.8142 / 0.8200 / 0.6179 / 0.8142 / 0.8214 / 0.6385 / 0.6425. Winsor changed 2,480 obs per tail. Held-only dropped 26,011 extensive-margin zeros then 25,963 unpaired.
Panels: observed 582,463 cells / 11,732 firms / 100 quarters / 1999Q1–2023Q4; NONUS 369,418 / US 213,045. c6 ownership panel 248,024 rows / 5,191 firms / 82 quarters, balanced 124,012 each side. Flow mean 1.016e-02, sd 3.674, p1 −0.1364, p99 0.1387, kurtosis 198,984.5, max 1724.924, min −0.9727538. dos mean −4.373e-05, sd 4.175e-02; os mean 0.0754749.
B9: 16 rows across `ownshare_vce_diag.csv`, `flowheldonly_vce_diag.csv`, `flowwinsor_vce_diag.csv` — **zero se_valid=0**. Non-PSD/CGM warnings on r1 and winsor_fq+gq.
**VERIFY VERDICT: DISCREPANT (numbers CONFIRMED, process claim FALSE).** r1 (−3.3293406870e-02) and r2 (−3.6142844738e-02) re-derived to 11 sig figs; winsor 3pw to 4 sig figs (residual gap is the Stata-winsor vs numpy-clip percentile definition); kurtosis 198,988 / sd 3.6738 / max 1724.924 / p99 0.1387. The f6 agent's claim that `run_ri_flow.py` PID 5699 was "alive and healthy" is wrong — the process is dead, the log holds only pandas warnings, and `ri_flow_results.csv` is still 2026-08-03 11:09. The instruction not to relaunch is now void; relaunch is required.
**Open issues:**
1. **Steps 6–8 owed:** `run_ri_flow.py` → `run_flow_decomposition.py` → `run_flow_decomp_step3.do`, strictly ordered. Steps 1–5 artifacts are current.
2. **STALE ARTIFACTS ON DISK — DO NOT CITE:** `ri_flow_results.csv` (08-03 11:09), `ri_flow_decomposition.csv`, `flow_decomposition_diag.csv`, `flow_decomposition_panel.parquet/.dta` (all 08-03 10:09), `flowdecomp_results.csv` and `flowdecomp_vce_diag.csv` (08-03 10:10). Reading any of these reproduces exactly the stale-artifact headline error logged 2026-06-08.
3. **Fat-tail pathology much worse post-P0** (kurtosis ~5,000 → 198,985). The raw-flow CRVE column must not be primary.
4. **B9 vintage flip:** the `dos` spec is no longer degenerate. The vintage-dependence warnings in the `.do` comments now point at the wrong spec.
5. `build_ownership_share_panel.py` line 204 still writes `ref_main_panel_firms=6854` into `ownership_share_diagnostics.csv`; the canonical is 6,867. Not edited (print/CSV field, not an assert).
6. **Universe shift:** 5,191 firms vs 6,867 canonical. Re-check whether the coverage gap moved with P0 before writing the §7.6 sample-comparison sentence.

---

### f7 — Extensive margin (§7.10, A1/G1)
**STATUS: null-held on CRVE, but PROVISIONAL — no RI arbiter delivered. Watch item survives, weakened.**

Table as §2/§7.10 above. Additional cells:

| Spec | Post-P0 β₃ / se / p | N (firms) |
|---|---|---|
| d_breadth it+gt | +1.4756e-05 / 3.6124e-05 / 0.6840 | 348,156 (6,867) |
| d_nh it+gt | −1.3584 / 1.0097 / 0.1823 | 348,156 (6,867) |
| **exit it+gt (WATCH)** | **−5.0551e-03 / 2.4673e-03 / 0.0437** (pre-P0 p=0.006) | 201,964 (3,898) |
| init it+gt | −2.1752e-03 / 2.7593e-03 / 0.4328 | 85,880 (3,700) |
| d_breadth 3pw riskset | +2.468393e-05 / 5.597988e-05 / 0.660429 | 270,386 (5,420) |
| d_breadth it+gt riskset | +2.1540e-05 / 5.2508e-05 / 0.6827 | 270,796 (5,625) |
| d_nh 3pw riskset | −2.053915 / 1.513384 / 0.178498 | 270,386 |
| d_nh it+gt riskset | −1.8825 / 1.4625 / 0.2017 | 270,796 |
| exit 3pw riskset | +1.415490e-03 / 3.715630e-03 / 0.704233 — **bit-identical to full sample** | 201,634 |
| exit it+gt riskset | −5.0551e-03 / 2.4673e-03 / 0.0437 — **bit-identical** | 201,964 |
| init 3pw riskset | −1.986981e-02 / 2.269067e-02 / 0.383825 — **THIN** | 7,824 (1,027) |
| init it+gt riskset | −1.9183e-03 / 7.0899e-03 / 0.7874 — **THIN** | 8,518 (1,374) |

Panel build: 2,558,200 rows = 12,791 firms × 100 quarters × 2 groups; d_nh non-null 2,532,618; d_breadth non-null 2,519,827; breadth non-null 2,545,409; held_lag missing 25,582 (series starts); rows with cn_lag & shock 348,156 (exit 232,120, init 116,036). All structural asserts (a)–(h) OK; 2,000-row spot-assert vs grid OK; `.dta` 165.9 MB.
B9: `extmargin_vce_diag.csv`, **32 rows, zero se_valid=0**.
**VERIFY VERDICT: DISCREPANT (numbers FULLY CONFIRMED, process claim FALSE).** d_breadth 3pw β₃ re-derived at +2.1971842697e-05 and us_cn at −1.6961934111e-05 (exact), N 347,690. All descriptives confirmed to the last digit against `extmargin_descriptive.csv` and `extmargin_fourcell.csv`. `run_ri_extensive.py` PID 19300 is dead; `ri_extensive_results.csv` is still 2026-08-04 00:39, i.e. pre-07:56 rebuild. (Note: the parquet's 14:42:14 mtime post-dating `extmargin_results.csv` at 14:40:53 is a benign write-order artifact, not a vintage mismatch — the parquet reproduces the .dta-based Stata run exactly.)
**Open issues:**
1. **RI arbiter missing for all four DVs.** Pre-P0 arbiter values, for reference: d_breadth 0.768, d_nh 0.549, exit 0.695, init 0.805. Relaunch `run_ri_extensive.py`.
2. `run_ri_extensive.py` has no `flush=True`, so a redirected run produces an empty log whether alive or dead. Add `python -u` on the relaunch (not changed here — no-edit rule).
3. **WATCH ITEM:** the exit it+gt cell survives negative-significant but weakens to p=0.0437, it is an it+gt *companion* not the primary, the 3pw primary has the OPPOSITE sign (+1.4155e-03, p=0.704), and RI covers the 3pw collapse only — so this cell has no design-based check at all.
4. exit riskset variants are **bit-identical** to their full-sample twins (samp_exit==1 implies risk1==1). Do not present as independent robustness.
5. init riskset cells (N 7,824 / 8,518) are underpowered; label uninformative.
6. The `n_active` breadth denominator shows sawtooth Q1/Q3 collapses in both groups (US ~2,400 → ~700–800 through 2000–2004; NONUS through 2013). P0 recovers weekend-batch funds but the odd/even reporting pattern persists, so d_breadth levels in 1999–2005 are denominator-driven.
7. The motivating zero-holder gap must specify its denominator: +12.23pp full grid vs +15.18pp estimable subset.

---

### f8 — Four-group active/passive (§7.8)
**STATUS: the family's only near-significant column is gone. RI NOT delivered.**

Coefficient table as §7.8 above. Additional detail: t_cn (two-way term) m_act_full +1.09612e-06 (p 0.8704); m_act_post −4.79750e-06 (p 0.5262); m_pas_full +6.30374e-06 (p 0.5580); m_pas_post −5.61864e-06 (**p 0.0742**, marginal — noted for completeness, carries no claim); m_usi_full −1.82558e-06 (p 0.7580); m_usi_post +2.51034e-06 (p 0.6267). R² 0.5709 / 0.7314 / 0.5372 / 0.7795 / 0.6485 / 0.8297. Root MSE 0.0004 / 0.0001 / 0.0007 / 0.0001 / 0.0003 / 0.0001. 466 singletons (full) / 622 (post-2018).
Builder: 696,312 rows / 6,867 firms / 82 contiguous quarters; recon-A max_abs_dev 0.000e+00 and max_rel_dev 0.000e+00 over 199 (side,quarter) cells (no join fan-out); recon-B max|Σw−1| = 1.55e-15 over 395 books, held-but-null cells 0; firm universe 12,791; grid 5,116,400 = 12,791×4×100; grp_agg 1,343,212 cells (ACTIVE 547,157 / PASSIVE 308,446 / UNKNOWN 487,609); funds master 106,965 fund_id; exposure_map 174,078; shock_map 100. Comparability check: 6,867 firms / 82 quarters, EXACT match to c6.
Coverage diagnostic (replacing the sum-to-1 assert): Σ_firm(w) per (grp,quarter) on the cn_lag-filtered panel: min 0.2177, p50 0.8362, max 0.9338 — all <1 as expected, since w is normalized on the full 12,791-firm grid.
Passive-share anchor (recomputed independently, not coded in the builder): US 2021Q4 = **0.3975** pooled / 0.4187 labelled-only (documented anchor 39.78%); NONUS 2021Q4 0.1878 / 0.2148; US 2010Q4 0.1316 / 0.1397; NONUS 2010Q4 0.0721 / 0.0883. Series in `output/_rerun_fourgroup_passive_share.csv`.
Matched-share decay: US join share 0.995 (1999) → 0.917 (2023), labelled 0.916; 2018 0.9926, 2019 0.9816, 2020 0.9656, 2021 0.9535, 2022 0.9487. NONUS 2023 join 0.845, labelled 0.836. Series in `output/fourgroup_matched_share.csv`.
B9: `fourgroup_vce_diag.csv`, 12 rows, zero se_valid=0. Non-PSD/CGM warnings on m_pas_full and m_usi_full — se_valid only tests se>0 and non-missing, so it cannot flag those.
**VERIFY VERDICT: DISCREPANT (numbers FULLY CONFIRMED, process claim FALSE).** Both MAIN columns re-derived to 12 sig figs (m_act_full −7.0646789350e-07 with t_cn +1.0961181867e-06 at N 347,690/466 singletons; m_act_post +5.0479192880e-07 with t_cn −4.7974964527e-06 at N 183,880/622 singletons). `run_ri_fourgroup.py` PID 29128 is dead; `ri_fourgroup_results.csv` is still 2026-08-03 00:02.
**Open issues:**
1. **RI not delivered.** Relaunch `run_ri_fourgroup.py`. Pre-P0 values captured before any overwrite (above) so the comparison is not lost.
2. `run_fourgroup.do` line 42 dilution-benchmark comment still cites the pre-P0 pooled +2.746e-06; the live value is −5.279916e-07. Doc-pass item.
3. `run_ri_fourgroup.py` lines 188–190 hardcode the pre-P0 Stata anchors and will print an "investigate if far off" mismatch — that mismatch IS the result.
4. `run_fourgroup.do` coverage comment cites "~6,854 of 12,743 universe firms"; the true figures are 6,867 of 12,791.
5. The "passive is the inert arm" framing must be restated: the passive post-2018 column flipped sign and is now the largest |β₃| in the family (still ns, p=0.177).
6. The "US 2021Q4 passive = 39.78%" anchor is NOT coded as a check inside `build_fourgroup_panel.py` (verified by grep). It was recomputed independently rather than assumed.
7. Menu contrasts (passive, US-internal) have no RI fold — RI covers the MAIN ACTIVE contrast only.

---

### f9 — Russia positive control (§7.9)
**STATUS: null-held under design-based inference. All eight documented figures moved; the qualitative verdict survives.**

Tables as §7.9 above. Additional detail: R1 β₂ (us_ru) +4.5391e-06 (se 2.9331e-06, p 0.1256); R2 β₂ +4.4750e-06 (se 9.2302e-06, p 0.6291); R3 β₂ +6.3460e-06 (se and p missing). R² 0.5902 / 0.5908. R1 clusters 6,867 firm / 82 rd_m; R2 6,634 / 82 (466 singletons). RI: R1 200,000 perms, n_fq 174,078; R2 5,000 perms with DEMEAN_ITERS=30, n_fq 174,078.
LP full table (ru_lag fixed at 2021Q4, base w_2021Q4, HC1 + 20,000-perm firm-level permutation, n=4,294 firms per horizon, LP panel 204,656 → 68,704 rows after dropna):

| h | β | HC1 se | t | perm_p |
|---|---|---|---|---|
| 0 | −2.52992e-05 | 3.98951e-05 | −0.6341 | 0.4004 |
| 1 | −4.03577e-05 | 3.99596e-05 | −1.0100 | **0.2703** |
| 2 | −3.14725e-05 | 3.91750e-05 | −0.8034 | 0.4521 |
| 3 | −4.37919e-05 | 4.03389e-05 | −1.0856 | 0.3042 |
| 4 | −3.39984e-05 | 4.13688e-05 | −0.8218 | 0.4826 |
| 5 | −4.28936e-05 | 4.11144e-05 | −1.0433 | 0.3778 |
| 6 | −4.18559e-05 | 4.17399e-05 | −1.0028 | 0.4375 |
| 7 | −3.76049e-05 | 4.14998e-05 | −0.9061 | 0.4536 |

Panel: 348,156 rows (US 174,078 / NONUS 174,078), 6,867 firms, 82 contiguous quarters, ru_lag>0 share 0.0520, all asserts passed. Grid: 2,558,200 cells; EU entity universe 12,791 (0 dual-listed); crosswalk 10,212; non-NULL delta_w 2,519,827; interior NULL delta_w = 0. HIGH-exposure cutoff 0.037; winsor bounds p1 −0.00086062 / p99 +0.00100374; corr((dUS−dNONUS), Shock) on HIGH-lag firms −0.0681 raw / −0.0663 winsorized over 81 quarters. Composition: US held/HIGH 2,821, US zero-filled/HIGH 1,611, NONUS held/HIGH 3,456, NONUS zero-filled/HIGH 976.
B9: `russia_headline_vce_diag.csv` — **r3/us_ru_post se_valid=0**, the single such row in the entire ten-file corpus, expected and pre-registered. r1 and r2 se_valid=1.
**VERIFY VERDICT: CONFIRMED.** 3pw re-derived at −3.8687606110e-06 (us_ru +4.4750089084e-06) and it+gt at −3.8418733610e-06 (us_ru +4.5390732647e-06); both match the RI CSVs to 11 digits. Frozen exposure inputs confirmed by mtime.
**Open issues:** (a) stale pre-P0 print-only anchors in `run_ri_russia.py` line 79 and `run_ri_russia_3pairwise.py` line 80; (b) pre-P0 vintage-warning headers still on `06_russia_grid.jl`, `build_russia_c6_panel.py`, `build_russia_lp_panel.py` — flag for the vintage register; (c) the §7.9 doc table and §13 rows are stale on all eight figures.

---

### f10 — Audit consumers + 07d/07e + cum4 hardening + DDD gate
**STATUS: LP significance gone; cum4 rejection dissolves on RI but the WCB disagrees. Gate constants updated and independently verified.**

STEP 1 (`run_audit_f1f2f7.do`):

| Cell | Pre-P0 | Post-P0 β₃ / se / p / RI | N |
|---|---|---|---|
| it+gt headline check | +2.081e-06 / 0.151 | −1.0727e-06 / 1.8191e-06 / 0.5570 / 0.5984 | 348,156 |
| F1a lead full | +4.85e-07 / 0.774 | −3.1768e-07 / 1.19e-06 / 0.7907 / 0.8775 | 338,076 |
| F1a lead in-span | — | −4.02e-07 / 1.85e-06 / 0.8285 | 265,904 |
| F1b LP h0 | +2.081e-06 / 0.151 | −1.07e-06 / 1.82e-06 / 0.5570 / 0.5984 | 348,156 |
| F1b LP h1 | +2.85e-06 / **0.017** | −1.2475e-06 / 2.36e-06 / **0.5989** / 0.5260 | 338,076 |
| F1b LP h2 | +5.03e-06 / **0.003** | +3.8475e-07 / 2.02e-06 / **0.8491** / 0.8647 | 328,366 |
| F1b LP h3 | — | +9.2787e-08 / 2.32e-06 / 0.9681 / 0.9711 | 318,812 |
| F1b LP h4 | +5.24e-06 / **0.035** | +3.1026e-06 / 1.89e-06 / **0.1045** / 0.2962 | 309,512 |
| F2 dw in-span | — | −1.5656e-06 / 2.81e-06 / 0.5788 / 0.6093 | 272,140 |
| F7 GPR full | 0.63 / 0.88 | us_cn_gpr −2.20e-06 (1.43e-06, 0.1262); gprlag +9.51e-07 (1.01e-06, 0.3496) | 348,156 |
| F7 GPR in-span | — | −3.37e-06 (2.19e-06, 0.1277); +1.47e-06 (1.54e-06, 0.3426) | 272,140 |

STEP 2 (`run_audit_f1b_robust.do`): quarter-cluster CRVE h1 −1.25e-06 / 2.49e-06 / **0.6178** (was 0.0166); h2 +3.85e-07 / 2.27e-06 / **0.8656** (was 0.0025); h4 +3.10e-06 / 2.57e-06 / **0.2302** (was 0.0349). **boottest WCB MISSING at all three horizons** — Mata error 3900, "unable to allocate real <tmp>[169038,10000]" / [164183,10000] / [154756,10000]; r(p) and CI returned missing. Known, documented OOM.
STEP 3 (`run_randomization_inference.py`, S2 RI): headline dw h0 −1.0727121277103862e-06 / 0.5984120079399603 (n_fq 174,078); lead flow −3.1767652198450873e-07 / 0.8774856125719371 (169,038); LP cum h1 −1.2475407639089605e-06 / 0.5259973700131500; cum h2 +3.847467624279218e-07 / 0.8647056764716177 (was 0.060); cum h3 +9.278663574037999e-08 / 0.9711151444242779; cum h4 +3.1026446876682723e-06 / 0.2961835190824046; in-span headline −1.5656365363050556e-06 / 0.6092569537152315; in-span cum h1 −1.7551730465687068e-06 / 0.5565472172639137; in-span cum h4 +4.876228330473203e-06 / 0.2705286473567632.
STEP 4 (07d): spec0 no-FE −9.2863e-08 / 5.0321e-07 / 0.8540, R²=0.0000, F=0.4625 p=0.6314; spec1 headline −1.0727e-06 / 0.5570, R²=0.590162, F=0.2884 p=0.7502 (was +2.081e-06 / 0.151 / R² 0.6235 / F 1.31); spec4 weak-FE +5.2406e-07 / 6.9523e-07 / 0.4532, R²=0.005103 (was +5.38e-07 / 0.411 / R² 0.0073), CGM-repaired.
STEP 5 (07e): A0 −1.0727e-06 / 0.5570; A1 firm×group FE −5.2799e-07 / 1.7432e-06 / 0.7627 / RI 0.8014, N 347,690; B1 k=1.645 −2.7150e-05 / 1.7704e-05 / 0.1290 (6 treated quarters, 40,914 rows); B2 k=2.0 −2.3023e-05 / 2.3014e-05 / 0.3201 (4 quarters, 29,010 rows); B3 k=3.0 −6.0506e-05 / 3.2898e-05 / 0.0696 (2 quarters, 14,586 rows). Shock distribution sd 2.57789, mean 0.16695, 82 quarters.
STEP 6 (cum4 hardening): full table in §15 above. Gates PASS at relerr 3.28e-09 / 1.26e-08 / 5.14e-09.
STEP 7 (DDD gate): Stage A gate β₃ = −5.2799e-07 vs target −5.279916e-07, N 347,690 → **GATE PASS**. Stage B no-FE full factorial: coefficients only (us_cn_shock −1.03e-06, us +3.30e-07, cn_lag −1.85e-06, shock −1.19e-08, us_cn −1.22e-06, us_shock −4.79e-08, cn_shock +1.09e-06, cons −1.01e-06); **every SE, t, p, CI and the F statistic are MISSING** — singular/nonsymmetric variance matrix. NO valid inference from that column. Stage C1 bilateral (no BIL control): −3.88e-07 / 1.98e-06 / 0.845, N 324,340 (414 singletons, 6,119 firm clusters). Stage C2 + US×BIL: −5.23e-07 / 1.98e-06 / 0.792, with us_bil +2.91e-05 / 1.67e-05 / 0.084.
**VERIFY VERDICT: CONFIRMED.** Gate integrity verified end to end: the constants in `run_cum4_inference.py` and `run_ddd_nofe_bil.do` equal `headline_3pairwise_canonical.csv` and `audit_ri_3pairwise.csv` verbatim; `git diff` shows ONLY constants and comments changed (the reghdfe line, the 0.05e-06 tolerance and `exit 459` are untouched) and only those two files are modified in the repo. The gate value was independently re-derived at −5.2799160827e-07. Every result artifact matched verbatim, including all nine S2 RI rows, all five F1b LP horizons in `audit_f1_lp_irf.dta`, and every cell of `cum4_inference_hardening.csv`.
**Edits made (constants only, verified before editing):** (1) `run_cum4_inference.py` — B3_GATES → {h0 −5.2799161e-07, cum1 −1.5346165e-07, cum4 +6.6006275e-06}, P_FREE_ANCHORS → {0.8014, 0.9368, 0.0636}, cross-check print block updated, pre-P0 values retained inline as "superseded". (2) `run_ddd_nofe_bil.do` — Stage-A gate target +2.746e-06/N=347,490 → −5.279916e-07/N=347,690; tolerance and fail-safe untouched.
**Open issues:**
1. **boottest OOM** at all three LP horizons. The score-based WCB in `run_cum4_inference.py` is the working substitute and did run (9,999 Webb draws).
2. **DDD Stage B carries no valid inference** — singular variance matrix, all SEs missing.
3. **Stale input:** `output/firm_ladder_panel.dta` is 2026-07-22, PRE-P0, and supplies the `bil_us_c` lookup for Stage C. The merge left 204 master rows unmatched and 23,402 `us_bil` values missing, defining the 324,340-row bilateral subsample. `bil_us_c` is exposure-side (bit-frozen) so its VALUES should be unchanged, but its firm-quarter COVERAGE has not been re-verified against the P0 rebuild. **This is the only pre-P0 input still feeding a live estimate anywhere in the battery.**
4. **No `*_vce_diag.csv`** from any of the seven scripts in this family; B9 N/A. The Stage-B all-missing-SE result is the substantive equivalent.
5. **cum4 arbiter disagreement** — see §15. Needs an explicit adjudication sentence.
6. Non-PSD/CGM repairs in F7 in-span and 07d spec4.
7. Stale header anchors deliberately left unedited across `run_audit_f1b_robust.do`, 07d and 07e. Every one is now a mismatch; that is the result.

---

## 4. CONSOLIDATED UNRESOLVED ISSUES

### 4.1 BLOCKING — must be run before the doc refresh can be called complete
1. **Relaunch `run_ri_flow.py`** (f6 step 6), then `run_flow_decomposition.py` (step 7), then `run_flow_decomp_step3.do` (step 8). Strictly ordered. Six decomposition cells plus the raw/winsor flow RI are owed.
2. **Relaunch `run_ri_extensive.py`** (f7). No RI arbiter exists for any of the four extensive-margin DVs. Use `python -u` — the script has no `flush=True`, so the log is uninformative either way.
3. **Relaunch `run_ri_fourgroup.py`** (f8). No post-P0 RI for the MAIN active contrast.
All three earlier launches are DEAD (verified: only `python.exe` PID 880, the stata-mcp helper, is running; all three logs are 446 B of pandas warnings; all three CSVs carry pre-P0 mtimes).

### 4.2 DATA-INTEGRITY / VINTAGE
4. `output/firm_ladder_panel.dta` (2026-07-22) is pre-P0 and feeds DDD Stage C. Coverage not re-verified. Either rebuild it or footnote the 324,340-row subsample as provisional.
5. Six stale artifacts on disk that will silently produce wrong headlines if read: `ri_flow_results.csv`, `ri_flow_decomposition.csv`, `flow_decomposition_diag.csv`, `flow_decomposition_panel.{parquet,dta}`, `flowdecomp_results.csv`, `flowdecomp_vce_diag.csv` — all 2026-08-03. Consider moving them to a `_preP0/` subdirectory.
6. `ri_extensive_results.csv` (2026-08-04 00:39) LOOKS same-day but pre-dates the 07:56 holdings rebuild. Date alone is not a freshness test here.

### 4.3 REAL DEFECTS IN WRITTEN OUTPUT
7. `tercile_results.csv` addnote hardcodes "MAIN drops 462 singletons"; the true count is **466**. Independently confirmed. Fix before quoting.
8. `ownership_share_diagnostics.csv` carries `ref_main_panel_firms=6854`; the canonical is 6,867.
9. **`shockmenu_vce_diag.csv` parsing trap:** its schema is `variant,dv,fe,coef,b,se,se_valid,smoke,n_firms` — `se_valid` is column 7, NOT the last. A naive last-column parse falsely flags all 56 rows as `se_valid=0`. Parsed correctly, all 56 are `se_valid=1`.

### 4.4 METHOD GAPS (would require new code — none written)
10. **No RI anywhere in f5** (risk set, country-pair, spell boundary): 8 specs with no design-based arbiter.
11. **No RI for the f1 INDICATOR spec** — precisely the spec that needed the CGM non-PSD repair.
12. **No RI for f3's 3pw h=0 cell** (p=0.0939, the family's lowest) — RI covers only the it+gt collapse.
13. **No RI for f7's exit it+gt WATCH cell** (p=0.0437) — RI covers only the 3pw collapse, where the sign is opposite.
14. **No RI for f8's menu contrasts** (passive, US-internal) — MAIN ACTIVE only.
15. **No `*_vce_diag.csv`** in f3, f4, f5 or f10. B9 is N/A, not "pass," for those families. This matters most for f3, where both D3 regressions hit the non-PSD condition a vce_diag file would flag.
16. **boottest WCB OOM** at LP h1/h2/h4.

### 4.5 REPRODUCIBILITY GAPS (numbers exist only in transient Stata logs)
17. f3's entire sagg Stata battery (all SEs, p-values, joint Wald, cumulative) — no results CSV.
18. f5's F4a/F4b/F4c and F8 SEs, p-values and small-cluster warnings.
19. f10's F1b quarter-cluster CRVE rows, the boottest OOM record, DDD Stage B/C, and the 07e shock distribution.
Point estimates for all of these were independently re-derived and match, but the inference figures cannot be re-checked from any artifact. Consider adding results-CSV writers before the next rerun.

### 4.6 INFERENTIAL / FRAMING ITEMS FOR THE DOC PASS
20. **cum4 disagreement:** four design-based routes ≥0.0558, one score-WCB at 0.0038. Adjudicate explicitly.
21. **sagg magnitude-vs-significance tension:** per-SD effect ~5× the headline's while staying null. Two readings are consistent; do not present magnitude growth as corroboration.
22. **f3 docstring contradicts its own diagnostic** (claims serially-uncorrelated innovations; prints corr = 0.357).
23. **Non-PSD/CGM repairs** occurred in: f1 indicator, f2 MAIN, f3 both D3 specs, f4 S_{t−1} 3pw, f5 F4c, f6 r1 and winsor_fq+gq, f8 m_pas_full and m_usi_full, f10 F7 in-span, 07d spec4, and DDD Stage B. `se_valid` cannot detect this condition. Disclose alongside any cited CRVE p from those specs.
24. **Stale in-script anchors and header comments** across roughly a dozen files were deliberately left unedited per the rerun rules (only the two gate files were touched). They will print apparent mismatches. Schedule a separate cleanup pass so a future reader is not misled.
25. **Framing rewrites owed:** §7.7 buy-side tilt, §7.6 positive/significant flow, §7.8 "passive is inert," §7.5 marginally-positive country-pair b₁, §7.4 risk-set-as-corroboration, §7.10 which zero-holder denominator.

## ADDENDUM (2026-08-04, main session): the four dead RI legs, relaunched and landed

verify caught f6/f7/f8's RI jobs dead (agents exited believing them launched;
zero processes, pre-P0 CSVs). Relaunched sequentially by the main session:

- ri_fourgroup (P0): full -7.06e-7 RI 0.780; post2018 +5.05e-7 RI 0.798.
  BOTH deep null; passive-dilution rejection stands (active ~ pooled ~ 0).
- ri_extensive (P0): arbiter (circular) d_breadth 0.573 / d_nh 0.220 /
  exit 0.683 / init 0.829; riskset variants 0.207-0.756. ALL NULL.
- ri_flow (P0): winsor (MAIN) +7.48e-5 RI 0.913 - null. RAW -3.61e-2
  RI 0.021 - REJECTS, but see below.
- flow decomposition (P0) + step3 CRVE: winsor cells all null (flow_r 0.913,
  flow_common 0.641, flow_diff 0.833; CRVE matches RI verbatim via shipped
  winsor columns); RAW cells all ~0.02 with absurd magnitudes (flow_r b3 =
  -1.07 = -107% of float per unit CN*S; flow_common raw CRVE SE=0 DEGENERATE,
  B9-flagged); identity 3.6e-12.

### OPEN ISSUE (flagged, not resolved): raw-flow outlier explosion post-P0
The as-of rule admits firm-quarters with monster flow ratios (max 1724.9 x
float, kurtosis ~199k; pre-P0 max was 9.8). A handful of such cells drive all
three raw RI "rejections" (near-identical p's ~0.02 = same few quarters) and
crush the +0.20 co-movement anchor (winsor high-CN corr now +0.027,
stable-float +0.008). Winsor remains MAIN by the pre-P0 convention (fat-tail
lesson), and all winsor inference is null; but §7.6's estimand prose leaned on
corr ~ +0.20 - that anchor DID NOT SURVIVE P0 and the prose must be rewritten.
TODO: identify the monster cells (likely stale tiny lagged float meeting
as-of-recovered holdings) and decide a predetermined guard (BAD-rule extension),
NOT outcome-driven trimming. Until then: winsor-MAIN + raw-disclosed-as-
outlier-dominated is the honest reporting.

### cum4 WCB-vs-RI tension (open)
P0 cum4: free 0.066 / circ 0.122 / moving-block ~0.107 / FWER 0.056-0.062 all
null; score-based quarter-cluster WCB 0.0038 rejects. WCB assumes cross-quarter
independence - exactly what the measured serial correlation violates; RI stays
the arbiter by convention, but the number must be disclosed with that frame.
