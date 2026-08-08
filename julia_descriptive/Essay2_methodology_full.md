# Essay 2 — Full Methodology, Replication Log, and Honest Caveats

**Purpose.** This document is a complete, deliberately unflattering technical record of Essay 2, written to be handed to an independent reviewer (human or AI). It states the research design, every variable and how it is built, the exact code and commands, the data-cleaning decisions, all estimation results, and every caveat, limitation, and mistake we found and corrected. Nothing is hidden. Where the design is weak, underpowered, or where an earlier build was wrong, it is flagged explicitly.

**One-line status.** The headline coefficient is **statistically indistinguishable from zero** under the preferred backward-timing specification with the fully-saturated **three-way pairwise FE** (firm×quarter + group×quarter + firm×group; see "Second external review round"). On the rebuilt **P0 holdings panel** (the as-of W=10 snapshot rule, §5.0) β₃ = **−5.279916×10⁻⁷**, SE 1.743166×10⁻⁶, CRVE p = **0.762748**, N = **347,690**; design-based randomization inference gives free-permutation p = **0.8154** and circular-shift p = **0.8293**. The it+gt companion is −1.072712×10⁻⁶ (p = 0.557028, N = 348,156, RI 0.5984).

Two claims died in this cycle. Both are retracted here, not renumbered. First, the pre-P0 headline (+2.746×10⁻⁶, p = 0.110, RI 0.31) is superseded: `03_eom_etl.jl` kept only rows stamped exactly on the calendar quarter-end, which silently dropped about 40% of funds whenever a quarter ended on a weekend, and the loss was US/NONUS-asymmetric (US-share selection tilt −3.41pp before the fix, +0.01pp after). Second, and more consequential for the narrative, **the "every S_t point estimate is positive" pattern was itself that calendar artifact**. Post-P0 the signs are mixed and the magnitudes are tiny. Everything is null.

The null then holds three levels deeper. A 10-construction shock menu (§7.3d) puts zero of 12 randomization-inference columns below 0.05; the two CRVE crossings dissolve under the arbiter (min RI p = 0.2439). All ten estimation families rerun on the P0 panel are null. Nine of them carry a design-based RI arbiter and every one clears it, including the extensive margin (§7.10, arbiter 0.21–0.83); the risk-set / country-pair / spell-boundary family has no permutation arbiter at all (eight specs, §7.4) and is null on CRVE alone. Five further individual cells also lack an arbiter, none of them rejects, and each is named at the point of use (§9, limitation 13b). The one pre-registered free-permutation rejection, LP cum4, is now null on every design-based route (free 0.0664, circular 0.1220, moving-block 0.1074/0.1076, within-family max-|t| FWER 0.0558/0.0618); a score-based quarter-cluster wild bootstrap still rejects at 0.0038, and that disagreement is disclosed rather than buried (§ "Cum4 inference hardening"). The shares-based ownership test (§7.6) is null on its winsorized MAIN outcome (β₃ = +7.48×10⁻⁵, CRVE p = 0.83, RI p = 0.91); the raw companion is dominated by a handful of post-P0 outlier cells and is flagged as an open investigation, not a result. An earlier significant estimate came from a forward-window specification vulnerable to post-treatment timing contamination, and is retracted (§8). The project's current contribution is a measurement/design one (the panel + identification), and the empirical question is treated as open, with a planned pivot to a firm-level price channel.

---

## Change log for this review round (read first)

This document was hardened over one working session in response to two external AI reviews plus several rounds of internal adversarial multi-agent review. Everything below is reflected in the body (§1–§13) and in the source code in the repo (`github.com/fidio728/practice`, branch `essay2-code-review`, under `julia_descriptive/`; see the §14 code inventory and §15 result→source map). This section states **what changed and why**, so a new reviewer can see the full provenance. No data was fabricated or altered; every number cited was re-derived from the on-disk parquet/dta/csv files.

### 0. P0 (2026-08-04) — the holdings as-of rebuild, and the two claims it killed

This is the most recent and the most consequential change. Read it before anything below; every section marked "(pre-P0: …)" carries its superseded value for traceability. The story runs in five steps.

**1. The defect and the fix.** `03_eom_etl.jl` kept only holdings rows whose `REPORT_DATE` equalled the exact calendar quarter-end. FactSet shifts report dates to the prior business day when a quarter-end falls on a weekend, so weekend quarter-ends silently dropped roughly 40% of funds (2022-12-31, a Saturday: 58.4% coverage against 86–87% on weekday quarter-ends; 2023-09-30, also a Saturday: 38.6%). The loss was not random across groups. The dropped Friday batch was 26% US against 21% US in the kept Saturday batch, an asymmetry group×quarter FE cannot absorb, so it fed straight into β₃. The fix is a per-(fund, security) as-of rule: take the latest report on or before the quarter-end within a W=10 day window, stamp it to the quarter-end, and keep `report_date_actual` plus `asof_gap_days` (§5.0). The new holdings panel is 208,418,523 rows, +11.57%; the old panel is exactly its gap = 0 subset and is bit-equal on those rows.

**2. The headline moved, and the null deepened.** 3-pairwise β₃ went from +2.746×10⁻⁶ (p 0.110, RI 0.31) to **−5.279916×10⁻⁷ (p 0.762748, RI free 0.8154 / circular 0.8293)**, N 347,690 with 466 singletons dropped. The it+gt companion went from +2.081×10⁻⁶ (p 0.151) to −1.072712×10⁻⁶ (p 0.557028), N 348,156. The core conclusion is unchanged. What changed is the sign story: **every prior sentence asserting that all S_t point estimates were positive is dead**, because that uniform positive lean was substantially the weekend-composition artifact (weekend quarters under-counted US holdings, so rebound quarters read as US buying). Signs are now mixed and tiny, and every family is null.

**3. The shock menu closes P1a/P1b/P1c.** Ten shock constructions across three direction variants, selected by a pre-registered whiteness rule written before any regression ran (§7.3d). The winner is `D_q_ar1_nla` (quarterly-mean GPR, AR(1) at quarterly frequency, fit through 2023Q4, LB(4) p = 0.354). Every column is null under the circular-shift arbiter, minimum p = 0.2439. The span-aligned variants lean negative, that is the H2.1 direction, and two of them cross CRVE 0.05 (0.026 and 0.031); randomization inference kills both. Variant B-i proves the full-sample look-ahead (F7) was innocuous, so the defect was the AR specification, not the look-ahead. The F7 exact-nesting item deferred earlier is **superseded by the menu**. The winner is rule-dependent and must be quoted that way: LB(4)-first picks D, C-ii is whiter at LB(8) (0.1319 against 0.0848), an |acf1|-first rule would have picked C-i, and the selected shock is not white.

**4. cum4 no longer rejects on any design-based route, and one bootstrap disagrees.** On the P0 panel: free 0.0664, circular 0.1220, moving-block 0.1074/0.1076, within-family max-|t| FWER 0.0558/0.0618. The score-based quarter-cluster WCB reads 0.0038 and rejects. That bootstrap assumes cross-quarter independence, which the measured serial correlation of S_t violates (acf1 +0.271, Ljung–Box Q(1) p = 0.013), so RI stays the arbiter by convention. The WCB number is disclosed, not hidden.

**5. One open data issue.** The as-of rule admits firm-quarters with monster raw-flow ratios (max 1724.9× float, kurtosis ≈ 199,000, against a pre-P0 max of 9.8). A handful of such cells drive all three raw-flow RI "rejections" at ≈0.02, and they broke the +0.20 co-movement anchor that §7.6's estimand paragraph leaned on (winsorized high-CN correlation is now +0.027, stable-float +0.008). Winsorized remains MAIN by the pre-existing fat-tail convention and all winsorized inference is null. The raw column is disclosed as outlier-dominated. The guard will be a predetermined BAD-rule extension, not outcome-driven trimming, and it is not written yet.

### A. Correctness and honesty fixes to the existing w-based pipeline (applied + verified)

1. **Universe wording downgraded to "holdings-observed European issuer universe"** (§3 title, §3 Caveat 2, §5.2, §7.4). The firm universe is every `sec_entity_id` that ever appears with a European `sec_country` in the FactSet Ownership holdings panel — **not** the full FactSet Security Coverage listing universe. Estimand phrasing corrected accordingly. This does not change β₃; it makes the claim honest. (Definition of "European-listed": a security in FactSet Ownership whose `SEC_FIRM_ISO_COUNTRY` ∈ the 28 jurisdictions of §3, restricted to those held by ≥1 institution.)
2. **Attenuation claim softened** (§3 Caveat 2, §9 limitation 11). Rebuilding from the full Security Coverage universe is *expected* to attenuate β₃ toward zero (added zero-difference rows), but this is the mechanical expectation under a simplified argument, to be **verified by running the rebuild, not asserted** — corrected an earlier over-strong "would reinforce the null." Also corrected an earlier wrong claim that never-held firms "do not affect β₃" (they do, via attenuation).
3. **Retraction language softened** (§0 one-liner, §8). "Causally clean specification" / "look-ahead artifact" → "preferred backward-timing specification" / "forward-window specification vulnerable to post-treatment timing contamination." We do not claim to have decomposed *why* the SE tripled (§8).
4. **Firm-count cascade clause added** (§2): 12,743 European securities → 10,171 Revere-matched → 8,014 carry a non-null all-`rel_type` China share; under the B7 CUSTOMER+SUPPLIER restriction plus the one-quarter lag, **6,854** distinct firms carry a non-null lagged CN and enter the estimation panel (the pre-B7 all-`rel_type` in-panel count was 7,928). *(These are the pre-P0 endpoints. On the P0 panel the cascade runs 12,791 → … → **6,867**; see §2.)* The gaps are the documented MISSING bucket, the CUSTOMER+SUPPLIER restriction, and the one-quarter lag, not contradictory counts. Figures re-derived from `merged_us_eu_zero_filled.parquet` and `05_unmatched_profile_by_country.csv`.
5. **Revere match country structure disclosed** (§9 limitation 12): GB 32.78% unmatched (1,307 of 3,987), the worst among large countries; GB is 60.85% of the country-pair subsample, so the matched sample drops ~⅓ of GB issuers (selection concern).
6. **Hard asserts added to the build scripts** (`build_c6_panel.py`, `build_spell_riskset.py`): duplicate-key fail, full US/NONUS pairing per firm-quarter, one common shock per quarter, `cn_lag ∈ [0,1]`, contiguous quarter coverage, and a **NULL-aware group-quarter weight-sum** (portfolio weights sum to 1 within each group-quarter; an all-NULL cell is allowed only when the group's book is empty, `I_ict = 0`, and a held-but-all-NULL cell hard-fails as a 06 bug — 0 found). Verified: 199 non-empty group-quarters sum to 1 with max abs deviation 4.44×10⁻¹⁶; 82 contiguous quarters.
7. **Schema-collision fix**: `06_cartesian_grid.jl` and `plots/fig13_two_panel.py` both wrote `05_diff_us_vs_nonus_high_c6.csv` with different schemas. fig13 now writes its own `fig13_diff_{all,high}_c6.csv`; the orphaned `05_diff_us_vs_nonus_all_c6.csv` was deleted; the three similarly-named CSVs are documented in the §16 artifact map. No downstream reader breaks (grep-verified).
8. **`04_us_ownership_european.jl` duplicate-key check upgraded from `@warn` (>1,000) to a hard `error` (n_dup > 0)**. Measured 0 duplicate `(fund_id, fsym_id, report_date)` keys on the **pre-P0** 186,800,295-row `holdings_eom.parquet` *(superseded; the live panel is 208,418,523 rows and the as-of rule re-stamps `report_date`, so this check has **not** been re-measured on the P0 panel and none of the three P0 ledgers records a re-run)*. On that pre-P0 evidence the change was de-risked; it fires on a future ETL run only if a raw pull reintroduces duplicates. This matters because a raw fund-level duplicate would double-count dollars in `SUM(adj_mv)` and neither the final dedup assert (aggregated keys) nor the weight-sum check would catch it.
9. **Cosmetic**: corrected the `I_ict` Stata variable label (`build_country_pair_shock.py`) from "ICT industry indicator" to "Group holding value, USD"; `I_ict` is not used in any regression.

### B. New empirical work — the shares-based ownership test (§7.6), the "make-or-break"

The market-value portfolio weight `w = H(USD)/T(USD)` mixes trading flow with price moves and portfolio-denominator reallocation, so a null on `w` supports "US do not reduce their portfolio *weight*" but not "US do not *sell* / do not reduce their *stake*." To license the flow claim we built a **pure trading-flow** outcome (§7.6, F6): the change in group shares held over the *lagged* primary-EQ float, immune to price and to float changes (buybacks/issuance), on the same C6 grid, three-way pairwise FE, and clustering as the main spec.

**Result on the P0 panel (2026-08-04, current).** The MAIN column is the winsorized p1/p99 flow: β₃ = **+7.48×10⁻⁵** (3-pairwise), CRVE p = 0.8265, **RI p = 0.913 — null**. The raw column is −3.61×10⁻² with CRVE p = 0.2366 and RI p = 0.021, and it is **not** reported as a result: post-P0 the raw flow carries kurtosis 198,985 and a maximum of 1724.9× float, and the winsorized-versus-raw gap (a sign flip and a ~480× magnitude collapse) shows the raw point estimate is entirely a handful of tiny-lagged-denominator cells. See the §7.6 open-issue paragraph. *(Pre-P0, superseded: β₃ = +0.00089 raw with CRVE p = 0.0001 and RI 0.37 / 0.38; the CRVE significance there was already labelled a fat-tail artifact, and the P0 rebuild dissolves the point estimate as well.)* Two required caveats stand: ADR exclusion (US holds **8.79%** of European exposure off the primary class against **3.22%** for non-US; pre-P0 8.92% / 3.32%) and limited power against small adjustments. New files: `build_ownership_share_panel.py`, `build_ownership_share_c6_panel.py`, `run_ownership_share.do` (all in §14–§15).

### C. Adversarial multi-agent reviews run this round (all findings resolved)

| review id | scope | verdict |
|---|---|---|
| `wbo5chyw9` | the code fixes in A6–A9 | ready, 0 defects |
| `wutg4fnkw` | the weight-sum / quarter-coverage asserts, artifact map, doc wording | ready, 0 must-fix |
| `wqtl83l3b` | doc-scoped: every quantitative claim re-derived vs code/data; embedded code vs disk | ready, provenance verified, 0 must-fix |
| `wsu1zupkc` | the shares-based build | **needs_fix → caught a real bug** (first draft pooled all `fsym_id` classes and MODE-masked a dual-universe `shares_out`, 5.28% of cells dispersed up to 1e12×). Fixed by restricting to `fsym_id = fsym_primary_id`, which drives dispersion to exactly 0. |
| `w86zvcvsx` | the shares-based result | result_trustworthy, 0 must-fix; β₃ independently reproduced to ~2×10⁻⁷; null confirmed not a zero-fill artifact |

### D. Current conclusions (detail in §7, §8)

The headline β₃ (US × China-exposure × tension shock on within-Europe reallocation) is a **robust statistical null**, across FE choices, shock definitions (now including a 10-construction menu, §7.3d), sample constructions, the value-weighted portfolio weight, the shares-based ownership stake, **and the extensive margin (holder breadth, count, exit and initiation, §7.10, circular-shift RI 0.21–0.83)**. The extensive margin matters because discretionary divestment shows up as funds leaving the register before it shows up in dollars. It is null too, even though US institutions descriptively hold these firms more sparsely (a **+12.23 pp** firm-quarter zero-holder gap on the full grid that does *not* deepen with tension).

**On sign, the honest statement has changed.** Before the P0 holdings rebuild this document said every S_t point estimate on the main outcome was positive, that is, opposite to disengagement. That pattern was a calendar artifact of the pre-P0 quarter-end stamping (change log §0). Post-P0 the headline is negative and tiny (−5.28×10⁻⁷), the shares-based MAIN flow is positive and tiny (+7.48×10⁻⁵), the tercile dose bins are positive and tiny, the risk-set family is uniformly small-negative, the direction split has a negative sell arm and a positive buy arm, and every one of these is null. There is no sign pattern left to report. Do not resurrect either the "all positive" claim or its mirror.

The one historically borderline coefficient in the country-pair family (β₁) is now dead outright: −1.36×10⁻⁶ with p = 0.910 / 0.943 / 0.947 across the three clustering levels, so the clustering audit that used to discriminate no longer discriminates anything (§7.5). The earlier "significant" headline was a retracted forward-window/few-cluster artifact (§8), and the positive-sign pattern is now retracted alongside it. The project's contribution is currently a measurement/identification one (the paired firm×quarter / group×quarter panel, plus the as-of holdings rule of §5.0), and the ownership-flow question is treated as **open but with no detectable differential US disengagement at this design's resolution** (limitation 1; the Russia calibration in §7.9), with the economic action, if any, to be sought in a firm-level price channel (H2.2).

### E. Next steps (detail in §10)

In priority order: **(#0) the raw-flow outlier guard** — write a predetermined BAD-rule extension for the monster flow cells the as-of rule admits (§7.6 open issue), never an outcome-driven trim; **(#7) inference robustness** — the score-based quarter-cluster wild bootstrap is completed for the local-projection horizons (§6.3, F1/F10, `run_cum4_inference.py`) and the shock-timing/construction menu is **done** (§7.3d), so what remains open is leave-one-quarter-out; **(#6) exposure robustness** — `rel_type` filter, edge dedup, quarter-as-of counterparty country; **(#8) holder-level panel** — holder × quarter FE, the true Khwaja–Mian bank×time analog; **(H2.2) price channel** — acquire European returns and test abnormal returns / valuation directly; **ADR-inclusive ownership** — map ADR holdings to underlying-share equivalents (needs the ADR conversion ratio) to close the §7.6 Caveat 1 gap.

---

## Second external review round (F1–F13) — responses, new tests, and the three-way pairwise FE headline

A second (cloud-based) AI review raised 13 findings — F1 critical, F2–F7 major, F8–F13 minor — all touching either the identification/timing of the headline or the reliability of the inference. Every one was verified against the on-disk data, addressed with new code (all in the repo; see the §14 inventory), and re-run. **The null survives every fix, and the only CRVE-significant coefficients (at cumulative local-projection horizons) are not significant under valid, design-based inference.** All numbers in this section have been re-derived on the P0 panel; pre-P0 values are kept in parentheses so the movement is traceable. Two headline changes follow directly.

### Headline change 1 — three-way pairwise FE (Khwaja–Mian / De Haas saturation)

The headline is now the **fully-saturated three-way pairwise FE**: firm×quarter (`it`, α_{i,t}) + group×quarter (`gt`, γ_{g,t}) + **firm×group (`ig`, μ_{i,g})**; the earlier `it+gt` is kept as a comparison column. On the balanced 2-per-firm-quarter panel, `gt` absorbs β₀(US) and β₁(US·S); `it` absorbs the CN/S/CN·S levels; so only β₂(US·CN) and β₃(US·CN·S) are identified. On the differenced outcome, `ig` is a firm-specific drift control on the US−NONUS difference (via the pairwise collapse, the 3-pairwise = firm FE + quarter FE on Δy). The null surviving the fully-saturated FE is a **stronger** null claim. (The true Khwaja–Mian holder×quarter saturation needs the holder-level panel — deferred, §10.)

All rows are P0-vintage. Pre-P0 values in the last column are kept only for traceability.

| specification | 3-pairwise (it+gt+ig) β₃ | it+gt β₃ | verdict | (pre-P0) |
|---|---|---|---|---|
| headline Δw ~ US·CN·S_t | −5.279916×10⁻⁷ (p=0.762748, RI free 0.8154 / circ 0.8293) | −1.072712×10⁻⁶ (p=0.557028, RI 0.5984) | **null** | +2.746 / +2.081×10⁻⁶ |
| F1a lead-flow Δw_{t+1} ~ S_t | +2.09×10⁻⁷ (p=0.868) | −3.18×10⁻⁷ (p=0.7907, RI 0.8775) | **null** | +1.01×10⁻⁶ / +4.85×10⁻⁷ |
| F2 headline, in-span only | — (not rebuilt) | −1.5656×10⁻⁶ (p=0.5788, RI 0.6093, N=272,140) | **null** | +3.33×10⁻⁶ (p 0.135) |
| F7 US·CN·GPR_t / GPR_{t−1} | — (not rebuilt) | −2.20×10⁻⁶ (0.1262) / +9.51×10⁻⁷ (0.3496) | **null** | −8.4×10⁻⁷ (0.63) / +2.6×10⁻⁷ (0.88) |
| ownership FLOW (F6), winsor MAIN | +7.48×10⁻⁵ (CRVE 0.8265, **RI 0.913**) | +7.52×10⁻⁵ (CRVE 0.796) | **null (RI arbiter)** | +8.9×10⁻⁴, CRVE 0.0001, RI 0.37 |
| F1b LP cum h1 | −1.53×10⁻⁷ (no persisted CRVE p; RI free 0.9376 / circ 0.9634) | −1.2475×10⁻⁶ (CRVE p=0.5989) | **null on CRVE and RI (free 0.9376 / circ 0.9634); the pre-P0 CRVE rejection is gone** | +4.09×10⁻⁶ (0.009) / +2.85×10⁻⁶ (0.017) |
| F1b LP cum h4 = cum4 | +6.6006×10⁻⁶ | +3.1026×10⁻⁶ (CRVE p=0.1045, RI 0.2962) | **free 0.0664 / circ 0.1220 / moving-block 0.1074–0.1076 / max-\|t\| FWER 0.0558–0.0618 all clear 0.05; score-WCB 0.0038 still rejects and is disclosed. Arbiter = circular-shift → null. F1/F10** | +8.83×10⁻⁶ (CRVE 0.053, free-RI 0.039–0.042) |

**Sign reading, corrected.** Earlier drafts of this section asserted that every quarterly-shock (S_t) point estimate was positive, so that even a CRVE-significant cumulative horizon would have pointed against disengagement. That is no longer true and was never a real feature of the data: the uniform positive lean was the pre-P0 weekend-composition artifact (change log §0). Post-P0 the headline and the lead-flow and the LP h1 horizon are negative, cum2–cum4 are positive, and all of them are null. The one surviving directional remark worth keeping is narrow: cum4, the only cell that ever rejected under free permutation, still carries a **positive** sign, so even the score-WCB rejection at 0.0038 points against H2.1 rather than toward it.

### Headline change 2 — β₁ removed from directional evidence (F4)

The country-pair β₁ (US·S_{c,t}) is **removed from all directional-evidence sentences** (§0 index, §7.5, §13). On the P0 panel it is dead outright: β₁ = **−1.36×10⁻⁶** with p = **0.9103** (firm×quarter), **0.9432** (3 country clusters, df_r=2) and **0.9467** (country×quarter). Magnitude fell about 16× and the sign flipped. The clustering argument that used to carry this finding no longer discriminates anything, because the point estimate is identical at all three levels and every p exceeds 0.91. *(Pre-P0: +2.22×10⁻⁵, p 0.086 at firm×quarter, 0.246 at the country level; pre-B7, +17.7 with p 0.098.)*

### Finding-by-finding

- **F1 (critical — timing / retraction logic).** The concern: S_t is the quarter-*end* AR(1) residual while backward Δw_t spans the whole quarter, so the preferred spec allows only ~1 month of reaction, and the natural t+1-quarter response window was never tested; the SE tripling in the centered→backward retraction is not the signature of "removing look-ahead noise." Resolution on the P0 panel: (i) the **lead-flow** Δw_{i,g,t+1} ~ US·CN·S_t (pure lagged shock, no look-ahead) is **null** (it+gt −3.18×10⁻⁷, p=0.7907, RI 0.8775, N=338,076; 3-pairwise +2.09×10⁻⁷, p=0.868); (ii) the **local-projection IRF** of the cumulative response w_{t+h}−w_{t−1} on US·CN·S_t (h=0..4) **loses every CRVE rejection it used to have**: it+gt h1 −1.2475×10⁻⁶ (p 0.5989, was 0.017), h2 +3.8475×10⁻⁷ (p 0.8491, was 0.003), h3 +9.28×10⁻⁸ (p 0.9681), h4 +3.1026×10⁻⁶ (p 0.1045, was 0.035). Randomization inference agrees at every horizon: h1 0.5260, h2 0.8647, h3 0.9711, h4 0.2962. On the 3-pairwise collapse, cum1 is −1.5346×10⁻⁷ (RI free 0.9376 / circ 0.9634) and cum4 is +6.6006×10⁻⁶ with free 0.0664, circular 0.1220, moving-block 0.1074/0.1076 and within-family max-|t| FWER 0.0558/0.0618. **The one free-permutation rejection this document previously reported is gone on two independent grounds**: the A0 serial-robust inference fix and, separately, the P0 data fix. What still rejects is the score-based quarter-cluster WCB at 0.0038, and that route assumes cross-quarter independence, which the measured serial correlation of S_t (acf1 +0.271, Ljung–Box Q(1) p = 0.013) violates. It is disclosed in the hardening table, not adopted. `boottest`'s dense-matrix path still exhausts memory at the ~155–169k×10k allocation, which is why the WCB is score-based. Net: there is no US-differential disengagement response at t, at t+1, or cumulatively; the "reaction is in t+1" alternative is rejected. `run_audit_f1f2f7.do`, `run_audit_f1b_robust.do`, `run_randomization_inference.py`, `run_ri_3pairwise.py`. *(Pre-P0, superseded: lead +4.85×10⁻⁷ / +1.01×10⁻⁶; LP h1 CRVE 0.017, h2 0.003, h4 0.035; 3pw cum1 +4.09×10⁻⁶ CRVE 0.009, cum4 +8.83×10⁻⁶ with free-RI 0.039–0.042.)* (The RI is exact only if the quarter shocks are exchangeable; the circular-shift block variant is the arbiter throughout. The sharp null it tests is slightly broader than β₃=0: were the shock to act through a firm-level channel other than CN, the test could reject for non-β₃ reasons — R3-C.)
- **F2 (Cartesian grid vs firm existence span).** `06_cartesian_grid.jl` crosses the universe with ALL quarters 1999Q1–2023Q4 without intersecting each firm's own existence span, so pre-IPO / post-delisting structural Δw=0 rows enter the estimation panel. Measured: ~25% of the panel is out-of-span, most of it exactly Δw=0. These attenuate β₃, inflate N, and understate SE. Re-running the headline on the in-span subset only (P0 panel, N=**272,140**): β₃ = **−1.5656×10⁻⁶**, still **null** (p=0.5788, RI 0.6093; the in-span cumulative LP horizons are also null under RI, cum1 0.5565 / cum4 0.2705). The conclusion is unchanged; the headline N / SE / §5.5 extensive-margin figures are affected. *(Pre-P0: +3.33×10⁻⁶, p 0.135, RI 0.389, N 269,998; in-span cum1 0.192 / cum4 0.116.)* `build_audit_panel_f1f2f7.py` (`in_span`).
- **F3 / F9 (inference — few clusters, overlapping windows).** The tail-dummy specs (§7.3, 6/4/2 treated quarters) and the overlapping-window LP have unreliable CRVE (MacKinnon–Webb; `reghdfe` flagged a non-positive-semi-definite VCV on the LP), and the previously-planned wild cluster bootstrap fails with few treated clusters. **Randomization inference** is the correct design-based test and is now the arbiter for these specs. Its exact algebra (US−NONUS pairwise difference + quarter FE reproduces the two-way-FE β₃; per-quarter sufficient statistics make a permutation an O(82) weighted sum) was independently verified to reproduce reghdfe's β₃ to 7 significant figures. `run_randomization_inference.py`. The tail-dummy k=2/3 specs (2 and 4 treated quarters) are **inference-invalid** and reported as such / dropped, not as "low power."
- **F4 (country-pair β₁ clustering).** See Headline change 2 above. `run_audit_f4f8.do`.
- **F5 (MDE units).** The §7.6 "25–35 bps" MDE conflated the per-unit-CN·S coefficient scale with the outcome scale. Corrected: with the 3-pairwise CRVE SE on the winsorized MAIN flow of 3.40×10⁻⁴ (P0 vintage; pre-P0 the raw-flow SE was 2.14×10⁻⁴) and σ(S_t)=2.578 (see σ_S note, §7.3), the 80%-power MDE for a *representative* firm-quarter (CN∈[0.05,0.15], 1σ shock) is on the order of **1–4 bps of float**, not the ~6 bps that 2.8·SE implies at the non-existent CN·S=1 point; the quarter-end-only shock is classical measurement error that attenuates β₃ and enlarges the true MDE. §7.6 Caveat 2 carries the refreshed figures.
- **F6 (ownership flow denominator).** The §7.6 dos = ownership_share_t − ownership_share_{t−1} was NOT denominator-immune: with each term over its own *current* float, a buyback/issuance moves it with zero trading, by a term ∝ the group's own lagged level (differs across US/NONUS, so not absorbed by firm×quarter FE). Fixed: the primary outcome is now the pure flow **(held_t − held_{t−1}) / out_{t−1}** (fixed lagged float). On the P0 panel the MAIN winsorized column is β₃ = **+7.48×10⁻⁵** (3-pairwise, CRVE 0.8265, **RI 0.913, null**); the raw column is −3.61×10⁻² and is outlier-dominated post-P0 (§7.6 open issue). The old dos is retained as a labelled comparison column, and it is **no longer the VCE-degenerate one** in this vintage (see §7.6). `build_ownership_share_c6_panel.py`. *(Pre-P0: raw +8.9×10⁻⁴, CRVE 0.0001, RI 0.37.)*
- **F7 (generated regressor / full-sample AR(1)).** The AR(1) shock is estimated once on the full ~1957–2023 monthly series (its (a,b) embed future data) and is a generated regressor. Robustness: replacing the composite shock with **US·CN·GPR_t + US·CN·GPR_{t−1}** (raw GPR level + previous-quarter lag) leaves both interactions **null** on the P0 panel: −2.20×10⁻⁶ (SE 1.43×10⁻⁶, p=0.1262) and +9.51×10⁻⁷ (SE 1.01×10⁻⁶, p=0.3496), N=348,156; in-span, −3.37×10⁻⁶ (p=0.1277) and +1.47×10⁻⁶ (p=0.3426), N=272,140. *(Pre-P0: −8.4×10⁻⁷ / +2.6×10⁻⁷, p 0.63 / 0.88.)* This spec remains the one robustness column that fits no AR model at all. **The deferred exact-nesting variant (`US·CN·gpr(M2)`) is SUPERSEDED by the shock menu (§7.3d) and is closed.** Its only purpose was to prove by proxy that the full-sample look-ahead did not matter; the menu answers that directly. `B_i_lvl_ar1_nla_qend` is the baseline construction with the look-ahead physically removed, it correlates 0.99961 with the baseline, and β₃ moves from −1.361×10⁻⁶ (p=0.763) to −1.227×10⁻⁶ (p=0.784). The look-ahead is worth about 1×10⁻⁷ in β₃ and nothing in inference. Disclosed in §9. `run_audit_f1f2f7.do`.
- **F8 (risk-set lead membership).** The main risk set (§7.4) conditions membership on t+1 holdings (a post-treatment variable). A **lag-only** variant (held at t or t−1, no look-ahead) gives β₃ = **−1.77×10⁻⁶** (SE 2.90×10⁻⁶, p=0.5443, N=268,800 / 5,613 firms), **null** and still statistically indistinguishable from the with-lead risk set (−1.73×10⁻⁶), so lead-conditioning is not driving the estimate. *(Pre-P0: +3.35×10⁻⁶, p 0.153, N 265,710 / 5,553 firms.)* `build_riskset_lagonly.py`.
- **F10 (multiple testing).** Across the ≥19 coefficient tests, every cell that ever carried a nominal CRVE p<0.10 (country-pair β₁, the flow, the Russia control, the full-period four-group column, the LP horizons) is adjudicated **null under valid design-based RI** on the P0 panel. Most are now null on CRVE as well. On the pre-registered LP family, the **cum4 free-permutation rejection is gone** (0.0388 pre-P0 → 0.0664 post-P0), so the exception this bullet used to carve out has closed. The **within-family single-step max-|t| FWER over h0..h4** (studentized, one shared draw stream) puts cum4 at **0.0558** free-permutation and **0.0618** moving-block; no horizon clears 0.05 after multiplicity correction. The **serial-robust RI** references agree: circular-shift 0.1220, moving-block L5/L8 0.1074/0.1076. One method still rejects, the score-based quarter-cluster WCB at **0.0038**, and it must be read two ways: it assumes cross-quarter independence that the measured serial correlation of S_t violates, and cum4's sign is **positive**, so even taken at face value it runs against H2.1. The earlier asymmetric use of β₁ as "direction opposite H2.1" is removed, and β₁ itself is now dead (p ≥ 0.91 at every clustering level). (See F4 and "Cum4 inference hardening".)
- **F11 (ADR-share docstring).** `build_ownership_share_panel.py` reported 91.57%/96.99% on-primary (an EQ+AD calc); corrected to the EQ-primary measure this build uses. On the P0 panel the figures are **US 91.2144% / NONUS 96.7794%** on-primary, i.e. **US 8.79% / NONUS 3.22%** off-primary, a 2.73× asymmetry, matching §7.6. *(Pre-P0: 91.08% / 96.68% on-primary, 8.92% / 3.32% off.)* Read from `ownership_share_diagnostics.csv`.
- **F12 (edge as-of classification).** `02_china_exposure.jl` classifies a supply-chain edge's counterparty country **as of the edge start date**, not per quarter — better than a look-ahead, but the "point-in-time" wording is clarified to mean edge-start, not quarter-by-quarter re-classification.
- **F13 (version control).** The `julia_descriptive/` pipeline was local-only (untracked, hard-coded absolute paths). The code is now committed and pushed to `github.com/fidio728/practice` so it is reviewable; a `.gitignore` keeps the 5 GB `output/` data out of version control.

### Cum4 inference hardening (G5 serial correlation, G7 within-family multiplicity, G8 feasible WCB) — `run_cum4_inference.py`

The one cell in the 3-pairwise LP family that ever rejected under free permutation is cum4 (positive sign, anti-H2.1). Three gaps in that single p-value are closed in one engine reading the same frozen `audit_c6_panel.dta` input as `run_ri_3pairwise.py` (5,000 free-permutation draws, seed 20260702). **All numbers below are P0-vintage.** The three β₃ gates were verified against `headline_3pairwise_canonical.csv` and `audit_ri_3pairwise.csv` before the engine's constants were touched, and they pass at relative error ≤ 1.3×10⁻⁸: h0 −5.2799161×10⁻⁷, cum1 −1.5346165×10⁻⁷, cum4 +6.6006275×10⁻⁶, with free-permutation anchors 0.8014 / 0.9368 / 0.0636.

**Shock serial structure (G5).** The 82-quarter headline shock S_t is serially correlated: re-derived lag-1 autocorrelation +0.270747, lag-2 +0.163492, Ljung–Box Q(1)=6.2336 (p=0.0125), Q(4)=11.3632 (p=0.0228) (hand-rolled biased ACF + Ljung–Box via `scipy.stats.chi2.sf`; `statsmodels.acorr_ljungbox` was broken in this environment). The shock side is bit-frozen across P0, so these values are unchanged. Free permutation assumes exchangeable quarters, so it is anti-conservative exactly where cum4's MA(4) overlapping outcome makes the dependence worst. For reference the aggregated shock s_agg carries a still-higher lag-1 autocorrelation of 0.357 (§7.3c), which is why the circular-shift variant is already the arbiter there.

**Three RI methods, a within-family FWER, and a score-based WCB, per horizon (P0 panel):**

| horizon | β₃ (×10⁻⁶) | free-perm | circular-shift | moving-block L5 / L8 | max-\|t\| FWER free / mb | WCB score, q-cluster | (pre-P0 free-perm) |
|---|---|---|---|---|---|---|---|
| h0 | −0.5280 | 0.8154 | 0.8293 | 0.7856 / 0.7766 | 0.9990 / 0.9990 | 0.8032 | 0.3083 (β +2.746) |
| cum1 | −0.1535 | 0.9376 | 0.9634 | 0.9400 / 0.9434 | 1.0000 / 1.0000 | 0.9463 | 0.1052 (β +4.094) |
| cum2 | +2.2694 | 0.3675 | 0.4268 | 0.4069 / 0.3919 | 0.6181 / 0.5957 | 0.1774 | 0.022 (β +7.131) |
| cum3 | +2.8071 | 0.3619 | 0.3537 | 0.3997 / 0.4173 | 0.5069 / 0.4869 | 0.1160 | 0.082 (β +6.467) |
| **cum4** | **+6.6006** | **0.0664** | **0.1220** | **0.1074 / 0.1076** | **0.0558 / 0.0618** | **0.0038** | 0.0388 (β +8.826) |

Circular-shift uses all 81 non-trivial rolls (min two-sided p = 1/82); moving-block permutes contiguous blocks (primary L=5, sensitivity L=8) with a random circular start, 5,000 draws; the FWER is single-step max-T over studentized |t*_h| across h0..h4 within one draw stream (studentized so cum4's scale cannot dominate cum1's). The WCB is a score-based quarter-cluster wild bootstrap (one-time FWL residualization, 82 quarter scores, B=9,999 Webb six-point weights; quarters are the binding cluster dimension, since firm clusters number in the thousands); `boottest`'s dense-matrix path OOMs at the ~155–169k×10k allocation, so this route replaces it, not the reverse. Anchor cross-check drift was +0.0140 / +0.0008 / +0.0028, inside the 0.02 Monte-Carlo tolerance, no DRIFT flag. One bookkeeping note, stated once and covering the whole free-permutation column: `audit_ri_3pairwise.csv` records h0 / cum1 / cum4 as **0.8014 / 0.9368 / 0.0636** while the hardening engine reports **0.8154 / 0.9376 / 0.0664**. All three gaps are Monte-Carlo noise across two draw streams, not discrepancies. This document quotes the hardening-engine stream in §0, the review-round table, the hardening table, §13 and §15, and the `audit_ri_3pairwise.csv` stream in §7.2, §7.3b and the anchor line above; each in-text use is tagged with its engine.

**Arbiter and verdict (cum4), with one live tension.** The reporting arbiter for the LP horizons is the **circular-shift RI**. Under it cum4 is **null** (0.1220), as is every horizon. So is free permutation now (0.0664), so is moving-block (0.1074/0.1076), and so is the within-family max-|t| FWER (0.0558 free, 0.0618 moving-block). **The pre-registered free-permutation rejection has dissolved twice over**: once through the A0 serial-robust inference fix, and independently again through the P0 data fix, which moved cum4's β₃ from +8.826×10⁻⁶ to +6.601×10⁻⁶. Nothing in the design-based family rejects.

One route disagrees, and it is reported rather than dropped. The **score-based quarter-cluster WCB reads p = 0.0038** (t = +2.666), which is *more* significant than its pre-P0 value. The adjudication is explicit: that bootstrap resamples scores independently across quarter clusters, so it assumes cross-quarter independence. The measured serial correlation of S_t (acf1 +0.271, Ljung–Box Q(1) p = 0.0125) is exactly the violation of that assumption, and cum4's overlapping four-quarter window is where the violation bites hardest. The WCB corrects the firms-within-quarter clustering cross-sectionally; it does not model dependence across quarters, so it does not answer the question cum4 turns on. By the pre-registered arbiter hierarchy, **RI governs and cum4 is reported as not surviving**, with the WCB number disclosed on the same line rather than buried. Read alongside cum4's **positive** sign, even the surviving rejection points against H2.1. Output: `output/cum4_inference_hardening.csv`.

### New files this round (in the repo; see the §14 inventory)
Second-review round: `build_audit_panel_f1f2f7.py`, `run_audit_f1f2f7.do`, `run_audit_f1b_robust.do`, `run_randomization_inference.py`, `run_ri_3pairwise.py`, `run_headline_3pairwise.do`, `build_riskset_lagonly.py`, `run_audit_f4f8.do`, and the F6 rewrite of `build_ownership_share_c6_panel.py`.

P0 round (2026-08-04): `diag_p0_asof_window.py` (the W-curve and coverage/staleness diagnostics behind the W=10 choice), `diag_p0_compare.py` (old-vs-new gates: coverage, US-share tilt, dollars, shock bit-identity, grid and c6 deltas), `archive_preP0.py` (56 artifacts archived as `*_preP0`, 16 scripts stamped with vintage banners), plus the shock-menu trio `build_shock_menu.py`, `run_shock_menu.do`, `run_ri_shockmenu.py` (§7.3d).

---

## 0. Reviewer's quick index of things most worth attacking

1. The holdings snapshot rule (§5.0): the as-of W=10 window replaced an exact-quarter-end filter that dropped ~40% of funds on weekend quarter-ends, US/NONUS-asymmetrically. Press on whether W=10 is the right window, on the 11.3pp ramp-up-era residual, and on the ~1-day valuation drift the rule introduces on 7.8% of rows.
2. The zero-fill (§5): does imputing "absence = zero weight" create or destroy the object of interest? We keep the full Cartesian grid as the main panel but a reviewer should press on whether the extensive-margin gap is causal or mechanical.
3. The backward-vs-centered differencing (§4.1, §8): the entire significance of the historical headline flips on this. We argue the centered version had look-ahead contamination; a reviewer should verify that argument, and should read §8b, where the *positive-sign pattern* is retracted on separate grounds.
4. Power (§7.3): the shock-tercile dose menu (which replaces the inference-invalid tail-dummy design) shows no monotone dose-response; the continuous shock is arguably underpowered given only 82 quarters.
5. The "US vs non-US" contrast requires both groups present within a firm-quarter (§7.4). Our first conditional-sample build violated this (per-group selection); it is documented as an error and corrected.
6. Data coverage assumption (§5.5): after zero-fill, Δw = 0 cannot separate "truly held nothing" from "FactSet did not capture the institution."
7. The raw shares-flow outliers (§7.6) and the cum4 WCB-versus-RI disagreement (§ "Cum4 inference hardening"): the two items this cycle leaves open rather than closes.

---

## 1. Research question and hypotheses

**Question.** When US–China geopolitical tension rises unexpectedly, do US institutional investors reallocate away from European-listed firms with high Chinese supply-chain exposure, relative to non-US investors holding the same firm in the same quarter? (The estimand is measured within the holdings-observed European issuer universe, not the full listing universe; see §3, Caveat 2.)

**H2.1 (ownership-flow channel, the one estimated).** US holders disengage (reduce portfolio weight) more than non-US holders from high-China-exposure European firms when tension rises. This predicts the triple-interaction coefficient **β₃ < 0**.

**H2.2 (price channel, planned, not yet estimated).** Independent of whether US capital flees, high-China-exposure European firms earn lower abnormal returns / carry lower valuations around high-tension quarters (firm is harmed in price even absent identifiable US-specific selling). This is the planned extension; it requires European stock-return data not yet assembled.

The design ports the within-borrower / across-lender identification of the cross-border bank-lending literature (Khwaja–Mian 2008; De Haas et al. 2025) to portfolio equity.

---

## 2. Data sources

| # | Source | Content | Coverage |
|---|---|---|---|
| 1 | FactSet Ownership v5 | Institutional positions (USD market value) per holder–security–quarter | 2003Q3–2023Q4 |
| 2 | FactSet Revere Supply Chain Relationships | Directed firm-to-firm links (supplier / customer / partner) with counterparty country | from 2003 |
| 3 | Iacoviello & Tong (2026) AI-driven country-pair GPR | Monthly bilateral US–China geopolitical-risk index (extends Caldara & Iacoviello 2022) | monthly |
| 4 | Compustat North America + Compustat Global | Firm accounting, listing country | 2003–2023 |
| 5 | Global Sanctions Data Base (Felbermayr, Kirilakha, Syropoulos, Yalçın, Yotov) | Sanctions episodes; **robustness only**, not in main spec | 1950–2022 |

**Identifier merge (coverage cascade, real counts):** the holdings-observed European security universe is **12,791** distinct `sec_entity_id` on the P0 panel (pre-P0: 12,743; the as-of rule of §5.0 recovers 48 firms whose only reports fell on shifted weekend dates). CUSIP is populated for all of them, ISIN for 1,177, SEDOL for 0; 10,171 match to Revere on the pre-P0 cascade. A security that fails to match is treated as **no observed supply-chain coverage**, not zero Chinese exposure.

The firm count then steps down further, and the figures below measure **different objects** (do not read them as inconsistent): of the 12,743 pre-P0 securities, 2,572 fail the Revere match (see `05_unmatched_profile_by_country.csv`, which sums to 12,743 = 10,171 matched + 2,572 unmatched). Of the matched, **8,014** carry a non-null all-`rel_type` China share in at least one quarter; the rest are matched but have no *active* supply-chain link in any observed quarter, so `NULLIF(n_total_links, 0)` routes them to the MISSING bucket (§5.5, and the `NULLIF` at 02/06). Under the **B7 CUSTOMER+SUPPLIER restriction** (§4.2), firms whose only China links were competitor/partner ties also route to NULL; lagging the exposure one quarter (`china_share_lag1q`) then leaves **6,867** securities carrying a non-null lagged China share, which is the distinct-firm count of the P0 estimation panel (pre-P0: 6,854; pre-B7 all-`rel_type`: 7,928). So 12,791 (universe) > 10,171 (identifier match) > 8,014 (non-null all-`rel_type` CN) > 6,867 (non-null lagged CUSTOMER+SUPPLIER CN, in-panel); the gaps are the documented MISSING bucket, the CUSTOMER+SUPPLIER restriction, and the one-quarter lag, not stale or contradictory counts. **The intermediate Revere-cascade counts (10,171 / 8,014) were not re-derived this cycle on the P0 universe**; only the endpoints (12,791 and 6,867) were.

---

## 3. Sample frame — the holdings-observed European issuer universe

28 listing jurisdictions (`00_setup.jl`, `EU_COUNTRIES`):

```
GB, DE, FR, NL, CH, IT, ES, SE, DK, NO, FI, BE, AT, IE, LU, PT,
PL, CZ, HU, GR, RO, SK, SI, BG, HR, EE, LV, LT
```

**Caveat 1 (EU label):** the set includes **GB, CH, NO**, none of which is a current EU member. The correct label is "European-listed," not "EU." GB alone carries most of the listing-country weight in the country-pair subsample.

**Caveat 2 (holdings-observed, not full listing universe).** The firm universe is built in `06_cartesian_grid.jl` from the FactSet Ownership holdings panel (`FROM read_parquet('$EOM_PATH') WHERE sec_country IN EU`), i.e. every `sec_entity_id` that ever appears with a European `sec_country` in the holdings data. It is **not** the full FactSet Security Coverage listing universe. The correct estimand phrasing is therefore "reallocation within the holdings-observed European issuer universe," not "all European-listed firms." Rebuilding from Security Coverage would change the estimand and add many zero-difference observations (securities never held by any institution). Because with non-missing CN those observations carry zero within-firm-quarter outcome variation but nonzero regressor variation, the mechanical expectation is that they **attenuate β₃ toward zero** rather than overturn the null; but this is the expected direction, not a proven one, and should be verified by actually running the full-universe rebuild rather than asserted — see §9. The full-universe rebuild is an optional robustness, not a correctness bug.

---

## 4. Variable construction (formulas + exact code)

### 4.1 Outcome: change in portfolio weight Δw

For firm *i*, holder group *g* ∈ {US, NONUS}, quarter *t*:

- H_{i,g,t} = Σ_{b∈g} position_value_{b,i,t}  (group g's total USD position in firm i)
- T_{g,t} = Σ_{i∈European-listed} H_{i,g,t}   (group g's total European book)
- w_{i,g,t} = H_{i,g,t} / T_{g,t}
- **Δw_{i,g,t} = w_{i,g,t} − w_{i,g,t−1}**  (BACKWARD difference)

Exact SQL (`06_cartesian_grid.jl`, after the C6 grid is built):

```sql
SELECT
    sec_entity_id, sec_country,
    holder_group,
    holder_group AS investor_country,
    quarter_end AS report_date,
    I_ict,
    portfolio_weight_eu,
    LAG(portfolio_weight_eu, 1) OVER (PARTITION BY sec_entity_id, holder_group ORDER BY quarter_end) AS w_prev,
    (portfolio_weight_eu
     - LAG(portfolio_weight_eu, 1) OVER (PARTITION BY sec_entity_id, holder_group ORDER BY quarter_end)) AS delta_w,
    china_share,
    LAG(china_share, 1) OVER (PARTITION BY sec_entity_id, holder_group ORDER BY quarter_end) AS china_share_lag1q,
    gpr_us_cn,
    shock_us_cn
FROM grid_zfilled
```

**Why backward, not centered (critical, load-bearing decision).** An earlier build used a centered difference Δw_t = w_{t+1} − w_{t−1}. That embeds w_{t+1} on the LHS, which includes the date-t+1 adjustment and therefore mixes contemporaneous with future response, misaligning the outcome with the stated timing of the causal estimand (a post-treatment / look-ahead window, not a proven mechanical shock correlation). Backward differencing removes that misalignment and drops only the first quarter of each (firm × group) series. **This choice changes the headline from "significant" to "null" — see §8.**

### 4.2 China exposure CN_{i,t−1}

CN_{i,t} = R^{CN}_{i,t} / R_{i,t}, then lagged one quarter.

- R_{i,t} = active **supply-chain** edges where firm *i* is on **either** side (**symmetric** counting), counting only the two buyer–seller `rel_type`s: `CUSTOMER` and `SUPPLIER`. `COMPETITOR` (14.85% of CN edges) and all `PARTNER-*` types (joint venture, manufacturing, licensing, marketing; 21% of CN edges) are **excluded**: they capture rivalry or looser cooperation, not input dependence. This matches the descriptive Figure 1 universe. (B7 fix, 2026-07-22; the earlier version summed all `rel_type`s. The all-type share is kept in the parquet as `china_share_alltypes` for diagnostics.)
- R^{CN}_{i,t} = **bilateral union**: {firm i = source, CN = target} ∪ {CN = source, firm i = target}.
- Home-region classification is **point-in-time** (time-versioned), not a single end-of-sample label → no look-ahead in exposure.

Exact SQL for the share (`02_china_exposure.jl`):

```sql
CAST(COALESCE(c.n_cn_customer, 0) + COALESCE(c.n_cn_supplier, 0) AS DOUBLE)
    / NULLIF(t.n_supplychain_links, 0) AS china_share
```

Note `NULLIF(..., 0)`: a firm-quarter with **no supply-chain links at all** gets china_share = **NULL** (not 0), and does not enter the exposure table. This is deliberate (see MISSING bucket, §5.5).

**HIGH-exposure cutoff.** `HIGH` if CN_{i,t−1} > **0.0625** = the median of strictly-positive CN_{i,t−1} across firm-quarters (CUSTOMER+SUPPLIER share, post-B7 rebuild 2026-07-22; the pre-B7 all-`rel_type` value was 0.0476). **Disclosed nuance:** "high exposure" therefore means "at least ~6% of a firm's reported supply-chain edges touch China" — meaningful but **not extreme**. The estimation universe is now **6,867 firms** (pre-P0 6,854; pre-B7 7,928 — firms whose only China links were competitor/partner ties, with no customer/supplier link, carry NULL exposure and drop out). The cutoff itself is unaffected by P0, since the exposure side is bit-frozen.

### 4.3 Shock S_t

S_t isolates the unexpected component of the monthly bilateral US|China GPR. Fit an AR(1) by **OLS (conditional least squares)** and keep the residual at the quarter's final month.

- GPR_m = a + b·GPR_{m−1} + S_m
- Estimator (mean-deviation OLS): b̂ = Σ(x−x̄)(y−ȳ)/Σ(x−x̄)², â = ȳ − b̂·x̄, where y = GPR_m, x = GPR_{m−1}.
- **Estimates (verified, USA|China): a = 0.6936, b = 0.6572, n = 796 months, R² = 0.432, implied long-run mean a/(1−b) = 2.02 ≈ sample mean 2.02.** First month has no lag → residual NaN (795 obs in the regression). Finite-sample (Kendall) bias on b ≈ −(1+3b)/n ≈ −0.0037 (negligible).

Python recipe (`build_country_pair_shock.py`, `fit_ar1`; identical to `05_combine_visualize.jl`):

```python
g = series.dropna().to_numpy(float)
y, yL = g[1:], g[:-1]
b_hat = ((yL-yL.mean())*(y-y.mean())).sum() / ((yL-yL.mean())**2).sum()
a_hat = y.mean() - b_hat*yL.mean()
resid = y - (a_hat + b_hat*yL)      # S_m
```

S_t > 0 = unexpected escalation. The pipeline's stored `shock_us_cn` was verified equal to this residual.

**Tail-dummy variants (advisor-requested, §7.3).** ShockTail_t = 1[ (S_t − mean)/sd > k ], one-sided right (escalation), sd computed over the **82 distinct quarters** (not the panel rows), for k ∈ {1.645, 2, 3}.

**The shock side is bit-frozen across the P0 rebuild.** Verified two ways: the 265-quarter shock artifact reproduces to max|d| = 0.0 on every column, and `sd(shock) = 2.57789`, `mean = 0.16695` over the 82 estimation quarters, identical to pre-P0. Everything that moved in this cycle moved on the holdings side. A menu of ten alternative shock constructions is in §7.3d.

### 4.4 US indicator

US_g = 1 if g = US, 0 if g = NONUS.

---

## 5. Panel construction (C6 Cartesian grid + zero-fill) and cleaning

### 5.0 Holdings snapshot rule — the as-of W=10 window (P0 fix, `03_eom_etl.jl`, 2026-08-04)

**What was wrong.** `03_eom_etl.jl` selected holdings rows with `REPORT_DATE` equal to the exact calendar quarter-end. FactSet shifts a fund's report date to the prior business day when the quarter-end falls on a weekend. The exact-match filter therefore kept only the subset of funds that happened to stamp the weekend date itself, and dropped the rest. The size of the hole is large and it is concentrated on specific quarters:

| quarter-end | day | fund coverage, pre-P0 |
|---|---|---|
| weekday quarter-ends | Mon–Fri | 86–87% |
| 2022-12-31 | Saturday | **58.4%** |
| 2023-09-30 | Saturday | **38.6%** |

**Why it biased β₃ rather than just adding noise.** The dropped and kept batches differ in composition. On 2022-12-31 the dropped Friday batch was 26% US against 21% US in the kept Saturday batch. A group-composition shift that varies by quarter is exactly what group×quarter FE cannot absorb, because it moves the within-firm-quarter US−NONUS difference itself. It also produced a fake "deepest drawdown ever" in the advisor-facing figure (§J note below).

**The rule.** Per `(fund_id, fsym_id)`, take the **latest report on or before the quarter-end within a W = 10 calendar-day window**; stamp `report_date` to the quarter-end; keep `report_date_actual` and `asof_gap_days` as provenance columns. W = 10 sits on a measured plateau (W = 3..14 give the same coverage on the full 1999–2023 series), and W = 31 is rejected because it imports 28.4% stale rows. The W curve and all gate outputs are archived in `output/diag_p0_*`.

**Only-change-one-thing evidence.** The new holdings panel is **208,418,523 rows**, +11.57% on the old 186,800,295. The old panel is exactly the `asof_gap_days = 0` subset of the new one and is bit-equal on those rows. 56 downstream artifacts were archived as `*_preP0` and 16 scripts carry vintage banners (`archive_preP0.py`). Infrastructure note: `00_setup.jl`'s duckdb connection now sets `max_temp_directory_size=300GB`, because the old 4.3GB default cap deadlocked step 04 on the larger panel.

**Gates (independently re-derived by an adversarial verifier using a different algorithm, exact to the last dollar).**

| gate | pre-P0 | post-P0 |
|---|---|---|
| coverage gap, weekday vs weekend (full sample) | 22.3pp | **4.2pp** |
| same, mid era | 17.7pp | 2.9pp |
| same, modern era | 34.0pp | 1.6pp |
| trend-free artifact | 11.1pp | 1.1pp |
| **US-share selection tilt (the bias channel)** | **−3.41pp** | **+0.01pp**, flat in all eras |
| 2022Q4 European book | $1.457T | $1.892T |
| 2021Q4 European book | $2.40T | $2.40T (unchanged) |
| 2022Q4 nominal YoY | −39.2% | −21.2% |
| shock series (265 quarters) | — | bit-identical, max\|d\| = 0.0 |
| grid exposure/shock columns | — | bit-frozen on all 2,548,600 common cells |
| c6 estimation panel | 347,952 rows / 6,854 firms | **348,156 / 6,867** |

The grid's `w` moved on 18.9% of cells, as it must, and **48 new firms enter legitimately** (204 exposed cells). The c6 row gain, +204, equals those new firms' exposed cells exactly.

**Residual disclosure, ramp-up era.** The full-sample weekday-vs-weekend coverage gap does not go to zero; it stops at 4.2pp, dragged there by the 1999–2005 ramp-up era, which sits at **11.3pp**. No choice of W fixes that, because it is a property of early FactSet coverage rather than of the stamping rule. It is disclosed rather than tuned away. The candidate robustness, not yet run, is to start the sample in 2006.

**Second-order artifact of the fix, disclosed.** Step 04's `I_ict` now mixes valuation dates within a quarter cell for shifted funds, so about 7.8% of rows carry roughly one day of price drift relative to the quarter-end. That is a real cost of the as-of rule and it is immaterial against the 40% coverage hole it closes.

### 5.1 The problem the zero-fill fixes (selection on the outcome)

Raw FactSet records **only held positions**. If a group's holding in a firm is zero, there is **no row**. But H2.1 predicts US moves *toward* zero on high-exposure firms; keeping only non-zero rows deletes exactly the entry/exit events the design must measure. That is conditioning on the outcome.

**Real example (verified in data).** Firm `sec_entity_id = 05HF13-E` (GB-listed, high-CN), US group: held ~$7.5B through 2021Q4, then **$0 in 2022Q1** (full liquidation). In raw data the firm simply disappears from the US panel after 2021Q4; the −Δw of the liquidation is unobservable. Under zero-fill, 2022Q1 becomes a row with Δw = 0 − w_{2021Q4} < 0, so the exit is captured.

### 5.2 Fix: full Cartesian grid + zero-fill (audit item C6)

Unit of observation = every holdings-observed European issuer (§3, Caveat 2) × quarter × {US, NONUS}, whether or not held. Δw is 0 when the group held nothing at both t−1 and t; < 0 at an exit (−w_{t−1}); > 0 at an entry (+w_t).

Grid diagnostic (backward diff) on the P0 grid: **2,558,200 cells; 2,519,827 non-null Δw; 38,373 first-quarter-null cells; 0 interior nulls.** The non-null count is read off the P0 rebuild (`build_extensive_margin_panel.py`, `d_breadth` non-null) and matches the isomorphic Russia grid's non-NULL Δw exactly; `06_cartesian_grid.jl` was not re-run as a standalone diagnostic this cycle. *(Pre-P0, superseded: 2,510,371 non-null and 38,229 first-quarter nulls, summing to the 2,548,600-cell pre-P0 grid.)*

### 5.3 Data-cleaning / audit fixes (C1–C6)

| Fix | What |
|---|---|
| C1 | Ownership denominator restricted to European-listed equity (w = within-Europe share) |
| C2 | Pre-2003 supply-chain exposure = NULL, not zero (Revere begins 2003) |
| C3 | One-quarter lag via window function (genuine LAG, not row shift) |
| C4 | Symmetric edge counting (bilateral union of firm-as-source and firm-as-target CN links); **CUSTOMER+SUPPLIER only (B7)** |
| C5 | Time-versioned (point-in-time) home-region classification → no look-ahead |
| C6 | Cartesian grid + zero-fill → corrects selection on the outcome |

### 5.4 Estimation panel (`build_c6_panel.py`)

Single duckdb SELECT (no multi-scan row-misalignment), then drop rows with missing dw / cn_lag / shock:

```python
SELECT
    CAST(sec_entity_id AS VARCHAR)                  AS firm_str,
    holder_group                                    AS hgroup,
    CAST(report_date AS TIMESTAMP)                  AS rdate,
    delta_w                                         AS dw,
    china_share_lag1q                               AS cn_lag,
    -- B7 direction split (sell_lag + buy_lag = cn_lag row-wise; §7.7):
    sell_share_lag1q                                AS sell_lag,   -- CUSTOMER-side China link share, lagged
    buy_share_lag1q                                 AS buy_lag,    -- SUPPLIER-side China link share, lagged
    shock_us_cn                                     AS shock,
    CASE WHEN holder_group = 'US' THEN 1 ELSE 0 END AS us
FROM read_parquet('merged_us_eu_zero_filled.parquet')
WHERE delta_w IS NOT NULL AND china_share_lag1q IS NOT NULL AND shock_us_cn IS NOT NULL
```

Result: **c6_panel.dta = 348,156 firm-group-quarter rows, 6,867 firms, 82 quarters (2003Q3–2023Q4), 50/50 US/NONUS** (P0 vintage; pre-P0 347,952 / 6,854; pre-B7 all-`rel_type` 462,564 rows / 7,928 firms). The 3-pairwise FE drops **466** singletons, leaving N = 347,690 and 6,634 firm clusters. Written with `to_stata(convert_dates={'rdate':'tc'}, version=118)`; pandas writes datetime as %tc so Stata must `dofc()` before `mofd()`.

**Known defect in a shipped artifact:** `tercile_results.csv`'s addnote hardcodes "MAIN drops 462 singletons". The true count is **466**, independently confirmed. Anywhere this document or a CSV says 462, read 466.

**Ownership/flow universe (§7.6) is smaller than the main panel.** 248,024 rows / 5,191 firms, roughly 76% of the main panel's firms, because the primary-EQ float restriction plus the BAD firm-quarter rule bind harder. Fraction of holdings market value on the primary EQ class: US **91.2144%**, NONUS **96.7794%** (pre-P0 91.08% / 96.68%). Whether the coverage gap between the two universes itself moved with P0 has **not been re-derived this cycle**; treat the sample-comparison sentence in §7.6 as provisional on that point.

### 5.5 Honest data caveats

- **Zero-fill ambiguity.** After the fill, Δw = 0 cannot distinguish "the group truly held nothing" from "FactSet did not capture an institution." US 13F reporting is mandatory above the $100M AUM threshold, so **material** US institutional ownership is largely observed; non-US coverage varies by jurisdiction. Absence is treated as zero **by assumption, not by observation**.
- **MISSING bucket.** Firms with no Revere coverage at t get CN = NULL and are bucketed MISSING, reported separately from HIGH/LOW; "no data" is never silently read as "zero exposure." These rows drop out of the estimation panel via the cn_lag-non-null filter.
- **Extensive-margin gap (descriptive only; P0 vintage, §7.10).** At the firm-quarter cell level the US zero-holder rate = 1,058,546/1,279,100 = **82.757%**, NONUS = 902,112/1,279,100 = **70.527%**, gap **+12.23 pp** in the H2.1 direction (US holds these firms more sparsely). *(Pre-P0: +11.30 pp on a 1,274,300-cell grid; pre-B7: +11.26 pp.)* On the estimable subset (cn_lag non-missing, 174,078 firm-quarters) the same gap is **+15.18 pp** (US 41.38% vs NONUS 26.20%). **The two denominators now differ by about 3pp, so any sentence quoting this fact must say which one it uses.** Pooled over the whole window; cannot separate a geopolitical channel from time-invariant US-vs-NONUS portfolio differences. Not causal, and the differential *response* to the tension shock is null (§7.10). Full detail, four-cell counts, and the holder-count DDD in §7.10.
- **Adding-up artifact.** Because portfolio weights sum to 1 within a group each quarter, the mean Δw across ALL firms is mechanically ≈0 (this is why the "all-firms" descriptive scatter, Panel A of the pre-regression figure, sits at ~1e-20).

---

## 6. Identification

### 6.1 Estimating equation (De Haas 4-coefficient form)

Δw_{i,g,t} = β₀·US_g + β₁·(US_g × S_t) + β₂·(US_g × CN_{i,t−1}) + β₃·(US_g × CN_{i,t−1} × S_t) + α_{i,t} + γ_{g,t} + ε_{i,g,t}

- α_{i,t} = firm × quarter FE; γ_{g,t} = holder-group × quarter FE.

### 6.2 Absorption (main spec)

Because S_t has no host-country variation:
- β₀·US_g and β₁·(US_g × S_t) → absorbed by γ_{g,t}
- levels CN_{i,t−1}, S_t, CN_{i,t−1}·S_t → absorbed by α_{i,t}
- **identified: β₂ and β₃ (headline).** β₃ = ∂³Δw/∂US ∂CN ∂S = the differential US-vs-NONUS response to a firm's Chinese exposure as the shock moves, measured within the same firm-quarter. H2.1 ⇒ β₃ < 0.

### 6.3 Clustering

Two-way cluster on **firm and quarter**. Quarter clustering matters because S_t is a single common series (effective independent shock draws = number of quarters, not rows). Backward differencing induces MA(1) in Δw, which the firm cluster absorbs.

**VCE-validity convention (B7).** Every Stata spec writes a companion `*_vce_diag.csv` recording, per coefficient, whether the clustered VCV is positive-definite (`se_valid`). When it is not — few treated clusters, a fat-tailed outcome, or an overlapping-window LP — the **B9 guard** suppresses the SE rather than reporting a bogus non-PSD one (`se_valid=0`); such columns are flagged "VCE degenerate" in the tables and adjudicated by design-based randomization inference instead of the CRVE. Separately, the wild-cluster bootstrap cross-check of the marginal local-projection CRVE p's **is completed** via a score-based quarter-cluster route (`run_cum4_inference.py`): one-time FWL residualization, 82 quarter scores, B=9,999 Webb six-point weights, studentized with the reweighted-score cluster SE. Per-horizon WCB p on the P0 panel = 0.8032 (h0) / 0.9463 (cum1) / 0.1774 (cum2) / 0.1160 (cum3) / **0.0038 (cum4)**. *(Pre-P0: 0.208 / 0.024 / 0.005 / 0.027 / 0.024.)* `boottest`'s own dense-matrix path still exhausts memory at 10k reps on the ~155–169k×10k allocation, so the score-based route replaces it; quarters are the binding cluster dimension (firm clusters number in the thousands). The serial-robustness question the LP horizons actually turn on is answered by the circular-shift / moving-block RI, not the WCB, which is why cum4's lone WCB rejection is disclosed but not adopted (§ Second review round, "Cum4 inference hardening", F1/F10).

**Non-PSD / CGM repairs, disclosed as a class.** `reghdfe`'s Cameron–Gelbach–Miller repair fired somewhere in most families this cycle: the direction-split indicator spec, the tercile MAIN, both sagg D3 specs, the S_{t−1} 3-pairwise spec, country-pair F4c, flow r1 and winsorized fq+gq, four-group `m_pas_full` and `m_usi_full`, F7 in-span, 07d spec4, and DDD Stage B. `se_valid` only tests that the SE is finite and positive, so it **cannot** detect this condition. Any CRVE p quoted from those specs carries the repair with it.

### 6.4 firm × group FE option (μ_{i,g})

The three pairwise FEs among {firm, group, quarter} are firm×quarter (have it), group×quarter (have it), and **firm×group (μ_{i,g})**. Adding μ_{i,g} absorbs any time-invariant US-vs-NONUS tilt toward each firm; β₂, β₃ then identify off within-(firm,group) time variation. Run as robustness (§7.2, §7.4). (A separate, heavier option — **holder × quarter** FE at the individual-institution level, the true Khwaja–Mian bank×time analog — is not yet built; it needs disaggregating from 2 groups to thousands of holders and conditioning on engagement.)

---

## 7. Estimation results — everything, honestly (coefficients ×10⁻⁶ of portfolio weight; two-way clustered SE in parentheses)

### 7.1 Main panel — three FE specifications (`07d_three_spec_table.do`, P0 panel, N = 348,156)

| | (1) No FE | (2) it+gt (α_{i,t}+γ_{g,t}) | (3) Weak FE (firm+quarter) |
|---|---|---|---|
| β₂ (US·CN_{t−1}) | −2.59 (3.77), p 0.4938 | −0.757 (4.14), p 0.8554 | −3.10 (4.00), p 0.4398 |
| β₃ (US·CN·S_t) | −0.0929 (0.503) | **−1.0727 (1.819)** | +0.5241 (0.695) |
| p(β₃) | 0.854 | **0.557** | 0.453 |
| R² | 0.0000 | 0.590162 | 0.005103 |
| F(2,81) | 0.4625 (p=0.6314) | 0.2884 (p=0.7502) | — |
| (pre-P0 β₃ / p) | +0.0006 / 0.999 | +2.08 / 0.151 (R² 0.6235, F 1.31) | +0.538 / 0.411 (R² 0.0073) |

**Every β₃ is null, and the smallest p in the table is now 0.45.** The current headline FE is the three-way pairwise (it+gt+ig; §7.2 and the "Second external review round" section); column (2) here is the it+gt comparison. The headline β₃ (3-pairwise) is **−5.279916×10⁻⁷ (p = 0.762748)**, negative, tiny, and indistinguishable from zero; the null is from a wide SE, not a tight zero. Spec 4 (weak FE) is CGM-repaired. (The "β₃-only" and "full triple" specs reproduce the same point estimate with lower-order terms auto-omitted, confirming the absorption logic.)

**Formal MDE (w-based headline).** With SE(β₃)=1.743166×10⁻⁶ (3-pairwise, P0) and σ_S=2.578, the 80%-power MDE for a representative firm-quarter (CN∈[0.05,0.15], 1σ shock) is 2.8·SE·CN·σ_S ≈ **0.63–1.89×10⁻⁶** in Δw units = **0.08–0.25% of σ(Δw)=7.6×10⁻⁴**. The design rules out within-Europe reallocations larger than about 0.25% of a typical quarterly weight change; it cannot rule out smaller ones. Both outcome families (w and shares-flow) carry an explicit MDE. *(Pre-P0 SE 1.697×10⁻⁶ → MDE 0.61–1.84×10⁻⁶.)*

### 7.2 + firm × group FE (`07e_firmgroup_tail.do`)

| | Headline (it+gt) | + firm×group FE (3-pairwise) |
|---|---|---|
| β₃ | −1.0727 (1.819), p=0.557 | **−0.5280 (1.7432), p=0.7627**, RI free 0.8014 (`audit_ri_3pairwise.csv`; the hardening engine's stream reads 0.8154, see the note in "Cum4 inference hardening") |
| N | 348,156 | 347,690 (466 singletons dropped) |
| (pre-P0) | +2.08 (1.44), p=0.151, N 347,952 | +2.746 (1.697), p=0.110, N 347,490 |

Still null. Interpretation: the null is **not** an artifact of structural US-vs-NONUS firm sorting.

**Tail-dummy companions on the P0 panel** (07e, k on the standardized shock; inference-invalid by design, reported for provenance only): k=1.645 β₃ = −2.7150×10⁻⁵ (SE 1.7704×10⁻⁵, p 0.1290; 6 treated quarters, 40,914 rows); k=2.0 −2.3023×10⁻⁵ (SE 2.3014×10⁻⁵, p 0.3201; 4 quarters, 29,010 rows); k=3.0 −6.0506×10⁻⁵ (SE 3.2898×10⁻⁵, p 0.0696; 2 quarters, 14,586 rows). With 2 to 6 treated clusters these p-values are not interpretable (MacKinnon–Webb; §7.3). Shock distribution on the estimation quarters: sd 2.57789, mean 0.16695, 82 quarters.

### 7.3 Shock-tercile dose menu (`run_tercile_3pairwise.do`, `run_ri_tercile.py`) — replaces the tail-dummy menu

σ_S over 82 quarters = **2.578** (verified independently on both the estimation-sample panel and the audit panel; a second-review-round "correction" to 2.42 was itself in error — it was computed over a wider, non-estimation-sample quarter range — and is reverted here, R3 shock-lag work, 2026-07-02).

The tail-dummy design (6/4/2 treated quarters, below) is **inference-invalid** with so few treated clusters, so the preferred dose test is now a **shock-tercile menu**: cut S_t into terciles over the 82 distinct quarters (realized bins 28/27/27), interact each with US·CN_{t−1}, T2 (middle) as base. A genuine dose-response would show a monotone, significant T3 (highest-escalation) coefficient. It does not:

| dose bin | 3-pairwise β₃ | fq+gq β₃ | RI p (3-pairwise) | (pre-P0 3pw β₃ / p / RI) |
|---|---|---|---|---|
| level (US·CN) | −8.66×10⁻⁷ (p=0.9551) | −1.79×10⁻⁶ (p=0.9076) | — | −1.02×10⁻⁵ / 0.5698 |
| T1 (lowest S_t) | +7.02×10⁻⁷ (p=0.9721) | +1.01×10⁻⁶ (p=0.9575) | — | +1.37×10⁻⁵ / 0.5347 |
| **T3 (highest S_t)** | **+2.77×10⁻⁶ (SE 1.87×10⁻⁵, p=0.8823)** | +4.55×10⁻⁷ (p=0.9796) | **0.8478** | +2.05×10⁻⁵ / 0.264 / RI 0.233 |
| T3 − T1 gradient | +2.07×10⁻⁶ (SE 1.03×10⁻⁵, CRVE p=0.841) | — | **0.8742** | +6.76×10⁻⁶ / 0.623 / RI 0.670 |

N = **347,690** (3-pairwise, R² 0.5908) / **348,156** (fq+gq, R² 0.5902). The family dissolves on the P0 panel: the highest-escalation tercile carries an RI p of **0.85** and the T3−T1 gradient **0.87**, against 0.23 and 0.67 before. There is no monotone dose-response, and the point estimates are now roughly an order of magnitude smaller than pre-P0. Cutpoints and bins are unchanged, as they must be, because the shock side is bit-frozen: p33.33 = −0.6383, p66.67 = +0.0116, bins 28/27/27, independently re-derived exact. RI is the design-based arbiter (`run_ri_tercile.py` re-cuts terciles per permuted shock); CRVE validity per spec is in `tercile_vce_diag.csv`, which has 4 rows and zero `se_valid=0`, though `reghdfe` did print a non-PSD/CGM warning on MAIN. Both routes agree on a decisive null, so nothing here rests on that choice.

**Superseded tail-dummy menu (retained for F3 provenance only, pre-B7 panel).** The earlier advisor-requested tail-dummy variants ShockTail_t = 1[(S_t−mean)/sd > k], one-sided right (escalation), on the pre-B7 all-`rel_type` panel:

| k | **treated quarters** | β₃^tail | SE | p |
|---|---|---|---|---|
| 1.645 | 6 / 82 | +1.07 | 15.7 | 0.946 |
| 2 | 4 / 82 | +14.0 | 18.0 | 0.439 |
| 3 | 2 / 82 | +8.59 | 39.0 | 0.826 |

All null; SE explodes as k rises. Empirical distribution is **fat-tailed** (k=1.645 gives 6 quarters = 7.3%, not the 5% of a normal). **With only 6 / 4 / 2 treated quarters the CRVE + t(81) inference here is not merely low-powered but statistically *invalid* (MacKinnon–Webb 2017): few treated clusters bias the CRVE and break the t(G−1) reference, and the standard wild bootstrap fails in the same regime.** These rows are **superseded** by the tercile dose menu above and kept only as F3 provenance; the dummy coefficient is on a different scale from the continuous one — do not compare point estimates.

### 7.3b Shock timing and units (advisor request, 2026-06-28 meeting)

The advisor's meeting comment on the shock was three-part: (a) the shock should *also* be lagged, since CN exposure is already CN_{t-1} — measuring both regressors as of the start of the period over which Δw_t is measured, rather than mixing a lagged CN with a contemporaneous S_t; (b) report it in standard-deviation units; (c) a two-standard-deviation threshold, matching the capital-flow-episode convention (Forbes and Warnock 2012, 2σ main / 3σ robustness) and the GPR-spike convention (Caldara and Iacoviello 2022, AER, 2σ; their earlier IFDP 1222 draft used 1.68σ on the AR(1) residual — structurally the closest precedent to our S_t). Point (c) is already covered by the k=2 row in §7.3 above. Points (a)-(b):

**(a) Lagged shock S_{t-1}: Δw_t ~ US·CN_{t-1}·S_{t-1}**, vs the current headline Δw_t ~ US·CN_{t-1}·S_t.

| spec | FE | β₃ | SE | p (CRVE) | RI p | (pre-P0 β₃ / p / RI) |
|---|---|---|---|---|---|---|
| S_t (current headline) | it+gt | −1.0727×10⁻⁶ | 1.82×10⁻⁶ | 0.5570 | 0.5984 | +2.081×10⁻⁶ / 0.151 / 0.412 |
| S_t (current headline) | 3-pairwise | −5.280×10⁻⁷ | 1.74×10⁻⁶ | 0.7627 | 0.8014 (`audit_ri_3pairwise.csv` stream; 0.8154 on the hardening stream) | +2.746×10⁻⁶ / 0.110 / 0.308 |
| **S_{t-1} (advisor spec)** | it+gt | **−1.0669×10⁻⁶** | 1.18×10⁻⁶ | **0.3696** | **0.6018** | −5.90×10⁻⁷ / 0.750 / 0.814 |
| S_{t-1} (advisor spec) | 3-pairwise | −7.30×10⁻⁷ | 1.24×10⁻⁶ | 0.5589 | — | −2.78×10⁻⁷ / 0.886 |

(σ_S(S_t)=2.5779, σ(S_{t−1})=2.5770, both bit-frozen; N=**348,156** it+gt / **347,690** 3-pairwise; grid 2,558,200 → 348,156 after requiring dw, cn_lag, S_t and S_{t−1}; RI 200,000 permutations, seed 20260702.)

The lagged-shock estimate roughly doubled in magnitude and its CRVE p fell from 0.750 to 0.370, but the design-based arbiter moved the other way (RI 0.814 → 0.602) and the cell stays comfortably null. **The pre-P0 framing of this section, which turned on S_t being positive and S_{t−1} flipping negative, no longer applies**: post-P0 both timings are negative, both tiny, both null. The point that survives is the weaker and more honest one, that β₃ is not stably signed or sized across reasonable timing conventions, which is informative about how little signal there is rather than evidence for disengagement under the "correct" timing. Two disclosures: the S_{t−1} 3-pairwise spec triggered the non-PSD/CGM repair, so its SE and p are post-adjustment, and no `shocklag_vce_diag.csv` exists, so B9 is not reportable for this family. `build_shocklag_panel.py`, `run_shocklag.do`, `run_ri_shocklag.py`.

**(b) SD-standardized reporting.** Since σ_S is a pure rescaling of the shock, standardizing changes only the coefficient's unit ("effect per 1-SD tension shock"), not any t-statistic or p-value. At σ_S=2.578 the current headline β₃ (3-pairwise) is **−1.36×10⁻⁶** per 1-SD shock. The advisor's S_{t−1} spec is **−2.75×10⁻⁶** per 1-SD (it+gt) and **−1.88×10⁻⁶** per 1-SD (3-pairwise), rescaling by σ(S_{t−1})=2.5770. Both remain null under the same p-values above.

### 7.3c Aggregated-shock robustness (`build_sagg_panel.py`, `run_sagg_distlag.do`, `run_ri_sagg.py`)

The headline S_t is the quarter-*end* AR(1) residual. As a robustness against that single-month timing, `s_agg` aggregates the shock over the quarter (a distributed-lag / within-quarter sum). σ(s_agg)=4.0846; its correlation with the stamped quarter-end shock is 0.425, so it is a genuinely different aggregation, not a rescaling. Both shock series are serially correlated: the quarter-end S_t has a re-derived lag-1 autocorrelation of +0.271 (lag-2 +0.163; Ljung–Box Q(1)=6.23, p=0.013; Q(4)=11.36, p=0.023, over the 82 quarters; hand-rolled ACF + `scipy.stats.chi2.sf` Ljung–Box), and s_agg is more serially dependent still, with lag-1 autocorrelation corr(s_agg_t, s_agg_{t−1})=0.357 (distinct from the 0.425 cross-correlation above). Free permutation assumes exchangeable quarters and is therefore anti-conservative on both; the circular-shift block variant below — and, for the overlapping-window LP horizons, the "Cum4 inference hardening" battery — preserves that serial structure and is the arbiter.

| spec | FE | β₃ | SE | p (CRVE) | RI p | (pre-P0) |
|---|---|---|---|---|---|---|
| D1 s_agg h=0 | it+gt (pairwise collapse) | **−1.55×10⁻⁶** | 9.23×10⁻⁷ | **0.0974** | free-perm **0.2353** / circular-shift **0.2195** | −1.06×10⁻⁶ / 0.348 / RI 0.512 / 0.463 |
| D1 s_agg h=0 | 3-pairwise | **−1.66×10⁻⁶** | 9.80×10⁻⁷ | **0.0939** | — (no RI arbiter) | −1.12×10⁻⁶ / 0.411 |
| D2 s_agg h=1 | 3-pairwise | +7.36×10⁻⁸ | 7.82×10⁻⁷ | 0.9253 | — | — |
| D2 s_agg h=1 | it+gt | −4.10×10⁻⁸ | 6.87×10⁻⁷ | 0.9526 | — | — |
| D3 distributed lag, h=0 | 3-pairwise | −1.67×10⁻⁶ | 9.78×10⁻⁷ | 0.0906 | — | — |
| D3 distributed lag, h=1 | 3-pairwise | +1.90×10⁻⁷ | 7.22×10⁻⁷ | 0.7928 | — | — |
| D3 joint Wald | 3-pairwise | F(2,81)=1.545 | — | 0.2194 | — | — |
| D3 cumulative (h0+h1) | 3-pairwise | −1.48×10⁻⁶ | 1.26×10⁻⁶ | 0.2421, CI [−3.99×10⁻⁶, +1.02×10⁻⁶] | — | — |
| D3 cumulative (h0+h1) | it+gt | −1.31×10⁻⁶ | 1.01×10⁻⁶ | 0.1982, CI [−3.31×10⁻⁶, +6.97×10⁻⁷] | — | — |

N = **347,690** (3-pairwise) / **348,156** (it+gt); the build drops zero rows, and 264 quarters carry all three monthly residuals (consistency assert max deviation 7.1×10⁻¹⁵). σ(s_agg)=4.0846, σ(stamped)=2.5779, cross-correlation 0.425, corr(s_agg_t, s_agg_{t−1})=0.357.

**Still null, but this is now the closest non-Russia cell in the battery on the design-based arbiter, and it moved.** |β₃| grew about 48% and the CRVE p fell roughly fourfold, to 0.094 (3pw) and 0.097 (it+gt). The design-based arbiter is what keeps the verdict: on the it+gt collapse RI free-permutation is 0.2353 and circular-shift 0.2195, against 0.512 / 0.463 pre-P0. So the CRVE and the arbiter moved in opposite directions here, and by the pre-registered hierarchy the arbiter governs. **The 3-pairwise h=0 cell, which carries the family's lowest p at 0.0939, has no permutation arbiter at all** — RI covers only the it+gt collapse. Its two-way analogue inflated from 0.0974 to 0.2353 under RI, so the 3pw p should be presumed similarly non-robust; that is an inference from the neighbouring cell, not a measurement.

**Magnitude-versus-significance caveat, keep it.** The per-SD sagg effect is −6.78×10⁻⁶ against the headline stamped-shock −1.36×10⁻⁶, about five times larger, which is more than measurement-error attenuation alone naturally produces, and it coexists with a null p. Two readings fit the data equally well: genuine de-attenuation from aligning the information window with the outcome window, or a low-power estimate wandering. The battery cannot separate them. Do not present the magnitude growth as corroboration without stating that the test does not reject zero.

**Four further disclosures.** (i) `build_sagg_panel.py`'s docstring asserts the AR(1) innovations are "serially uncorrelated by construction, so the quarter's total news surprise is their sum", while the script's own diagnostic prints corr = 0.357 with the inline comment "(should be ~0)". The diagnostic is right and the docstring is wrong; it would not survive external review as written. (ii) No `sagg_vce_diag.csv` exists, so B9 cannot be reported here, which matters because both D3 regressions emitted the non-PSD/CGM warning that such a file would flag. (iii) The circular-shift resolution floor is 1/82 = 0.0122, not binding at 0.2195 but a hard limit on how small this test's p can ever be. (iv) The SEs, joint Wald and cumulative figures above live only in the transient Stata log; no results CSV persists them.

### 7.3d Shock-construction menu (`build_shock_menu.py`, `run_shock_menu.do`, `run_ri_shockmenu.py`, 2026-08-04)

The headline S_t has three documented defects. It is serially correlated at quarterly frequency (P1a: acf1 = +0.2707, Ljung–Box Q(1) p = 0.0125), because a monthly level-AR(1) under-cleans GPR persistence. It uses only the `USA|China` direction of a bilateral index that also publishes `China|USA` (P1b: level correlation between them 0.5632 full sample, 0.5755 on months ≤ 2023-12). And its AR is fit on the full series including 2024–2026, a look-ahead in a generated regressor (P1c, the F7 disclosure). This subsection replaces all three open threats with measurements.

**Eleven columns, one validation gate.** All variants are built at monthly frequency (except D, which aggregates first), brought to the 82-quarter panel grid, then standardized to mean 0 and sd 1 over those quarters so β₃ magnitudes are comparable. The gate column `A_baseline_repro` must reproduce the shipped `shock_us_cn` to max|diff| ≤ 1e-8; it achieves **1.776×10⁻¹⁵** against both the frozen 265-quarter parquet and the panel column. The tolerance is hard-coded at `build_shock_menu.py:180` and was not tuned to the observed value. Nothing existing was modified; the frozen parquet and `c6_panel.dta` were read-only and verified byte-unchanged afterwards.

| variant | direction | spec | agg | no look-ahead | sd_raw | acf1 | LB(4) p | LB(8) p | corr w/ baseline |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_existing` | USA\|China | level AR(1), full-sample | qtr-end month | no | 2.578 | **+0.2707** | **0.0228** | **0.0040** | 1.000 |
| `A_baseline_repro` (gate) | USA\|China | level AR(1), full-sample | qtr-end month | no | 2.578 | +0.2707 | 0.0228 | 0.0040 | 1.000 |
| `B_i_lvl_ar1_nla_qend` | USA\|China | level AR(1), fit ≤ 2023-12 | qtr-end month | yes | 2.590 | +0.2676 | 0.0260 | 0.0028 | 0.99961 |
| `B_ii_lvl_ar1_nla_q3sum` | USA\|China | level AR(1), fit ≤ 2023-12 | 3-month sum | yes | 3.970 | +0.3128 | 3.3e-12 | 2.2e-18 | 0.436 |
| `C_i_dgpr_ar4_nla_qend` | USA\|China | AR(4) on Δmonthly GPR | qtr-end month | yes | 2.413 | −0.0309 | 0.2534 | 0.0792 | 0.880 |
| `C_ii_dgpr_ar4_nla_q3sum` | USA\|China | AR(4) on Δmonthly GPR | 3-month sum | yes | 3.850 | −0.2136 | 0.2958 | **0.1319** | 0.320 |
| **`D_q_ar1_nla` (WINNER)** | USA\|China | quarterly MEAN then AR(1), fit ≤ 2023Q4 | native quarterly | yes | 1.786 | **−0.1288** | **0.3544** | 0.0848 | 0.254 |
| `E_C_i_bidir` | sum of both directions | AR(4) on Δmonthly GPR | qtr-end month | yes | 2.815 | +0.0317 | 0.1169 | 0.0401 | 0.829 |
| `E_C_i_cnus` | China\|USA | AR(4) on Δmonthly GPR | qtr-end month | yes | 0.887 | +0.1342 | 0.1730 | 0.1486 | 0.170 |
| `E_B_i_bidir` | sum of both directions | level AR(1), fit ≤ 2023-12 | qtr-end month | yes | 3.032 | +0.2723 | 0.0138 | 0.0032 | 0.939 |
| `E_B_i_cnus` | China\|USA | level AR(1), fit ≤ 2023-12 | qtr-end month | yes | 1.003 | +0.2610 | 0.1006 | 0.3845 | 0.266 |

Level-AR(1) does not whiten GPR at quarterly frequency, with or without look-ahead: B-i is essentially the baseline and carries the same +0.27 acf1. The 3-month sum of a level-AR(1) residual is the worst column in the menu, with LB p ≈ 0 at every lag, because summing re-injects the persistence the AR failed to remove. Differencing first (C) or aggregating first (D) is what actually whitens.

**The pre-registered selection rule, verbatim from the `build_shock_menu.py` header, written before any regression ran:**

> "the preferred NEW construction = the no-look-ahead variant with the whitest QUARTERLY residual series on the 82 panel quarters, judged by LB(4) p (tie-break: LB(8) p, then acf1 magnitude), within the USA|China direction; the direction axis (bidirectional vs directional) is reported as a parallel column set, not selected on outcomes. beta3 results play NO role in selection. Baseline S_t stays the headline-continuity column regardless (its defects are disclosed, not hidden)."

The candidate pool is exactly {B-i, B-ii, C-i, C-ii, D}; A and `baseline_existing` are excluded by look-ahead and the four E columns by direction. Sorting on LB(4) p descending gives no ties (0.3544 > 0.2958 > 0.2534 > 0.0260 > 3.3e-12), so the winner is **`D_q_ar1_nla`**: quarterly mean of monthly `USA|China` GPR, then AR(1) at quarterly frequency, fit on quarters ≤ 2023Q4, residual native quarterly.

**Provenance of the selection, on disk.** `shock_menu_preferred.txt` and `shock_menu_diagnostics.csv` were written at 13:13:46; `shockmenu_vce_diag.csv` at 13:14:55 and the results CSV after that. β₃ appears nowhere in the selection block, and the `.do` file reads the winner from `shock_menu_preferred.txt` rather than a hard-coded macro, echoing the cross-check.

**Honest caveat, keep it in the paper.** The winner is rule-dependent. D wins on LB(4) p, but its LB(8) p of 0.0848 is *worse* than the runner-up C-ii's 0.1319, and an |acf1|-first rule would have selected C-i. The pre-registered rule is LB(4)-first and was applied as written, so the ordering stands, but higher-order dependence in the selected series is not fully cleaned. Do not describe the selected shock as white.

**Re-derivation of prior external measurements.** Six of eight quoted diagnostics reproduce from the raw CSV (`shock_menu_rederivation_check.csv`). Two do not: D's acf1 (prior −0.029, re-derived **−0.1288**) and D's LB(8) p (prior 0.017, re-derived **0.0848**). Confirmed by refit, the prior numbers came from a **full-sample look-ahead** quarterly AR(1); refitting that way returns −0.0292 and 0.0175 exactly. Variant D as built is no-look-ahead, so the re-derived values govern.

**Results.** `reghdfe dw us_cn us_cn_shock_v, absorb(fq gq ig) vce(cluster firm_n rd_m)` (3-pairwise, N = 347,690, 6,634 firm clusters, 82 month clusters, df_r = 81) and the `absorb(fq gq)` companion (N = 348,156). RI mirrors `run_ri_3pairwise.py`: 5,000 free permutations, seed 20260702, 81 circular shifts, 174,078 firm-quarters, zero quarters dropped, `arbiter_p = p_circ`. β₃ in units of 10⁻⁶.

| variant | β₃ 3pw | SE | CRVE p 3pw | β₃ it+gt | CRVE p it+gt | RI free | **RI circ (arbiter)** | verdict |
|---|---|---|---|---|---|---|---|---|
| `baseline_panel_shock_raw` (= current headline) | −0.528 | 1.743 | 0.7627 | −1.073 | 0.5570 | 0.8154 | **0.8293** | null |
| `s_baseline_existing` = `s_A_baseline_repro` | −1.361 | 4.494 | 0.7627 | −2.765 | 0.5570 | 0.8154 | **0.8293** | null |
| `s_B_i_lvl_ar1_nla_qend` | −1.227 | 4.468 | 0.7842 | −2.635 | 0.5745 | 0.8348 | **0.8659** | null |
| `s_B_ii_lvl_ar1_nla_q3sum` | −6.734 | 3.951 | 0.0922 | −6.339 | 0.0969 | 0.2228 | **0.3293** | null (CRVE marginal) |
| `s_C_i_dgpr_ar4_nla_qend` | −2.916 | 3.783 | 0.4431 | −4.004 | 0.3490 | 0.6089 | **0.5732** | null |
| `s_C_ii_dgpr_ar4_nla_q3sum` | −6.252 | 2.761 | **0.0262** | −7.054 | **0.0179** | 0.2611 | **0.2805** | null under arbiter |
| **`s_D_q_ar1_nla` (PREFERRED)** | **−7.557** | 3.452 | **0.0315** | −8.031 | **0.0195** | 0.1780 | **0.2439** | null under arbiter |
| `s_E_C_i_bidir` | −1.530 | 3.636 | 0.6750 | −2.751 | 0.4908 | 0.7822 | **0.7561** | null |
| `s_E_C_i_cnus` | +3.515 | 2.387 | 0.1446 | +2.620 | 0.2368 | 0.5299 | **0.5244** | null |
| `s_E_B_i_bidir` | +0.448 | 4.063 | 0.9125 | −0.954 | 0.8217 | 0.9402 | **0.9146** | null |
| `s_E_B_i_cnus` | +3.975 | 2.741 | 0.1509 | +3.424 | 0.1762 | 0.4621 | **0.4512** | null |

Lead spec (dv = `dw_lead1`), run for baseline and preferred only: baseline +0.177×10⁻⁶ (3pw p 0.890) / −0.332×10⁻⁶ (it+gt p 0.790); preferred +2.994×10⁻⁶ (3pw p 0.363) / +1.931×10⁻⁶ (it+gt p 0.499). β₂ is null everywhere, p ∈ [0.68, 0.98] across all 12 columns.

**Reading.** Zero of twelve RI columns rejects at 0.05 under either free permutation or circular shift; the minimum arbiter p is **0.2439**, at the preferred variant. Two columns cross CRVE 0.05, `s_D_q_ar1_nla` (0.031 / 0.020) and `s_C_ii_dgpr_ar4_nla_q3sum` (0.026 / 0.018), and both dissolve under RI. They are **not two independent hits**: they correlate 0.892 with each other and only 0.254 and 0.320 with the baseline, so treat them as roughly one column. Both lean **negative**, which is the H2.1 direction, and these are the only shock constructions in this document that cross CRVE 0.05 in that direction; the lagged shock (§7.3b) and the aggregated shock (§7.3c) also lean negative on CRVE without crossing. The corroborating check goes against them: the preferred variant's lead spec is null and sign-flipped (+2.994×10⁻⁶ against −7.557×10⁻⁶ contemporaneous), which a real lagged disengagement effect would not do. B-ii is the mechanical warning, with the third-lowest CRVE p in the menu (0.092) and the least white residual in the menu (LB(8) p = 2×10⁻¹⁸): serial correlation inflating apparent CRVE significance is visible directly in the data.

**Direction axis (P1b), closed.** All four direction columns are null on both CRVE and RI. The bidirectional sum with the C-i spec gives −1.530×10⁻⁶ (RI circ 0.756) and correlates 0.949 with directional C-i; with the B-i spec it gives +0.448×10⁻⁶ (RI circ 0.915) and correlates 0.939 with the baseline. `China|USA` alone flips sign in both specs and is null in both. The sign flip is not a finding; it is what a null column with a different noise realization looks like. The bidirectional sum is the conceptually-matching construction and it gives the same answer as the baseline, so P1b moves from an open threat to a tested and closed one. Note the sum is dominated by the US-initiated leg (0.94–0.95 correlation with the directional column, 0.52–0.56 with `China|USA` alone).

**Look-ahead (P1c), closed, and F7's deferred item superseded.** `B_i_lvl_ar1_nla_qend` is the baseline construction with the look-ahead physically removed. It correlates 0.99961 with the baseline and moves β₃ from −1.361×10⁻⁶ (p 0.763) to −1.227×10⁻⁶ (p 0.784). The look-ahead is worth about 1×10⁻⁷ in β₃ and nothing in inference. **The AR specification, not the look-ahead, was the defect.** This supersedes the deferred `US·CN·gpr(M2)` exact-nesting column of F7, whose only purpose was to make the same argument by proxy.

**What the menu does and does not license.** Licensed: the null is robust to the AR specification (level AR(1), AR(4) on first differences, quarterly-frequency AR(1)), to the aggregation rule (quarter-end month versus within-quarter sum), to removing the full-sample look-ahead, and to the bilateral direction; and P1a is real, measured, and reduced in the preferred variant (LB(4) p 0.023 → 0.354) without changing the answer. **Not licensed:** any claim that the preferred variant produces a significant negative effect (it does on CRVE, it does not under the arbiter, and the arbiter governs), and any claim that the selected shock is white (LB(8) p = 0.085).

**Reporting rule adopted.** The baseline stays the headline-continuity column everywhere. `s_D_q_ar1_nla` becomes the primary shock-construction robustness column and is always reported with β₃, CRVE p **and** RI p_circ on the same row, plus the rule-dependence caveat. The full diagnostics table is appendix material. The direction axis is a two-row appendix block, not a headline column.

**Execution health.** 28 result rows, all `status = ok`, zero `reghdfe` failures. `shockmenu_vce_diag.csv` has **56** data rows, every one `se_valid = 1`, so no degenerate two-way-cluster VCE. Its schema is `variant,dv,fe,coef,b,se,se_valid,smoke,n_firms`, so `se_valid` is **column 7, not the last**; a naive last-column parse falsely flags all 56 rows as invalid. The drift gate against `headline_3pairwise_canonical.csv` printed PASS on both FE sets (relative error 6.5×10⁻⁸ and 7.0×10⁻⁸).

### 7.4 Conditional "spell-boundary" sample (advisor request) — including an error we made and fixed

**Design intent:** keep held quarters + one boundary zero at each entry/exit, conditional on firm-groups the group ever engaged; drop deep never-held zeros and never-held firms.

**Error (first build, `build_spell_boundary.py` / `07f`, SUPERSEDED).** Selection was done **per (firm, group)**. That broke the US-vs-NONUS pairing: many firm-quarters retained only one group → singletons dropped by α_{i,t} → the estimate degenerated to "US's own change in selected firms," not "US relative to non-US." **Do not use.** On the P0 panel the biased spec now returns β₃ = **−2.6226×10⁻⁶** (SE 4.3314×10⁻⁶, p 0.5466, N 209,056 after 30,870 singleton drops) and, with firm×group, −2.3654×10⁻⁶ (SE 4.3717×10⁻⁶, p 0.5899, N 208,874). *(Pre-P0: +4.38×10⁻⁶ (5.66), p 0.441, N 250,918; S2 +4.32×10⁻⁶, p 0.4517.)* **Power note owed if 07f is ever cited:** its estimation N fell 16.7% under P0 (250,918 → 209,056) and firm clusters fell to 3,939, while the risk-set samples *grew*. Also do not quote the 239,926 build count and the 209,056 regression N interchangeably; the gap is the 30,870 singleton drop that the per-group selection creates.

**Correction (`build_spell_riskset.py` / `07g`).** Define the risk set at the **firm-quarter** level: (i,t) enters if EITHER group has a spell-boundary there (held at t, t−1, or t+1); then keep **both** group rows. This preserves pairing.

Build output on the P0 panel: 898,566 risk-set rows before the drop, **balanced 135,398 US / 135,398 NONUS** in the estimation subset, 100% paired (449,283 of 449,283 exactly-two-row firm-quarters, 0 unpaired). Estimation subset (drop missing dw/cn_lag/shock) = **270,796 rows, 5,625 firms, 0 group singletons** (pre-P0: 268,282 / 5,564; pre-B7 all-`rel_type`: 342,262 / 6,355).

| | R1 headline (fq gq) | R2 + firm×group | F8 lag-only (no look-ahead) |
|---|---|---|---|
| β₃ | **−1.7300×10⁻⁶** (2.8399×10⁻⁶), p=0.5441 | **−1.7063×10⁻⁶** (2.9983×10⁻⁶), p=0.5709 | **−1.7674×10⁻⁶** (2.90×10⁻⁶), p=0.5443 |
| N | 270,796 (5,625 firms, 0 singletons) | 270,386 (410 singletons) | 268,800 (5,613 firms) |
| (pre-P0) | +3.26×10⁻⁶ (2.24), p 0.1505, N 268,282 | +3.51×10⁻⁶ (2.38), p 0.1432, N 267,898 | +3.35×10⁻⁶, p 0.153, N 265,710 |

Still null, cleanly identified within firm-quarter. **The narrative here has to change, not just the numbers.** Pre-P0 all five specs in this family were positive and this was written up as a consistently-corroborative pattern. Post-P0 all five flip negative (07g R1 and R2, F8, and both 07f columns) and every p sits in 0.54–0.59. Nothing was significant before or after, so no conclusion is overturned; what is gone is the coherent sign pattern that made the risk set look like it was pulling in one direction. The one substantive point that survives is the F8 comparison: the lag-only estimate (−1.7674×10⁻⁶) is essentially the with-lead estimate (−1.7300×10⁻⁶), so lead-conditioning is not driving anything.

Lower-order β₂ (US·CN) for the record: R1 −1.2244×10⁻⁶ (p 0.8437), R2 −1.8065×10⁻⁶ (p 0.8154), F8 −1.22×10⁻⁶ (p 0.846), 07f S1 −5.0172×10⁻⁶ (p 0.5611), 07f S2 −1.6904×10⁻⁶ (p 0.8721).

**Trade-off to disclose:** conditioning on engagement is closer to "real reallocation behavior," but because US engages fewer firm-quarters, the paired risk set discards deep zeros and the identifying variation narrows; the estimand is "within engaged firms" rather than "across all holdings-observed European issuers."

**No design-based arbiter exists anywhere in this family.** Neither `07g_spell_riskset.do`, nor `07f_spell_boundary.do`, nor `run_audit_f4f8.do` contains a permutation block, so eight specs here rest on CRVE alone. That is a method gap, not a missing output file; supplying an RI arbiter would require new code. No `*_vce_diag.csv` is produced either, so B9 is not applicable rather than passing. F4a/F4b/F4c and F8 SEs and p-values persist only in the transient Stata log.

### 7.5 Country-pair subsample (`run_audit_f4f8.do`, `07b`/`07c`) — GB+DE+FR, replaces S_t with country-specific S_{c,t}

N = **172,920** of a 1,315,200-row country-pair panel (GB 799,400 / DE 260,200 / FR 255,600; 795 shock rows, 0 unmatched). Because S_{c,t} varies across listing country within (g,t), β₁ (US_g × S_{c,t}) becomes separately identified (3 coefficients instead of 2).

| coefficient | clustering | estimate | SE | p | (pre-P0) |
|---|---|---|---|---|---|
| β₁ (US·S_{c,t}) | firm × quarter | **−1.36×10⁻⁶** | 1.20×10⁻⁵ | **0.9103** | +2.22×10⁻⁵, p 0.086 |
| β₁ | country (3 clusters, df_r=2) | −1.36×10⁻⁶ | 1.69×10⁻⁵ | **0.9432** | p 0.246 |
| β₁ | country × quarter | −1.36×10⁻⁶ | 1.80×10⁻⁵ | **0.9467** | p 0.242 |
| β₃ (US·CN·S_{c,t}) | three levels | −8.13×10⁻⁶ | — | 0.764 / 0.521 / 0.408 | −1.21×10⁻⁶, p 0.93–0.97 |
| β₂ (US·CN) | firm × quarter | +8.42×10⁻⁸ | — | 0.984 | — |

**The clustering audit that used to carry this section no longer discriminates.** Pre-P0 the argument was that β₁'s nominal p=0.086 under firm×quarter clustering was an artifact of the wrong level, since S_{c,t} varies only across 3 listing countries × quarter, and that clustering at the country level moved it to 0.246. On the P0 panel that argument is moot: the point estimate fell about 16-fold and flipped sign, it is identical at all three clustering levels, and all three p-values exceed 0.91. β₁ is dead, not fragile. It is not evidence of anything and is not cited as directional support (Second review round, F4). β₃ is likewise null at every level.

Three pre-existing limits stay on the record. **87% of the country-pair panel is unestimable** (172,920 of 1,315,200; `cn_lag` missing on 1,142,280 rows), which is why the panel size and the reported N differ so widely. The 3-country specs draw small-cluster warnings ("missing F statistic … too few clusters", df_r = 2) and F4c required a CGM repair; those are design limits of 3 country units, not P0 artifacts. The country-pair AR(1) coefficients are frozen across P0 (`cmp` proves both country-pair shock CSVs byte-identical to their pre-P0 backups): US a=0.693556 b=0.657211 n=796; GB a=0.040330 b=0.234908; DE a=0.010026 b=0.049298; FR a=0.013404 b=0.469693.

---

### 7.6 Shares-based robustness: ownership share (`build_ownership_share_panel.py`, `build_ownership_share_c6_panel.py`, `run_ownership_share.do`)

**Why this exists.** The main outcome w = H(USD)/T(USD) is a market-value portfolio weight, so it mixes trading flow with price moves and portfolio-denominator reallocation. A null on w supports "US do not reduce their within-Europe portfolio *weight*", but not the stronger "US do not *sell* / do not reduce their *stake*". For the flow claim we use a pure quantity — the group's ownership share of each firm:

- ownership_share_{i,g,t} = ( Σ_{b∈g} shares held on the primary EQ class ) / (primary-EQ shares outstanding).

It is immune to price (numerator and denominator are both in shares) and to portfolio-denominator reallocation (no T term). Shares are not additive across firms, so a shares-based *portfolio weight* is meaningless; the ownership share is the correct object. `adj_shares_out` is a per-security-class attribute, so we restrict both numerator and denominator to the **primary EQ class** (`fsym_id = fsym_primary_id`, matching the market-cap rule in `04_us_ownership_european.jl`); this makes shares_out constant within each security-quarter (verified: dispersion drops from 5.28% of cells to exactly 0). The **primary outcome is the pure trading FLOW** `(held_t − held_{t−1}) / out_{t−1}` (fixed lagged float; Second review round, F6), on the **same C6 grid, three-way pairwise FE, and clustering** as the main spec. (An earlier draft used Δ(ownership_share) = held_t/out_t − held_{t−1}/out_{t−1} with each term over its *own* current float; that is not float-immune — a buyback/issuance moves it with zero trading — so it is retained only as a labelled comparison column.)

**Result (FLOW, raw fraction-of-float units — NOT ×10⁻⁶; P0 panel, 2026-08-04).** N = **248,024** rows / **5,191** firms / 82 quarters, balanced 124,012 per side (3-pairwise: 247,582). The ownership observed panel behind it is 582,463 firm-group-quarter cells over 11,732 firms and 100 quarters, 1999Q1–2023Q4; `shrout` is missing on 0.1835% of cells, `os>1` is impossible in 31 cells (0.0053%), and os has median 0.0325 / p90 0.1713 / p99 0.3800. Flow mean 1.016×10⁻², sd 3.674.

**The MAIN column is the winsorized one, and it is null.**

| coefficient | fq+gq (comparison) | **3-pairwise (headline FE)** | (pre-P0 3pw) |
|---|---|---|---|
| **β₃ winsorized p1/p99 — MAIN** | **+7.5197×10⁻⁵** (SE 2.9005×10⁻⁴, p 0.796) | **+7.4816×10⁻⁵** (SE 3.4026×10⁻⁴, CRVE p 0.8265, **RI p 0.913**) | +0.000649, p 0.056, RI 0.38 |
| β₃ raw — disclosed, outlier-dominated | −3.3293×10⁻² (SE 2.6793×10⁻², p 0.2176) | −3.6143×10⁻² (SE 3.0312×10⁻², CRVE p 0.2366, RI p 0.021) | **+8.90×10⁻⁴, CRVE p 0.0001**, RI 0.3691 |
| β₃, old Δ(ownership_share) `dos` | −8.5888×10⁻⁵ (SE 3.4349×10⁻⁴, p 0.8032), `se_valid=1` | — | +0.000549, `se_valid=0` |
| held-only (os>0, re-paired) — **raw, outlier-dominated** | −5.4410×10⁻² (p 0.2031, N 196,050) | −6.0805×10⁻² (p 0.2234, N 195,742) | +0.00135, CRVE p 0.0018 |

R²: 0.8142 (r1) / 0.8200 (r2) / 0.6179 (r3) / 0.8142 and 0.8214 (held-only) / 0.6385 and 0.6425 (winsor). Winsorizing changed 2,480 observations per tail. Held-only dropped 26,011 extensive-margin zeros and then 25,963 unpaired rows. β₂ (US·CN) is null throughout: −1.4336×10⁻² (r1, p 0.3671), +1.2669×10⁻³ (r2, p 0.91), −6.8353×10⁻⁴ (r3, p 0.5175), −2.9582×10⁻⁴ and −2.2734×10⁻⁴ on the winsorized columns.

**What died here.** The old §7.6 headline was **+0.00089 with CRVE p = 0.0001**, presented as positive and opposite to disengagement, with the CRVE labelled fat-tail-deflated and RI (0.37) the arbiter. On the P0 panel the identical spec returns **−0.036 with p = 0.2366**: a sign flip and a roughly 40× magnitude change. Every CRVE spec in the family is now null, and every **unwinsorized** column (r1, r2, dos, held-only) flips negative; the MAIN winsorized column is positive and tiny (+7.48×10⁻⁵) and equally null. So the sentence "US investors' stake response is positive, opposite to disengagement" is retracted. The correct statement is narrower and rests on the winsorized column: **the shares-based test finds no differential US stake adjustment, in either direction, and it agrees with the w-based null.**

**Not a zero-fill artifact — a consistency check, not a claim.** Dropping the extensive-margin zeros (observed-held-only, re-paired) and re-running the triple difference on the FLOW outcome gives −5.44×10⁻² (fq+gq, p 0.2031) and −6.08×10⁻² (3-pairwise, p 0.2234), both null. Both are raw-scale columns and are subject to the open outlier issue below, so this is a consistency check rather than a claim; no winsorized held-only column was run this cycle. *(Pre-P0 this row read +0.00135 with CRVE p = 0.0018 and was described as "further from the disengagement prediction"; that reading is dead with the raw column it came from.)*

**OPEN ISSUE — the raw flow is outlier-dominated post-P0, and it is not resolved.** The as-of rule of §5.0 admits firm-quarters with extreme flow ratios. Post-P0 the raw flow carries **kurtosis 198,985** (pre-P0 about 5,000), sd 3.674, **max 1724.9× float** (pre-P0 max 9.8), min −0.9728, against p99 = 0.1387 and p1 = −0.1364. The consequences are visible and consistent:

- Winsorizing at p1/p99 moves the 3-pairwise estimate from −3.61×10⁻² to +7.48×10⁻⁵. That is a sign flip and a ~480× magnitude collapse, which proves the raw point estimate is a handful of tiny-lagged-denominator cells rather than a population feature.
- **All three raw RI cells land at ≈0.02** (flow raw 0.0206, flow_common raw 0.0198, flow_r raw 0.0196). Near-identical p-values across three different outcomes are the signature of the same few quarters driving all of them, not three independent findings.
- The raw `flow_r` coefficient is −1.07, i.e. −107% of float per unit of CN×S, which is not an economically possible effect size.
- `flow_common` raw returns a CRVE SE of exactly 0 and is B9-flagged degenerate.

Reporting rule until this is fixed: **winsorized is MAIN and null; raw is disclosed as outlier-dominated and carries no claim.** That follows the pre-existing fat-tail convention, it is not a post-hoc choice. The fix owed is to identify the monster cells (most likely a stale tiny lagged float meeting an as-of-recovered holdings numerator) and write a **predetermined BAD-rule extension**, never an outcome-driven trim. It is not written yet.

**Estimand (accounting decomposition of observed institutional and residual ownership flows; rewritten 2026-08-04).** We interpret β₃ as the differential ownership flow of US institutions relative to non-US institutions, rather than the gross response of US investors in isolation. At the firm-quarter level flow_US + flow_NONUS + flow_R equals the growth in float, so any US repositioning is measured net of the common institutional flow and of the offsetting flow of an unobserved residual sector (retail, insiders, non-reporting institutions, strategic holders). A null on the differential is therefore informative about relative US repositioning, not about whether any investor traded.

**The co-movement anchor this paragraph used to rest on did not survive P0 and is withdrawn.** Earlier drafts stated that in the high-exposure tercile the two reporting groups move together, with corr(flow_US, flow_NONUS) ≈ +0.20. On the P0 panel that correlation is **+0.027** winsorized and **+0.008** on the stable-float subsample (`flow_decomposition_diag.csv`, high-CN tercile, 6,813 and 5,160 firm-quarters). The raw-sample figure is +0.073, and the full-sample raw figure of +0.995 is a fat-tail artifact of the same monster cells, not a co-movement fact. We therefore no longer claim the two groups move together in the high-exposure tercile. Whether the anchor is genuinely gone or is another casualty of the outlier cells is **an open investigation**, tied to the BAD-rule item above. Because these are quantity data, they still cannot distinguish a world in which US demand does not change from one in which it changes but is absorbed by price; that distinction rests on the H2.2 price evidence, and we do not claim the finding holds under both.

**Decomposition test (weak-FE, descriptive; `run_flow_decomposition.py` + `run_flow_decomp_step3.do`, P0 rerun 2026-08-04).** The identity flow_US + flow_NONUS + flow_R = float growth holds to **3.6×10⁻¹²** on the P0 panel. Because CN_{t−1}×S_t is a firm-quarter attribute, this step can only use firm + quarter FE (a firm×quarter FE would absorb the treatment), so it is a deliberately weaker, descriptive design; inference is anchored by the design-based RI (5,000 quarter-shock permutations, seed 20260702, 124,012 firm-quarters).

| cell | β₃ winsor (MAIN) | RI p | β₃ raw | RI p raw |
|---|---|---|---|---|
| flow_r (residual sector) | −1.1104×10⁻⁴ | **0.913** | −1.0726 | 0.020 |
| flow_common | +3.4568×10⁻⁴ | **0.641** | −7.5704×10⁻² | 0.020 |
| flow_diff (US − NONUS) | +1.5285×10⁻⁴ | **0.833** | −3.6143×10⁻² | 0.021 |

All three winsorized MAIN cells are null. The CRVE companion (`flowdecomp_results.csv`, N 123,791) reproduces them: flow_r −1.11×10⁻⁴ (p 0.9018), flow_common +3.46×10⁻⁴ (p 0.5369), flow_diff +1.53×10⁻⁴ (p 0.6593). The two engines agree because the Python producer ships the winsorized outcome columns, so both read bitwise-identical data. The raw companions are the outlier cells described above, including `flow_common` raw with SE = 0. Neither the residual sector nor the common institutional flow responds to CN×S on the MAIN outcome, and the US-minus-non-US differential remains null in flow form. *(Pre-P0, superseded: winsorized flow_R +3.81×10⁻⁴ RI 0.69, flow_common +3.31×10⁻⁵ RI 0.97, flow_diff +6.65×10⁻⁴ RI 0.41.)*

**B9 status.** 16 rows across `ownshare_vce_diag.csv`, `flowheldonly_vce_diag.csv` and `flowwinsor_vce_diag.csv`, **zero `se_valid=0`**. Non-PSD/CGM warnings fired on r1 and on winsorized fq+gq. **Vintage flip to note:** pre-P0 the `dos` comparison column was the pre-registered degenerate case; on the P0 panel it returns a finite SE and is `se_valid=1`. The vintage-dependence warnings written into the `.do` comments now point at the wrong spec. Separately, `build_ownership_share_panel.py` line 204 still writes `ref_main_panel_firms=6854` into `ownership_share_diagnostics.csv`; the canonical count is **6,867**.

**Caveat 1 (ADR / non-primary exclusion).** The measure is primary-EQ only, so it does not observe stake adjustment through ADR/GDR or non-primary classes. On the P0 panel US investors hold **8.79%** of their European exposure off the primary class versus **3.22%** for non-US, a 2.73× asymmetry (on-primary shares 91.2144% and 96.7794%; pre-P0 8.92% / 3.32%). The shares-based null is therefore *complementary* to the USD portfolio-weight main spec, which does capture the ADR channel, not a substitute. An ADR-inclusive measure needs the ADR conversion ratio and is deferred (§10).

**Caveat 2 (power / MDE).** With 82 quarter-clusters the design rules out *large* stake reductions but has limited power against small ones. Using the MAIN winsorized 3-pairwise CRVE SE of 3.40×10⁻⁴ and σ_S = 2.578, the 80%-power MDE for a representative firm-quarter (CN ∈ [0.05, 0.15], 1σ shock) is on the order of **1.2–3.7 bps of float** (about 2.5 bps at CN = 0.10), well below the ~6 bps that 2.8·SE implies at the non-existent CN·S = 1 point (Second review round, F5). *(Pre-P0 this caveat quoted 0.8–2.3 bps off the raw-flow SE of 2.14×10⁻⁴.)* The quarter-end-only shock is additionally classical measurement error that attenuates β₃ and enlarges the true MDE. Read the estimate as bounding the effect near zero, not as proving an exact zero.

**Sample-comparison caveat.** The flow universe is 5,191 firms against the main panel's 6,867, because the primary-EQ float restriction and the BAD firm-quarter rule bind harder. Whether that coverage gap itself moved with P0 was **not re-derived this cycle**, so any sentence comparing the two samples is provisional on that check.

---

### 7.7 Direction split — buy-side vs sell-side China exposure (`run_direction_split.do`, `run_ri_direction.py`)

The B7 exposure `cn_lag` sums two link types with opposite economic content: **sell-side** (the firm is a China customer's supplier, i.e. it *sells to* China — revenue exposure) and **buy-side** (the firm is a China supplier's customer, i.e. it *buys from* China — input dependence). They split additively (`sell_lag + buy_lag = cn_lag`, §5.4). If disengagement acts through only one channel, pooling them into a single `cn_lag` could dilute it. It does not:

| channel | β₃ (link-share) | SE | CRVE p | RI p | indicator β₃ (SE, p) | (pre-P0 β₃ / p / RI) |
|---|---|---|---|---|---|---|
| sell-side (US·sell_lag·S_t) | **−1.9639×10⁻⁶** | 1.3749×10⁻⁶ | **0.1570** | **0.4735** | −1.2825×10⁻⁶ (1.1067×10⁻⁶, 0.2499) | +2.53×10⁻⁷ / 0.6181 / RI 0.931 |
| buy-side (US·buy_lag·S_t) | **+1.0131×10⁻⁶** | 2.7750×10⁻⁶ | **0.7160** | **0.6701** | −9.241×10⁻⁷ (2.8035×10⁻⁶, 0.7425) | +5.50×10⁻⁶ / 0.1473 / RI 0.136 |
| sell − buy (contrast) | −2.977×10⁻⁶ | 2.58×10⁻⁶ | 0.253 | **0.3171** | −3.58×10⁻⁷ (1.94×10⁻⁶, 0.854) | −5.25×10⁻⁶ / RI 0.2046 |
| joint β₃ₛ = β₃ᵦ = 0 | F(2,81) = 1.51 | — | 0.2281 | — | — | not reported |
| **pooling validity** (β₂ₛ=β₂ᵦ ∧ β₃ₛ=β₃ᵦ) | **F(2,81) = 0.77** | — | **0.4651** | — | — | p 0.424 |

Lower-order terms: US·sell +5.4292×10⁻⁶ (8.8654×10⁻⁶, p 0.5420), US·buy −2.1217×10⁻⁶ (8.2310×10⁻⁶, p 0.7972); indicator US·dsell −6.1699×10⁻⁸ (p 0.9907), US·dbuy −7.6830×10⁻⁶ (p 0.4751). N = **347,690**, R² 0.5908 (pre-P0 347,490 / 0.6244); 466 singletons, 6,634 firm clusters, 82 month clusters; fq groups 173,845, gq 164, ig 13,268. Model fit F(4,81) = 0.81 (p 0.5198) for MAIN and 0.90 (p 0.4696) for the indicator spec. RI uses 5,000 permutations on 174,078 firm-quarters. The `indicator` column replaces the link-share with a 0/1 exposure flag, which is immune to reciprocal double-records; it is null in both channels too.

**The narrative reverses, and the reversal is the point.** Pre-P0 the buy arm carried whatever suggestive signal this family had (+5.50×10⁻⁶, RI p = 0.136) and the sell arm sat at essentially zero, which supported a "buy-side tilt, anti-decoupling" reading. Post-P0 the buy arm collapses to +1.01×10⁻⁶ (p = 0.716, RI 0.670) and the **sell** arm becomes the larger-|t| one at −1.96×10⁻⁶ (p = 0.157, RI 0.474). Both are null and the contrast is null (RI 0.317). **Any sentence asserting a buy-side directional tilt is withdrawn**, and it is not replaced by a sell-side one: the arms simply swapped ranks inside the noise, which is what a family of nulls does when the data changes underneath it.

Two structural facts are unaffected by P0 and were independently re-derived exact. The channels are near-orthogonal: corr(sell_lag, buy_lag | cn_lag>0) = **−0.1353** on 24,346 exposed firm-quarters, with cells sell-only 11,429, buy-only 8,533, both 4,384, neither 0. And **pooling remains licensed**: the joint test of β₂ₛ=β₂ᵦ and β₃ₛ=β₃ᵦ is not rejected (F(2,81)=0.77, p=0.4651), so the pooled single-`cn_lag` headline is still the right object. The offset alternative, a negative sell channel cancelled by a positive buy channel, is not supported either, since neither arm is distinguishable from zero.

B9: `direction_vce_diag.csv`, 4 rows, all `se_valid=1`. **Open gap:** there is no design-based RI for the INDICATOR spec, which is precisely the spec whose CRVE required the Cameron–Gelbach–Miller non-PSD repair. Note also that `run_ri_direction.py` lines 111–113 hard-code pre-P0 drift anchors, so the script prints a large apparent mismatch on this vintage; that mismatch is the result, not an error.

### 7.8 Four-group active/passive split — is the null a passive-fund dilution artifact? (`build_fourgroup_panel.py`, `run_fourgroup.do`, `run_ri_fourgroup.py`)

A natural worry about the pooled null: index/passive funds mechanically do not reallocate on news, so pooling them with active managers could dilute a real active-manager disengagement toward zero. We split both US and NONUS into **ACTIVE** and **PASSIVE** managers (FactSet funds master) and re-run the triple difference. **The active/passive labels come from a ~2018-08 master snapshot and are predetermined only at/after 2018m8, so the post-2018 subsample is the PRIMARY report and the full-period columns carry a look-ahead caveat.**

| contrast | full-period β₃ (SE, CRVE p, RI p) | **post-2018 β₃ (PRIMARY; SE, CRVE p, RI p)** | (pre-P0 full / post-2018) |
|---|---|---|---|
| **US_ACTIVE vs NONUS_ACTIVE (MAIN)** | **−7.0647×10⁻⁷** (1.7270×10⁻⁶, p 0.6836, **RI 0.7802**) | **+5.0479×10⁻⁷** (1.7146×10⁻⁶, p 0.7713, **RI 0.7976**) | +2.79×10⁻⁶ (p 0.069, RI 0.289) / +1.16×10⁻⁶ (p 0.376, RI 0.561) |
| US_PASSIVE vs NONUS_PASSIVE (benchmark) | +3.3374×10⁻⁷ (1.3136×10⁻⁶, p 0.8001) | **+1.5258×10⁻⁶** (1.0913×10⁻⁶, p 0.1766) | +1.03×10⁻⁶ / −0.14×10⁻⁶ |
| US-internal ACTIVE vs PASSIVE | +1.6021×10⁻⁶ (1.0290×10⁻⁶, p 0.1234) | −2.0162×10⁻⁷ (9.3370×10⁻⁷, p 0.8311) | +1.56×10⁻⁶ / −0.19×10⁻⁶ |

N = **347,690** (full, 466 singletons) / **183,880** (post-2018, 622 singletons, 22 quarters). Panel 696,312 rows / 6,867 firms / 82 quarters; funds master 106,965 unique `fund_id` (ACTIVE 39,429 / PASSIVE 9,353 / UNKNOWN 58,183). R² 0.5709 / 0.7314 / 0.5372 / 0.7795 / 0.6485 / 0.8297 across the six columns. The two-way `t_cn` terms are null throughout, with one marginal cell noted for completeness and carrying no claim: `m_pas_post` −5.61864×10⁻⁶ (p 0.0742).

**The MAIN contrast is null in both windows, and the family's one near-significant column is gone.** Pre-P0 the full-period MAIN sat at +2.79×10⁻⁶ with CRVE p = 0.069, the only sub-0.10 cell here; post-P0 it is −7.06×10⁻⁷ with p = 0.684 and RI 0.780. The post-2018 PRIMARY is +5.05×10⁻⁷ with RI 0.798. Both are deep nulls.

**The passive-dilution hypothesis is still rejected, for a cleaner reason than before.** The old argument was that active-only ≈ pooled at +2.79 versus +2.75×10⁻⁶. The new version is simply that active-only (−7.06×10⁻⁷) and pooled (−5.28×10⁻⁷) are both indistinguishable from zero, so restricting to active managers does not reveal an effect that pooling hid.

**Framing change owed, and applied here: "passive is the inert arm" is weaker post-P0.** The passive post-2018 benchmark flipped sign and is now the **largest |β₃| of the six columns** at +1.53×10⁻⁶, though still not significant (p = 0.177). We therefore no longer describe the passive arm as mechanically inert; we say only that no contrast in this family is distinguishable from zero, and that the passive benchmark is not visibly quieter than the active one on the P0 panel.

Panel integrity: reconciliation A gives max absolute deviation 0.000e+00 over 199 (side, quarter) cells; reconciliation B gives max|Σw−1| = 1.55×10⁻¹⁵ over 395 books with zero held-but-null cells; the firm universe is 12,791 and the grid 5,116,400 = 12,791 × 4 × 100; comparability against c6 is exact at 6,867 firms / 82 quarters. Coverage diagnostic on the `cn_lag`-filtered panel: Σ_firm(w) per (group, quarter) has min 0.2177, median 0.8362, max 0.9338, all below 1 as expected because w is normalized on the full 12,791-firm grid. The US 2021Q4 passive-share anchor holds: documented 39.78%, recomputed **39.75%** pooled (41.87% labelled-only) on the identical denominator, so the fund-universe change did not move it; NONUS 2021Q4 18.78% / 21.48%; US 2010Q4 13.16% / 13.97%. This anchor is **not** coded as a check inside `build_fourgroup_panel.py`; it was recomputed independently. Matched-share decay: US join share 0.995 (1999) → 0.917 (2023); NONUS 0.845 (2023).

B9: `fourgroup_vce_diag.csv`, 12 rows, zero `se_valid=0`; non-PSD/CGM warnings on `m_pas_full` and `m_usi_full`, which `se_valid` cannot flag. **Open gaps:** RI covers the MAIN ACTIVE contrast only, so the passive and US-internal menu contrasts have no design-based arbiter. Two stale in-script anchors will print apparent mismatches on this vintage and are left unedited deliberately: `run_fourgroup.do` line 42 still cites the pre-P0 pooled +2.746×10⁻⁶ as its dilution benchmark (the live value is −5.279916×10⁻⁷), and its coverage comment cites "~6,854 of 12,743 universe firms" against the true 6,867 of 12,791. `run_ri_fourgroup.py` lines 188–190 hard-code the pre-P0 Stata anchors for the same reason.

### 7.9 Russia positive control (`build_russia_c6_panel.py`, `run_russia_headline.do`, `run_ri_russia*.py`, `run_russia_lp_test.py`)

To make the China null informative we reran the *identical* pipeline (02→06→estimation→RI) on **Russia** exposure with a `USA|Russia` GPR shock; post-2022 is a known large divestment. A working positive control would detect that divestment and license reading the China null as "truly absent" rather than "undetectable."

| spec | β₃ (US·RU·S_t) | CRVE p | **RI p (headline inference)** | N | (pre-P0) |
|---|---|---|---|---|---|
| R1 it+gt | **−3.84187×10⁻⁶** (SE 1.60434×10⁻⁶) | **0.0189** | **0.2085** | 348,156 | −4.21×10⁻⁶ (1.97×10⁻⁶), 0.035, RI 0.243, N 347,952 |
| R2 3-pairwise | **−3.86876×10⁻⁶** (SE 1.59921×10⁻⁶) | **0.0178** | **0.2198** | 347,690 | −4.28×10⁻⁶ (2.01×10⁻⁶), 0.036, RI 0.246, N 347,490 |
| R3 2022Q1–Q2 event dummy | **−2.8858×10⁻⁵** | SE = 0, VCE degenerate (`se_valid=0`) | placebo **0.4815** | 348,156 | −3.44×10⁻⁵, placebo 0.457 |

β₂ (US·RU) for the record: R1 +4.5391×10⁻⁶ (SE 2.9331×10⁻⁶, p 0.1256); R2 +4.4750×10⁻⁶ (SE 9.2302×10⁻⁶, p 0.6291); R3 +6.3460×10⁻⁶ with SE and p missing. R² 0.5902 / 0.5908. R1 has 6,867 firm clusters and 82 month clusters; R2 has 6,634 after 466 singletons. RI: R1 200,000 permutations, R2 5,000 with 30 demeaning iterations, both on 174,078 firm-quarters.

**RI is the headline inference here**, not the CRVE: the shock is a single concentrated event, so the few-cluster CRVE overstates significance. The sign is **negative**, the divestment direction, as hoped. Under valid RI **no spec clears 0.05** (0.2085 / 0.2198). Note the two routes moved in opposite directions under P0: CRVE got *more* significant (0.035 → 0.019 and 0.036 → 0.018) while RI stayed flatly null, and the event-window placebo got slightly worse (0.457 → 0.4815). R3's CRVE p is `.` and must not be read as inference; the placebo is the valid design-based read for that column.

The cumulative event study confirms the picture. All **8 of 8** horizons are negative, with β from −2.53×10⁻⁵ to −4.38×10⁻⁵, HC1 t-statistics between −0.634 and −1.086 (none exceeding |t| = 1.09), and firm-level permutation p from **0.2703 to 0.4826** on n = **4,294** firms per horizon (pre-P0: 0.218–0.413, n = 4,289). The minimum is 0.2703 at h = 1.

| h | β | HC1 SE | t | perm p |
|---|---|---|---|---|
| 0 | −2.52992×10⁻⁵ | 3.98951×10⁻⁵ | −0.6341 | 0.4004 |
| 1 | −4.03577×10⁻⁵ | 3.99596×10⁻⁵ | −1.0100 | **0.2703** |
| 2 | −3.14725×10⁻⁵ | 3.91750×10⁻⁵ | −0.8034 | 0.4521 |
| 3 | −4.37919×10⁻⁵ | 4.03389×10⁻⁵ | −1.0856 | 0.3042 |
| 4 | −3.39984×10⁻⁵ | 4.13688×10⁻⁵ | −0.8218 | 0.4826 |
| 5 | −4.28936×10⁻⁵ | 4.11144×10⁻⁵ | −1.0433 | 0.3778 |
| 6 | −4.18559×10⁻⁵ | 4.17399×10⁻⁵ | −1.0028 | 0.4375 |
| 7 | −3.76049×10⁻⁵ | 4.14998×10⁻⁵ | −0.9061 | 0.4536 |

**So the Russia control shows the right *sign* but does not reach significance under valid design-based inference; it cannot be cited as passing design validation.** Every one of the eight figures this section used to quote has moved, and the qualitative verdict is unchanged and if anything better supported. This *reinforces* the known power limitation (limitation 1 / B12): with 82 quarters and a triple difference, even a large, concentrated, correctly-signed divestment is not detectable at conventional levels. Russia is the lowest-p family in the battery on CRVE and it still fails the arbiter, which is the cleanest available calibration of what this design can and cannot see.

Panel and exposure detail: 348,156 rows (US 174,078 / NONUS 174,078), 6,867 firms, 82 contiguous quarters, `ru_lag>0` share 0.0520, all asserts passed; grid 2,558,200 cells; EU entity universe 12,791 with 0 dual-listed; crosswalk 10,212. HIGH-exposure cutoff 0.037; winsor bounds p1 −0.00086062 / p99 +0.00100374; corr((dUS−dNONUS), Shock) on HIGH-lag firms −0.0681 raw / −0.0663 winsorized over 81 quarters. B9: `russia_headline_vce_diag.csv` carries the **single `se_valid=0` row in the entire ten-file corpus** (r3 / `us_ru_post`), which is the pre-registered expected one.

**Superseded:** an earlier pre-B7-fix run (all-relationship-type denominator) reported LP h=0 perm_p=0.040 / h=1 perm_p=0.052; those are **dead**, they came from the wrong denominator, and must not be cited as validation. Stale pre-P0 print-only anchors remain in `run_ri_russia.py` line 79 and `run_ri_russia_3pairwise.py` line 80, and pre-P0 vintage-warning headers remain on `06_russia_grid.jl`, `build_russia_c6_panel.py` and `build_russia_lp_panel.py`.

### 7.10 Extensive margin — holder breadth and exit (`build_extensive_margin_panel.py`, `run_extensive_margin.do`, `run_ri_extensive.py`)

**Why this exists.** Every outcome so far is value/weight-based (Δw, the shares-flow of §7.6, the flow decomposition). The divestment literature's *first* margin is the **number of holders** — breadth (Chen–Hong–Stein 2002) and outright position exit (Hong–Kacperczyk 2009) — because discretionary selling shows up as funds *leaving the register* before it shows up in aggregate dollars. Our main pipeline collapses `fund_id` into a single US/NONUS group before the outcome forms (`04_us_ownership_european.jl`, `SUM(adj_mv) GROUP BY`), so if 5 of 10 US funds exit and the survivors add, the group Δw barely moves and the exodus is invisible. This section rebuilds the outcome at the holder-count level from the same `holdings_eom.parquet` on the same C6 grid, and runs the identical 3-pairwise triple difference + design-based RI.

**Outcomes.** `n_holders` = COUNT(DISTINCT `fund_id`) in group *g* holding firm *i* at quarter *t* (zero-filled on the full grid). (a) **d_breadth** (PRIMARY) = Δ of the Chen–Hong–Stein-normalized breadth `n_holders / n_active`, where `n_active` is the group×quarter count of funds holding any grid firm — this nets out the secular rise in 13F filers; **d_nh** = Δ(raw n_holders) companion. (b) **exit** = 1{held at *t−1*, zero at *t*} and **init** = 1{zero at *t−1*, held at *t*}, linear-probability, conditioned on the lagged holding state.

**Pairing caveat (load-bearing).** Conditioning exit/init on a lagged holding state breaks the US–NONUS pairing that the 3-pairwise FE relies on (an unpaired firm-quarter is an fq-FE singleton contributing nothing), so exit is estimated only on **both-held** firm-quarters and init only on **neither-held** ones. Four-cell counts on the P0 grid: both-held **206,809** (16.33%), US-only **11,003** (0.87%), NONUS-only **166,137** (13.12%), neither **882,360** (69.68%) of a **1,266,309**-cell domain (pre-P0: 204,314 / 12,356 / 155,352 / 889,535 of 1,261,557). The **15× asymmetry between NONUS-only and US-only** firm-quarters is itself a fact: at the extensive margin US institutions hold a *narrower* set of these firms. The both-held / neither restriction selects toward larger, dual-held firms and narrows the exit/init estimand away from US divestment broadly.

**Result — the extensive margin is null too.** DDD coefficient (US·CN·S_t), 3-pairwise headline; circular-shift RI is the arbiter (post-A0 convention). All rows are P0-vintage.

| outcome | β₃ | SE | CRVE p | free-RI p | **circular-shift RI (arbiter)** | riskset RI (arbiter) | (pre-P0 β₃ / RI) |
|---|---|---|---|---|---|---|---|
| **d_breadth (PRIMARY)** | +2.197184×10⁻⁵ | 3.834406×10⁻⁵ | 0.5682 | 0.7001 | **0.5732 (null)** | 0.6585 | +2.42×10⁻⁵ / 0.768 |
| d_nh | −1.453061 | 1.041273 | 0.1667 | 0.3207 | **0.2195 (null)** | 0.2073 | −1.17 / 0.549 |
| exit | +1.415490×10⁻³ | 3.715630×10⁻³ | 0.7042 | 0.6951 | **0.6829 (null)** | 0.6829 | +1.73×10⁻³ / 0.695 |
| init | −6.622296×10⁻⁴ | 2.361000×10⁻³ | 0.7798 | 0.8680 | **0.8293 (null)** | 0.7561 | −1.11×10⁻³ / 0.805 |

N = **347,690** (breadth / d_nh) / **201,634** (exit) / **85,136** (init); the RI collapse runs on 174,078 / 100,982 / 42,940 firm-quarters and reproduces each Stata estimate to relative error ≤ 7×10⁻⁴. **Every outcome is null under the RI arbiter**, on both the full grid and the engaged (riskset) subsample, with arbiter p spanning 0.21 to 0.83. So discretionary exit does not respond differentially to tension either: the null holds on the margin where divestment would appear first.

it+gt companions: d_breadth +1.4756×10⁻⁵ (p 0.6840), d_nh −1.3584 (p 0.1823), init −2.1752×10⁻³ (p 0.4328), and **exit −5.0551×10⁻³ (SE 2.4673×10⁻³, p 0.0437)**.

**The exit watch cell, stated precisely.** That last companion is the only extensive-margin cell below 0.05 on any route, and it is weaker than it was (pre-P0 p = 0.006). Four things must be said with it. It is the **it+gt companion, not the 3-pairwise primary**. The 3-pairwise primary carries the **opposite sign** (+1.4155×10⁻³, p = 0.704). RI covers the 3-pairwise collapse only, so **this cell has no design-based arbiter at all**. And its sign is negative, meaning US funds exit *less*, the anti-disengagement direction. It is a watch item, not a finding.

Risk-set variants: d_breadth +2.468393×10⁻⁵ (p 0.6604, N 270,386) and d_nh −2.053915 (p 0.1785). The **exit riskset rows are bit-identical to their full-sample twins**, because `samp_exit==1` implies `risk1==1`, so the risk-set restriction is mechanically non-binding for exit; do not present it as an independent robustness result. The init riskset cells (N 7,824 / 8,518) are underpowered and are labelled uninformative.

**Live motivating fact (P0 vintage).** At the firm-quarter cell level the US zero-holder rate is **82.757%** (1,058,546 / 1,279,100) versus **70.527%** for NONUS (902,112 / 1,279,100), a **+12.23 pp** gap in the disengagement direction. *(Pre-P0: +11.30 pp; pre-B7: +11.26 pp.)* On the **estimable subset** (cn_lag non-missing, 174,078 firm-quarters) the same gap is **+15.18 pp** (US 41.38% vs NONUS 26.20%). **The two denominators now differ by about 3pp, so the paper must state which one it quotes.** At the *firm* level US institutions have ever held **56.59%** of the 12,791 firms against **92.11%** for NONUS, a **−35.52 pp** gap (unchanged by P0). Separately, **98.87%** of firms (12,646 / 12,791) have at least one quarter with zero US holders, against **100%** for NONUS. The honest reading pairs the two facts. US institutions hold these firms **more sparsely** to begin with, **but that sparsity does not deepen differentially when tension rises** (the DDD null above).

**Coverage-asymmetry caveat:** count/exit outcomes are more exposed than Δw to the US-13F-vs-NONUS reporting-coverage asymmetry (a coverage-driven disappearance reads as an exit); the riskset-subsample rerun is the partial mitigation, since it restricts rows and not the `n_active` denominator. **Denominator caveat:** the `n_active` breadth denominator shows sawtooth Q1/Q3 collapses in both groups (US from ~2,400 to ~700–800 through 2000–2004; NONUS through 2013). P0 recovers weekend-batch funds but the odd/even reporting pattern persists, so d_breadth levels in 1999–2005 are denominator-driven. Panel build: 2,558,200 rows = 12,791 firms × 100 quarters × 2 groups; rows with `cn_lag` and shock 348,156 (exit 232,120, init 116,036); all structural asserts (a)–(h) OK. B9: `extmargin_vce_diag.csv`, 32 rows, zero `se_valid=0`.

---

## 8. The centered-vs-backward disclosure (retraction of the earlier "result")

| Δw definition | β₃ | SE | p |
|---|---|---|---|
| Centered (w_{t+1} − w_{t−1}), prior build (pre-B7) | +1.38 | 0.54 | **0.013 (significant, opposite H2.1)** |
| Backward (w_t − w_{t−1}), pre-B7 comparison | +1.28 | 1.66 | 0.443 (null) |

Both rows are on the **pre-B7 all-`rel_type` panel**, held on a common sample so the SE change is not confounded by the B7 rebuild; the current P0 backward headline is −1.072712×10⁻⁶ (it+gt) / −5.279916×10⁻⁷ (3-pairwise), §7.1, still null. Point estimate barely moves (+1.38 → +1.28); **SE roughly triples (0.54 → 1.66)**. We read the earlier significance as a **look-ahead / post-treatment window contamination**: the centered LHS includes post-t holdings (w_{t+1}), so the outcome mixes contemporaneous with future adjustment and is not aligned with the estimand's timing. We deliberately do **not** claim a proven "mechanical correlation of shocks" (S_t is an AR(1) residual, so S_t and S_{t+1} need not be strongly correlated), and we do **not** claim to have decomposed *why* the SE tripled. Three forces move together between the two builds, namely look-ahead removal, a small right-edge sample expansion, and entry/exit reweighting, and they are not separately identified here; a sample-matched centered-window comparison on a common sample (deferred, §10) is needed to attribute the SE change. Identification rests on the within-firm-quarter β₃, which is null. **The retraction is on firm ground:** the clean lead-flow test (Δw_{t+1} ~ S_t) and the local-projection IRF are both null under design-based randomization inference on the P0 panel (Second review round, F1), so the null is not an artifact of the centered window's timing. No US-differential response exists at t, at t+1, or cumulatively.

### 8b. The second retraction: the positive-sign pattern was a data artifact (P0, 2026-08-04)

The centered-window retraction above now has a sibling, and it is worth stating separately because it retracts a *pattern* rather than a single coefficient.

Through every prior version of this document, the headline reading was "null, and every S_t point estimate on the main outcome is positive, i.e. opposite to the disengagement prediction." That sentence appeared in §0, in the second-review-round summary, in §7.1, §7.3, §7.3b, §7.3c, §7.6, §7.7 and §13. It was used to argue that even the CRVE-significant cumulative horizons, if believed, would strengthen rather than overturn the no-disengagement reading.

**That pattern was not in the data. It was in the stamping rule.** `03_eom_etl.jl` kept only exact-quarter-end report dates, dropping roughly 40% of funds on weekend quarter-ends, and the drop was US/NONUS-asymmetric (§5.0). Weekend quarters under-counted US holdings, so the following rebound quarters read as US buying, which tilted β₃ positive across the specs that share the shock timing. Once the as-of rule fixes the stamping, the US-share selection tilt falls from −3.41pp to +0.01pp and the sign pattern disappears: the headline is −5.28×10⁻⁷, the lead flow is +2.09×10⁻⁷, LP h1 is negative and cum2–cum4 positive, the tercile bins are positive and tiny, the risk-set family is uniformly small-negative, the direction split has a negative sell arm and a positive buy arm, and the shares-based MAIN flow is +7.48×10⁻⁵. Everything is null.

So both halves of the original story are retracted. The **timing** story was retracted first: the significant centered-window estimate was a forward-window artifact. The **sign** story is retracted now: the uniform positive lean was a calendar artifact. What survives both retractions is the same null, measured on better data, with wider honesty about what it does and does not rule out. No sentence in this document should now claim a directional reading of β₃ on the main outcome. The one narrow sign remark that is still allowed is about cum4 specifically, which remains positive and therefore anti-H2.1 even where its lone WCB rejection stands.

---

## 9. Limitations and threats (exhaustive)

1. **Null could be power, not absence.** 82 quarters; effective time variation is small; two-way cluster leaves 82 quarter-clusters. We cannot reject H2.1 or its opposite.
2. **Zero-fill assumption** (§5.5): absence coded as zero by assumption; non-US coverage uneven.
3. **Backward-diff timing** removes t+1 leakage but not within-quarter anticipation (investors trading in months 1–2 of quarter t on early news that the quarter-end residual misses).
4. **Shock uses only the quarter-end month** (~1/3 of intra-quarter GPR news).
5. **Entry/exit mechanical asymmetry** under backward diff (exit = −w_{t−1}, entry = +w_t) mechanically favors the H2.1 direction, yet β₃ is null — arguably reassuring, but it complicates magnitude interpretation.
6. **Estimand shifts** across samples (full grid = all holdings-observed European issuers; risk-set = engaged firms only).
7. **"European" includes GB/CH/NO** (non-EU); GB dominates the country-pair subsample.
8. **HIGH cutoff is ~6%** (B7 CUSTOMER+SUPPLIER median of positive CN; pre-B7 all-`rel_type` was ~4%), so "high exposure" is not extreme.
8b. **Ramp-up-era coverage residual (P0).** The as-of rule cuts the weekday-vs-weekend coverage gap from 22.3pp to 4.2pp overall, but the 1999–2005 ramp-up era still sits at **11.3pp**. No choice of the window W fixes it, because it reflects early FactSet coverage rather than the stamping rule. Disclosed rather than tuned away. The candidate robustness, not yet run, is to start the sample in 2006. A second, smaller P0 artifact: step 04's `I_ict` now mixes valuation dates within a quarter cell for shifted funds, giving about one day of price drift on roughly 7.8% of rows.
8c. **Raw shares-flow outliers are an open data issue (P0).** The as-of rule admits firm-quarters with flow ratios up to 1724.9× float (kurtosis ≈ 199,000, against a pre-P0 max of 9.8). A handful of such cells drive all three raw-flow RI rejections at ≈0.02 and broke the +0.20 co-movement anchor that §7.6's estimand paragraph used to rest on. Winsorized inference is MAIN and null; the raw column is disclosed and carries no claim. The fix owed is a predetermined BAD-rule extension, not an outcome-driven trim, and it is not written. Until it exists, treat every raw-flow number in §7.6 as descriptive of the outliers rather than of the population.
8d. **One inference route disagrees at cum4 and is not resolved.** Four design-based routes put cum4 at or above 0.0558; the score-based quarter-cluster WCB puts it at 0.0038. RI is the arbiter by the pre-registered hierarchy, and the WCB assumes exactly the cross-quarter independence that the measured serial correlation of S_t violates, so cum4 is reported as not surviving. The disagreement is a live limitation of the inference toolkit rather than a settled question. See "Cum4 inference hardening" and §15.
9. **Group-level (2 groups), not holder-level.** The true Khwaja–Mian holder×time control is not yet in.
10. **Convention-match not verified componentwise** against De Haas (do their diff and shock-timing both match ours?).
11. **Universe is holdings-observed, not full listing** (§3, Caveat 2). The C6 zero-fill corrects selection on the outcome *within* the holdings-observed universe (a firm held by anyone gets a full grid, so US exits are captured), but not securities never held by any institution. Adding those from full Security Coverage would, for firms with non-missing CN, contribute zero within-firm-quarter outcome variation against nonzero regressor variation, whose **mechanical expectation is to attenuate β₃ toward zero**. On that reasoning the current holdings-observed universe is, if anything, the *higher-powered* universe for detecting β₃, and the full-universe rebuild is an optional robustness. But this is the expected direction under a simplified argument, not a proven one — in the full FE model the added rows also shift the group×quarter effects, so the net movement of β₃ should be **verified by actually running the rebuild, not asserted**. (This corrects an earlier claim that never-held firms "do not affect β₃" — they do, and the leading-order effect is attenuation.)
12. **Revere match has country structure.** Of the holdings-observed European securities, ~20% do not match to Revere overall, but the miss rate is uneven by listing country (`05_unmatched_profile_by_country.csv`). **GB is the worst among large countries: 32.78% unmatched (1,307 of 3,987)**, versus FR 17.9%, DE 11.6%. Because GB is 60.85% of the GB+DE+FR country-pair subsample, the Revere-matched sample systematically drops about one-third of GB issuers, a selection concern for both the main and country-pair specifications that should be characterized (is the unmatched set systematically smaller or different-sector?), not just reported. *(These match rates are pre-P0 cascade figures and were **not re-derived this cycle** on the 12,791-firm P0 universe; only the cascade endpoints were.)*
13. **Shares-based test excludes ADRs** (§7.6, Caveat 1). The ownership-share robustness is primary-EQ only, so it cannot see stake adjustment through ADR/GDR or non-primary classes. On the P0 panel US holds **8.79%** of its European exposure off the primary class versus **3.22%** for non-US (2.73× asymmetric; pre-P0 8.92% / 3.32%), so the shares-based null is complementary to, not a substitute for, the USD portfolio-weight main spec that does capture the ADR channel. Its power is limited against small adjustments; the 80%-power MDE for a representative firm-quarter is on the order of **1.2–3.7 bps of float** on the MAIN winsorized SE (Second review round, F5), so it bounds the stake effect near zero rather than proving an exact zero.
13b. **Several families have no design-based arbiter.** RI is missing entirely for the risk-set / country-pair / spell-boundary family (8 specs, §7.4–§7.5), for the direction-split INDICATOR spec (the one that needed the CGM repair, §7.7), for the sagg 3-pairwise h=0 cell (the family's lowest p, §7.3c), for the extensive-margin exit it+gt watch cell (§7.10), and for the four-group passive and US-internal menu contrasts (§7.8). Supplying arbiters there requires new code, not a rerun. Where a family's verdict rests on CRVE alone, this document says so at the point of use.
13c. **Some inference figures survive only in transient logs.** The sagg Stata battery (all SEs, the joint Wald, the cumulative figures), the F4a/F4b/F4c and F8 SEs and p-values, the F1b quarter-cluster CRVE rows, the `boottest` OOM record, and DDD Stage B/C all live in Stata logs with no results CSV behind them. Every point estimate in that set was independently re-derived and matched, but the inference figures cannot be re-checked from an artifact. Results-CSV writers should be added before the next rerun.
14. **Group is FactSet registration domicile, not decision nationality** (third review round). A US manager's Luxembourg / Ireland UCITS shell is grouped into NONUS, though the allocation decision and any political-pressure channel are US — treatment-group misclassification that mechanically **compresses β₃ toward zero**. Not yet corrected; the root fix is to regroup by the management company's **ultimate-parent nationality** (deferred, §10). The EU-domiciled-only control treats the symptom, not the cause.
15. **NONUS is not an untreated control.** European institutions have their own domestic de-risking pressure (from ~2019), so the triple difference identifies only *differential* US disengagement and is **blind to common de-risking**: a zero β₃ is consistent with both groups reallocating identically. Consistent with this, the weak-FE column (which would pick up any *common* reaction) is also null — at the quarterly frequency no group reallocates within Europe on the shock.
16. **The continuous AR(1) shock averages over opposite-signed episodes.** 2018 tariffs (bad for exposed firms) vs the 2022 export controls (partly *good* for EU semicap substitutes) may push allocations opposite ways; one continuous shock aggregates them toward zero. A named-event study (H2.2) is the cleaner test.

---

## 10. Deferred robustness (planned, NOT done — do not present as completed)

- Continuing-positions-only subsample (w_{t−1}>0 AND w_t>0) to neutralize entry/exit mechanical asymmetry.
- Winsorize Δw at 1/99%.
- Leave-one-quarter-out (which escalation episodes carry any result).
- PPML via `ppmlhdfe` for the zero-heavy weights (Silva–Tenreyro 2006; Correia et al. 2020).
- ~~Local projections (Jordà 2005) for dynamics~~ — **DONE** (Second review round, F1). On the P0 panel every horizon is null on CRVE and on all four design-based routes; the pre-registered cum4 free-permutation rejection is gone (0.0388 → 0.0664). The score-based quarter-cluster wild bootstrap is completed and is the one route that still rejects cum4 (0.0038), disclosed with its independence assumption stated; only `boottest`'s dense-matrix path OOMs. `run_cum4_inference.py`.
- ~~Shock-timing / shock-construction menu~~ — **DONE** (§7.3d): 10 constructions × 3 directions, pre-registered whiteness selection, all null under the circular-shift arbiter (min 0.2439). Closes P1a, P1b and P1c, and supersedes the F7 exact-nesting column.
- **Predetermined guard for the raw shares-flow outliers** (§7.6 open issue): identify the monster firm-quarters the as-of rule admits (max 1724.9× float) and extend the BAD firm-quarter rule to cover them. Must be predetermined, never an outcome-driven trim. Blocking a clean §7.6 raw column and the co-movement anchor.
- **Sample start 2006 robustness** (§5.0): the residual 11.3pp ramp-up-era coverage gap is a data property no window W fixes; re-running the headline from 2006 is the candidate check.
- **Holder-level fixed effects** (holder × quarter), the true bank×time analog.
- **ADR-inclusive ownership share** (§7.6): map ADR/GDR holdings to underlying-share equivalents via the ADR conversion ratio so the shares-based test captures the ADR channel (US 8.79% of exposure off-primary on the P0 panel); needs the ratio data.
- Sample-matched centered-window comparison (quantify the §8 disclosure on a common sample).
- OFAC SDN / Entity-List exclusion; size, industry, listing-country × shock controls.
- **Positive controls** (make the null informative): (i) **Russia exposure** post-Ukraine (2022) — **DONE (§7.9)**: the isomorphic 02→06→RI pipeline with `'RU'` and a `USA|Russia` shock returns the correct (divestment) *sign* but no spec is significant under valid RI, so it validates *direction only* and reinforces the power limitation; (ii) the same investors' **direct China holdings** (ADR/HK, HFCAA-era divestment) as an in-pipeline benchmark — still planned. If the design detects those known divestments, the China null becomes "truly absent," not "undetectable."
- **Ultimate-parent regrouping** (limitation 14): reclassify holders by management-company ultimate-parent nationality; EU-domiciled-only as one column.
- **H2.2 price channel**: acquire European stock returns (Datastream / Compustat Global Security / FactSet Prices); abnormal returns from a factor model; Tobin's q from Compustat Global; named-event studies (2018 tariffs, 2019 Huawei, 2022-08, 2022-10) at daily/monthly frequency to bypass the 82-quarter power ceiling.

---

## 11. Open questions pending advisor confirmation

- Is the "another two-way FE" the advisor mentioned = **firm × group** (μ_{i,g})? (We assume yes; §6.4, §7.2, §7.4.)
- "Δw = 100% / −100%" at entry/exit: does the advisor want the **continuous** Δw (current) or a **normalized 0-1 / percentage** entry-exit indicator as the outcome?
- Table-3 "US-only / Europe-only" columns: does this mean **US-listed vs European-listed firm** subsamples? (We assume yes; not yet built.)

---

## 12. Command inventory (all under `julia_descriptive/`; run order top-to-bottom)

| File | Role | Key output |
|---|---|---|
| `00_setup.jl` | EU country list, paths, DB helper (P0: `max_temp_directory_size=300GB` on the duckdb connection) | — |
| `02_china_exposure.jl` | CN exposure (symmetric, bilateral, time-versioned; C2/C4/C5) | `firm_quarter_china_exposure.parquet` |
> **[UPDATE 2026-08-08 — REBUILD v3; this table is P0-vintage and PARTLY RETIRED].** Since this table was written: (i) `03_eom_etl.jl` default is the advisor QUARTER rule at **FUND grain** (`DPN_SNAPSHOT_GRAIN=fund`; the W=10 as-of and per-security grain survive as overrides); (ii) the MAIN outcome `dw` is the **GLOBAL full-portfolio-denominator** Δw (EU-restricted Δw lives in `dw_eu`, labeled within-Europe-reallocation diagnostic); (iii) PRIMARY timing is **S_{t−1}** (`s_lag`; S_t is the labeled timing diagnostic); (iv) `headline_3pairwise_canonical.csv` uses the LOCKED layout `spec,denom,timing,fe,b3,se,p,N` (primary + primary_itgt + diag 2x2 + diag itgt-st anchor row). Row counts below are P0-era. See `VINTAGE_P0.md`, `VINTAGE_PREEM.md` and the REBUILD v3 chain log for current numbers.

| `03_eom_etl.jl` | FactSet Ownership EOM ETL. **P0: the as-of W=10 snapshot rule** (latest report ≤ quarter-end within 10 days, stamped to quarter-end, `report_date_actual` + `asof_gap_days` retained); §5.0 | `holdings_eom.parquet` (208,418,523 rows) |
| `06_cartesian_grid.jl` | Cartesian grid + zero-fill + **backward Δw** + lagged CN (C1/C6) | `merged_us_eu_zero_filled.parquet` |
| `build_c6_panel.py` | → main estimation panel (adds `sell_lag`/`buy_lag` for §7.7) | `c6_panel.dta` (348,156) |
| `build_country_pair_shock.py` | AR(1) per country-pair (UK/DE/FR) + subsample | `c6_panel_country_pair.dta` |
| `build_spell_riskset.py` | **correct** firm-quarter risk-set conditional sample | `c6_panel_riskset.dta` (270,796) |
| ~~`build_spell_boundary.py`~~ | **SUPERSEDED — per-group selection, broke pairing** | ~~`c6_panel_spell.dta`~~ |
| `07_regression.do` | 4 specs (β₂+β₃ / β₃-only / full triple / weak FE) | `07_reghdfe_results.txt` |
| `07b_country_pair_robustness.do` | country-pair S_{c,t}, recovers β₁ | `07b_country_pair_results.*` |
| `07c_strict_4term.do` | explicit 4-term form, main vs country-pair | `07c_strict_4term_results.*` |
| `07d_three_spec_table.do` | No-FE / Headline / Weak-FE headline table | `07d_three_spec_results.*` |
| `07e_firmgroup_tail.do` | firm×group FE + tail-dummy menu (k=1.645/2/3) | `07e_firmgroup_tail.*` |
| `07g_spell_riskset.do` | risk-set conditional sample regression | `07g_spell_riskset.*` |
| ~~`07f_spell_boundary.do`~~ | **SUPERSEDED (ran the biased per-group sample)** | — |
| `run_headline_3pairwise.do` | **headline** = three-way pairwise FE, all main specs | `audit_c6_panel.dta`, `headline_3pairwise_canonical.csv` |
| `run_audit_f1f2f7.do` / `run_audit_f4f8.do` | second-round tests (F1/F2/F7, F4/F8) | audit CSVs |
| `run_randomization_inference.py` / `run_ri_3pairwise.py` | design-based RI (F1/F3) | RI CSVs |
| `build_shocklag_panel.py` / `run_shocklag.do` / `run_ri_shocklag.py` | advisor shock-timing revision (§7.3b): S_{t-1} spec + RI | `shocklag_panel.dta`, `shocklag_ri_results.csv` |
| `02_russia_exposure.jl` / `06_russia_grid.jl` / `build_russia_shock.py` / `build_russia_c6_panel.py` | Russia positive control (isomorphic pipeline) | `c6_panel_russia.dta` |
| `run_russia_headline.do` / `run_ri_russia*.py` / `build_russia_lp_panel.py` / `run_russia_lp_test.py` | Russia positive-control regressions + inference (§7.9: correct divestment sign, not significant under RI — validates direction only) | Russia result CSVs |
| `build_audit_panel_f1f2f7.py` / `build_riskset_lagonly.py` | second-round panels | audit / lag-only dta |
| `run_tercile_3pairwise.do` / `run_ri_tercile.py` | §7.3 shock-tercile dose menu + RI | `tercile_results.csv`, `ri_tercile_results.csv`, `tercile_vce_diag.csv` |
| `build_sagg_panel.py` / `run_sagg_distlag.do` / `run_ri_sagg.py` | §7.3c aggregated-shock robustness + RI (free-perm + circular-shift) | `sagg_panel.dta`, `sagg_ri_check.csv` |
| `run_direction_split.do` / `run_ri_direction.py` | §7.7 buy-side vs sell-side split + RI | `direction_results.csv`, `ri_direction_results.csv`, `direction_vce_diag.csv` |
| `build_fourgroup_panel.py` / `run_fourgroup.do` / `run_ri_fourgroup.py` | §7.8 active/passive four-group split + RI (post-2018 primary) | `fourgroup_panel.dta`, `fourgroup_results.csv`, `ri_fourgroup_results.csv`, `fourgroup_vce_diag.csv` |
| `run_flow_decomposition.py` / `run_flow_decomp_step3.do` | §7.6 flow-identity decomposition (six cells) | `flow_decomposition_panel.dta`, `ri_flow_decomposition.csv` |
| `run_cum4_inference.py` | cum4 inference hardening: shock serial diagnostics + RI three ways (free / all-81 circular-shift / moving-block L5,L8) + within-family max-\|t\| FWER + score-based quarter-cluster Webb WCB, over h0..cum4 | `cum4_inference_hardening.csv` |
| `build_extensive_margin_panel.py` / `run_extensive_margin.do` / `run_ri_extensive.py` | §7.10 extensive-margin outcomes (holder breadth, count, exit, init) from `holdings_eom` + RI (free + circular-shift) | `extensive_margin_panel.dta`, `extmargin_results.csv`, `extmargin_fourcell.csv`, `extmargin_descriptive.csv`, `ri_extensive_results.csv`, `extmargin_vce_diag.csv` |
| `build_shock_menu.py` / `run_shock_menu.do` / `run_ri_shockmenu.py` | §7.3d shock-construction menu: 11 columns + validation gate + pre-registered whiteness selection; 28 reghdfe specs with the B9 guard and a drift gate; RI for 12 columns | `shock_menu_quarterly.{csv,parquet,dta}`, `shock_menu_diagnostics.csv`, `shock_menu_corr.csv`, `shock_menu_rederivation_check.csv`, `shock_menu_preferred.txt`, `shockmenu_c6_panel.parquet`, `shockmenu_results.csv`, `shockmenu_vce_diag.csv`, `ri_shockmenu.csv` |
| `diag_p0_asof_window.py` / `diag_p0_compare.py` / `archive_preP0.py` | §5.0 P0 rebuild: W-curve and coverage/staleness diagnostics; old-vs-new gates (coverage, US-share tilt, dollars, shock bit-identity, grid and c6 deltas); archiving of 56 pre-P0 artifacts and vintage banners on 16 scripts | `output/diag_p0_*` (18 files), `*_preP0` artifacts |

Stata specification pattern (headline; all reghdfe calls share it):

```stata
gen rd_day = dofc(rdate)                         // pandas writes %tc; convert before mofd
egen firm_n = group(firm_str)
egen fq     = group(firm_str rd_day)             // firm x quarter  (alpha_{i,t})
gen  rd_m   = mofd(rd_day)
egen gq     = group(hgroup rd_day)               // group x quarter (gamma_{g,t})
egen ig     = group(firm_str hgroup)             // firm x group    (mu_{i,g}, robustness)
gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock
reghdfe dw us_cn us_cn_shock, absorb(fq gq)      vce(cluster firm_n rd_m)   // headline
reghdfe dw us_cn us_cn_shock, absorb(fq gq ig)   vce(cluster firm_n rd_m)   // + firm x group
```

Risk-set construction (the correct pairing logic, `build_spell_riskset.py`):

```sql
-- per (firm, group): held + neighbors
CAST(I_ict>0 AS INT) AS held,
COALESCE(LAG(...)  OVER w, 0) AS held_lag,
COALESCE(LEAD(...) OVER w, 0) AS held_lead
WINDOW w AS (PARTITION BY sec_entity_id, holder_group ORDER BY report_date)
-- firm-quarter risk set = union over groups; keep BOTH rows
MAX(CASE WHEN held=1 OR held_lag=1 OR held_lead=1 THEN 1 ELSE 0 END)
    OVER (PARTITION BY sec_entity_id, report_date) AS risk
... WHERE risk = 1
```

---

## 13. Master results table (β₃, ×10⁻⁶, all specs)

All rows are **P0-vintage** unless the row itself says otherwise. β₃ values are ×10⁻⁶ of portfolio weight unless noted. The last column carries the pre-P0 value for traceability.

| Spec | Sample / shock | β₃ | SE | p | N | (pre-P0 β₃ / p) |
|---|---|---|---|---|---|---|
| **Headline (3-pairwise it+gt+ig)** | full grid, S_t | **−0.5280** | 1.7432 | **0.762748** (RI free 0.8154 / circ 0.8293) | **347,690** | +2.746 / 0.110 (RI 0.31) |
| Headline (it+gt) | full grid, S_t | −1.0727 | 1.8191 | 0.557028 (RI 0.5984) | 348,156 | +2.081 / 0.151 (RI 0.41) |
| No FE | full grid | −0.0929 | 0.5032 | 0.854 | 348,156 | +0.0006 / 0.999 |
| Weak FE (firm+quarter) | full grid | +0.5241 | 0.6952 | 0.453 | 348,156 | +0.538 / 0.411 |
| Conditional risk-set (fq gq) | paired, S_t | −1.7300 | 2.8399 | 0.5441 | 270,796 | +3.26 / 0.1505 |
| Conditional + firm×group | paired | −1.7063 | 2.9983 | 0.5709 | 270,386 | +3.51 / 0.1432 |
| Risk-set lag-only (F8) | paired, S_t | −1.7674 | 2.90 | 0.5443 | 268,800 | +3.35 / 0.153 |
| Country-pair β₁ (GB+DE+FR) | S_{c,t} | −1.36 | 12.0 / 16.9 / 18.0 | 0.9103 / 0.9432 / 0.9467 | 172,920 | +22.2 / 0.086→0.246 |
| Country-pair β₃ (GB+DE+FR) | S_{c,t} | −8.13 | — | 0.764 / 0.521 / 0.408 | 172,920 | −1.21 / 0.93–0.97 |
| Lead-flow Δw_{t+1} ~ S_t (it+gt) | full grid | −0.3177 | 1.19 | 0.7907 (RI 0.8775) | 338,076 | +0.485 / 0.774 |
| In-span (drop ~25% phantom, it+gt) | in-span, S_t | −1.5656 | 2.81 | 0.5788 (RI 0.6093) | 272,140 | +3.33 / 0.135 |
| Local projection cum1 (3-pairwise) | full grid, S_t | −0.1535 | — | free-RI 0.9376 / circ 0.9634 / mb 0.940–0.943 / FWER 1.000 / WCB 0.9463 | 169,038 fq | +4.094, CRVE 0.009 |
| Local projection cum2 (3-pairwise) | full grid, S_t | +2.2694 | — | free 0.3675 / circ 0.4268 / mb 0.407–0.392 / FWER 0.618 / WCB 0.1774 | —‡ | +7.131, free 0.022 |
| Local projection cum3 (3-pairwise) | full grid, S_t | +2.8071 | — | free 0.3619 / circ 0.3537 / mb 0.400–0.417 / FWER 0.507 / WCB 0.1160 | —‡ | +6.467, free 0.082 |
| Local projection cum4 (3-pairwise) | full grid, S_t | **+6.6006** | — | free **0.0664** / **circ 0.1220 (arbiter → null)** / mb 0.1074–0.1076 / FWER 0.0558–0.0618 / **WCB 0.0038 (disagrees, disclosed)**; *positive* sign | 154,756 fq | +8.826, free 0.0388 |
| Advisor lagged shock S_{t−1} (§7.3b) | full grid, S_{t−1} | −1.0669 | 1.18 | 0.3696 (RI 0.6018) | 348,156 | −0.59 / 0.750 (RI 0.81) |
| Aggregated shock s_agg h=0 (§7.3c) | full grid, s_agg | −1.55 (it+gt) / −1.66 (3pw) | 0.923 / 0.980 | 0.0974 / 0.0939 (RI free 0.2353 / circ **0.2195**) | 348,156 / 347,690 | −1.06 / 0.348 (RI 0.51/0.46) |
| Shock-tercile T3 (§7.3) | full grid, tercile | +2.77 | 18.7 | 0.8823 (RI **0.8478**) | 347,690 | +20.5 / 0.264 (RI 0.23) |
| Shock menu, preferred `s_D_q_ar1_nla` (§7.3d) | full grid, quarterly-AR(1) shock | −7.557 | 3.452 | CRVE **0.0315** / RI free 0.1780 / **circ 0.2439 (arbiter → null)**; rule-dependent winner: LB(4)-first, C-ii is whiter at LB(8) 0.1319 vs 0.0848, an \|acf1\|-first rule would have picked C-i, and the selected shock is not white | 347,690 | — (new this cycle) |
| Direction buy-side (§7.7) | full grid, buy_lag | +1.0131 | 2.7750 | 0.7160 (RI 0.6701) | 347,690 | +5.50 / 0.147 (RI 0.14) |
| Direction sell-side (§7.7) | full grid, sell_lag | −1.9639 | 1.3749 | 0.1570 (RI 0.4735) | 347,690 | +0.25 / 0.618 (RI 0.93) |
| Four-group active-vs-active, full (§7.8) | active, S_t | −0.7065 | 1.7270 | 0.6836 (RI **0.7802**) | 347,690 | +2.789 / 0.069 (RI 0.289) |
| Four-group active-vs-active, post-2018 (§7.8, PRIMARY) | active, S_t | +0.5048 | 1.7146 | 0.7713 (RI **0.7976**) | 183,880 | +1.16 / 0.376 (RI 0.56) |
| Ownership FLOW winsor, MAIN (F6)† | flow, S_t | +0.0000748 | 0.000340 | CRVE 0.8265 / **RI 0.913** | 247,582 | +0.000649 / 0.056 (RI 0.38) |
| Ownership FLOW raw, disclosed only (F6)† | flow, S_t | −0.0361 | 0.0303 | CRVE 0.2366 / RI 0.021 — **outlier-dominated, no claim** | 247,582 | +0.00089 / 0.0001 (RI 0.37) |
| GPR two-interaction (F7) | GPR_t / GPR_{t−1} | −2.20 / +0.951 | 1.43 / 1.01 | 0.1262 / 0.3496 | 348,156 | −0.84 / +0.26 (0.63 / 0.88) |
| Russia positive control it+gt (§7.9) | full grid, S^{RU}_t | −3.8419 | 1.6043 | CRVE **0.0189** / **RI 0.2085** | 348,156 | −4.21 / 0.035 (RI 0.243) |
| Russia positive control 3pw (§7.9) | full grid, S^{RU}_t | −3.8688 | 1.5992 | CRVE **0.0178** / **RI 0.2198** | 347,690 | −4.28 / 0.036 (RI 0.246) |
| Extensive margin d_breadth, PRIMARY (§7.10) | full grid, S_t | +21.97 (=+2.197×10⁻⁵ of breadth) | — | CRVE 0.5682 / **circ RI 0.5732** | 347,690 | +2.42×10⁻⁵ / RI 0.768 |
| Extensive margin exit it+gt, WATCH (§7.10) | both-held, S_t | −5.0551×10⁻³ | 2.4673×10⁻³ | **0.0437**, no RI arbiter, 3pw primary has opposite sign | 201,964 | p 0.006 |
| Tail k=1.645/2/3 (07e, inference-invalid) | full grid, dummy | −27.15 / −23.02 / −60.51 | 17.70 / 23.01 / 32.90 | 0.129 / 0.320 / 0.070 | 40,914 / 29,010 / 14,586 | pre-B7 +1.07 / +14.0 / +8.59 |
| **[retracted] centered diff (pre-B7)** | full grid | +1.38 | 0.54 | 0.013 | ~450k | — |
| **[retracted, biased] per-group spell** | broke pairing | −2.6226 | 4.3314 | 0.5466 | 209,056 | +4.38 / 0.441, N 250,918 |

†Ownership FLOW β₃ is in raw fraction-of-float units, not ×10⁻⁶. The winsorized column is MAIN; the raw column is disclosed as outlier-dominated (§7.6 open issue) and carries no claim. RI = design-based randomization-inference p.

‡cum2/cum3 panel row-N are not stored in the hardening CSV, which records n_draws and the quarter-cluster count (cum2 80, cum3 79 of 82); their β₃ and p's come from `run_cum4_inference.py`. For the LP horizons the reporting arbiter is the **circular-shift RI**, the most conservative of the three design-based variants against the measured serial correlation of S_t (lag-1 autocorrelation +0.271), and under it every horizon is null. Legend: free-RI = free-permutation RI; circ = all-81 circular-shift; mb = moving-block L5(–L8); FWER = within-family single-step max-|t| over h0..cum4; WCB = score-based quarter-cluster Webb wild bootstrap (§ "Cum4 inference hardening").

**Bottom line for the reviewer.** Every valid specification returns a null β₃, on the rebuilt P0 holdings panel, under the three-way pairwise FE headline (−5.279916×10⁻⁷, p = 0.762748, RI free 0.8154 / circular 0.8293). All ten estimation families rerun on the P0 panel are null. Nine of them carry a design-based RI arbiter and all nine clear it; the risk-set / country-pair / spell-boundary family has no permutation arbiter (eight specs, §7.4) and is null on CRVE alone, and five further individual cells have no arbiter either, none of which rejects (§9, limitation 13b). The null survives a 10-construction shock menu with zero of 12 RI columns rejecting and a minimum arbiter p of 0.2439, and it survives at the extensive margin (arbiter 0.21–0.83), where divestment would appear first.

**Two retractions, both in this document rather than around it.** The historically significant result was a forward-window / few-cluster artifact and was retracted earlier (§8). The uniform positive-sign pattern, which every prior version used as a secondary reading, was a calendar artifact of the pre-P0 quarter-end stamping and is retracted now (§8b). Post-P0 the signs are mixed and the magnitudes are tiny; there is no directional reading of β₃ left to report.

**What is closest to rejecting, named honestly.** Outside the Russia control, the lowest **design-based arbiter** p in the battery is the aggregated-shock spec (RI circular-shift 0.2195, negative-signed, CRVE 0.094 on 3pw and 0.097 on it+gt, with no arbiter at all on the 3pw cell); it is the closest cell on the arbiter and it is still null. Three cells carry *lower CRVE* p's — the preferred menu variant at 0.0315, `s_C_ii` at 0.0262, and the extensive-margin exit it+gt companion at 0.0437 — and each of them either dissolves under RI or has no arbiter. The Russia positive control is the lowest CRVE p anywhere (0.0178 / 0.0189), correctly signed, and it fails the arbiter at 0.2085 / 0.2198 with all eight LP horizons negative and every permutation p ≥ 0.27. Russia validates *direction only* and reinforces the power limitation. The preferred shock-menu variant crosses CRVE at 0.0315 and dissolves at RI 0.2439. The country-pair β₁, once the one borderline coefficient here, is now dead at p ≥ 0.91 at every clustering level.

**Two things this cycle leaves open, stated rather than smoothed.** At cum4, four design-based routes clear 0.05 and one score-based wild bootstrap reads 0.0038; RI governs by the pre-registered hierarchy because the WCB assumes cross-quarter independence that the measured serial correlation violates, so cum4 is reported as not surviving with the disagreement disclosed. And the raw shares-flow column is dominated by post-P0 outlier cells up to 1724.9× float, which is why the winsorized column is MAIN and why the +0.20 co-movement anchor in §7.6 has been withdrawn pending a predetermined guard.

The honest reading is that the ownership-flow channel shows no detectable differential US disengagement, and the project should be judged on (a) the panel, the as-of holdings rule, and the identification construction, and (b) the planned firm-level price channel (H2.2), not on a confirmed behavioral effect.

---

## 14. Code inventory (all files live in the repo — open them on GitHub)

The full source is in `github.com/fidio728/practice`, branch `essay2-code-review`, under `julia_descriptive/`. This document no longer embeds the code verbatim (that risked doc-vs-code drift); instead it maps each file to what it does. Run order for a rebuild: Julia `00`→`06`, then the Python panel builds, then the Stata do-files.

### Julia data-construction pipeline (run order 00→06)
| file | builds / does |
|---|---|
| `00_setup.jl` | config: the 28 `EU_COUNTRIES`, data paths, helpers |
| `01_master_files.jl` | master-file assembly |
| `02_china_exposure.jl` | Chinese supply-chain exposure from Revere edges; edge counterparty country **as of edge-start** (F12); `china_share` = CN CUSTOMER+SUPPLIER edges / total CUSTOMER+SUPPLIER edges (**B7**; competitor/partner excluded, `china_share_alltypes` kept for diagnostics); also emits `sell_share`/`buy_share` for §7.7 |
| `03_eom_etl.jl` (+ `03a_decompress_to_parquet.jl`, `03b_phase_c_diagnostics.jl`) | FactSet Ownership EOM ETL → `holdings_eom.parquet`. **P0 (2026-08-04): the as-of W=10 snapshot rule** replaces the exact-quarter-end filter; per (fund, security) take the latest report ≤ quarter-end within 10 days, stamp to quarter-end, retain `report_date_actual` and `asof_gap_days`. Panel grows 186,800,295 → **208,418,523** rows (+11.57%); the old panel is the `asof_gap_days=0` subset, bit-equal. See §5.0. Duplicate-key **hard-fail** (F13/dup-key) |
| `04_us_ownership_european.jl` | `I_ict` (group USD holdings) + market cap; US/NONUS split; **primary-EQ float** (`fsym_id = fsym_primary_id`) |
| `05_combine_visualize.jl` | AR(1) GPR shock (quarter-end residual); combine; diagnostics |
| `06_cartesian_grid.jl` | Cartesian grid firm×group×quarter + zero-fill + backward Δw → `merged_us_eu_zero_filled.parquet`. **Caveat F2:** grid spans ALL quarters, not each firm's existence span |
| `debug_cusip_match.jl`, `debug_id_coverage.jl` | identifier-match diagnostics |

### Python panel builds
| file | builds |
|---|---|
| `build_c6_panel.py` | main w-based estimation panel `c6_panel.dta` + hard asserts (pairing, dedup, NULL-aware weight-sum, quarter-coverage) |
| `build_country_pair_shock.py` | country-pair S_{c,t} panel `c6_panel_country_pair.dta` |
| `build_spell_riskset.py` | firm-quarter risk-set panel `c6_panel_riskset.dta` (`build_spell_boundary.py` superseded) |
| `build_ownership_share_panel.py` | #5 step 1-2: primary-EQ ownership + diagnostics → `ownership_share_observed.parquet`, `ownership_share_float.parquet` |
| `build_ownership_share_c6_panel.py` | #5 step 3: ownership **FLOW** panel `ownership_c6_panel.dta` (F6: `(held_t−held_{t−1})/out_{t−1}`) |
| `build_audit_panel_f1f2f7.py` | F1/F2/F7 audit panel `audit_c6_panel.dta` (leads, cumulative, `in_span`, raw GPR) |
| `build_riskset_lagonly.py` | F8 lag-only risk set `c6_panel_riskset_lagonly.dta` |
| `build_shocklag_panel.py` | §7.3b: shock lagged to S_{t-1}, paired with the existing CN_{t-1} → `shocklag_panel.dta` |
| `build_sagg_panel.py` | §7.3c: within-quarter aggregated shock `s_agg` → `sagg_panel.dta` |
| `build_fourgroup_panel.py` | §7.8: active/passive four-group panel (funds master ~2018-08 snapshot) → `fourgroup_panel.dta` |
| `02_russia_exposure.jl` (isomorphic copy of `02_china_exposure.jl`) | Russia positive-control exposure panel `firm_quarter_russia_exposure.parquet` |
| `06_russia_grid.jl` (isomorphic copy of `06_cartesian_grid.jl`) | Russia positive-control zero-filled grid `merged_us_ru_zero_filled.parquet` |
| `build_russia_shock.py` | US-Russia AR(1) GPR shock (isomorphic to `build_country_pair_shock.py`) → `russia_shock_monthly.csv` |
| `build_russia_c6_panel.py` | Russia estimation panel (isomorphic to `build_c6_panel.py`) → `c6_panel_russia.dta` |
| `build_russia_lp_panel.py` | Russia cumulative event-study panel (2022Q1-2023Q4 vs 2021Q4 base) → `russia_lp_panel.parquet` |

### Stata regressions
| file | runs |
|---|---|
| `run_headline_3pairwise.do` | **HEADLINE** — three-way pairwise FE (it+gt+ig) for every main spec, vs it+gt |
| `run_ownership_share.do` | ownership FLOW triple difference (§7.6) |
| `run_audit_f1f2f7.do` | F1 lead-flow + local-projection IRF, F2 in-span headline, F7 GPR two-interaction |
| `run_audit_f4f8.do` | F4 country-pair β₁ under 3 clustering levels, F8 lag-only risk set |
| `run_audit_f1b_robust.do` | F1b CRVE-vs-alternate-FE sensitivity (see also RI) |
| `07g_spell_riskset.do` | risk-set regression; `07_regression.do`/`07b`/`07c`/`07d`/`07e` legacy main/robustness (`07f` superseded) |
| `run_shocklag.do` | §7.3b advisor shock-timing spec (S_{t-1}) — CRVE + SD-standardized reporting |
| `run_tercile_3pairwise.do` | §7.3 shock-tercile dose menu (T2 base; replaces tail-dummy menu) |
| `run_sagg_distlag.do` | §7.3c aggregated-shock (`s_agg`) spec |
| `run_direction_split.do` | §7.7 buy-side vs sell-side split (link-share + indicator columns) |
| `run_fourgroup.do` | §7.8 active/passive four-group split (post-2018 = primary) |
| `run_flow_decomp_step3.do` | §7.6 flow-identity decomposition regressions (six cells) |
| `run_russia_headline.do` | Russia positive control: continuous-shock (it+gt, 3-pairwise) + 2022Q1-Q2 event dummy |

### Design-based inference — Russia extension
| file | does |
|---|---|
| `run_ri_shocklag.py` | RI for the S_{t−1} spec (§7.3b); on the P0 panel RI p = 0.6018, null (pre-P0 0.814) |
| `run_ri_russia.py` | Russia continuous-shock RI (it+gt) + 2022Q1-Q2 vs all rolling 2-quarter windows placebo |
| `run_ri_russia_3pairwise.py` | Russia 3-pairwise RI |
| `run_russia_lp_test.py` | Russia cumulative event-study, firm-level permutation test per horizon (h=0..7). §7.9 on the P0 panel: all 8 horizons negative, **none below 0.05 under RI** (perm p 0.2703–0.4826, minimum at h=1; h0 = 0.4004; n = 4,294 firms/horizon); validates divestment *sign* only |

### Design-based inference (Python)
| file | does |
|---|---|
| `run_randomization_inference.py` | exact RI: US−NONUS pairwise-difference collapse reproduces the two-way-FE β₃; permutes the 82 quarter shocks → `audit_randomization_inference.csv` |
| `run_ri_3pairwise.py` | two-way-FE (firm+quarter) RI for the 3-pairwise headline + LP horizons → `audit_ri_3pairwise.csv` |
| `run_ri_tercile.py` | §7.3 tercile RI (re-cuts terciles per permuted shock) → `ri_tercile_results.csv` |
| `run_ri_sagg.py` | §7.3c aggregated-shock RI (free-permutation + circular-shift) → `sagg_ri_check.csv` |
| `run_ri_direction.py` | §7.7 direction-split RI (sell / buy / sell−buy) → `ri_direction_results.csv` |
| `run_ri_fourgroup.py` | §7.8 four-group RI (full + post-2018) → `ri_fourgroup_results.csv` |
| `run_ri_flow.py` | §7.6 flow RI (raw + winsorized) → `ri_flow_results.csv` |
| `run_flow_decomposition.py` | §7.6 flow-identity decomposition builder + RI → `flow_decomposition_panel.dta`, `ri_flow_decomposition.csv` |
| `run_cum4_inference.py` | G5/G7/G8 hardening of the LP family: 82-quarter shock serial diagnostics (ACF, Ljung–Box), RI three ways (free / all-81 circular-shift / moving-block L5,L8), within-family single-step max-\|t\| FWER over h0..cum4, and a score-based quarter-cluster Webb WCB; reads the same frozen `audit_c6_panel.dta` as `run_ri_3pairwise.py` → `cum4_inference_hardening.csv` |
| `run_ri_extensive.py` | §7.10 extensive-margin RI (d_breadth / d_nh / exit / init; free + circular-shift; full + riskset) → `ri_extensive_results.csv` |
| `run_ri_shockmenu.py` | §7.3d shock-menu RI for 12 columns: 5,000 free permutations + 81 circular shifts, seed 20260702, `arbiter_p = p_circ` → `ri_shockmenu.csv` |

### P0 rebuild and shock menu (2026-08-04)
| file | does |
|---|---|
| `diag_p0_asof_window.py` | the W-curve behind W=10, plus coverage / staleness / duplicate-key / US-share diagnostics → `output/diag_p0_*` (18 files) |
| `diag_p0_compare.py` | old-vs-new gates: weekday-vs-weekend coverage by era, US-share selection tilt, dollar totals, shock bit-identity on 265 quarters, grid and c6 deltas |
| `archive_preP0.py` | archives 56 pre-P0 artifacts as `*_preP0` and stamps vintage banners on 16 scripts |
| `build_shock_menu.py` | builds all 11 menu columns, runs the 1e-8 validation gate (hard-coded at line 180), computes whiteness diagnostics, applies the pre-registered selection rule stated verbatim in its header → `shock_menu_*` |
| `run_shock_menu.do` | 28 reghdfe specs (12 variants × 2 FE for dv=dw, 2 × 2 for dv=dw_lead1), B9 VCE guard, drift gate against the canonical headline |

**The three P0 ledger documents** (repo root, not `julia_descriptive/`). They are the authoritative sources behind this refresh and every number in it; a reviewer should read them alongside this document.

| file | role |
|---|---|
| `P0_REBUILD_RESULTS_2026_08_04.md` | the as-of W=10 data fix: the defect, the rule, the old-vs-new gates, the figure corrections, and the new canonical headline |
| `SHOCK_MENU_RESULTS_2026_08_04.md` | the 10 shock constructions, the pre-registered whiteness selection and its provenance, and the all-null result set (§7.3d) |
| `P0_BATTERY_RESULTS_2026_08_04.md` | every estimation family old-vs-new, plus the ADDENDUM with the relaunched RI legs and the two OPEN issues (raw-flow outliers, cum4 WCB tension) |

### Plots
`plots/fig13_two_panel.py`, `plots/plot_all.jl`, `plots/plots_python.py`.

**Figure corrections from P0 (advisor-facing).** Figure 2 (`fig_us_holdings_real_yoy_growth`) changes materially: 2022Q4 real YoY moves from **−42.8% to −25.9%**, while 2009Q1 stays **bit-unchanged at −39.7%** because that quarter-end was a clean Tuesday. The deepest-drawdown ranking therefore **reverses back to 2009Q1**, and the pre-P0 "deepest drawdown ever" reading of 2022Q4 was an artifact of the dropped weekend batch. 2023Q3 moves from −1.0% to **+29.1%**. The pattern-3 text already sent to the advisor needs the corresponding correction. Figure 1 (`fig_china_link_fraction`) is immune by design, because numerator and denominator come from the same source: maximum absolute change is **0.22pp** over 7,568 cells. Pre-P0 versions of both figures are retained on disk as `*_preP0.{png,pdf}`.

---

## 15. Result → source map (which file produces each headline number)

All values are **P0-vintage (2026-08-04)**.

> **[UPDATE 2026-08-08 — REBUILD v3: every value in this table is a RETIRED vintage.]** The panel has since been rebuilt twice (EM quarter-snapshot 2026-08-06; MM-FIX v2 + fund-grain REBUILD v3 2026-08-08) and the primary spec moved to GLOBAL denominator × S_{t−1}. Current headline cells live in `output/headline_3pairwise_canonical.csv` (LOCKED layout `spec,denom,timing,fe,b3,se,p,N`); do not cite the numbers below as current.

| result | value | produced by | reads / writes |
|---|---|---|---|
| Headline β₃ (w-based, it+gt+ig) | **−5.279916×10⁻⁷** (SE 1.743166×10⁻⁶, p 0.762748, N 347,690) | `run_headline_3pairwise.do` | `audit_c6_panel.dta` → `headline_3pairwise_canonical.csv` |
| Headline RI, hardening stream | RI free **0.8154** / circ **0.8293** (the `audit_ri_3pairwise.csv` free-permutation stream reads 0.8014; Monte-Carlo noise, see the note in "Cum4 inference hardening") | `run_cum4_inference.py` / `run_ri_shockmenu.py` | `cum4_inference_hardening.csv` (h0 row), `ri_shockmenu.csv` (`baseline_panel_shock_raw` row) |
| Headline β₃ (w-based, it+gt) | −1.072712×10⁻⁶ (SE 1.819055×10⁻⁶, p 0.557028, N 348,156; RI 0.5984) | `run_headline_3pairwise.do` / `build_c6_panel.py` | `audit_c6_panel.dta` / `c6_panel.dta` |
| Holdings as-of rule (P0) | W=10; panel 208,418,523 rows; US-share tilt −3.41pp → +0.01pp; c6 348,156 / 6,867 / 82; grid 12,791 firms (+48) | `03_eom_etl.jl` / `diag_p0_asof_window.py` / `diag_p0_compare.py` | `holdings_eom.parquet`, `output/diag_p0_*` |
| Ownership FLOW β₃ (F6), winsor MAIN | **+7.48×10⁻⁵** (CRVE 0.8265; **RI 0.913**) | `run_ownership_share.do` / `run_ri_flow.py` | `ownership_c6_panel.dta` → `ownership_share_results.csv`, `ri_flow_results.csv` |
| Ownership FLOW β₃, raw (disclosed only) | −3.61×10⁻² (CRVE 0.2366; RI 0.021) — outlier-dominated, no claim | same | same |
| F1 lead-flow / local projection | all horizons null on CRVE and RI; 3pw cum4 free-RI 0.0636 in the RI artifact | `run_audit_f1f2f7.do` / `run_audit_f1b_robust.do` | `audit_c6_panel.dta` → `audit_f1f2f7_results.csv`, `audit_f1_lp_irf.dta` |
| F1/F3 randomization-inference p (it+gt) | headline 0.5984, lead 0.8775, LP h1 0.5260, h2 0.8647, h3 0.9711, h4 0.2962 | `run_randomization_inference.py` | `audit_c6_panel.parquet` → `audit_randomization_inference.csv` |
| randomization-inference p (3-pairwise) | headline 0.8014, LP cum1 0.9368, **cum4 0.0636** | `run_ri_3pairwise.py` | `audit_c6_panel.parquet` → `audit_ri_3pairwise.csv` |
| cum4 inference hardening (LP four routes + FWER + WCB) | free 0.0664 / **circ 0.1220 (arbiter → null)** / mb 0.1074–0.1076 / max-\|t\| FWER 0.0558–0.0618 / **WCB 0.0038 (disagrees)**; S_t acf1 +0.270747, LB(1) p 0.0125 | `run_cum4_inference.py` | `audit_c6_panel.dta` → `cum4_inference_hardening.csv` |
| F2 in-span headline | −1.5656×10⁻⁶ (p 0.5788, RI 0.6093, N 272,140) | `run_audit_f1f2f7.do` (`in_span==1`) | `audit_c6_panel.dta` |
| F4 country-pair β₁ across three clusterings | −1.36×10⁻⁶; p 0.9103 / 0.9432 / 0.9467 | `run_audit_f4f8.do` | `c6_panel_country_pair.dta` |
| F7 GPR two-interaction | −2.20×10⁻⁶ (0.1262) / +9.51×10⁻⁷ (0.3496) | `run_audit_f1f2f7.do` | `audit_c6_panel.dta` |
| F8 risk-set lag-only β₃ | −1.7674×10⁻⁶ (p 0.5443, N 268,800) | `run_audit_f4f8.do` | `c6_panel_riskset_lagonly.dta` |
| ADR off-primary share | US **8.79%** / NONUS **3.22%** (on-primary 91.2144% / 96.7794%) | `build_ownership_share_panel.py` | `ownership_share_diagnostics.csv` |
| Firm-count cascade | 12,791 universe → 10,171 Revere-matched → 8,014 non-null all-type CN → **6,867** in-panel (middle two are pre-P0, not re-derived) | `05_combine_visualize.jl` / `build_c6_panel.py` | `05_unmatched_profile_by_country.csv` |
| §7.3b shock-timing S_{t−1} | −1.0669×10⁻⁶ (CRVE 0.3696, RI 0.6018) | `run_shocklag.do` / `run_ri_shocklag.py` | `shocklag_panel.dta` → `shocklag_results.csv`, `shocklag_ri_results.csv` |
| §7.3c aggregated shock s_agg h=0 | −1.55×10⁻⁶ it+gt (CRVE 0.0974; RI free 0.2353 / circ 0.2195); −1.66×10⁻⁶ 3pw (CRVE 0.0939, no RI) | `run_sagg_distlag.do` / `run_ri_sagg.py` | `sagg_panel.dta` → `sagg_ri_check.csv` |
| §7.3d shock menu, preferred variant | `s_D_q_ar1_nla` −7.557×10⁻⁶ (CRVE 0.0315; RI free 0.1780 / **circ 0.2439**); gate 1.776×10⁻¹⁵; LB(4) p 0.3544; rule-dependent winner: LB(4)-first, C-ii is whiter at LB(8) 0.1319 vs 0.0848, an \|acf1\|-first rule would have picked C-i, and the selected shock is not white | `build_shock_menu.py` / `run_shock_menu.do` / `run_ri_shockmenu.py` | `shock_menu_diagnostics.csv`, `shock_menu_preferred.txt`, `shockmenu_results.csv`, `ri_shockmenu.csv` |
| §7.3 shock-tercile T3 | +2.77×10⁻⁶ (p 0.8823, RI **0.8478**); T3−T1 +2.07×10⁻⁶ (RI 0.8742) | `run_tercile_3pairwise.do` / `run_ri_tercile.py` | `tercile_results.csv` / `ri_tercile_results.csv` |
| §7.7 direction split | sell −1.9639×10⁻⁶ (RI 0.4735) / buy +1.0131×10⁻⁶ (RI 0.6701) / contrast RI 0.3171; pooling F(2,81)=0.77 p 0.4651 | `run_direction_split.do` / `run_ri_direction.py` | `direction_results.csv` / `ri_direction_results.csv` |
| §7.8 four-group active | full −7.065×10⁻⁷ (RI **0.7802**); post-2018 PRIMARY +5.048×10⁻⁷ (RI **0.7976**) | `build_fourgroup_panel.py` / `run_fourgroup.do` / `run_ri_fourgroup.py` | `fourgroup_results.csv` / `ri_fourgroup_results.csv` |
| §7.6 flow decomposition, winsor MAIN | flow_r −1.11×10⁻⁴ (RI 0.913), flow_common +3.46×10⁻⁴ (RI 0.641), flow_diff +1.53×10⁻⁴ (RI 0.833); identity 3.6×10⁻¹² | `run_flow_decomposition.py` / `run_flow_decomp_step3.do` | `flow_decomposition_panel.dta` → `ri_flow_decomposition.csv`, `flowdecomp_results.csv`, `flow_decomposition_diag.csv` |
| §7.6 co-movement anchor (withdrawn) | high-CN tercile corr(flow_US, flow_NONUS) = **+0.027** winsor / **+0.008** stable-float (was quoted as ≈ +0.20) | `run_flow_decomposition.py` | `flow_decomposition_diag.csv` |
| σ_S (bit-frozen across P0) | 2.57789, mean 0.16695, over 82 estimation quarters | `run_shocklag.do` / `build_shock_menu.py` | `shocklag_panel.dta`, `shock_menu_diagnostics.csv` |
| §7.9 Russia positive control | it+gt −3.84187×10⁻⁶ (CRVE 0.0189, **RI 0.2085**); 3pw −3.86876×10⁻⁶ (CRVE 0.0178, **RI 0.2198**); 8/8 LP horizons negative, perm p 0.2703–0.4826, n 4,294 | `run_russia_headline.do` / `run_ri_russia*.py` / `run_russia_lp_test.py` | `c6_panel_russia.dta` → `russia_headline_results.csv`, `russia_lp_test_results.csv` |
| §7.10 extensive margin | d_breadth +2.197184×10⁻⁵ (arbiter 0.5732), d_nh −1.453061 (0.2195), exit +1.41549×10⁻³ (0.6829), init −6.622296×10⁻⁴ (0.8293) | `run_extensive_margin.do` / `run_ri_extensive.py` | `extmargin_results.csv` → `ri_extensive_results.csv` |
| §7.10 motivating zero-holder gap | **+12.23pp** full grid (US 82.757% vs NONUS 70.527% of 1,279,100 cells) / **+15.18pp** estimable subset | `build_extensive_margin_panel.py` | `extmargin_descriptive.csv`, `extmargin_fourcell.csv` |

**Known stale strings in shipped artifacts (fix before quoting the CSV, not the number).** `tercile_results.csv` addnote says "MAIN drops 462 singletons"; the true count is **466**. `ownership_share_diagnostics.csv` carries `ref_main_panel_firms=6854`; the canonical is **6,867**. `shockmenu_vce_diag.csv` puts `se_valid` in column 7, not last, so a naive last-column parse falsely flags all 56 rows. `gpr_ar1_coefficients.csv` still persists a=0, b=0 from an old Julia let-scope bug; the shock itself is unaffected and `shock_menu_diagnostics.csv` is now the reliable AR-coefficient record. Roughly a dozen scripts retain pre-P0 header comments and hard-coded drift anchors that will print apparent mismatches; those were left unedited deliberately, and the mismatch is the result.

**One pre-P0 input still feeds a live estimate.** `output/firm_ladder_panel.dta` (2026-07-22) supplies the `bil_us_c` lookup for the DDD Stage-C bilateral control. `bil_us_c` is exposure-side and bit-frozen, so its values should be unchanged, but its firm-quarter *coverage* has not been re-verified against the P0 rebuild. The merge left 204 master rows unmatched and 23,402 `us_bil` values missing, which defines the 324,340-row bilateral subsample. Treat Stage C as provisional until it is rebuilt.

**DDD gate status (P0).** Stage A gate β₃ = −5.2799×10⁻⁷ against the target −5.279916×10⁻⁷ at N = 347,690: **PASS**. Stage B (no-FE full factorial) returns coefficients only, with **every SE, t, p, CI and the F statistic missing** because the variance matrix is singular/nonsymmetric, so **no valid inference comes from that column**. Stage C1 (bilateral, no BIL control) −3.88×10⁻⁷ (SE 1.98×10⁻⁶, p 0.845, N 324,340, 414 singletons, 6,119 firm clusters); Stage C2 (+ US×BIL) −5.23×10⁻⁷ (SE 1.98×10⁻⁶, p 0.792), with `us_bil` +2.91×10⁻⁵ (SE 1.67×10⁻⁵, p 0.084).
