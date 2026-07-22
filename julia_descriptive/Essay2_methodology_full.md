# Essay 2 — Full Methodology, Replication Log, and Honest Caveats

**Purpose.** This document is a complete, deliberately unflattering technical record of Essay 2, written to be handed to an independent reviewer (human or AI). It states the research design, every variable and how it is built, the exact code and commands, the data-cleaning decisions, all estimation results, and every caveat, limitation, and mistake we found and corrected. Nothing is hidden. Where the design is weak, underpowered, or where an earlier build was wrong, it is flagged explicitly.

**One-line status.** The headline coefficient is **statistically indistinguishable from zero** under the preferred backward-timing specification with the fully-saturated **three-way pairwise FE** (firm×quarter + group×quarter + firm×group; see "Second external review round"), with standard errors large enough that economically meaningful effects of either sign cannot be ruled out; the null is robust across fixed-effect choices, shock definitions, sample constructions, a firm-existence-span restriction, a raw-GPR-level shock, and — decisively — **design-based randomization inference** (which shows the only CRVE-significant coefficients, at cumulative local-projection horizons, to be few-cluster/overlapping-window artifacts; the lowest randomization-inference p is 0.07, at one cumulative horizon under the saturated FE — every single-quarter spec is ≥ 0.23 — and every point estimate is *positive*, i.e. opposite to disengagement). It also holds on a **shares-based ownership measure** (§7.6): US investors do not reduce their ownership *stake* — a pure trading flow, the change in group shares held over the lagged primary-EQ float (F6) — any more than non-US investors when tension rises (β₃ = +0.00088, p=0.44, positive and insignificant), so the null is not an artifact of value-weighting — subject to an ADR-exclusion caveat. An earlier significant estimate came from a forward-window specification vulnerable to post-treatment timing contamination, and is retracted. The project's current contribution is a measurement/design one (the panel + identification), and the empirical question is treated as open, with a planned pivot to a firm-level price channel.

---

## Change log for this review round (read first)

This document was hardened over one working session in response to two external AI reviews plus several rounds of internal adversarial multi-agent review. Everything below is reflected in the body (§1–§13) and in the source code in the repo (`github.com/fidio728/practice`, branch `essay2-code-review`, under `julia_descriptive/`; see the §14 code inventory and §15 result→source map). This section states **what changed and why**, so a new reviewer can see the full provenance. No data was fabricated or altered; every number cited was re-derived from the on-disk parquet/dta/csv files.

### A. Correctness and honesty fixes to the existing w-based pipeline (applied + verified)

1. **Universe wording downgraded to "holdings-observed European issuer universe"** (§3 title, §3 Caveat 2, §5.2, §7.4). The firm universe is every `sec_entity_id` that ever appears with a European `sec_country` in the FactSet Ownership holdings panel — **not** the full FactSet Security Coverage listing universe. Estimand phrasing corrected accordingly. This does not change β₃; it makes the claim honest. (Definition of "European-listed": a security in FactSet Ownership whose `SEC_FIRM_ISO_COUNTRY` ∈ the 28 jurisdictions of §3, restricted to those held by ≥1 institution.)
2. **Attenuation claim softened** (§3 Caveat 2, §9 limitation 11). Rebuilding from the full Security Coverage universe is *expected* to attenuate β₃ toward zero (added zero-difference rows), but this is the mechanical expectation under a simplified argument, to be **verified by running the rebuild, not asserted** — corrected an earlier over-strong "would reinforce the null." Also corrected an earlier wrong claim that never-held firms "do not affect β₃" (they do, via attenuation).
3. **Retraction language softened** (§0 one-liner, §8). "Causally clean specification" / "look-ahead artifact" → "preferred backward-timing specification" / "forward-window specification vulnerable to post-treatment timing contamination." We do not claim to have decomposed *why* the SE tripled (§8).
4. **Firm-count cascade clause added** (§2): 12,743 European securities → 10,171 Revere-matched → 8,014 carry a non-null China share → 7,928 carry a non-null lagged CN and enter the estimation panel. The gaps are the documented MISSING bucket and the one-quarter lag, not contradictory counts. All five figures re-derived from `merged_us_eu_zero_filled.parquet` and `05_unmatched_profile_by_country.csv`.
5. **Revere match country structure disclosed** (§9 limitation 12): GB 32.78% unmatched (1,307 of 3,987), the worst among large countries; GB is 60.85% of the country-pair subsample, so the matched sample drops ~⅓ of GB issuers (selection concern).
6. **Hard asserts added to the build scripts** (`build_c6_panel.py`, `build_spell_riskset.py`): duplicate-key fail, full US/NONUS pairing per firm-quarter, one common shock per quarter, `cn_lag ∈ [0,1]`, contiguous quarter coverage, and a **NULL-aware group-quarter weight-sum** (portfolio weights sum to 1 within each group-quarter; an all-NULL cell is allowed only when the group's book is empty, `I_ict = 0`, and a held-but-all-NULL cell hard-fails as a 06 bug — 0 found). Verified: 199 non-empty group-quarters sum to 1 with max abs deviation 4.44×10⁻¹⁶; 82 contiguous quarters.
7. **Schema-collision fix**: `06_cartesian_grid.jl` and `plots/fig13_two_panel.py` both wrote `05_diff_us_vs_nonus_high_c6.csv` with different schemas. fig13 now writes its own `fig13_diff_{all,high}_c6.csv`; the orphaned `05_diff_us_vs_nonus_all_c6.csv` was deleted; the three similarly-named CSVs are documented in the §16 artifact map. No downstream reader breaks (grep-verified).
8. **`04_us_ownership_european.jl` duplicate-key check upgraded from `@warn` (>1,000) to a hard `error` (n_dup > 0)**. Measured 0 duplicate `(fund_id, fsym_id, report_date)` keys on the 186,800,295-row `holdings_eom.parquet`, so the change is de-risked; it fires on a future ETL run only if a raw pull reintroduces duplicates. This matters because a raw fund-level duplicate would double-count dollars in `SUM(adj_mv)` and neither the final dedup assert (aggregated keys) nor the weight-sum check would catch it.
9. **Cosmetic**: corrected the `I_ict` Stata variable label (`build_country_pair_shock.py`) from "ICT industry indicator" to "Group holding value, USD"; `I_ict` is not used in any regression.

### B. New empirical work — the shares-based ownership test (§7.6), the "make-or-break"

The market-value portfolio weight `w = H(USD)/T(USD)` mixes trading flow with price moves and portfolio-denominator reallocation, so a null on `w` supports "US do not reduce their portfolio *weight*" but not "US do not *sell* / do not reduce their *stake*." To license the flow claim we built a **pure trading-flow** outcome (§7.6, F6): the change in group shares held over the *lagged* primary-EQ float — immune to price and to float changes (buybacks/issuance) — on the same C6 grid, three-way pairwise FE, and clustering as the main spec. **Result: β₃ = +0.00088 (SE 0.00115, p=0.44), null and if anything positive** — agrees with the w-based null (the earlier Δ(ownership_share) form, +0.000599, is retained only as a comparison column). Robust to dropping the extensive-margin zeros. Two required caveats: ADR exclusion (US holds 8.92% of European exposure off the primary class vs 3.32% for non-US) and limited power against small adjustments (80%-power MDE ≈ 4–12 bps of float). New files: `build_ownership_share_panel.py`, `build_ownership_share_c6_panel.py`, `run_ownership_share.do` (all in §14–§15).

### C. Adversarial multi-agent reviews run this round (all findings resolved)

| review id | scope | verdict |
|---|---|---|
| `wbo5chyw9` | the code fixes in A6–A9 | ready, 0 defects |
| `wutg4fnkw` | the weight-sum / quarter-coverage asserts, artifact map, doc wording | ready, 0 must-fix |
| `wqtl83l3b` | doc-scoped: every quantitative claim re-derived vs code/data; embedded code vs disk | ready, provenance verified, 0 must-fix |
| `wsu1zupkc` | the shares-based build | **needs_fix → caught a real bug** (first draft pooled all `fsym_id` classes and MODE-masked a dual-universe `shares_out`, 5.28% of cells dispersed up to 1e12×). Fixed by restricting to `fsym_id = fsym_primary_id`, which drives dispersion to exactly 0. |
| `w86zvcvsx` | the shares-based result | result_trustworthy, 0 must-fix; β₃ independently reproduced to ~2×10⁻⁷; null confirmed not a zero-fill artifact |

### D. Current conclusions (detail in §7, §8)

The headline β₃ (US × China-exposure × tension shock on within-Europe reallocation) is a **robust statistical null** — across FE choices, shock definitions, sample constructions, the value-weighted portfolio weight, **and** the shares-based ownership stake. If anything, every point estimate is *positive* — the opposite sign to the disengagement hypothesis (e.g. the shares-based flow β₃ = +0.00088). The one historically borderline coefficient (country-pair β₁) is not reliably inferred once clustered at the honest country level (p 0.098→0.23) and is not cited as evidence. The earlier "significant" result was a retracted forward-window/few-cluster artifact. The project's contribution is currently a measurement/identification one (the paired firm×quarter / group×quarter panel), and the ownership-flow question is treated as answered-in-the-null, with the economic action (if any) to be sought in a firm-level price channel (H2.2).

### E. Next steps (detail in §10)

In priority order: **(#7) inference robustness** — wild-cluster bootstrap (important with only 82 quarter-clusters), leave-one-quarter-out, and a shock-timing menu (quarterly sum/avg vs quarter-end); **(#6) exposure robustness** — `rel_type` filter, edge dedup, quarter-as-of counterparty country; **(#8) holder-level panel** — holder × quarter FE, the true Khwaja–Mian bank×time analog; **(H2.2) price channel** — acquire European returns and test abnormal returns / valuation directly; **ADR-inclusive ownership** — map ADR holdings to underlying-share equivalents (needs the ADR conversion ratio) to close the §7.6 Caveat 1 gap.

---

## Second external review round (F1–F13) — responses, new tests, and the three-way pairwise FE headline

A second (cloud-based) AI review raised 13 findings — F1 critical, F2–F7 major, F8–F13 minor — all touching either the identification/timing of the headline or the reliability of the inference. Every one was verified against the on-disk data, addressed with new code (all in the repo; see the §14 inventory), and re-run. **The disengagement null (β₃ < 0) survives every fix, and the only CRVE-significant coefficients (at cumulative local-projection horizons) are not significant under valid, design-based inference.** All new numbers were re-derived from the parquet/dta files; nothing was fabricated. Two headline changes follow directly.

### Headline change 1 — three-way pairwise FE (Khwaja–Mian / De Haas saturation)

The headline is now the **fully-saturated three-way pairwise FE**: firm×quarter (`it`, α_{i,t}) + group×quarter (`gt`, γ_{g,t}) + **firm×group (`ig`, μ_{i,g})**; the earlier `it+gt` is kept as a comparison column. On the balanced 2-per-firm-quarter panel, `gt` absorbs β₀(US) and β₁(US·S); `it` absorbs the CN/S/CN·S levels; so only β₂(US·CN) and β₃(US·CN·S) are identified. On the differenced outcome, `ig` is a firm-specific drift control on the US−NONUS difference (via the pairwise collapse, the 3-pairwise = firm FE + quarter FE on Δy). The null surviving the fully-saturated FE is a **stronger** null claim. (The true Khwaja–Mian holder×quarter saturation needs the holder-level panel — deferred, §10.)

| specification | 3-pairwise (it+gt+ig) β₃ | it+gt β₃ | verdict |
|---|---|---|---|
| headline Δw ~ US·CN·S_t | +1.80×10⁻⁶ (p=0.35) | +1.28×10⁻⁶ (p=0.44) | **null** |
| F1a lead-flow Δw_{t+1} ~ S_t | +1.9×10⁻⁷ (p=0.92) | −2.4×10⁻⁷ (p=0.90) | **null** |
| F2 headline, in-span only | +2.5×10⁻⁶ (p=0.42) | +2.4×10⁻⁶ (p=0.43) | **null** |
| F7 US·CN·GPR_t / GPR_{t−1} | −2.0×10⁻⁶ (0.34) / +7×10⁻⁷ (0.74) | −2.1×10⁻⁶ (0.28) / +9×10⁻⁷ (0.64) | **null** |
| ownership FLOW (F6) | +8.8×10⁻⁴ (p=0.44) | +8.9×10⁻⁴ (p=0.44) | **null** |
| F1b LP cum h1 (CRVE only) | +2.4×10⁻⁶ (p=0.03) | +1.4×10⁻⁶ (p=0.01) | **CRVE invalid here → not significant under RI (0.23/0.47), F1/F3** |
| F1b LP cum h4 (CRVE only) | +6.6×10⁻⁶ (p=0.08) | +3.9×10⁻⁶ (p=0.04) | **CRVE invalid here → not significant under RI (0.07/0.23), F1/F3** |

Every point estimate is **positive** — the opposite sign to the disengagement prediction (β₃ < 0) — so even the CRVE-significant cumulative horizons, if believed, would *strengthen* the no-disengagement reading, not overturn it.

### Headline change 2 — β₁ removed from directional evidence (F4)

The country-pair β₁ (US·S_{c,t}, formerly "+17.7, p=0.098, opposite H2.1") is **removed from all directional-evidence sentences** (§0 index, §7.5, §13). S_{c,t} varies only across 3 listing countries × quarter, so the honest clustering level is the country: clustering by `sec_country` (3 clusters, df=2) moves β₁ from p=0.098 to **p=0.227** (β₁ unchanged at +1.77×10⁻⁵; the SE barely moves but the degrees of freedom collapse). It is not reliable evidence of anything.

### Finding-by-finding

- **F1 (critical — timing / retraction logic).** The concern: S_t is the quarter-*end* AR(1) residual while backward Δw_t spans the whole quarter, so the preferred spec allows only ~1 month of reaction, and the natural t+1-quarter response window was never tested; the SE tripling in the centered→backward retraction is not the signature of "removing look-ahead noise." Resolution: (i) the **lead-flow** Δw_{i,g,t+1} ~ US·CN·S_t (pure lagged shock, no look-ahead) is **null** (p=0.92); (ii) a **local-projection IRF** of the cumulative response w_{t+h}−w_{t−1} on US·CN·S_t (h=0..4) has positive point estimates that reach CRVE p<0.05 at h=1 and h=4 — **but the CRVE is not valid there** (overlapping cumulative windows + few quarter-clusters; `reghdfe` flags a non-PSD VCV, and the p flips 0.013→0.111 under an equivalent FE parameterization). Under valid design-based **randomization inference** (which has exact size; permuting the 82 quarter shocks) these horizons are **not distinguishable from zero** — every horizon null (headline 0.62, lead 0.92, LP h1 **0.47**, h2 0.25, h3 0.32, h4 **0.23**). Under the saturated three-way pairwise FE the RI likewise rejects the CRVE significances (`run_ri_3pairwise.py`: headline 0.51, LP cum1 **0.23** [CRVE 0.03], LP cum4 **0.07** [CRVE 0.08]) — all null, all positive. So there is no US-differential response at t, at t+1, or cumulatively; the "reaction is in t+1" alternative is rejected. `run_audit_f1f2f7.do`, `run_randomization_inference.py`. (The RI is exact only if the quarter shocks are exchangeable; under residual serial correlation the p is approximate — a block-permutation / circular-shift variant is a ~5-line robustness, deferred (R2-N1). The sharp null it tests is slightly broader than β₃=0: were the shock to act through a firm-level channel other than CN, the test could reject for non-β₃ reasons — R3-C.)
- **F2 (Cartesian grid vs firm existence span).** `06_cartesian_grid.jl` crosses the universe with ALL quarters 1999Q1–2023Q4 without intersecting each firm's own existence span, so pre-IPO / post-delisting structural Δw=0 rows enter the estimation panel. Measured: **25.03%** of the panel is out-of-span, ~96.6% of it exactly Δw=0. These attenuate β₃, inflate N, and understate SE. Re-running the headline on the in-span subset only: β₃ ≈ 1.86× larger (+1.28→+2.4×10⁻⁶) but still **null** (RI p=0.58). The conclusion is unchanged; the headline N / SE / §5.5 extensive-margin figures are affected. `build_audit_panel_f1f2f7.py` (`in_span`).
- **F3 / F9 (inference — few clusters, overlapping windows).** The tail-dummy specs (§7.3, 6/4/2 treated quarters) and the overlapping-window LP have unreliable CRVE (MacKinnon–Webb; `reghdfe` flagged a non-positive-semi-definite VCV on the LP), and the previously-planned wild cluster bootstrap fails with few treated clusters. **Randomization inference** is the correct design-based test and is now the arbiter for these specs. Its exact algebra (US−NONUS pairwise difference + quarter FE reproduces the two-way-FE β₃; per-quarter sufficient statistics make a permutation an O(82) weighted sum) was independently verified to reproduce reghdfe's β₃ to 7 significant figures. `run_randomization_inference.py`. The tail-dummy k=2/3 specs (2 and 4 treated quarters) are **inference-invalid** and reported as such / dropped, not as "low power."
- **F4 (country-pair β₁ clustering).** See Headline change 2 above. `run_audit_f4f8.do`.
- **F5 (MDE units).** The §7.6 "25–35 bps" MDE conflated the per-unit-CN·S coefficient scale with the outcome scale. Corrected: with SE(β₃_flow)=1.15×10⁻³ and σ(S_t)=2.578 (see σ_S note, §7.3), the 80%-power MDE for a *representative* firm-quarter (CN∈[0.05,0.15], 1σ shock) is **≈ 4.1–12.4 bps of float** (0.7–2.2% of the outcome sd), not 25–35 bps; the quarter-end-only shock is classical measurement error that attenuates β₃ and enlarges the true MDE. §7.6 Caveat 2 corrected.
- **F6 (ownership flow denominator).** The §7.6 dos = ownership_share_t − ownership_share_{t−1} was NOT denominator-immune: with each term over its own *current* float, a buyback/issuance moves it with zero trading, by a term ∝ the group's own lagged level (differs across US/NONUS, so not absorbed by firm×quarter FE). Fixed: the primary outcome is now the pure flow **(held_t − held_{t−1}) / out_{t−1}** (fixed lagged float). β₃ = +8.8×10⁻⁴ (p=0.44), **null** — same conclusion, but the "net buying/selling" language is now literally correct. The old dos is retained as a labelled comparison column. `build_ownership_share_c6_panel.py`.
- **F7 (generated regressor / full-sample AR(1)).** The AR(1) shock is estimated once on the full ~1957–2023 monthly series (its (a,b) embed future data) and is a generated regressor. Robustness: replacing the composite shock with **US·CN·GPR_t + US·CN·GPR_{t−1}** (raw GPR level + previous-quarter lag) gives both interactions **null** (p=0.34 / 0.74). This nests the *quarterly* AR(1) family and removes the full-sample look-ahead, but note it does not exactly reproduce the monthly-fitted residual, whose autoregressive term is the quarter's second-to-last month gpr(M2), not the previous quarter's gpr — an exact-nesting variant adding a `US·CN·gpr(M2)` interaction is a one-column addition, deferred (Second review round, R2-F2). Disclosed in §9. `run_audit_f1f2f7.do`.
- **F8 (risk-set lead membership).** The main risk set (§7.4) conditions membership on t+1 holdings (a post-treatment variable). A **lag-only** variant (held at t or t−1, no look-ahead) gives β₃ = +2.6×10⁻⁶ (p=0.44), **null** — insensitive to the membership rule. `build_riskset_lagonly.py`.
- **F10 (multiple testing).** ≥19 coefficient tests were reported; the single p<0.10 (country-pair β₁) is the expected number of false positives at α=0.10 and was used asymmetrically as "direction opposite H2.1." That framing is removed; the narrative is a clean null. (See F4.)
- **F11 (ADR-share docstring).** `build_ownership_share_panel.py` reported 91.57%/96.99% on-primary (an EQ+AD calc); corrected to the EQ-primary measure this build uses, 91.08%/96.68% on-primary (US 8.92% / NONUS 3.32% off-primary), matching §7.6.
- **F12 (edge as-of classification).** `02_china_exposure.jl` classifies a supply-chain edge's counterparty country **as of the edge start date**, not per quarter — better than a look-ahead, but the "point-in-time" wording is clarified to mean edge-start, not quarter-by-quarter re-classification.
- **F13 (version control).** The `julia_descriptive/` pipeline was local-only (untracked, hard-coded absolute paths). The code is now committed and pushed to `github.com/fidio728/practice` so it is reviewable; a `.gitignore` keeps the 5 GB `output/` data out of version control.

### New files this round (in the repo; see the §14 inventory)
`build_audit_panel_f1f2f7.py`, `run_audit_f1f2f7.do`, `run_audit_f1b_robust.do`, `run_randomization_inference.py`, `run_ri_3pairwise.py`, `run_headline_3pairwise.do`, `build_riskset_lagonly.py`, `run_audit_f4f8.do`, and the F6 rewrite of `build_ownership_share_c6_panel.py`.

---

## 0. Reviewer's quick index of things most worth attacking

1. The zero-fill (§5): does imputing "absence = zero weight" create or destroy the object of interest? We keep the full Cartesian grid as the main panel but a reviewer should press on whether the extensive-margin gap is causal or mechanical.
2. The backward-vs-centered differencing (§4.1, §8): the entire significance of the headline flips on this. We argue the centered version had look-ahead contamination; a reviewer should verify that argument.
3. Power (§7.3): the tail-shock design has 6/4/2 treated quarters at k = 1.645/2/3. This is thin. The continuous shock is also arguably underpowered given only 82 quarters.
4. The "US vs non-US" contrast requires both groups present within a firm-quarter (§7.4). Our first conditional-sample build violated this (per-group selection); it is documented as an error and corrected.
5. Data coverage assumption (§5.5): after zero-fill, Δw = 0 cannot separate "truly held nothing" from "FactSet did not capture the institution."

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

**Identifier merge (coverage cascade, real counts):** of 12,743 distinct European-listed securities in the holdings universe, CUSIP populated for all 12,743, ISIN for 1,177, SEDOL for 0; 10,171 match to Revere. A security that fails to match is treated as **no observed supply-chain coverage**, not zero Chinese exposure.

The firm count then steps down further, and the three figures below measure **different objects** (do not read them as inconsistent): of the 12,743 securities, 2,572 fail the Revere match (see `05_unmatched_profile_by_country.csv`, which sums to 12,743 = 10,171 matched + 2,572 unmatched), leaving **10,171 Revere-matched**. Of those, **8,014** carry a non-null China share in at least one quarter — the rest are matched but have no *active* supply-chain link in any observed quarter, so `NULLIF(n_total_links, 0)` routes them to the MISSING bucket (§5.5, and the `NULLIF` at 02/06). Lagging the exposure one quarter (`china_share_lag1q`) leaves **7,928** securities carrying a non-null lagged China share, which is the distinct-firm count of the estimation panel. So 10,171 (identifier match) > 8,014 (non-null CN) > 7,928 (non-null lagged CN, in-panel); the gaps are the documented MISSING bucket and the one-quarter lag, not stale or contradictory counts.

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

**HIGH-exposure cutoff.** `HIGH` if CN_{i,t−1} > **0.0625** = the median of strictly-positive CN_{i,t−1} across firm-quarters (CUSTOMER+SUPPLIER share, post-B7 rebuild 2026-07-22; the pre-B7 all-`rel_type` value was 0.0476). **Disclosed nuance:** "high exposure" therefore means "at least ~6% of a firm's reported supply-chain edges touch China" — meaningful but **not extreme**. The estimation universe is now **6,854 firms** (was 7,928; firms whose only China links were competitor/partner ties, with no customer/supplier link, now carry NULL exposure and drop out).

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

**Tail-dummy variants (advisor-requested, §7.3).** ShockTail_t = 1[ (S_t − mean)/sd > k ], one-sided right (escalation), sd computed over the **82 distinct quarters** (not the 462k rows), for k ∈ {1.645, 2, 3}.

### 4.4 US indicator

US_g = 1 if g = US, 0 if g = NONUS.

---

## 5. Panel construction (C6 Cartesian grid + zero-fill) and cleaning

### 5.1 The problem the zero-fill fixes (selection on the outcome)

Raw FactSet records **only held positions**. If a group's holding in a firm is zero, there is **no row**. But H2.1 predicts US moves *toward* zero on high-exposure firms; keeping only non-zero rows deletes exactly the entry/exit events the design must measure. That is conditioning on the outcome.

**Real example (verified in data).** Firm `sec_entity_id = 05HF13-E` (GB-listed, high-CN), US group: held ~$7.5B through 2021Q4, then **$0 in 2022Q1** (full liquidation). In raw data the firm simply disappears from the US panel after 2021Q4; the −Δw of the liquidation is unobservable. Under zero-fill, 2022Q1 becomes a row with Δw = 0 − w_{2021Q4} < 0, so the exit is captured.

### 5.2 Fix: full Cartesian grid + zero-fill (audit item C6)

Unit of observation = every holdings-observed European issuer (§3, Caveat 2) × quarter × {US, NONUS}, whether or not held. Δw is 0 when the group held nothing at both t−1 and t; < 0 at an exit (−w_{t−1}); > 0 at an entry (+w_t).

Grid diagnostic (`06_cartesian_grid.jl`, backward diff): **2,510,371 non-null Δw cells; 38,229 first-quarter-null cells; 0 interior nulls.**

### 5.3 Data-cleaning / audit fixes (C1–C6)

| Fix | What |
|---|---|
| C1 | Ownership denominator restricted to European-listed equity (w = within-Europe share) |
| C2 | Pre-2003 supply-chain exposure = NULL, not zero (Revere begins 2003) |
| C3 | One-quarter lag via window function (genuine LAG, not row shift) |
| C4 | Symmetric edge counting (bilateral union of firm-as-source and firm-as-target CN links) |
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
    shock_us_cn                                     AS shock,
    CASE WHEN holder_group = 'US' THEN 1 ELSE 0 END AS us
FROM read_parquet('merged_us_eu_zero_filled.parquet')
WHERE delta_w IS NOT NULL AND china_share_lag1q IS NOT NULL AND shock_us_cn IS NOT NULL
```

Result: **c6_panel.dta = 462,564 firm-group-quarter rows, 7,928 firms, 82 quarters (2003Q3–2023Q4), 50/50 US/NONUS.** Written with `to_stata(convert_dates={'rdate':'tc'}, version=118)`; pandas writes datetime as %tc so Stata must `dofc()` before `mofd()`.

### 5.5 Honest data caveats

- **Zero-fill ambiguity.** After the fill, Δw = 0 cannot distinguish "the group truly held nothing" from "FactSet did not capture an institution." US 13F reporting is mandatory above the $100M AUM threshold, so **material** US institutional ownership is largely observed; non-US coverage varies by jurisdiction. Absence is treated as zero **by assumption, not by observation**.
- **MISSING bucket.** Firms with no Revere coverage at t get CN = NULL and are bucketed MISSING, reported separately from HIGH/LOW; "no data" is never silently read as "zero exposure." These rows drop out of the estimation panel via the cn_lag-non-null filter.
- **Extensive-margin gap (descriptive only).** On HIGH-CN firm-quarters, US zero-fill rate = 5,733/17,027 = **33.67%**, NONUS = 3,815/17,027 = **22.41%**, gap **+11.26 pp** in the H2.1 direction. Pooled over the whole window; cannot separate a geopolitical channel from time-invariant US-vs-NONUS portfolio differences. Not causal.
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

### 6.4 firm × group FE option (μ_{i,g})

The three pairwise FEs among {firm, group, quarter} are firm×quarter (have it), group×quarter (have it), and **firm×group (μ_{i,g})**. Adding μ_{i,g} absorbs any time-invariant US-vs-NONUS tilt toward each firm; β₂, β₃ then identify off within-(firm,group) time variation. Run as robustness (§7.2, §7.4). (A separate, heavier option — **holder × quarter** FE at the individual-institution level, the true Khwaja–Mian bank×time analog — is not yet built; it needs disaggregating from 2 groups to thousands of holders and conditioning on engagement.)

---

## 7. Estimation results — everything, honestly (coefficients ×10⁻⁶ of portfolio weight; two-way clustered SE in parentheses)

### 7.1 Main panel — three FE specifications (`07d_three_spec_table.do`, N = 462,564)

| | (1) No FE | (2) it+gt (α_{i,t}+γ_{g,t}) | (3) Weak FE (firm+quarter) |
|---|---|---|---|
| β₂ (US·CN_{t−1}) | −2.76 (3.72) | +1.98 (5.29) | −0.67 (3.63) |
| β₃ (US·CN·S_t) | −0.35 (0.60) | **+1.28 (1.66)** | −0.12 (0.71) |
| p(β₃) | 0.565 | **0.443** | 0.865 |
| R² | ~0.000 | 0.6215 | 0.0058 |
| F(2,81) | 0.81 (p=0.449) | 0.41 (p=0.668) | 0.04 (p=0.960) |

**Every coefficient is null (p ≥ 0.44).** **The current headline FE is the three-way pairwise (it+gt+ig; §7.2 and the "Second external review round" section); column (2) here is the it+gt comparison, β₃ = +1.28.** The headline β₃ (3-pairwise) is +1.80 (p=0.35), positive (opposite the H2.1 prediction) but indistinguishable from zero; the null is from a large SE, not a tight zero. (Also run: "β₃-only" spec = +1.37 (1.63), p=0.405; "full triple" spec identical to Headline with lower-order terms auto-omitted — confirms the absorption logic.)

**Formal MDE (w-based headline).** With SE(β₃)=1.91×10⁻⁶ (3-pairwise) and σ_S=2.578, the 80%-power MDE for a representative firm-quarter (CN∈[0.05,0.15], 1σ shock) is ≈ 0.7–2.1×10⁻⁶ in Δw units = **0.09–0.27% of σ(Δw)=7.6×10⁻⁴**. The design rules out within-Europe reallocations larger than ~0.2% of a typical quarterly weight change; it cannot rule out smaller ones. (Both outcome families — w and shares-flow — now carry an explicit MDE.)

### 7.2 + firm × group FE (`07e_firmgroup_tail.do`)

| | Headline | + firm×group FE |
|---|---|---|
| β₃ | +1.28 (1.66), p=0.443 | +1.80 (1.91), p=0.350 |
| N | 462,564 | 462,096 (468 singletons dropped; 15,388 firm-group cells) |

Still null. Interpretation: the null is **not** an artifact of structural US-vs-NONUS firm sorting.

### 7.3 Tail-dummy shock menu (`07e_firmgroup_tail.do`) — with power diagnostic

σ_S over 82 quarters = **2.578** (verified independently on both the estimation-sample panel and the audit panel; a second-review-round "correction" to 2.42 was itself in error — it was computed over a wider, non-estimation-sample quarter range — and is reverted here, R3 shock-lag work, 2026-07-02). One-sided right (escalation):

| k | **treated quarters** | β₃^tail | SE | p |
|---|---|---|---|---|
| 1.645 | 6 / 82 | +1.07 | 15.7 | 0.946 |
| 2 | 4 / 82 | +14.0 | 18.0 | 0.439 |
| 3 | 2 / 82 | +8.59 | 39.0 | 0.826 |

All null; SE explodes as k rises. Empirical distribution is **fat-tailed** (k=1.645 gives 6 quarters = 7.3%, not the 5% of a normal). **With only 6 / 4 / 2 treated quarters the CRVE + t(81) inference here is not merely low-powered but statistically *invalid* (MacKinnon–Webb 2017): few treated clusters bias the CRVE and break the t(G−1) reference, and the standard wild bootstrap fails in the same regime.** These tail rows are reported as *descriptive* robustness only; the correct inference is design-based randomization (Second review round, F3). The dummy coefficient is on a different scale from the continuous one — do not compare point estimates.

### 7.3b Shock timing and units (advisor request, 2026-06-28 meeting)

The advisor's meeting comment on the shock was three-part: (a) the shock should *also* be lagged, since CN exposure is already CN_{t-1} — measuring both regressors as of the start of the period over which Δw_t is measured, rather than mixing a lagged CN with a contemporaneous S_t; (b) report it in standard-deviation units; (c) a two-standard-deviation threshold, matching the capital-flow-episode convention (Forbes and Warnock 2012, 2σ main / 3σ robustness) and the GPR-spike convention (Caldara and Iacoviello 2022, AER, 2σ; their earlier IFDP 1222 draft used 1.68σ on the AR(1) residual — structurally the closest precedent to our S_t). Point (c) is already covered by the k=2 row in §7.3 above. Points (a)-(b):

**(a) Lagged shock S_{t-1}: Δw_t ~ US·CN_{t-1}·S_{t-1}**, vs the current headline Δw_t ~ US·CN_{t-1}·S_t.

| spec | FE | β₃ | SE | p (CRVE) | RI p |
|---|---|---|---|---|---|
| S_t (current headline) | it+gt | +1.28×10⁻⁶ | 1.66×10⁻⁶ | 0.443 | 0.617 |
| S_t (current headline) | 3-pairwise | +1.80×10⁻⁶ | 1.91×10⁻⁶ | 0.350 | 0.508 |
| **S_{t-1} (advisor spec)** | it+gt | **−8.65×10⁻⁷** | 2.08×10⁻⁶ | 0.679 | **0.735** |
| S_{t-1} (advisor spec) | 3-pairwise | −6.51×10⁻⁷ | 2.03×10⁻⁶ | 0.749 | — |

Under the advisor's timing the point estimate **flips sign** (now negative, i.e. nominally in the disengagement direction) but remains solidly null — if anything *less* distinguishable from zero than the contemporaneous spec (RI p=0.735 vs 0.617). Given randomization inference (design-based, CRVE-independent) shows both signs are equally consistent with the sharp null, **the sign flip is noise, not a finding**: β₃ is not stably signed across reasonable timing conventions, which is itself informative about how weak any signal is, not evidence of disengagement under the "correct" timing. `build_shocklag_panel.py`, `run_shocklag.do`, `run_ri_shocklag.py`.

**(b) SD-standardized reporting.** Since σ_S is a pure rescaling of the shock, standardizing changes only the coefficient's unit (interpretable as "effect per 1-SD tension shock"), not any t-statistic or p-value. At σ_S=2.578: the current headline β₃ (3-pairwise) is +4.64×10⁻⁶ per 1-SD shock; the advisor's S_{t-1} spec is −1.68×10⁻⁶ per 1-SD shock (3-pairwise). Both remain null under the same p-values reported above.

### 7.4 Conditional "spell-boundary" sample (advisor request) — including an error we made and fixed

**Design intent:** keep held quarters + one boundary zero at each entry/exit, conditional on firm-groups the group ever engaged; drop deep never-held zeros and never-held firms.

**Error (first build, `build_spell_boundary.py` / `07f`, SUPERSEDED).** Selection was done **per (firm, group)**. That broke the US-vs-NONUS pairing: many firm-quarters retained only one group → 45,672 singletons dropped by α_{i,t} → the estimate degenerated to "US's own change in selected firms," not "US relative to non-US." Result was β₃ = +4.38 (5.66), p=0.441, N=250,918. **Do not use.**

**Correction (`build_spell_riskset.py` / `07g`).** Define the risk set at the **firm-quarter** level: (i,t) enters if EITHER group has a spell-boundary there (held at t, t−1, or t+1); then keep **both** group rows. This preserves pairing.

Build output: 885,802 risk-set rows → **balanced 442,901 US / 442,901 NONUS**, 100% paired (0 unpaired). Estimation subset (drop missing dw/cn_lag/shock) = **342,262 rows, balanced 171,131 / 171,131, 6,355 firms, 0 group singletons.**

| | R1 headline (fq gq) | R2 + firm×group |
|---|---|---|
| β₃ | +2.46 (3.15), p=0.436 | +2.46 (3.19), p=0.443 |
| N | 342,262 (0 group singletons) | 341,858 (404 firm-group singletons) |

Still null, now cleanly identified within firm-quarter.

**Trade-off to disclose:** conditioning on engagement is closer to "real reallocation behavior," but because US engages fewer firm-quarters, the paired risk set discards deep zeros and the identifying variation narrows; the estimand is "within engaged firms" rather than "across all holdings-observed European issuers."

### 7.5 Country-pair subsample (`07b`, `07c`) — GB+DE+FR, replaces S_t with country-specific S_{c,t}

N = 227,310, 3,678 firms. Because S_{c,t} varies across listing country within (g,t), β₁ (US_g × S_{c,t}) becomes separately identified (3 coefficients instead of 2).

| coefficient | estimate |
|---|---|
| β₁ (US·S_{c,t}) — newly identified | **+17.7 (10.6)** — *not reliably inferred* (see below; not cited as evidence) |
| β₂ (US·CN) | +5.64 (6.92), p=0.417 |
| β₃ (US·CN·S_{c,t}) | +3.48 (30.2), p=0.909 |

The p=0.098 on β₁ under (firm, quarter) clustering is **an artifact of the wrong clustering level**: S_{c,t} varies only across 3 listing countries × quarter, so the honest cluster is the country. Clustering by `sec_country` (3 clusters, df=2) gives **p=0.227** (β₁ unchanged; SE barely moves but the degrees of freedom collapse), and with only 3 clusters no CRVE p is reliable here at all. β₁ is therefore **not evidence of anything** and is not cited as directional support (Second review round, F4).

---

### 7.6 Shares-based robustness: ownership share (`build_ownership_share_panel.py`, `build_ownership_share_c6_panel.py`, `run_ownership_share.do`)

**Why this exists.** The main outcome w = H(USD)/T(USD) is a market-value portfolio weight, so it mixes trading flow with price moves and portfolio-denominator reallocation. A null on w supports "US do not reduce their within-Europe portfolio *weight*", but not the stronger "US do not *sell* / do not reduce their *stake*". For the flow claim we use a pure quantity — the group's ownership share of each firm:

- ownership_share_{i,g,t} = ( Σ_{b∈g} shares held on the primary EQ class ) / (primary-EQ shares outstanding).

It is immune to price (numerator and denominator are both in shares) and to portfolio-denominator reallocation (no T term). Shares are not additive across firms, so a shares-based *portfolio weight* is meaningless; the ownership share is the correct object. `adj_shares_out` is a per-security-class attribute, so we restrict both numerator and denominator to the **primary EQ class** (`fsym_id = fsym_primary_id`, matching the market-cap rule in `04_us_ownership_european.jl`); this makes shares_out constant within each security-quarter (verified: dispersion drops from 5.28% of cells to exactly 0). The **primary outcome is the pure trading FLOW** `(held_t − held_{t−1}) / out_{t−1}` (fixed lagged float; Second review round, F6), on the **same C6 grid, three-way pairwise FE, and clustering** as the main spec. (An earlier draft used Δ(ownership_share) = held_t/out_t − held_{t−1}/out_{t−1} with each term over its *own* current float; that is not float-immune — a buyback/issuance moves it with zero trading — so it is retained only as a labelled comparison column.)

**Result (FLOW, raw fraction-of-float units — NOT ×10⁻⁶).** N = 300,866 (5,704 firms with a primary-EQ float, 82 quarters, balanced 150,433 / 150,433). Flow mean ≈ 0, sd ≈ 0.053; ownership level mean 6.3%, median 3.4%.

| coefficient | FLOW, it+gt (comparison) | **FLOW, it+gt+ig (headline)** | old Δ(ownership_share) |
|---|---|---|---|
| β₂ (US·CN_{t−1}) | −0.00048 (p=0.85) | −0.00139 (p=0.65) | −0.00108 (p=0.60) |
| β₃ (US·CN_{t−1}·S_t) | +0.00089 (0.00113), p=0.44 | **+0.00088 (0.00115), p=0.44** | +0.00060 (p=0.61) |

β₃ is null and, if anything, **positive** — opposite to the disengagement prediction (β₃<0). The shares-based test agrees with the w-based null: **US investors do not reduce their ownership stake in high-China-exposure European firms more than non-US investors when tension rises.**

**Not a zero-fill artifact.** Dropping the extensive-margin zeros (observed-held-only, re-paired, N = 231,860) gives β₃ = **+0.00122** — *further* from the disengagement prediction, not toward it. The null is therefore not zero-fill attenuation. (Verified by re-running the triple difference on the held-only subset.)

**Independently reproduced.** β₃ = +0.0005988 via a Python two-way within-transformation, matching the Stata `reghdfe` value to ~2×10⁻⁷; three independent adversarial re-derivations agree on the point estimate, the null, and the construction.

**Caveat 1 (ADR / non-primary exclusion).** The measure is primary-EQ only, so it does not observe stake adjustment through ADR/GDR or non-primary classes. US investors hold **8.92%** of their European exposure off the primary class versus **3.32%** for non-US (2.69× asymmetric). The shares-based null is therefore *complementary* to the USD portfolio-weight main spec, which does capture the ADR channel — not a substitute. An ADR-inclusive ownership measure needs the ADR conversion ratio and is deferred (§10).

**Caveat 2 (power / MDE).** With 82 quarter-clusters the design rules out *large* stake reductions but has limited power against small ones. The 80%-power MDE for a *representative* firm-quarter (CN∈[0.05,0.15], 1σ shock, σ_S=2.578) is **≈ 4.1–12.4 bps of float** (≈8.3 bps at CN=0.10), i.e. 0.7–2.2% of the outcome sd — an order of magnitude below the ~25–35 bps that 2.8·SE(β₃) implies at the non-existent CN·S=1 point (Second review round, F5). The quarter-end-only shock is classical measurement error that attenuates β₃ and *enlarges* the true MDE. Read the estimate as bounding the effect near zero, not as proving an exact zero.

---

## 8. The centered-vs-backward disclosure (retraction of the earlier "result")

| Δw definition | β₃ | SE | p |
|---|---|---|---|
| Centered (w_{t+1} − w_{t−1}), prior build | +1.38 | 0.54 | **0.013 (significant, opposite H2.1)** |
| Backward (w_t − w_{t−1}), current | +1.28 | 1.66 | 0.443 (null) |

Point estimate barely moves (+1.38 → +1.28); **SE roughly triples (0.54 → 1.66)**. We read the earlier significance as a **look-ahead / post-treatment window contamination**: the centered LHS includes post-t holdings (w_{t+1}), so the outcome mixes contemporaneous with future adjustment and is not aligned with the estimand's timing. We deliberately do **not** claim a proven "mechanical correlation of shocks" (S_t is an AR(1) residual, so S_t and S_{t+1} need not be strongly correlated), and we do **not** claim to have decomposed *why* the SE tripled. Three forces move together between the two builds — look-ahead removal, a small right-edge sample expansion, and entry/exit reweighting — and they are not separately identified here; a sample-matched centered-window comparison on a common sample (deferred, §10) is needed to attribute the SE change. Identification rests on the within-firm-quarter β₃, which is null. **The retraction is now on firmer ground:** the clean lead-flow test (Δw_{t+1} ~ S_t) and the local-projection IRF are both null under design-based randomization inference (Second review round, F1), so the null is not an artifact of the centered window's timing — no US-differential response exists at t, at t+1, or cumulatively.

---

## 9. Limitations and threats (exhaustive)

1. **Null could be power, not absence.** 82 quarters; effective time variation is small; two-way cluster leaves 82 quarter-clusters. We cannot reject H2.1 or its opposite.
2. **Zero-fill assumption** (§5.5): absence coded as zero by assumption; non-US coverage uneven.
3. **Backward-diff timing** removes t+1 leakage but not within-quarter anticipation (investors trading in months 1–2 of quarter t on early news that the quarter-end residual misses).
4. **Shock uses only the quarter-end month** (~1/3 of intra-quarter GPR news).
5. **Entry/exit mechanical asymmetry** under backward diff (exit = −w_{t−1}, entry = +w_t) mechanically favors the H2.1 direction, yet β₃ is null — arguably reassuring, but it complicates magnitude interpretation.
6. **Estimand shifts** across samples (full grid = all holdings-observed European issuers; risk-set = engaged firms only).
7. **"European" includes GB/CH/NO** (non-EU); GB dominates the country-pair subsample.
8. **HIGH cutoff is ~4%**, so "high exposure" is not extreme.
9. **Group-level (2 groups), not holder-level.** The true Khwaja–Mian holder×time control is not yet in.
10. **Convention-match not verified componentwise** against De Haas (do their diff and shock-timing both match ours?).
11. **Universe is holdings-observed, not full listing** (§3, Caveat 2). The C6 zero-fill corrects selection on the outcome *within* the holdings-observed universe (a firm held by anyone gets a full grid, so US exits are captured), but not securities never held by any institution. Adding those from full Security Coverage would, for firms with non-missing CN, contribute zero within-firm-quarter outcome variation against nonzero regressor variation, whose **mechanical expectation is to attenuate β₃ toward zero**. On that reasoning the current holdings-observed universe is, if anything, the *higher-powered* universe for detecting β₃, and the full-universe rebuild is an optional robustness. But this is the expected direction under a simplified argument, not a proven one — in the full FE model the added rows also shift the group×quarter effects, so the net movement of β₃ should be **verified by actually running the rebuild, not asserted**. (This corrects an earlier claim that never-held firms "do not affect β₃" — they do, and the leading-order effect is attenuation.)
12. **Revere match has country structure.** Of the holdings-observed European securities, ~20% do not match to Revere overall, but the miss rate is uneven by listing country (`05_unmatched_profile_by_country.csv`). **GB is the worst among large countries: 32.78% unmatched (1,307 of 3,987)**, versus FR 17.9%, DE 11.6%. Because GB is 60.85% of the GB+DE+FR country-pair subsample, the Revere-matched sample systematically drops about one-third of GB issuers — a selection concern for both the main and country-pair specifications that should be characterized (is the unmatched set systematically smaller / different-sector?), not just reported.
13. **Shares-based test excludes ADRs** (§7.6, Caveat 1). The ownership-share robustness is primary-EQ only, so it cannot see stake adjustment through ADR/GDR or non-primary classes. US holds 8.92% of its European exposure off the primary class versus 3.32% for non-US (2.69× asymmetric), so the shares-based null is complementary to — not a substitute for — the USD portfolio-weight main spec that does capture the ADR channel. Its power is limited against small adjustments; the 80%-power MDE for a representative firm-quarter is ≈ 4–12 bps of float (Second review round, F5), so it bounds the stake effect near zero rather than proving an exact zero.
14. **Group is FactSet registration domicile, not decision nationality** (third review round). A US manager's Luxembourg / Ireland UCITS shell is grouped into NONUS, though the allocation decision and any political-pressure channel are US — treatment-group misclassification that mechanically **compresses β₃ toward zero**. Not yet corrected; the root fix is to regroup by the management company's **ultimate-parent nationality** (deferred, §10). The EU-domiciled-only control treats the symptom, not the cause.
15. **NONUS is not an untreated control.** European institutions have their own domestic de-risking pressure (from ~2019), so the triple difference identifies only *differential* US disengagement and is **blind to common de-risking**: a zero β₃ is consistent with both groups reallocating identically. Consistent with this, the weak-FE column (which would pick up any *common* reaction) is also null — at the quarterly frequency no group reallocates within Europe on the shock.
16. **The continuous AR(1) shock averages over opposite-signed episodes.** 2018 tariffs (bad for exposed firms) vs the 2022 export controls (partly *good* for EU semicap substitutes) may push allocations opposite ways; one continuous shock aggregates them toward zero. A named-event study (H2.2) is the cleaner test.

---

## 10. Deferred robustness (planned, NOT done — do not present as completed)

- Continuing-positions-only subsample (w_{t−1}>0 AND w_t>0) to neutralize entry/exit mechanical asymmetry.
- Winsorize Δw at 1/99%.
- Leave-one-quarter-out (which escalation episodes carry any result).
- PPML via `ppmlhdfe` for the zero-heavy weights (Silva–Tenreyro 2006; Correia et al. 2020).
- ~~Local projections (Jordà 2005) for dynamics~~ — **DONE** (Second review round, F1; all horizons null under randomization inference).
- **Holder-level fixed effects** (holder × quarter), the true bank×time analog.
- **ADR-inclusive ownership share** (§7.6): map ADR/GDR holdings to underlying-share equivalents via the ADR conversion ratio so the shares-based test captures the ADR channel (US 8.92% of exposure off-primary); needs the ratio data.
- Sample-matched centered-window comparison (quantify the §8 disclosure on a common sample).
- OFAC SDN / Entity-List exclusion; size, industry, listing-country × shock controls.
- **Positive controls** (make the null informative): (i) **Russia exposure** post-Ukraine (2022) — isomorphic reuse of 02→06→RI with `'RU'` and a `USA|Russia` GPR shock; (ii) the same investors' **direct China holdings** (ADR/HK, HFCAA-era divestment) as an in-pipeline benchmark. If the design detects those known divestments, the China null becomes "truly absent," not "undetectable."
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
| `00_setup.jl` | EU country list, paths, DB helper | — |
| `02_china_exposure.jl` | CN exposure (symmetric, bilateral, time-versioned; C2/C4/C5) | `firm_quarter_china_exposure.parquet` |
| `06_cartesian_grid.jl` | Cartesian grid + zero-fill + **backward Δw** + lagged CN (C1/C6) | `merged_us_eu_zero_filled.parquet` |
| `build_c6_panel.py` | → main estimation panel | `c6_panel.dta` (462,564) |
| `build_country_pair_shock.py` | AR(1) per country-pair (UK/DE/FR) + subsample | `c6_panel_country_pair.dta` |
| `build_spell_riskset.py` | **correct** firm-quarter risk-set conditional sample | `c6_panel_riskset.dta` (342,262) |
| ~~`build_spell_boundary.py`~~ | **SUPERSEDED — per-group selection, broke pairing** | ~~`c6_panel_spell.dta`~~ |
| `07_regression.do` | 4 specs (β₂+β₃ / β₃-only / full triple / weak FE) | `07_reghdfe_results.txt` |
| `07b_country_pair_robustness.do` | country-pair S_{c,t}, recovers β₁ | `07b_country_pair_results.*` |
| `07c_strict_4term.do` | explicit 4-term form, main vs country-pair | `07c_strict_4term_results.*` |
| `07d_three_spec_table.do` | No-FE / Headline / Weak-FE headline table | `07d_three_spec_results.*` |
| `07e_firmgroup_tail.do` | firm×group FE + tail-dummy menu (k=1.645/2/3) | `07e_firmgroup_tail.*` |
| `07g_spell_riskset.do` | risk-set conditional sample regression | `07g_spell_riskset.*` |
| ~~`07f_spell_boundary.do`~~ | **SUPERSEDED (ran the biased per-group sample)** | — |
| `run_headline_3pairwise.do` | **headline** = three-way pairwise FE, all main specs | `audit_c6_panel.dta` |
| `run_audit_f1f2f7.do` / `run_audit_f4f8.do` | second-round tests (F1/F2/F7, F4/F8) | audit CSVs |
| `run_randomization_inference.py` / `run_ri_3pairwise.py` | design-based RI (F1/F3) | RI CSVs |
| `build_shocklag_panel.py` / `run_shocklag.do` / `run_ri_shocklag.py` | advisor shock-timing revision (§7.3b): S_{t-1} spec + RI | `shocklag_panel.dta`, `shocklag_ri_results.csv` |
| `02_russia_exposure.jl` / `06_russia_grid.jl` / `build_russia_shock.py` / `build_russia_c6_panel.py` | Russia positive control (isomorphic pipeline) | `c6_panel_russia.dta` |
| `run_russia_headline.do` / `run_ri_russia*.py` / `build_russia_lp_panel.py` / `run_russia_lp_test.py` | Russia positive-control regressions + inference (mixed evidence, pending re-review) | Russia result CSVs |
| `build_audit_panel_f1f2f7.py` / `build_riskset_lagonly.py` | second-round panels | audit / lag-only dta |

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

| Spec | Sample / shock | β₃ | SE | p | N |
|---|---|---|---|---|---|
| Headline | full grid, continuous S_t | +1.28 | 1.66 | 0.443 | 462,564 |
| No FE | full grid | −0.35 | 0.60 | 0.565 | 462,564 |
| Weak FE (firm+quarter) | full grid | −0.12 | 0.71 | 0.865 | 462,564 |
| + firm×group FE (= 3-pairwise headline; see row below) | full grid | +1.80 | 1.91 | 0.350 | 462,096 |
| Tail k=1.645 (6 qtrs) | full grid, dummy | +1.07 | 15.7 | 0.946 | 462,564 |
| Tail k=2 (4 qtrs) | full grid, dummy | +14.0 | 18.0 | 0.439 | 462,564 |
| Tail k=3 (2 qtrs) | full grid, dummy | +8.59 | 39.0 | 0.826 | 462,564 |
| Conditional risk-set | paired, continuous S_t | +2.46 | 3.15 | 0.436 | 342,262 |
| Conditional + firm×group | paired | +2.46 | 3.19 | 0.443 | 341,858 |
| Country-pair (GB+DE+FR) | S_{c,t} | +3.48 | 30.2 | 0.909 | 227,310 |
| **Headline (3-pairwise it+gt+ig)** | full grid, S_t | +1.80 | 1.91 | 0.350 (RI 0.51) | 462,096 |
| Lead-flow Δw_{t+1} ~ S_t | full grid | −0.24 | 1.98 | 0.905 (RI 0.92) | 449,938 |
| Local projection cum h1 | full grid | +1.38 | 0.54 | CRVE 0.013 / **RI 0.47** | 449,938 |
| Local projection cum h4 | full grid | +3.88 | 1.89 | CRVE 0.044 / **RI 0.23** | 414,198 |
| In-span (drop ~25% phantom) | in-span, S_t | +2.38 | 3.01 | 0.430 (RI 0.58) | 346,764 |
| Ownership FLOW (F6)† | flow, S_t | +0.00088 | 0.00115 | 0.44 | 300,866 |
| GPR two-interaction (F7) | GPR_t / GPR_{t−1} | −2.07 / +0.89 | — | 0.28 / 0.64 | 462,564 |
| **[retracted] centered diff** | full grid | +1.38 | 0.54 | 0.013 | ~450k |
| **[retracted, biased] per-group spell** | broke pairing | +4.38 | 5.66 | 0.441 | 250,918 |

†Ownership FLOW β₃ is in raw fraction-of-float units, not ×10⁻⁶. RI = design-based randomization-inference p (Second review round, F1/F3).

**Bottom line for the reviewer:** every valid specification returns a null β₃, including under the three-way pairwise FE headline and under design-based randomization inference (which shows the CRVE-significant local-projection horizons to be few-cluster/overlapping-window artifacts; every RI p ≥ 0.07, every single-quarter spec ≥ 0.23, every point estimate *positive*). The historically significant result is a retracted forward-window/few-cluster artifact; the one borderline coefficient (country-pair β₁) is not reliably inferred once clustered at the country level (p 0.098→0.23). The honest reading is that the ownership-flow channel shows no detectable differential US disengagement, and the project should be judged on (a) the panel/identification construction and (b) the planned firm-level price channel (H2.2), not on a confirmed behavioral effect.

---

## 14. Code inventory (all files live in the repo — open them on GitHub)

The full source is in `github.com/fidio728/practice`, branch `essay2-code-review`, under `julia_descriptive/`. This document no longer embeds the code verbatim (that risked doc-vs-code drift); instead it maps each file to what it does. Run order for a rebuild: Julia `00`→`06`, then the Python panel builds, then the Stata do-files.

### Julia data-construction pipeline (run order 00→06)
| file | builds / does |
|---|---|
| `00_setup.jl` | config: the 28 `EU_COUNTRIES`, data paths, helpers |
| `01_master_files.jl` | master-file assembly |
| `02_china_exposure.jl` | Chinese supply-chain exposure from Revere edges; edge counterparty country **as of edge-start** (F12); `china_share` = CN edges / total edges |
| `03_eom_etl.jl` (+ `03a_decompress_to_parquet.jl`, `03b_phase_c_diagnostics.jl`) | FactSet Ownership EOM ETL → `holdings_eom.parquet` (186,800,295 rows); duplicate-key **hard-fail** (F13/dup-key) |
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
| `run_russia_headline.do` | Russia positive control: continuous-shock (it+gt, 3-pairwise) + 2022Q1-Q2 event dummy |

### Design-based inference — Russia extension
| file | does |
|---|---|
| `run_ri_shocklag.py` | RI for the S_{t-1} spec (§7.3b) — confirms sign flip is noise (RI p=0.735) |
| `run_ri_russia.py` | Russia continuous-shock RI (it+gt) + 2022Q1-Q2 vs all rolling 2-quarter windows placebo |
| `run_ri_russia_3pairwise.py` | Russia 3-pairwise RI |
| `run_russia_lp_test.py` | Russia cumulative event-study, firm-level permutation test per horizon (h=0..7) — **mixed evidence, pending adversarial re-review (workflow wy0r0wgv2 failed on a session-limit error, not yet re-run)** |

### Design-based inference (Python)
| file | does |
|---|---|
| `run_randomization_inference.py` | exact RI: US−NONUS pairwise-difference collapse reproduces the two-way-FE β₃; permutes the 82 quarter shocks → `audit_randomization_inference.csv` |
| `run_ri_3pairwise.py` | two-way-FE (firm+quarter) RI for the 3-pairwise LP horizons → `audit_ri_3pairwise.csv` |

### Plots
`plots/fig13_two_panel.py`, `plots/plot_all.jl`, `plots/plots_python.py`.

---

## 15. Result → source map (which file produces each headline number)

| result | value | produced by | reads / writes |
|---|---|---|---|
| Headline β₃ (w-based, it+gt+ig) | +1.80×10⁻⁶ (p=0.35) | `run_headline_3pairwise.do` | `audit_c6_panel.dta` |
| Headline β₃ (w-based, it+gt) | +1.28×10⁻⁶ (p=0.44) | `run_headline_3pairwise.do` / `build_c6_panel.py` | `audit_c6_panel.dta` / `c6_panel.dta` |
| Ownership FLOW β₃ (F6) | +8.8×10⁻⁴ (p=0.44) | `run_ownership_share.do` | `ownership_c6_panel.dta` → `ownership_share_results.csv` |
| F1 lead-flow / local projection | all null | `run_audit_f1f2f7.do` | `audit_c6_panel.dta` → `audit_f1f2f7_results.csv`, `audit_f1_lp_irf.dta` |
| F1/F3 randomization-inference p (it+gt) | headline 0.62, lead 0.92, LP h1 0.47, h4 0.23 | `run_randomization_inference.py` | `audit_c6_panel.parquet` → `audit_randomization_inference.csv` |
| randomization-inference p (3-pairwise) | headline 0.51, LP cum1 0.23, cum4 0.07 | `run_ri_3pairwise.py` | `audit_c6_panel.parquet` → `audit_ri_3pairwise.csv` |
| F2 in-span headline | +2.4×10⁻⁶ (p=0.42) | `run_audit_f1f2f7.do` (`in_span==1`) | `audit_c6_panel.dta` |
| F4 country-pair β₁ clustering | p 0.098 → 0.23 | `run_audit_f4f8.do` | `c6_panel_country_pair.dta` |
| F7 GPR two-interaction | both null | `run_audit_f1f2f7.do` | `audit_c6_panel.dta` |
| F8 risk-set lag-only β₃ | +2.6×10⁻⁶ (p=0.44) | `run_audit_f4f8.do` | `c6_panel_riskset_lagonly.dta` |
| ADR off-primary share | US 8.92% / NONUS 3.32% | `build_ownership_share_panel.py` | `ownership_share_diagnostics.csv` |
| Revere match cascade | 12,743→10,171→8,014→7,928 | `05_combine_visualize.jl` / `build_c6_panel.py` | `05_unmatched_profile_by_country.csv` |
| §7.3b shock-timing S_{t-1} | −8.65×10⁻⁷ (CRVE p=0.68, RI p=0.735) | `run_shocklag.do` / `run_ri_shocklag.py` | `shocklag_panel.dta` |
| σ_S (corrected) | 2.578 over 82 estimation quarters | `run_shocklag.do` | `shocklag_panel.dta` |
| Russia positive control | mixed/fragile; **pending adversarial re-review** | `run_russia_headline.do` / `run_ri_russia*.py` | `c6_panel_russia.dta` |
