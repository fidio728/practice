# Essay 2 — Full Methodology, Replication Log, and Honest Caveats

**Purpose.** This document is a complete, deliberately unflattering technical record of Essay 2, written to be handed to an independent reviewer (human or AI). It states the research design, every variable and how it is built, the exact code and commands, the data-cleaning decisions, all estimation results, and every caveat, limitation, and mistake we found and corrected. Nothing is hidden. Where the design is weak, underpowered, or where an earlier build was wrong, it is flagged explicitly.

**One-line status.** The headline coefficient is **statistically indistinguishable from zero** under the preferred backward-timing specification with the fully-saturated **three-way pairwise FE** (firm×quarter + group×quarter + firm×group; see "Second external review round"), with standard errors large enough that economically meaningful effects of either sign cannot be ruled out; the null is robust across fixed-effect choices, shock definitions, sample constructions, a firm-existence-span restriction, a raw-GPR-level shock, and — decisively — **design-based randomization inference** (which shows the only CRVE-significant coefficients, at cumulative local-projection horizons, to be few-cluster/overlapping-window artifacts; every randomization-inference p ≥ 0.17, and every point estimate is *positive*, i.e. opposite to disengagement). It also holds on a **shares-based ownership measure** (§7.6): US investors do not reduce their ownership *stake* (shares held as a fraction of the primary-EQ float) any more than non-US investors when tension rises (β₃ = +0.000599, p=0.61, positive and insignificant), so the null is not an artifact of value-weighting — subject to an ADR-exclusion caveat. An earlier significant estimate came from a forward-window specification vulnerable to post-treatment timing contamination, and is retracted. The project's current contribution is a measurement/design one (the panel + identification), and the empirical question is treated as open, with a planned pivot to a firm-level price channel.

---

## Change log for this review round (read first)

This document was hardened over one working session in response to two external AI reviews plus several rounds of internal adversarial multi-agent review. Everything below is already reflected in the body (§1–§16) and in the on-disk code embedded verbatim in §14–§16. This section states **what changed and why**, so a new reviewer can see the full provenance. No data was fabricated or altered; every number cited was re-derived from the on-disk parquet/dta/csv files.

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

The market-value portfolio weight `w = H(USD)/T(USD)` mixes trading flow with price moves and portfolio-denominator reallocation, so a null on `w` supports "US do not reduce their portfolio *weight*" but not "US do not *sell* / do not reduce their *stake*." To license the flow claim we built an **ownership-share** outcome (§7.6): group shares held on the primary EQ class / primary-EQ shares outstanding, price- and denominator-immune, on the same C6 grid, FE, and clustering as the main spec. **Result: β₃ = +0.000599 (SE 0.001165, p=0.609), null and if anything positive** — the shares-based test agrees with the w-based null. Robust to dropping the extensive-margin zeros (observed-only β₃ = +0.00122). Two required caveats carried: ADR exclusion (US holds 8.92% of European exposure off the primary class vs 3.32% for non-US) and limited power against small (~25–35 bps) adjustments. New files: `build_ownership_share_panel.py`, `build_ownership_share_c6_panel.py`, `run_ownership_share.do` (all in §14–§15).

### C. Adversarial multi-agent reviews run this round (all findings resolved)

| review id | scope | verdict |
|---|---|---|
| `wbo5chyw9` | the code fixes in A6–A9 | ready, 0 defects |
| `wutg4fnkw` | the weight-sum / quarter-coverage asserts, artifact map, doc wording | ready, 0 must-fix |
| `wqtl83l3b` | doc-scoped: every quantitative claim re-derived vs code/data; embedded code vs disk | ready, provenance verified, 0 must-fix |
| `wsu1zupkc` | the shares-based build | **needs_fix → caught a real bug** (first draft pooled all `fsym_id` classes and MODE-masked a dual-universe `shares_out`, 5.28% of cells dispersed up to 1e12×). Fixed by restricting to `fsym_id = fsym_primary_id`, which drives dispersion to exactly 0. |
| `w86zvcvsx` | the shares-based result | result_trustworthy, 0 must-fix; β₃ independently reproduced to ~2×10⁻⁷; null confirmed not a zero-fill artifact |

### D. Current conclusions (detail in §7, §8)

The headline β₃ (US × China-exposure × tension shock on within-Europe reallocation) is a **robust statistical null** — across FE choices, shock definitions, sample constructions, the value-weighted portfolio weight, **and** the shares-based ownership stake. If anything, the few non-null coefficients point *opposite* to the disengagement hypothesis (country-pair β₁ = +17.7, p=0.098; shares-based β₃ = +0.000599 positive). The earlier "significant" result was a retracted forward-window artifact. The project's contribution is currently a measurement/identification one (the paired firm×quarter / group×quarter panel), and the ownership-flow question is treated as answered-in-the-null, with the economic action (if any) to be sought in a firm-level price channel (H2.2).

### E. Next steps (detail in §10)

In priority order: **(#7) inference robustness** — wild-cluster bootstrap (important with only 82 quarter-clusters), leave-one-quarter-out, and a shock-timing menu (quarterly sum/avg vs quarter-end); **(#6) exposure robustness** — `rel_type` filter, edge dedup, quarter-as-of counterparty country; **(#8) holder-level panel** — holder × quarter FE, the true Khwaja–Mian bank×time analog; **(H2.2) price channel** — acquire European returns and test abnormal returns / valuation directly; **ADR-inclusive ownership** — map ADR holdings to underlying-share equivalents (needs the ADR conversion ratio) to close the §7.6 Caveat 1 gap.

---

## Second external review round (F1–F13) — responses, new tests, and the three-way pairwise FE headline

A second (cloud-based) AI review raised 13 findings — F1 critical, F2–F7 major, F8–F13 minor — all touching either the identification/timing of the headline or the reliability of the inference. Every one was verified against the on-disk data, addressed with new code (all embedded in §14–§16), and re-run. **The disengagement null (β₃ < 0) survives every fix, and several apparent significances were shown to be clustering artifacts.** All new numbers were re-derived from the parquet/dta files; nothing was fabricated. Two headline changes follow directly.

### Headline change 1 — three-way pairwise FE (Khwaja–Mian / De Haas saturation)

The headline is now the **fully-saturated three-way pairwise FE**: firm×quarter (`it`, α_{i,t}) + group×quarter (`gt`, γ_{g,t}) + **firm×group (`ig`, μ_{i,g})**; the earlier `it+gt` is kept as a comparison column. On the balanced 2-per-firm-quarter panel, `gt` absorbs β₀(US) and β₁(US·S); `it` absorbs the CN/S/CN·S levels; so only β₂(US·CN) and β₃(US·CN·S) are identified. On the differenced outcome, `ig` is a firm-specific drift control on the US−NONUS difference (via the pairwise collapse, the 3-pairwise = firm FE + quarter FE on Δy). The null surviving the fully-saturated FE is a **stronger** null claim. (The true Khwaja–Mian holder×quarter saturation needs the holder-level panel — deferred, §10.)

| specification | 3-pairwise (it+gt+ig) β₃ | it+gt β₃ | verdict |
|---|---|---|---|
| headline Δw ~ US·CN·S_t | +1.80×10⁻⁶ (p=0.35) | +1.28×10⁻⁶ (p=0.44) | **null** |
| F1a lead-flow Δw_{t+1} ~ S_t | +1.9×10⁻⁷ (p=0.92) | −2.4×10⁻⁷ (p=0.90) | **null** |
| F2 headline, in-span only | +2.5×10⁻⁶ (p=0.42) | +2.4×10⁻⁶ (p=0.43) | **null** |
| F7 US·CN·GPR_t / GPR_{t−1} | −2.0×10⁻⁶ (0.34) / +7×10⁻⁷ (0.74) | −2.1×10⁻⁶ (0.28) / +9×10⁻⁷ (0.64) | **null** |
| ownership FLOW (F6) | +8.8×10⁻⁴ (p=0.44) | +8.9×10⁻⁴ (p=0.44) | **null** |
| F1b LP cum h1 (CRVE only) | +2.4×10⁻⁶ (p=0.03) | +1.4×10⁻⁶ (p=0.01) | **CRVE artifact → null under RI, see F1/F3** |
| F1b LP cum h4 (CRVE only) | +6.6×10⁻⁶ (p=0.08) | +3.9×10⁻⁶ (p=0.04) | **CRVE artifact → null under RI, see F1/F3** |

Every point estimate is **positive** — the opposite sign to the disengagement prediction (β₃ < 0) — so even the CRVE-significant cumulative horizons, if believed, would *strengthen* the no-disengagement reading, not overturn it.

### Headline change 2 — β₁ removed from directional evidence (F4)

The country-pair β₁ (US·S_{c,t}, formerly "+17.7, p=0.098, opposite H2.1") is **removed from all directional-evidence sentences** (§0 index, §7.5, §13). S_{c,t} varies only across 3 listing countries × quarter, so the honest clustering level is the country: clustering by `sec_country` (3 clusters, df=2) moves β₁ from p=0.098 to **p=0.227** (β₁ unchanged at +1.77×10⁻⁵; the SE barely moves but the degrees of freedom collapse). It is not reliable evidence of anything.

### Finding-by-finding

- **F1 (critical — timing / retraction logic).** The concern: S_t is the quarter-*end* AR(1) residual while backward Δw_t spans the whole quarter, so the preferred spec allows only ~1 month of reaction, and the natural t+1-quarter response window was never tested; the SE tripling in the centered→backward retraction is not the signature of "removing look-ahead noise." Resolution: (i) the **lead-flow** Δw_{i,g,t+1} ~ US·CN·S_t (pure lagged shock, no look-ahead) is **null** (p=0.92); (ii) a **local-projection IRF** of the cumulative response w_{t+h}−w_{t−1} on US·CN·S_t (h=0..4) has positive point estimates that reach CRVE p<0.05 at h=1 and h=4 — **but these are overlapping-window / few-cluster CRVE artifacts**: under design-based **randomization inference** (permuting the 82 quarter shocks; §F3) every horizon is null (headline 0.62, lead 0.92, LP h1 **0.47**, h2 0.25, h3 0.32, h4 **0.23**). So there is no US-differential response at t, at t+1, or cumulatively; the "reaction is in t+1" alternative is rejected. `run_audit_f1f2f7.do`, `run_randomization_inference.py`.
- **F2 (Cartesian grid vs firm existence span).** `06_cartesian_grid.jl` crosses the universe with ALL quarters 1999Q1–2023Q4 without intersecting each firm's own existence span, so pre-IPO / post-delisting structural Δw=0 rows enter the estimation panel. Measured: **25.03%** of the panel is out-of-span, ~96.6% of it exactly Δw=0. These attenuate β₃, inflate N, and understate SE. Re-running the headline on the in-span subset only: β₃ ≈ 1.86× larger (+1.28→+2.4×10⁻⁶) but still **null** (RI p=0.58). The conclusion is unchanged; the headline N / SE / §5.5 extensive-margin figures are affected. `build_audit_panel_f1f2f7.py` (`in_span`).
- **F3 / F9 (inference — few clusters, overlapping windows).** The tail-dummy specs (§7.3, 6/4/2 treated quarters) and the overlapping-window LP have unreliable CRVE (MacKinnon–Webb; `reghdfe` flagged a non-positive-semi-definite VCV on the LP), and the previously-planned wild cluster bootstrap fails with few treated clusters. **Randomization inference** is the correct design-based test and is now the arbiter for these specs. Its exact algebra (US−NONUS pairwise difference + quarter FE reproduces the two-way-FE β₃; per-quarter sufficient statistics make a permutation an O(82) weighted sum) was independently verified to reproduce reghdfe's β₃ to 7 significant figures. `run_randomization_inference.py`. The tail-dummy k=2/3 specs (2 and 4 treated quarters) are **inference-invalid** and reported as such / dropped, not as "low power."
- **F4 (country-pair β₁ clustering).** See Headline change 2 above. `run_audit_f4f8.do`.
- **F5 (MDE units).** The §7.6 "25–35 bps" MDE conflated the per-unit-CN·S coefficient scale with the outcome scale. Corrected: with SE(β₃_flow)=1.13×10⁻³ and σ(S_t)=2.42, the 80%-power MDE for a *representative* firm-quarter (CN∈[0.05,0.15], 1σ shock) is **≈ 4–12 bps of float** (0.7–2.2% of the outcome sd), not 25–35 bps; the quarter-end-only shock is classical measurement error that attenuates β₃ and enlarges the true MDE. §7.6 Caveat 2 corrected.
- **F6 (ownership flow denominator).** The §7.6 dos = ownership_share_t − ownership_share_{t−1} was NOT denominator-immune: with each term over its own *current* float, a buyback/issuance moves it with zero trading, by a term ∝ the group's own lagged level (differs across US/NONUS, so not absorbed by firm×quarter FE). Fixed: the primary outcome is now the pure flow **(held_t − held_{t−1}) / out_{t−1}** (fixed lagged float). β₃ = +8.8×10⁻⁴ (p=0.44), **null** — same conclusion, but the "net buying/selling" language is now literally correct. The old dos is retained as a labelled comparison column. `build_ownership_share_c6_panel.py`.
- **F7 (generated regressor / full-sample AR(1)).** The AR(1) shock is estimated once on the full ~1957–2023 monthly series (its (a,b) embed future data) and is a generated regressor. Robustness: replacing the composite shock with **US·CN·GPR_t + US·CN·GPR_{t−1}** (raw GPR level + lag, which nests every (a,b) and removes both the generated-regressor and the full-sample look-ahead) gives both interactions **null** (p=0.34 / 0.74). Disclosed in §9. `run_audit_f1f2f7.do`.
- **F8 (risk-set lead membership).** The main risk set (§7.4) conditions membership on t+1 holdings (a post-treatment variable). A **lag-only** variant (held at t or t−1, no look-ahead) gives β₃ = +2.6×10⁻⁶ (p=0.44), **null** — insensitive to the membership rule. `build_riskset_lagonly.py`.
- **F10 (multiple testing).** ≥19 coefficient tests were reported; the single p<0.10 (country-pair β₁) is the expected number of false positives at α=0.10 and was used asymmetrically as "direction opposite H2.1." That framing is removed; the narrative is a clean null. (See F4.)
- **F11 (ADR-share docstring).** `build_ownership_share_panel.py` reported 91.57%/96.99% on-primary (an EQ+AD calc); corrected to the EQ-primary measure this build uses, 91.08%/96.68% on-primary (US 8.92% / NONUS 3.32% off-primary), matching §7.6.
- **F12 (edge as-of classification).** `02_china_exposure.jl` classifies a supply-chain edge's counterparty country **as of the edge start date**, not per quarter — better than a look-ahead, but the "point-in-time" wording is clarified to mean edge-start, not quarter-by-quarter re-classification.
- **F13 (version control).** The `julia_descriptive/` pipeline was local-only (untracked, hard-coded absolute paths). The code is now committed and pushed to `github.com/fidio728/practice` so it is reviewable; a `.gitignore` keeps the 5 GB `output/` data out of version control.

### New files this round (all embedded in §14–§16)
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

- R_{i,t} = active supply-chain edges where firm *i* is on **either** side (**symmetric** counting), summed across five `rel_type`s: `CUSTOMER`, `SUPPLIER`, `PARTNER-JVENTUR`, `PARTNER-MANUFAC`, other `PARTNER-*`.
- R^{CN}_{i,t} = **bilateral union**: {firm i = source, CN = target} ∪ {CN = source, firm i = target}.
- Home-region classification is **point-in-time** (time-versioned), not a single end-of-sample label → no look-ahead in exposure.

Exact SQL for the share (`02_china_exposure.jl`):

```sql
CAST(COALESCE(c.n_cn_total, 0) AS DOUBLE) / NULLIF(t.n_total_links, 0) AS china_share
```

Note `NULLIF(..., 0)`: a firm-quarter with **no supply-chain links at all** gets china_share = **NULL** (not 0), and does not enter the exposure table. This is deliberate (see MISSING bucket, §5.5).

**HIGH-exposure cutoff.** `HIGH` if CN_{i,t−1} > **0.0476** = the median of strictly-positive CN_{i,t−1} across firm-quarters. **Disclosed nuance:** "high exposure" therefore means "at least ~4% of a firm's reported supply-chain edges touch China" — meaningful but **not extreme**.

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

| | (1) No FE | (2) Headline α_{i,t}+γ_{g,t} | (3) Weak FE (firm+quarter) |
|---|---|---|---|
| β₂ (US·CN_{t−1}) | −2.76 (3.72) | +1.98 (5.29) | −0.67 (3.63) |
| β₃ (US·CN·S_t) | −0.35 (0.60) | **+1.28 (1.66)** | −0.12 (0.71) |
| p(β₃) | 0.565 | **0.443** | 0.865 |
| R² | ~0.000 | 0.6215 | 0.0058 |
| F(2,81) | 0.81 (p=0.449) | 0.41 (p=0.668) | 0.04 (p=0.960) |

**Every coefficient is null (p ≥ 0.44).** The headline β₃ is positive (opposite the H2.1 prediction) but indistinguishable from zero; the null is from a large SE, not a tight zero. (Also run: "β₃-only" spec = +1.37 (1.63), p=0.405; "full triple" spec identical to Headline with lower-order terms auto-omitted — confirms the absorption logic.)

### 7.2 + firm × group FE (`07e_firmgroup_tail.do`)

| | Headline | + firm×group FE |
|---|---|---|
| β₃ | +1.28 (1.66), p=0.443 | +1.80 (1.91), p=0.350 |
| N | 462,564 | 462,096 (468 singletons dropped; 15,388 firm-group cells) |

Still null. Interpretation: the null is **not** an artifact of structural US-vs-NONUS firm sorting.

### 7.3 Tail-dummy shock menu (`07e_firmgroup_tail.do`) — with power diagnostic

σ_S over 82 quarters = 2.58. One-sided right (escalation):

| k | **treated quarters** | β₃^tail | SE | p |
|---|---|---|---|---|
| 1.645 | 6 / 82 | +1.07 | 15.7 | 0.946 |
| 2 | 4 / 82 | +14.0 | 18.0 | 0.439 |
| 3 | 2 / 82 | +8.59 | 39.0 | 0.826 |

All null; SE explodes as k rises. Empirical distribution is **fat-tailed** (k=1.645 gives 6 quarters = 7.3%, not the 5% of a normal). **k=3 is essentially uninformative (2 treated quarters).** The dummy coefficient is on a different scale from the continuous one — do not compare point estimates. The tail specs are a robustness layer on top of the continuous S_t, not a replacement.

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
| β₁ (US·S_{c,t}) — newly identified | **+17.7 (10.6), p=0.098** (borderline; only non-null; sign opposite H2.1) |
| β₂ (US·CN) | +5.64 (6.92), p=0.417 |
| β₃ (US·CN·S_{c,t}) | +3.48 (30.2), p=0.909 |

The single borderline coefficient (β₁, p=0.098) says: in GB/DE/FR, US holders lean *marginally toward* European equity when host-country CN tension rises — opposite the disengagement story, and only suggestive.

---

### 7.6 Shares-based robustness: ownership share (`build_ownership_share_panel.py`, `build_ownership_share_c6_panel.py`, `run_ownership_share.do`)

**Why this exists.** The main outcome w = H(USD)/T(USD) is a market-value portfolio weight, so it mixes trading flow with price moves and portfolio-denominator reallocation. A null on w supports "US do not reduce their within-Europe portfolio *weight*", but not the stronger "US do not *sell* / do not reduce their *stake*". For the flow claim we use a pure quantity — the group's ownership share of each firm:

- ownership_share_{i,g,t} = ( Σ_{b∈g} shares held on the primary EQ class ) / (primary-EQ shares outstanding).

It is immune to price (numerator and denominator are both in shares) and to portfolio-denominator reallocation (no T term). Shares are not additive across firms, so a shares-based *portfolio weight* is meaningless; the ownership share is the correct object. `adj_shares_out` is a per-security-class attribute, so we restrict both numerator and denominator to the **primary EQ class** (`fsym_id = fsym_primary_id`, matching the market-cap rule in `04_us_ownership_european.jl`); this makes shares_out constant within each security-quarter (verified: dispersion drops from 5.28% of cells to exactly 0). Outcome = backward Δ(ownership_share) on the **same C6 grid, FE, and clustering** as the main spec, so the test runs on the same firm-quarter universe.

**Result (raw ownership-share units — NOT ×10⁻⁶).** N = 300,866 (5,704 firms that carry a primary-EQ float, 82 quarters 2003Q3–2023Q4, balanced 150,433 US / 150,433 NONUS). Outcome mean ≈ 0, sd ≈ 0.047; ownership level mean 6.3%, median 3.4%.

| coefficient | R1 (firm×qtr, group×qtr FE) | R2 (+ firm×group FE) |
|---|---|---|
| β₂ (US·CN_{t−1}) | −0.00108 (0.00205), p=0.601 | −0.00105 (0.00287), p=0.715 |
| β₃ (US·CN_{t−1}·S_t) | **+0.000599 (0.001165), p=0.609** | +0.000572 (0.001135), p=0.616 |
| joint F (β₂, β₃) | p=0.774 | p=0.801 |

β₃ is null and, if anything, **positive** — opposite to the disengagement prediction (β₃<0). The shares-based test agrees with the w-based null: **US investors do not reduce their ownership stake in high-China-exposure European firms more than non-US investors when tension rises.**

**Not a zero-fill artifact.** Dropping the extensive-margin zeros (observed-held-only, re-paired, N = 231,860) gives β₃ = **+0.00122** — *further* from the disengagement prediction, not toward it. The null is therefore not zero-fill attenuation. (Verified by re-running the triple difference on the held-only subset.)

**Independently reproduced.** β₃ = +0.0005988 via a Python two-way within-transformation, matching the Stata `reghdfe` value to ~2×10⁻⁷; three independent adversarial re-derivations agree on the point estimate, the null, and the construction.

**Caveat 1 (ADR / non-primary exclusion).** The measure is primary-EQ only, so it does not observe stake adjustment through ADR/GDR or non-primary classes. US investors hold **8.92%** of their European exposure off the primary class versus **3.32%** for non-US (2.69× asymmetric). The shares-based null is therefore *complementary* to the USD portfolio-weight main spec, which does capture the ADR channel — not a substitute. An ADR-inclusive ownership measure needs the ADR conversion ratio and is deferred (§10).

**Caveat 2 (power / MDE).** With 82 quarter-clusters the design rules out *large* stake reductions but has limited power against small economically-meaningful adjustments (order 25–35 bps); |β₃| is only ~1.3% of the outcome standard deviation. Read the estimate as bounding the effect near zero, not as proving an exact zero.

---

## 8. The centered-vs-backward disclosure (retraction of the earlier "result")

| Δw definition | β₃ | SE | p |
|---|---|---|---|
| Centered (w_{t+1} − w_{t−1}), prior build | +1.38 | 0.54 | **0.013 (significant, opposite H2.1)** |
| Backward (w_t − w_{t−1}), current | +1.28 | 1.66 | 0.443 (null) |

Point estimate barely moves (+1.38 → +1.28); **SE roughly triples (0.54 → 1.66)**. We read the earlier significance as a **look-ahead / post-treatment window contamination**: the centered LHS includes post-t holdings (w_{t+1}), so the outcome mixes contemporaneous with future adjustment and is not aligned with the estimand's timing. We deliberately do **not** claim a proven "mechanical correlation of shocks" (S_t is an AR(1) residual, so S_t and S_{t+1} need not be strongly correlated), and we do **not** claim to have decomposed *why* the SE tripled. Three forces move together between the two builds — look-ahead removal, a small right-edge sample expansion, and entry/exit reweighting — and they are not separately identified here; a sample-matched centered-window comparison on a common sample (deferred, §10) is needed to attribute the SE change. Identification rests on the within-firm-quarter β₃, which is null.

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
13. **Shares-based test excludes ADRs** (§7.6, Caveat 1). The ownership-share robustness is primary-EQ only, so it cannot see stake adjustment through ADR/GDR or non-primary classes. US holds 8.92% of its European exposure off the primary class versus 3.32% for non-US (2.69× asymmetric), so the shares-based null is complementary to — not a substitute for — the USD portfolio-weight main spec that does capture the ADR channel. Its power is also limited against small (~25–35 bps) adjustments; it bounds the stake effect near zero rather than proving an exact zero.

---

## 10. Deferred robustness (planned, NOT done — do not present as completed)

- Continuing-positions-only subsample (w_{t−1}>0 AND w_t>0) to neutralize entry/exit mechanical asymmetry.
- Winsorize Δw at 1/99%.
- Leave-one-quarter-out (which escalation episodes carry any result).
- PPML via `ppmlhdfe` for the zero-heavy weights (Silva–Tenreyro 2006; Correia et al. 2020).
- Local projections (Jordà 2005) for dynamics.
- **Holder-level fixed effects** (holder × quarter), the true bank×time analog.
- **ADR-inclusive ownership share** (§7.6): map ADR/GDR holdings to underlying-share equivalents via the ADR conversion ratio so the shares-based test captures the ADR channel (US 8.92% of exposure off-primary); needs the ratio data.
- Sample-matched centered-window comparison (quantify the §8 disclosure on a common sample).
- OFAC SDN / Entity-List exclusion; size, industry, listing-country × shock controls.
- **H2.2 price channel**: acquire European stock returns (Datastream / Compustat Global Security / FactSet Prices); abnormal returns from a factor model; Tobin's q from Compustat Global.

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
| + firm×group FE | full grid | +1.80 | 1.91 | 0.350 | 462,096 |
| Tail k=1.645 (6 qtrs) | full grid, dummy | +1.07 | 15.7 | 0.946 | 462,564 |
| Tail k=2 (4 qtrs) | full grid, dummy | +14.0 | 18.0 | 0.439 | 462,564 |
| Tail k=3 (2 qtrs) | full grid, dummy | +8.59 | 39.0 | 0.826 | 462,564 |
| Conditional risk-set | paired, continuous S_t | +2.46 | 3.15 | 0.436 | 342,262 |
| Conditional + firm×group | paired | +2.46 | 3.19 | 0.443 | 341,858 |
| Country-pair (GB+DE+FR) | S_{c,t} | +3.48 | 30.2 | 0.909 | 227,310 |
| **[retracted] centered diff** | full grid | +1.38 | 0.54 | 0.013 | ~450k |
| **[retracted, biased] per-group spell** | broke pairing | +4.38 | 5.66 | 0.441 | 250,918 |

**Bottom line for the reviewer:** every valid specification returns a null β₃. The one historically significant result is a retracted forward-window estimate (post-treatment timing contamination); one borderline coefficient (country-pair β₁ = +17.7, p=0.098) points *opposite* to the disengagement hypothesis. The honest reading is that the ownership-flow channel shows no detectable differential US disengagement, and the project should be judged on (a) the panel/identification construction and (b) the planned firm-level price channel (H2.2), not on a confirmed behavioral effect.

---

## 14. Full Stata do-files (verbatim, on-disk copies)
All under `julia_descriptive/`. `07f_spell_boundary.do` is **superseded** (§7.4) and omitted. `run_headline_3pairwise.do` is the new three-way pairwise FE headline; `run_audit_f1f2f7.do` / `run_audit_f1b_robust.do` / `run_audit_f4f8.do` implement the second-review-round tests (F1/F2/F7, LP robust inference, F4/F8). **Refreshed from disk after both review rounds.**

### `07_regression.do`

```stata
* 07_regression.do
* Headline triple-difference β_3 on the C6 zero-filled panel.
*
* Spec:  Δw_{b,i,c,t} = β_2·US_c × CN_{i,t-1} + β_3·US_c × CN_{i,t-1} × Shock_t
*                    + α_{i,t} + α_{c×t} + ε
*
* All other De Haas 4-coefficient terms are absorbed by the high-dim FE:
*   - β_0·US_c            absorbed by α_{c×t}  (US_c constant within c)
*   - β_1·US_c × Shock_t   absorbed by α_{c×t}  (Shock_t constant within t)
*   - CN_{i,t-1}           absorbed by α_{i,t}  (CN_lag constant within (i,t))
*   - CN_{i,t-1} × Shock_t absorbed by α_{i,t}  (Shock_t constant within t)
*
* Controls: no separate control variables are added — every firm-quarter level
* and holder-group-quarter level control is absorbed by α_{i,t} + α_{c×t}.

clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/c6_panel.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

display _newline "=== Sample composition ==="
count
display _newline "Holder group:"
tab hgroup
display _newline "Quarters covered:"
sum rdate, format

* Numeric IDs for FE / clustering
* pandas to_stata writes datetime64 as %tc (milliseconds); convert to daily
* with dofc() before extracting month/quarter.
gen rd_day = dofc(rdate)
format rd_day %td

egen firm_n   = group(firm_str)
egen fq       = group(firm_str rd_day)
gen  rd_m     = mofd(rd_day)
format rd_m %tm
egen gq       = group(hgroup rd_day)

* Sanity: rd_day must not be all missing
count if missing(rd_day)
assert r(N) == 0

* Pre-compute interaction terms explicitly so reghdfe doesn't auto-drop
gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock

* Label for output
label var us_cn       "US × CN(t-1)"
label var us_cn_shock "US × CN(t-1) × Shock"

* ============================================================
* Spec 1: minimal identified pair (β_2 + β_3) with full FE stack
*         α_{i,t} + α_{c×t}
* ============================================================
display _newline _newline "=== Spec 1: β_2 + β_3, FE = (firm × quarter) + (hgroup × quarter) ==="
reghdfe dw us_cn us_cn_shock, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store m1

* ============================================================
* Spec 2: triple only (no β_2) — does β_3 survive without γ?
* ============================================================
display _newline _newline "=== Spec 2: β_3 only, FE = (firm × quarter) + (hgroup × quarter) ==="
reghdfe dw us_cn_shock, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store m2

* ============================================================
* Spec 3: full De Haas 4-coefficient form (let reghdfe drop absorbed terms)
* ============================================================
display _newline _newline "=== Spec 3: full c.us##c.cn_lag##c.shock with all FE ==="
reghdfe dw c.us##c.cn_lag##c.shock, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store m3

* ============================================================
* Spec 4: weaker FE (separate firm + quarter, no interactions) — just for
*         comparison to show how much β_3 changes when FE saturate.
* ============================================================
display _newline _newline "=== Spec 4: weaker FE = firm + quarter (no interactions) ==="
reghdfe dw us_cn us_cn_shock, ///
    absorb(firm_n rd_m) ///
    vce(cluster firm_n rd_m)
estimates store m4

* ============================================================
* Summary table
* ============================================================
display _newline _newline "=== Summary table ==="

* Use esttab if available; otherwise fall back to estimates table
capture which esttab
if _rc == 0 {
    esttab m4 m3 m1 m2 using "`OUT'/07_reghdfe_results.txt", replace ///
        cells("b(fmt(6) star) se(fmt(6))") ///
        stats(N r2 r2_a, fmt(%9.0gc %6.4f %6.4f) labels("N" "R²" "Adj R²")) ///
        keep(us_cn us_cn_shock) ///
        title("β_3 on C6 zero-filled panel — different FE specifications") ///
        mtitles("Sep firm+qtr" "Full saturation" "β_2 + β_3" "β_3 only") ///
        note("Cluster SE on (firm, quarter). C6 zero-filled panel, n ≈ 450k.")
    di "Wrote esttab output to `OUT'/07_reghdfe_results.txt"
}
else {
    estimates table m4 m3 m1 m2, b(%9.6f) se(%9.6f) ///
        stats(N r2 r2_a) keep(us_cn us_cn_shock)
}

* Always also print to log for capture
display _newline "=== FINAL HEADLINE β_3 estimate (Spec 1, recommended) ==="
estimates restore m1
estimates table, b(%9.6f) se(%9.6f) p(%6.4f)

display _newline "Done."
```

### `07b_country_pair_robustness.do`

```stata
* 07b_country_pair_robustness.do
* Country-pair GPR shock robustness for the GB+DE+FR subsample.
*
* Robustness redesign:
*   Main spec uses Shock_t = AR(1) residual of USA|China bilateral GPR_AI,
*   applied identically to every EU listing country. This do-file replaces
*   the homogeneous Shock_t with country-pair-specific shocks S_{c,t}
*   constructed as AR(1) residuals of (UK|China), (Germany|China),
*   (France|China), and merges S_{c,t} to the panel by (sec_country, quarter).
*
* Identification with c.us##c.cn_lag##c.shock_c and FE = alpha_{i,t} + gamma_{g,t}:
*   - beta_0 * S_{c,t}            absorbed by alpha_{i,t} (c(i) constant in t)
*   - beta_1 * US_g * S_{c,t}     IDENTIFIED (NEW vs main)
*   - CN_{i,t-1}                  absorbed by alpha_{i,t}
*   - US_g * CN_{i,t-1}           IDENTIFIED (= main beta_2)
*   - CN * S_{c,t}                absorbed by alpha_{i,t}
*   - US * CN * S_{c,t}           IDENTIFIED (= main beta_3 with new shock)
* So three coefficients survive on the pooled subsample.
*
* Spec C3 reruns the MAIN US-CN shock (shock_us_cn) on the SAME subsample
* (GB+DE+FR firms) so the country-pair result is benchmarked against the
* homogeneous-shock baseline on identical observations.
*
* Cluster on (firm, quarter); NOT on country (only 3 countries -> too few).
*
* Stata gotcha (from 07_regression.do): pandas to_stata writes datetime64
* as %tc (milliseconds); MUST apply dofc() before mofd().

clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/c6_panel_country_pair.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

* ============================================================
* Sanity counts
* ============================================================
display _newline "=== Sample composition (country-pair robustness subsample) ==="
count
display _newline "Holder group:"
tab hgroup
display _newline "Listing country (must be GB/DE/FR only):"
tab sec_country
display _newline "Quarters covered:"
sum rdate, format

* Hard assertion: only GB/DE/FR firms allowed
gen byte _bad_country = !inlist(sec_country, "GB", "DE", "FR")
count if _bad_country == 1
assert r(N) == 0
drop _bad_country

* Numeric IDs for FE / clustering
gen rd_day = dofc(rdate)
format rd_day %td

egen firm_n   = group(firm_str)
egen fq       = group(firm_str rd_day)
gen  rd_m     = mofd(rd_day)
format rd_m %tm
egen gq       = group(hgroup rd_day)

* Sanity: rd_day must not be all missing
count if missing(rd_day)
assert r(N) == 0

* Diagnostic: how many rows have missing shock_c or shock_us_cn?
count if missing(shock_c)
display "rows with missing shock_c: " r(N)
count if missing(shock_us_cn)
display "rows with missing shock_us_cn: " r(N)

* ============================================================
* Pre-compute interaction terms (clean names for esttab)
* ============================================================
gen us_cn          = us * cn_lag
gen us_s_c         = us * shock_c
gen us_cn_s_c      = us * cn_lag * shock_c
gen us_cn_s_main   = us * cn_lag * shock_us_cn

label var us_cn        "US x CN(t-1)"
label var us_s_c       "US x Shock_c"
label var us_cn_s_c    "US x CN(t-1) x Shock_c"
label var us_cn_s_main "US x CN(t-1) x Shock_USA-CN"

* Per-country flags
gen byte _gb = (sec_country == "GB")
gen byte _de = (sec_country == "DE")
gen byte _fr = (sec_country == "FR")

* ============================================================
* Spec C1: explicit triple-interaction form (screen display only)
*          reghdfe auto-omits absorbed terms; serves as a sanity check
*          that surviving coefs match Spec C2's hand-built form.
* ============================================================
display _newline _newline "=== Spec C1: c.us##c.cn_lag##c.shock_c (screen check) ==="
reghdfe dw c.us##c.cn_lag##c.shock_c, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store c1

* ============================================================
* Spec C2 (HEADLINE): explicit beta_1 + beta_2 + beta_3 with clean names
* ============================================================
display _newline _newline "=== Spec C2 (HEADLINE): us_cn + us_s_c + us_cn_s_c ==="
reghdfe dw us_cn us_s_c us_cn_s_c, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store c2

* ============================================================
* Spec C3: MAIN US-CN shock on the SAME GB+DE+FR subsample (benchmark)
* ============================================================
display _newline _newline "=== Spec C3: MAIN US-CN shock on GB+DE+FR subsample ==="
reghdfe dw us_cn us_cn_s_main, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store c3

* ============================================================
* Per-country breakdowns: within one country, S_{c,t} varies only in t,
* so US x Shock_c is constant within (g,t) and gets absorbed by gq.
* Only us_cn and us_cn_s_c are identified per-country.
* ============================================================
display _newline _newline "=== Spec C1-GB: GB-only subsample ==="
capture noisily reghdfe dw us_cn us_cn_s_c if _gb == 1, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
if _rc == 0 estimates store c1_gb

display _newline _newline "=== Spec C1-DE: DE-only subsample ==="
capture noisily reghdfe dw us_cn us_cn_s_c if _de == 1, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
if _rc == 0 estimates store c1_de

display _newline _newline "=== Spec C1-FR: FR-only subsample ==="
capture noisily reghdfe dw us_cn us_cn_s_c if _fr == 1, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
if _rc == 0 estimates store c1_fr

* ============================================================
* Summary tables (esttab if available)
* ============================================================
display _newline _newline "=== Summary tables ==="

capture which esttab
if _rc == 0 {
    * Main table: C2 (country-pair shock, HEADLINE) vs C3 (main shock benchmark)
    esttab c2 c3 using "`OUT'/07b_country_pair_results.txt", replace ///
        cells("b(fmt(6) star) se(fmt(6))") ///
        stats(N r2 r2_a, fmt(%9.0gc %6.4f %6.4f) labels("N" "R2" "Adj R2")) ///
        keep(us_cn us_s_c us_cn_s_c us_cn_s_main) ///
        title("Country-pair GPR shock robustness - GB+DE+FR subsample") ///
        mtitles("C2: country-pair shock" "C3: main USA-CN shock") ///
        note("FE = firm x qtr + hgroup x qtr. Cluster SE on (firm, quarter). Subsample: sec_country in GB/DE/FR.")

    * Per-country breakdown table
    esttab c2 c1_gb c1_de c1_fr using "`OUT'/07b_country_pair_by_country.txt", replace ///
        cells("b(fmt(6) star) se(fmt(6))") ///
        stats(N r2 r2_a, fmt(%9.0gc %6.4f %6.4f) labels("N" "R2" "Adj R2")) ///
        keep(us_cn us_s_c us_cn_s_c) ///
        title("Country-pair GPR shock - per-country breakdown") ///
        mtitles("All (C2)" "GB only" "DE only" "FR only") ///
        note("FE = firm x qtr + hgroup x qtr. us_s_c absorbed within single country.")

    * CSV version
    esttab c2 c3 using "`OUT'/07b_country_pair_results.csv", replace ///
        cells("b(fmt(6)) se(fmt(6)) p(fmt(4))") ///
        stats(N r2 r2_a, fmt(%9.0gc %6.4f %6.4f)) ///
        keep(us_cn us_s_c us_cn_s_c us_cn_s_main) ///
        mtitles("C2_country_pair" "C3_main_shock") ///
        nonumbers plain

    di "Wrote esttab outputs to `OUT'/07b_country_pair_*.txt and .csv"
}
else {
    estimates table c2 c3, b(%9.6f) se(%9.6f) ///
        stats(N r2 r2_a) keep(us_cn us_s_c us_cn_s_c us_cn_s_main)
    estimates table c2 c1_gb c1_de c1_fr, b(%9.6f) se(%9.6f) ///
        stats(N r2 r2_a) keep(us_cn us_cn_s_c)
}

* ============================================================
* Final headline: Spec C2 (country-pair shock) coefficients
* ============================================================
display _newline "=== FINAL HEADLINE: Spec C2 coefficients (beta_1, beta_2, beta_3) ==="
estimates restore c2
estimates table, b(%9.6f) se(%9.6f) p(%6.4f)

display _newline "Done."
```

### `07c_strict_4term.do`

```stata
* 07c_strict_4term.do
* Strict De Haas 4-term symmetric notation, slide form:
*
*   Delta_w_{i,g,t} = beta_0 * US_g
*                   + beta_1 * US_g * S_t
*                   + beta_2 * US_g * CN_{i,t-1}
*                   + beta_3 * US_g * CN_{i,t-1} * S_t
*                   + alpha_{i,t}  (firm x quarter)
*                   + gamma_{g,t}  (group x quarter)
*                   + eps_{i,g,t}
*
* Two specs, hand-built interaction terms, NO extra robustness columns.
*
* Spec M (MAIN, full panel, US-CN shock):
*   reghdfe dw us_lvl us_s us_cn us_cn_s, absorb(fq gq) vce(cluster firm_n rd_m)
*   Expected omissions:
*     us_lvl  -> omitted (absorbed by gamma_{g,t})
*     us_s    -> omitted (S_t has no host-country variation -> absorbed by gamma_{g,t})
*     us_cn   -> IDENTIFIED (beta_2)
*     us_cn_s -> IDENTIFIED (beta_3) HEADLINE
*
* Spec P (COUNTRY-PAIR, GB+DE+FR subsample, shock_c):
*   reghdfe dw us_lvl us_s_c us_cn us_cn_s_c, absorb(fq gq) vce(cluster firm_n rd_m)
*   Expected omissions:
*     us_lvl    -> omitted (absorbed by gamma_{g,t})
*     us_s_c    -> IDENTIFIED (beta_1) NEW vs Spec M
*     us_cn     -> IDENTIFIED (beta_2)
*     us_cn_s_c -> IDENTIFIED (beta_3) HEADLINE
*
* Stata gotchas (per project notes):
*   - pandas to_stata writes datetime64 as %tc (ms). MUST apply dofc() before mofd().
*   - In c6_panel.dta the shock column is named "shock". In
*     c6_panel_country_pair.dta the columns are "shock_us_cn" and "shock_c".

clear all
set more off

local DTA_M "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/c6_panel.dta"
local DTA_P "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/c6_panel_country_pair.dta"
local OUT   "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

* ============================================================
* SPEC M: MAIN (full panel, US-CN shock)
* ============================================================
use "`DTA_M'", clear

display _newline "=== Spec M: sample composition (full panel) ==="
count
display _newline "Holder group:"
tab hgroup
display _newline "Quarters covered:"
sum rdate, format

* Numeric IDs for FE / clustering.
* pandas to_stata writes datetime64 as %tc; dofc() before mofd().
gen rd_day = dofc(rdate)
format rd_day %td

egen firm_n = group(firm_str)
egen fq     = group(firm_str rd_day)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen gq     = group(hgroup rd_day)

* Sanity
count if missing(rd_day)
assert r(N) == 0

* Hand-built 4 interaction terms matching slide notation.
* In this file the AR(1) residual column is literally named "shock".
gen us_lvl  = us
gen us_s    = us * shock
gen us_cn   = us * cn_lag
gen us_cn_s = us * cn_lag * shock

label var us_lvl  "US_g (beta_0)"
label var us_s    "US_g x S_t (beta_1)"
label var us_cn   "US_g x CN(t-1) (beta_2)"
label var us_cn_s "US_g x CN(t-1) x S_t (beta_3)"

display _newline _newline "=== Spec M: reghdfe dw us_lvl us_s us_cn us_cn_s, absorb(fq gq) ==="
reghdfe dw us_lvl us_s us_cn us_cn_s, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store specM

* ============================================================
* SPEC P: COUNTRY-PAIR (GB+DE+FR subsample, shock_c)
* ============================================================
use "`DTA_P'", clear

display _newline "=== Spec P: sample composition (GB+DE+FR subsample) ==="
count
display _newline "Listing country (must be GB/DE/FR only):"
tab sec_country
display _newline "Holder group:"
tab hgroup
display _newline "Quarters covered:"
sum rdate, format

* Hard guard: only GB/DE/FR firms allowed
gen byte _bad_country = !inlist(sec_country, "GB", "DE", "FR")
count if _bad_country == 1
assert r(N) == 0
drop _bad_country

* Numeric IDs for FE / clustering (same recipe as Spec M).
gen rd_day = dofc(rdate)
format rd_day %td

egen firm_n = group(firm_str)
egen fq     = group(firm_str rd_day)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen gq     = group(hgroup rd_day)

count if missing(rd_day)
assert r(N) == 0

* Diagnostic on shock_c coverage
count if missing(shock_c)
display "rows with missing shock_c: " r(N)

* Hand-built 4 interaction terms matching slide notation,
* with shock replaced by country-pair shock_c.
gen us_lvl    = us
gen us_s_c    = us * shock_c
gen us_cn     = us * cn_lag
gen us_cn_s_c = us * cn_lag * shock_c

label var us_lvl    "US_g (beta_0)"
label var us_s_c    "US_g x S_{c,t} (beta_1)"
label var us_cn     "US_g x CN(t-1) (beta_2)"
label var us_cn_s_c "US_g x CN(t-1) x S_{c,t} (beta_3)"

display _newline _newline "=== Spec P: reghdfe dw us_lvl us_s_c us_cn us_cn_s_c, absorb(fq gq) ==="
reghdfe dw us_lvl us_s_c us_cn us_cn_s_c, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estimates store specP

* ============================================================
* Side-by-side summary table (Spec M and Spec P)
* ============================================================
display _newline _newline "=== Side-by-side: Spec M | Spec P ==="

capture which esttab
if _rc == 0 {
    esttab specM specP using "`OUT'/07c_strict_4term_results.txt", replace ///
        cells("b(fmt(6) star) se(fmt(6))") ///
        stats(N r2 r2_a, fmt(%9.0gc %6.4f %6.4f) labels("N" "R2" "Adj R2")) ///
        keep(us_lvl us_s us_s_c us_cn us_cn_s us_cn_s_c) ///
        order(us_lvl us_s us_s_c us_cn us_cn_s us_cn_s_c) ///
        title("Strict 4-term form (slide notation): Spec M vs Spec P") ///
        mtitles("Spec M (main, S_t)" "Spec P (country-pair, S_{c,t})") ///
        note("FE = firm x qtr + hgroup x qtr. Cluster SE on (firm, quarter). " ///
             "Spec M: full panel, US-CN AR(1) residual. " ///
             "Spec P: GB+DE+FR subsample, country-pair AR(1) residual S_{c,t}. " ///
             "us_lvl absorbed by gamma_{g,t} in both. us_s absorbed in Spec M (no c-variation in S_t).")

    esttab specM specP using "`OUT'/07c_strict_4term_results.csv", replace ///
        cells("b(fmt(6)) se(fmt(6)) p(fmt(4))") ///
        stats(N r2 r2_a, fmt(%9.0gc %6.4f %6.4f)) ///
        keep(us_lvl us_s us_s_c us_cn us_cn_s us_cn_s_c) ///
        order(us_lvl us_s us_s_c us_cn us_cn_s us_cn_s_c) ///
        mtitles("SpecM_main" "SpecP_country_pair") ///
        nonumbers plain

    di "Wrote esttab outputs to `OUT'/07c_strict_4term_results.txt and .csv"
}
else {
    estimates table specM specP, b(%9.6f) se(%9.6f) p(%6.4f) ///
        stats(N r2 r2_a) ///
        keep(us_lvl us_s us_s_c us_cn us_cn_s us_cn_s_c)
}

* ============================================================
* Final headline display
* ============================================================
display _newline "=== FINAL HEADLINE: Spec M (main, full panel) ==="
estimates restore specM
estimates table, b(%9.6f) se(%9.6f) p(%6.4f)

display _newline "=== FINAL HEADLINE: Spec P (country-pair, GB+DE+FR) ==="
estimates restore specP
estimates table, b(%9.6f) se(%9.6f) p(%6.4f)

display _newline "Done."
```

### `07d_three_spec_table.do`

```stata
* 07d_three_spec_table.do
* Slide-9 headline three-column table on the C6 zero-filled (backward-diff) panel.
*
* Columns (left to right in slide):
*   (1) Col 1  - SPEC 0  -  NO FE BASELINE       -> estimates m0
*       reghdfe dw us_cn us_cn_shock, noabsorb vce(cluster firm_n rd_m)
*       Pure OLS with two-way cluster; returns _cons; "unconditional
*       differential association" interpretation of beta_2 and beta_3.
*       (Note: noabsorb is documented as a no-op in current reghdfe but
*        the syntax is harmless; coefs match `regress` exactly.)
*
*   (2) Col 2  - SPEC 1  -  HEADLINE             -> estimates m1
*       reghdfe dw us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
*       fq = group(firm_str rd_day), gq = group(hgroup rd_day).
*       Already locked: beta_3 = +1.28e-06, SE = 1.66e-06, p = 0.443,
*       N = 462,564, R2 = 0.6215, F(2,81) = 0.41 p = 0.668.
*
*   (3) Col 3  - SPEC 4  -  WEAK FE              -> estimates m4
*       reghdfe dw us_cn us_cn_shock, absorb(firm_n rd_m) vce(cluster firm_n rd_m)
*       Already locked: beta_3 = -1.22e-07, SE = 7.14e-07, p = 0.865,
*       N = 462,564, R2 = 0.0058.
*
* BLOCKING fix from adversarial review: reghdfe does NOT save e(p_F),
* so we add `estadd scalar p_F = Ftail(e(df_m), e(df_r), e(F))` after
* each regression so the esttab Prob>F row is populated.

clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/c6_panel.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

display _newline "=== Sample composition ==="
count
display _newline "Holder group:"
tab hgroup
display _newline "Quarters covered:"
sum rdate, format

* Panel keys (identical to 07_regression.do)
gen rd_day = dofc(rdate)
format rd_day %td

egen firm_n   = group(firm_str)
egen fq       = group(firm_str rd_day)
gen  rd_m     = mofd(rd_day)
format rd_m %tm
egen gq       = group(hgroup rd_day)

count if missing(rd_day)
assert r(N) == 0

gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock

label var us_cn       "us_cn"
label var us_cn_shock "us_cn_shock"

* ============================================================
* Col 1 / Spec 0: NO FE baseline (pure OLS, two-way cluster).
* ============================================================
display _newline _newline "=== Col 1 / Spec 0: NO FE baseline (reghdfe, noabsorb) ==="
reghdfe dw us_cn us_cn_shock, ///
    noabsorb ///
    vce(cluster firm_n rd_m)
estadd scalar p_F = Ftail(e(df_m), e(df_r), e(F))
estimates store m0

display _newline "Spec 0 -- F and R2 (for slide footer):"
display "  F            = " %12.4f e(F)
display "  df_m         = " %12.0f e(df_m)
display "  df_r         = " %12.0f e(df_r)
display "  r2           = " %12.6f e(r2)
display "  r2_a         = " %12.6f e(r2_a)
display "  N            = " %12.0f e(N)

display _newline "Spec 0 -- full coefficient table including _cons:"
estimates table m0, b(%12.4e) se(%12.4e) p(%6.4f)

* ============================================================
* Col 2 / Spec 1: HEADLINE -- alpha_{i,t} + alpha_{c x t}
* ============================================================
display _newline _newline "=== Col 2 / Spec 1: HEADLINE, FE = (firm x quarter) + (hgroup x quarter) ==="
reghdfe dw us_cn us_cn_shock, ///
    absorb(fq gq) ///
    vce(cluster firm_n rd_m)
estadd scalar p_F = Ftail(e(df_m), e(df_r), e(F))
estimates store m1

display _newline "Spec 1 -- F and R2 (lock check):"
display "  F            = " %12.4f e(F)
display "  df_m         = " %12.0f e(df_m)
display "  df_r         = " %12.0f e(df_r)
display "  r2           = " %12.6f e(r2)
display "  N            = " %12.0f e(N)

* ============================================================
* Col 3 / Spec 4: WEAK FE -- firm + quarter (no interactions)
* ============================================================
display _newline _newline "=== Col 3 / Spec 4: WEAK FE, absorb(firm_n rd_m) ==="
reghdfe dw us_cn us_cn_shock, ///
    absorb(firm_n rd_m) ///
    vce(cluster firm_n rd_m)
estadd scalar p_F = Ftail(e(df_m), e(df_r), e(F))
estimates store m4

display _newline "Spec 4 -- F and R2 (lock check):"
display "  F            = " %12.4f e(F)
display "  df_m         = " %12.0f e(df_m)
display "  df_r         = " %12.0f e(df_r)
display "  r2           = " %12.6f e(r2)
display "  N            = " %12.0f e(N)

* ============================================================
* Side-by-side table: order (m0, m1, m4) = slide cols (1, 2, 3).
* Drop _cons from keep() to keep table clean (per reviewer P2).
* ============================================================
display _newline _newline "=== Slide-9 three-column headline table ==="

capture which esttab
if _rc == 0 {
    esttab m0 m1 m4 using "`OUT'/07d_three_spec_results.txt", replace ///
        cells("b(fmt(%9.3e) star) se(fmt(%9.3e))") ///
        stats(N r2 r2_a F p_F, ///
              fmt(%9.0gc %6.4f %6.4f %9.4f %6.4f) ///
              labels("N" "R2" "Adj R2" "F" "Prob>F")) ///
        keep(us_cn us_cn_shock) ///
        order(us_cn us_cn_shock) ///
        title("Slide 9 -- three-column headline table on C6 zero-filled (backward) panel") ///
        mtitles("(1) No FE" "(2) Headline" "(3) Weak FE") ///
        nonumbers ///
        note("Two-way cluster (firm, quarter) in all columns. Coefficients in scientific notation; slide rescales by 10^-6.")

    esttab m0 m1 m4 using "`OUT'/07d_three_spec_results.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2 r2_a F p_F, ///
              fmt(%9.0gc %6.4f %6.4f %9.4f %6.4f) ///
              labels("N" "R2" "Adj R2" "F" "Prob>F")) ///
        keep(us_cn us_cn_shock) ///
        order(us_cn us_cn_shock) ///
        mtitles("SpecM0_no_FE" "SpecM1_headline" "SpecM4_weak_FE") ///
        nonumbers plain

    di "Wrote esttab outputs to:"
    di "  `OUT'/07d_three_spec_results.txt"
    di "  `OUT'/07d_three_spec_results.csv"
}
else {
    di as error "esttab not installed -- falling back to estimates table"
    estimates table m0 m1 m4, b(%12.4e) se(%12.4e) p(%6.4f) ///
        stats(N r2 r2_a F) keep(us_cn us_cn_shock)
}

* Echo all three for eyeball verification
display _newline "=== Lock-check: Spec 0 (No FE) ==="
estimates restore m0
estimates table, b(%12.4e) se(%12.4e) p(%6.4f)

display _newline "=== Lock-check: Spec 1 (Headline) ==="
estimates restore m1
estimates table, b(%12.4e) se(%12.4e) p(%6.4f)

display _newline "=== Lock-check: Spec 4 (Weak FE) ==="
estimates restore m4
estimates table, b(%12.4e) se(%12.4e) p(%6.4f)

display _newline "Done."
```

### `07e_firmgroup_tail.do`

```stata
* 07e_firmgroup_tail.do
* Two robustness blocks on the backward-diff C6 panel (c6_panel.dta):
*   (A) Add firm x group FE  mu_{i,g}  -- the "missing third pairwise" FE
*       (advisor's "another two-way interaction"). absorb(fq gq ig).
*   (B) Tail-dummy shock: replace continuous S_t with 1[z_t > k] one-sided
*       (escalation = right tail), k = 1.645 / 2 / 3. Report treated-quarter
*       counts at each k (power diagnostic).
*
* Headline reference: beta_3 = +1.28 (SE 1.66, p=0.443), N=462,564.
*
* Design notes / self-review:
*   - sigma_S computed over the 82 DISTINCT quarters (tag(rd_m)), NOT over the
*     462k rows, so it is not weighted by firms-per-quarter.
*   - one-sided right tail: H1 is about tension RISING, so ShockTail = 1 when
*     z_t > k (large unexpected escalation).
*   - firm x group FE (ig) is the third pairwise FE among {firm, group, quarter};
*     beta_2, beta_3 stay identified off within-(firm,group) time variation in
*     CN_{t-1} and S_t.
*   - Stata gotcha: dofc() before mofd() (pandas %tc). z>k guarded for missing.

clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/c6_panel.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

display _newline "=== Sample ==="
count

* keys
gen rd_day = dofc(rdate)
format rd_day %td
egen firm_n = group(firm_str)
egen fq     = group(firm_str rd_day)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen gq     = group(hgroup rd_day)
egen ig     = group(firm_str hgroup)      // NEW: firm x group

count if missing(rd_day)
assert r(N) == 0

* continuous-shock interactions
gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock
label var us_cn       "US x CN(t-1)"
label var us_cn_shock "US x CN(t-1) x S_t (continuous)"

* ============================================================
* (A) firm x group FE robustness
* ============================================================
display _newline _newline "=== A0: headline (absorb fq gq) -- reference ==="
reghdfe dw us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store base

display _newline _newline "=== A1: + firm x group FE (absorb fq gq ig) ==="
reghdfe dw us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
estimates store mig

* ============================================================
* (B) Tail-dummy shock
*     sigma over DISTINCT quarters; one-sided right (escalation)
* ============================================================
egen qtag = tag(rd_m)
quietly summarize shock if qtag==1
local smean = r(mean)
local ssd   = r(sd)
local nq    = r(N)
display _newline "=== Shock distribution over distinct quarters ==="
display "  n quarters = `nq'"
display "  mean(S_t)  = " %9.5f `smean'
display "  sd(S_t)    = " %9.5f `ssd'

gen z = (shock - `smean')/`ssd'

display _newline "=== Tail-dummy treated-quarter counts (one-sided right) ==="
foreach k in 1645 2000 3000 {
    local kk = `k'/1000
    gen byte tail`k' = (z > `kk') if !missing(z)
    quietly count if qtag==1 & tail`k'==1
    local ntq = r(N)
    quietly count if tail`k'==1
    local nrows = r(N)
    display "  k=`kk':  treated quarters = `ntq'  (of `nq')   treated rows = `nrows'"
    gen us_cn_tail`k' = us * cn_lag * tail`k'
    label var us_cn_tail`k' "US x CN(t-1) x 1[S_t > `kk' SD]"
}

display _newline _newline "=== B1: tail k=1.645 ==="
reghdfe dw us_cn us_cn_tail1645, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store t1645

display _newline _newline "=== B2: tail k=2.0 (headline tail) ==="
reghdfe dw us_cn us_cn_tail2000, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store t2000

display _newline _newline "=== B3: tail k=3.0 ==="
reghdfe dw us_cn us_cn_tail3000, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store t3000

* ============================================================
* Summary
* ============================================================
display _newline _newline "=== Summary: continuous + firm-group FE + tail menu ==="
capture which esttab
if _rc == 0 {
    esttab base mig t1645 t2000 t3000 using "`OUT'/07e_firmgroup_tail.txt", replace ///
        cells("b(fmt(%9.3e) star) se(fmt(%9.3e))") ///
        stats(N r2, fmt(%9.0gc %6.4f) labels("N" "R2")) ///
        keep(us_cn us_cn_shock us_cn_tail1645 us_cn_tail2000 us_cn_tail3000) ///
        order(us_cn us_cn_shock us_cn_tail1645 us_cn_tail2000 us_cn_tail3000) ///
        mtitles("base" "+firmXgroup" "tail k=1.645" "tail k=2" "tail k=3") ///
        title("Essay 2 robustness: firm-group FE and tail-dummy shock") ///
        note("Coefs x10^-6. Two-way cluster (firm, quarter). Tail one-sided right (escalation).")

    esttab base mig t1645 t2000 t3000 using "`OUT'/07e_firmgroup_tail.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) ///
        keep(us_cn us_cn_shock us_cn_tail1645 us_cn_tail2000 us_cn_tail3000) ///
        mtitles("base" "firmXgroup" "tail1645" "tail2000" "tail3000") ///
        nonumbers plain
    di "Wrote `OUT'/07e_firmgroup_tail.txt and .csv"
}

* on-screen lock check of each beta_3
display _newline "=== beta_3 across specs (b / se / p) ==="
foreach m in base mig t1645 t2000 t3000 {
    estimates restore `m'
    display _newline "--- spec: `m' ---"
    estimates table, b(%12.4e) se(%12.4e) p(%6.4f)
}

display _newline "Done."
```

### `07g_spell_riskset.do`

```stata
* 07g_spell_riskset.do
* CORRECTED conditional sample: firm-quarter risk set (both groups kept), so the
* within-firm-quarter US-vs-NONUS comparison is preserved (no group singletons).
* Sample: c6_panel_riskset.dta (342,262 rows, balanced 171,131 US / 171,131 NONUS,
* 6,355 firms). Supersedes 07f (per-group selection, which broke the pairing).
*
* Compare: full grid beta_3 = +1.28 (p=0.443, N=462,564);
*          per-group spell (07f, biased) beta_3 = +4.38 (p=0.441, 45,672 singletons).

clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/c6_panel_riskset.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

display _newline "=== Risk-set conditional sample (balanced, paired) ==="
count
tab hgroup
sum rdate, format

gen rd_day = dofc(rdate)
format rd_day %td
egen firm_n = group(firm_str)
egen fq     = group(firm_str rd_day)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen gq     = group(hgroup rd_day)
egen ig     = group(firm_str hgroup)
count if missing(rd_day)
assert r(N) == 0

gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock
label var us_cn       "US x CN(t-1)"
label var us_cn_shock "US x CN(t-1) x S_t"

display _newline _newline "=== R1: headline (absorb fq gq), risk-set sample ==="
reghdfe dw us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store r1

display _newline _newline "=== R2: + firm x group FE (absorb fq gq ig) ==="
reghdfe dw us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
estimates store r2

display _newline _newline "=== Summary (risk-set conditional sample) ==="
capture which esttab
if _rc == 0 {
    esttab r1 r2 using "`OUT'/07g_spell_riskset.txt", replace ///
        cells("b(fmt(%9.3e) star) se(fmt(%9.3e))") ///
        stats(N r2, fmt(%9.0gc %6.4f) labels("N" "R2")) ///
        keep(us_cn us_cn_shock) ///
        mtitles("riskset: fq gq" "riskset: + firmXgroup") ///
        title("Essay 2: firm-quarter risk-set conditional sample (paired US/NONUS)") ///
        note("Coefs x10^-6. Two-way cluster (firm, quarter). Both groups kept for any firm-quarter with a spell nearby.")
    esttab r1 r2 using "`OUT'/07g_spell_riskset.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) ///
        keep(us_cn us_cn_shock) mtitles("riskset_fqgq" "riskset_firmgroup") nonumbers plain
    di "Wrote `OUT'/07g_spell_riskset.txt and .csv"
}

display _newline "=== beta_2 / beta_3 (b / se / p) ==="
foreach m in r1 r2 {
    estimates restore `m'
    display _newline "--- `m' ---"
    estimates table, b(%12.4e) se(%12.4e) p(%6.4f)
}

display _newline "Done."
```

### `run_ownership_share.do`

```stata
* run_ownership_share.do — Essay 2 #5 STEP 3 regression.
* Shares-based outcome: dos = backward Delta(ownership_share), primary-EQ float.
* Same De Haas 4-coef triple difference as the w-based main spec, same FE + clustering.
*   dos = b2 (US x CN_{t-1}) + b3 (US x CN_{t-1} x S_t) + firm#quarter FE + group#quarter FE
* b3 (us_cn_shock) is the make-or-break: b3<0 => US reduce their STAKE more when tension rises.
* Compare to the w-based headline (null b3). Panel: ownership_c6_panel.dta.

clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/ownership_c6_panel.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

display _newline "=== ownership_c6_panel ==="
count
tab hgroup
summarize flow dos os, detail

gen rd_day = dofc(rdate)
format rd_day %td
egen firm_n = group(firm_str)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen fq     = group(firm_str rd_day)
egen gq     = group(hgroup rd_day)
egen ig     = group(firm_str hgroup)
count if missing(rd_day)
assert r(N) == 0

gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock
label var us_cn       "US x CN(t-1)"
label var us_cn_shock "US x CN(t-1) x S_t"

display _newline _newline "=== R1: PRIMARY outcome = FLOW (held_t-held_{t-1})/out_{t-1}, absorb fq gq ==="
reghdfe flow us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store r1

display _newline _newline "=== R2: FLOW + firm x group FE (absorb fq gq ig) ==="
reghdfe flow us_cn us_cn_shock, absorb(fq gq ig) vce(cluster firm_n rd_m)
estimates store r2

display _newline _newline "=== R3: OLD dos (share-of-float change, F6-confounded) for comparison ==="
reghdfe dos us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store r3

capture which esttab
if _rc == 0 {
    esttab r1 r2 r3 using "`OUT'/ownership_share_results.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) ///
        keep(us_cn us_cn_shock) mtitles("flow_fqgq" "flow_firmXgroup" "dos_compare") nonumbers plain
    di "Wrote `OUT'/ownership_share_results.csv"
}

display _newline "=== b2 / b3 (b / se / p) ==="
foreach m in r1 r2 r3 {
    estimates restore `m'
    display _newline "--- `m' ---"
    estimates table, b(%12.4e) se(%12.4e) p(%6.4f)
}
display _newline "Done."
```

### `run_headline_3pairwise.do`

```stata
* run_headline_3pairwise.do — adopt the fully-saturated three-way pairwise FE
* (firm×quarter it + group×quarter gt + firm×group ig) as the HEADLINE, per the
* Khwaja-Mian / De Haas saturated design. Reports it+gt+ig (headline) alongside
* it+gt (comparison) for every main specification. The null must survive the
* most demanding FE. Two-way cluster (firm, quarter) throughout.

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

* helper: print b3/se/p/N for us_cn_shock under a given FE set + optional if
capture program drop _row
program define _row
    args lbl y FE ifc
    qui reghdfe `y' us_cn us_cn_shock `ifc', absorb(`FE') vce(cluster firm_n rd_m)
    display %-24s "`lbl'" %-11s "`FE'" "  b3=" %10.3e _b[us_cn_shock] ///
            "  p=" %6.4f 2*ttail(e(df_r), abs(_b[us_cn_shock]/_se[us_cn_shock])) ///
            "  N=" %9.0gc e(N)
end

*=======================================================================
* PART A — w-based specs on the audit panel
*=======================================================================
use "`OUT'/audit_c6_panel.dta", clear
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
gen us_cn        = us*cn_lag
gen us_cn_shock  = us*cn_lag*shock
gen us_cn_gpr    = us*cn_lag*gpr
gen us_cn_gprlag = us*cn_lag*gpr_lag

display _newline "===== 3-PAIRWISE HEADLINE (fq gq ig) vs it+gt (fq gq) ====="
_row "headline dw"      dw       "fq gq ig" ""
_row "headline dw"      dw       "fq gq"    ""
_row "F1a lead dw_t+1"  dw_lead1 "fq gq ig" ""
_row "F1a lead dw_t+1"  dw_lead1 "fq gq"    ""
_row "F1b LP cum1"      cum1     "fq gq ig" ""
_row "F1b LP cum1"      cum1     "fq gq"    ""
_row "F1b LP cum2"      cum2     "fq gq ig" ""
_row "F1b LP cum4"      cum4     "fq gq ig" ""
_row "F1b LP cum4"      cum4     "fq gq"    ""
_row "F2 in-span dw"    dw       "fq gq ig" "if in_span==1"
_row "F2 in-span dw"    dw       "fq gq"    "if in_span==1"

display _newline "===== F7 GPR two-interaction (3-pairwise vs it+gt) ====="
foreach FE in "fq gq ig" "fq gq" {
    qui reghdfe dw us_cn us_cn_gpr us_cn_gprlag, absorb(`FE') vce(cluster firm_n rd_m)
    display "`FE': b(GPR_t)=" %9.3e _b[us_cn_gpr] " p=" %6.4f 2*ttail(e(df_r),abs(_b[us_cn_gpr]/_se[us_cn_gpr])) ///
            "  b(GPR_t-1)=" %9.3e _b[us_cn_gprlag] " p=" %6.4f 2*ttail(e(df_r),abs(_b[us_cn_gprlag]/_se[us_cn_gprlag]))
}

*=======================================================================
* PART B — ownership FLOW (F6)
*=======================================================================
use "`OUT'/ownership_c6_panel.dta", clear
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen rd_m = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ig = group(firm_str hgroup)
gen us_cn       = us*cn_lag
gen us_cn_shock = us*cn_lag*shock
display _newline "===== ownership FLOW (F6), 3-pairwise vs it+gt ====="
_row "flow"  flow  "fq gq ig" ""
_row "flow"  flow  "fq gq"    ""
display _newline "Done."
```

### `run_audit_f1f2f7.do`

```stata
* run_audit_f1f2f7.do — remediation regressions for review findings F1, F2, F7.
* Panel: audit_c6_panel.dta. Same FE (firm#quarter, group#quarter) + two-way
* cluster (firm, quarter) as the main spec. Coefficients are on Delta w (raw;
* multiply by 1e6 to match the doc's x10^-6 convention).

clear all
set more off

local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/audit_c6_panel.dta"
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

use "`DTA'", clear

gen rd_day = dofc(rdate)
format rd_day %td
egen firm_n = group(firm_str)
gen  rd_m   = mofd(rd_day)
format rd_m %tm
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)

* interactions
gen us_cn        = us * cn_lag
gen us_cn_shock  = us * cn_lag * shock
gen us_cn_gpr    = us * cn_lag * gpr
gen us_cn_gprlag = us * cn_lag * gpr_lag
label var us_cn       "US x CN(t-1)"
label var us_cn_shock "US x CN(t-1) x S_t"

*==============================================================
* CHECK: reproduce the headline (contemporaneous dw ~ S_t) on this panel
*==============================================================
display _newline "===== CHECK: headline dw ~ US x CN x S_t (full grid) ====="
reghdfe dw us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)

*==============================================================
* F1a: LEAD-FLOW — Delta w_{t+1} ~ US x CN x S_t  (pure lagged shock, no look-ahead)
*==============================================================
display _newline "===== F1a: LEAD-FLOW dw_lead1 ~ US x CN x S_t (full grid) ====="
reghdfe dw_lead1 us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store f1a_full

display _newline "===== F1a (in-span only) ====="
reghdfe dw_lead1 us_cn us_cn_shock if in_span==1, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store f1a_span

*==============================================================
* F1b: LOCAL PROJECTION IRF — cum_h = w_{t+h} - w_{t-1} ~ US x CN x S_t, h=0..4
*==============================================================
display _newline "===== F1b: LOCAL PROJECTION IRF (b3 by horizon) ====="
capture postclose lp
postfile lp int h double b3 double se3 double p3 double b2 double p2 double nobs ///
    using "`OUT'/audit_f1_lp_irf.dta", replace
forvalues h = 0/4 {
    quietly reghdfe cum`h' us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
    local b3 = _b[us_cn_shock]
    local se3 = _se[us_cn_shock]
    local t3 = `b3'/`se3'
    local p3 = 2*ttail(e(df_r), abs(`t3'))
    local b2 = _b[us_cn]
    local se2 = _se[us_cn]
    local p2 = 2*ttail(e(df_r), abs(`b2'/`se2'))
    post lp (`h') (`b3') (`se3') (`p3') (`b2') (`p2') (e(N))
    display "  h=`h'  b3=" %9.3e `b3' "  se=" %9.3e `se3' "  p=" %6.4f `p3' "  N=" %9.0gc e(N)
}
postclose lp

*==============================================================
* F2: HEADLINE restricted to firm existence span (drop ~25% phantom zeros)
*==============================================================
display _newline "===== F2: headline dw ~ US x CN x S_t, IN-SPAN ONLY ====="
reghdfe dw us_cn us_cn_shock if in_span==1, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store f2_span

*==============================================================
* F7: replace the AR(1) shock with GPR level + lag (no generated regressor / look-ahead)
*==============================================================
display _newline "===== F7: dw ~ US x CN x GPR_t + US x CN x GPR_{t-1} (full grid) ====="
reghdfe dw us_cn us_cn_gpr us_cn_gprlag, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store f7_full

display _newline "===== F7 (in-span) ====="
reghdfe dw us_cn us_cn_gpr us_cn_gprlag if in_span==1, absorb(fq gq) vce(cluster firm_n rd_m)
estimates store f7_span

*==============================================================
* Export summary
*==============================================================
capture which esttab
if _rc == 0 {
    esttab f1a_full f1a_span f2_span f7_full f7_span using "`OUT'/audit_f1f2f7_results.csv", replace ///
        cells("b(fmt(%9.3e)) se(fmt(%9.3e)) p(fmt(4))") ///
        stats(N r2, fmt(%9.0gc %6.4f)) ///
        mtitles("F1a_lead_full" "F1a_lead_span" "F2_dw_span" "F7_gpr_full" "F7_gpr_span") ///
        nonumbers plain
    di "Wrote `OUT'/audit_f1f2f7_results.csv"
}
display _newline "Done."
```

### `run_audit_f1b_robust.do`

```stata
* run_audit_f1b_robust.do — robust inference for the F1b local-projection
* horizons significant under CRVE (h=1 p=0.013, h=4 p=0.044).
* Overlapping cumulative windows + only 82 quarter-clusters => CRVE SE likely
* understated. Re-test with wild cluster bootstrap (Webb weights), bootstrapping
* on the QUARTER cluster (the few-cluster / serial-overlap dimension), keeping the
* same two-way (firm, quarter) error clustering and the same firm#quarter +
* group#quarter FE.

clear all
set more off
local DTA "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output/audit_c6_panel.dta"
use "`DTA'", clear

gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen  rd_m   = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock

capture which boottest
if _rc ssc install boottest, replace

* absorb ONLY firm#quarter (fq); include group#quarter as explicit i.us#i.rd_m
* dummies so boottest works (it rejects >1 absorbed FE set).
foreach h in 1 2 4 {
    display _newline "===== LP h=`h' : CRVE vs wild-cluster bootstrap ====="
    reghdfe cum`h' us_cn us_cn_shock i.us#i.rd_m, absorb(fq) vce(cluster firm_n rd_m)
    local crve_p = 2*ttail(e(df_r), abs(_b[us_cn_shock]/_se[us_cn_shock]))
    display "  CRVE:  b3=" %9.3e _b[us_cn_shock] "  se=" %9.3e _se[us_cn_shock] "  p=" %6.4f `crve_p'
    * wild cluster bootstrap, Webb weights, bootstrap on quarter (few-cluster / overlap dim)
    boottest us_cn_shock, weighttype(webb) reps(9999) bootcluster(rd_m) nograph
    display "  WCB (Webb, bootcluster=quarter, 9999 reps): p=" %6.4f r(p)
    matrix ci = r(CI)
    display "     95% CI: " %9.3e ci[1,1] "  to  " %9.3e ci[1,2]
}
display _newline "Done."
```

### `run_audit_f4f8.do`

```stata
* run_audit_f4f8.do — review fixes F8 (risk-set lag-only) and F4 (country-pair
* β₁ clustering level).

clear all
set more off
local OUT "c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output"

*==============================================================
* F8: risk-set with LAG-ONLY membership (no t+1 look-ahead). Compare β₃ to the
* with-lead risk set headline.
*==============================================================
use "`OUT'/c6_panel_riskset_lagonly.dta", clear
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen  rd_m   = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
gen us_cn       = us * cn_lag
gen us_cn_shock = us * cn_lag * shock
display _newline "===== F8: risk-set LAG-ONLY membership (dw ~ US x CN x S_t) ====="
reghdfe dw us_cn us_cn_shock, absorb(fq gq) vce(cluster firm_n rd_m)
display "  b3=" %9.3e _b[us_cn_shock] "  se=" %9.3e _se[us_cn_shock] ///
        "  p=" %6.4f 2*ttail(e(df_r), abs(_b[us_cn_shock]/_se[us_cn_shock]))
di "  (with-lead risk set 07g R1 b3 was ~ +1.28e-6-scale null; check stability)"

*==============================================================
* F4: country-pair β₁ (US x S_c) under three clustering levels. S_c varies only
* across GB/DE/FR x quarter, so (firm, quarter) clustering ignores within-country
* serial correlation with only ~3 country units.
*==============================================================
use "`OUT'/c6_panel_country_pair.dta", clear
drop if missing(shock_c)
gen rd_day = dofc(rdate)
egen firm_n = group(firm_str)
gen  rd_m   = mofd(rd_day)
egen fq = group(firm_str rd_day)
egen gq = group(hgroup rd_day)
egen ctry_n = group(sec_country)
egen ctryq  = group(sec_country rd_day)

gen us_sc      = us * shock_c
gen us_cn      = us * cn_lag
gen us_cn_sc   = us * cn_lag * shock_c
label var us_sc    "US x S_c (β1)"
label var us_cn    "US x CN (β2)"
label var us_cn_sc "US x CN x S_c (β3)"

tab sec_country
display _newline "===== F4a: country-pair, cluster(firm_n rd_m) [as in doc] ====="
reghdfe dw us_sc us_cn us_cn_sc, absorb(fq gq) vce(cluster firm_n rd_m)
display "  b1(US x S_c)=" %9.3e _b[us_sc] "  se=" %9.3e _se[us_sc] ///
        "  p=" %6.4f 2*ttail(e(df_r), abs(_b[us_sc]/_se[us_sc]))

display _newline "===== F4b: country-pair, cluster(sec_country) [3 clusters, honest level] ====="
reghdfe dw us_sc us_cn us_cn_sc, absorb(fq gq) vce(cluster ctry_n)
display "  b1(US x S_c)=" %9.3e _b[us_sc] "  se=" %9.3e _se[us_sc] ///
        "  p=" %6.4f 2*ttail(e(df_r), abs(_b[us_sc]/_se[us_sc])) "  (df_r=" e(df_r) ")"

display _newline "===== F4c: country-pair, cluster(sec_country rd_m) [country x quarter] ====="
reghdfe dw us_sc us_cn us_cn_sc, absorb(fq gq) vce(cluster ctry_n rd_m)
display "  b1(US x S_c)=" %9.3e _b[us_sc] "  se=" %9.3e _se[us_sc] ///
        "  p=" %6.4f 2*ttail(e(df_r), abs(_b[us_sc]/_se[us_sc]))
display _newline "Done."
```


---

## 15. Python build scripts (verbatim)
`build_spell_boundary.py` superseded (§7.4), omitted. Second-review-round additions: `build_audit_panel_f1f2f7.py` (lead/cumulative/in-span/GPR panel, F1/F2/F7), `run_randomization_inference.py` (design-based RI, exact pairwise-difference algebra, F3/F9), `run_ri_3pairwise.py` (two-way-FE RI for the 3-pairwise headline), `build_riskset_lagonly.py` (F8 lag-only membership).

### `build_c6_panel.py`

```python
"""
build_c6_panel.py

Build the Stata-ready C6 firm-quarter-holdergroup panel for 07_regression.do
from the rebuilt merged_us_eu_zero_filled.parquet (now carrying BACKWARD Δw:
delta_w = w_t - w_{t-1}, per slide 4 formula).

Columns written (matches what 07_regression.do `use`s):
  firm_str  (= sec_entity_id, string)
  hgroup    (= holder_group: 'US' / 'NONUS')
  rdate     (= report_date, datetime64 -> Stata %tc)
  dw        (= delta_w)
  cn_lag    (= china_share_lag1q)
  shock     (= shock_us_cn)
  us        (= 1 if holder_group == 'US' else 0)

Sample filter:
  delta_w IS NOT NULL AND china_share_lag1q IS NOT NULL AND shock_us_cn IS NOT NULL

Single duckdb SELECT projects + filters in one shot — no separate scans, no
row-order misalignment.

Stata gotcha: pandas.to_stata writes datetime64 as %tc (milliseconds since
1960-01-01); 07_regression.do already does `gen rd_day = dofc(rdate)`.
"""

from pathlib import Path

import duckdb
import pandas as pd

PROJ_DIR = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT_DIR  = PROJ_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_PARQUET = OUT_DIR / "merged_us_eu_zero_filled.parquet"
PANEL_DTA     = OUT_DIR / "c6_panel.dta"

assert PANEL_PARQUET.exists(), (
    f"Missing input parquet: {PANEL_PARQUET}\n"
    f"Re-run julia_descriptive/06_cartesian_grid.jl first."
)

print(f"[1/3] Reading {PANEL_PARQUET}")
panel_uri = str(PANEL_PARQUET).replace("\\", "/")

con = duckdb.connect()
df = con.execute(f"""
    SELECT
        CAST(sec_entity_id AS VARCHAR)                  AS firm_str,
        holder_group                                    AS hgroup,
        CAST(report_date AS TIMESTAMP)                  AS rdate,
        delta_w                                         AS dw,
        china_share_lag1q                               AS cn_lag,
        shock_us_cn                                     AS shock,
        CASE WHEN holder_group = 'US' THEN 1 ELSE 0 END AS us
    FROM read_parquet('{panel_uri}')
    WHERE delta_w           IS NOT NULL
      AND china_share_lag1q IS NOT NULL
      AND shock_us_cn       IS NOT NULL
""").df()

# Upstream integrity on the FULL grid (external-review P2): portfolio weights
# must sum to 1 within each (group, quarter). Checked on the full parquet, NOT
# the filtered estimation sample above — filtering drops rows and would break
# the simplex adding-up. (Observed max abs deviation ~4.4e-16 = float noise.)
# Simplex integrity on the FULL grid (external-review P2), NULL-aware.
# portfolio_weight_eu = w = H/T is NULL by construction (06 ELSE NULL) when a
# group's European book is empty that quarter (T_{g,t}=0) — e.g. NONUS holds
# nothing in 1999Q1, I_ict all 0. Such all-NULL cells are LEGITIMATE and allowed.
# The two things that WOULD be bugs, and are hard-failed here:
#   (i)  a cell with positive holdings (SUM(I_ict)>0) but all-NULL weights;
#   (ii) any cell that DOES carry weights whose sum is not 1.
_wchk = con.execute(f"""
    WITH cell AS (
        SELECT holder_group, report_date,
               SUM(portfolio_weight_eu)   AS s,
               COUNT(portfolio_weight_eu) AS nn,
               SUM(COALESCE(I_ict, 0))    AS tot_hold
        FROM read_parquet('{panel_uri}')
        GROUP BY 1, 2)
    SELECT
        MAX(CASE WHEN nn > 0 THEN ABS(s - 1) END)          AS max_dev,
        COUNT(CASE WHEN nn > 0 THEN 1 END)                 AS n_checked,
        COUNT(CASE WHEN nn = 0 AND tot_hold > 0 THEN 1 END) AS n_held_but_null
    FROM cell
""").df()
_wdev = _wchk["max_dev"].iloc[0]
_nchk = int(_wchk["n_checked"].iloc[0])
_nbug = int(_wchk["n_held_but_null"].iloc[0])
assert _nbug == 0, \
    f"{_nbug} (group, quarter) cell(s) have positive holdings but all-NULL weights — 06 weight bug"
assert _wdev < 1e-9, \
    f"weights do not sum to 1 by (group, quarter): max abs dev {_wdev:.2e} over {_nchk} non-empty cells"
con.close()

df["firm_str"] = df["firm_str"].astype(str)
df["hgroup"]   = df["hgroup"].astype(str)
df["rdate"]    = pd.to_datetime(df["rdate"])
df["dw"]       = pd.to_numeric(df["dw"],     errors="raise").astype("float64")
df["cn_lag"]   = pd.to_numeric(df["cn_lag"], errors="raise").astype("float64")
df["shock"]    = pd.to_numeric(df["shock"],  errors="raise").astype("float64")
df["us"]       = df["us"].astype("int8")

n = len(df)
print(f"[2/3] Filtered panel rows: {n:,}")
print("       Breakdown by hgroup:")
print(df["hgroup"].value_counts().to_string())
print("       us flag tab:")
print(df["us"].value_counts().to_string())
print(f"       rdate range: {df['rdate'].min()} -> {df['rdate'].max()}")
print(f"       n unique firm_str: {df['firm_str'].nunique():,}")

# Sanity assertions (hardened per external review — hard fail, not print)
assert n > 0, "Empty panel after filter — check upstream parquet."
assert df["dw"].notna().all(),     "dw has NaN after filter"
assert df["cn_lag"].notna().all(), "cn_lag has NaN after filter"
assert df["shock"].notna().all(),  "shock has NaN after filter"
assert set(df["hgroup"].unique()) <= {"US", "NONUS"}, (
    f"Unexpected hgroup values: {df['hgroup'].unique()}"
)
assert ((df["hgroup"] == "US") == (df["us"] == 1)).all(), "us flag mismatch with hgroup"
# (a) unique key
assert not df.duplicated(["firm_str", "hgroup", "rdate"]).any(), \
    "duplicate (firm_str, hgroup, rdate) rows"
# (b) every firm-quarter paired US + NONUS (within-firm-quarter identification needs both)
_pair = df.groupby(["firm_str", "rdate"])["hgroup"].nunique()
assert (_pair == 2).all(), \
    f"panel not fully paired: {(_pair != 2).sum():,} firm-quarters lack both US and NONUS"
# (c) shock is one common value per quarter (no within-quarter variation)
assert (df.groupby("rdate")["shock"].nunique() == 1).all(), \
    "shock varies within a quarter — expected a single common S_t per quarter"
# (d) cn_lag in [0, 1] (it is a share)
assert df["cn_lag"].between(0, 1).all(), "cn_lag outside [0,1]"
# (e) quarter coverage is contiguous — no missing quarters in the estimation span
_q = df["rdate"].dt.to_period("Q").drop_duplicates().sort_values()
_expected = pd.period_range(_q.iloc[0], _q.iloc[-1], freq="Q")
assert len(_q) == len(_expected) and (_q.to_numpy() == _expected.to_numpy()).all(), (
    f"gap in quarter coverage: {len(_q)} distinct quarters vs {len(_expected)} expected "
    f"between {_q.iloc[0]} and {_q.iloc[-1]}"
)
print(f"       quarter coverage: {len(_q)} contiguous quarters "
      f"{_q.iloc[0]} -> {_q.iloc[-1]}")

print(f"[3/3] Writing Stata file: {PANEL_DTA}")
df.to_stata(
    PANEL_DTA,
    write_index=False,
    convert_dates={"rdate": "tc"},
    version=118,
)
print(f"      done. {n:,} rows written.")
```

### `build_country_pair_shock.py`

```python
"""
build_country_pair_shock.py

Robustness build for `07b_country_pair_robustness.do`.

Replaces the single USA-China AI-GPR shock used in the main spec with a
COUNTRY-PAIR-SPECIFIC shock S_{c,t} computed on each EU-listing country's
own AI-GPR series vs China. Three EU countries covered: GB (UK|China),
DE (Germany|China), FR (France|China). PT excluded (single direction,
magnitude ~0, only 79 firms).

Two artifacts written:
  1. output/country_pair_shocks_monthly.csv  -- diagnostic monthly series
  2. output/c6_panel_country_pair.dta        -- Stata-ready firm-qtr-group panel
     restricted to sec_country IN ('GB','DE','FR') with `shock_c` (country-pair)
     and `shock_us_cn` (original) on identical observations.

AR(1) recipe = byte-for-byte copy of 05_combine_visualize.jl L115-126:
plain OLS on non-missing monthly series, residual at quarter-end month.

REVIEW FIXES applied (vs first draft):
  - month_end via dt.to_period(M).to_timestamp(M) (the np.timedelta64('M')
    arithmetic crashes at runtime)
  - us indicator pulled in the SAME duckdb SELECT as the panel (avoids row-
    order misalignment across two separate parquet scans)
  - shock column kept as `shock_us_cn` (matches main panel naming + do-file ref)
  - assert shock_long key uniqueness before merge
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb

PROJ_DIR = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT_DIR  = PROJ_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GPR_CSV       = Path(r"E:/Data/Data/ai_gpr_bilateral_monthly.csv")
PANEL_PARQUET = OUT_DIR / "merged_us_eu_zero_filled.parquet"

SHOCKS_CSV  = OUT_DIR / "country_pair_shocks_monthly.csv"
COEFS_CSV   = OUT_DIR / "country_pair_shocks_coefficients.csv"
PANEL_DTA   = OUT_DIR / "c6_panel_country_pair.dta"

# ISO2 listing country -> AI-GPR CSV column. Direction = first-country-perspective
# matching the main USA|China spec.
COUNTRY_PAIR = {
    "GB": "UK|China",
    "DE": "Germany|China",
    "FR": "France|China",
}

# ---------------------------------------------------------------
# 1. Load raw monthly AI-GPR.
# ---------------------------------------------------------------
print(f"[1/6] Reading raw GPR CSV: {GPR_CSV}")
need_cols = ["Date", "USA|China"] + list(COUNTRY_PAIR.values())
gpr_raw = pd.read_csv(GPR_CSV, usecols=need_cols)
gpr_raw["Date"] = pd.to_datetime(gpr_raw["Date"])
# Use to_period for safe month-end arithmetic (np.timedelta64('M') is rejected
# by recent numpy as "ambiguous duration").
gpr_raw["month_first"] = gpr_raw["Date"].dt.to_period("M").dt.to_timestamp()
gpr_raw["month_end"]   = gpr_raw["Date"].dt.to_period("M").dt.to_timestamp("M")
gpr_raw = gpr_raw.sort_values("month_first").reset_index(drop=True)
print(f"      {len(gpr_raw)} months, "
      f"{gpr_raw['month_first'].min().date()} -> "
      f"{gpr_raw['month_first'].max().date()}")

# ---------------------------------------------------------------
# 2. AR(1) per series. Recipe matches julia 05_combine_visualize.jl L115-126.
# ---------------------------------------------------------------
def fit_ar1(series: pd.Series) -> dict:
    g = series.dropna().to_numpy(dtype=float)
    if len(g) < 3:
        raise ValueError("AR(1) fit needs >=3 non-missing months")
    y  = g[1:]
    yL = g[:-1]
    yL_mean = yL.mean()
    y_mean  = y.mean()
    b_hat = ((yL - yL_mean) * (y - y_mean)).sum() / ((yL - yL_mean) ** 2).sum()
    a_hat = y_mean - b_hat * yL_mean
    resid = y - (a_hat + b_hat * yL)
    aligned = np.concatenate([[np.nan], resid])
    return {"a": a_hat, "b": b_hat, "n": int(len(g)), "resid_aligned": aligned}

# Sanity: interior NaNs would silently shift residuals onto wrong months.
for col in ["USA|China"] + list(COUNTRY_PAIR.values()):
    s = gpr_raw[col]
    nn = s.notna()
    if nn.any():
        first = nn.idxmax()
        if s.iloc[first:].isna().any():
            raise ValueError(f"Interior NaN in '{col}' after {gpr_raw['month_first'].iloc[first].date()}")

print("[2/6] Fitting AR(1) per series (USA|China + 3 country-pairs)")
diag_rows = []
wide_shocks = gpr_raw[["month_first", "month_end"]].copy()

# US baseline first (sanity check vs main spec).
us_fit = fit_ar1(gpr_raw["USA|China"])
us_full = np.full(len(gpr_raw), np.nan)
us_full[gpr_raw["USA|China"].notna().to_numpy().nonzero()[0]] = us_fit["resid_aligned"]
wide_shocks["gpr_us_cn"]   = gpr_raw["USA|China"].to_numpy()
wide_shocks["shock_us_cn"] = us_full
diag_rows.append({"sec_country": "US", "ai_gpr_column": "USA|China",
                  "a_hat": us_fit["a"], "b_hat": us_fit["b"], "n_months": us_fit["n"]})
print(f"      US baseline (USA|China): a={us_fit['a']:.4f}  b={us_fit['b']:.4f}  n={us_fit['n']}")

for ctry_iso, col in COUNTRY_PAIR.items():
    fit = fit_ar1(gpr_raw[col])
    full_resid = np.full(len(gpr_raw), np.nan)
    full_resid[gpr_raw[col].notna().to_numpy().nonzero()[0]] = fit["resid_aligned"]
    wide_shocks[f"gpr_{ctry_iso.lower()}_cn"]   = gpr_raw[col].to_numpy()
    wide_shocks[f"shock_{ctry_iso.lower()}_cn"] = full_resid
    diag_rows.append({"sec_country": ctry_iso, "ai_gpr_column": col,
                      "a_hat": fit["a"], "b_hat": fit["b"], "n_months": fit["n"]})
    print(f"      {ctry_iso} ({col}): a={fit['a']:.4f}  b={fit['b']:.4f}  n={fit['n']}")

wide_shocks["is_quarter_end"] = wide_shocks["month_end"].dt.month.isin([3, 6, 9, 12])

# ---------------------------------------------------------------
# 3. Write diagnostics.
# ---------------------------------------------------------------
print(f"[3/6] Writing diagnostics")
pd.DataFrame(diag_rows).to_csv(COEFS_CSV, index=False)
wide_shocks.to_csv(SHOCKS_CSV, index=False)
print(f"      coefficients -> {COEFS_CSV.name}")
print(f"      monthly      -> {SHOCKS_CSV.name}")

# ---------------------------------------------------------------
# 4. Build the long-format quarter-end shock lookup.
# ---------------------------------------------------------------
qe = wide_shocks.loc[wide_shocks["is_quarter_end"]].copy()
shock_long_rows = []
for ctry_iso in COUNTRY_PAIR:
    col = f"shock_{ctry_iso.lower()}_cn"
    sub = qe[["month_end", col]].rename(columns={col: "shock_c"})
    sub["sec_country"] = ctry_iso
    shock_long_rows.append(sub)
shock_long = pd.concat(shock_long_rows, ignore_index=True)
shock_long = shock_long.rename(columns={"month_end": "quarter_end"})
shock_long["quarter_end"] = pd.to_datetime(shock_long["quarter_end"])

# Belt-and-braces key uniqueness check
assert not shock_long.duplicated(["sec_country", "quarter_end"]).any(), \
    "duplicate (sec_country, quarter_end) key in shock_long"
print(f"[4/6] Country-pair quarter-end shock rows: {len(shock_long)}  "
      f"({shock_long['sec_country'].nunique()} countries x ~"
      f"{shock_long['quarter_end'].nunique()} quarters)")

# ---------------------------------------------------------------
# 5. Load panel + us indicator in ONE duckdb query (avoids row-order
#    misalignment that would happen with two separate parquet scans).
# ---------------------------------------------------------------
print(f"[5/6] Reading panel parquet (single SELECT) -> {PANEL_PARQUET}")
con = duckdb.connect(":memory:")
con.execute("SET memory_limit='6GB'")
panel = con.execute(f"""
    SELECT sec_entity_id           AS firm_str,
           sec_country,
           holder_group             AS hgroup,
           report_date              AS rdate,
           I_ict,
           portfolio_weight_eu      AS w,
           w_prev,
           delta_w                  AS dw,
           china_share              AS cn_share,
           china_share_lag1q        AS cn_lag,
           gpr_us_cn,
           shock_us_cn,
           CASE WHEN investor_country = 'US' THEN 1 ELSE 0 END AS us
    FROM read_parquet('{PANEL_PARQUET.as_posix()}')
    WHERE sec_country IN ('GB','DE','FR')
""").df()
print(f"      panel rows (GB+DE+FR): {len(panel):,}")
print(panel["sec_country"].value_counts().to_string())

# ---------------------------------------------------------------
# 6. Join the country-pair shock by (sec_country, quarter_end).
# ---------------------------------------------------------------
panel["rdate"] = pd.to_datetime(panel["rdate"])
panel["rdate_qe"] = (panel["rdate"].dt.to_period("M")
                                   .dt.to_timestamp("M")
                                   .dt.normalize())
shock_long["quarter_end"] = shock_long["quarter_end"].dt.normalize()

before = len(panel)
panel = panel.merge(
    shock_long.rename(columns={"quarter_end": "rdate_qe"}),
    on=["sec_country", "rdate_qe"],
    how="left",
    validate="m:1",
)
assert len(panel) == before, "merge changed row count"
unmatched = panel["shock_c"].isna().sum()
print(f"[6/6] Joined shock_c. Unmatched (NaN shock_c): {unmatched:,} / {len(panel):,}")
panel = panel.drop(columns=["rdate_qe"])

# Final column order. shock_us_cn kept as-is (matches main panel naming).
out_cols = ["firm_str", "sec_country", "hgroup", "rdate",
            "w", "dw", "cn_share", "cn_lag", "I_ict",
            "gpr_us_cn", "shock_us_cn", "shock_c", "us"]
panel = panel[out_cols]

print(f"      writing Stata file -> {PANEL_DTA}")
panel.to_stata(
    PANEL_DTA,
    write_index=False,
    variable_labels={
        "firm_str":    "FactSet sec_entity_id",
        "sec_country": "EU listing country (GB/DE/FR)",
        "hgroup":      "Holder group",
        "rdate":       "Report date (quarter-end, %tc)",
        "w":           "Portfolio weight (EU)",
        "dw":          "Delta w",
        "cn_share":    "China share",
        "cn_lag":      "China share, lag 1q",
        "I_ict":       "Group holding value, USD (I_{i,c,t})",
        "gpr_us_cn":   "AI-GPR US-CN level",
        "shock_us_cn": "AR(1) residual on USA|China (main shock)",
        "shock_c":     "AR(1) residual on country|China (robustness)",
        "us":          "US investor indicator",
    },
    convert_dates={"rdate": "tc"},
)
print("Done.")
print(f"  Panel rows written: {len(panel):,}")
print(f"  Country breakdown:\n{panel['sec_country'].value_counts().to_string()}")
print(f"  shock_c missing: {panel['shock_c'].isna().sum():,} / {len(panel):,}")
print(f"  shock_us_cn missing: {panel['shock_us_cn'].isna().sum():,} / {len(panel):,}")
```

### `build_spell_riskset.py`

```python
"""
build_spell_riskset.py

CORRECTED spell-boundary sample (fixes build_spell_boundary.py, which selected
per-(firm,group) and broke the US-vs-NONUS pairing).

Risk set is defined at the FIRM-QUARTER level:
  A firm-quarter (i,t) enters the risk set if EITHER group (US or NONUS) has a
  spell-boundary row there, i.e. is held at t, or held at t-1, or held at t+1.
Once (i,t) is in the risk set, KEEP BOTH group rows -- (i,US,t) and (i,NONUS,t) --
even if one side is a boundary zero or a deep zero. This preserves the within-
firm-quarter US-vs-NONUS comparison that alpha_{i,t} identifies off.

delta_w reused from the full-grid parquet (correct full-grid lag).
Output: output/c6_panel_riskset.dta
"""

from pathlib import Path
import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT  = PROJ / "output"
PARQUET = OUT / "merged_us_eu_zero_filled.parquet"
DTA = OUT / "c6_panel_riskset.dta"
uri = str(PARQUET).replace("\\", "/")

con = duckdb.connect()

print("[1/4] Flagging held + neighbors per (firm, group)...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE flagged AS
SELECT
    sec_entity_id, sec_country, holder_group, investor_country,
    report_date, I_ict, portfolio_weight_eu, delta_w,
    china_share_lag1q, shock_us_cn,
    CAST(I_ict > 0 AS INTEGER) AS held,
    COALESCE(LAG(CAST(I_ict > 0 AS INTEGER))  OVER w, 0) AS held_lag,
    COALESCE(LEAD(CAST(I_ict > 0 AS INTEGER)) OVER w, 0) AS held_lead
FROM read_parquet('{uri}')
WINDOW w AS (PARTITION BY sec_entity_id, holder_group ORDER BY report_date)
""")

print("[2/4] Building FIRM-QUARTER risk set (union over groups)...")
con.execute("""
CREATE OR REPLACE TEMP TABLE riskset AS
SELECT *,
    CASE WHEN held=1 OR held_lag=1 OR held_lead=1 THEN 1 ELSE 0 END AS sb,
    MAX(CASE WHEN held=1 OR held_lag=1 OR held_lead=1 THEN 1 ELSE 0 END)
        OVER (PARTITION BY sec_entity_id, report_date) AS risk
FROM flagged
""")

print("[3/4] Keeping BOTH group rows for risk-set firm-quarters...")
df = con.execute("""
SELECT
    CAST(sec_entity_id AS VARCHAR)                  AS firm_str,
    sec_country,
    holder_group                                    AS hgroup,
    CAST(report_date AS TIMESTAMP)                  AS rdate,
    delta_w                                         AS dw,
    china_share_lag1q                               AS cn_lag,
    shock_us_cn                                     AS shock,
    CASE WHEN holder_group = 'US' THEN 1 ELSE 0 END AS us,
    held, sb,
    CASE WHEN held=1 THEN 'held'
         WHEN sb=1  THEN 'boundary_zero'
         ELSE 'deep_zero_paired' END               AS row_type
FROM riskset
WHERE risk = 1
""").df()

n_all = len(df)
print(f"      risk-set rows (before dropping NULLs): {n_all:,}")
print(f"      hgroup split (should be balanced):\n{df['hgroup'].value_counts().to_string()}")
print(f"      row type:\n{df['row_type'].value_counts().to_string()}")
# verify pairing: every (firm, quarter) has exactly 2 rows
pair = df.groupby(['firm_str','rdate']).size()
print(f"      firm-quarters with exactly 2 rows: {(pair==2).sum():,} / {len(pair):,}"
      f"  (unpaired: {(pair!=2).sum():,})")
# hard fail: risk-set MUST keep both group rows per firm-quarter (external-review fix)
assert (pair == 2).all(), \
    f"risk-set not fully paired before drop: {(pair != 2).sum():,} unpaired firm-quarters"

print("[4/4] Estimation subset: drop missing dw / cn_lag / shock...")
est = df.dropna(subset=["dw", "cn_lag", "shock"]).copy()
n_est = len(est)
print(f"      estimation rows: {n_est:,}  (dropped {n_all - n_est:,})")
print(f"      hgroup split:\n{est['hgroup'].value_counts().to_string()}")
pair2 = est.groupby(['firm_str','rdate']).size()
print(f"      paired firm-quarters after drop: {(pair2==2).sum():,} / {len(pair2):,}"
      f"  (unpaired singletons: {(pair2!=2).sum():,})")
print(f"      unique firms: {est['firm_str'].nunique():,}")
print(f"      rdate range: {est['rdate'].min()} -> {est['rdate'].max()}")

est["firm_str"] = est["firm_str"].astype(str)
est["hgroup"]   = est["hgroup"].astype(str)
est["rdate"]    = pd.to_datetime(est["rdate"])
for c in ["dw", "cn_lag", "shock"]:
    est[c] = pd.to_numeric(est[c], errors="raise").astype("float64")
est["us"] = est["us"].astype("int8")

assert n_est > 0
assert ((est["hgroup"] == "US") == (est["us"] == 1)).all()
# hard asserts (external-review fix): dedup + pairing + shock uniqueness + cn_lag range
assert not est.duplicated(["firm_str", "hgroup", "rdate"]).any(), "duplicate rows in estimation subset"
_p2 = est.groupby(["firm_str", "rdate"])["hgroup"].nunique()
assert (_p2 == 2).all(), \
    f"estimation subset not fully paired: {(_p2 != 2).sum():,} unpaired firm-quarters"
assert (est.groupby("rdate")["shock"].nunique() == 1).all(), "shock varies within a quarter"
assert est["cn_lag"].between(0, 1).all(), "cn_lag outside [0,1]"

keep_cols = ["firm_str", "hgroup", "rdate", "dw", "cn_lag", "shock", "us"]
est[keep_cols].to_stata(DTA, write_index=False, convert_dates={"rdate": "tc"}, version=118)
print(f"      wrote {DTA}  ({n_est:,} rows)")
```

### `build_ownership_share_panel.py`

```python
"""
build_ownership_share_panel.py  —  Essay 2, task #5 (shares-based outcome), STEP 1-2 only.

Motivation. The main outcome is a MARKET-VALUE portfolio weight w = H(USD)/T(USD),
which mixes trading flow with price changes and portfolio-denominator reallocation.
To support a "US investors do not SELL / do not reduce their stake" reading (not just
"do not reduce portfolio weight"), we need a pure QUANTITY measure. Shares are not
additive across firms (different prices/units), so a shares-based portfolio weight is
meaningless. The right object is the group's OWNERSHIP SHARE of each firm:

    ownership_share_{i,g,t} = ( sum_{b in g} adj_holding_{b,i,t} ) / adj_shares_out_{i,t}

= fraction of firm i's float held by group g. It is immune to price (numerator and
denominator are both in shares) and to portfolio-denominator reallocation (no T term).
Delta(ownership_share) is net buying/selling of firm i by group g as a share of float.

--- MULTI-AGENT REVIEW FIX (must-fix, review wsu1zupkc) ---
A `sec_entity_id` maps to MULTIPLE `fsym_id` security classes (primary equity, other
classes, ADR/GDR). Each class has its OWN adj_shares_out and its OWN adj_holding, in
DIFFERENT units, so pooling them under one sec_entity_id is unit-inconsistent for a
SHARES measure. The first draft pooled all classes and papered over the resulting
"dual-universe" shares_out with MODE(); verified that this left 19,771 sec-quarters
(5.28%) with >1 distinct shares_out (up to 1e12x) and inflated ownership_share ~2.6x on
the 6.7% of dispersed cells. Restricting to `fsym_id = fsym_primary_id` (the primary
equity class, exactly as the reference market-cap in 04_us_ownership_european.jl:171-183)
drives within-(sec,quarter) shares_out dispersion to EXACTLY ZERO, so we no longer need
MODE — adj_shares_out is a constant and AVG() returns it. We further restrict to
issue_type='EQ' (the primary listing float; the market-cap reference is EQ-only), so
numerator and denominator live in the same well-defined universe.

--- LIMITATION this creates (documented, quantified) ---
Restricting to the primary EQ class DROPS holdings on non-primary classes and on ADR/GDR
(AD). Measured on the EQ-primary measure this build uses (the `frac_*_on_primary_eq`
diagnostics): the fraction of European holdings USD on the primary EQ class is 96.68% for
NONUS but only 91.08% for US — i.e. US holds ~8.92% of its European exposure off the
primary class (ADR channel) vs 3.32% for NONUS. This ASYMMETRY matters: if US
investors adjust via ADRs, the primary-class ownership-share test will not see it, which
could bias the "do US sell" test toward a null. The USD portfolio-weight main spec DOES
capture the ADR channel, so the two outcomes are complementary. An ADR-inclusive
ownership-share robustness would need the ADR conversion ratio and is deferred.

This script does STEP 1-2 ONLY: aggregate to (firm, group, quarter), attach the
primary-class float, compute ownership_share, run integrity diagnostics (incl. a
COMBINED US+NONUS > 1 check and a by-country missingness breakdown), and stop. It does
NOT build the Cartesian grid, zero-fill, difference, or run any regression — those are
step 3, and the entry/exit zero-fill convention (mirroring build_c6_panel.py) is a
step-3 design item flagged by the review.

Outputs:
  output/ownership_share_observed.parquet         (observed firm-group-quarter cells)
  output/ownership_share_diagnostics.csv          (one-row summary of data quality)
  output/ownership_share_missing_by_country.csv   (residual shrout missingness by country)
"""

from pathlib import Path
import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
EOM = (OUT / "holdings_eom.parquet").as_posix()
OBS_PARQUET = OUT / "ownership_share_observed.parquet"
FLOAT_PARQUET = OUT / "ownership_share_float.parquet"   # shares_out per (firm, quarter), for step-3 zero-fill
DIAG_CSV = OUT / "ownership_share_diagnostics.csv"
MISS_CSV = OUT / "ownership_share_missing_by_country.csv"

# 28 European listing jurisdictions — MUST match 00_setup.jl EU_COUNTRIES exactly.
EU = ("GB", "DE", "FR", "NL", "CH", "IT", "ES", "SE", "DK", "NO", "FI", "BE",
      "AT", "IE", "LU", "PT", "PL", "CZ", "HU", "GR", "RO", "SK", "SI", "BG",
      "HR", "EE", "LV", "LT")
EU_SQL = "(" + ",".join(f"'{c}'" for c in EU) + ")"

# Primary EQ class only: matches the reference market-cap float rule and makes
# shares_out constant within each security-quarter (numerator + denominator same universe).
BASE_WHERE = f"issue_type = 'EQ' AND sec_country IN {EU_SQL} AND fsym_id = fsym_primary_id"

con = duckdb.connect()
con.execute("SET memory_limit='6GB'")

print("[1/5] Primary-class float per (sec, quarter) — shares_out now constant, AVG()...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE shr AS
SELECT sec_entity_id,
       report_date,
       AVG(adj_shares_out)            AS shares_out,
       COUNT(DISTINCT adj_shares_out) AS n_distinct_shrout,
       COUNT(*)                       AS n_holder_rows
FROM read_parquet('{EOM}')
WHERE {BASE_WHERE}
  AND adj_shares_out IS NOT NULL AND adj_shares_out > 0
GROUP BY 1, 2
""")

# Integrity: after the primary-class filter, shares_out must be a within-cell constant.
_disp = con.execute("SELECT COUNT(*) FROM shr WHERE n_distinct_shrout > 1").fetchone()[0]
assert _disp == 0, f"{_disp} (sec, quarter) cells still have dispersed shares_out after fsym=primary — investigate"

# Persist the primary-EQ float per (firm, quarter) so step 3 can zero-fill the grid:
# a grid cell where a group holds nothing but the firm HAS a valid float -> ownership 0;
# a firm-quarter with NO primary-EQ float -> ownership NULL (excluded). The distinction
# needs this table, not just the observed (held>0) cells.
con.execute(f"COPY (SELECT sec_entity_id, report_date, shares_out FROM shr) "
            f"TO '{FLOAT_PARQUET.as_posix()}' (FORMAT PARQUET)")

print("[2/5] Group (US / NONUS) shares held per (firm, quarter)...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE grp AS
SELECT sec_entity_id,
       any_value(sec_country)                                   AS sec_country,
       report_date,
       CASE WHEN investor_country = 'US' THEN 'US' ELSE 'NONUS' END AS hgroup,
       SUM(adj_holding)                                          AS shares_held,
       COUNT(*)                                                  AS n_positions
FROM read_parquet('{EOM}')
WHERE {BASE_WHERE}
  AND adj_holding IS NOT NULL AND adj_holding > 0
GROUP BY sec_entity_id, report_date,
         CASE WHEN investor_country = 'US' THEN 'US' ELSE 'NONUS' END
""")

print("[3/5] Join + ownership_share, flag impossible (>1)...")
obs = con.execute("""
SELECT g.sec_entity_id,
       g.sec_country,
       g.report_date,
       g.hgroup,
       g.shares_held,
       g.n_positions,
       s.shares_out,
       g.shares_held / NULLIF(s.shares_out, 0)                   AS ownership_share,
       CASE WHEN g.shares_held / NULLIF(s.shares_out, 0) > 1
            THEN 1 ELSE 0 END                                    AS os_impossible,
       CASE WHEN s.shares_out IS NULL THEN 1 ELSE 0 END          AS shrout_missing
FROM grp g
LEFT JOIN shr s USING (sec_entity_id, report_date)
""").df()
obs.to_parquet(OBS_PARQUET, index=False)

print("[4/5] Integrity diagnostics (single-group + COMBINED US+NONUS)...")
# Combined per-(firm, quarter) ownership across BOTH groups must also be <= 1
# (FactSet covers a subset of holders, so institutional ownership cannot exceed float).
# The per-group os_impossible flag misses cases where the combined total > 1 but
# neither group individually exceeds 1 — check that separately.
combined = con.execute("""
SELECT COUNT(*) AS n_combined_gt1
FROM (
    SELECT sec_entity_id, report_date,
           SUM(shares_held) / NULLIF(AVG(shares_out), 0) AS combined_os
    FROM (
        SELECT g.sec_entity_id, g.report_date, g.shares_held, s.shares_out
        FROM grp g LEFT JOIN shr s USING (sec_entity_id, report_date)
    )
    GROUP BY sec_entity_id, report_date
    HAVING SUM(shares_held) / NULLIF(AVG(shares_out), 0) > 1
)
""").df()["n_combined_gt1"].iloc[0]

# ADR/non-primary materiality (what the primary-class restriction drops), by group.
adr = con.execute(f"""
SELECT CASE WHEN investor_country='US' THEN 'US' ELSE 'NONUS' END AS g,
       SUM(CASE WHEN issue_type='EQ' AND fsym_id=fsym_primary_id THEN adj_mv ELSE 0 END)
         / SUM(adj_mv) AS frac_mv_on_primary_eq
FROM read_parquet('{EOM}')
WHERE issue_type IN ('EQ','AD') AND sec_country IN {EU_SQL} AND adj_mv > 0
GROUP BY 1
""").df().set_index("g")["frac_mv_on_primary_eq"].to_dict()

# Residual missingness by country (post-fix) — computed on the observed frame.
mb = (obs.groupby("sec_country")
        .agg(n_cells=("shrout_missing", "size"), frac_shrout_missing=("shrout_missing", "mean"))
        .sort_values("frac_shrout_missing", ascending=False).reset_index())
mb.to_csv(MISS_CSV, index=False)

print("[5/5] Summary...")
n = len(obs)
os_clean = obs.loc[(obs["shrout_missing"] == 0) & (obs["ownership_share"] <= 1), "ownership_share"]
diag = {
    "n_firm_group_quarter_cells": n,
    "n_distinct_firms": obs["sec_entity_id"].nunique(),
    "n_distinct_quarters": obs["report_date"].nunique(),
    "rdate_min": str(obs["report_date"].min()),
    "rdate_max": str(obs["report_date"].max()),
    "frac_cells_shrout_missing": round(float((obs["shrout_missing"] == 1).mean()), 6),
    "n_cells_os_impossible_gt1": int(obs["os_impossible"].sum()),
    "frac_cells_os_impossible_gt1": round(float(obs["os_impossible"].mean()), 8),
    "n_firmquarters_combined_os_gt1": int(combined),
    # distribution over CLEAN cells only (<=1, non-missing) — excludes impossibles
    "os_median_clean": round(float(os_clean.median()), 8),
    "os_p90_clean": round(float(os_clean.quantile(0.90)), 8),
    "os_p99_clean": round(float(os_clean.quantile(0.99)), 8),
    "os_max_clean": round(float(os_clean.max()), 8),
    # ADR / non-primary materiality (dropped by the primary-class restriction)
    "frac_us_holdings_mv_on_primary_eq": round(float(adr.get("US", float("nan"))), 6),
    "frac_nonus_holdings_mv_on_primary_eq": round(float(adr.get("NONUS", float("nan"))), 6),
    "ref_main_panel_firms": 7928,
    "ref_main_panel_quarters": 82,
}
pd.DataFrame([diag]).to_csv(DIAG_CSV, index=False)

print("\n===== ownership_share panel (observed, pre-grid, PRIMARY EQ class) =====")
for k, v in diag.items():
    print(f"  {k:38s} {v}")
print(f"\n  hgroup split:\n{obs['hgroup'].value_counts().to_string()}")
print(f"\n  top-5 countries by residual shrout missingness:\n{mb.head(5).to_string(index=False)}")
print(f"\n  wrote {OBS_PARQUET.name}, {DIAG_CSV.name}, {MISS_CSV.name}")
print("\nSTEP 1-2 complete. Grid / zero-fill / entry-exit convention / difference / "
      "regression = step 3 (not run).")
con.close()
```

### `build_ownership_share_c6_panel.py`

```python
"""
build_ownership_share_c6_panel.py — Essay 2 #5 STEP 3 (panel build).

Merge the primary-EQ ownership onto the SAME C6 grid used by the w-based main spec
(merged_us_eu_zero_filled.parquet), zero-fill, and compute a shares-based flow for the
triple-difference. Structurally identical to c6_panel.dta but with a shares outcome.

--- REVIEW FIX F6 (float-denominator confound) ---
The first version used dos = ownership_share_t − ownership_share_{t−1} with EACH term
divided by its OWN current float: os = held/out_t, dos = held_t/out_t − held_{t−1}/out_{t−1}.
That is NOT denominator-immune: a buyback/issuance (out changes) moves dos even with zero
trading, by −os_{g,t−1}·(Δout/out_t), a term ∝ the group's own lagged ownership level, which
differs across US/NONUS within a firm-quarter and so is NOT absorbed by firm×quarter FE.

The correct pure-trading FLOW fixes the denominator at the LAGGED float:

    flow_{i,g,t} = ( shares_held_{i,g,t} − shares_held_{i,g,t−1} ) / shares_out_{i,t−1}

Numerator = the actual change in the group's share count (real net buying/selling; a firm
buyback does not change it unless the group trades). Denominator = fixed lagged float, so a
same-quarter corporate action cannot mechanically move it. This IS the "net buying/selling as
a fraction of float" object §7.6 intends. We also keep the OLD `dos` (share-of-float change)
as a labelled comparison column, not the primary outcome.

Zero-fill / entry-exit (grid cell with a valid primary-EQ float, group holds nothing → held=0;
firm with no primary-EQ float → NULL, excluded). BAD firm-quarters (any group os>1 OR combined
os>1) are nulled for BOTH groups (current AND lagged) to keep US/NONUS pairing and to stop a
corrupted float from leaking into flow.

Output: output/ownership_c6_panel.dta   (outcome `flow`; `dos` kept for comparison)
"""

from pathlib import Path
import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
GRID = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
OBS = (OUT / "ownership_share_observed.parquet").as_posix()
FLOAT = (OUT / "ownership_share_float.parquet").as_posix()
DTA = OUT / "ownership_c6_panel.dta"

con = duckdb.connect()
con.execute("SET memory_limit='6GB'")

print("[1/4] Merge held + primary-EQ float onto the main C6 grid...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE joined AS
WITH grid AS (
    SELECT sec_entity_id, holder_group, report_date, china_share_lag1q, shock_us_cn
    FROM read_parquet('{GRID}')
),
held AS (
    SELECT sec_entity_id, hgroup AS holder_group, report_date, shares_held
    FROM read_parquet('{OBS}')
),
flt AS (
    SELECT sec_entity_id, report_date, shares_out FROM read_parquet('{FLOAT}')
)
SELECT g.sec_entity_id, g.holder_group, g.report_date,
       g.china_share_lag1q, g.shock_us_cn,
       f.shares_out,
       -- held is 0 where the firm HAS a float but the group holds nothing; NULL where no float
       CASE WHEN f.shares_out IS NULL THEN NULL ELSE COALESCE(h.shares_held, 0) END AS held,
       CASE WHEN f.shares_out IS NULL THEN NULL
            ELSE COALESCE(h.shares_held, 0) / f.shares_out END AS os_raw
FROM grid g
LEFT JOIN held h USING (sec_entity_id, holder_group, report_date)
LEFT JOIN flt  f USING (sec_entity_id, report_date)
""")

print("[2/4] Flag BAD firm-quarters (any group os>1 OR combined>1); null both groups...")
con.execute("""
CREATE OR REPLACE TEMP TABLE clean AS
WITH bad AS (
    SELECT sec_entity_id, report_date
    FROM joined GROUP BY 1, 2
    HAVING MAX(os_raw) > 1 OR SUM(os_raw) > 1
)
SELECT j.sec_entity_id, j.holder_group, j.report_date,
       j.china_share_lag1q, j.shock_us_cn, j.shares_out,
       CASE WHEN b.sec_entity_id IS NOT NULL THEN NULL ELSE j.held   END AS held,
       CASE WHEN b.sec_entity_id IS NOT NULL THEN NULL ELSE j.os_raw END AS os
FROM joined j
LEFT JOIN bad b USING (sec_entity_id, report_date)
""")

print("[3/4] Flow = (held_t - held_{t-1}) / out_{t-1}, backward, within (firm, group)...")
df = con.execute("""
SELECT
    CAST(sec_entity_id AS VARCHAR)                  AS firm_str,
    holder_group                                    AS hgroup,
    CAST(report_date AS TIMESTAMP)                  AS rdate,
    os,
    held,
    -- F6 primary outcome: pure trading flow with FIXED lagged float
    (held - LAG(held) OVER w) / NULLIF(LAG(shares_out) OVER w, 0)  AS flow,
    -- old share-of-float change, kept for comparison (has the F6 float-confound)
    os - LAG(os) OVER w                             AS dos,
    china_share_lag1q                               AS cn_lag,
    shock_us_cn                                     AS shock,
    CASE WHEN holder_group = 'US' THEN 1 ELSE 0 END AS us
FROM clean
WINDOW w AS (PARTITION BY sec_entity_id, holder_group ORDER BY report_date)
""").df()
con.close()

df["firm_str"] = df["firm_str"].astype(str)
df["hgroup"] = df["hgroup"].astype(str)
df["rdate"] = pd.to_datetime(df["rdate"])
for c in ["os", "held", "flow", "dos", "cn_lag", "shock"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
df["us"] = df["us"].astype("int8")

# estimation subset: non-null FLOW (primary) / cn_lag / shock
est = df.dropna(subset=["flow", "cn_lag", "shock"]).copy()

print("[4/4] Hard asserts + write...")
n = len(est)
assert n > 0
assert set(est["hgroup"].unique()) <= {"US", "NONUS"}
assert ((est["hgroup"] == "US") == (est["us"] == 1)).all()
assert not est.duplicated(["firm_str", "hgroup", "rdate"]).any(), "duplicate rows"
_pair = est.groupby(["firm_str", "rdate"])["hgroup"].nunique()
assert (_pair == 2).all(), f"not fully paired: {(_pair != 2).sum():,} firm-quarters"
assert (est.groupby("rdate")["shock"].nunique() == 1).all(), "shock varies within a quarter"
assert est["cn_lag"].between(0, 1).all(), "cn_lag outside [0,1]"
_q = est["rdate"].dt.to_period("Q").drop_duplicates().sort_values()
_exp = pd.period_range(_q.iloc[0], _q.iloc[-1], freq="Q")
assert len(_q) == len(_exp) and (_q.to_numpy() == _exp.to_numpy()).all(), "gap in quarter coverage"

keep = ["firm_str", "hgroup", "rdate", "flow", "dos", "os", "cn_lag", "shock", "us"]
est[keep].to_stata(DTA, write_index=False, convert_dates={"rdate": "tc"}, version=118)

print(f"\n===== ownership_c6_panel (estimation subset, outcome=flow) =====")
print(f"  rows: {n:,}   firms: {est['firm_str'].nunique():,}   quarters: {len(_q)}"
      f"   ({_q.iloc[0]} -> {_q.iloc[-1]})")
print(f"  hgroup split:\n{est['hgroup'].value_counts().to_string()}")
print(f"  flow: mean {est['flow'].mean():.3e}  sd {est['flow'].std():.3e}"
      f"  p1 {est['flow'].quantile(.01):.3e}  p99 {est['flow'].quantile(.99):.3e}")
print(f"  (old dos for comparison: mean {est['dos'].mean():.3e}  sd {est['dos'].std():.3e})")
print(f"  wrote {DTA.name} ({n:,} rows)")
```

### `build_audit_panel_f1f2f7.py`

```python
"""
build_audit_panel_f1f2f7.py — remediation panel for review findings F1, F2, F7.

Built on the SAME C6 grid (merged_us_eu_zero_filled.parquet), adding:
- F1 (timing): dw_lead1 = Delta w_{t+1} = w_{t+1} - w_t (pure lagged-shock outcome:
  regress on US x CN x S_t with NO look-ahead), and cumulative responses
  cum_h = w_{t+h} - w_{t-1} for h = 0..4 for a local-projection IRF.
- F2 (existence span): in_span = 1 iff report_date in [first_active, last_active],
  where first/last_active = min/max quarter with I_ict > 0 for the firm (either group).
  Lets the headline be re-run WITHOUT the ~25% (in_span==0 = 25.03%) pre-IPO /
  post-delisting phantom zeros.
- F7 (generated regressor / full-sample AR(1)): carry the RAW GPR level gpr_us_cn and
  its lag, so the shock can be replaced by US x CN x GPR_t + US x CN x GPR_{t-1}
  (nests every AR(1) (a,b), no generated regressor, no full-sample look-ahead).

Leads/lags are computed on the FULL contiguous grid (Cartesian, so LEAD/LAG over
report_date within (firm, group) is the true adjacent quarter), THEN the in_span flag
is attached — so span-boundary leads are not corrupted.

Output: output/audit_c6_panel.dta
"""

from pathlib import Path
import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
GRID = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
DTA = OUT / "audit_c6_panel.dta"

con = duckdb.connect()
con.execute("SET memory_limit='6GB'")

print("[1/3] Firm existence span (first/last quarter with I_ict>0)...")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE span AS
SELECT sec_entity_id,
       MIN(CASE WHEN I_ict > 0 THEN report_date END) AS first_active,
       MAX(CASE WHEN I_ict > 0 THEN report_date END) AS last_active
FROM read_parquet('{GRID}')
GROUP BY 1
""")

print("[2/3] Leads/lags on the full contiguous grid + in_span flag...")
df = con.execute(f"""
WITH g AS (
    SELECT
        CAST(m.sec_entity_id AS VARCHAR)                         AS firm_str,
        m.holder_group                                          AS hgroup,
        CAST(m.report_date AS TIMESTAMP)                        AS rdate,
        m.delta_w                                               AS dw,
        m.portfolio_weight_eu                                   AS w,
        m.w_prev                                                AS w_prev,
        m.china_share_lag1q                                     AS cn_lag,
        m.shock_us_cn                                           AS shock,
        m.gpr_us_cn                                             AS gpr,
        CASE WHEN m.holder_group = 'US' THEN 1 ELSE 0 END       AS us,
        s.first_active, s.last_active,
        LEAD(m.delta_w, 1)            OVER w                     AS dw_lead1,
        LEAD(m.portfolio_weight_eu,1) OVER w                     AS w_l1,
        LEAD(m.portfolio_weight_eu,2) OVER w                     AS w_l2,
        LEAD(m.portfolio_weight_eu,3) OVER w                     AS w_l3,
        LEAD(m.portfolio_weight_eu,4) OVER w                     AS w_l4,
        LAG(m.gpr_us_cn, 1)          OVER w                      AS gpr_lag
    FROM read_parquet('{GRID}') m
    JOIN span s USING (sec_entity_id)
    WINDOW w AS (PARTITION BY m.sec_entity_id, m.holder_group ORDER BY m.report_date)
)
SELECT firm_str, hgroup, rdate, us, dw, cn_lag, shock, gpr, gpr_lag,
       dw_lead1,
       (w      - w_prev) AS cum0,   -- = dw
       (w_l1   - w_prev) AS cum1,
       (w_l2   - w_prev) AS cum2,
       (w_l3   - w_prev) AS cum3,
       (w_l4   - w_prev) AS cum4,
       CASE WHEN rdate BETWEEN first_active AND last_active THEN 1 ELSE 0 END AS in_span
FROM g
WHERE cn_lag IS NOT NULL AND shock IS NOT NULL   -- keep regressor-complete rows; outcomes may be null at edges
""").df()
con.close()

df["firm_str"] = df["firm_str"].astype(str)
df["hgroup"] = df["hgroup"].astype(str)
df["rdate"] = pd.to_datetime(df["rdate"])
for c in ["dw", "cn_lag", "shock", "gpr", "gpr_lag", "dw_lead1",
          "cum0", "cum1", "cum2", "cum3", "cum4"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
df["us"] = df["us"].astype("int8")
df["in_span"] = df["in_span"].astype("int8")

# sanity: cum0 must equal dw where both present
_chk = df.dropna(subset=["dw", "cum0"])
assert (_chk["dw"] - _chk["cum0"]).abs().max() < 1e-9, "cum0 != dw"

print("[3/3] Diagnostics + write...")
n = len(df)
print(f"  rows (regressor-complete): {n:,}")
print(f"  in_span share: {df['in_span'].mean():.4f}  (phantom/out-of-span = {1-df['in_span'].mean():.4f})")
print(f"  dw non-null: {df['dw'].notna().mean():.4f}   dw_lead1 non-null: {df['dw_lead1'].notna().mean():.4f}")
print(f"  cum4 non-null: {df['cum4'].notna().mean():.4f}")
print(f"  gpr non-null: {df['gpr'].notna().mean():.4f}  gpr_lag non-null: {df['gpr_lag'].notna().mean():.4f}")
df.to_stata(DTA, write_index=False, convert_dates={"rdate": "tc"}, version=118)
print(f"  wrote {DTA.name} ({n:,} rows)")
```

### `run_randomization_inference.py`

```python
"""
run_randomization_inference.py — design-based inference for the triple-difference
β₃, robust to the few-quarter-cluster + overlapping-window problems that make the
CRVE unreliable (review findings F1b, F3, F9).

Exact algebra. The panel is balanced 2-per-(firm, quarter) {US, NONUS}. The De Haas
spec  y_g = β₂·(us_g·cn) + β₃·(us_g·cn·S) + α_{firm×qtr} + γ_{group×qtr}
collapses, on the US−NONUS within-firm-quarter difference Δy, to

    Δy_it = β₂·cn_it + β₃·(cn_it·S_t) + δ_t + Δε_it        (δ_t = quarter FE)

which reproduces the two-way-FE β₂, β₃ EXACTLY. After within-quarter demeaning of cn
(→ c̃n; note Σ_i c̃n_it = 0), and because S_t is constant within a quarter,

    β₃ = solve the 2×2 OLS of Δỹ on [c̃n, S·c̃n],

whose sufficient statistics are, per quarter t,  A_t = Σ_i c̃n²,  C_t = Σ_i c̃n·Δỹ.
For ANY assignment of the 82 quarter shocks S_t:
    X'X = [[ΣA,      Σ S_t A_t],
           [Σ S_t A_t, Σ S_t² A_t]],   X'y = [ΣC,  Σ S_t C_t].
So a randomization test that PERMUTES the 82 shocks across quarters is just weighted
sums over 82 numbers → millions of permutations are instant.

RI p-value (two-sided) = share of permutations with |β₃_perm| ≥ |β₃_obs|.
Reported for the headline (h=0), the lead-flow (Δw_{t+1}), and the LP horizons h=1..4,
on the full grid and the in-span subset.
"""

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
DTA = (OUT / "audit_c6_panel.dta").as_posix()
N_PERM = 200000
SEED_STREAM = 12345  # deterministic; Math.random unavailable-style reproducibility

con = duckdb.connect()

def load_diffs(where=""):
    # collapse to firm-quarter US−NONUS differences for each outcome
    q = f"""
    WITH p AS (SELECT * FROM read_parquet('{OUT.as_posix()}/audit_c6_panel.parquet') {where})
    SELECT firm_str, rdate,
           any_value(cn_lag) AS cn, any_value(shock) AS s,
           MAX(CASE WHEN us=1 THEN dw       END) - MAX(CASE WHEN us=0 THEN dw       END) AS d_dw,
           MAX(CASE WHEN us=1 THEN dw_lead1 END) - MAX(CASE WHEN us=0 THEN dw_lead1 END) AS d_lead1,
           MAX(CASE WHEN us=1 THEN cum0 END) - MAX(CASE WHEN us=0 THEN cum0 END) AS d_c0,
           MAX(CASE WHEN us=1 THEN cum1 END) - MAX(CASE WHEN us=0 THEN cum1 END) AS d_c1,
           MAX(CASE WHEN us=1 THEN cum2 END) - MAX(CASE WHEN us=0 THEN cum2 END) AS d_c2,
           MAX(CASE WHEN us=1 THEN cum3 END) - MAX(CASE WHEN us=0 THEN cum3 END) AS d_c3,
           MAX(CASE WHEN us=1 THEN cum4 END) - MAX(CASE WHEN us=0 THEN cum4 END) AS d_c4
    FROM p GROUP BY firm_str, rdate
    """
    return con.execute(q).df()

def beta3_and_ri(df, ycol, n_perm=N_PERM, seed=0):
    d = df.dropna(subset=[ycol, "cn", "s"]).copy()
    # within-quarter demean of cn
    d["cn_c"] = d["cn"] - d.groupby("rdate")["cn"].transform("mean")
    # per-quarter sufficient stats
    g = d.groupby("rdate")
    A = g.apply(lambda x: np.sum(x["cn_c"].values**2)).values.astype(float)          # A_t
    C = g.apply(lambda x: np.sum(x["cn_c"].values * x[ycol].values)).values.astype(float)  # C_t
    S = g["s"].first().values.astype(float)
    def solve_b3(Svec):
        sA = np.sum(A); sSA = np.sum(Svec*A); sS2A = np.sum(Svec*Svec*A)
        sC = np.sum(C); sSC = np.sum(Svec*C)
        # X'X = [[sA, sSA],[sSA, sS2A]] ; X'y = [sC, sSC] ; want second coef
        det = sA*sS2A - sSA*sSA
        if abs(det) < 1e-300: return np.nan
        b3 = (sA*sSC - sSA*sC)/det
        return b3
    b3_obs = solve_b3(S)
    rng = np.random.default_rng(seed)
    nq = len(S)
    cnt = 0
    # vectorize permutations in blocks
    block = 20000
    done = 0
    while done < n_perm:
        b = min(block, n_perm - done)
        # generate b permutations of S
        perms = np.array([rng.permutation(S) for _ in range(b)])  # b x nq
        sA = np.sum(A);
        sSA = perms @ A
        sS2A = (perms*perms) @ A
        sC = np.sum(C)
        sSC = perms @ C
        det = sA*sS2A - sSA*sSA
        b3p = np.where(np.abs(det) < 1e-300, np.nan, (sA*sSC - sSA*sC)/det)
        cnt += np.sum(np.abs(b3p) >= abs(b3_obs) - 1e-300)
        done += b
    ri_p = (cnt + 1) / (n_perm + 1)
    return b3_obs, ri_p, len(S), len(d)

# write a parquet copy (duckdb reads dta poorly; use pandas->parquet once)
import pyreadstat  # noqa
_df, _ = pyreadstat.read_dta(DTA)
_df.to_parquet(OUT / "audit_c6_panel.parquet", index=False)

full = load_diffs("")
span = load_diffs("WHERE in_span=1")

print(f"{'spec':22s} {'b3':>12s} {'RI_p(2side)':>12s} {'nq':>4s} {'n_fq':>9s}")
rows = []
for label, df, y in [
    ("headline dw (h0)",       full, "d_dw"),
    ("lead-flow dw_{t+1}",     full, "d_lead1"),
    ("LP cum h=1",             full, "d_c1"),
    ("LP cum h=2",             full, "d_c2"),
    ("LP cum h=3",             full, "d_c3"),
    ("LP cum h=4",             full, "d_c4"),
    ("headline dw IN-SPAN",    span, "d_dw"),
    ("LP cum h=1 IN-SPAN",     span, "d_c1"),
    ("LP cum h=4 IN-SPAN",     span, "d_c4"),
]:
    b3, p, nq, nfq = beta3_and_ri(df, y, seed=SEED_STREAM)
    print(f"{label:22s} {b3:12.3e} {p:12.4f} {nq:4d} {nfq:9,d}")
    rows.append({"spec": label, "b3": b3, "ri_p_2sided": p, "n_quarters": nq, "n_firmquarters": nfq})

pd.DataFrame(rows).to_csv(OUT / "audit_randomization_inference.csv", index=False)
print(f"\nwrote audit_randomization_inference.csv  ({N_PERM:,} permutations each)")
```

### `run_ri_3pairwise.py`

```python
"""
run_ri_3pairwise.py — randomization inference for the 3-PAIRWISE headline
(it firm×quarter + gt group×quarter + ig firm×group).

On the balanced 2-per-(firm,quarter) panel the three pairwise FE collapse, on the
US−NONUS within-firm-quarter difference Δy, to

    Δy_it = β₂·cn_it + β₃·(cn_it·S_t) + φ_i (firm FE) + δ_t (quarter FE) + error

i.e. FIRM + QUARTER two-way FE on the difference panel (ig → the firm intercept φ_i;
gt → the quarter intercept δ_t; it absorbed by the pairwise difference itself).

RI (sharp null β₃=0): permute the 82 quarter shocks S_t. Unlike the quarter-FE-only
case there is no closed-form sufficient statistic (firm-demeaning of cn·S mixes S
across a firm's quarters), so we two-way-demean cn·S each permutation via fast
bincount alternating projections. β₃ via FWL against the (once-)demeaned Δy and cn.
Reported for the headline (h0) and the LP horizons that looked significant under CRVE.
"""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")
N_PERM = 5000
DEMEAN_ITERS = 30
SEED = 20260702

con = duckdb.connect()
d = con.execute(f"""
SELECT firm_str, rdate,
       any_value(cn_lag) AS cn, any_value(shock) AS s,
       MAX(CASE WHEN us=1 THEN dw   END) - MAX(CASE WHEN us=0 THEN dw   END) AS d_dw,
       MAX(CASE WHEN us=1 THEN cum1 END) - MAX(CASE WHEN us=0 THEN cum1 END) AS d_c1,
       MAX(CASE WHEN us=1 THEN cum2 END) - MAX(CASE WHEN us=0 THEN cum2 END) AS d_c2,
       MAX(CASE WHEN us=1 THEN cum4 END) - MAX(CASE WHEN us=0 THEN cum4 END) AS d_c4
FROM read_parquet('{(OUT/'audit_c6_panel.parquet').as_posix()}')
GROUP BY firm_str, rdate
""").df()
con.close()

d["firm_c"] = pd.factorize(d["firm_str"])[0]
d["q_c"] = pd.factorize(d["rdate"])[0]

def make_demeaner(firm_c, q_c, iters):
    nf = firm_c.max() + 1
    nq = q_c.max() + 1
    fcount = np.bincount(firm_c, minlength=nf).astype(float)
    qcount = np.bincount(q_c, minlength=nq).astype(float)
    def demean(x):
        x = x.copy()
        for _ in range(iters):
            fm = np.bincount(firm_c, weights=x, minlength=nf) / fcount
            x = x - fm[firm_c]
            qm = np.bincount(q_c, weights=x, minlength=nq) / qcount
            x = x - qm[q_c]
        return x
    return demean

def ri_twoway(df, ycol, n_perm=N_PERM, seed=SEED):
    sub = df.dropna(subset=[ycol, "cn", "s"]).reset_index(drop=True)
    firm_c = pd.factorize(sub["firm_str"])[0]
    q_c = pd.factorize(sub["rdate"])[0]
    demean = make_demeaner(firm_c, q_c, DEMEAN_ITERS)
    y = demean(sub[ycol].to_numpy(float))          # two-way demeaned outcome (once)
    cn = demean(sub["cn"].to_numpy(float))          # two-way demeaned cn (once)
    cn_raw = sub["cn"].to_numpy(float)
    # per-quarter shock lookup + row->quarter map
    qs = sub.groupby("q_c" if "q_c" in sub else q_c)  # noqa
    # map quarter code -> its shock
    qcode = q_c
    nq = qcode.max() + 1
    Svec = np.zeros(nq)
    Svec[qcode] = sub["s"].to_numpy(float)          # shock constant within quarter
    cn_dot_c = float(cn @ cn)                        # denom piece (fixed)
    def beta3(Sq):
        x = cn_raw * Sq[qcode]                       # cn * S (row level)
        xt = demean(x)                              # two-way demean
        # FWL: residualize xt on cn (already demeaned), then regress y on that
        b = (cn @ xt) / cn_dot_c
        xr = xt - b * cn
        denom = xr @ xr
        return (y @ xr) / denom if denom > 0 else np.nan
    b_obs = beta3(Svec)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        Sp = rng.permutation(Svec)                  # permute quarter shocks
        bp = beta3(Sp)
        if abs(bp) >= abs(b_obs) - 1e-300:
            cnt += 1
    return b_obs, (cnt + 1) / (n_perm + 1), sub.shape[0]

print(f"{'spec':16s} {'b3(chk vs Stata)':>18s} {'RI_p_2side':>12s} {'n_fq':>9s}   (N_PERM={N_PERM}, iters={DEMEAN_ITERS})")
print("  Stata 3-pairwise b3: headline +1.800e-6, cum1 +2.373e-6, cum4 +6.621e-6")
rows = []
for lbl, y in [("headline dw", "d_dw"), ("LP cum1", "d_c1"), ("LP cum4", "d_c4")]:
    b, p, n = ri_twoway(d, y)
    print(f"{lbl:16s} {b:12.3e} {p:12.4f} {n:9,d}")
    rows.append({"spec": lbl, "fe": "it+gt+ig (3-pairwise)", "b3": b, "ri_p_2sided": p, "n_firmquarters": n})
pd.DataFrame(rows).to_csv(OUT / "audit_ri_3pairwise.csv", index=False)
print(f"\nwrote audit_ri_3pairwise.csv")
```

### `build_riskset_lagonly.py`

```python
"""
build_riskset_lagonly.py — review fix F8.

The main risk-set (build_spell_riskset.py) defines firm-quarter membership as
held at t OR t-1 OR t+1 (LEAD). The t+1 lead conditions membership on a
POST-treatment outcome — the same family of concern as the retracted centered
diff. This builds a LAG-ONLY variant (held at t OR t-1, no look-ahead) so β₃ can
be checked for sensitivity. Everything else matches build_spell_riskset.py.

Output: output/c6_panel_riskset_lagonly.dta
"""
from pathlib import Path
import duckdb
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
uri = (OUT / "merged_us_eu_zero_filled.parquet").as_posix()
DTA = OUT / "c6_panel_riskset_lagonly.dta"

con = duckdb.connect()
con.execute(f"""
CREATE OR REPLACE TEMP TABLE flagged AS
SELECT sec_entity_id, holder_group, report_date, I_ict, delta_w,
       china_share_lag1q, shock_us_cn,
       CAST(I_ict > 0 AS INTEGER) AS held,
       COALESCE(LAG(CAST(I_ict > 0 AS INTEGER)) OVER w, 0) AS held_lag
FROM read_parquet('{uri}')
WINDOW w AS (PARTITION BY sec_entity_id, holder_group ORDER BY report_date)
""")
con.execute("""
CREATE OR REPLACE TEMP TABLE riskset AS
SELECT *,
       MAX(CASE WHEN held=1 OR held_lag=1 THEN 1 ELSE 0 END)
           OVER (PARTITION BY sec_entity_id, report_date) AS risk
FROM flagged
""")
df = con.execute("""
SELECT CAST(sec_entity_id AS VARCHAR) AS firm_str,
       holder_group AS hgroup,
       CAST(report_date AS TIMESTAMP) AS rdate,
       delta_w AS dw, china_share_lag1q AS cn_lag, shock_us_cn AS shock,
       CASE WHEN holder_group='US' THEN 1 ELSE 0 END AS us
FROM riskset WHERE risk = 1
""").df()
con.close()

est = df.dropna(subset=["dw", "cn_lag", "shock"]).copy()
est["firm_str"] = est["firm_str"].astype(str)
est["hgroup"] = est["hgroup"].astype(str)
est["rdate"] = pd.to_datetime(est["rdate"])
for c in ["dw", "cn_lag", "shock"]:
    est[c] = pd.to_numeric(est[c], errors="raise").astype("float64")
est["us"] = est["us"].astype("int8")

# pairing assert (lag-only membership is still firm-quarter -> both groups kept)
_p = est.groupby(["firm_str", "rdate"])["hgroup"].nunique()
assert (_p == 2).all(), f"not fully paired: {(_p != 2).sum():,}"
assert not est.duplicated(["firm_str", "hgroup", "rdate"]).any()

keep = ["firm_str", "hgroup", "rdate", "dw", "cn_lag", "shock", "us"]
est[keep].to_stata(DTA, write_index=False, convert_dates={"rdate": "tc"}, version=118)
print(f"lag-only risk set: {len(est):,} rows, {est['firm_str'].nunique():,} firms, "
      f"{est.groupby(['firm_str','rdate']).ngroups:,} firm-quarters")
print(f"  (compare with-lead risk set c6_panel_riskset.dta = 342,262 rows)")
print(f"  wrote {DTA.name}")
```


---

## 16. Julia data-construction pipeline (verbatim, run order 00 to 06)
`00`–`06` build the zero-filled panel from raw FactSet + Revere + GPR. **Note (F2, second review round):** `06_cartesian_grid.jl` crosses the universe with ALL quarters and does NOT intersect each firm's existence span — see the in-span remediation in `build_audit_panel_f1f2f7.py`.

### `00_setup.jl`

```julia
# 00_setup.jl
# Run this FIRST. Sets up packages, paths, a DuckDB connection helper, atomic
# writes, and TEST_MODE controls.
#
# Audited 2026-06-01 — see AUDIT_2026_06_01_julia_descriptive.md.
# Resulting changes:
#   - ENV-driven paths (DPN_DATA_ROOT) so the pipeline runs off any disk.
#   - TEST_MODE from ENV (DPN_TEST_MODE), with downstream guard that forces
#     suffixed output filenames when set so a test run cannot silently
#     overwrite canonical artifacts.
#   - Atomic-write helper (atomic_copy_to) for every COPY ... TO parquet.
#   - Manifest writer for reproducibility (records git sha, julia version,
#     row count, input fingerprints).
#
# One-time package install (uncomment and run once, then comment again):
# using Pkg
# Pkg.add(["DuckDB", "DataFrames", "CSV", "StatsBase", "Dates",
#          "Statistics", "CategoricalArrays", "GZip", "Printf", "SHA", "JSON3"])

using DuckDB, DataFrames, CSV, StatsBase, Dates, Statistics, Printf, SHA

# ============================================================
# PATHS — ENV-driven with sensible defaults
# ============================================================
const DATA_ROOT = get(ENV, "DPN_DATA_ROOT", raw"E:\Data\Data")
const OWN_DIR   = joinpath(DATA_ROOT, "Factset Ownership")
const REV_DIR   = joinpath(DATA_ROOT, "Factset Revere")

const INSTITUTIONS_PATH = joinpath(OWN_DIR, "Factset_LionShares_Institutions.gz")
const FUNDS_PATH        = joinpath(OWN_DIR, "Factset_LionShares_Funds.gz")
const SEC_MAP_PATH      = joinpath(OWN_DIR, "Factset_Security_Map.gz")
const SEC_COVERAGE_PATH = joinpath(OWN_DIR, "Factset_Security_coverage.gz")
const HOLDINGS_GLOB     = joinpath(OWN_DIR, "Factset_FundOwners_*.gz")

const REVERE_REL_PATH = joinpath(REV_DIR, "data_giorgio.csv")
const REVERE_CO_PATH  = joinpath(REV_DIR, "revere_company_wrds.csv")

const GPR_PATH      = joinpath(DATA_ROOT, "ai_gpr_bilateral_monthly.csv")
const GSDB_PATH     = joinpath(DATA_ROOT, "GSDB_V4_dates.csv")
const GRAVITY_PATH  = joinpath(DATA_ROOT, "gravity_vars_2021.csv")
const CONFLICT_PATH = joinpath(DATA_ROOT, "conflict_monthly.csv")

const SCRIPT_DIR = @__DIR__
const OUT_DIR    = joinpath(SCRIPT_DIR, "output")
isdir(OUT_DIR) || mkpath(OUT_DIR)

# Raw-parquet cache lives OFF the OneDrive-synced path. ~30 GB total.
# 03a writes here; 03 reads from here.
const RAW_PARQUET_DIR = get(ENV, "DPN_RAW_PARQUET_DIR", raw"E:\Data\Data\raw_parquet")
isdir(RAW_PARQUET_DIR) || mkpath(RAW_PARQUET_DIR)

# ============================================================
# TEST_MODE controls — set DPN_TEST_MODE=true in environment to subset.
# When TEST_MODE is on, downstream MUST add a suffix to output filenames
# (see test_suffix_path) so test runs cannot overwrite canonical outputs.
# ============================================================
const TEST_MODE = parse(Bool, lowercase(get(ENV, "DPN_TEST_MODE", "false")))
const TEST_SUFFIX = "_TESTMODE"

"""
    test_suffix_path(path) -> String

If `TEST_MODE` is on, inserts `_TESTMODE` before the file extension so a test
run cannot silently overwrite the canonical artifact at the same path. If
`TEST_MODE` is off, returns `path` unchanged.
"""
function test_suffix_path(path::AbstractString)
    TEST_MODE || return path
    base, ext = splitext(path)
    return base * TEST_SUFFIX * ext
end

# ============================================================
# EU COUNTRY LIST — single source of truth (was duplicated across 02/03/04/05)
# ============================================================
const EU_COUNTRIES = ("GB","DE","FR","NL","CH","IT","ES","SE","DK","NO","FI",
                      "BE","AT","IE","LU","PT","PL","CZ","HU","GR","RO","SK",
                      "SI","BG","HR","EE","LV","LT")
const EU_SQL_TUPLE = "(" * join(["'" * c * "'" for c in EU_COUNTRIES], ",") * ")"

# ============================================================
# DUCKDB CONNECTION HELPER
# ============================================================
# RAM budget: 6GB out of 8-10GB usable (leave headroom for Julia + OS).
# Spill to disk when RAM exceeded.
function dbcon(; memory_gb::Int=6, threads::Int=4)
    spill = joinpath(tempdir(), "duckdb_spill")
    isdir(spill) || mkpath(spill)
    con = DBInterface.connect(DuckDB.DB, ":memory:")
    DBInterface.execute(con, "SET memory_limit='$(memory_gb)GB'")
    DBInterface.execute(con, "SET threads=$threads")
    DBInterface.execute(con, "SET temp_directory='$(replace(spill, "\\" => "/"))'")
    # Critical for large GROUP BY / COPY operations on tight RAM:
    # lets DuckDB pipeline through without buffering full input order.
    DBInterface.execute(con, "SET preserve_insertion_order=false")
    return con
end

# Convenience: run SQL and return DataFrame
qdf(con, sql::AbstractString) = DataFrame(DBInterface.execute(con, sql))

# ============================================================
# ATOMIC-WRITE HELPER — for every COPY ... TO parquet.
# Pattern: write to <path>.tmp.<pid>, verify non-empty, then mv to <path>.
# On error, the .tmp is cleaned up so canonical paths never carry a partial
# write. Same filesystem assertion guarantees the rename is atomic.
# ============================================================
"""
    atomic_copy_to(con, select_sql, out_path; format="parquet", compression="zstd", row_group_size=122880)

Executes `COPY (\$select_sql) TO '<out_path>.tmp.<pid>' (FORMAT 'parquet', ...)`,
verifies the temp file has non-zero size, then atomically renames it onto
`out_path`. Clears any stale `.tmp` first. On error, removes the partial temp.

Uses pinned `COMPRESSION_LEVEL 3` and `ROW_GROUP_SIZE 122880` so file SHA-256
is reproducible across DuckDB versions (per audit recommendation).
"""
function atomic_copy_to(con, select_sql::AbstractString, out_path::AbstractString;
                        format::AbstractString="parquet",
                        compression::AbstractString="zstd",
                        compression_level::Int=3,
                        row_group_size::Int=122_880)
    out_path_fwd = replace(out_path, "\\" => "/")
    tmp_path     = out_path_fwd * ".tmp." * string(getpid())
    # Same-filesystem assertion for atomic rename.
    @assert dirname(tmp_path) == dirname(out_path_fwd) "tmp and target must share volume for atomic rename"
    # Clear stale tmp from a previous failed run.
    isfile(replace(tmp_path, "/" => "\\")) && rm(replace(tmp_path, "/" => "\\"); force=true)
    try
        if format == "parquet"
            DBInterface.execute(con, """
                COPY ($select_sql) TO '$tmp_path' (
                    FORMAT 'parquet',
                    COMPRESSION '$compression',
                    COMPRESSION_LEVEL $compression_level,
                    ROW_GROUP_SIZE $row_group_size
                )
            """)
        else
            DBInterface.execute(con, "COPY ($select_sql) TO '$tmp_path' (FORMAT '$format')")
        end
        tmp_native = replace(tmp_path, "/" => "\\")
        @assert isfile(tmp_native) && filesize(tmp_native) > 0 "atomic_copy_to: temp file empty or missing"
        mv(tmp_native, replace(out_path_fwd, "/" => "\\"); force=true)
    catch e
        tmp_native = replace(tmp_path, "/" => "\\")
        isfile(tmp_native) && rm(tmp_native; force=true)
        rethrow(e)
    end
    return out_path_fwd
end

# ============================================================
# MANIFEST WRITER — per-step reproducibility sidecar.
# Records git sha (if available), Julia version, row count, output sha256,
# input file fingerprints, build timestamp. Downstream scripts can read and
# assert against expected fingerprints.
# ============================================================
function _git_sha()
    try
        return strip(read(`git -C $SCRIPT_DIR rev-parse --short HEAD`, String))
    catch
        return "unknown"
    end
end

function _file_sha256(path::AbstractString)
    isfile(path) || return ""
    open(path) do io
        return bytes2hex(SHA.sha256(io))
    end
end

"""
    write_manifest(step_name, out_path; row_count=missing, input_paths=String[])

Writes `<out_path>.meta.json` (single-line JSON) with reproducibility info
beside `out_path`. Use after `atomic_copy_to` lands the canonical artifact.
"""
function write_manifest(step_name::AbstractString, out_path::AbstractString;
                        row_count=missing,
                        input_paths::Vector{<:AbstractString}=String[])
    out_native = replace(out_path, "/" => "\\")
    meta_path = out_native * ".meta.json"
    input_fp = [Dict("path"=>p, "size_bytes"=>(isfile(p) ? filesize(p) : 0),
                     "mtime"=>(isfile(p) ? string(Dates.unix2datetime(mtime(p))) : ""))
                for p in input_paths]
    meta = Dict(
        "step"             => step_name,
        "out_path"         => out_native,
        "git_sha"          => _git_sha(),
        "julia_version"    => string(VERSION),
        "duckdb_version"   => (try qdf(dbcon(), "SELECT version() AS v").v[1] catch; "unknown" end),
        "test_mode"        => TEST_MODE,
        "row_count"        => row_count === missing ? -1 : row_count,
        "sha256"           => _file_sha256(out_native),
        "build_ts"         => string(now()),
        "input_files"      => input_fp,
    )
    # Minimal JSON without bringing in JSON3 dep — fast escape for filenames.
    json_str = "{" * join([
        "\"step\":\"$(meta["step"])\"",
        "\"out_path\":\"$(replace(meta["out_path"], "\\" => "\\\\"))\"",
        "\"git_sha\":\"$(meta["git_sha"])\"",
        "\"julia_version\":\"$(meta["julia_version"])\"",
        "\"duckdb_version\":\"$(meta["duckdb_version"])\"",
        "\"test_mode\":$(meta["test_mode"])",
        "\"row_count\":$(meta["row_count"])",
        "\"sha256\":\"$(meta["sha256"])\"",
        "\"build_ts\":\"$(meta["build_ts"])\"",
        "\"input_count\":$(length(input_paths))",
    ], ",") * "}"
    open(meta_path, "w") do io
        write(io, json_str)
    end
    return meta_path
end

# ============================================================
# FILE EXISTENCE SMOKE TEST
# ============================================================
function smoke_test()
    files = [INSTITUTIONS_PATH, FUNDS_PATH, SEC_MAP_PATH, SEC_COVERAGE_PATH,
             REVERE_REL_PATH, REVERE_CO_PATH,
             GPR_PATH, GSDB_PATH, GRAVITY_PATH, CONFLICT_PATH]
    println("File existence check:")
    for f in files
        size_mb = isfile(f) ? round(filesize(f) / 1024^2, digits=1) : 0.0
        mark = isfile(f) ? "OK " : "MISS"
        println("  [$mark] $(basename(f)) ($size_mb MB)")
    end
    # Check holdings chunks
    holdings_files = filter(f -> startswith(basename(f), "Factset_FundOwners_") && endswith(f, ".gz"),
                            readdir(OWN_DIR, join=true))
    println("Holdings chunks: $(length(holdings_files)) files")
    for f in sort(holdings_files)
        size_gb = round(filesize(f) / 1024^3, digits=2)
        println("  $(basename(f))  $size_gb GB")
    end

    # Test DuckDB connection
    print("\nDuckDB version: ")
    con = dbcon()
    res = qdf(con, "SELECT version() AS v")
    println(res.v[1])
    DBInterface.close!(con)

    println("\nOutput directory: $OUT_DIR")
    println("TEST_MODE: $TEST_MODE (set DPN_TEST_MODE=true to enable)")
    println("DATA_ROOT: $DATA_ROOT (override with DPN_DATA_ROOT)")
    println("\nSetup OK. Proceed to 01_master_files.jl")

    # Write input fingerprints CSV for downstream provenance checks.
    fp_path = joinpath(OUT_DIR, "00_input_fingerprints.csv")
    open(fp_path, "w") do io
        write(io, "path,size_bytes,mtime,sha256_head\n")
        for f in files
            sz = isfile(f) ? filesize(f) : 0
            mt = isfile(f) ? string(Dates.unix2datetime(mtime(f))) : ""
            # SHA256 of large files is slow; skip for files > 100 MB here.
            sh = (isfile(f) && filesize(f) < 100_000_000) ? _file_sha256(f) : ""
            write(io, "$f,$sz,$mt,$sh\n")
        end
    end
    println("Input fingerprints -> $fp_path")
end

# If run as a script (not include), execute smoke test
if abspath(PROGRAM_FILE) == @__FILE__
    smoke_test()
end
```

### `01_master_files.jl`

```julia
# 01_master_files.jl
# Descriptive stats for all master files (small, runs in <1 minute).
# Covers:
#   - Section 1 sample prep (security coverage)
#   - Section 2 country mapping (institutions, funds)
#   - Section 9 heterogeneity audit (pension fund candidates)

include("00_setup.jl")

con = dbcon()

# ============================================================
# (1) INSTITUTIONS MASTER
# ============================================================
println("\n========== Factset_LionShares_Institutions ==========")

insts = qdf(con, """
    SELECT * FROM read_csv_auto('$(replace(INSTITUTIONS_PATH, "\\" => "/"))',
                                 compression='gzip')
""")
println("rows: $(nrow(insts)), unique entity_id: $(length(unique(insts.FACTSET_ENTITY_ID)))")

# Country distribution
country_dist = qdf(con, """
    SELECT ISO_COUNTRY, COUNT(*) AS n
    FROM read_csv_auto('$(replace(INSTITUTIONS_PATH, "\\" => "/"))', compression='gzip')
    GROUP BY ISO_COUNTRY ORDER BY n DESC LIMIT 20
""")
CSV.write(joinpath(OUT_DIR, "01_inst_country_top20.csv"), country_dist)
println("Top investor countries -> 01_inst_country_top20.csv")
println(first(country_dist, 10))

# Manager style breakdown
style_dist = qdf(con, """
    SELECT MANAGER_STYLE, COUNT(*) AS n
    FROM read_csv_auto('$(replace(INSTITUTIONS_PATH, "\\" => "/"))', compression='gzip')
    GROUP BY MANAGER_STYLE ORDER BY n DESC
""")
CSV.write(joinpath(OUT_DIR, "01_inst_manager_style.csv"), style_dist)
println("\nManager style -> 01_inst_manager_style.csv")
println(style_dist)

# ============================================================
# (2) HETEROGENEITY AUDIT: PENSION / STATE / CONSTRAINED CANDIDATES
# ============================================================
# Section 9 — find candidate "constrained" US institutions by name pattern.
# Patterns based on US public pension fund nomenclature.
println("\n========== Section 9 Heterogeneity Audit ==========")

const PENSION_PATTERNS = [
    "PENSION", "RETIREMENT", "RETIREES",
    "CALPERS", "CALSTRS", "TEACHERS",
    "PUBLIC EMPLOYEES", "STATE EMPLOYEES", "MUNICIPAL",
    "FIREFIGHTERS", "POLICE", "JUDICIAL",
    "ENDOWMENT", "FOUNDATION"
]

# Build SQL LIKE pattern
pension_like = join(["UPPER(ENTITY_PROPER_NAME) LIKE '%$p%'" for p in PENSION_PATTERNS], " OR ")

constrained = qdf(con, """
    SELECT FACTSET_ENTITY_ID, ENTITY_PROPER_NAME, ISO_COUNTRY,
           MANAGER_STYLE, TOTAL_AUM
    FROM read_csv_auto('$(replace(INSTITUTIONS_PATH, "\\" => "/"))', compression='gzip')
    WHERE ISO_COUNTRY = 'US'
      AND ($pension_like)
    ORDER BY TOTAL_AUM DESC NULLS LAST
""")
CSV.write(joinpath(OUT_DIR, "01_us_constrained_candidates.csv"), constrained)
println("US constrained candidates: $(nrow(constrained))")
println("  -> 01_us_constrained_candidates.csv")
println("Top 20 by AUM:")
println(first(constrained, 20))

# US hedge fund candidates (unconstrained)
hedge = qdf(con, """
    SELECT FACTSET_ENTITY_ID, ENTITY_PROPER_NAME, ISO_COUNTRY,
           MANAGER_STYLE, TOTAL_AUM
    FROM read_csv_auto('$(replace(INSTITUTIONS_PATH, "\\" => "/"))', compression='gzip')
    WHERE ISO_COUNTRY = 'US'
      AND MANAGER_STYLE = 'Hedge Fund'
    ORDER BY TOTAL_AUM DESC NULLS LAST
""")
CSV.write(joinpath(OUT_DIR, "01_us_hedge_candidates.csv"), hedge)
println("\nUS hedge fund (unconstrained candidates): $(nrow(hedge))")

# ============================================================
# (3) FUNDS MASTER
# ============================================================
println("\n========== Factset_LionShares_Funds ==========")

fund_type = qdf(con, """
    SELECT FUND_TYPE, COUNT(*) AS n,
           SUM(PORTFOLIO_VALUE)/1e9 AS total_pv_billions
    FROM read_csv_auto('$(replace(FUNDS_PATH, "\\" => "/"))', compression='gzip')
    GROUP BY FUND_TYPE ORDER BY n DESC
""")
CSV.write(joinpath(OUT_DIR, "01_fund_type.csv"), fund_type)
println("Fund type breakdown -> 01_fund_type.csv")
println(fund_type)

# Pension plan funds (FUND_TYPE = PLP) — alternate constrained candidate signal
plp = qdf(con, """
    SELECT f.FACTSET_FUND_ID, f.FACTSET_ENTITY_ID, f.ENTITY_PROPER_NAME,
           f.ISO_COUNTRY, f.PORTFOLIO_VALUE
    FROM read_csv_auto('$(replace(FUNDS_PATH, "\\" => "/"))', compression='gzip') f
    WHERE f.FUND_TYPE = 'PLP' AND f.ISO_COUNTRY = 'US'
    ORDER BY f.PORTFOLIO_VALUE DESC NULLS LAST
""")
CSV.write(joinpath(OUT_DIR, "01_us_plp_funds.csv"), plp)
println("\nUS PLP (pension plan) funds: $(nrow(plp))")

# ============================================================
# (4) SECURITY COVERAGE — for Section 1 sample selection
# ============================================================
# Purpose: report how many European securities are in FactSet's master
# universe (upper bound on potential sample). NOT the same as 03's
# diagnostic, which counts securities ACTUALLY HELD by institutions.
#
# All filters are made EXPLICIT here (no silent EQ+AD or ACTIVE=1) so the
# user can see the cascade and decide where to cut.
println("\n========== Factset_Security_coverage ==========")

EU_countries = "('GB','DE','FR','NL','CH','IT','ES','SE','DK','NO','FI','BE','AT','IE','LU','PT','PL','CZ','HU','GR','RO','SK','SI','BG','HR','EE','LV','LT')"

# Full breakdown: ISSUE_TYPE × ACTIVE — see what each filter drops
eu_full = qdf(con, """
    SELECT ISSUE_TYPE, ACTIVE, COUNT(*) AS n
    FROM read_csv_auto('$(replace(SEC_COVERAGE_PATH, "\\" => "/"))', compression='gzip')
    WHERE ISO_COUNTRY IN $EU_countries
    GROUP BY ISSUE_TYPE, ACTIVE
    ORDER BY ISSUE_TYPE, ACTIVE
""")
CSV.write(joinpath(OUT_DIR, "01_eu_universe_breakdown.csv"), eu_full)
println("\nEuropean securities by ISSUE_TYPE x ACTIVE:")
println(eu_full)

# Cascade: show how the sample shrinks at each filter step
cascade = qdf(con, """
    SELECT
        COUNT(*) AS total_european,
        SUM(CASE WHEN ACTIVE = 1 THEN 1 ELSE 0 END) AS active_only,
        SUM(CASE WHEN ACTIVE = 1 AND ISSUE_TYPE IN ('EQ','AD') THEN 1 ELSE 0 END) AS active_and_eq_ad,
        SUM(CASE WHEN ACTIVE = 1 AND ISSUE_TYPE = 'EQ' THEN 1 ELSE 0 END) AS active_and_eq_only,
        SUM(CASE WHEN ACTIVE = 1 AND ISSUE_TYPE IN ('EQ','AD')
                  AND CAP_GROUP IN ('MEGA','LARGE','MID') THEN 1 ELSE 0 END) AS active_eq_ad_midplus,
        SUM(CASE WHEN ACTIVE = 1 AND ISSUE_TYPE IN ('EQ','AD')
                  AND CAP_GROUP IN ('MEGA','LARGE') THEN 1 ELSE 0 END) AS active_eq_ad_largeplus,
        SUM(CASE WHEN ACTIVE = 1 AND ISSUE_TYPE IN ('EQ','AD')
                  AND CAP_GROUP = 'MEGA' THEN 1 ELSE 0 END) AS active_eq_ad_mega
    FROM read_csv_auto('$(replace(SEC_COVERAGE_PATH, "\\" => "/"))', compression='gzip')
    WHERE ISO_COUNTRY IN $EU_countries
""")
CSV.write(joinpath(OUT_DIR, "01_eu_universe_cascade.csv"), cascade)
println("\nEuropean security-universe cascade (each col is one further filter):")
println(cascade)

# Country × cap, no ISSUE_TYPE filter, but ACTIVE=1 (most actionable cut)
eu_by_country_cap = qdf(con, """
    SELECT ISO_COUNTRY, CAP_GROUP, COUNT(*) AS n
    FROM read_csv_auto('$(replace(SEC_COVERAGE_PATH, "\\" => "/"))', compression='gzip')
    WHERE ISO_COUNTRY IN $EU_countries AND ACTIVE = 1
    GROUP BY ISO_COUNTRY, CAP_GROUP
    ORDER BY ISO_COUNTRY, CAP_GROUP
""")
CSV.write(joinpath(OUT_DIR, "01_eu_universe_by_country_cap.csv"), eu_by_country_cap)
println("\nEuropean ACTIVE securities by country x cap (no ISSUE_TYPE filter):")
println("  -> 01_eu_universe_by_country_cap.csv")
show(stdout, eu_by_country_cap; allrows=true)
println()

DBInterface.close!(con)

println("\n========== DONE ==========")
println("Outputs in $OUT_DIR")
println("Files written: 01_*.csv (7 files)")
println("\nNext: run 02_china_exposure.jl")
```

### `02_china_exposure.jl`

```julia
# 02_china_exposure.jl
# Section 5 — build China exposure measure from Revere supply chain data.
# Uses both directions (Path 1: EU=source, CN=target; Path 2: CN=source, EU=target).
# Runs in ~3-5 minutes on 8-10GB RAM machine.
#
# Audited 2026-06-01 — see AUDIT_2026_06_01_julia_descriptive.md.
# Critical-fix changes vs prior version:
#
# C5 (MOST_RECENT look-ahead): The previous version dedup'd Revere companies
#     to most-recent row per company_id, then stamped that row's home_region
#     / CUSIP / ISIN / SEDOL onto every quarter back to 2003. A firm that
#     redomiciled to the EU in 2020 entered the EU universe for 2005.
#     Fix: build a time-versioned (rev_co_asof_q) table that resolves each
#     company's attributes AS-OF every quarter-end, and use it for the EU
#     universe membership filter AND for edge SRC/TGT region classification.
#     Static rev_co (now using FIRST_VALUE IGNORE NULLS per field, with a
#     deterministic company_id tiebreaker) is preserved for non-time-sensitive
#     uses but is NEVER used for membership or region classification.
#
# C4 (asymmetric edge counting): The previous eu_any_edge denominator's
#     Path 2 had `WHERE src_region <> tgt_region`, which dropped legitimate
#     EU-EU edges from the target firm's denominator. Numerator (eu_china_edge)
#     had no analogous filter, so china_share was inflated for firms with
#     predominantly inbound EU links.
#     Fix: remove the `<>` filter. Each EU firm counts edges where IT is
#     either source or target; EU-EU edges contribute once to each endpoint's
#     denominator — symmetric with how every other edge is counted.
#
# Medium fixes also applied here:
#   - Removed firm_month_china_exposure.parquet alias (silent type-pun;
#     month_end column held quarter-end values). Downstream now reads
#     firm_quarter_china_exposure.parquet directly.
#   - Added deterministic tiebreaker (, company_id ASC) to every ROW_NUMBER.
#   - Sentinel '4000-01-01' replaced with NULL at load time.
#   - dedup-edge diagnostic on rev_rel.
#   - Atomic writes via atomic_copy_to from 00_setup.

include("00_setup.jl")

# Source paths feeding this step (recorded in manifest)
const STEP_INPUTS = [REVERE_CO_PATH, REVERE_REL_PATH]

con = dbcon()

# ============================================================
# (1) Load Revere company master (RAW). Diagnostics on within-company variation.
# ============================================================
println("Loading Revere company master (raw, time-versioned)...")

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE rev_co_raw AS
    SELECT company_id,
           home_region,
           country,
           cusip,
           isin,
           sedol,
           covered,
           CAST(start_ AS DATE) AS start_d_raw,
           CAST(end_   AS DATE) AS end_d_raw,
           -- Normalise sentinel '4000-01-01' (and similar) to NULL at load time
           -- so downstream BETWEEN checks behave correctly.
           CAST(start_ AS DATE) AS start_d,
           CASE WHEN CAST(end_ AS DATE) >= DATE '4000-01-01' THEN NULL
                ELSE CAST(end_ AS DATE) END AS end_d
    FROM read_csv_auto('$(replace(REVERE_CO_PATH, "\\" => "/"))', sample_size=-1)
""")

n_raw = qdf(con, "SELECT COUNT(*) AS n FROM rev_co_raw").n[1]
n_unique = qdf(con, "SELECT COUNT(DISTINCT company_id) AS n FROM rev_co_raw").n[1]
println("  raw rows: $n_raw ; unique company_id: $n_unique ; avg rows per company: $(round(n_raw/n_unique, digits=2))")

# Sentinel diagnostic
sentinel_diag = qdf(con, """
    SELECT COUNT(*) FILTER (WHERE end_d_raw >= DATE '4000-01-01') AS n_sentinel_end,
           COUNT(*) FILTER (WHERE end_d IS NULL)                  AS n_null_end_post_norm
    FROM rev_co_raw
""")
println("  sentinel end_d normalised:")
println(sentinel_diag)

# Within-company-id variation diagnostic
println("\n--- Within-company-id variation diagnostic ---")
diag = qdf(con, """
    SELECT
        SUM(CASE WHEN n_regions > 1 THEN 1 ELSE 0 END) AS n_co_multi_region,
        SUM(CASE WHEN n_countries > 1 THEN 1 ELSE 0 END) AS n_co_multi_country,
        SUM(CASE WHEN n_cusips > 1 THEN 1 ELSE 0 END) AS n_co_multi_cusip,
        SUM(CASE WHEN n_isins > 1 THEN 1 ELSE 0 END) AS n_co_multi_isin,
        SUM(CASE WHEN n_sedols > 1 THEN 1 ELSE 0 END) AS n_co_multi_sedol,
        COUNT(*) AS n_co_total
    FROM (
        SELECT company_id,
               COUNT(DISTINCT home_region) AS n_regions,
               COUNT(DISTINCT country)     AS n_countries,
               COUNT(DISTINCT cusip)       AS n_cusips,
               COUNT(DISTINCT isin)        AS n_isins,
               COUNT(DISTINCT sedol)       AS n_sedols
        FROM rev_co_raw
        GROUP BY company_id
    )
""")
println(diag)
println("  -> % companies with >1 home_region: $(round(100*diag.n_co_multi_region[1]/diag.n_co_total[1], digits=2))%")
println("  -> % companies with >1 country:     $(round(100*diag.n_co_multi_country[1]/diag.n_co_total[1], digits=2))%")
println("  -> % companies with >1 cusip:       $(round(100*diag.n_co_multi_cusip[1]/diag.n_co_total[1], digits=2))%")
println("  -> % companies with >1 isin:        $(round(100*diag.n_co_multi_isin[1]/diag.n_co_total[1], digits=2))%")
println("  -> % companies with >1 sedol:       $(round(100*diag.n_co_multi_sedol[1]/diag.n_co_total[1], digits=2))%")
CSV.write(joinpath(OUT_DIR, "02_revere_co_variation_diag.csv"), diag)

# ============================================================
# (1a) Static dedup — FIRST_VALUE IGNORE NULLS per field, deterministic.
# Used for ID-based joins where time-invariance is acceptable. NEVER used
# for membership filters or for edge region classification — those use the
# time-versioned rev_co_asof_q built below.
# ============================================================
println("\n--- Static dedup (FIRST_VALUE IGNORE NULLS per field, company_id tiebreaker) ---")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE rev_co AS
    SELECT DISTINCT
        company_id,
        FIRST_VALUE(home_region IGNORE NULLS) OVER w AS home_region,
        FIRST_VALUE(country     IGNORE NULLS) OVER w AS country,
        FIRST_VALUE(cusip       IGNORE NULLS) OVER w AS cusip,
        FIRST_VALUE(isin        IGNORE NULLS) OVER w AS isin,
        FIRST_VALUE(sedol       IGNORE NULLS) OVER w AS sedol
    FROM rev_co_raw
    WINDOW w AS (
        PARTITION BY company_id
        ORDER BY end_d DESC NULLS FIRST, start_d DESC, company_id ASC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )
""")
DBInterface.execute(con, """
    ALTER TABLE rev_co ADD COLUMN ever_covered INTEGER DEFAULT 0
""")
DBInterface.execute(con, """
    UPDATE rev_co SET ever_covered = (
        SELECT MAX(CASE WHEN r.covered='Y' THEN 1 ELSE 0 END)
        FROM rev_co_raw r WHERE r.company_id = rev_co.company_id
    )
""")
n_co = qdf(con, "SELECT COUNT(*) AS n FROM rev_co").n[1]
println("  unique Revere companies after static dedup: $n_co")

# Country breakdown of STATIC home_region (for reference only — universe
# membership uses the time-versioned table below).
co_country = qdf(con, """
    SELECT home_region, COUNT(*) AS n
    FROM rev_co
    WHERE home_region IS NOT NULL
    GROUP BY home_region ORDER BY n DESC LIMIT 25
""")
CSV.write(joinpath(OUT_DIR, "02_revere_company_country_top25.csv"), co_country)

# ============================================================
# (1b) Quarter calendar — single source for time-versioning + edge filter.
# Spans 2003-Q1 (Revere coverage begin) through 2025-Q2.
# ============================================================
const PANEL_START_QUARTER_END = "2003-03-31"
const PANEL_END_QUARTER_END   = "2025-06-30"

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE quarters AS
    SELECT LAST_DAY(MAKE_DATE(y, m, 1)) AS qend
    FROM range(2003, 2026) y(y)
    CROSS JOIN (VALUES (3),(6),(9),(12)) AS month_tbl(m)
    WHERE LAST_DAY(MAKE_DATE(y, m, 1)) BETWEEN DATE '$PANEL_START_QUARTER_END'
                                           AND DATE '$PANEL_END_QUARTER_END'
    ORDER BY qend
""")

# ============================================================
# (1c) Time-versioned company attributes — AS-OF every quarter-end.
# For each (company_id, qend), pick the rev_co_raw row that was active at qend
# (start_d <= qend AND (end_d IS NULL OR end_d >= qend)). If multiple match,
# prefer the row with the latest start_d, then latest end_d, then company_id
# for a deterministic tiebreaker.
#
# This is the CANONICAL source of company attributes (esp. home_region) for
# membership filters and edge classification. NEVER use rev_co for those.
# ============================================================
println("\nBuilding time-versioned rev_co_asof_q (point-in-time attributes per quarter)...")

# (1c-pre) Deterministic dedup at the (company_id, start_d) grain. This is
# shared by rev_co_asof_q and the rev_rel src/tgt joins below. Pre-deduping
# the history avoids tiebreaker ambiguity inside ASOF JOIN — ASOF picks the
# row with the LATEST start_d <= probe; we resolve same-start_d collisions
# here so that ASOF's pick is unique.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE rev_co_dedup AS
    SELECT company_id, home_region, country, cusip, isin, sedol, start_d, end_d
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY company_id, start_d
                   ORDER BY end_d DESC NULLS LAST, company_id ASC
               ) AS rn
        FROM rev_co_raw
    )
    WHERE rn = 1
""")
n_dedup = qdf(con, "SELECT COUNT(*) AS n FROM rev_co_dedup").n[1]
println("  rev_co_dedup rows (one per (company_id, start_d)): $n_dedup")

# (1c) ASOF JOIN version of rev_co_asof_q.
# For each (company_id, qend) probe, ASOF selects the latest history row with
# start_d <= qend. We then filter on end_d to enforce interval coverage —
# equivalent to the original BETWEEN-style join but linear in 'companies x
# quarters' instead of cross-product * ROW_NUMBER cost.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE rev_co_asof_q AS
    WITH co_ids AS (
        SELECT DISTINCT company_id FROM rev_co_dedup
    ),
    probe AS (
        SELECT c.company_id, q.qend
        FROM co_ids c
        CROSS JOIN quarters q
    )
    SELECT p.qend, p.company_id, h.home_region, h.country, h.cusip, h.isin, h.sedol
    FROM probe p
    ASOF LEFT JOIN rev_co_dedup h
      ON p.company_id = h.company_id
     AND p.qend >= h.start_d
    WHERE h.company_id IS NOT NULL
      AND (h.end_d IS NULL OR h.end_d >= p.qend)
""")

# Diagnostic: how often does the as-of attribute differ from static rev_co?
asof_drift = qdf(con, """
    SELECT
        COUNT(*) AS n_pairs,
        SUM(CASE WHEN a.home_region IS NULL THEN 1 ELSE 0 END)            AS n_asof_null_region,
        SUM(CASE WHEN c.home_region <> a.home_region THEN 1 ELSE 0 END)   AS n_region_drift,
        SUM(CASE WHEN c.cusip       <> a.cusip       THEN 1 ELSE 0 END)   AS n_cusip_drift,
        SUM(CASE WHEN c.isin        <> a.isin        THEN 1 ELSE 0 END)   AS n_isin_drift
    FROM rev_co_asof_q a
    JOIN rev_co c USING (company_id)
""")
println("As-of vs static (drift counts; non-zero = real look-ahead in old version):")
println(asof_drift)
CSV.write(joinpath(OUT_DIR, "02_rev_co_asof_drift_diag.csv"), asof_drift)

# ============================================================
# (1d) Supply-chain relationships, with src/tgt attributes joined AS-OF rel_start.
# Each edge classified by what its endpoints were AT THE TIME it began, not
# what they are today. Eliminates the look-ahead in edge classification.
# ============================================================
println("\nLoading supply chain relationships (5.5M rows) + AS-OF rel_start join...")

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE rev_rel_raw AS
    SELECT  r.source_company_id,
            r.target_company_id,
            r.rel_type,
            CAST(r.start_ AS DATE) AS rel_start,
            CASE WHEN CAST(r.end_ AS DATE) >= DATE '4000-01-01' THEN NULL
                 ELSE CAST(r.end_ AS DATE) END AS rel_end,
            r.revenue_percent
    FROM read_csv_auto('$(replace(REVERE_REL_PATH, "\\" => "/"))', sample_size=-1) r
""")

# Dedup-edge diagnostic — per audit
dup_edge_diag = qdf(con, """
    SELECT COUNT(*) AS n_duplicate_edges
    FROM (
        SELECT source_company_id, target_company_id, rel_type, rel_start, rel_end,
               COUNT(*) AS c
        FROM rev_rel_raw
        GROUP BY 1,2,3,4,5
        HAVING COUNT(*) > 1
    )
""")
println("Duplicate-edge diagnostic on rev_rel: $(dup_edge_diag.n_duplicate_edges[1]) duplicate keys")

# For AS-OF lookup at rel_start, use the pre-deduped rev_co_dedup history
# (deterministic tiebreaker). Two chained ASOF LEFT JOINs (src then tgt)
# replace the previous BETWEEN-style join + ROW_NUMBER. We then NULL-out the
# region/cusip/isin/sedol columns when the picked history row's end_d is
# strictly before rel_start (the edge fell into a gap in the company's
# coverage). This preserves the original semantics: only history rows whose
# [start_d, end_d] interval covers rel_start contribute a region tag.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE rev_rel AS
    WITH src_asof AS (
        SELECT r.source_company_id, r.target_company_id, r.rel_type, r.rel_start, r.rel_end,
               r.revenue_percent,
               s.home_region AS src_region_raw,
               s.cusip       AS src_cusip_raw,
               s.isin        AS src_isin_raw,
               s.sedol       AS src_sedol_raw,
               s.end_d       AS src_end_d
        FROM rev_rel_raw r
        ASOF LEFT JOIN rev_co_dedup s
          ON r.source_company_id = s.company_id
         AND r.rel_start >= s.start_d
    ),
    src_picked AS (
        SELECT source_company_id, target_company_id, rel_type, rel_start, rel_end,
               revenue_percent,
               CASE WHEN src_end_d IS NULL OR src_end_d >= rel_start THEN src_region_raw ELSE NULL END AS src_region,
               CASE WHEN src_end_d IS NULL OR src_end_d >= rel_start THEN src_cusip_raw  ELSE NULL END AS src_cusip,
               CASE WHEN src_end_d IS NULL OR src_end_d >= rel_start THEN src_isin_raw   ELSE NULL END AS src_isin,
               CASE WHEN src_end_d IS NULL OR src_end_d >= rel_start THEN src_sedol_raw  ELSE NULL END AS src_sedol
        FROM src_asof
    ),
    tgt_asof AS (
        SELECT sp.*,
               t.home_region AS tgt_region_raw,
               t.cusip       AS tgt_cusip_raw,
               t.isin        AS tgt_isin_raw,
               t.sedol       AS tgt_sedol_raw,
               t.end_d       AS tgt_end_d
        FROM src_picked sp
        ASOF LEFT JOIN rev_co_dedup t
          ON sp.target_company_id = t.company_id
         AND sp.rel_start >= t.start_d
    )
    SELECT source_company_id, target_company_id, rel_type, rel_start, rel_end,
           revenue_percent,
           src_region, src_cusip, src_isin, src_sedol,
           CASE WHEN tgt_end_d IS NULL OR tgt_end_d >= rel_start THEN tgt_region_raw ELSE NULL END AS tgt_region,
           CASE WHEN tgt_end_d IS NULL OR tgt_end_d >= rel_start THEN tgt_cusip_raw  ELSE NULL END AS tgt_cusip,
           CASE WHEN tgt_end_d IS NULL OR tgt_end_d >= rel_start THEN tgt_isin_raw   ELSE NULL END AS tgt_isin,
           CASE WHEN tgt_end_d IS NULL OR tgt_end_d >= rel_start THEN tgt_sedol_raw  ELSE NULL END AS tgt_sedol
    FROM tgt_asof
""")
n_rel = qdf(con, "SELECT COUNT(*) AS n FROM rev_rel").n[1]
println("  relationships (with AS-OF rel_start regions): $n_rel")

# ============================================================
# (2) China-edge view: i has a CN counterparty (either side). Region tags
# are AS-OF the edge's rel_start, so an EU firm that became EU in 2020 does
# NOT have its 2010 edges retroactively reclassified.
# ============================================================
println("\nBuilding China-edge view (Path 1 ∪ Path 2)...")

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE eu_china_edge AS
    -- Path 1: EU company i is source, CN counterparty is target
    SELECT source_company_id AS eu_company_id,
           target_company_id AS cn_company_id,
           rel_type,
           rel_start, rel_end,
           src_cusip AS eu_cusip,
           src_isin AS eu_isin,
           src_sedol AS eu_sedol,
           'EU_SRC' AS path
    FROM rev_rel
    WHERE src_region IN $EU_SQL_TUPLE AND tgt_region = 'CN'

    UNION ALL

    -- Path 2: CN company is source, EU company i is target
    SELECT target_company_id AS eu_company_id,
           source_company_id AS cn_company_id,
           rel_type,
           rel_start, rel_end,
           tgt_cusip AS eu_cusip,
           tgt_isin AS eu_isin,
           tgt_sedol AS eu_sedol,
           'CN_SRC' AS path
    FROM rev_rel
    WHERE src_region = 'CN' AND tgt_region IN $EU_SQL_TUPLE
""")

stats = qdf(con, """
    SELECT
        path,
        COUNT(*) AS n_relations,
        COUNT(DISTINCT eu_company_id) AS n_eu,
        COUNT(DISTINCT cn_company_id) AS n_cn
    FROM eu_china_edge GROUP BY path
""")
println("\nPath breakdown:")
println(stats)
CSV.write(joinpath(OUT_DIR, "02_china_edge_path_summary.csv"), stats)

# Union counts
union_stats = qdf(con, """
    SELECT
        COUNT(*) AS total_relations,
        COUNT(DISTINCT eu_company_id) AS unique_eu_with_cn,
        COUNT(DISTINCT cn_company_id) AS unique_cn_with_eu
    FROM eu_china_edge
""")
println("\nUnion totals:")
println(union_stats)

# ============================================================
# (3) EU "any-edge" view — SYMMETRIC fix (C4).
# Every relation where AT LEAST ONE side is an EU firm contributes once to
# each EU endpoint's denominator. EU-EU edges contribute once to BOTH
# endpoints' denominators (was previously dropped from the target endpoint's
# denominator via `src_region <> tgt_region`, inflating china_share for
# inbound-link firms).
# ============================================================
println("\nBuilding EU any-edge view (SYMMETRIC denominator)...")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE eu_any_edge AS
    -- EU company i is source, any counterparty is target
    SELECT source_company_id AS eu_company_id,
           rel_type,
           rel_start, rel_end,
           'EU_SRC' AS path
    FROM rev_rel
    WHERE src_region IN $EU_SQL_TUPLE

    UNION ALL

    -- Any company is source, EU company i is target.
    -- (C4 FIX): no `src_region <> tgt_region` filter. EU-EU edges are
    -- counted once on each endpoint, symmetric with EU-CN edges.
    SELECT target_company_id AS eu_company_id,
           rel_type,
           rel_start, rel_end,
           'EU_TGT' AS path
    FROM rev_rel
    WHERE tgt_region IN $EU_SQL_TUPLE
""")
any_stats = qdf(con, """
    SELECT COUNT(*) AS n_relations,
           COUNT(DISTINCT eu_company_id) AS n_eu_companies
    FROM eu_any_edge
""")
println("EU any-edge totals (after C4 symmetric fix):")
println(any_stats)

# C4 verification: every EU-EU edge appears in BOTH endpoints' denominators.
c4_check = qdf(con, """
    WITH eu_eu_edges AS (
        SELECT source_company_id, target_company_id, rel_type, rel_start, rel_end
        FROM rev_rel
        WHERE src_region IN $EU_SQL_TUPLE AND tgt_region IN $EU_SQL_TUPLE
    ),
    src_seen AS (
        SELECT COUNT(*) AS n FROM eu_eu_edges e
        JOIN eu_any_edge a
          ON a.eu_company_id = e.source_company_id
         AND a.rel_type = e.rel_type
         AND a.rel_start = e.rel_start
         AND (a.rel_end = e.rel_end OR (a.rel_end IS NULL AND e.rel_end IS NULL))
         AND a.path = 'EU_SRC'
    ),
    tgt_seen AS (
        SELECT COUNT(*) AS n FROM eu_eu_edges e
        JOIN eu_any_edge a
          ON a.eu_company_id = e.target_company_id
         AND a.rel_type = e.rel_type
         AND a.rel_start = e.rel_start
         AND (a.rel_end = e.rel_end OR (a.rel_end IS NULL AND e.rel_end IS NULL))
         AND a.path = 'EU_TGT'
    ),
    eu_eu_total AS (SELECT COUNT(*) AS n FROM eu_eu_edges)
    SELECT (SELECT n FROM eu_eu_total)  AS n_eu_eu_edges,
           (SELECT n FROM src_seen)     AS n_eu_eu_in_src_path,
           (SELECT n FROM tgt_seen)     AS n_eu_eu_in_tgt_path
""")
println("C4 symmetric-counting verification:")
println(c4_check)

# ============================================================
# (4) EU Revere universe — time-versioned. For each quarter q, the set of
# Revere companies whose home_region AT q is in the EU country list.
# (C5 fix.)
# ============================================================
println("\nBuilding time-versioned EU Revere universe (per quarter)...")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE eu_revere_universe_qend AS
    SELECT a.company_id AS eu_company_id,
           a.qend,
           a.home_region AS eu_home_region,
           a.cusip       AS eu_cusip,
           a.isin        AS eu_isin,
           a.sedol       AS eu_sedol
    FROM rev_co_asof_q a
    WHERE a.home_region IN $EU_SQL_TUPLE
""")

eu_univ_qend_stats = qdf(con, """
    SELECT
        COUNT(*) AS n_company_quarters,
        COUNT(DISTINCT eu_company_id) AS n_eu_companies_ever,
        MIN(qend) AS first_qend,
        MAX(qend) AS last_qend
    FROM eu_revere_universe_qend
""")
println("EU Revere universe (time-versioned) stats:")
println(eu_univ_qend_stats)

# For downstream consumers that need a static snapshot (the SAMPLE END-DATE
# universe), we also emit a flat parquet. Note: this is for documentation
# only — sample-membership joins MUST use eu_revere_universe_qend AS-OF the
# joining quarter.
universe_static_path = test_suffix_path(joinpath(OUT_DIR, "eu_revere_universe.parquet"))
atomic_copy_to(con, """
    SELECT DISTINCT
        u.eu_company_id,
        FIRST_VALUE(u.eu_home_region) OVER w AS eu_home_region,
        FIRST_VALUE(u.eu_cusip)       OVER w AS eu_cusip,
        FIRST_VALUE(u.eu_isin)        OVER w AS eu_isin,
        FIRST_VALUE(u.eu_sedol)       OVER w AS eu_sedol
    FROM eu_revere_universe_qend u
    WINDOW w AS (PARTITION BY u.eu_company_id ORDER BY u.qend DESC
                 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
""", universe_static_path)
println("  -> latest-snapshot universe saved to $(basename(universe_static_path))")

# Also emit the time-versioned panel — this is what 05 MUST use.
universe_qend_path = test_suffix_path(joinpath(OUT_DIR, "eu_revere_universe_qend.parquet"))
atomic_copy_to(con, "SELECT * FROM eu_revere_universe_qend", universe_qend_path)
println("  -> time-versioned universe saved to $(basename(universe_qend_path))")
write_manifest("02_eu_revere_universe_qend", universe_qend_path;
               row_count=eu_univ_qend_stats.n_company_quarters[1],
               input_paths=STEP_INPUTS)

# ============================================================
# (5) Firm-QUARTER China exposure panel — uses time-versioned EU universe.
# A firm i appears in the panel for quarter q ONLY if it was an EU firm at q
# (home_region AS-OF q in EU). Pre-2003 quarters are not produced; downstream
# must preserve NULL for those.
# ============================================================
println("\nBuilding firm-quarter exposure panel (2003-Q1 to 2025-Q2, EU AS-OF q)...")

# CN counts per (firm, quarter) — restricted to EU firms AS-OF q.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE firm_quarter_cn AS
    SELECT
        e.eu_company_id,
        u.eu_cusip,
        u.eu_isin,
        u.eu_sedol,
        q.qend,
        SUM(CASE WHEN e.rel_type = 'CUSTOMER' THEN 1 ELSE 0 END) AS n_cn_customer,
        SUM(CASE WHEN e.rel_type = 'SUPPLIER' THEN 1 ELSE 0 END) AS n_cn_supplier,
        SUM(CASE WHEN e.rel_type = 'PARTNER-JVENTUR' THEN 1 ELSE 0 END) AS n_cn_jv,
        SUM(CASE WHEN e.rel_type = 'PARTNER-MANUFAC' THEN 1 ELSE 0 END) AS n_cn_manuf,
        SUM(CASE WHEN e.rel_type LIKE 'PARTNER%' THEN 1 ELSE 0 END) AS n_cn_partner_any,
        COUNT(*) AS n_cn_total
    FROM eu_china_edge e
    JOIN quarters q
      ON q.qend >= e.rel_start
     AND (e.rel_end IS NULL OR q.qend <= e.rel_end)
    JOIN eu_revere_universe_qend u
      ON u.eu_company_id = e.eu_company_id
     AND u.qend          = q.qend
    GROUP BY e.eu_company_id, u.eu_cusip, u.eu_isin, u.eu_sedol, q.qend
""")

# Total link counts per (firm, quarter) — DENOMINATOR for share. SYMMETRIC.
# Also restricted to EU firms AS-OF q.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE firm_quarter_total AS
    SELECT
        a.eu_company_id,
        q.qend,
        COUNT(*) AS n_total_links
    FROM eu_any_edge a
    JOIN quarters q
      ON q.qend >= a.rel_start
     AND (a.rel_end IS NULL OR q.qend <= a.rel_end)
    JOIN eu_revere_universe_qend u
      ON u.eu_company_id = a.eu_company_id
     AND u.qend          = q.qend
    GROUP BY a.eu_company_id, q.qend
""")

# Join to compute share. Firms in the universe with NO supply-chain link
# (CN or otherwise) at q do not appear; downstream treats their china_share
# as NULL (NOT zero), per pre-2003 convention.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE firm_quarter_exposure AS
    SELECT
        t.eu_company_id,
        u.eu_cusip,
        u.eu_isin,
        u.eu_sedol,
        t.qend AS quarter_end,
        COALESCE(c.n_cn_customer, 0) AS n_cn_customer,
        COALESCE(c.n_cn_supplier, 0) AS n_cn_supplier,
        COALESCE(c.n_cn_jv,       0) AS n_cn_jv,
        COALESCE(c.n_cn_manuf,    0) AS n_cn_manuf,
        COALESCE(c.n_cn_partner_any, 0) AS n_cn_partner_any,
        COALESCE(c.n_cn_total,    0) AS n_cn_total,
        t.n_total_links,
        CAST(COALESCE(c.n_cn_total, 0) AS DOUBLE) / NULLIF(t.n_total_links, 0) AS china_share
    FROM firm_quarter_total t
    JOIN eu_revere_universe_qend u
      ON u.eu_company_id = t.eu_company_id AND u.qend = t.qend
    LEFT JOIN firm_quarter_cn c
        ON t.eu_company_id = c.eu_company_id AND t.qend = c.qend
""")

n_panel = qdf(con, "SELECT COUNT(*) AS n FROM firm_quarter_exposure").n[1]
n_firms = qdf(con, "SELECT COUNT(DISTINCT eu_company_id) AS n FROM firm_quarter_exposure").n[1]
n_exposed = qdf(con, "SELECT COUNT(*) AS n FROM firm_quarter_exposure WHERE n_cn_total > 0").n[1]
println("  panel rows: $n_panel ; unique EU firms with any supply-chain link: $n_firms ; rows with positive CN exposure: $n_exposed")

# Save panel as parquet for downstream use (atomic write)
firm_quarter_path = test_suffix_path(joinpath(OUT_DIR, "firm_quarter_china_exposure.parquet"))
atomic_copy_to(con, "SELECT * FROM firm_quarter_exposure", firm_quarter_path)
println("  -> saved to $(basename(firm_quarter_path))")
write_manifest("02_firm_quarter_china_exposure", firm_quarter_path;
               row_count=n_panel, input_paths=STEP_INPUTS)

# NOTE: previous version also emitted firm_month_china_exposure.parquet as an
# "alias" — but it contained QUARTER-END dates under a column named month_end,
# which silently misled downstream readers. The alias has been REMOVED per
# audit recommendation. 05 now reads firm_quarter_china_exposure.parquet
# directly.

# ============================================================
# (6) Descriptive: distribution of share-based exposure at one snapshot
# ============================================================
const SNAP_QUARTER = "2018-12-31"
println("\n========== Descriptive snapshot: $SNAP_QUARTER ==========")

snap = qdf(con, """
    SELECT n_cn_total,
           COUNT(*) AS n_firms
    FROM firm_quarter_exposure
    WHERE quarter_end = DATE '$SNAP_QUARTER' AND n_cn_total > 0
    GROUP BY n_cn_total
    ORDER BY n_cn_total
""")
println("Distribution of CN relation count ($SNAP_QUARTER, firms with positive CN exposure only):")
println(first(snap, 30))
CSV.write(joinpath(OUT_DIR, "02_dist_cn_total_2018.csv"), snap)

# Summary percentiles for both the count and the new share measure
pct = qdf(con, """
    SELECT
        COUNT(*) AS n_firms,
        AVG(n_cn_total) AS mean_count,
        QUANTILE_CONT(n_cn_total, 0.5) AS count_p50,
        QUANTILE_CONT(n_cn_total, 0.75) AS count_p75,
        QUANTILE_CONT(n_cn_total, 0.90) AS count_p90,
        MAX(n_cn_total) AS count_max,
        AVG(china_share) AS mean_share,
        QUANTILE_CONT(china_share, 0.5) AS share_p50,
        QUANTILE_CONT(china_share, 0.75) AS share_p75,
        QUANTILE_CONT(china_share, 0.90) AS share_p90,
        MAX(china_share) AS share_max
    FROM firm_quarter_exposure
    WHERE quarter_end = DATE '$SNAP_QUARTER' AND n_total_links > 0
""")
println("\nPercentiles ($SNAP_QUARTER):")
println(pct)

# Time series: per-quarter mean China share and mean CN relation count
ts = qdf(con, """
    SELECT quarter_end,
           COUNT(*) FILTER (WHERE n_cn_total > 0)            AS n_firms_with_cn,
           COUNT(*)                                          AS n_firms_with_any_link,
           AVG(n_cn_total) FILTER (WHERE n_cn_total > 0)     AS avg_cn_rels,
           AVG(n_cn_customer) FILTER (WHERE n_cn_total > 0)  AS avg_cn_customer,
           AVG(n_cn_supplier) FILTER (WHERE n_cn_total > 0)  AS avg_cn_supplier,
           AVG(n_cn_jv) FILTER (WHERE n_cn_total > 0)        AS avg_cn_jv,
           AVG(china_share)                                  AS avg_china_share,
           AVG(china_share) FILTER (WHERE n_cn_total > 0)    AS avg_china_share_among_exposed
    FROM firm_quarter_exposure
    WHERE n_total_links > 0
    GROUP BY quarter_end ORDER BY quarter_end
""")
CSV.write(joinpath(OUT_DIR, "02_china_exposure_timeseries.csv"), ts)
println("\nTime series -> 02_china_exposure_timeseries.csv")

try
    DBInterface.close!(con)
catch e
    @warn "DBInterface.close! failed" exception=e
end

println("\n========== DONE ==========")
println("Key outputs:")
println("  firm_quarter_china_exposure.parquet  (firm-quarter panel)")
println("  eu_revere_universe.parquet           (latest snapshot — DOC only)")
println("  eu_revere_universe_qend.parquet      (time-versioned — required by 05)")
println("  02_china_edge_path_summary.csv")
println("  02_china_exposure_timeseries.csv")
println("  02_rev_co_asof_drift_diag.csv         (look-ahead drift diagnostic)")
println("\nNext: 03_eom_etl.jl  (heavy — runs ~30-60 min for full panel)")
```

### `03_eom_etl.jl`

```julia
# 03_eom_etl.jl
# Build slim month-end (h, i, t) holdings panel from raw parquet cache.
#
# PREREQUISITE: run 03a_decompress_to_parquet.jl FIRST (one-time, ~30-60 min)
# to populate the raw parquet cache. This script reads from that cache, not
# from the gz files directly, so it now runs in ~1-2 min per chunk.
#
# OUTPUT:
#   holdings_eom.parquet  -- slim quarter-end panel (see column list below)
#
# Audited 2026-06-01 — see AUDIT_2026_06_01_julia_descriptive.md.
# Changes vs prior version:
#   - TEST_MODE now ENV-driven (DPN_TEST_MODE=true) via 00_setup.jl, AND
#     output filenames carry _TESTMODE suffix in test mode so a test run
#     cannot silently overwrite the canonical artifact.
#   - Atomic write via atomic_copy_to (.tmp + verify + mv).
#   - Post-ETL duplicate-key audit: SELECT COUNT(*) over (fund_id, fsym_id,
#     report_date) duplicates and hard-fail above threshold.
#   - Weekend-EOM audit: per-quarter row-counts vs neighbour quarters to
#     surface 2016-12-31 (Saturday) / 2017-12-31 (Sunday) silent drops.
#   - Materialize EOM panel once into a DuckDB table for diagnostics so 8
#     diagnostic queries don't re-scan the parquet 8x.
#   - SKIP_AUDIT now ENV-driven (DPN_SKIP_AUDIT=false to enable Phase A).

include("00_setup.jl")

# TEST_MODE is now read from ENV via 00_setup.jl; do NOT redefine here.
# SKIP_AUDIT defaults to true (audit overhead is non-trivial on a full run);
# set DPN_SKIP_AUDIT=false to run Phase A.
const SKIP_AUDIT = parse(Bool, lowercase(get(ENV, "DPN_SKIP_AUDIT", "true")))

con = dbcon(memory_gb=6, threads=4)

# Source: raw parquet cache produced by 03a_decompress_to_parquet.jl.
holdings_pattern = if TEST_MODE
    replace(joinpath(RAW_PARQUET_DIR, "Factset_FundOwners_2022_2023.parquet"),
            "\\" => "/")
else
    replace(joinpath(RAW_PARQUET_DIR, "Factset_FundOwners_*.parquet"),
            "\\" => "/")
end

const EXPECTED_CHUNKS = TEST_MODE ?
    ["Factset_FundOwners_2022_2023.parquet"] :
    ["Factset_FundOwners_1999_2005.parquet",
     "Factset_FundOwners_2006_2011.parquet",
     "Factset_FundOwners_2012_2013.parquet",
     "Factset_FundOwners_2014_2015.parquet",
     "Factset_FundOwners_2016_2017.parquet",
     "Factset_FundOwners_2018_2019.parquet",
     "Factset_FundOwners_2020_2021.parquet",
     "Factset_FundOwners_2022_2023.parquet"]

let missing_chunks = filter(c -> !isfile(joinpath(RAW_PARQUET_DIR, c)), EXPECTED_CHUNKS)
    if !isempty(missing_chunks)
        error("Raw parquet cache is incomplete. Missing chunks:\n  " *
              join(missing_chunks, "\n  ") *
              "\n\nRun 03a_decompress_to_parquet.jl to build the missing ones.")
    end
end
# Surface unexpected extras as a warning (per audit recommendation)
let actual_chunks = filter(f -> startswith(basename(f), "Factset_FundOwners_") && endswith(f, ".parquet"),
                           readdir(RAW_PARQUET_DIR, join=true))
    extras = setdiff(basename.(actual_chunks), EXPECTED_CHUNKS)
    if !isempty(extras)
        @warn "Raw parquet cache contains unexpected chunks (will still be picked up by wildcard):\n  " * join(extras, "\n  ")
    end
end

if TEST_MODE
    println("\n" * "!"^70)
    println("!!  TEST_MODE = true (DPN_TEST_MODE)                              !!")
    println("!!  Output filenames carry _TESTMODE suffix.                      !!")
    println("!"^70 * "\n")
else
    println("\nETL mode: FULL (all 8 chunks, 1999-2023)")
end
println("Holdings pattern: $holdings_pattern")

eom_path = test_suffix_path(joinpath(OUT_DIR, "holdings_eom.parquet"))
const STEP_INPUTS = [joinpath(RAW_PARQUET_DIR, c) for c in EXPECTED_CHUNKS]

# ============================================================
# PHASE A: PRE-ETL AUDIT (optional via DPN_SKIP_AUDIT)
# ============================================================
if !SKIP_AUDIT
    println("\n========== PHASE A: pre-ETL audit ==========")
    DBInterface.execute(con, """
        CREATE OR REPLACE TABLE audit_cache AS
        SELECT FACTSET_FUND_ID, factset_sec_entity_id, ISSUE_TYPE, ADJ_MV,
               CAST(REPORT_DATE AS DATE) AS report_date,
               EXTRACT(DAY FROM CAST(REPORT_DATE AS DATE)) AS dom
        FROM read_parquet('$holdings_pattern')
        WHERE EXTRACT(YEAR FROM CAST(REPORT_DATE AS DATE)) = 2022
          AND EXTRACT(MONTH FROM CAST(REPORT_DATE AS DATE)) = 1
    """)
    cache_n = qdf(con, "SELECT COUNT(*) AS n FROM audit_cache").n[1]
    println("  cached rows: $cache_n")

    audit_issue = qdf(con, """
        SELECT ISSUE_TYPE, COUNT(*) AS n_rows,
               COUNT(DISTINCT FACTSET_FUND_ID)       AS n_funds,
               COUNT(DISTINCT factset_sec_entity_id) AS n_companies,
               SUM(ADJ_MV)/1e9                       AS total_mv_billions
        FROM audit_cache
        WHERE report_date = DATE '2022-01-31'
        GROUP BY ISSUE_TYPE ORDER BY n_rows DESC
    """)
    println("\nISSUE_TYPE breakdown (2022-01-31):")
    println(audit_issue)
    CSV.write(joinpath(OUT_DIR, "03_audit_issue_type_holdings.csv"), audit_issue)

    audit_dates = qdf(con, "SELECT dom, COUNT(*) AS n_rows FROM audit_cache GROUP BY dom ORDER BY dom")
    println("\nReport-date day-of-month distribution (Jan 2022):")
    println(audit_dates)
    CSV.write(joinpath(OUT_DIR, "03_audit_dom_distribution.csv"), audit_dates)

    DBInterface.execute(con, "DROP TABLE audit_cache")
else
    println("\n========== PHASE A SKIPPED (DPN_SKIP_AUDIT = true) ==========")
end

# ============================================================
# PHASE B: ETL — atomic write via atomic_copy_to
# Filters:
#   * NO ISSUE_TYPE filter (downstream 04/05 apply their own)
#   * ADJ_MV > 0
#   * Quarter-end calendar last day (Mar/Jun/Sep/Dec)
# Weekend-EOM caveat: the LAST_DAY filter drops quarter-ends that fell on a
# weekend if FactSet recorded the snapshot on the prior business day. Audit
# Phase C computes per-quarter row counts so the operator can spot drop-outs;
# if material, switch to DATE_TRUNC('quarter') + INTERVAL approach.
# ============================================================
println("\n========== PHASE B: ETL ==========")
println("Filters: QUARTER-end report date + ADJ_MV > 0 (NO ISSUE_TYPE filter)")
println("Output: $eom_path")

@time atomic_copy_to(con, """
    SELECT
        FACTSET_FUND_ID            AS fund_id,
        ENTITY_PROPER_NAME         AS entity_name,
        ISO_COUNTRY                AS investor_country,
        ENTITY_TYPE                AS entity_type,
        FSYM_ID                    AS fsym_id,
        FSYM_PRIMARY_EQUITY_ID     AS fsym_primary_id,
        LISTING_FLAG               AS listing_flag,
        CUSIP                      AS cusip,
        ISIN                       AS isin,
        SEDOL                      AS sedol,
        SEC_FIRM_ISO_COUNTRY       AS sec_country,
        ISSUE_TYPE                 AS issue_type,
        CAP_GROUP                  AS cap_group,
        factset_sec_entity_id      AS sec_entity_id,
        SEC_ENTITY_PROPER_NAME     AS sec_entity_name,
        CAST(REPORT_DATE AS DATE)  AS report_date,
        ADJ_HOLDING                AS adj_holding,
        ADJ_MV                     AS adj_mv,
        ADJ_SHARES_OUTSTANDING     AS adj_shares_out,
        ADJ_PRICE                  AS adj_price
    FROM read_parquet('$holdings_pattern')
    WHERE ADJ_MV IS NOT NULL
      AND ADJ_MV > 0
      AND CAST(REPORT_DATE AS DATE) = LAST_DAY(CAST(REPORT_DATE AS DATE))
      AND EXTRACT(MONTH FROM CAST(REPORT_DATE AS DATE)) IN (3, 6, 9, 12)
""", eom_path)
eom_path_fwd = replace(eom_path, "\\" => "/")
println("File size: $(round(filesize(eom_path)/1024^3, digits=2)) GB")

# Use a VIEW (not a TABLE) so each diagnostic query streams parquet rows
# rather than materialising 4-5 GB into the 6 GB memory_limit (which OOMs on
# Phase C otherwise). Each query independently scans the parquet; DuckDB's
# query optimiser pushes projection / filter down so the per-query cost is
# manageable.
DBInterface.execute(con, "CREATE OR REPLACE VIEW eom AS SELECT * FROM read_parquet('$eom_path_fwd')")
total_rows = qdf(con, "SELECT COUNT(*) AS n FROM eom").n[1]
println("Total rows in EOM panel: $total_rows")
write_manifest("03_eom_etl", eom_path; row_count=total_rows, input_paths=STEP_INPUTS)

# ============================================================
# PHASE C: POST-ETL DIAGNOSTICS
# ============================================================
println("\n========== PHASE C: diagnostics on holdings_eom (single materialised scan) ==========")

# C1: Duplicate-key check on (fund_id, fsym_id, report_date) — audit hard-flag
dup_check = qdf(con, """
    SELECT COUNT(*) AS n_dup_keys
    FROM (
        SELECT fund_id, fsym_id, report_date, COUNT(*) c
        FROM eom GROUP BY 1,2,3 HAVING COUNT(*) > 1
    )
""")
println("Duplicate (fund_id, fsym_id, report_date) keys: $(dup_check.n_dup_keys[1])")
CSV.write(joinpath(OUT_DIR, "03_eom_dup_key_count.csv"), dup_check)
if dup_check.n_dup_keys[1] > 10000
    @warn "Duplicate key count is high: $(dup_check.n_dup_keys[1]). Downstream SUM(adj_mv) will over-count. Investigate raw chunks."
end

# C2: Weekend-EOM audit — check per-quarter row counts vs neighbours
weekend_audit = qdf(con, """
    SELECT EXTRACT(YEAR FROM report_date) AS yr,
           EXTRACT(MONTH FROM report_date) AS mo,
           EXTRACT(DAY FROM report_date) AS dom,
           COUNT(*) AS n_rows
    FROM eom
    GROUP BY 1,2,3 ORDER BY yr, mo, dom
""")
CSV.write(joinpath(OUT_DIR, "03_eom_weekend_audit.csv"), weekend_audit)
println("Weekend-EOM audit -> 03_eom_weekend_audit.csv (look for quarters with anomalously low row counts).")

# C3: ISSUE_TYPE breakdowns
issue_breakdown = qdf(con, """
    SELECT issue_type, COUNT(*) AS n_rows,
           COUNT(DISTINCT fund_id)       AS n_funds,
           COUNT(DISTINCT sec_entity_id) AS n_companies,
           COUNT(DISTINCT fsym_id)       AS n_securities,
           SUM(adj_mv)/1e9               AS total_mv_b
    FROM eom GROUP BY issue_type ORDER BY n_rows DESC
""")
println("\nISSUE_TYPE breakdown in EOM:")
println(issue_breakdown)
CSV.write(joinpath(OUT_DIR, "03_eom_issue_type_breakdown.csv"), issue_breakdown)

issue_eu = qdf(con, """
    SELECT issue_type, COUNT(*) AS n_rows,
           COUNT(DISTINCT sec_entity_id) AS n_companies,
           SUM(adj_mv)/1e9 AS total_mv_b
    FROM eom WHERE sec_country IN $EU_SQL_TUPLE
    GROUP BY issue_type ORDER BY n_rows DESC
""")
CSV.write(joinpath(OUT_DIR, "03_eom_issue_type_breakdown_europe.csv"), issue_eu)

issue_us_inv = qdf(con, """
    SELECT issue_type, COUNT(*) AS n_rows,
           COUNT(DISTINCT sec_entity_id) AS n_companies,
           SUM(adj_mv)/1e9 AS total_mv_b
    FROM eom WHERE investor_country = 'US'
    GROUP BY issue_type ORDER BY n_rows DESC
""")
CSV.write(joinpath(OUT_DIR, "03_eom_issue_type_breakdown_us_investors.csv"), issue_us_inv)

# C4: Coverage by year
yr = qdf(con, """
    SELECT EXTRACT(YEAR FROM report_date) AS year,
           COUNT(*) AS n_rows,
           COUNT(DISTINCT fund_id) AS n_funds,
           COUNT(DISTINCT sec_entity_id) AS n_firms
    FROM eom GROUP BY year ORDER BY year
""")
CSV.write(joinpath(OUT_DIR, "03_eom_coverage_by_year.csv"), yr)
println("\nYearly coverage:")
println(yr)

# C5: Sample-date investor / company country breakdowns
ic = qdf(con, """
    SELECT investor_country, COUNT(*) AS n_holdings
    FROM eom WHERE report_date = DATE '2018-12-31'
    GROUP BY investor_country ORDER BY n_holdings DESC LIMIT 20
""")
CSV.write(joinpath(OUT_DIR, "03_eom_investor_country_2018_12.csv"), ic)

cc = qdf(con, """
    SELECT sec_country, COUNT(*) AS n_holdings,
           COUNT(DISTINCT sec_entity_id) AS n_firms
    FROM eom WHERE report_date = DATE '2018-12-31'
    GROUP BY sec_country ORDER BY n_holdings DESC LIMIT 20
""")
CSV.write(joinpath(OUT_DIR, "03_eom_company_country_2018_12.csv"), cc)

us_eu_cells = qdf(con, """
    SELECT report_date,
           COUNT(*) AS n_holdings,
           COUNT(DISTINCT fund_id) AS n_us_funds,
           COUNT(DISTINCT sec_entity_id) AS n_eu_firms,
           SUM(adj_mv)/1e9 AS total_mv_billions
    FROM eom
    WHERE investor_country = 'US' AND sec_country IN $EU_SQL_TUPLE
    GROUP BY report_date ORDER BY report_date
""")
CSV.write(joinpath(OUT_DIR, "03_us_x_eu_cells_by_month.csv"), us_eu_cells)
println("\nUS investor × EU firm panel (last 12 quarters):")
println(last(us_eu_cells, 12))

# Free the materialised diagnostic table.
DBInterface.execute(con, "DROP TABLE eom")

try
    DBInterface.close!(con)
catch e
    @warn "DBInterface.close! failed" exception=e
end

println("\n========== ETL DONE ==========")
println("Output: $eom_path")
println("\nKey diagnostic files:")
println("  03_eom_dup_key_count.csv         (hard-flag if non-trivial)")
println("  03_eom_weekend_audit.csv         (look for low-rowcount quarters)")
println("  03_eom_coverage_by_year.csv")
println("  03_eom_issue_type_breakdown.csv")
println("  03_us_x_eu_cells_by_month.csv")
println("\nNext: 04_us_ownership_european.jl")
```

### `04_us_ownership_european.jl`

```julia
# 04_us_ownership_european.jl
# Sections 2-4 descriptive analysis using the EOM panel from script 03.
# Builds:
#   - I_{i,c,t}                   (country-firm aggregation)
#   - w_{i,c,t}    [portfolio_weight_eu]  (EU-restricted; this is the REGRESSION input)
#   - w_{i,c,t}    [portfolio_weight_global]  (global; kept for diagnostic only)
#   - Ownership_{i,US,t}          (US ownership share of European firms)
# And computes distributions, percentiles, top firms.
#
# Audited 2026-06-01 — see AUDIT_2026_06_01_julia_descriptive.md.
# Critical-fix changes vs prior version:
#
# C1 (denominator): The previous portfolio_weight used a GLOBAL denominator —
#     sum of holder-country I across ALL sec_country, not just EU. US weights
#     were deflated ~6-7x relative to the regression spec. Fix: build a parallel
#     country_total_holdings_eu column (denominator restricted to EU sec_country)
#     and emit portfolio_weight_eu alongside portfolio_weight_global. The
#     EU-restricted one is what 05 must use; the global one is kept for the
#     fig2 "US-EU engagement" descriptive only.
#
# Market-cap MAX×MAX (high): The previous MAX(adj_shares_out) × MAX(adj_price)
#     silently selected the upper bound of intra-group dispersion (same firm-MAX
#     leak pattern as the sibling robots project). Fix: use AVG within
#     (sec_entity_id, fsym_id, report_date) — if rows are identical (the usual
#     case), AVG = MIN = MAX so the result is unambiguous. STDDEV per group is
#     captured as a dispersion diagnostic; downstream can flag (sec, fsym, t)
#     groups where stddev > 0.
#
# C6 (selection-on-outcome universe — PARTIAL): The EU firm universe is still
#     derived from FactSet ownership ("firms ever held by ≥1 institution"),
#     not from Factset_Security_coverage. This is selection-on-outcome and
#     remains a STRUCTURAL TODO requiring a fresh build from Factset_Security_coverage
#     + Cartesian (firm × country × quarter) grid + zero-fill. Marked at top
#     of file; do NOT treat the current panel as the regression panel until
#     this is rebuilt. See AUDIT file for the full fix recipe.
#
# Also applied:
#   - ownership_share > 1 cells: diagnostic emitted BEFORE silent drop.
#   - Atomic writes via atomic_copy_to.

include("00_setup.jl")

const EOM_PATH = replace(joinpath(OUT_DIR, test_suffix_path("holdings_eom.parquet")), "\\" => "/")
if !isfile(replace(EOM_PATH, "/" => "\\"))
    error("$EOM_PATH not found. Run 03_eom_etl.jl first.")
end

# Step 04 reads only one external artifact — the EOM panel from 03.
const STEP_INPUTS = [replace(EOM_PATH, "/" => "\\")]

# (kept legacy name for any in-file references; canonical list is in 00_setup.jl)
EU_str = EU_SQL_TUPLE

con = dbcon()

# Runtime banner mirroring 05's end-of-run notice. Printed at script start so
# anyone reading stdout (not just the source comments) sees the PROVISIONAL
# state of the EU firm universe.
println("\n" * "!"^70)
println("!! 04_us_ownership_european.jl — DESCRIPTIVE PIPELINE                !!")
println("!! EU firm universe is currently SELECTION-ON-OUTCOME (C6 deferred). !!")
println("!! Outputs are DESCRIPTIVE ONLY; downstream regression panel must    !!")
println("!! be rebuilt from Factset_Security_coverage + Cartesian grid +      !!")
println("!! zero-fill. See AUDIT_2026_06_01_julia_descriptive.md.             !!")
println("!"^70 * "\n")

# ============================================================
# STRUCTURAL TODO (C6) — flagged at the top so a reviewer cannot miss it.
# The EU firm universe used here is derived from "appears in FactSet ownership
# panel with EU sec_country" — i.e., firms ever held by at least one investor.
# That is selection-on-outcome for a regression whose dependent variable is
# institutional holdings. Extensive-margin exits (firms US investors fully
# exit) silently vanish from the panel. Fix requires:
#   1. Pull Factset_Security_coverage to enumerate ALL EU-listed equities.
#   2. Build a (firm × holder-country × quarter) Cartesian grid.
#   3. LEFT-JOIN ownership_ict onto the grid; zero-fill (i,c,t) cells where
#      the institution did not hold (legitimate zeros, not missingness).
#   4. Drop gap_months=6 hard filter in 05; require only that t-1 and t+1
#      exist on the grid (zeros allowed).
# Until done, the merged panel is descriptive-only; do NOT use as regression
# panel.
# ============================================================

# WARNING: if 03_eom_etl.jl was run with TEST_MODE=true, the EOM panel covers
# only a subset. Time series plots will reflect only those months.
year_range = qdf(con, """
    SELECT MIN(EXTRACT(YEAR FROM report_date)) AS ymin,
           MAX(EXTRACT(YEAR FROM report_date)) AS ymax,
           COUNT(DISTINCT EXTRACT(YEAR FROM report_date)) AS n_years
    FROM read_parquet('$EOM_PATH')
""")
if year_range.n_years[1] < 5
    println("\n" * "!"^70)
    println("WARNING: EOM panel covers only $(year_range.ymin[1])-$(year_range.ymax[1]) ($(year_range.n_years[1]) year(s)).")
    println("This looks like TEST_MODE output from 03_eom_etl.jl.")
    println("Trend plots will NOT represent the full 1999-2023 sample.")
    println("!"^70 * "\n")
end

# ============================================================
# (0) Pre-aggregation duplicate-key audit on holdings panel.
# If the EOM panel has multiple rows per (fund_id, fsym_id, report_date), a
# downstream SUM(adj_mv) silently double-counts dollars. We hard-fail on ANY
# duplicate (measured 0 on the current 186,800,295-row holdings_eom.parquet).
# ============================================================
println("Auditing holdings panel for duplicate keys...")
dup_check = qdf(con, """
    SELECT COUNT(*) AS n_dup_keys
    FROM (
        SELECT fund_id, fsym_id, report_date, COUNT(*) AS c
        FROM read_parquet('$EOM_PATH')
        GROUP BY 1,2,3
        HAVING COUNT(*) > 1
    )
""")
n_dup = dup_check.n_dup_keys[1]
println("  duplicate (fund_id, fsym_id, report_date) keys: $n_dup")
if n_dup > 0
    error("Holdings panel has $n_dup duplicate (fund_id, fsym_id, report_date) keys — " *
          "downstream SUM(adj_mv) would double-count dollars and distort I_ict / portfolio weights. " *
          "Fix the 03_eom_etl.jl dedup rule before proceeding. " *
          "(Neither the build_c6_panel dedup assert nor the weight-sum check catches raw fund-level double-counting.)")
end

# ============================================================
# (1) Build I_{i,c,t} — investor-country × firm × quarter aggregation
# ============================================================
# ISSUE_TYPE filter applied here (NOT in 03):
#   EQ = common equity of operating companies (the main object of analysis)
#   AD = ADR/GDR/depositary receipts (US investors' main path to hold foreign
#        firms — must include or US holdings of EU firms gets under-counted)
# Excluded: OE/ET/CE/UIT (funds), PF/CP (preferred), WT (warrants),
#           AI/EP/DR/BC (alt/private). See 03_eom_issue_type_breakdown.csv.
#
# Atomically write.
# ============================================================
println("Building I_{i,c,t} country-firm aggregation (ISSUE_TYPE in EQ, AD)...")

ict_path = test_suffix_path(joinpath(OUT_DIR, "I_ict_panel.parquet"))

@time atomic_copy_to(con, """
    SELECT
        sec_entity_id,
        sec_country,
        investor_country,
        report_date,
        SUM(adj_mv) AS I_ict
    FROM read_parquet('$EOM_PATH')
    WHERE sec_entity_id IS NOT NULL
      AND investor_country IS NOT NULL
      AND issue_type IN ('EQ', 'AD')
    GROUP BY sec_entity_id, sec_country, investor_country, report_date
""", ict_path)
ict_path_fwd = replace(ict_path, "\\" => "/")

n_ict = qdf(con, "SELECT COUNT(*) AS n FROM read_parquet('$ict_path_fwd')").n[1]
println("  rows in I_ict panel: $n_ict")
write_manifest("04_I_ict_panel", ict_path; row_count=n_ict, input_paths=STEP_INPUTS)

# ============================================================
# (2) MarketCap_{i,t} — built from PRIMARY EQUITY only.
# Fix (high): the previous version used MAX × MAX which silently selected the
# upper bound of intra-group dispersion (firm-MAX leak pattern). Now uses AVG
# within group: if all rows are identical (the usual case), AVG = MIN = MAX
# so the result is exact. STDDEV per group is reported as a dispersion
# diagnostic so a reviewer can spot-check.
# ============================================================
mcap_path = test_suffix_path(joinpath(OUT_DIR, "marketcap_it.parquet"))

@time atomic_copy_to(con, """
    WITH primary_only AS (
        SELECT sec_entity_id, sec_country, report_date, fsym_id,
               AVG(adj_shares_out) AS shares_out,
               AVG(adj_price)      AS price,
               STDDEV_SAMP(adj_shares_out) AS shares_out_stddev,
               STDDEV_SAMP(adj_price)      AS price_stddev,
               COUNT(*) AS n_rows_in_group
        FROM read_parquet('$EOM_PATH')
        WHERE fsym_id = fsym_primary_id
          AND issue_type = 'EQ'
          AND adj_shares_out IS NOT NULL AND adj_shares_out > 0
          AND adj_price IS NOT NULL AND adj_price > 0
        GROUP BY sec_entity_id, sec_country, report_date, fsym_id
    )
    SELECT sec_entity_id, sec_country, report_date,
           SUM(shares_out * price) AS market_cap,
           COUNT(*) AS n_primary_classes,
           MAX(shares_out_stddev) AS max_class_shares_stddev,
           MAX(price_stddev)      AS max_class_price_stddev
    FROM primary_only
    GROUP BY sec_entity_id, sec_country, report_date
""", mcap_path)
mcap_path_fwd = replace(mcap_path, "\\" => "/")

# Diagnostic: how many companies got a clean mcap? Any dispersion?
mcap_diag = qdf(con, """
    SELECT COUNT(*) AS n_company_quarters,
           COUNT(DISTINCT sec_entity_id) AS n_companies,
           AVG(n_primary_classes) AS avg_primary_classes,
           SUM(CASE WHEN max_class_shares_stddev > 0 THEN 1 ELSE 0 END) AS n_shares_dispersion,
           SUM(CASE WHEN max_class_price_stddev > 0 THEN 1 ELSE 0 END)  AS n_price_dispersion
    FROM read_parquet('$mcap_path_fwd')
""")
println("  mcap coverage + dispersion diagnostic:")
println(mcap_diag)
if mcap_diag.n_shares_dispersion[1] > 0 || mcap_diag.n_price_dispersion[1] > 0
    println("  -> $(mcap_diag.n_shares_dispersion[1]) groups had within-group shares dispersion;")
    println("     $(mcap_diag.n_price_dispersion[1]) groups had within-group price dispersion.")
    println("     AVG was used. Spot-check via marketcap_it.parquet if these counts are non-trivial.")
end
write_manifest("04_marketcap_it", mcap_path; row_count=mcap_diag.n_company_quarters[1], input_paths=STEP_INPUTS)

# ============================================================
# (3a) Country totals — TWO versions.
#
# country_total_holdings_GLOBAL  = sum_j I_{j,c,t} across ALL sec_country
# country_total_holdings_EU      = sum_{j in EU} I_{j,c,t}  (regression input)
#
# C1 FIX: the regression spec requires the EU-restricted denominator so US and
# NONUS series use the same scope. The global version is preserved for the
# fig2 descriptive ONLY.
# ============================================================
country_total_path = test_suffix_path(joinpath(OUT_DIR, "country_total_ct.parquet"))

@time atomic_copy_to(con, """
    SELECT
        investor_country,
        report_date,
        SUM(I_ict) AS country_total_holdings_global,
        SUM(CASE WHEN sec_country IN $EU_str THEN I_ict ELSE 0 END) AS country_total_holdings_eu
    FROM read_parquet('$ict_path_fwd')
    GROUP BY investor_country, report_date
""", country_total_path)
country_total_path_fwd = replace(country_total_path, "\\" => "/")
write_manifest("04_country_total_ct", country_total_path; input_paths=STEP_INPUTS)

# ============================================================
# (3b) Ownership_{i,c,t} AND portfolio weight w_{i,c,t}
# Emit BOTH portfolio_weight_eu (regression input) and portfolio_weight_global
# (diagnostic). Downstream / 05 must use portfolio_weight_eu.
# ============================================================
println("\nBuilding Ownership_{i,c,t} and portfolio weight w_{i,c,t} (EU + global)...")

own_path = test_suffix_path(joinpath(OUT_DIR, "ownership_ict.parquet"))

@time atomic_copy_to(con, """
    SELECT i.sec_entity_id,
           i.sec_country,
           i.investor_country,
           i.report_date,
           i.I_ict,
           m.market_cap,
           i.I_ict / NULLIF(m.market_cap, 0) AS ownership_share,
           -- C1-FIX regression input: denominator restricted to EU sec_country
           CASE WHEN i.sec_country IN $EU_str
                THEN i.I_ict / NULLIF(ct.country_total_holdings_eu, 0)
                ELSE NULL END AS portfolio_weight_eu,
           -- Diagnostic only: global denominator (do NOT use in regression)
           i.I_ict / NULLIF(ct.country_total_holdings_global, 0) AS portfolio_weight_global
    FROM read_parquet('$ict_path_fwd') i
    LEFT JOIN read_parquet('$mcap_path_fwd') m
        ON i.sec_entity_id = m.sec_entity_id
       AND i.report_date = m.report_date
    LEFT JOIN read_parquet('$country_total_path_fwd') ct
        ON i.investor_country = ct.investor_country
       AND i.report_date      = ct.report_date
""", own_path)
own_path_fwd = replace(own_path, "\\" => "/")
write_manifest("04_ownership_ict", own_path; input_paths=STEP_INPUTS)

# Sanity-check: sum of portfolio_weight_eu over (investor_country, report_date)
# should be ~1 for EU sec_country only.
pw_sanity = qdf(con, """
    SELECT investor_country, report_date,
           SUM(CASE WHEN sec_country IN $EU_str THEN portfolio_weight_eu ELSE 0 END) AS sum_w_eu
    FROM read_parquet('$own_path_fwd')
    WHERE investor_country = 'US' AND report_date IN (DATE '2018-12-31', DATE '2022-12-31')
    GROUP BY 1,2 ORDER BY 1,2
""")
println("Sanity (sum w_eu in EU per US-quarter; should be ~1):")
println(pw_sanity)

# ============================================================
# (4) Descriptive: US ownership of European firms
# ============================================================
println("\n========== US Ownership of European firms ==========")

# Pick snapshot date: prefer 2018-12-31 if in panel, else most recent month-end
snap_q = qdf(con, """
    SELECT MAX(report_date) AS d
    FROM read_parquet('$own_path_fwd')
    WHERE report_date <= DATE '2018-12-31'
""")
snap_date = (snap_q.d[1] === missing || snap_q.d[1] === nothing || nrow(snap_q) == 0) ?
    qdf(con, "SELECT MAX(report_date) AS d FROM read_parquet('$own_path_fwd')").d[1] :
    snap_q.d[1]
println("Snapshot date for cross-sectional descriptives: $snap_date")

# Diagnostic BEFORE the silent BETWEEN(0,1) drop: how many cells get filtered?
share_filter_diag = qdf(con, """
    SELECT
        COUNT(*) AS n_total,
        SUM(CASE WHEN ownership_share IS NULL THEN 1 ELSE 0 END) AS n_null,
        SUM(CASE WHEN ownership_share < 0  THEN 1 ELSE 0 END) AS n_negative,
        SUM(CASE WHEN ownership_share > 1  THEN 1 ELSE 0 END) AS n_above_one,
        SUM(CASE WHEN ownership_share BETWEEN 0 AND 1 THEN 1 ELSE 0 END) AS n_in_range
    FROM read_parquet('$own_path_fwd')
    WHERE investor_country = 'US'
      AND sec_country IN $EU_str
      AND report_date = DATE '$snap_date'
""")
println("Ownership-share filter diagnostic ($snap_date):")
println(share_filter_diag)
if share_filter_diag.n_above_one[1] > 0
    above_one_top = qdf(con, """
        SELECT sec_entity_id, sec_country, market_cap, I_ict, ownership_share
        FROM read_parquet('$own_path_fwd')
        WHERE investor_country = 'US'
          AND sec_country IN $EU_str
          AND report_date = DATE '$snap_date'
          AND ownership_share > 1
        ORDER BY ownership_share DESC LIMIT 5
    """)
    println("  Top-5 sec_entity_ids with ownership_share > 1 (typically dual-class / inversion):")
    println(above_one_top)
end

us_eu_snap = qdf(con, """
    SELECT sec_entity_id, sec_country, I_ict, market_cap, ownership_share
    FROM read_parquet('$own_path_fwd')
    WHERE investor_country = 'US'
      AND sec_country IN $EU_str
      AND report_date = DATE '$snap_date'
      AND market_cap > 0
""")
println("US ownership cells on $snap_date: $(nrow(us_eu_snap))")

# Drop pathological values (with the above_one_top diagnostic capturing what's lost)
clean = filter(:ownership_share => x -> !ismissing(x) && 0 <= x <= 1, us_eu_snap)
println("After filtering 0 <= ownership <= 1: $(nrow(clean))")

if nrow(clean) == 0
    try
        DBInterface.close!(con)
    catch
    end
    error("Empty snapshot — check EOM panel and ISSUE_TYPE filter in 04.")
end

# Percentiles
qs = quantile(skipmissing(clean.ownership_share), [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
println("\nOwnership share distribution (US holdings of EU firms, $snap_date):")
for (p, q) in zip([10,25,50,75,90,95,99], qs)
    @printf("  P%-3d: %.4f (%.2f%%)\n", p, q, 100q)
end
@printf("  mean: %.4f (%.2f%%)\n", mean(skipmissing(clean.ownership_share)),
        100*mean(skipmissing(clean.ownership_share)))
@printf("  max:  %.4f\n", maximum(skipmissing(clean.ownership_share)))

CSV.write(joinpath(OUT_DIR, "04_us_ownership_eu_snapshot.csv"), clean)

# By country
by_country = qdf(con, """
    SELECT sec_country,
           COUNT(*) AS n_firms,
           AVG(ownership_share) AS mean_us_own,
           QUANTILE_CONT(ownership_share, 0.5) AS median_us_own,
           SUM(I_ict)/1e9 AS total_us_holding_b
    FROM read_parquet('$own_path_fwd')
    WHERE investor_country = 'US'
      AND sec_country IN $EU_str
      AND report_date = DATE '$snap_date'
      AND ownership_share IS NOT NULL AND ownership_share BETWEEN 0 AND 1
    GROUP BY sec_country ORDER BY total_us_holding_b DESC
""")
println("\nUS ownership of EU firms by country ($snap_date):")
println(by_country)
CSV.write(joinpath(OUT_DIR, "04_us_own_by_eu_country_snapshot.csv"), by_country)

# ============================================================
# (5) Time series: US ownership of EU firms over time
# ============================================================
println("\n========== Time series of US ownership in EU ==========")
ts = qdf(con, """
    SELECT report_date,
           COUNT(DISTINCT sec_entity_id) AS n_eu_firms,
           AVG(ownership_share) AS mean_us_own,
           SUM(I_ict)/1e9 AS total_us_holding_b
    FROM read_parquet('$own_path_fwd')
    WHERE investor_country = 'US'
      AND sec_country IN $EU_str
      AND ownership_share IS NOT NULL AND ownership_share BETWEEN 0 AND 1
    GROUP BY report_date ORDER BY report_date
""")
CSV.write(joinpath(OUT_DIR, "04_us_ownership_eu_timeseries.csv"), ts)

# Top 30 EU firms by US ownership at snapshot date
top_us = qdf(con, """
    SELECT sec_entity_id, sec_country, market_cap/1e9 AS mcap_b,
           I_ict/1e9 AS us_holding_b, ownership_share
    FROM read_parquet('$own_path_fwd')
    WHERE investor_country = 'US'
      AND sec_country IN $EU_str
      AND report_date = DATE '$snap_date'
      AND ownership_share BETWEEN 0 AND 1
      AND market_cap > 1e9
    ORDER BY ownership_share DESC LIMIT 30
""")
CSV.write(joinpath(OUT_DIR, "04_top30_us_owned_eu_firms.csv"), top_us)
println("\nTop 30 EU firms by US ownership share at $snap_date (mcap > 1B) -> 04_top30_us_owned_eu_firms.csv")

try
    DBInterface.close!(con)
catch e
    @warn "DBInterface.close! failed" exception=e
end

println("\n========== DONE ==========")
println("Key outputs:")
println("  I_ict_panel.parquet")
println("  ownership_ict.parquet     (portfolio_weight_eu = regression input; portfolio_weight_global = diagnostic)")
println("  marketcap_it.parquet      (AVG-not-MAX; dispersion stddev columns)")
println("  country_total_ct.parquet  (both global and EU-restricted totals)")
println("  04_us_ownership_eu_*.csv")
println("\nNext: 05_combine_visualize.jl  (Sections 6-7 pre-regression checks)")
```

### `05_combine_visualize.jl`

```julia
# 05_combine_visualize.jl
# Sections 6-7 PRE-REGRESSION descriptive checks (audited revision).
#
# !!! IMPORTANT !!!
# This script produces merged_us_eu_matched.parquet, but per the 2026-06-01
# audit (AUDIT_2026_06_01_julia_descriptive.md) the EU firm universe is still
# selection-on-outcome (firms ever held by ≥1 institution in FactSet). DO NOT
# use the output as a REGRESSION PANEL. It is DESCRIPTIVE ONLY. The full fix
# requires rebuilding the universe from Factset_Security_coverage with a
# Cartesian (firm × holder-country × quarter) grid + zero-fill — see C6 in
# the audit file. Until that rebuild lands, treat all numerical claims here
# as PROVISIONAL.
#
# Critical-fix changes in THIS script (per audit):
#
# C1 (denominator): switches to portfolio_weight_eu from 04 for the US side
#     and builds a parallel nonus_aggregate_eu so US and NONUS series share
#     scope. The previous nonus_aggregate used a global denominator and was
#     not comparable to US portfolio_weight (which was also global, see 04
#     fix). Sanity check sum_w_eu ≈ 1 at the US-quarter level is printed.
#
# C2 (pre-2003 NULL): drops the silent COALESCE→0 for pre-Revere quarters.
#     Pre-2003 firm-quarters now carry NULL china_share / n_cn_* and are
#     excluded from the exposure bucket CTE rather than populating ZERO.
#
# C3 (lag): adds LAG(china_share) OVER (PARTITION BY sec_entity_id ORDER BY
#     report_date) as china_share_lag1q to the merged panel. The bucket CTE
#     and any regression interaction MUST use the lagged column.
#
# C5 propagation: switched the merged-panel firm filter from the static
#     matched_eu_sec table to a time-versioned (sec_entity_id, quarter_end)
#     filter that respects the eu_revere_universe_qend AS-OF table built
#     in 02.
#
# High (AR(1)): refit on MONTHLY GPR (the spec). Residuals at the quarter-end
#     month are taken as Shock^{US-CN}_t. Coefficients (a, b) and the full
#     monthly + quarterly shock series are persisted to disk so the regression
#     input is reproducible from the repo alone.
#
# High (multi-match aggregation): replace MAX(n_cn_total) / MAX(china_share)
#     with SUM(n_cn_total) / SUM(n_total_links) ratio — the only self-
#     consistent ratio when one FactSet sec_entity_id maps to multiple Revere
#     companies. DISTINCT applied to the exposure_share_by_sec_entity JOIN.
#
# Other: drop firm_month_china_exposure.parquet alias (now reads
#     firm_quarter_china_exposure.parquet directly); replace hard ±1% trim
#     with winsorization in the descriptive scatter; emit gap_months
#     diagnostic.

include("00_setup.jl")

const EOM_PATH      = replace(joinpath(OUT_DIR, test_suffix_path("holdings_eom.parquet")), "\\" => "/")
const OWN_PATH      = replace(joinpath(OUT_DIR, test_suffix_path("ownership_ict.parquet")), "\\" => "/")
const EXP_PATH      = replace(joinpath(OUT_DIR, test_suffix_path("firm_quarter_china_exposure.parquet")), "\\" => "/")
const UNIV_PATH     = replace(joinpath(OUT_DIR, test_suffix_path("eu_revere_universe.parquet")), "\\" => "/")
const UNIV_QEND_PATH= replace(joinpath(OUT_DIR, test_suffix_path("eu_revere_universe_qend.parquet")), "\\" => "/")
const ICT_PATH      = replace(joinpath(OUT_DIR, test_suffix_path("I_ict_panel.parquet")), "\\" => "/")

for p in (EOM_PATH, OWN_PATH, EXP_PATH, UNIV_PATH, UNIV_QEND_PATH, ICT_PATH)
    if !isfile(replace(p, "/" => "\\"))
        error("$(basename(p)) not found. Run 02 (China exposure) and 03/04 (ETL) first.")
    end
end

# Note: EU_SQL_TUPLE is defined in 00_setup.jl as the single source of truth.
const EUROPE_str = EU_SQL_TUPLE

const STEP_INPUTS = [
    replace(EOM_PATH, "/" => "\\"),
    replace(OWN_PATH, "/" => "\\"),
    replace(EXP_PATH, "/" => "\\"),
    replace(UNIV_QEND_PATH, "/" => "\\"),
    GPR_PATH,
]

con = dbcon()

# Sample-period warning
yr = qdf(con, """
    SELECT MIN(EXTRACT(YEAR FROM report_date)) AS ymin,
           MAX(EXTRACT(YEAR FROM report_date)) AS ymax
    FROM read_parquet('$EOM_PATH')
""")
if yr.ymax[1] - yr.ymin[1] < 5
    println("\n" * "!"^70)
    println("WARNING: EOM panel covers only $(yr.ymin[1])-$(yr.ymax[1]).")
    println("This looks like TEST_MODE output. All time series below are restricted.")
    println("!"^70 * "\n")
end

# ============================================================
# (0) GeoPressure series — AR(1) decomposition on MONTHLY data (audit fix).
# Spec calls for monthly bilateral AI-GPR. The previous version fit AR(1) on
# the quarterly mean, which (a) does not match the spec, (b) attenuates the
# innovation magnitude, and (c) leaves the persistence coefficient on a
# different time scale than the data are documented at. We now fit AR(1) on
# monthly, then take the residual at the quarter-end month as Shock^{US-CN}_t.
# Coefficients and the full series are persisted to disk so the regression
# input is reproducible from the repo alone.
# ============================================================
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE gpr_monthly AS
    SELECT CAST(Date AS DATE)               AS month_first,
           LAST_DAY(CAST(Date AS DATE))     AS month_end,
           "USA|China"                      AS gpr_us_cn,
           GPR_AI                           AS gpr_global
    FROM read_csv_auto('$(replace(GPR_PATH, "\\" => "/"))', sample_size=-1)
    ORDER BY month_first
""")

# AR(1) on monthly series.
gpr_m = qdf(con, "SELECT * FROM gpr_monthly ORDER BY month_end")
println("GPR monthly: $(nrow(gpr_m)) months, $(gpr_m.month_end[1]) → $(gpr_m.month_end[end])")

a_hat_m = 0.0; b_hat_m = 0.0
let g = collect(skipmissing(gpr_m.gpr_us_cn))
    y  = g[2:end]
    yL = g[1:end-1]
    yL_mean = mean(yL); y_mean = mean(y)
    b_hat_m = sum((yL .- yL_mean) .* (y .- y_mean)) / sum((yL .- yL_mean).^2)
    a_hat_m = y_mean - b_hat_m * yL_mean
    println("AR(1) on MONTHLY USA|China GPR: a = $(round(a_hat_m, digits=4)), b = $(round(b_hat_m, digits=4))")
    # Monthly innovation = y_t - (a + b·y_{t-1}). First obs has no lag → missing.
    shock_m = vcat([missing], y .- (a_hat_m .+ b_hat_m .* yL))
    gpr_m.shock_us_cn_monthly = shock_m
end

# Persist monthly coefficients + series.
ar1_coef_csv = joinpath(OUT_DIR, "gpr_ar1_coefficients.csv")
open(ar1_coef_csv, "w") do io
    write(io, "param,value,frequency\n")
    write(io, "a,$(a_hat_m),monthly\n")
    write(io, "b,$(b_hat_m),monthly\n")
    write(io, "n_months,$(nrow(gpr_m)),monthly\n")
    write(io, "sample_first,$(gpr_m.month_end[1]),monthly\n")
    write(io, "sample_last,$(gpr_m.month_end[end]),monthly\n")
end
println("  AR(1) coefficients persisted -> $(basename(ar1_coef_csv))")

DuckDB.register_data_frame(con, gpr_m, "gpr_monthly_with_shock")

gpr_monthly_path = test_suffix_path(joinpath(OUT_DIR, "gpr_monthly_with_shock.parquet"))
atomic_copy_to(con, "SELECT * FROM gpr_monthly_with_shock", gpr_monthly_path)
write_manifest("05_gpr_monthly_with_shock", gpr_monthly_path; row_count=nrow(gpr_m), input_paths=[GPR_PATH])

# For joins to quarterly holdings, take the quarter-end-month innovation as
# Shock^{US-CN}_t. (Robustness via SUM of three monthly innovations is a
# documented next step.)
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE gpr_ts AS
    SELECT month_end AS quarter_end,
           gpr_us_cn,
           gpr_global,
           shock_us_cn_monthly AS shock_us_cn
    FROM gpr_monthly_with_shock
    WHERE EXTRACT(MONTH FROM month_end) IN (3, 6, 9, 12)
""")
gpr_quarterly_path = test_suffix_path(joinpath(OUT_DIR, "gpr_quarterly_with_shock.parquet"))
atomic_copy_to(con, "SELECT * FROM gpr_ts", gpr_quarterly_path)
write_manifest("05_gpr_quarterly_with_shock", gpr_quarterly_path; input_paths=[GPR_PATH])
println("GPR (monthly + quarter-end-month) persisted and registered as gpr_ts.")

# ============================================================
# (1) BUILD MATCHED-EUROPEAN-FIRM CROSSWALK (CUSIP-first).
# FactSet Ownership stores identifiers for European firms mostly in CUSIP.
# Use CUSIP first, keep ISIN/SEDOL as secondary diagnostics.
# ============================================================
println("\nBuilding CUSIP-first crosswalk: FactSet sec_entity_id → Revere company_id")

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE eu_sec_ids AS
    SELECT DISTINCT
           sec_entity_id,
           NULLIF(TRIM(cusip), '') AS cusip,
           NULLIF(TRIM(isin),  '') AS isin,
           NULLIF(TRIM(sedol), '') AS sedol
    FROM read_parquet('$EOM_PATH')
    WHERE sec_country IN $EUROPE_str
      AND sec_entity_id IS NOT NULL
""")

id_cov = qdf(con, """
    SELECT
        COUNT(DISTINCT sec_entity_id) AS n_eu_sec_total,
        COUNT(DISTINCT CASE WHEN cusip IS NOT NULL THEN sec_entity_id END) AS n_eu_sec_with_cusip,
        COUNT(DISTINCT CASE WHEN isin  IS NOT NULL THEN sec_entity_id END) AS n_eu_sec_with_isin,
        COUNT(DISTINCT CASE WHEN sedol IS NOT NULL THEN sec_entity_id END) AS n_eu_sec_with_sedol,
        COUNT(*) AS n_sec_id_rows
    FROM eu_sec_ids
""")
println("  EU identifier coverage in holdings:")
println(id_cov)

# Crosswalk uses the latest-snapshot universe IDs (CUSIP/ISIN/SEDOL).
# Per audit C5: ID stability over the panel is assumed for these joiners
# (the time-versioned EU MEMBERSHIP filter below is what actually closes the
# look-ahead loop).
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE matched_eu_sec_links AS
    WITH univ AS (
        SELECT eu_company_id,
               NULLIF(TRIM(eu_cusip), '') AS eu_cusip,
               NULLIF(TRIM(eu_isin),  '') AS eu_isin,
               NULLIF(TRIM(eu_sedol), '') AS eu_sedol
        FROM read_parquet('$UNIV_PATH')
    )
    SELECT DISTINCT s.sec_entity_id, u.eu_company_id,
           'CUSIP' AS match_type, s.cusip AS matched_id
    FROM eu_sec_ids s
    JOIN univ u ON s.cusip = u.eu_cusip
    WHERE s.cusip IS NOT NULL AND u.eu_cusip IS NOT NULL

    UNION ALL

    SELECT DISTINCT s.sec_entity_id, u.eu_company_id,
           'ISIN_EXACT' AS match_type, s.isin AS matched_id
    FROM eu_sec_ids s
    JOIN univ u ON s.isin = u.eu_isin
    WHERE s.isin IS NOT NULL AND u.eu_isin IS NOT NULL

    UNION ALL

    SELECT DISTINCT s.sec_entity_id, u.eu_company_id,
           'SEDOL' AS match_type, s.sedol AS matched_id
    FROM eu_sec_ids s
    JOIN univ u ON s.sedol = u.eu_sedol
    WHERE s.sedol IS NOT NULL AND u.eu_sedol IS NOT NULL
""")

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE matched_eu_sec AS
    SELECT DISTINCT sec_entity_id FROM matched_eu_sec_links
""")

n_matched = qdf(con, "SELECT COUNT(*) AS n FROM matched_eu_sec").n[1]
n_eu_sec_total = id_cov.n_eu_sec_total[1]
println("  Matched to Revere European universe: $n_matched / $n_eu_sec_total = $(round(100*n_matched/n_eu_sec_total, digits=1))% of EU sec_entity_ids")

# Profile unmatched per audit: characterize the dropped firms.
unmatched_profile = qdf(con, """
    WITH all_eu AS (SELECT DISTINCT sec_entity_id, sec_country FROM read_parquet('$EOM_PATH') WHERE sec_country IN $EUROPE_str)
    SELECT a.sec_country,
           COUNT(*) AS n_all,
           COUNT(*) FILTER (WHERE m.sec_entity_id IS NULL) AS n_unmatched,
           ROUND(100.0 * COUNT(*) FILTER (WHERE m.sec_entity_id IS NULL) / COUNT(*), 2) AS pct_unmatched
    FROM all_eu a
    LEFT JOIN matched_eu_sec m USING (sec_entity_id)
    GROUP BY a.sec_country
    ORDER BY n_unmatched DESC
""")
CSV.write(joinpath(OUT_DIR, "05_unmatched_profile_by_country.csv"), unmatched_profile)
println("  Unmatched profile by country -> 05_unmatched_profile_by_country.csv")

match_type = qdf(con, """
    SELECT match_type,
           COUNT(DISTINCT sec_entity_id) AS n_sec_entities,
           COUNT(*) AS n_links
    FROM matched_eu_sec_links
    GROUP BY match_type ORDER BY n_sec_entities DESC
""")
CSV.write(joinpath(OUT_DIR, "05_match_type_distribution.csv"), match_type)

multi_match = qdf(con, """
    SELECT n_revere_companies, COUNT(*) AS n_sec_entities
    FROM (
        SELECT sec_entity_id, COUNT(DISTINCT eu_company_id) AS n_revere_companies
        FROM matched_eu_sec_links GROUP BY sec_entity_id
    )
    GROUP BY n_revere_companies ORDER BY n_revere_companies
""")
CSV.write(joinpath(OUT_DIR, "05_multi_match_per_sec_entity.csv"), multi_match)

# ============================================================
# (1a) PRE-AGGREGATE exposure to (sec_entity_id × quarter_end) using
# SELF-CONSISTENT ratio aggregation: SUM(n_cn_total)/SUM(n_total_links).
# Previous version used MAX which (1) is the upper bound, not "conservative",
# and (2) takes numerator and denominator from potentially different sub-entities.
# DISTINCT applied to crosswalk subquery (mirrors exposure_by_sec_entity).
# ============================================================
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE exposure_by_sec_entity AS
    WITH crosswalk AS (
        SELECT DISTINCT sec_entity_id, eu_company_id FROM matched_eu_sec_links
    )
    SELECT m.sec_entity_id,
           e.quarter_end,
           SUM(e.n_cn_total)     AS n_cn_total,
           SUM(e.n_cn_customer)  AS n_cn_customer,
           SUM(e.n_cn_supplier)  AS n_cn_supplier,
           SUM(e.n_cn_jv)        AS n_cn_jv,
           SUM(e.n_total_links)  AS n_total_links,
           SUM(e.n_cn_total)::DOUBLE / NULLIF(SUM(e.n_total_links), 0) AS china_share
    FROM crosswalk m
    JOIN read_parquet('$EXP_PATH') e ON m.eu_company_id = e.eu_company_id
    GROUP BY m.sec_entity_id, e.quarter_end
""")
n_exp_se = qdf(con, "SELECT COUNT(*) AS n FROM exposure_by_sec_entity").n[1]
println("  Pre-aggregated exposure rows (sec_entity_id × quarter, SUM-based): $n_exp_se")

# Time-versioned EU membership at sec_entity level. A sec_entity is included
# in the merged panel for quarter q ONLY if at least one matched Revere
# company was an EU firm AS-OF q. This propagates the C5 fix from 02.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE matched_eu_sec_qend AS
    SELECT DISTINCT m.sec_entity_id, u.qend AS quarter_end
    FROM matched_eu_sec_links m
    JOIN read_parquet('$UNIV_QEND_PATH') u
      ON m.eu_company_id = u.eu_company_id
""")
n_se_qend = qdf(con, "SELECT COUNT(*) AS n FROM matched_eu_sec_qend").n[1]
println("  Time-versioned (sec_entity × quarter) membership rows: $n_se_qend")

# Coverage cascade
coverage = qdf(con, """
    WITH all_eu_sec AS (
        SELECT DISTINCT sec_entity_id FROM read_parquet('$EOM_PATH')
        WHERE sec_country IN $EUROPE_str
    )
    SELECT
        (SELECT COUNT(*) FROM all_eu_sec) AS n_eu_sec_total,
        (SELECT COUNT(DISTINCT CASE WHEN cusip IS NOT NULL THEN sec_entity_id END) FROM eu_sec_ids) AS n_eu_sec_with_cusip,
        (SELECT COUNT(DISTINCT CASE WHEN isin IS NOT NULL THEN sec_entity_id END) FROM eu_sec_ids) AS n_eu_sec_with_isin,
        (SELECT COUNT(DISTINCT CASE WHEN sedol IS NOT NULL THEN sec_entity_id END) FROM eu_sec_ids) AS n_eu_sec_with_sedol,
        (SELECT COUNT(DISTINCT sec_entity_id) FROM matched_eu_sec) AS n_eu_sec_matched
""")
println("\nCoverage cascade:")
println(coverage)
CSV.write(joinpath(OUT_DIR, "05_coverage_cascade.csv"), coverage)

# ============================================================
# (2) BUILD MERGED PANEL
# Time-versioned EU membership filter (matched_eu_sec_qend) closes C5 from 02.
# Exposure joined via the SUM-aggregated table.
# C2 fix: pre-2003 quarters carry NULL china_share / n_cn_* (NOT zero).
# C3 fix: LAG(china_share) over (sec_entity_id, report_date) → china_share_lag1q.
# C1 fix: regression weight is portfolio_weight_eu (from 04), not _global.
# ============================================================
println("\nBuilding merged panel (US-investor × matched-EU-AS-OF-q × quarter)...")

merged_path = test_suffix_path(joinpath(OUT_DIR, "merged_us_eu_matched.parquet"))

atomic_copy_to(con, """
    WITH ownership_matched AS (
        SELECT o.sec_entity_id, o.sec_country, o.investor_country, o.report_date,
               o.I_ict, o.market_cap, o.ownership_share,
               o.portfolio_weight_eu      AS portfolio_weight_eu,
               o.portfolio_weight_global  AS portfolio_weight_global
        FROM read_parquet('$OWN_PATH') o
        JOIN matched_eu_sec_qend mq
          ON mq.sec_entity_id = o.sec_entity_id
         AND mq.quarter_end   = o.report_date
        WHERE o.sec_country IN $EUROPE_str
    ),
    base AS (
        SELECT
            om.sec_entity_id,
            om.sec_country,
            om.investor_country,
            om.report_date,
            om.I_ict,
            om.market_cap,
            om.ownership_share,
            om.portfolio_weight_eu,
            om.portfolio_weight_global,
            -- C2 fix: pre-2003-Q1 firm-quarters carry NULL exposure (NOT zero).
            CASE WHEN om.report_date < DATE '2003-03-31' THEN NULL
                 ELSE e.n_cn_total END         AS n_cn_total_raw,
            CASE WHEN om.report_date < DATE '2003-03-31' THEN NULL
                 ELSE e.n_cn_customer END      AS n_cn_customer_raw,
            CASE WHEN om.report_date < DATE '2003-03-31' THEN NULL
                 ELSE e.n_cn_supplier END      AS n_cn_supplier_raw,
            CASE WHEN om.report_date < DATE '2003-03-31' THEN NULL
                 ELSE e.n_cn_jv END            AS n_cn_jv_raw,
            CASE WHEN om.report_date < DATE '2003-03-31' THEN NULL
                 ELSE e.n_total_links END      AS n_total_links_raw,
            CASE WHEN om.report_date < DATE '2003-03-31' THEN NULL
                 ELSE e.china_share END        AS china_share_raw,
            (om.report_date >= DATE '2003-03-31'
             AND e.quarter_end IS NOT NULL)    AS in_revere_coverage,
            g.gpr_us_cn,
            g.gpr_global,
            g.shock_us_cn
        FROM ownership_matched om
        LEFT JOIN exposure_by_sec_entity e
               ON om.sec_entity_id = e.sec_entity_id
              AND om.report_date   = e.quarter_end
        LEFT JOIN gpr_ts g ON om.report_date = g.quarter_end
    )
    SELECT
        sec_entity_id,
        sec_country,
        investor_country,
        report_date,
        I_ict,
        market_cap,
        ownership_share,
        portfolio_weight_eu,
        portfolio_weight_global,
        n_cn_total_raw     AS n_cn_total,
        n_cn_customer_raw  AS n_cn_customer,
        n_cn_supplier_raw  AS n_cn_supplier,
        n_cn_jv_raw        AS n_cn_jv,
        n_total_links_raw  AS n_total_links,
        china_share_raw    AS china_share,
        -- C3 FIX: regression-spec ChinaExposure_{i,t-1} as a lagged column.
        LAG(china_share_raw)   OVER (PARTITION BY sec_entity_id, investor_country
                                      ORDER BY report_date) AS china_share_lag1q,
        LAG(n_cn_total_raw)    OVER (PARTITION BY sec_entity_id, investor_country
                                      ORDER BY report_date) AS n_cn_total_lag1q,
        (n_cn_total_raw > 0)   AS has_cn_exposure,
        in_revere_coverage,
        gpr_us_cn,
        gpr_global,
        shock_us_cn
    FROM base
""", merged_path)
merged_path_fwd = replace(merged_path, "\\" => "/")

n_merged = qdf(con, "SELECT COUNT(*) AS n FROM read_parquet('$merged_path_fwd')").n[1]
println("  merged panel rows (matched, EU-AS-OF-q): $n_merged")
write_manifest("05_merged_us_eu_matched", merged_path; row_count=n_merged, input_paths=STEP_INPUTS)

# Composition diagnostics
comp = qdf(con, """
    SELECT investor_country = 'US' AS is_us,
           in_revere_coverage,
           has_cn_exposure,
           COUNT(*) AS n
    FROM read_parquet('$merged_path_fwd')
    GROUP BY 1,2,3 ORDER BY 1 DESC, 2 DESC, 3 DESC
""")
println("\nMerged panel composition (US / in-coverage / has-cn):")
println(comp)
CSV.write(joinpath(OUT_DIR, "05_merged_panel_composition.csv"), comp)

# ============================================================
# (2b) NONUS AGGREGATE — EU-restricted AND matched-only (C1 fix v2).
# Old version used a global denominator (sum of all non-US holdings worldwide).
# v1 of the fix restricted to EU sec_country but kept the denominator at
# all-EU; verifier flagged that the bucket numerator was matched-only while
# the denominator was all-EU, breaking apples-to-apples comparability with the
# matched-only US numerator. Fix v2: BOTH numerator and denominator restricted
# to the matched_eu_sec_qend universe — the SAME scope as the US side uses
# via merged_path_fwd. Result: bucket within_europe_share now sums to 1 on
# both sides per quarter, and the cross-holder comparison is clean.
# ============================================================
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE nonus_aggregate_eu AS
    WITH matched AS (
        SELECT DISTINCT sec_entity_id, quarter_end FROM matched_eu_sec_qend
    ),
    per_firm AS (
        SELECT i.sec_entity_id, i.report_date,
               SUM(i.I_ict) AS nonus_I
        FROM read_parquet('$ICT_PATH') i
        JOIN matched m
          ON m.sec_entity_id = i.sec_entity_id
         AND m.quarter_end   = i.report_date
        WHERE i.investor_country != 'US' AND i.investor_country IS NOT NULL
          AND i.sec_country IN $EUROPE_str
        GROUP BY i.sec_entity_id, i.report_date
    ),
    per_quarter AS (
        SELECT i.report_date,
               SUM(i.I_ict) AS nonus_total_eu
        FROM read_parquet('$ICT_PATH') i
        JOIN matched m
          ON m.sec_entity_id = i.sec_entity_id
         AND m.quarter_end   = i.report_date
        WHERE i.investor_country != 'US' AND i.investor_country IS NOT NULL
          AND i.sec_country IN $EUROPE_str
        GROUP BY i.report_date
    )
    SELECT p.sec_entity_id, p.report_date,
           p.nonus_I,
           q.nonus_total_eu,
           p.nonus_I / NULLIF(q.nonus_total_eu, 0) AS nonus_portfolio_weight_eu
    FROM per_firm p
    JOIN per_quarter q USING (report_date)
""")
nonus_diag = qdf(con, """
    SELECT
        COUNT(*) AS n_quarters,
        MIN(sum_w_eu) AS min_sum,
        MAX(sum_w_eu) AS max_sum,
        MAX(ABS(sum_w_eu - 1.0)) AS max_abs_dev_from_1
    FROM (
        SELECT report_date, SUM(nonus_portfolio_weight_eu) AS sum_w_eu
        FROM nonus_aggregate_eu
        GROUP BY report_date
    )
""")
println("\nNONUS-EU aggregate sanity check (sum_w_eu should be ≈ 1 per quarter, ALL quarters):")
println(nonus_diag)
if nonus_diag.max_abs_dev_from_1[1] > 1e-6
    @warn "nonus_aggregate_eu does not sum to 1 within tolerance" max_abs_dev=nonus_diag.max_abs_dev_from_1[1]
end

# ============================================================
# (A) SCATTER: US ownership vs ChinaExposure (snapshot)
# Uses CHINA_SHARE_LAG1Q for the descriptive analogue of the regression
# specification (ChinaExposure_{i,t-1}).
# ============================================================
println("\n(A) Scatter: US ownership vs CN exposure (snapshot, LAGGED china_share)")

snap_date_q = qdf(con, """
    SELECT MAX(report_date) AS d
    FROM read_parquet('$merged_path_fwd')
    WHERE report_date <= DATE '2018-12-31'
""")
snap_date = snap_date_q.d[1]
if snap_date === missing || snap_date === nothing
    snap_date_q = qdf(con, "SELECT MAX(report_date) AS d FROM read_parquet('$merged_path_fwd')")
    snap_date = snap_date_q.d[1]
end
println("  using snapshot date: $snap_date")

scatter_df = qdf(con, """
    SELECT us_ownership_share, n_cn_total, china_share, china_share_lag1q, has_cn_exposure
    FROM (
        SELECT investor_country, sec_entity_id,
               n_cn_total, china_share, china_share_lag1q, has_cn_exposure,
               ownership_share AS us_ownership_share, market_cap
        FROM read_parquet('$merged_path_fwd')
        WHERE investor_country = 'US'
          AND report_date = DATE '$snap_date'
          AND market_cap > 0
          AND ownership_share IS NOT NULL
          AND ownership_share BETWEEN 0 AND 1
    )
""")
println("  cells: $(nrow(scatter_df))")
if nrow(scatter_df) > 10
    if hasproperty(scatter_df, :china_share_lag1q)
        keep = .!ismissing.(scatter_df.china_share_lag1q)
        if sum(keep) > 10
            println("  Correlation(US ownership, china_share_lag1q): ",
                    round(cor(scatter_df.us_ownership_share[keep],
                              identity.(scatter_df.china_share_lag1q[keep])), digits=4))
        end
    end
end
CSV.write(joinpath(OUT_DIR, "05_scatter_own_vs_cn_data.csv"), scatter_df)

# ============================================================
# (B) TIME SERIES — US vs NONUS allocation by exposure group (lagged).
# Bucket built from china_share_lag1q (NOT contemporaneous) — descriptive
# analogue of ChinaExposure_{i,t-1} in the regression.
# Pre-2003 quarters drop out automatically because china_share_lag1q is NULL.
# Bucket is a firm-quarter attribute (not investor-conditional).
# ============================================================
println("\n(B) Within-Europe US allocation by exposure group (china_share_lag1q)")

ts_alloc = qdf(con, """
    WITH median_cutoff AS (
        -- Median of china_share_lag1q across firm-quarter cells WITH positive
        -- exposure. Data-driven cutoff replaces the earlier arbitrary 0.20
        -- threshold. Cells with zero or NULL exposure are excluded from the
        -- median computation but classified separately below.
        SELECT QUANTILE_CONT(china_share_lag1q, 0.5) AS med
        FROM read_parquet('$merged_path_fwd')
        WHERE china_share_lag1q > 0
    ),
    bucket AS (
        SELECT DISTINCT sec_entity_id, report_date, china_share_lag1q,
               CASE
                   WHEN china_share_lag1q IS NULL                                           THEN 'MISSING'
                   WHEN china_share_lag1q > (SELECT med FROM median_cutoff)                 THEN 'HIGH'
                   ELSE                                                                          'LOW'
               END AS exp_grp
        FROM read_parquet('$merged_path_fwd')
    ),
    us_side AS (
        SELECT m.report_date, b.exp_grp, SUM(m.portfolio_weight_eu) AS w_in_grp
        FROM read_parquet('$merged_path_fwd') m
        JOIN bucket b ON m.sec_entity_id = b.sec_entity_id AND m.report_date = b.report_date
        WHERE m.investor_country = 'US' AND m.portfolio_weight_eu IS NOT NULL
        GROUP BY m.report_date, b.exp_grp
    ),
    us_total AS (
        SELECT m.report_date, SUM(m.portfolio_weight_eu) AS w_total
        FROM read_parquet('$merged_path_fwd') m
        WHERE m.investor_country = 'US' AND m.portfolio_weight_eu IS NOT NULL
        GROUP BY m.report_date
    ),
    nonus_side AS (
        SELECT n.report_date, b.exp_grp, SUM(n.nonus_portfolio_weight_eu) AS w_in_grp
        FROM nonus_aggregate_eu n
        JOIN bucket b ON n.sec_entity_id = b.sec_entity_id AND n.report_date = b.report_date
        GROUP BY n.report_date, b.exp_grp
    ),
    nonus_total AS (
        SELECT report_date, SUM(nonus_portfolio_weight_eu) AS w_total
        FROM nonus_aggregate_eu GROUP BY report_date
    )
    SELECT u.report_date, 'US' AS investor_country, u.exp_grp,
           u.w_in_grp                              AS abs_portfolio_weight,
           u.w_in_grp / NULLIF(ut.w_total, 0)      AS within_europe_share,
           g.gpr_us_cn, g.shock_us_cn
    FROM us_side u
    LEFT JOIN us_total ut USING (report_date)
    LEFT JOIN gpr_ts g ON u.report_date = g.quarter_end
    UNION ALL
    SELECT n.report_date, 'NONUS' AS investor_country, n.exp_grp,
           n.w_in_grp                              AS abs_portfolio_weight,
           n.w_in_grp / NULLIF(nt.w_total, 0)      AS within_europe_share,
           g.gpr_us_cn, g.shock_us_cn
    FROM nonus_side n
    LEFT JOIN nonus_total nt USING (report_date)
    LEFT JOIN gpr_ts g ON n.report_date = g.quarter_end
    ORDER BY report_date, investor_country, exp_grp
""")
CSV.write(joinpath(OUT_DIR, "05_within_europe_share_by_group.csv"), ts_alloc)

# US-vs-non-US HIGH-only
hi_nonus_agg = let
    sub = filter(r -> r.investor_country == "NONUS" && r.exp_grp == "HIGH", ts_alloc)
    DataFrame(
        report_date           = sub.report_date,
        nonus_within_eu_share = sub.within_europe_share,
        nonus_abs_pw          = sub.abs_portfolio_weight,
        gpr_us_cn             = sub.gpr_us_cn,
        shock_us_cn           = sub.shock_us_cn,
    )
end
CSV.write(joinpath(OUT_DIR, "05_us_vs_nonus_high_share_data.csv"), hi_nonus_agg)

# ============================================================
# (C) DIFFERENTIAL — Δw vs GPR AR(1) shock, HIGH-LAGGED group.
# Uses portfolio_weight_eu (US) and nonus_portfolio_weight_eu (NONUS).
# Winsorize Δw at 1st/99th percentile rather than hard-trim at ±1%, so
# extreme moves (the behavior of interest) are not silently dropped.
# Bucket built from LAGGED china_share.
# ============================================================
println("\n(C) Differential (descriptive) — Δw vs GPR shock, HIGH-lag")

diff_panel_path = test_suffix_path(joinpath(OUT_DIR, "us_vs_nonus_diff.parquet"))

atomic_copy_to(con, """
    WITH us_w_t AS (
        SELECT sec_entity_id, report_date, china_share_lag1q,
               SUM(portfolio_weight_eu) AS us_w
        FROM read_parquet('$merged_path_fwd')
        WHERE investor_country = 'US' AND portfolio_weight_eu IS NOT NULL
        GROUP BY sec_entity_id, report_date, china_share_lag1q
    ),
    nonus_w_t AS (
        SELECT sec_entity_id, report_date,
               nonus_portfolio_weight_eu AS nonus_w
        FROM nonus_aggregate_eu
    ),
    firm_q AS (
        SELECT u.sec_entity_id, u.report_date, u.china_share_lag1q, u.us_w,
               COALESCE(n.nonus_w, 0) AS nonus_w
        FROM us_w_t u
        LEFT JOIN nonus_w_t n
               ON u.sec_entity_id = n.sec_entity_id
              AND u.report_date   = n.report_date
    ),
    windowed AS (
        SELECT sec_entity_id, report_date, china_share_lag1q,
               LAG(us_w,    1) OVER (PARTITION BY sec_entity_id ORDER BY report_date) AS us_w_prev,
               LEAD(us_w,   1) OVER (PARTITION BY sec_entity_id ORDER BY report_date) AS us_w_next,
               LAG(nonus_w, 1) OVER (PARTITION BY sec_entity_id ORDER BY report_date) AS nonus_w_prev,
               LEAD(nonus_w,1) OVER (PARTITION BY sec_entity_id ORDER BY report_date) AS nonus_w_next,
               DATEDIFF('month',
                        LAG(report_date,  1) OVER (PARTITION BY sec_entity_id ORDER BY report_date),
                        LEAD(report_date, 1) OVER (PARTITION BY sec_entity_id ORDER BY report_date)) AS gap_months
        FROM firm_q
    )
    SELECT sec_entity_id, report_date, china_share_lag1q, gap_months,
           (us_w_next    - us_w_prev)    AS d_us_w,
           (nonus_w_next - nonus_w_prev) AS d_nonus_w
    FROM windowed
    WHERE us_w_prev IS NOT NULL AND us_w_next IS NOT NULL
      AND nonus_w_prev IS NOT NULL AND nonus_w_next IS NOT NULL
      AND gap_months = 6
""", diff_panel_path)
diff_panel_path_fwd = replace(diff_panel_path, "\\" => "/")

# Gap-months diagnostic (per audit medium fix)
gap_diag = qdf(con, """
    WITH all_pairs AS (
        SELECT sec_entity_id, report_date,
               LAG(report_date,  1) OVER (PARTITION BY sec_entity_id ORDER BY report_date) AS prev,
               LEAD(report_date, 1) OVER (PARTITION BY sec_entity_id ORDER BY report_date) AS next,
               DATEDIFF('month',
                        LAG(report_date, 1) OVER (PARTITION BY sec_entity_id ORDER BY report_date),
                        LEAD(report_date,1) OVER (PARTITION BY sec_entity_id ORDER BY report_date)) AS gap
        FROM read_parquet('$merged_path_fwd')
        WHERE investor_country = 'US'
    )
    SELECT
        SUM(CASE WHEN prev IS NULL THEN 1 ELSE 0 END) AS n_missing_prev,
        SUM(CASE WHEN next IS NULL THEN 1 ELSE 0 END) AS n_missing_next,
        SUM(CASE WHEN gap IS NOT NULL AND gap <> 6 THEN 1 ELSE 0 END) AS n_gap_not_6,
        SUM(CASE WHEN gap = 6 THEN 1 ELSE 0 END) AS n_gap_6
    FROM all_pairs
""")
println("Δw gap_months diagnostic (firm-quarters dropped by reason):")
println(gap_diag)
CSV.write(joinpath(OUT_DIR, "05_gap_months_diagnostic.csv"), gap_diag)

# HIGH-exposure threshold: median of china_share_lag1q across firm-quarter
# cells with positive exposure. Data-driven; replaces the earlier arbitrary
# 0.20 cutoff. We compute the median from the differential panel itself so
# the cutoff is consistent with the panel used for Δw.
med_pos_q = qdf(con, """
    SELECT QUANTILE_CONT(china_share_lag1q, 0.5) AS med
    FROM read_parquet('$diff_panel_path_fwd')
    WHERE china_share_lag1q > 0
""")
high_cutoff = (nrow(med_pos_q) > 0 && !ismissing(med_pos_q.med[1])) ? med_pos_q.med[1] : 0.0
println("HIGH-exposure cutoff (median of positive china_share_lag1q): $(round(high_cutoff, digits=4))")

# Winsorize at p1/p99 instead of hard ±1% trim.
ws_bounds_q = qdf(con, """
    SELECT
        QUANTILE_CONT(d_us_w,    0.01) AS d_us_p1,
        QUANTILE_CONT(d_us_w,    0.99) AS d_us_p99,
        QUANTILE_CONT(d_nonus_w, 0.01) AS d_nonus_p1,
        QUANTILE_CONT(d_nonus_w, 0.99) AS d_nonus_p99
    FROM read_parquet('$diff_panel_path_fwd')
    WHERE china_share_lag1q > $high_cutoff
""")
d_us_p1    = ws_bounds_q.d_us_p1[1];    d_us_p99    = ws_bounds_q.d_us_p99[1]
d_nonus_p1 = ws_bounds_q.d_nonus_p1[1]; d_nonus_p99 = ws_bounds_q.d_nonus_p99[1]

ts_diff = qdf(con, """
    SELECT report_date,
           AVG(LEAST(GREATEST(d_us_w, $d_us_p1), $d_us_p99))       AS mean_d_us_ws,
           AVG(LEAST(GREATEST(d_nonus_w, $d_nonus_p1), $d_nonus_p99)) AS mean_d_nonus_ws,
           AVG(LEAST(GREATEST(d_us_w, $d_us_p1), $d_us_p99)
              - LEAST(GREATEST(d_nonus_w, $d_nonus_p1), $d_nonus_p99)) AS mean_diff_ws,
           AVG(d_us_w)                 AS mean_d_us_raw,
           AVG(d_nonus_w)              AS mean_d_nonus_raw,
           AVG(d_us_w - d_nonus_w)     AS mean_diff_raw,
           COUNT(*)                    AS n_firms,
           ANY_VALUE(g.gpr_us_cn)      AS gpr_us_cn,
           ANY_VALUE(g.shock_us_cn)    AS shock_us_cn
    FROM read_parquet('$diff_panel_path_fwd') d
    LEFT JOIN gpr_ts g ON d.report_date = g.quarter_end
    WHERE china_share_lag1q > $high_cutoff
      AND g.gpr_us_cn IS NOT NULL
    GROUP BY report_date ORDER BY report_date
""")
CSV.write(joinpath(OUT_DIR, "05_diff_us_vs_nonus_high.csv"), ts_diff)

if nrow(ts_diff) > 10
    # Use winsorized series for headline reporting; raw is also in the CSV.
    mask_ws = .!ismissing.(ts_diff.shock_us_cn)
    if sum(mask_ws) > 10
        c_shock_ws = cor(ts_diff.mean_diff_ws[mask_ws], identity.(ts_diff.shock_us_cn[mask_ws]))
        println("  cor((ΔUS − ΔnonUS)_winsor, Shock^{US-CN}) for HIGH-lag firms: $(round(c_shock_ws, digits=4))")
        println("  obs: $(nrow(ts_diff)) (winsorized at p1/p99)")
    end
end

try
    DBInterface.close!(con)
catch e
    @warn "DBInterface.close! failed" exception=e
end

println("\n========== DONE ==========")
println("Files written (CSV data; figures generated separately by plots/plot_all.jl):")
println("  merged_us_eu_matched.parquet            (DESCRIPTIVE ONLY — see C6 banner at top)")
println("  gpr_ar1_coefficients.csv                (monthly a, b — regression-ready)")
println("  gpr_monthly_with_shock.parquet")
println("  gpr_quarterly_with_shock.parquet")
println("  05_coverage_cascade.csv                 (matched vs unmatched cascade)")
println("  05_unmatched_profile_by_country.csv     (composition of dropped firms)")
println("  05_match_type_distribution.csv          (CUSIP / ISIN / SEDOL contribution)")
println("  05_multi_match_per_sec_entity.csv       (# Revere companies per sec_entity)")
println("  05_merged_panel_composition.csv         (US / coverage / has-CN breakdown)")
println("  05_within_europe_share_by_group.csv     (4-group within-Europe share, lagged bucket)")
println("  05_us_vs_nonus_high_share_data.csv      (non-US side for compare plot)")
println("  05_scatter_own_vs_cn_data.csv           (A: scatter data, lagged exposure)")
println("  05_diff_us_vs_nonus_high.csv            (C: Δw differential, HIGH-lag, winsorized)")
println("  05_gap_months_diagnostic.csv            (Δw drop reasons)")
println()
println("Interpretation notes:")
println("  * All figures are DESCRIPTIVE. Visual correlation is suggestive, not causal.")
println("  * portfolio_weight_eu (regression input) restricts denominator to EU sec_country.")
println("  * NONUS aggregate now also EU-restricted (nonus_aggregate_eu) — fair comparison.")
println("  * Pre-2003 quarters carry NULL china_share (NOT zero).")
println("  * Bucket uses china_share_LAG1Q to match the regression spec.")
println("  * Δw winsorized at p1/p99; raw mean_diff_raw also in CSV for comparison.")
println("  * C6 (selection-on-outcome universe) is NOT fixed here. The merged panel is")
println("    NOT a regression panel. See AUDIT_2026_06_01_julia_descriptive.md.")
```

### `06_cartesian_grid.jl`

```julia
# 06_cartesian_grid.jl
# Audit fix C6 — zero-fill EU firm × holder-group × quarter Cartesian grid.
#
# v2 (post-adversarial-review). Patches applied vs v1:
#   1. Scatter CSV now carries us_ownership_share (I_ict_US/market_cap)
#      so fig10 in plots_python.py doesn't crash on DPN_USE_C6=true.
#   2. Crosswalk picks ONE eu_company_id per sec_entity_id with priority
#      CUSIP > ISIN > SEDOL (was: SUM across multi-match → bias).
#   3. ict_grouped keys off eu_entity_universe instead of duplicating the
#      sec_country EU filter (which could disagree with holdings_eom).
#   4. Merged parquet carries BOTH holder_group AND investor_country (alias)
#      so downstream consumers expecting 05's investor_country column name
#      work without code changes.
#   5. GPR LEFT JOIN uses SELECT DISTINCT to guard against duplicate
#      quarter_end rows blowing up the grid.
#   6. mean_diff_ws is actually winsorised at p1/p99 (not a duplicate of
#      mean_diff).
#   7. country_total_grouped uses COALESCE so NULL totals don't silently
#      drop entire (holder_group × quarter) cells.
#   8. Median cutoff for HIGH/LOW computed ONCE on firm-quarter distinct
#      cells (de-duplicates across holder_group) and reused as a Julia
#      constant — no drift across the three subqueries.
#   9. Asserts: sec_entity_id unique in canonical universe, unique in
#      crosswalk, china_share ∈ [0,1].
#  10. Diagnostic counts: n_delta_w_null_at_boundary vs n_delta_w_null_interior,
#      held-vs-zero-filled composition table.
#
# Why this exists. The 05 panel keeps only (sec_entity_id, holder_country,
# quarter) cells with a positive holding. That is selection-on-outcome for
# a regression whose outcome is institutional holdings: an extensive-margin
# exit (positive → zero) and an extensive-margin entry (zero → positive)
# both disappear because FactSet stores no row for "holding = 0". This
# biases the headline β_3 toward zero.

include("00_setup.jl")

const EOM_PATH        = replace(joinpath(OUT_DIR, test_suffix_path("holdings_eom.parquet")), "\\" => "/")
const ICT_PATH        = replace(joinpath(OUT_DIR, test_suffix_path("I_ict_panel.parquet")), "\\" => "/")
const COUNTRY_TOT_PATH= replace(joinpath(OUT_DIR, test_suffix_path("country_total_ct.parquet")), "\\" => "/")
const EXP_PATH        = replace(joinpath(OUT_DIR, test_suffix_path("firm_quarter_china_exposure.parquet")), "\\" => "/")
const UNIV_PATH       = replace(joinpath(OUT_DIR, test_suffix_path("eu_revere_universe.parquet")), "\\" => "/")
const GPR_Q_PATH      = replace(joinpath(OUT_DIR, test_suffix_path("gpr_quarterly_with_shock.parquet")), "\\" => "/")
const MCAP_PATH       = replace(joinpath(OUT_DIR, test_suffix_path("marketcap_it.parquet")), "\\" => "/")

for p in (EOM_PATH, ICT_PATH, COUNTRY_TOT_PATH, EXP_PATH, UNIV_PATH, GPR_Q_PATH, MCAP_PATH)
    isfile(replace(p, "/" => "\\")) || error("Required input missing: $p")
end

const STEP_INPUTS = [EOM_PATH, ICT_PATH, COUNTRY_TOT_PATH, EXP_PATH, UNIV_PATH, GPR_Q_PATH, MCAP_PATH]

println("\n" * "!"^70)
println("!! 06_cartesian_grid.jl v2 — C6 zero-fill panel build                 !!")
println("!! Adversarial review patches applied (priority crosswalk, scatter    !!")
println("!! us_ownership_share, gpr DISTINCT, winsor, NULL guards, asserts).   !!")
println("!"^70 * "\n")

con = dbcon()

# ============================================================
# (1) EU entity universe — every sec_entity_id ever appearing with an EU
#     sec_country in the holdings panel.
# ============================================================
println("Building EU entity universe from holdings_eom...")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE eu_entity_universe AS
    WITH primary_country AS (
        SELECT sec_entity_id, sec_country,
               COUNT(*) AS n_rows
        FROM read_parquet('$EOM_PATH')
        WHERE sec_country IN $EU_SQL_TUPLE
          AND sec_entity_id IS NOT NULL
        GROUP BY sec_entity_id, sec_country
    ),
    canonical AS (
        SELECT sec_entity_id, sec_country,
               ROW_NUMBER() OVER (PARTITION BY sec_entity_id
                                  ORDER BY n_rows DESC, sec_country ASC) AS rn
        FROM primary_country
    )
    SELECT sec_entity_id, sec_country
    FROM canonical
    WHERE rn = 1
""")
n_eu = qdf(con, "SELECT COUNT(*) AS n FROM eu_entity_universe").n[1]
n_eu_unique = qdf(con, "SELECT COUNT(DISTINCT sec_entity_id) AS n FROM eu_entity_universe").n[1]
@assert n_eu == n_eu_unique "eu_entity_universe is not unique on sec_entity_id ($n_eu rows, $n_eu_unique unique IDs)"
println("  EU entity universe size: $n_eu (unique sec_entity_ids ✓)")

# Diagnostic: how many firms are dual-listed across EU sec_countries?
dual = qdf(con, """
    SELECT COUNT(*) AS n_dual_listed
    FROM (
        SELECT sec_entity_id
        FROM read_parquet('$EOM_PATH')
        WHERE sec_country IN $EU_SQL_TUPLE
        GROUP BY sec_entity_id
        HAVING COUNT(DISTINCT sec_country) > 1
    )
""")
println("  Dual-listed (>1 EU sec_country): $(dual.n_dual_listed[1])")

# ============================================================
# (2) Quarter calendar.
# ============================================================
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE quarters AS
    SELECT LAST_DAY(MAKE_DATE(y, m, 1)) AS quarter_end
    FROM range(1999, 2024) y(y)
    CROSS JOIN (VALUES (3),(6),(9),(12)) AS month_tbl(m)
    WHERE LAST_DAY(MAKE_DATE(y, m, 1)) BETWEEN DATE '1999-03-31' AND DATE '2023-12-31'
    ORDER BY quarter_end
""")
n_q = qdf(con, "SELECT COUNT(*) AS n FROM quarters").n[1]
println("  Quarter calendar: $n_q quarter-ends")

# ============================================================
# (3) Holder-group dimension (binary US / NONUS).
# ============================================================
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE holder_groups AS
    SELECT 'US' AS holder_group UNION ALL SELECT 'NONUS' AS holder_group
""")

# ============================================================
# (4) Cartesian grid: EU firm × holder_group × quarter.
# ============================================================
println("\nBuilding Cartesian grid (firm × holder_group × quarter)...")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE cartesian_grid AS
    SELECT u.sec_entity_id, u.sec_country, h.holder_group, q.quarter_end
    FROM eu_entity_universe u
    CROSS JOIN holder_groups h
    CROSS JOIN quarters q
""")
n_grid = qdf(con, "SELECT COUNT(*) AS n FROM cartesian_grid").n[1]
println("  Cartesian grid size: $n_grid cells")

# ============================================================
# (5) ict_grouped: I_ict aggregated to (sec, holder_group, quarter).
#     Patch 3: filter on sec_entity_id ∈ universe rather than re-applying
#     EU sec_country filter (which could disagree across sources).
# ============================================================
println("\nAggregating I_ict to holder-group level (universe-filtered)...")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE ict_grouped AS
    SELECT i.sec_entity_id,
           CASE WHEN i.investor_country = 'US' THEN 'US' ELSE 'NONUS' END AS holder_group,
           i.report_date AS quarter_end,
           SUM(i.I_ict) AS I_ict
    FROM read_parquet('$ICT_PATH') i
    WHERE i.sec_entity_id IN (SELECT sec_entity_id FROM eu_entity_universe)
    GROUP BY i.sec_entity_id, holder_group, i.report_date
""")

# (5b) country_total_grouped — Patch 7: use COALESCE so NULL totals don't
# nuke an entire (group × quarter).
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE country_total_grouped AS
    SELECT CASE WHEN investor_country = 'US' THEN 'US' ELSE 'NONUS' END AS holder_group,
           report_date AS quarter_end,
           SUM(COALESCE(country_total_holdings_eu, 0)) AS total_holdings_eu
    FROM read_parquet('$COUNTRY_TOT_PATH')
    GROUP BY holder_group, report_date
""")
ct_check = qdf(con, """
    SELECT
        COUNT(*) AS n_rows,
        COUNT(*) FILTER (WHERE total_holdings_eu IS NULL OR total_holdings_eu = 0) AS n_zero_or_null
    FROM country_total_grouped
""")
println("  country_total_grouped: $(ct_check.n_rows[1]) rows; $(ct_check.n_zero_or_null[1]) zero/NULL")

# ============================================================
# (6) ChinaExposure crosswalk — Patch 2: priority CUSIP > ISIN > SEDOL,
#     ONE eu_company_id per sec_entity_id.
# ============================================================
println("\nBuilding ChinaExposure crosswalk (CUSIP > ISIN > SEDOL priority)...")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE eu_sec_ids AS
    SELECT DISTINCT
           sec_entity_id,
           NULLIF(TRIM(cusip), '') AS cusip,
           NULLIF(TRIM(isin),  '') AS isin,
           NULLIF(TRIM(sedol), '') AS sedol
    FROM read_parquet('$EOM_PATH')
    WHERE sec_country IN $EU_SQL_TUPLE
      AND sec_entity_id IS NOT NULL
""")

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE matched_eu_sec_links AS
    WITH univ AS (
        SELECT eu_company_id,
               NULLIF(TRIM(eu_cusip), '') AS eu_cusip,
               NULLIF(TRIM(eu_isin),  '') AS eu_isin,
               NULLIF(TRIM(eu_sedol), '') AS eu_sedol
        FROM read_parquet('$UNIV_PATH')
    ),
    all_matches AS (
        SELECT 1 AS prio, s.sec_entity_id, u.eu_company_id
        FROM eu_sec_ids s JOIN univ u ON s.cusip = u.eu_cusip
        WHERE s.cusip IS NOT NULL AND u.eu_cusip IS NOT NULL
        UNION ALL
        SELECT 2 AS prio, s.sec_entity_id, u.eu_company_id
        FROM eu_sec_ids s JOIN univ u ON s.isin = u.eu_isin
        WHERE s.isin IS NOT NULL AND u.eu_isin IS NOT NULL
        UNION ALL
        SELECT 3 AS prio, s.sec_entity_id, u.eu_company_id
        FROM eu_sec_ids s JOIN univ u ON s.sedol = u.eu_sedol
        WHERE s.sedol IS NOT NULL AND u.eu_sedol IS NOT NULL
    )
    SELECT sec_entity_id, eu_company_id
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY sec_entity_id
                                  ORDER BY prio ASC, eu_company_id ASC) AS rn
        FROM all_matches
    )
    WHERE rn = 1
""")
cw_n = qdf(con, "SELECT COUNT(*) AS n, COUNT(DISTINCT sec_entity_id) AS u FROM matched_eu_sec_links").n[1]
cw_u = qdf(con, "SELECT COUNT(DISTINCT sec_entity_id) AS u FROM matched_eu_sec_links").u[1]
@assert cw_n == cw_u "matched_eu_sec_links is not unique on sec_entity_id ($cw_n rows, $cw_u unique)"
println("  Crosswalk: $cw_n unique (sec_entity_id → eu_company_id) pairs ✓")

DBInterface.execute(con, """
    CREATE OR REPLACE TABLE exposure_by_sec_entity AS
    SELECT m.sec_entity_id,
           e.quarter_end,
           e.n_cn_total,
           e.n_total_links,
           e.n_cn_total::DOUBLE / NULLIF(e.n_total_links, 0) AS china_share
    FROM matched_eu_sec_links m
    JOIN read_parquet('$EXP_PATH') e ON m.eu_company_id = e.eu_company_id
""")
# Patch 4 assertion: china_share ∈ [0,1]
oob = qdf(con, """
    SELECT COUNT(*) AS n FROM exposure_by_sec_entity
    WHERE china_share IS NOT NULL AND (china_share < 0 OR china_share > 1)
""")
@assert oob.n[1] == 0 "exposure_by_sec_entity has $(oob.n[1]) rows with china_share ∉ [0,1]"
println("  exposure_by_sec_entity: china_share ∈ [0,1] ✓")

# ============================================================
# (7) Stitch everything onto the grid + zero-fill.
#     Patch 5: gpr LEFT JOIN with SELECT DISTINCT to guard against
#     duplicate quarter_end rows.
# ============================================================
println("\nStitching the zero-filled merged panel...")
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE grid_zfilled AS
    SELECT g.sec_entity_id, g.sec_country, g.holder_group, g.quarter_end,
           COALESCE(i.I_ict, 0) AS I_ict,
           COALESCE(ct.total_holdings_eu, 0) AS total_holdings_eu,
           CASE WHEN COALESCE(ct.total_holdings_eu, 0) > 0
                THEN COALESCE(i.I_ict, 0) / ct.total_holdings_eu
                ELSE NULL END AS portfolio_weight_eu,
           CASE WHEN g.quarter_end < DATE '2003-03-31' THEN NULL
                ELSE e.china_share END AS china_share,
           gpr.gpr_us_cn,
           gpr.shock_us_cn
    FROM cartesian_grid g
    LEFT JOIN ict_grouped i USING (sec_entity_id, holder_group, quarter_end)
    LEFT JOIN country_total_grouped ct USING (holder_group, quarter_end)
    LEFT JOIN exposure_by_sec_entity e USING (sec_entity_id, quarter_end)
    LEFT JOIN (
        SELECT DISTINCT quarter_end, gpr_us_cn, shock_us_cn
        FROM read_parquet('$GPR_Q_PATH')
    ) gpr USING (quarter_end)
""")

# ============================================================
# (8) Lag exposure + backward Δw + emit merged parquet.
#     Patch 4: holder_group AS investor_country alias for back-compat.
# ============================================================
println("\nComputing lagged exposure + backward Δw on the zero-filled panel...")
merged_path = test_suffix_path(joinpath(OUT_DIR, "merged_us_eu_zero_filled.parquet"))
atomic_copy_to(con, """
    SELECT
        sec_entity_id, sec_country,
        holder_group,
        holder_group AS investor_country,  -- back-compat alias
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
""", merged_path)
merged_path_fwd = replace(merged_path, "\\" => "/")
n_merged = qdf(con, "SELECT COUNT(*) AS n FROM read_parquet('$merged_path_fwd')").n[1]
println("  Zero-filled panel rows: $n_merged")
write_manifest("06_merged_us_eu_zero_filled", merged_path; row_count=n_merged, input_paths=STEP_INPUTS)

# ============================================================
# (9) Patch 8: compute the HIGH-vs-LOW cutoff ONCE on firm-quarter distinct
#     rows (deduped across holder_group) and reuse as a Julia constant.
# ============================================================
med_q = qdf(con, """
    SELECT QUANTILE_CONT(china_share_lag1q, 0.5) AS med
    FROM (
        SELECT DISTINCT sec_entity_id, report_date, china_share_lag1q
        FROM read_parquet('$merged_path_fwd')
        WHERE china_share_lag1q IS NOT NULL AND china_share_lag1q > 0
    )
""")
const HIGH_CUTOFF = (nrow(med_q) > 0 && !ismissing(med_q.med[1]) && !isnan(med_q.med[1])) ? med_q.med[1] : 0.0
println("\nHIGH-exposure cutoff (firm-quarter median on positive cells): $(round(HIGH_CUTOFF, digits=4))")
@assert HIGH_CUTOFF > 0 "HIGH_CUTOFF degenerated to $HIGH_CUTOFF — no positive china_share_lag1q observations in the merged panel; check exposure_by_sec_entity"

# Patch 10: delta_w NULL diagnostic (backward Δw: NULL iff w_prev is NULL,
# i.e. the first quarter of each (firm × holder_group) series).
ndiag = qdf(con, """
    SELECT
        COUNT(*) FILTER (WHERE delta_w IS NULL AND w_prev IS NULL)     AS n_null_first_quarter,
        COUNT(*) FILTER (WHERE delta_w IS NULL AND w_prev IS NOT NULL) AS n_null_interior,
        COUNT(*) FILTER (WHERE delta_w IS NOT NULL)                    AS n_delta_w_nonnull
    FROM read_parquet('$merged_path_fwd')
""")
println("Δw NULL diagnostic (backward diff):")
println("  first quarter of series (w_prev missing): $(ndiag.n_null_first_quarter[1])")
println("  interior (w_prev present but Δw NULL):    $(ndiag.n_null_interior[1])  ← should be 0")
println("  non-NULL Δw: $(ndiag.n_delta_w_nonnull[1])")

# Held vs zero-filled composition (Patch 10)
comp = qdf(con, """
    SELECT holder_group,
           CASE WHEN I_ict > 0 THEN 'held' ELSE 'zero_filled' END AS hold_status,
           CASE WHEN china_share_lag1q IS NULL    THEN 'MISSING'
                WHEN china_share_lag1q > $HIGH_CUTOFF THEN 'HIGH'
                ELSE                                       'LOW' END AS exp_grp,
           COUNT(*) AS n
    FROM read_parquet('$merged_path_fwd')
    GROUP BY 1,2,3 ORDER BY 1,2,3
""")
CSV.write(joinpath(OUT_DIR, "06_panel_composition_c6.csv"), comp)
println("\nC6 panel composition (held vs zero-filled, by exposure bucket):")
println(comp)

# ============================================================
# (10) Descriptive analogues — _c6 suffixed CSVs.
# ============================================================
println("\nBuilding descriptive analogues on C6 panel...")

# (10a) Cross-section scatter — Patch 1: us_ownership_share = I_ict_US / market_cap.
snap_q = qdf(con, """
    SELECT MAX(report_date) AS d FROM read_parquet('$merged_path_fwd')
    WHERE report_date <= DATE '2018-12-31'
""")
snap = (nrow(snap_q) > 0 && !ismissing(snap_q.d[1])) ? snap_q.d[1] :
       qdf(con, "SELECT MAX(report_date) AS d FROM read_parquet('$merged_path_fwd')").d[1]
println("  Scatter snapshot date: $snap")

sc = qdf(con, """
    SELECT z.sec_entity_id,
           z.china_share,
           z.china_share_lag1q,
           z.I_ict,
           z.portfolio_weight_eu,
           m.market_cap,
           CASE WHEN m.market_cap > 0
                THEN z.I_ict / m.market_cap
                ELSE NULL END AS us_ownership_share
    FROM read_parquet('$merged_path_fwd') z
    LEFT JOIN read_parquet('$MCAP_PATH') m
      ON z.sec_entity_id = m.sec_entity_id
     AND z.report_date   = m.report_date
    WHERE z.holder_group = 'US'
      AND z.report_date = DATE '$snap'
      AND z.portfolio_weight_eu IS NOT NULL
""")
CSV.write(joinpath(OUT_DIR, "05_scatter_own_vs_cn_data_c6.csv"), sc)
println("  fig10 scatter data → 05_scatter_own_vs_cn_data_c6.csv ($(nrow(sc)) rows; us_ownership_share included)")

# (10b) Bucket time series — Patch 7: bucket from DISTINCT firm-quarter,
# both holder groups share the same bucket assignment.
ts_alloc = qdf(con, """
    WITH bucket AS (
        SELECT DISTINCT sec_entity_id, report_date,
               CASE WHEN china_share_lag1q IS NULL          THEN 'MISSING'
                    WHEN china_share_lag1q > $HIGH_CUTOFF   THEN 'HIGH'
                    ELSE                                         'LOW'
               END AS exp_grp
        FROM read_parquet('$merged_path_fwd')
    ),
    sided AS (
        SELECT z.report_date, z.holder_group, b.exp_grp,
               SUM(z.portfolio_weight_eu) AS abs_portfolio_weight
        FROM read_parquet('$merged_path_fwd') z
        JOIN bucket b USING (sec_entity_id, report_date)
        WHERE z.portfolio_weight_eu IS NOT NULL
        GROUP BY 1, 2, 3
    ),
    totals AS (
        SELECT report_date, holder_group, SUM(portfolio_weight_eu) AS w_total
        FROM read_parquet('$merged_path_fwd')
        WHERE portfolio_weight_eu IS NOT NULL
        GROUP BY 1, 2
    )
    SELECT s.report_date,
           CASE WHEN s.holder_group = 'US' THEN 'US' ELSE 'NONUS' END AS investor_country,
           s.exp_grp,
           s.abs_portfolio_weight,
           s.abs_portfolio_weight / NULLIF(t.w_total, 0) AS within_europe_share,
           gpr.gpr_us_cn, gpr.shock_us_cn
    FROM sided s
    LEFT JOIN totals t USING (report_date, holder_group)
    LEFT JOIN (SELECT DISTINCT quarter_end, gpr_us_cn, shock_us_cn FROM read_parquet('$GPR_Q_PATH')) gpr
           ON s.report_date = gpr.quarter_end
    ORDER BY s.report_date, s.holder_group, s.exp_grp
""")
CSV.write(joinpath(OUT_DIR, "05_within_europe_share_by_group_c6.csv"), ts_alloc)
println("  fig11 bucket data → 05_within_europe_share_by_group_c6.csv ($(nrow(ts_alloc)) rows)")

# (10c) Differential ΔUS - ΔnonUS for HIGH-exposure firms vs Shock.
#       Patch 6: actually winsorize mean_diff_ws at p1/p99 of the diff series.
println("\nComputing differential (with REAL winsorization)...")
# Step 1: get raw per-firm-quarter (d_us - d_nonus) on HIGH-exposure cells.
DBInterface.execute(con, """
    CREATE OR REPLACE TABLE diff_high_raw AS
    WITH us_w AS (
        SELECT sec_entity_id, report_date, delta_w AS d_us_w, china_share_lag1q
        FROM read_parquet('$merged_path_fwd')
        WHERE holder_group = 'US' AND delta_w IS NOT NULL
    ),
    nonus_w AS (
        SELECT sec_entity_id, report_date, delta_w AS d_nonus_w
        FROM read_parquet('$merged_path_fwd')
        WHERE holder_group = 'NONUS' AND delta_w IS NOT NULL
    )
    SELECT u.sec_entity_id, u.report_date, u.china_share_lag1q,
           u.d_us_w, n.d_nonus_w,
           (u.d_us_w - n.d_nonus_w) AS d_diff
    FROM us_w u JOIN nonus_w n USING (sec_entity_id, report_date)
    WHERE u.china_share_lag1q > $HIGH_CUTOFF
""")
# Step 2: compute p1/p99 bounds for winsorization.
ws_b = qdf(con, """
    SELECT QUANTILE_CONT(d_diff, 0.01) AS p1, QUANTILE_CONT(d_diff, 0.99) AS p99
    FROM diff_high_raw
""")
p1   = nrow(ws_b) > 0 && !ismissing(ws_b.p1[1])  ? ws_b.p1[1]  : -Inf
p99  = nrow(ws_b) > 0 && !ismissing(ws_b.p99[1]) ? ws_b.p99[1] :  Inf
println("  winsorization bounds: p1=$(round(p1, digits=8)) p99=$(round(p99, digits=8))")

ts_diff = qdf(con, """
    SELECT report_date,
           AVG(d_diff)                                          AS mean_diff,
           AVG(LEAST(GREATEST(d_diff, $p1), $p99))              AS mean_diff_ws,
           AVG(d_us_w)                                          AS mean_d_us_raw,
           AVG(d_nonus_w)                                       AS mean_d_nonus_raw,
           AVG(d_us_w - d_nonus_w)                              AS mean_diff_raw,
           COUNT(*)                                             AS n_firms,
           ANY_VALUE(g.gpr_us_cn)                               AS gpr_us_cn,
           ANY_VALUE(g.shock_us_cn)                             AS shock_us_cn
    FROM diff_high_raw d
    LEFT JOIN (SELECT DISTINCT quarter_end, gpr_us_cn, shock_us_cn FROM read_parquet('$GPR_Q_PATH')) g
           ON d.report_date = g.quarter_end
    WHERE g.gpr_us_cn IS NOT NULL
    GROUP BY report_date ORDER BY report_date
""")
CSV.write(joinpath(OUT_DIR, "05_diff_us_vs_nonus_high_c6.csv"), ts_diff)
println("  fig13 differential data → 05_diff_us_vs_nonus_high_c6.csv ($(nrow(ts_diff)) rows; mean_diff_ws is REALLY winsorized)")

if nrow(ts_diff) > 10
    # Joint mask: drop rows where ANY of the three series is Missing/NaN so
    # cor() doesn't MethodError on Union{Missing,Float64}. Then disallowmissing
    # to coerce the eltype.
    mm = .!ismissing.(ts_diff.shock_us_cn) .&
         .!ismissing.(ts_diff.mean_diff)   .&
         .!ismissing.(ts_diff.mean_diff_ws)
    if sum(mm) > 10
        s  = Vector{Float64}(ts_diff.shock_us_cn[mm])
        mr = Vector{Float64}(ts_diff.mean_diff[mm])
        mw = Vector{Float64}(ts_diff.mean_diff_ws[mm])
        c_raw = cor(mr, s)
        c_ws  = cor(mw, s)
        println("\n  cor((ΔUS − ΔnonUS), Shock^{US-CN}) on C6 panel for HIGH-lag firms:")
        println("    raw:        $(round(c_raw, digits=4))")
        println("    winsorized: $(round(c_ws,  digits=4))")
        println("  obs: $(sum(mm)) (of $(nrow(ts_diff)) total ts_diff rows)")
    else
        println("  too few non-missing rows ($(sum(mm))) to compute correlation")
    end
end

# (10d) US vs NONUS HIGH-exposure aggregate data (fig12).
hi = qdf(con, """
    WITH high_cells AS (
        SELECT z.sec_entity_id, z.report_date, z.holder_group, z.portfolio_weight_eu
        FROM read_parquet('$merged_path_fwd') z
        WHERE z.china_share_lag1q > $HIGH_CUTOFF
    ),
    summed AS (
        SELECT report_date, holder_group, SUM(portfolio_weight_eu) AS abs_portfolio_weight
        FROM high_cells
        WHERE portfolio_weight_eu IS NOT NULL
        GROUP BY 1, 2
    ),
    wide AS (
        SELECT report_date,
               SUM(CASE WHEN holder_group = 'NONUS' THEN abs_portfolio_weight ELSE 0 END) AS nonus_abs_pw,
               SUM(CASE WHEN holder_group = 'US'    THEN abs_portfolio_weight ELSE 0 END) AS us_abs_pw
        FROM summed
        GROUP BY report_date
    )
    SELECT w.report_date,
           w.nonus_abs_pw, w.us_abs_pw,
           g.gpr_us_cn, g.shock_us_cn
    FROM wide w
    LEFT JOIN (SELECT DISTINCT quarter_end, gpr_us_cn, shock_us_cn FROM read_parquet('$GPR_Q_PATH')) g
           ON w.report_date = g.quarter_end
    ORDER BY w.report_date
""")
CSV.write(joinpath(OUT_DIR, "05_us_vs_nonus_high_share_data_c6.csv"), hi)
println("  fig12 US-vs-NONUS HIGH data → 05_us_vs_nonus_high_share_data_c6.csv ($(nrow(hi)) rows)")

try
    DBInterface.close!(con)
catch e
    @warn "DBInterface.close! failed" exception=e
end

println("\n========== 06 v2 DONE ==========")
println("Key output:")
println("  merged_us_eu_zero_filled.parquet  (REGRESSION-READY zero-filled panel,")
println("                                      holder_group AND investor_country columns)")
println("  05_scatter_own_vs_cn_data_c6.csv         (fig10 input incl us_ownership_share)")
println("  05_within_europe_share_by_group_c6.csv   (fig11 input)")
println("  05_us_vs_nonus_high_share_data_c6.csv    (fig12 input)")
println("  05_diff_us_vs_nonus_high_c6.csv          (fig13 input, mean_diff_ws REALLY winsorized)")
println("  06_panel_composition_c6.csv              (held vs zero-filled diagnostic)")
println()
println("To re-generate figures with the C6 panel:")
println("  set DPN_USE_C6=true && python plots_python.py")
```

