# Multi-Agent Review: De Haas et al. (2025) vs. Geoecon Project

**Paper:** "Violent Conflict and Cross-Border Lending"
**Authors:** Ralph De Haas (EBRD, KU Leuven, CEPR); Mikhail Mamonov (Toulouse Business School); Alexander Popov (ECB, CEPR); Iliriana Shala (Deutsche Bundesbank, Goethe University)
**Status:** Working paper, Dec 19, 2025, 93 pp. Presented at EFA 2025, EEA 2025, BIS-BoE-ECB-IMF Spillover Conference 2025, CEPR Summer Conference on Financial Intermediation, FRB Summer Workshop. Positioning suggests JF / RFS / JFE.
**Headline finding:** During violent conflicts, foreign banks reduce non-military lending by **27%** but increase military-sector lending by **24%** relative to domestic banks; reallocation driven by military-specialized but country-non-specialized banks; channeled to politically non-aligned countries; dissipates within 3 years; military firms expand revenue/employment while civilian firms contract.

## 1. Honest verdict on the advisor's "exactly the same" framing

**SUPERFICIAL parallel.** At the **rhetorical template** level the analogy is real — triple-difference, progressive FE saturation, Khwaja-Mian mechanism decomposition, leave-one-out robustness, Jordà-LP mediation. At the **methodological** level the analogy fails on **6 of 6 dimensions**:

| Dimension | De Haas et al. | Geoecon project | Verdict |
|---|---|---|---|
| Triple structural form | Foreign × Conflict × Military | US × ChinaExposure × Shock | **parallel** |
| Lender / holder contrast | Foreign-vs-**Domestic** (within-country, domestic benchmark stays put) | US-vs-non-US (both foreign to EU firm, no domestic anchor) | **divergent** |
| Sectoral exposure | Time-**invariant** Military_s (SIC) | Time-**varying** ChinaExposure_{i,t-1} | **divergent** |
| Shock measurement | Binary UCDP conflict, cross-country variation, discrete onset | Continuous AR(1) residual, **single global time series**, no cross-country variation | **divergent** |
| LHS structure | USD loan levels (PPML appropriate) / log-loan (Table 2) | **Signed** Δw (PPML mathematically inadmissible) | **divergent** |
| FE stack | 8 FE blocks (μ_b, ν_f, π_{ht}, ω_{ct}, ν_{vs}, δ_{gs}, φ_{gt}, θ_{gc}) | 2 FE blocks (α_{i,t}, α_{c×t}) — **missing investor FE and holder-country × quarter** | **divergent** |

The advisor's framing is correct at "this is a triple-DD paper too, look how they wrote it up" but **fails as a methodological blueprint**. Treat the paper as the **best available rhetorical template**, not a substitute for independent defense of your spec.

## 2. The single most important takeaway

**The paper's β3>0 (military rents) prediction has NO analog in your setup.** Their β3 requires a sector with a *positive* return shock (defense procurement rents). China-exposed EU firms have no such rent channel during US-CN tension — they only lose. **Drop H2. Reframe as one-sided "flight-from-exposure" — the analog of their β1, not their β3.**

If you keep H2 ("US investors strategically expand to some beneficiary subsample") you'll be reverse-engineering to fit the paper's rhetorical structure. Any referee who reads the paper will flag it.

## 3. Estimator decision: stay LINEAR

**Do NOT switch to PPML for the headline.** Three reasons in order of weight:

1. PPML requires y ≥ 0. **Δw is signed** (centered change in portfolio weight, routinely negative). PPML is mathematically inadmissible — not negotiable.
2. The paper itself **switches to log-OLS with HDFE at the loan level** (Table 2, 1.3M obs) precisely because at that granularity zeros disappear. Your 2.27M investor-firm-quarter panel is the direct analog of their **Table 2**, not Table 1.
3. PPML's semi-elasticity interpretation does not apply to a differenced outcome — differencing already converts the LHS to a change-like object.

**Recommended compromise to preempt advisor PPML pushback:** add an *aggregate* PPML companion table (US-vs-non-US × EU-country × CN-exposure-bucket × quarter, zero-filled, ppmlhdfe on USD-holdings *levels*) framed as "suggestive aggregate evidence" à la their Eq. 7. Gives the advisor the PPML column, captures the extensive margin Δw can't, and frames the linear Δw spec as the *identified* test (their Table 2 logic).

## 4. Two referee-mandatory FE blocks you are missing

Both are non-negotiable for top-journal review:

- **Holder-country × quarter FE** (paper's φ_{gt} bank-group × year analog). Absorbs all US-wide reallocation trends unrelated to China — yield curve, tax, Fed regulation coincident with GPR. Caveat: this absorbs the level US_c × Shock_t double effect, but per the paper's own saturation logic the triple is the headline anyway.
- **Investor FE** (paper's μ_b analog). Absorbs mandate / style / vintage effects of each US institution. Feasible on 2.27M obs.

## 5. Action list — priorities for the next 1-2 weeks

### HIGH priority

1. **Drop H2.** Reframe as one-sided flight-from-exposure (analog of their β1<0), abandon the lincom (β2+β3) test apparatus.
2. **Add holder-country × quarter FE and investor FE.** Show progressive saturation in a 4-6 column table where β stays essentially constant. Be transparent that holder-country × quarter absorbs US_c × Shock_t.
3. **Change comparison-group framing.** Either (a) include EU-domiciled institutional investors as the "domestic" arm to mirror the paper, OR (b) keep two-foreign and reframe as "differential retrenchment among foreign holders, with US-CN as the salient pair." Abandon "flight-home" language under option (b).
4. **Defend linear by citing the paper's Table 2** (loan-level, log-OLS, HDFE, 1.3M obs), NOT Table 1 (PPML). State explicitly that PPML is mathematically inadmissible on signed Δw.
5. **Build a Khwaja-Mian 4-column decomposition** mirroring their Table 3: (a) baseline, (b) +Investor×Quarter (demand isolation, paper's Bank×Year), (c) drop firm×quarter / keep only investor×quarter, (d) +Investor×Firm (Chodorow-Reich 2014 + Schwert 2018 assortative matching). This is your headline mechanism table.
6. **Address time-varying ChinaExposure endogeneity.** Either (a) freeze exposure at first observation (cross-sectional, mirrors their Military_s), OR (b) keep lagged-time-varying but run a placebo using t-12 lag to show pre-determined exposure is exogenous to current shock.

### MEDIUM priority

7. Construct a binary shock variant (top-decile AR(1) residual quarter = 1) and run their threshold-monotonicity ladder (top-quartile / decile / 5%). Continuous stays headline; binary enables their internal-validity check.
8. Engineer an event-study around shock "onset" (residual crosses 90th pctile after a 4Q quiet stretch, anchored to 2018-Q3 trade war / 2020-Q1 COVID / 2022-Q3 Pelosi-Taiwan). Plot β at event-time −8 to +8 quarters. Adopt their honest-pretrend disclosure protocol (footnote 12).
9. Pre-commit signed hypothesis BEFORE Table 1: "β < 0 on US × ChinaExposure × Shock, reflecting rational rebalancing under integrated capital markets (Holmström-Tirole 1997; Morgan-Rime-Strahan 2004)." Quote magnitudes in the abstract as percentages, not t-stats.
10. Leave-one-out investor robustness: drop BlackRock, Vanguard, State Street, Fidelity, JPM one at a time. Top-500 13F filer restriction. Exclude US-affiliate funds domiciled in LU / IE.
11. Build firm-level mediation panel à la Sec 5.3.2: Jordà (2005) local projections of ln(CN revenue), ln(CN COGS), margin onto Shock × CN-exposure, h=1..8 quarters. Validates retrenchment as **rational**, not sentiment.
12. Extensive-margin companion: probit/logit for P(US investor entirely exits EU firm | held in t−1). Reviewers will ask.

### LOW priority

13. IHS-transform Δw robustness; binary high-CN-exposure (top-quintile) robustness; alternative clustering (investor, quarter) and (investor-country, quarter) alongside (firm, quarter).
14. Construct-validity placebo: re-run with US-exposure × US-CN-shock instead of CN-exposure. Should be null or opposite sign.

## 6. Talking points for the advisor meeting (verbatim)

1. "I want to flag three asymmetries before we commit to 'exactly the same'."
2. **Lender contrast:** their Foreign-vs-Domestic is within-country (Citi-US vs local Nigerian bank). Mine is two-foreign — both US and UK institutional investors are foreign to an EU firm. **Question:** should I add EU-domiciled institutional investors as the 'domestic' arm to mirror their setup, or keep two-foreign and reframe?
3. **Treatment time-variation:** their Military_s is time-invariant SIC; mine is lagged time-varying. I want to run two specs side-by-side — lagged-time-varying as headline plus first-observation-frozen as robustness — and a t-12 placebo for pre-determined exposure.
4. **Estimator:** I'm NOT switching to PPML for the headline. Their Table 2 (loan-level, log-OLS, HDFE, 1.3M obs) is the direct analog of my 2.27M panel. PPML is for the aggregate-level USD LHS with zeros — different research question. I'll add an aggregate PPML companion.
5. **Fixed effects:** adding two referee-mandatory blocks — holder-country × quarter (their φ_{gt} analog) and investor FE (their μ_b analog). Yes, this absorbs US_c × Shock_t but the triple is the headline.
6. **Mechanism table:** direct Khwaja-Mian port of their Table 3. If β survives investor × quarter but dies under firm × quarter, it's supply-driven (my preferred story). If β dies under investor × quarter, it's investor-side common rebalancing and I have a problem.
7. **Hypothesis:** dropping H2 (β2+β3>0). Their β3>0 requires a sector with a positive return shock; my setup has none. Reframing as one-sided flight-from-exposure (analog of their β1). Open to pushback if you see a credible 'beneficiary' subsample.
8. **Theory framing:** borrowing Morgan-Rime-Strahan (2004) + Holmström-Tirole (1997) language — **rational rebalancing**, not preference shift.
9. **Event-study:** my continuous shock has no natural onset. I'll engineer one anchored to 2018-Q3 trade war / 2020-Q1 COVID / 2022-Q3 Pelosi-Taiwan.
10. **Headline magnitude pre-commit:** "1-SD US-CN GPR shock causes US holders to reduce weight on top-quintile-CN-exposed EU firms by X bp relative to non-US holders." Percentages, not t-stats.

## 7. Things your project does BETTER than the paper

The mapping is not all loss — your design has real strengths the paper doesn't:

- **α_{i,t} firm × quarter FE is stronger** than their (separate firm + country×year) combo on a richer panel.
- **Two-way clustering on (firm, quarter)** is more conservative than their single-cluster.
- **Continuous AR(1)-residual shock** is more sophisticated than their binary onset (preserves intensity variation).
- **Richer panel** (2.27M obs) allows Investor × Firm FE without halving the sample (which crushes them in Table 3).
- **Bilateral hegemon-pair shock** is a novel measurement (their conflicts are one-sided host-country events).

## 8. Citations to add to your paper

- **De Haas, Mamonov, Popov, Shala (2025)** — primary template (cite at empirical strategy + mechanism table + robustness sections)
- **Khwaja and Mian (2008, AER)** — saturating-FE supply/demand decomposition (cite at mechanism table)
- **Chodorow-Reich (2014, QJE)** — bank-firm relationship FE (cite when adding Investor × Firm FE)
- **Schwert (2018, JF)** — bank-firm matching FE (assortative matching column)
- **Morgan, Rime, and Strahan (2004, QJE)** — internal capital markets (theory subsection, rational rebalancing framing)
- **Holmström and Tirole (1997, QJE)** — pledgeable income / collateral (theory subsection)
- **Jordà (2005, AER)** — local projections (firm-level mediation panel)
- **Iacoviello and Tong (2026)** — bilateral AI-GPR construction (shock-measurement subsection — already cited in your slides)
- **Mian (2006, JF)** and **Giannetti and Laeven (2012, JFE)** — foreign-vs-domestic lender definition (when defining US vs non-US holder)
- **Silva and Tenreyro (2006, REStat)** + **Correia, Guimaraes, Zylkin (2020, Stata Journal)** — only if running the PPML aggregate companion
- **De Haas and Van Horen (2013, AER)** — analog for any imputation robustness

## 9. Largest residual risks for your design (none new — but newly sharp)

Beyond the audit's open C6 (selection-on-outcome universe), the paper-by-paper deep read surfaces:

- **Single-time-series shock with no cross-country variation.** Once holder-country × quarter FE is added, US_c × Shock_t is fully absorbed. The triple becomes your *only* identified coefficient — a real referee critique unless framed correctly.
- **Two time-varying interactions in the triple** (ChinaExposure × Shock both vary in t). The paper's design has only one time-varying interaction. Standard pretrend robustness is mechanically harder.
- **Differenced LHS interacts non-trivially with α_{i,t}.** w_{t+1} − w_{t−1} carries information about t+1 and t−1 fundamentals; α_{i,t} only absorbs at t. There's a genuine spec-coherence question the paper does not face because they don't difference.

These are not necessarily fatal — but each needs an explicit defense paragraph in your empirical strategy section.

---

**Bottom line for the advisor meeting:** the De Haas paper is the **best available rhetorical template** for the project but not a methodological substitute. Adopt their robustness ladder, mechanism table architecture, hypothesis-pre-commit + magnitude-framing style. **Defend your spec independently** on the six dimensions where the analogy breaks (lender contrast, time-varying treatment, single-time-series shock, signed-Δw estimator, missing FE blocks, no rent-channel for H2).
