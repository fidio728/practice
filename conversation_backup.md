# Conversation Backup: Cherry-Picking Model Development

> Date: 2026-03-02
> Context: Developing a game theory model of pharma firm selective disclosure, based on Min (2023)

---

## Background

The user is studying the pharmaceutical "file drawer" problem using game theory. The base framework is Min (2023), which studies FDA screening of firm types through experiment design. The advisor's requirement: "only change one thing" from Min's model.

### Key documents provided:
- **药企文件抽屉问题研究框架.md**: Extended chat history discussing model development with another AI
- **Feb 27.tex**: Previous LaTeX draft (single-type model, FDA sees (π, ∅))
- **pharma_hidden_trials.md**: Empirical evidence on pharma hiding trial results
- **pharma_selective_reporting_2018_2026.md**: Additional empirical evidence
- **Herresthal (2022, JET)**: "Hidden testing and selective disclosure of evidence" — key reference paper

---

## Advisor Feedback on Feb 27

1. **结论太trivial**: Cherry-picking is strategically irrelevant (A(∅) = 0 always, model collapses to Min)
2. **FDA看见∅的假设不convincing**: "都看见了, FDA response不是很容易了嘛"
3. **需要tradeoff**: 像Min一样，对不同type的firm有不同的approval probability

---

## Diagnosis: Why Feb 27 Is Trivial

In Feb 27, the FDA observes (π, ∅) — it knows which experiment was conducted. So:
- Silence = bad result with certainty (since δ_g = 1)
- Complete unraveling: μ(G|∅) = μ(G|b) < 1/2
- A(∅) = 0 in any equilibrium
- Model is identical to Min with A_b = 0

Even with two types and screening, the model is **isomorphic to Min**: the firm effectively takes max(A_b, A_∅) as bad-news approval probability. Since FDA controls both, it sets one "effective bad-news approval prob" — exactly Min's A_b.

---

## Solution: The Mar 2 Model

### Core modification
**When the firm discloses**, FDA sees (π, j) — experiment and verified outcome.
**When the firm stays silent**, FDA sees only ∅ — doesn't know which experiment was conducted or what result was obtained.

### Why this breaks unraveling
Silence pools information from different types and experiments. The FDA can't determine:
- Which type of firm stayed silent
- Which experiment produced the hidden bad result
- The informativeness of the hidden experiment

The silence posterior becomes a weighted average across types:

$$\mu(G|\emptyset) = \frac{p \sum_i t_i (1 - y_i)}{p \sum_i t_i (1 - y_i) + (1-p) \sum_i t_i x_i}$$

### Why this is NOT isomorphic to Min
In Min, FDA sets different A_b^i for each type (6 instruments: π₁, π₂, A_g^1, A_g^2, A_b^1, A_b^2).
In this model, FDA has a **single pooled A(∅)** (5 instruments: π₁, π₂, A_g^1, A_g^2, A(∅)).

The pooling constraint A_b^1 = A_b^2 = A(∅) is the source of new tradeoffs.

### Key results in Mar 2

1. **Lemma (Disclosure equilibrium)**: Good results always disclosed (δ_g = 1). Bad results always hidden when A(∅) > 0 (δ_b = 0).

2. **Lemma (Silence bound)**: Under ex post optimality, μ(G|∅) < 1/2, so A(∅) = 0 and model collapses to Min. This is the benchmark.

3. **Proposition (Benchmark)**: Under ex post optimal FDA behavior, cherry-picking is irrelevant (same as Feb 27).

4. **Key insight**: The FDA may benefit from **committing** to A(∅) > 0 (sacrificing ex post optimality) to improve screening, analogous to Min's commitment to A_b > 0.

5. **Within-option neutrality**: Firm's experiment choice within a menu option does NOT depend on A(∅). This is the **same as Min** (firm always maximizes Pr(g|π) regardless of A_b).

6. **Screening channel**: A(∅) affects which menu option each type prefers (IC constraints). This is where the genuine new effects arise.

7. **How A(∅) changes IC**: If big type's experiment produces g more often, raising A(∅) tightens IC₁ (makes deviation to small type's option more attractive). If big type's experiment produces g less often, raising A(∅) relaxes IC₁.

---

## Analysis of Herresthal (2022, JET)

### What makes Herresthal's model non-trivial
- DM doesn't know how long the agent tested or what he found when silent
- Silence is informationally ambiguous (not just "bad news")
- Key result: hidden testing can Pareto-dominate observable testing when conflict is large

### Connection to Mar 2
- Herresthal compares two regimes (observable vs hidden testing)
- Mar 2 studies optimal mechanism design within hidden-disclosure regime
- Both share the mechanism: discretion over disclosure → agent acquires more/better information → potential mutual benefit

### Herresthal's core mechanism
1. Observable testing: agent stops early at t* (to prevent DM from getting bad news)
2. Hidden testing: agent keeps testing to T (can hide bad results)
3. Extra testing sometimes reveals ω₀ (both agree on withdrawal) → Pareto improvement

---

## Literature Recommendations (from GPT, assessed by Claude)

### Must-read and cite:
- **Herresthal (2022, JET)**: Core reference, hidden testing + selective disclosure
- **Henry & Ottaviani (2019, AER)**: Sequential costly research + approval mechanism, commitment power
- **Dahm, González, Porteiro (2018, JPubE)**: Mandatory disclosure + detection probability + fines, imperfect enforcement optimal
- **Dahm, González, Porteiro (2009, JHE)**: Trial registry + voluntary results, transparency vs trial incentives

### Useful but secondary:
- **Di Tillio, Ottaviani, Sørensen (2021, Econometrica)**: Strategic sample selection
- **Libgober (2022, AEJ Micro)**: Experiment dimensions partly hidden, "more transparency not always better"
- **Henry (2009, EJ)**: Hidden research effort + verifiable selective disclosure

### For motivation/empirics only:
- **Kao (2025, ManSci)**: Competition-driven selective withholding (empirical)
- **Tyagi, Hermosilla, Shah (2025, MSOM)**: Transparency → innovation composition change
- **Jin, Luca, Martin (2022, ManSci)**: Truthful-but-complex disclosure (obfuscation)

### Assessment: No need to borrow model settings from these papers
The Mar 2 model's core structure is sound. The "within-option neutrality" is not a bug — Min has the same feature. The contribution is the pooling constraint on A(∅) and its implications for screening.

---

## Next Steps

1. **Solve the optimization problem**: Characterize optimal A(∅) for specific feasible-set structures
2. **Compare with Min**: Show when pooling constraint binds and how optimal mechanism differs
3. **Welfare comparison**: Full transparency (Min) vs hidden disclosure with optimal A(∅) vs ex post optimal (trivial)
4. **Extensions**: Audit probability α (from Dahm et al. 2018), pre-registration as policy instrument
5. **Write related literature section**: Position relative to Herresthal, Henry & Ottaviani, Dahm et al.

---

## Files Created
- `Mar2_cherry_picking.tex`: New LaTeX model file with modified information structure
- `conversation_backup.md`: This file

---

## Key Takeaway

The single modification — FDA doesn't observe π under silence — transforms the model from trivial (Feb 27) to interesting (Mar 2):

| | Feb 27 | Mar 2 |
|--|--------|-------|
| FDA sees under silence | (π, ∅) | Only ∅ |
| Unraveling | Complete | Broken (type uncertainty) |
| A(∅) | = 0 always | Design variable, possibly > 0 |
| Instruments | Same as Min | Pooling constraint: one A(∅) for all types |
| Contribution | None (trivial) | Constrained screening with pooled bad-news instrument |
