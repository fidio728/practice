# Session Notes: Mar2_v1_theory.tex
Date: 2026-03-03

---

## Q1: Why does the FDA's payoff from non-participation equal $1-p$?

**Passage:** "The FDA's payoff from non-participation equals $1-p$ (equivalent to always rejecting: correct on bad drugs, incorrect on good ones)."

**User confusion:** If the firm doesn't develop, there is no drug — so how can the FDA "reject" anything?

**Answer:**
- The phrasing is a mathematical equivalence, not a literal action description.
- If no drug exists, the FDA makes no decision. Her payoff is $1-p$ purely because:
  - $p = \Pr(\text{drug is Good}) < 1/2$, so most drugs are bad.
  - "Being correct" in a world with no drugs means bad drugs were never approved — this happens with probability $1-p$.
- The "always reject" framing is just a mnemonic to interpret the number $1-p$:

| Drug type | FDA action | Correct? | Payoff | Probability |
|-----------|-----------|----------|--------|-------------|
| Good | Reject | No | 0 | $p$ |
| Bad | Reject | Yes | 1 | $1-p$ |

Expected payoff $= p \cdot 0 + (1-p) \cdot 1 = 1-p$.

**Key point:** $1-p$ is the FDA's **outside option** (baseline). Any mechanism must deliver more than this to be worthwhile.

---

## Q2: Why is the firm's payoff $EU^A_i = A_g^i \cdot h_i + A(\emptyset) \cdot (1-h_i)$? Why no $A(b) \cdot \Pr(b|\pi_i)$ term?

**Formula in paper:**
$$EU^A_i = A_g^i \cdot h_i + A(\emptyset) \cdot (1-h_i)$$
where $h_i = \Pr(g|\pi_i) = py_i + (1-p)(1-x_i)$.

**User question:** Shouldn't the firm sometimes disclose bad outcomes? Why is there no $A(b)(1-h_i)$ term?

**Answer — two lemmas work together:**

1. **Equilibrium disclosure lemma:** The firm discloses $j$ iff $A(\pi,j) \geq A(\emptyset)$. It only discloses if doing so raises its approval probability above the baseline.

2. **Bad news lemma:** $\mu(G|\pi,b) < 1/2$. A bad result makes the drug look more likely bad. So the FDA approves less upon seeing $b$ than upon seeing silence: $A(b) < A(\emptyset)$.

**Consequence:** Since $A(b) < A(\emptyset)$, the firm **always hides $b$** and receives $A(\emptyset)$ instead. So the $b$-outcome is captured by the $A(\emptyset)(1-h_i)$ term — it just appears as silence.

---

## Q3: The missing WLOG lemma — is it really WLOG that $A(b) \leq A(\emptyset)$?

**User's observation:** The paper never formally proves $A(b) \leq A(\emptyset)$ in any optimal mechanism. The bad news lemma alone doesn't close the argument. There should be a third WLOG lemma.

**My thought process:**
- The bad news lemma gives $\mu(G|b) < 1/2$, meaning $b$ is bad news.
- But this doesn't automatically imply the committed approval rule satisfies $A(b) \leq A(\emptyset)$.
- The key gap: why wouldn't the FDA ever set $A(b) > A(\emptyset)$ to incentivize the firm?
- Answer: because the FDA is the mechanism designer and strictly prefers low $A(b)$ (approving bad news is a mistake). And crucially, the firm can always guarantee $A(\emptyset)$ by hiding $b$, so reducing $A(b)$ below $A(\emptyset)$ never hurts the firm's payoff.

**Gap in the paper:** Missing a lemma: WLOG $A(\pi_i, b) \leq A(\emptyset)$.

**Proof of the missing lemma:**
- *FDA side:* Since $\mu(G|b) < 1/2$, the FDA's ex-ante payoff is strictly decreasing in $A(b)$. She wants it as low as possible.
- *Firm side:* By the disclosure lemma, whenever $A(b) < A(\emptyset)$, the firm hides $b$ and receives $A(\emptyset)$. So the firm's effective payoff from the $b$-outcome is always $A(\emptyset)$, regardless of $A(b) \leq A(\emptyset)$. This means $A(b)$ below the threshold is irrelevant to every IC and IR constraint.
- Therefore: the FDA freely sets $A(b) \leq A(\emptyset)$, and the firm hides $b$.

---

## Q4: Mathematical proof that silence dominates bad news informationally

**User's intuition:** When FDA sees $\emptyset$ ("preliminary outcome" / silence), it can't be sure whether the trial had a good or bad result — it's ambiguous. But when it sees $b$, it knows for certain the result was bad. So silence should be weakly better news: $\mu(G|\emptyset) \geq \mu(G|\pi,b)$.

**Formalization:**

Let $\sigma_g := \Pr(\text{hide}|g)$ and $\sigma_b := \Pr(\text{hide}|b)$.

Define:
- $H_g$: firm hides a good result
- $H_b$: firm hides a bad result

Then $\Pr(\emptyset) = \Pr(H_g) + \Pr(H_b)$, and by the law of total probability:

$$\mu(G|\emptyset) = w \cdot \mu(G|g) + (1-w) \cdot \mu(G|b)$$

where $w := \Pr(H_g)/\Pr(\emptyset) \in [0,1]$ is the probability weight on silence having come from a hidden good result.

**Sub-claim:** $\mu(G|g) \geq \mu(G|b)$ for any $(x,y)$ with $y \geq 1-x$.

**Proof of sub-claim:**
$$\frac{py}{py+(1-p)(1-x)} \geq \frac{p(1-y)}{p(1-y)+(1-p)x}$$

Cross-multiplying:
$$p(1-p)xy \geq p(1-p)(1-x)(1-y)$$
$$\Leftrightarrow xy \geq (1-x)(1-y) \Leftrightarrow x+y \geq 1 \Leftrightarrow y \geq 1-x \checkmark$$

This holds exactly by the feasibility constraint.

**Conclusion:**
$$\mu(G|\emptyset) \geq \mu(G|\pi,b)$$
with **strict inequality** iff $w > 0$, i.e., iff $\sigma_g > 0$ (silence is genuinely ambiguous because the firm sometimes hides good results).

**Caveats:**
- In the post-WLOG equilibrium where $g$ is always disclosed ($\sigma_g = 0$), equality holds: $\mu(G|\emptyset) = \mu(G|b)$. Silence is perfectly revealing of $b$ in that equilibrium.
- The informational argument is strongest **before** applying WLOG — it captures the ambiguity the FDA faces at the commitment stage.
- After WLOG, the maintained $A(\emptyset) > A(b)=0$ is a design/incentive choice, not purely informational.

---

## Q5: Are the Remark and the WLOG lemma alternatives, or do you need both?

**Answer: They are complementary — keep both.**

| | Statement | Type |
|---|---|---|
| **Remark** | $\mu(G\|\emptyset) \geq \mu(G\|b)$ always | Information-theoretic (about posteriors) |
| **WLOG Lemma** | $A(b) \leq A(\emptyset)$ in any optimal mechanism | Mechanism design (about optimal rules) |

**Logical chain in the paper:**

1. **Bad news lemma** → $\mu(G|b) < 1/2$
2. **Remark** → $\mu(G|\emptyset) \geq \mu(G|b)$ (silence $\geq$ bad news informationally)
3. **WLOG ($A_b^i \leq A(\emptyset)$)** → FDA never rewards $b$ more than silence; sets $A(b) \leq A(\emptyset)$
4. **Disclosure lemma** → firm hides $b$
5. **WLOG ($A_g^i \geq A(\emptyset)$)** → firm discloses $g$
6. **Corollary** → simplified menu $\{A_g^i, A(\emptyset)\}$ is WLOG

The Remark answers *why* from a Bayesian perspective — it gives the FDA's informational rationale for the design choice in step 3. Without the Remark, the WLOG proof is still valid but less transparent.

---

## Changes made to Mar2_v1_theory.tex

### Added: Proof for `[Equilibrium disclosure]` lemma
Previously had the statement but no proof. Added:
> The firm compares $A(\pi,j)$ vs $A(\emptyset)$ and picks whichever is higher. The FDA cannot distinguish hidden outcomes, so hiding always yields $A(\emptyset)$.

### Added: New WLOG lemma — $A_b^i \leq A(\emptyset)$
**Label:** `lem:wlog_Ab`

Previously missing entirely. Now inserted between `[Equilibrium disclosure]` and `[WLOG: $A_g^i \geq A(\emptyset)$]`.

### Updated: Proof of `[WLOG: $A_g^i \geq A(\emptyset)$]`
Now explicitly cites the disclosure lemma.

### Added: `[Corollary: Simplified disclosure structure]`
**Label:** `cor:disclosure`

Ties the three lemmas together, explicitly stating that the full rule $\{A(\pi,g), A(\pi,b), A(\emptyset)\}$ reduces to $\{A_g^i, A(\emptyset)\}$.

### Added: `[Remark: Silence dominates bad news informationally]`
**Label:** `rem:silence`

Inserted between `[Bad news]` lemma and `[Equilibrium disclosure]` lemma. Proves $\mu(G|\emptyset) \geq \mu(G|b)$ as a general result using the convex-combination decomposition and the feasibility constraint $y \geq 1-x$.

---

## Session 2 (2026-03-15): Overlapping Feasible Sets & Final Paper

### Q6: What happens when $\pi_1 \in S_2$ (both IC constraints active)?

**Problem:** When the small type can mimic the big type's experiment, IC_2 is no longer automatic. Both IC constraints must bind simultaneously.

**Analytical results (Proposition 1):**

Both ICs binding gives: $(A_g^1 - A(\emptyset))h_1 = (A_g^2 - A(\emptyset))h_2$.

Since $h_1 > h_2$ and $h_1/h_2 > 1$: the small type's contract must offer higher approval ($A_g^2 > A_g^1$). Setting $A_g^2 = 1$ (optimal):

$$A_g^1 = A(\emptyset)(1 - h_2/h_1) + h_2/h_1 < 1$$

**Key findings:**
1. **Same $\alpha^*$**: The IR threshold $A(\emptyset) = (c - h_2)/(1 - h_2)$ is identical to the separate feasible sets case
2. **Distortion at the top**: $A_g^1 < 1$ (big type's approval distorted downward)
3. **No distortion at bottom**: $A_g^2 = 1$ (small type undistorted)
4. **Reversed screening**: Unlike standard Maskin-Riley screening (distort bottom, no distortion at top), here the TOP type is distorted because the bottom type wants to mimic upward
5. **V still decreasing in $A(\emptyset)$**: coefficient $= t[(2p-1) - \Delta_g(\pi_1)h_2/h_1] + (1-t)\Delta_b(\pi_2) < 0$

**Welfare loss from IC_2:**
$$V_A - V_B = t \cdot \Delta_g(\pi_1) \cdot (1 - A_g^1) > 0$$

### Numerical validation

**Parameters:** $p = 0.45, t = 0.5, \pi_1 = (0.70, 0.95), \pi_2 = (0.70, 0.75)$

| $c$ | Case A: $A_g^1$ | Case B: $A_g^1$ | $A(\emptyset)$ | $V_A$ | $V_B$ |
|-----|-----------------|-----------------|-----------------|-------|-------|
| 0.30 | 1.000 | 0.848 | 0.000 | 0.768 | 0.748 |
| 0.55 | 1.000 | 0.863 | 0.095 | 0.737 | 0.719 |
| 0.80 | 1.000 | 0.939 | 0.598 | 0.578 | 0.570 |

Scipy numerical solver (Case C) matches Case B to 3 decimal places.

### Files created

- **`Mar2_final.tex`**: Complete mini-paper with:
  - Abstract
  - Introduction with embedded literature review (Min 2023, Turner et al. 2008, Maskin-Riley 1984, Mussa-Rosen 1978)
  - Connection to class content (screening, IC, IR, revelation principle)
  - Full model, disclosure subgame lemmas, mechanism design problem
  - Closed-form results (Theorems 1-2, Corollaries)
  - New Section 6: Extension with overlapping feasible sets (Proposition 1 + numerical)
  - Figures referenced: fig1_approval_vs_cost.png, fig2_fda_payoff.png
  - Bibliography with proper \bibitem entries

- **`numerical_both_IC.py`**: Python script generating:
  - fig1_approval_vs_cost.png (3-panel: Cases A, B, C)
  - fig2_fda_payoff.png (FDA payoff comparison)
  - fig3_distortion.png (IC_2 distortion of $A_g^1$)
  - fig4_welfare_loss.png (welfare cost of IC_2)
