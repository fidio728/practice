# PRE-REGISTERED DECISION RULE v2 — robots_spec construct-validity / selection test
**v2 supersedes v1 (signed 2026-06-03 AM). v1's gate logic was adversarially refuted (workflow
wj7rwjj13); composition verified (wmyslsnym). NEEDS RE-SIGN before any run.**
**When numbers come back, the answer is THIS rule — not a new construct, not a moved cutoff.**

## WHY v2 (what the two reviews changed in v1)
- **`robots_any` is NOT the gate.** Its control pool (1,226 in-sample) is a STRICT SUBSET of the
  others' (most-selected, not least), it is the LEAST precise construct (largest SE), and its
  result is UNIDENTIFIED (95% CI [−0.006,+0.056] includes both 0 AND +0.045) — it cannot adjudicate.
  It is also ~89% site-wide-wildcard events, so it doesn't isolate AI/search restriction.
- **Drop v1 Amendment-1's +0.045 benchmark.** Benchmarking one construct's CI against another
  construct's coefficient is a category error, and MDE@80%≈0.044≈0.045 makes "precise-null" unreachable.
- **The headline `robots_spec` estimate is SOUND, not an inert artifact** (b=+0.045 p=0.001, clean
  pre_p=0.42, jump-then-decay = identification signature). v1's "inert/fake-pretrend" label is RETRACTED.
- **The pretrend FAILURE is wildcard/full-site-lockdown-specific** (robots_root/root_wild), NOT a
  general access-state problem (robots_any pre_p=.99, robots_nroot pre_p=.99 PASS). = distressed-firm
  selection into blanket lockdown.
- **The real threat is SELECTION + CONSTRUCT-MIX, not identification failure.**

## VERIFIED COMPOSITION (publishable — 3 independent re-derivations, exact to the firm)
`robots_spec` is a MIXED construct. **In-sample (N=217, the regression population):**
**84 genuine open→blocked (38.7%) / 133 naming-on-wildcard (61.3%) / 12 truly-inert (5.5%).**
Firm-cohort (N=1,022) decomposition: genuine 413 (of which **379 open→ROOT** = cleanest access flip,
33 open→nonroot, 1 root-after); naming 609 (**429 nonroot→named-root escalation**, 148 nonroot-only
[path impact unverifiable from this file], 27 inert, 5 root-later). Controls: 13,022 clean-never, of
which **7,923 (60.8%) restrict the bots via wildcard/nonroot anyway.** For ~56% of canon treated firms
the spec onset is a named rule layered on an ALREADY-ACTIVE wildcard block (567/609 admitted via a
`bk_any==1` "clean" month; 474 at the 202101 panel edge).
"genuine" = FIRM-level "no observed block of ANY of the 23 bots before the named-rule month."

## THE QUESTION (corrected)
Does `robots_spec` carry the +0.045 effect on the GENUINE access-flip core (open→blocked), with a
clean lead-shape pretrend, against a HARMONIZED control pool — or is it concentrated in the
naming-on-wildcard firms (a specificity/escalation phenomenon) and/or driven by selection?

## THE RUN — ONE pre-specified attempt. No re-runs on other samples/windows/cohorts.
**Constructs (in-sample, X-strict, ALL against one HARMONIZED common never-restrict-anything control pool):**
- **H.** `robots_spec` FULL (217) — current headline, for reference.
- **G.** `robots_spec` ACCESS-CLEAN = in-sample genuine open-before subset (≈84); LEAD with the
  open→ROOT sub-core. **This is the gate.**
- **N.** `robots_spec` NAMING-ON-WILDCARD (≈133) — the specificity/escalation firms, separately.
- **Placebo:** search16 `sb_spec` vs control7 `cb_spec` (report descriptively; disclose ~99% sb/cb
  any-overlap + spec-filter confound — it cannot cleanly separate bot-type from selection).
**Spec:** filing-window `[0,+5]` (+ `[+6,+30]` placebo); TWFE (`reghdfe`) + BJS (`did_imputation`,
`leaveout`, `avgeffectsby(_grpD)` real var, `maxit(5000)`); `pretrends(12)` WITH the 12 lead
coefficients SAVED (do-file currently saves only e(pre_F)/e(pre_p) — must add lead extraction);
**winsorize the COMPONENT means** before differencing (not the difference only) + leave-one-firm-out
on distant leads; report 95% CI + **MDE/power per construct**; **covariate balance** G vs controls.
**Robustness columns (not gates):** clean-sample (drop SIC 6000-6999, SHRCD{10,11}, EXCHD{1,2,3},
per-filing `in_sample` join); the EA announcement-window decomposition (per signed 2026-05-28 memo).

## THRESHOLDS (committed; do NOT move post-hoc)
- **clean pretrend** ≡ BJS `pre_p ≥ 0.10` AND lead coefficients show no monotone drift toward onset
  (component-winsorized, leave-one-firm-out-robust). [shape + p, locked before viewing the plot.]
- **effect present** ≡ BJS `[0,+5]` p<0.05 AND TWFE p<0.10, both positive.
- **precise null** ≡ effect-not-present AND 95% CI excludes **δ** = a PRE-SPECIFIED economically-
  meaningful margin on the SAME construct (NOT +0.045). **δ TO BE SET AT RE-SIGN** — proposal:
  δ = +0.020 (≈ half the filing-window headline). Honest caveat: at ≈84 firms / SE≈0.013–0.016 the
  attainable verdict space is likely {effect-present, unidentified}; precise-null may be unreachable,
  in which case the honest label is **unidentified → LIMITATION**, not "no effect."
- **adequately powered** ≡ n_canon ≥ 80 (G is right at the edge; report MDE).

## DECISION (committed before looking)
1. **CAUSAL HEADLINE STANDS (restriction-onset)** ⟺ **G** clean-pretrend AND effect-present AND
   balanced vs harmonized controls. (Lead with the open→ROOT core.)
2. **RECAST AS SPECIFICITY/ESCALATION (descriptive, NOT causal "restrict→illiquidity")** ⟺ effect
   concentrated in **N** (naming-on-wildcard), absent/weak in **G**. No-clean-mechanism caveat; not a headline.
3. **SELECTION-CONFOUNDED → cannot identify** ⟺ **G** pretrend fails (lead-shape) OR G unbalanced and
   not fixed by harmonized controls. Descriptive + EA cross-check; pivot per below.
4. **UNIDENTIFIED → LIMITATION** ⟺ **G** effect-not-present AND 95% CI includes δ (plausible at n≈84).
   State as a limitation; do NOT claim "no effect."
- **Pivot target (branches 3/4) = the AI-citation-gap paper, NOT specificity-as-primary.** The
  citation-gap result gets its OWN non-path-dependent audit before carrying a paper.
- **Branch 4-naming (within-broad23 spec-vs-wild) inadmissible as a PRIMARY mechanism; search16-vs-
  control7 admissible as a descriptive placebo only (with the overlap/confound caveat).**

## ANTI-SPEC-SEARCH LOCKS
- ONE sample. A failing branch is NOT retried with other filters/windows/cohorts.
- Report the FULL composition + all windows + event-study shape; do not hide window-fragility.
- When G lands near a boundary, the answer is the rule, NOT a fourth construct.

## BASE RATE (honest, before the run)
The headline estimate is sound but the genuine core is thin (~84 in-sample) and ~61% of canon is
naming-on-wildcard. Plausible outcomes are 1 (stands on the genuine core), 2 (specificity/escalation),
or 4 (genuine core underpowered → unidentified). I am NOT pre-committed to any; the run + this rule decide.
```
CHANGELOG v1→v2: removed robots_any-as-gate + the +0.045 precise-null branch; retracted inert label;
restated the pretrend-failure as lockdown-specific; added verified composition + exact subsamples
(H/G/N); harmonized control pool + MDE/power; lead-coefficient extraction; component-winsor + LOO;
split Branch 4; demoted clean-sample to robustness; added EA-window; δ to be set at re-sign.
```
