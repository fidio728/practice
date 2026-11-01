# Memo — event-date redesign: announcement-date MAIN, filing-date secondary

**For:** progress meeting / responding to the professor's critique.
**Date:** 2026-05-28 (rev. 2). Decision taken: **adopt the redesign.** All
citations verified against a fetched source; corrections + tier flags below.

---

## 0. Decision

**Filing date is no longer the sole main event date.** New structure:

- **MAIN result → earnings-announcement window.** Abnormal Amihud anchored on
  the earnings-announcement date (Compustat `rdq`, or 8-K Item 2.02 release
  date as a refinement):
  `mean Amihud[EA, EA+5] − mean Amihud[EA−30, EA−6]`. The robots.txt broad23
  DID runs on this. This puts us on the disclosure event the accounting
  literature recognizes.
- **FILING date → secondary / mechanism.** Kept, but the filing-date sample is
  split three ways relative to the earnings announcement:
  1. filing concurrent with the EA (same day / tight window),
  2. filing *after* the EA (the common case),
  3. filing with *no* nearby EA.
  Filing-date results are then framed as **incremental-filing-content / EDGAR
  discovery-access** tests, not as the market's primary reaction to earnings.

**Why this is not a concession but a strengthening:** investor and machine
attention concentrates at the earnings announcement (Li, Nekrasov & Teoh
2020). If crawler-blocking impairs *discovery* of the firm's disclosures, the
liquidity bite should be **largest exactly at the announcement** — so the
announcement anchor is the natural home for the discovery mechanism, and the
filing-date split isolates whatever *incremental* filing-specific effect
remains.

---

## 1. Why the critique must be taken seriously

- **Timing gap (verified).** 8-K earnings release ~41 days after FY-end;
  10-K ~12 days later. By filing time the earnings news is ~2 weeks old.
- **Li & Ramesh (2009)** TAR 84(4):1171–1208 — **CONFIRMED**. 10-Q filing
  reaction subsumed by the earlier earnings release; 10-K only a ~25%
  quarter-end residual after removing concurrent earnings. They explicitly
  note earnings announcements increasingly preempt the financial-statement
  disclosures in periodic filings.
- **The liquidity-event convention itself anchors on the announcement** —
  this is the sharpest part because it is our exact outcome class:
  - **Lee, Mucklow & Ready (1993)** RFS 6(2):345–374 — spreads widen, depth
    falls around the *announcement*. **CONFIRMED**.
  - **Krinsky & Lee (1996)** JF 51(4):1523–1535 — adverse selection spikes
    around the *announcement*. **CONFIRMED**.
  - **Affleck-Graves, Callahan & Chipalkatti (2002)** JAR 40(3):561–583 —
    same. **CONFIRMED**.
  - **Ball & Brown (1968)**, **Beaver (1968)** JAR 6 — the announcement is the
    information event (foundational). **CONFIRMED**.
- **Li, Nekrasov & Teoh (2020)** Review of Accounting Studies 25(1):159–200 —
  **CONFIRMED** (Yifan Li, Alexander Nekrasov, Siew Hong Teoh). Define a
  delayed-disclosure ratio = fraction of 10-Q items withheld at the EA;
  higher DD → greater delay in investor/analyst response, and analysts defer
  forecasts from the EA to after the 10-Q filing. **This is the keystone for
  the redesign:** attention is at the EA; whatever is held back to the filing
  is under-reacted. It both justifies announcement-as-main and motivates the
  filing-date secondary test (late-disclosed content).

---

## 2. Why filing-date stays in, as secondary / mechanism

A real literature treats the EDGAR filing/availability moment as its own
event — this is where the discovery/access mechanism lives:

- **Christensen, Heninger & Stice (2013)** Research in Accounting Regulation
  25(2):133–148 — **CONFIRMED**. Stock-price reactions to SEC filings are
  significant in the EDGAR era but not pre-EDGAR; analyst forecast revisions
  occur around filing dates. Directly supports a filing-date secondary spec.
- **Filzen (2015)** Accounting Horizons 29(4):887–916 — **CONFIRMED**. 10-Q
  *risk-factor updates* generate negative abnormal returns at the filing and
  predict future negative earnings shocks — concrete incremental 10-Q content
  that exists only at the filing, not the EA. Ideal for the
  "filing with no nearby EA / incremental content" arm.
- **Chang, Ljungqvist & Tseng (2023)** Review of Financial Studies
  36(8):3163–3212 — **CONFIRMED** (Yen-Cheng Chang, Alexander Ljungqvist,
  Kevin Tseng). EDGAR implementation as a natural experiment varying the
  timeliness/cost of accessing mandatory disclosures; mandatory disclosure
  substitutes for analyst information production. **Top-tier (RFS) anchor for
  the disclosure-access channel** — this is the cite that carries the access
  argument, in place of the weak-journal Haghighi paper.
- **Drake, Roulstone & Thornock (2015)** Contemporary Accounting Research
  32(3):1128–1161 — **CONFIRMED**. EDGAR access concentrates around corporate
  events; improves efficiency w.r.t. earnings news.
- **Gibbons, Iliev & Kalodimos (2021)** Management Science 67(2):769–793 —
  **CONFIRMED**. Analysts acquire info via EDGAR.
- **Blankespoor, deHaan & Marinovic (2020)** JAE 70(2) — **CONFIRMED**.
  Disclosure processing/acquisition costs — conceptual home for a
  crawler-access-friction mechanism.
- **Gao, Huang & Goldstein (2020)** RFS 33(4):1367–1411 — **CONFIRMED (never
  drop Goldstein)**. EDGAR raised outsider information production.
- Supporting (filing-day reactions exist): Griffin (2003) RAST 8(4):433–460;
  Qi, Wu & Haw (2000) JAAF 15(1):25–46; Asthana & Balsam (2001) J. Accounting
  & Public Policy 20(4–5):349–372; You & Zhang (2009) RAST 14(4):559–586;
  Bartov & Konchitchki (2017) Accounting Horizons 31(4):109–131. All
  **CONFIRMED**.

---

## 3. The redesign in operational terms

### MAIN — announcement-window abnormal Amihud
- Event date: `EA = rdq` (Compustat); refinement: 8-K Item 2.02 release date.
- Outcome: `abn_Amihud = mean Amihud[EA, EA+5] − mean Amihud[EA−30, EA−6]`.
- Spec: broad23 DID, firm FE + announcement-month FE, firm-clustered SE,
  winsor 1/99 — same machinery as the current filing-window headline.
- **DATA GAP (confirmed locally):** no announcement-anchored Amihud panel
  exists. `edgar_filing_event_panel.csv` is filing-date only;
  `rdq_drift_return_panel_serious.csv` has BHAR returns but no Amihud. → build
  an announcement-anchored Amihud panel reusing the daily-CRSP machinery in
  `build_edgar_filing_event_panel.py`, anchored on `ann_date = min(rdq,
  ANNDATS_ACT)` (already used in the PEAD code).

### SECONDARY / mechanism — filing-date, 3-way concurrent split
- Merge EA date into the filing panel (key: gvkey + fiscal quarter; EA date
  already available from the PEAD build).
- Classify each filing: (1) concurrent with EA, (2) filed after EA,
  (3) no nearby EA.
- Re-run the robots.txt-Amihud DID per class. The arm that matters is
  **(2)+(3)**: if the effect survives where the filing is *not* concurrent
  with the EA, it isolates the filing/discovery event and rebuts "stale
  earnings news" — answering Li & Ramesh on their own terms. Frame with
  Filzen (2015) / Li-Nekrasov-Teoh (2020) (incremental + late-disclosed
  content) and Chang-Ljungqvist-Tseng (2023) / Drake-Roulstone-Thornock
  (2015) (discovery access).

---

## 4. Citation hygiene

**REMOVED (tier too low to anchor an argument):**
- ~~Haghighi, Zhang, Oliver & Faff (2023), Global Finance Journal~~ — drop as
  a keystone access cite. The access channel is now carried by
  Chang-Ljungqvist-Tseng (2023, RFS), Drake-Roulstone-Thornock (2015, CAR),
  Blankespoor-deHaan-Marinovic (2020, JAE), Gao-Huang-Goldstein (2020, RFS).

**DO-NOT-CITE for the stated claim (failed verification earlier):**
- Lee, Ma & Wang (2015) JFE 116(2):410–431 — "5:30 p.m. filing-cutoff timing"
  claim is **REFUTED** (that is Cuñat & Groen-Xu). Lee-Ma-Wang is
  search-based peer firms.
- Easton & Zmijewski (1993) JAR 31(1):113–129 — pre/post-EDGAR reaction claim
  **UNVERIFIABLE** (paper is about reporting lags).

**Newly verified (this revision), all CONFIRMED:**
- Christensen, Heninger & Stice (2013) Research in Accounting Regulation
  25(2):133–148.
- Filzen, Joshua J. (2015) Accounting Horizons 29(4):887–916.
- Li, Yifan; Nekrasov, Alexander; Teoh, Siew Hong (2020) Review of Accounting
  Studies 25(1):159–200. (Distinct from Edward Xuejun Li of Li & Ramesh.)
- Chang, Yen-Cheng; Ljungqvist, Alexander; Tseng, Kevin (2023) Review of
  Financial Studies 36(8):3163–3212.

**Metadata fixes (carry these):** Gao-Huang-**Goldstein** (2020, RFS — never
drop the third author); Ryans (2017) = The Accounting Review 92(2):161–184
(not JBF); Gomez-Heflin-Hall is a **2025** working paper.

---

## 5. One-line answer in the meeting

"We've taken the point: the main result now anchors on the earnings
announcement, where the accounting and the liquidity-event literatures both
say the disclosure event is (Ball-Brown, Beaver; Lee-Mucklow-Ready,
Krinsky-Lee). The filing date moves to a secondary, mechanism role, split by
whether the filing coincides with the announcement (à la Li & Ramesh), so we
can show any residual liquidity effect on non-concurrent filings reflects the
incremental disclosure / EDGAR-discovery channel (Filzen 2015; Li-Nekrasov-Teoh
2020; Chang-Ljungqvist-Tseng 2023; Drake-Roulstone-Thornock 2015), not stale
earnings news."

---

*All citations verified against fetched sources. Re-read any paper before
quoting a number into the manuscript.*
