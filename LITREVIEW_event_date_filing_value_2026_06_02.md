# Lit-review draft — event date, filing value, and information acquisition

**Date:** 2026-06-02. Paper-ready prose, organized around four literature
threads + the AI/discoverability mechanism + a regulator-synthesis anchor.
All citations verified against a fetched source (status note at the end).

**Spine sentence (organizing thesis):**
> The earnings announcement is the primary accounting information event;
> 10-Q/10-K filings retain incremental value when they contain non-preempted,
> hard-to-process, or text-based information; and EDGAR/web access captures
> costly information acquisition. Our paper studies a friction in the *first
> stages* of that acquisition — whether search/discovery crawlers can reach
> the firm's disclosures at all — and its consequence for liquidity.

---

## Thread 1 — The earnings announcement is the primary information event, and increasingly so

The information-content tradition anchors the market's reaction to accounting
information on the **earnings announcement** (Ball and Brown 1968; Beaver
1968). For *liquidity* specifically — our outcome — the convention is the
same: spreads widen and depth falls around the announcement (Lee, Mucklow, and
Ready 1993), the adverse-selection component of the spread spikes around it
(Krinsky and Lee 1996; Affleck-Graves, Callahan, and Chipalkatti 2002).
Crucially, this is not a static fact: **Beaver, McNichols, and Wang (2020)**
document that the market response to earnings announcements *rose* over
2001–2016 precisely because management guidance, analyst forecasts, and
disaggregated financial-statement line items are increasingly **bundled with
the announcement**. The corollary, which **Li and Ramesh (2009)** establish
directly, is that the subsequent periodic filing is often an information
non-event: the 10-Q reaction is largely subsumed by the earlier earnings
release, and the 10-K reaction, after removing concurrent releases, survives
only for the minority of filings near calendar quarter-ends. **Arif, Marshall,
Schroeder, and Yohn (2019)** show the bundling has intensified, with the rise
of concurrently released earnings announcements and 10-Ks.

*Where our paper sits:* we take this seriously and **anchor our main
specification on the earnings-announcement date**, not the filing date,
placing the headline liquidity result on the event the accounting literature
recognizes.

## Thread 2 — Filings retain incremental value when information is *not* preempted at the announcement

The filing is not always redundant. **Li, Nekrasov, and Teoh (2020)** show
that when firms withhold financial-statement items from the earnings
announcement and delay them to the 10-Q (a high "delayed-disclosure ratio"),
investor and analyst responses to the earnings news are correspondingly
delayed, and analysts defer forecasts from the announcement to after the 10-Q
filing. The incremental value of the filing is thus *conditional* — it
appears exactly where the announcement did not preempt it.

*Where our paper sits:* this legitimizes our **filing-date secondary
analysis** and tells us how to cut it — by whether the filing coincides with,
or is separated from, the earnings announcement.

## Thread 3 — Specific filing content produces filing-date reactions

Beyond timing, specific 10-Q content moves prices at the filing. **Balsam,
Bartov, and Marquardt (2002)** find a negative market reaction to
discretionary-accrual information around the 10-Q filing date, varying with
investor sophistication. **Filzen (2015)** finds that 10-Q risk-factor
*updates* generate negative abnormal returns at filing and predict future
negative earnings shocks — content that exists only in the filing, not the
announcement.

*Where our paper sits:* these establish that a filing-date liquidity response
is not prima facie spurious; there is genuine, sometimes hard-to-process,
incremental content whose discovery our treatment plausibly impairs.

## Thread 4 — EDGAR/web access captures costly information acquisition

A distinct literature treats access to disclosures as a costly,
behaviorally-revealed act. **Drake, Roulstone, and Thornock (2015)** show
EDGAR downloads cluster around corporate events and improve the price
formation of earnings news; **Gibbons, Iliev, and Kalodimos (2021)** show
analysts acquire information via EDGAR in a meaningful share of their forecast
updates; **Chang, Ljungqvist, and Tseng (2023)**, using the staggered EDGAR
rollout as a natural experiment, show that cheaper, timelier access to
mandatory disclosures substitutes for analysts' own information production.
**Neilson (2022)** adds an interpretive caveat we adopt: information-gathering
proxies reflect *both* information demand *and* unresolved uncertainty, so
access-based outcomes should not be read mechanically.

*Where our paper sits:* our treatment operates one step earlier than this
literature measures. These papers study *who chooses to acquire* a disclosure;
we study whether automated search/discovery agents *can reach* the firm's
disclosures at all when the firm restricts crawlers — a supply-side friction
on machine-mediated discovery.

---

## The mechanism: disclosure processing costs and impaired discovery

Our mechanism is best framed through **Blankespoor, deHaan, and Marinovic
(2020)**, whose review formalizes that disclosure is not free public
information: investors bear *awareness*, *acquisition*, and *integration*
costs. Search/discovery crawlability changes the first two stages — whether a
disclosure is found and retrieved at all. Two findings make the "information
exists but is not discovered/processed" channel concrete: **Cohen, Malloy, and
Nguyen (2020)** show that changes in 10-K/10-Q text predict future returns yet
produce *no announcement-day reaction* — investors are inattentive to public
filings sitting in plain sight; and **Matsumoto, Pronk, and Roelofsen (2011)**
show conference calls (typically bundled with the announcement) carry
incremental content, which we must control for or discuss when anchoring on
the announcement date.

*Where our paper sits:* restricting search/discovery crawlers raises the
awareness/acquisition cost of the firm's disclosures for the machine-mediated
investor base; the consequence we document is a deterioration in liquidity
(price impact), distinct from the analyst-intermediation and
information-diffusion channels we test and find null.

## Regulator synthesis (timely framing)

The policy debate confirms this is a mainstream concern, not a niche one. The
**SEC's May 2026 semiannual-reporting proposal** (Form 10-S in lieu of 10-Q)
explicitly weighs whether quarterly filings carry information incremental to
earnings releases, synthesizing exactly this literature — that quarterly
reports contain information but are frequently preempted by earnings releases,
so the value of 10-Q reporting must be assessed by content *type* and
*timing*. That a sitting regulator is actively litigating the
filing-vs-announcement information question is direct evidence that our
event-date design choice is a live, first-order issue.

---

## Citation status

**Verified (authors / venue / claim, against a fetched source):**
Ball and Brown (1968, JAR); Beaver (1968, JAR); Lee, Mucklow, and Ready (1993,
RFS 6(2):345–374); Krinsky and Lee (1996, JF 51(4):1523–1535); Affleck-Graves,
Callahan, and Chipalkatti (2002, JAR 40(3):561–583); Li and Ramesh (2009, TAR
84(4):1171–1208); **Beaver, McNichols, and Wang (2020, JAE 69(1), art. 101244)**;
Arif, Marshall, Schroeder, and Yohn (2019, JAE 68(1), art. 101221); **Li,
Nekrasov, and Teoh (2020, Review of Accounting Studies 25(1):159–200)**;
**Balsam, Bartov, and Marquardt (2002, JAR 40(4):987–1012)**; **Filzen (2015,
Accounting Horizons 29(4):887–916)**; Drake, Roulstone, and Thornock (2015,
CAR 32(3):1128–1161); Gibbons, Iliev, and Kalodimos (2021, Management Science
67(2):769–793); Chang, Ljungqvist, and Tseng (2023, RFS 36(8):3163–3212);
**Neilson (2022, JAE 74(1)** — title "Investor Information Gathering and the
Resolution of Uncertainty"; single author Jed Neilson)**; Blankespoor, deHaan,
and Marinovic (2020, JAE 70(2)); **Cohen, Malloy, and Nguyen (2020, JF
75(3):1371–1415)**; **Matsumoto, Pronk, and Roelofsen (2011, TAR
86(4):1383–1414)**; SEC semiannual-reporting proposal (May 2026, Form 10-S;
Federal Register 2026-09095).

**Do NOT cite (unresolved):**
- *Chiorean et al. (2025)* — could not independently verify; appears only as a
  citation inside the SEC proposal. Locate the actual paper before citing.

**Metadata reminders carried from prior checks:** Gao, Huang, **and Goldstein**
(2020, RFS) — never drop the third author; Ryans (2017) = TAR 92(2):161–184;
**dropped** Haghighi et al. (2023, Global Finance Journal) as a keystone
(journal tier too low). Do not cite Lee, Ma, and Wang (2015) for any
"5:30 p.m. filing-cutoff" claim (refuted — that is Cuñat and Groen-Xu).
