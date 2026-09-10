# GeoEcon project — meeting note, 4 August 2026

Participants: Emanuele Rizzo, Stefano Cassella, Xueyang Liu. Duration 55 min.
Source: Teams transcript (`Emanuele, Stefano, Xueyang.docx`). ASR artifacts noted at the end.

---

## PART 1 — Email-ready minute (this is the ≤10-minute version Stefano asked for)

**Measurement of supply-chain exposure to China.** The current variable is an extensive-margin
indicator (firm has at least one link to China), averaged across firms within a country. Two
limitations were raised: a firm cutting China links from 10 to 5 is still coded as exposed, and
substituting a Chinese upstream supplier with a non-Chinese one leaves the measure unchanged.
Whether the extensive or the intensive margin drives investor reaction is an empirical question,
so we will construct both. Three country-level measures were agreed:

1. **M1 — country aggregation.** Numerator: total number of China links summed over all firms in
   the country. Denominator: total number of links (suppliers and customers) summed over the same
   firms.
2. **M2 — value-weighted firm ratio.** Compute the China-link share firm by firm, then average
   across firms weighting by market capitalisation (assets or sales are alternatives).
3. **M3 — equal-weighted firm ratio.** Same firm-level ratio, averaged with 1/N weights.

M1 and M3 are not equivalent: when a few firms hold a large share of all links, they receive
different weight in the denominator.

Per-link sales values do not exist in the data. Emanuele may be able to obtain maritime shipment
data, which would give an intensity measure by shipment value, but only for manufacturing firms.
This is kept aside as a later refinement, not part of the first pass.

**Data correction.** A bug was found in the construction of the quarterly panel. Quarter-end dates
sometimes fall on a weekend, when there is no trading, so retaining only the last calendar day of
the quarter dropped a large number of observations. The fix is to stop working at daily frequency:
convert dates to quarterly periods, sort within quarter, take the last available observation, and
compute growth as the change relative to the previous quarter. After the correction, the first
figure is unchanged (it does not involve quarters) and the decline in US institutional holdings of
European firms after 2015 remains but is far smaller than in the earlier version, which had shown
an implausible drop exceeding 2008.

**Design.** The first analysis is a country-by-quarter panel, the counterpart of the state-level
design in Emanuele and Stefano's earlier paper, with the country replacing the US state. The
perspective is that of US institutional investors. The hypothesis is that a US-induced preference
to reduce ties with China is incorporated by US institutions, which tilt away from countries whose
firms have larger supply-chain exposure to China. We start with this single direction and extend
later to the reverse direction and to other geopolitical pairs. Both the tension measure and the
China-exposure measure enter lagged one quarter; holdings respond in the following quarter.

**Figures agreed.** Rather than one time series per country, which is hard to read collectively,
two summary figures were agreed, each with the bilateral US–China tension index plotted in the
background so that spikes are visible:

- *Figure A (investor side).* Two lines: holdings of US investors in firms with zero ties to
  China, and holdings of US investors in firms worldwide with positive ties to China. The object
  of interest is whether the wedge between the two lines widens at tension spikes.
- *Figure B (firm side).* Two lines: growth in China links for firms that have US investors, and
  growth in China links for firms with no or low US ownership. This tests whether firms
  pre-emptively cut China links in order not to lose US investors.

Figure A has one version per exposure measure (M1, M2, M3); in the meeting the
multiple-versions request was tied to the wedge figure. Figure B's arms are defined by US
ownership, so the exposure measures do not enter its construction; a per-measure Figure B appears
below only as part of the proposed ordering, not as something the advisors asked for.
[Corrected 2026-08-18 after a transcript re-audit: this paragraph previously said "each figure",
which over-generalized the transcript.]

**Endogeneity and identification.** The concern raised at the seminar is that US investors may
reduce holdings in China-linked firms not because of government pressure but because tension
shocks predict lower future profitability for firms exposed to China: a shock hurts Chinese firms,
and any firm with a Chinese supplier or customer suffers, so investors rationally sell on the same
news. The proposed answer is a triple difference. The sample contains non-US institutional
investors as well as US ones, and all of them share the same forecast about the fundamentals of
China-linked firms. If US investors react more strongly, the incremental reaction is attributable
to a characteristic specific to US investors, namely responsiveness to their own government,
rather than to exposure to China as such.

A second question was whether compliance with government pressure can be separated from
nationalistic or ESG-type preferences held by the investors themselves, which could spike at the
same time. Two routes were discussed, both amounting to a further difference on the investor side:

- Classify investors by how state-dependent they plausibly are, for example tax-exempt status
  (endowments, charities, pension funds), managerial ties to government, or a large portfolio
  allocation to regulated sectors such as utilities.
- Compare mutual funds with separate institutional accounts. Portfolio changes at mutual funds may
  partly reflect the preferences of retail investors supplying flows, whereas separate accounts
  have no such flows, which gives a falsification test.

Neither route requires the effect to be zero in one arm. It is sufficient that the effect is
stronger where the proxy for state-dependence is higher.

**Process.** A short written update is to be circulated roughly three days before each call, so
that the call starts from a shared picture rather than from reconstruction. Materials go into the
shared repository (Dropbox; Overleaf if the documents live there), and updates are to be announced
rather than left to be discovered. Two further rules from the same exchange (Emanuele, endorsed by
Stefano): presentations and written updates must not jump straight into regression tables; each
opens with what we are trying to achieve, the hypotheses, the basic data construction, and
motivating graphical evidence, and only then reaches the tables. And each meeting starts from a
structured account of what has been done and what was previously discussed, so the advisors are
not left doing detective work.

**Next steps and dates.** Xueyang sends this minute together with a proposed ordering of the next
steps and an updated version of the project document; per Stefano's request, the updated document
folds in the research design discussed in the chat before the seminar presentation. Emanuele and
Stefano then confirm the ordering. The figures are to be sent by the weekend of 15–16 August,
despite the summer school that week; the advisors will look at the delivered plots and send
feedback before the call so that changes can be made. The next call is Monday 24 August.
Subsequent cadence is roughly every ten days. Emanuele and Stefano both attend DFA and will
discuss the project in person in Ghent.

---

## PART 1b — Stefano's follow-up email, 4 August 2026, 18:03

Sent to Xueyang and Emanuele a few hours after the call, and not covered by the
minute above. Three items, quoted.

**1. How firms with no links are treated.** "You mentioned something about how you
treat 'no links' firms, but I did not quite catch that. Can you clarify?" This is
an open question awaiting an answer. The substance: the zero-exposure group is not
one thing. It pools firms that are covered by the relationship data and have
supply-chain relationships but none to China, and firms that have no measured
supply-chain relationships at all. At the end of 2023 the split is 4,276 against
4,241, and pooled across quarters the second kind is about 65 percent of the
group. The two are economically different and the answer should say so.

**2. What the "potential" measure is for.** "The 'potential' is not meant to solve
the problem addressed by the triple difference involving other investors, it is
meant to separate effects due to investor preferences (e.g., mutual fund investors
really start to dislike china-linked firms) from institutional investor ones."
This scopes the investor-heterogeneity step: it is aimed at preference-driven
versus institution-driven responses, not at the fundamentals confound, which the
triple difference already handles.

**3. Quartile sorts, and the timing.** "I would try also low versus high quartile
firms/countries by china-link share. Be mindful of the timeline (ideally a rise in
tensions/risk leads to rebalancing in a subsequent period)." Two directives in one
sentence. The quartile split of both firms and countries by China-link share is
advisor-specified, so the 25/75 cut is not a free researcher choice. The second
half restates in writing the one-quarter lag agreed on the call.

Stefano also asked Emanuele for a first version of a blueprint, and said the two
of them would give input on the ordering of next steps.

## PART 2 — Proposed ordering of next steps (Xueyang's proposal, for confirmation)

**Step 0 — rebuild the quarterly panel with the date fix.** Prerequisite for everything below.
Convert to quarterly periods, take the last observation within quarter, recompute growth rates.
Re-issue the existing descriptive figures on the corrected panel so that all later output rests on
one panel.

**Step 1 — construct M1, M2, M3 at country by quarter.** Report, for each, the cross-sectional
distribution and the time-series behaviour, so the three can be compared before any of them is
used in a regression. Note explicitly how much of the variation in each is extensive versus
intensive.

**Step 2 — Figure A, one panel per measure.** This is the deliverable due by 15–16 August. Include
the tension index in the background and state the exact rule used to define the zero-tie and
positive-tie groups, including how firms are reassigned when their exposure changes.

**Step 3 — Figure B, one panel per measure.** Firm-side response. Requires a rule for splitting
firms by US ownership, which should be predetermined relative to the tension shock rather than
contemporaneous.

**Step 4 — baseline country by quarter regression.** Holdings or holdings growth on the
interaction of lagged tension with lagged China exposure, with country and quarter fixed effects.
One specification per measure.

**Step 5 — triple difference on investor domicile.** Add non-US investor countries and interact,
so that the identifying variation is the differential response of US investors relative to
investors elsewhere facing the same fundamentals news. This is the specification that answers the
seminar critique, so it should be reached early rather than left to the end.

**Step 6 — investor heterogeneity.** State-dependence proxies first (tax-exempt status is the
cheapest to construct), then the mutual fund versus separate account comparison as a falsification
test.

**Step 7 — intensive margin from shipment values.** Deferred, and conditional on Emanuele
obtaining the maritime shipment data. Manufacturing only, so it is a robustness exercise rather
than a main specification.

**Step 8 — extensions.** Reverse direction (Chinese pressure on European firms with US ties),
other countries' investors reacting to maintain their own US relationship, and other geopolitical
pairs (US–Iran, US–Russia, South Korea–China). Not before the single-direction result is settled.

The spoken weekend commitment for 15–16 August was to finish and send the plots; Steps 0 to 2
serve that directly, and Step 3 is included in the same window as this proposal's addition. Steps
4 and 5 are the natural content of the 24 August call.

---

## PART 3 — Open items and things to verify

- **Identity of the bilateral tension index.** The transcript renders the authors as "Jakovell and
  Tongs", which is an ASR error. Confirm the correct citation before it goes into any document.
- **Bilateral versus unilateral construction.** The index was recently updated to be genuinely
  bilateral, where the previous version only captured US pressure on China. Confirm which version
  the current figures use, since the earlier figures may mix the two.
- **Weighting variable for M2.** Market capitalisation was the default suggested, with assets and
  sales as alternatives. Sales may be preferable if the mechanism runs through trade, but
  availability is the binding constraint.
- **Lead-lag structure.** One quarter was agreed for both the tension measure and the exposure
  measure. Whether the response also appears at longer horizons is worth showing, but the one
  quarter specification is the baseline.
- **Shared repository.** Confirm the existing Dropbox folder, add Stefano if he is not on it, and
  send the link explicitly rather than assuming it will be found.

## ASR artifacts in the source transcript

The speaker's name is variously rendered as Shen, Silvia, Leah, Xinli and "Mammalian LI"; all
refer to Xueyang. GPR appears as J.P.R., JPR and GPS. The tension index authors appear as
"Jakovell and Tongs". The advisors' earlier paper appears as "our GFE paper". "Vicente" is named
as the source of the investor-heterogeneity comment at the seminar. None of these have been
silently corrected in this note beyond what is stated here.
