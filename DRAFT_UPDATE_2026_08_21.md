# Draft text blocks for the written update of 2026-08-21

Prepared 2026-08-10. Numbers are taken from the canonical v3.1 artifacts
(c6_panel.dta, country_measures_m1m2m3.csv, country_panel_step4 outputs, the
dq_variants run of 2026-08-10, and a new extensive/intensive decomposition
computed for this draft). Each block below is meant to be pasted into the
update with light editing only.

## (a0) The estimating equation, and its timing

For firm i, holder group h in {US, non-US} and quarter t, the primary
specification is

    dw_iht = b2 (US_h x CN_i,t-1) + b3 (US_h x CN_i,t-1 x S_t-1)
             + a_it + a_ht + a_ih + e_iht

estimated by reghdfe with the three pairwise fixed effects absorbed and
standard errors two-way clustered on firm and quarter. The outcome dw is the
change in the group's portfolio weight on the firm, computed against that
group's full global equity book. CN is the firm's share of supply-chain links
running to China. S is the geopolitical tension shock.

Both right-hand-side terms are lagged one quarter, as agreed on 4 August: CN
is measured at t-1 and the shock at t-1, with the holdings response measured
at t. Specifications carrying the contemporaneous shock S_t are retained and
labelled as timing diagnostics, not as the primary.

One property of the shock is worth stating because it qualifies what "lagged
one quarter" delivers. S is the AR(1) innovation of the monthly bilateral
tension index, and only the quarter-closing month is kept. So S_t-1 is the
March, June, September or December innovation of the previous quarter, and the
other two months of that quarter are not used. The shock is also mildly
serially correlated, corr(S_t, S_t-1) = 0.27, so the lagged and contemporaneous
cells are not orthogonal tests. Two alternative constructions address this
directly: summing the quarter's three monthly innovations, and fitting the
AR(1) at quarterly frequency to begin with. Both were run at both timings and
the null holds throughout (three-month sum, lagged: p = 0.34 to 0.58; native
quarterly, lagged: p = 0.29).

## (a1) How firms with no links are treated (answering Stefano's question of 4 August)

The zero-exposure group is not one thing, and that is the part worth clarifying.
A firm's exposure is the share of its active customer and supplier relationships
that run to China. Three cases produce something that looks like a zero.

A firm may be covered by the relationship data, have supply-chain relationships,
and have none of them running to China. Its exposure is a measured zero, and it is
the natural comparison for an exposed firm. A firm may be covered but carry no
active supply-chain relationship of any kind, in which case the ratio has no
denominator; we set it to zero rather than dropping the firm, so it enters as a
zero without any measurement behind it. A firm may be absent from the relationship
data altogether, and that firm is left unclassified and excluded rather than
treated as a zero.

The first two are pooled in every figure and in the regression sample, and the
second kind is the larger part. At the end of 2023 the split is 4,276 firms with a
positive supply-chain denominator and no Chinese counterparty against 4,241 with no
supply-chain links at all, of which 2,926 have no recorded relationship of any
type. Pooled across quarters the no-denominator kind is about 65 percent of the
zero group. Firm counts are in output/fig_A_firm_buckets.csv and the recode is
applied in 02_china_exposure.jl.

This matters for two things. Separating the two would change what the zero arm of
Figure A means, since most of that arm is currently firms with nothing to measure
rather than firms measured at zero. It also affects the quartile figures, which is
why the no-ties line there is drawn separately rather than folded into the bottom
quartile. Whether the second kind belongs in the comparison group at all is a
question we would like to settle on the call.

## (a) What the estimand is, and what it is not

The estimation compares one treated holder group, US investors, with a single
merged non-US comparison group. The coefficient of interest is therefore a
group-level average: it asks whether the aggregate US position in China-linked
European firms was reallocated differently from the aggregate non-US position.
It answers the aggregate-reallocation question. It does not identify the
government-compliance mechanism, because compliance operates at the level of
individual institutions and mandates. Investor-level heterogeneity is the
agreed next step for that question. The DDD structure mitigates the common
firm-fundamentals alternative, but it does not by itself rule out US-specific
funding, regulatory, or client-preference shocks.

## (b) Precision, not proof of zero

The primary coefficient is close to zero: b3 = -1.39e-7 (SE 1.03e-6,
p = 0.89, N = 909,638, two-way clustered on firm and quarter, 10,250 firm and
82 quarter clusters). The 95% confidence interval is [-2.17e-6, +1.89e-6] and
the minimum detectable effect at 80% power and 5% size is 2.89e-6. The result
is a bounded null, not evidence of a true zero.

That bound is looser than it needs to be, for a reason worth correcting. The
comparison arm is defined as every non-US investor country, which puts Chinese
investors themselves in the control group. They are 2.95 percent of the non-US
global book that forms the denominator of the outcome, rising to 9.2-9.8
percent in the June and December quarters and falling to 0.00 percent in March
and September on the semiannual reporting cycle, while contributing 0.0063
percent of the non-US holdings of European securities that form the numerator.
The control arm's portfolio weights therefore inherit a large alternating
movement in the denominator that carries almost no information about European
firms. Removing Chinese investors from the control arm, on the same panel and
the same specification, leaves the null intact and sharpens it considerably:
b3 = +1.07e-7 with SE 3.59e-7 (p = 0.77, N = 909,638), an SE roughly one third
of the reported one, implying a minimum detectable effect near 1.0e-6. Adding
Hong Kong to the exclusion changes nothing (b3 = +1.05e-7, SE 3.59e-7,
p = 0.77).

We report the wider bound as the headline because the specification that
produced it is the pre-registered one, and the tighter bound as the correction
we believe is right. Whether Chinese investors belong in the control group is
a question for the call: the literal reading of the design keeps them, since
they hold China-linked firms and face no US government pressure, but their
reporting cycle is what widens the interval. The same denominator hygiene issue
is not unique to China, and is listed as an open item below.

## (c) The cum3 horizon and the inference hierarchy

At the three-quarter cumulative horizon the wild cluster bootstrap gives
p = 0.043. The family-wise adjusted value across horizons is 0.132. The
circular-shift randomization gives p = 0.329. We report the circular-shift
number as the headline for this cell, following the pre-registered hierarchy.
The reason is specific. The shock series is serially correlated, so the
quarter-independence approximation that underlies the bootstrap becomes more
suspect as the horizon lengthens. The circular-shift test preserves the
shock's own autocorrelation structure under the null. This is a choice of the
more conservative and better-matched reference distribution at long horizons.
We do not claim the wild cluster bootstrap is invalid.

## (d) The fund-grain snapshot rule

The holdings data are read at fund grain with an as-of snapshot rule. Three
separate reappearance-window diagnostics were run to check that rule. All
three point the same way. We describe them as strong and consistent direct
evidence for the fund-grain rule, not as a proof.

## (e) Data-quality filter, its residual, and the four-version check

The shipped filter is V1. It is a union of two legs, applied only to positions
above $5bn. The market-cap leg fires when the position's market value exceeds
the issuer's inline market cap. The shares leg fires when the holding exceeds
shares outstanding. A row is dropped if either leg fires. V1 drops exactly 3
rows, $452.6bn in total. Each of the three fails both legs, so the leg
decomposition does not change V1's dollar count.

We reran the pipeline under four filter versions. V0 drops nothing. V1 is the
shipped filter, 3 rows and $452.6bn. V2 is the shares leg only, applied at any
size, with no market-cap proxy anywhere in the rule; it drops 304 rows and
$483.0bn. V3 is V2 plus a sentinel-price screen, 30,586 rows and $496.8bn. V2
is not V1 re-evaluated at a lower size threshold. It is narrower than V1 in one
dimension and wider in another. It is narrower because it drops the market-cap
leg. It is wider because it drops the size threshold. It was built that way on
purpose. The market-cap proxy is derived from the same feed the filter is meant
to police, so a proxy error could manufacture the filter's own justification,
and V2 is the version that cannot.

Two residual figures follow from those two definitions. They count different
sets of rows, and both are correct. The first is the union-leg residual below
the size threshold. Running both legs with the threshold set to zero drops 376
rows and $499.6bn. Subtracting V1 leaves 373 rows and $47.0bn, spread over 196
holder-country-quarters, as recorded in the audit of the filter. The second is
the shares-leg-only residual. V2 minus V1 leaves 301 rows and $30.38bn, read
live from the census as 304 minus 3 rows and $483.01bn minus $452.63bn; 163 of
the quarter-end holder-country cells in the deltas table move under V2. The
union figure is the larger of the two because it also catches sub-threshold
rows that fail only the market-cap leg. Neither figure supersedes the other.
Neither is stale. Any text that reports one of them has to say which one it is.

Across the 19,340-row deltas table, 1,408 country-quarter cells move at all.
Eight of those cells move by more than 1% of their own country-quarter total.
No cell moves by as much as 1% of the global V1 book. The movers are the ones
the filter was built around. Under V0 the dropped rows return: Poland 2017Q4 at
+1479.7% and +$239.7bn, Canada 2020Q4 at +9.74%, and the United States 2019Q4
at +0.80%. Under V2 and V3 the sub-threshold residual comes out: Indonesia
2017Q4 at -65.9% and -$4.20bn, falling from $6.38bn to $2.18bn, Slovenia 2015Q2
at -4.7%, and Mauritius 2004Q4 at -1.7%.

The two remaining checks had a narrower scope than the deltas table, and the
earlier text did not say so. The country-ranking comparison was run at 2018Q4
and 2022Q4 only. The Figure-A bucket-share comparison was run at 2018Q4, 2020Q1
and 2022Q4 only. Both spot sets exclude 2017Q4 and 2020Q4, which are exactly
the quarters carrying the two largest anomalies in the delta table. The earlier
figure of 0.0010pp on the global denominator and 0.0022pp on the EU denominator
is therefore a three-quarter maximum, not a full-sample maximum. The earlier
claim of zero rank moves is a two-quarter statement. Neither supports a claim
about the whole sample as it stands. Both checks have now been run on every
quarter, with per-quarter maxima and argmax cells, and with four quarters
reported by name whatever the maxima turn out to be. The results follow.

All-quarter result, ranking. Over the 100 quarter-ends of the ranking calendar
(1999Q1-2023Q4) and the three non-V1 versions, 4 country-rank positions differ
from V1, in 2 of the 300 version-quarters; every one of those is a single-place
swap and all four are under V3. The ordered top five is identical to V1 in all
300 version-quarters, and so is the top-five set. The largest value move for any
one EU country is 5.13 percent (V3, 2008Q2, Greece), and it moves no rank.

One caveat on that denominator, which the earlier spot-quarter version of this
check did not surface. The ranking family is built on investor_country = US and
EU security countries, and none of the three rows the shipped filter drops
satisfies both predicates: two are US securities and the third is held by a
Polish investor. V0, the do-nothing version, is therefore identical to V1 at
every quarter by construction rather than by evidence, and one third of the 300
version-quarters cannot move. The evidential denominator for the ranking claim
is the 200 V2 and V3 version-quarters.

All-quarter result, Figure A bucket shares. On the bucket calendar (66 quarters,
2007Q3-2023Q4; the membership file carries a 67th quarter, 2024Q1, which lies
outside the holdings panel and is excluded and reported in
dq_bucket_quarter_calendar_census.csv), the largest share movement against V1 is
3.6444pp on the global denominator and 8.7881pp on the EU denominator, both
attained at the same cell: V0, equal-weighted measure, 2017Q4, bottom quartile,
non-US holders.

That maximum belongs entirely to V0, which is not a candidate rule but the null
of applying no filter at all. Among the three impossibility filters that are
actually candidates, the figure barely moves: the largest movement across V2 and
V3 over the full calendar is 0.021pp global (V3, value-weighted, 2017Q4) and
0.060pp on the EU denominator (V3, value-weighted, 2018Q2). The result is
therefore a vindication of the shipped rule rather than a robustness problem.
What the exercise does establish is that not filtering at all is not an option:
the single Polish row is 3.8 percent of the non-US global book in 2017Q4.

Named quarters, reported regardless of the maxima. The two originally named
quarters were identified from the holder-country deltas table and are both
holder-country anomalies, 2017Q4 Polish and 2020Q4 Canadian, so the US-held
ranking family cannot move on either and its zeros there are structural rather
than evidential. The named set has been widened to the two quarters where the
US-held book does move, 2016Q4 and 2018Q2. Across all four named quarters and
all three non-V1 versions the ranking shows 0 rank moves; the largest value
moves are 0.705 percent at 2016Q4 (V2 and V3) and 0.426 percent at 2018Q2 (V3).
On the bucket side the two original quarters do carry the anomalies, since the
Polish and Canadian holders sit in the non-US group: 2017Q4 moves 3.6444pp
global and 8.7881pp EU under V0, and 2020Q4 moves 0.1533pp global and exactly
0.0000pp EU, the latter because the Canadian holder's security is American and
never enters the EU book.

Source files, all under output/dq_variants/: dq_ranking_quarter_summary.csv (400
version-quarter rows), dq_bucket_share_quarter_summary.csv (792 rows),
dq_named_quarter_rows.csv (64 rows), and dq_bucket_quarter_calendar_census.csv
for the excluded quarter. The old spot tables are retained as
dq_country_ranking_us_held_SPOT_SUBSET.csv and
dq_fig_A_country_bucket_shares_SPOT_SUBSET.csv and are a subset, not evidence on
their own.

## (f) Country-panel results (Step 4)

The outcome here is the change in log dollar holdings at the country-quarter
level. It is not the portfolio-weight estimand you specified on 4 August. The
country-level weight panel has not been built yet, so we cannot report the
object you asked for at this level; that is an open item, not a result.

The US leg is clean and is the part worth stating. Regressing the change in
log US holdings on the lagged interaction of tension with country exposure
gives coefficients indistinguishable from zero on all three measures: M1
b = -0.025 (p = 0.77, N = 1,940), M2 b = -0.076 (p = 0.44, N = 2,016), M3
b = -0.008 (p = 0.98, N = 2,036). This is the leg that speaks to the US
perspective, and it shows nothing.

The differenced block, US minus non-US, is negative on all three measures
(M1 -0.695, M2 -0.459, M3 -1.310), but we do not think that pattern is
interpretable, for three reasons we checked directly.

First, it is carried by two or three small countries. Dropping Poland from M1
moves the coefficient from -0.695 to -0.421 and the p-value from 0.02 to 0.21.
Dropping Lithuania and Latvia from M2 collapses it from -0.459 to -0.167.
Dropping Latvia from M3 halves it, from -1.310 to -0.655. Latvia is the
clearest case: within the M1, M2 and M3 estimation samples its exposure
measure is exactly zero in every one of its rows, so by construction it cannot
respond to the interaction, yet removing it moves the estimate substantially.
Effective cluster counts, adjusting for how concentrated the regressor
variance is across countries, are roughly 7 for M1, 7 for M2 and 14 for M3
rather than the nominal 27.

Second, the movement lives in the non-US leg, not the US one. On the same
samples the non-US leg gives +0.672, +0.383 and +1.309 against the three US-leg
zeros above, so the differenced coefficient is essentially the non-US leg with
its sign reversed. That is the opposite of the reading the design intends: the
triple difference is meant to isolate a US-specific increment, and there is no
US-specific movement here to isolate.

Third, the non-US leg is the one contaminated by reporting frequency. Its
quarter-on-quarter log change has negative first-order autocorrelation in all
28 countries, median -0.73, which is the semiannual reporting pattern
described in section (a0) showing up in the outcome. Adding country-by-
calendar-quarter fixed effects, the natural correction, cuts the M1 coefficient
roughly in half.

We therefore take no position on the differenced block and would rather not
build on it. What is needed before it means anything is the portfolio-weight
outcome at country level, a rule for the exposure measure's denominator (M1
currently has no floor on the link count: Poland reaches M1 = 0.5 on two
recorded links against a sample median of 0.012), and a fix for the reporting
sawtooth in the non-US leg. The circular-shift randomization test, the arbiter
under our pre-registered hierarchy, has not been run on these cells either.

## (g) Sample-span variants: a proposal for discussion

The main grid carries every firm-quarter cell, including cells outside the
firm's holdings-based existence span. In the audit panel 42.0% of cells sit
outside that span. The two estimates sit side by side as follows. Full grid:
b3 = -1.39e-7, p = 0.893, N = 909,638, 10,250 firms. Holdings-span
restriction: b3 = -1.84e-7, p = 0.890, N = 527,380, 7,988 firms. The null is
unchanged; the coefficient moves within a fraction of the standard error.

Our question for you is which sample should be the main specification. Option
one promotes the span-restricted sample to main, on the argument that cells
outside a firm's observed existence span are not economically at risk of
holding changes. Option two keeps the full grid as main and reports the span
restriction as robustness, on the argument that the span itself is derived
from the holdings data and so conditions on the outcome process. We lean to
option two but will follow your call. A third, cleaner variant based on
listing coverage windows is not currently buildable: the security coverage
file's ADJDATE field is a corporate-actions date, not a delisting date
(median ADJDATE minus last activity is -523 days), so a listing-window span
needs future symbology data.

## (h) Extensive vs intensive variation in M1/M2/M3

Method, in brief. For each country and adjacent quarter pair we restrict to
firms present in both quarters, split them into firms that switch
any-China-link status (extensive) and firms linked in both quarters
(intensive), and compute a two-player Shapley decomposition of the change in
each measure, with each margin's contribution defined as its average marginal
effect of moving that group from its t-1 states to its t states; the gap
between this balanced-set change and the actual measure change (panel entry
and exit, plus denominator and weight drift among never-linked firms) is a
residual, and quarter-to-quarter variance shares are covariance betas of each
component on the actual change across country-quarter cells.

The intensive margin contributes little under M1 and M3, at 5 and 4 percent,
but carries 35 percent under M2. The extensive and
composition margins carry most of the variation, and which of the two leads
depends on the measure: the blanket phrase "the extensive margin dominates" is
wrong for M3, where composition leads at 61% against 35% extensive. The
extensive margin is the majority contributor for M1 and M2 only.
For M1 the shares
are 57% extensive, 5% intensive, 39% residual (composition). For M2 they are
51% extensive, 35% intensive, 14% residual. For M3 they are 35% extensive,
4% intensive, 61% residual; the equal-weighted measure is the most exposed to
universe composition because every entrant shifts the mean directly. Within
the balanced continuing-firm set the extensive share rises to 93% (M1), 60%
(M2), and 91% (M3). The full Revere universe gives the same picture for M1
(55% extensive) and a larger composition share for M3 (75%). The
quarter-to-quarter movement in country exposure is mostly firms gaining or
losing China links, and universe composition, not deepening of link counts
among firms already linked. M2 is the partial exception: cap-weighting gives
continuing large firms a substantial intensive share.

## (i) M2 is a conditional-on-coverage measure

Market cap covers a declining share of the exposure-defined universe. In the
estimation universe the covered share of firm-quarters is 66.6% in 2010,
48.0% in 2015, 45.5% in 2020, and 42.9% in 2023. At the distinct-firm level
the 2023 figure is 46.9%. The reviewer's ~42.9% figure is confirmed at the
firm-quarter grain. M2 is therefore an average over the covered subset only,
which tilts toward larger listed firms, and its coverage falls over time as
the Revere-matched universe adds small unlisted firms. We report
mcap_covered_share for every country-quarter cell and read M2 jointly with
M1 and M3 rather than on its own. In the full Revere universe M2 is not
computable at all, since unmatched firms have no FactSet market cap.

## (j) Figures A and B: status and read
[ADDED 2026-08-18 after the coverage audit. Re-read every number below off the
current CSVs (E: drive) before sending; the Figure B percentages are the v1
vintage read.]

The figure package agreed on 4 August is built and current as of 10 August on
the v3.1 panel. Figure A exists at the firm level in a rolling
zero-tie-versus-positive-tie version, a fixed-cohort version frozen at 2017Q4,
and equal-weighted and market-cap-weighted quartile versions; at the country
level it exists in one version per exposure measure (M1, M2, M3), as
requested. Figure B exists in a two-bucket (no-US versus any-US) and a
three-bucket (no-US, low-US, high-US) version. All nine figures carry the
tension index as background. A LaTeX document with full construction notes in
the captions is at julia_descriptive/Essay2_figures_AB.tex.

The read on Figure A. Two denominators are available and they are an order of
magnitude apart, so the label matters. Against the US investors' full global
equity book, which is the denominator the regressions use and the one the
figures plot, the zero-tie share falls from 5.15 percent in 2018Q4 to 3.64
percent in 2022Q4 while the positive-tie share is 7.53 and 7.55 percent.
Against the European book alone, which is the within-Europe reallocation
diagnostic, the same three cells read 37.2 percent, 29.6 percent and 61.3
percent. An earlier draft of this block quoted the European figures under the
global label; the figures themselves use the global denominator throughout. The fixed-cohort version is much flatter than the rolling one, so a
large part of the raw wedge is composition (Revere reclassifying firms into
the positive arm) rather than reallocation. Any discussion should lead with
the fixed-cohort panel.

The read on Figure B runs opposite to the hypothesis it was built to test.
Over 2018 to 2023, year-over-year growth in China links averages 19.1 percent
for firms with US institutional owners against 9.3 percent for firms with
none, with ranges of 3.1 to 37.3 percent and 0.6 to 20.0 percent respectively.
The ordering is not uniform: the no-US group grows faster in 6 of those 24
quarters. No pre-emptive de-linking is visible in any quarter.

Delivery status: [state here when the PDFs were sent, or send them with this
update].

## (k) Proposed ordering of next steps
[ADDED 2026-08-18. This is the ordering proposal the meeting asked for.]

1. Figures discussion and sign-off (the package above), including the
   composition caveat and whether the fixed-cohort framing becomes the lead.
2. The sample-span decision from block (g), which gates the full battery
   rerun.
3. Investor heterogeneity, in two stages: first the state-dependence
   classification (tax-exempt endowments, charities, pension funds are the
   cheapest to construct), then the mutual-fund versus separate-account
   comparison as a falsification test.
4. Firm-side response beyond the descriptive Figure B.
5. Extensions (reverse direction, other countries' investors, other
   geopolitical pairs) only after the single-direction result is settled.
