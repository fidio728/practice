# Draft text blocks for the written update of 2026-08-21

Prepared 2026-08-10. Numbers are taken from the canonical v3.1 artifacts
(c6_panel.dta, country_measures_m1m2m3.csv, country_panel_step4 outputs, the
dq_variants run of 2026-08-10, and a new extensive/intensive decomposition
computed for this draft). Each block below is meant to be pasted into the
update with light editing only.

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
p = 0.89). The 95% confidence interval is [-2.17e-6, +1.89e-6]. That interval
still admits economically nonzero effects in both directions. The minimum
detectable effect at 80% power and 5% size is 2.89e-6. The design therefore
excludes only average differential responses larger than the CI and MDE scale.
The result should be read as a bounded null. It is not evidence of a true zero.

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
about the whole sample as it stands. Both checks have now been extended to every
quarter, with per-quarter maxima and argmax cells, and with 2017Q4 and 2020Q4
reported by name whatever the maxima turn out to be.

<<< PLACEHOLDER, to be filled from the Run phase of 2026-08-10 >>>
All-quarter ranking result: ___ rank positions differ from V1 across ___ non-V1
version-quarters. Quarters where the ordered top five differs from V1: ___.
Worst single rank move: ___. Worst percentage move: ___.
All-quarter Figure-A bucket result: the largest share movement against V1 is
___pp on the global denominator and ___pp on the EU denominator, attained at
___.
Named quarters, reported regardless of the maxima: 2017Q4 ___, 2020Q4 ___.
Source files, all under output/dq_variants/:
dq_ranking_quarter_summary.csv, dq_bucket_share_quarter_summary.csv,
dq_named_quarter_rows.csv. The old spot tables are retained as
dq_country_ranking_us_held_SPOT_SUBSET.csv and
dq_fig_A_country_bucket_shares_SPOT_SUBSET.csv and are a subset, not evidence
on their own.
Only once these are filled in can this block state whether the filter choice is
irrelevant for the figures and the rankings over the full sample.
<<< END PLACEHOLDER >>>

## (f) Country-panel results (Step 4)

These are preliminary estimates on an alternative outcome. The outcome is the
change in log dollar holdings at the country-quarter level. It is not the
portfolio-weight estimand you specified, so these coefficients do not answer
the reallocation question the design was built for. We report them because
they were run. We draw no inferential conclusion from them.

Two inference routines were computed. The first is cluster-robust standard
errors, reported below as p_crve. The second is a score bootstrap with Webb
weights and 9999 replications, following Kline and Santos, reported below as
p_wild. There are 27 to 28 clusters. Neither routine is the arbiter for this
panel. The bootstrap engine was switched at run time, because boottest does
not run after reghdfe with two absorbed fixed-effect sets, so the do-file's
documented fallback was used for all cells. The circular-shift randomization
test, which is the arbiter under the pre-registered hierarchy, has not been
run on these cells.

The us_dlog block is small and indistinguishable from zero on both routines.
M1: b = -0.025, p_crve = 0.77, p_wild = 0.76, N = 1,940. M2: b = -0.076,
p_crve = 0.44, p_wild = 0.45, N = 2,016. M3: b = -0.008, p_crve = 0.98,
p_wild = 0.98, N = 2,036.

The us_minus_nonus block, the DDD analogue, is negative for all three
measures. M1: b = -0.695, p_crve = 0.024, p_wild = 0.060, N = 1,938. M2:
b = -0.459, p_crve = 0.104, p_wild = 0.025, N = 2,016. M3: b = -1.310,
p_crve = 0.092, p_wild = 0.080, N = 2,034. The two routines disagree, and
they disagree in both directions. For M1 the bootstrap p-value is the larger
of the pair. For M2 it is the smaller, and M2 is the only cell that falls
below 0.05 on either routine. The common sign is a description of these
estimates. It is not a finding. With an outcome that is not the target
estimand, 27 clusters, two routines that contradict each other, and no
arbiter test on the cells, we take no position on whether these differences
are distinguishable from zero. That position waits on the portfolio-weight
outcome and on the circular-shift run.

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

The intensive margin contributes little under every measure. The extensive and
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
