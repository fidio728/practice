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

The shipped filter (V1) drops exactly 3 impossible rows, positions above $5bn
where market value exceeds market cap or the holding exceeds shares
outstanding, totaling $452.6bn. A documented residual remains below the $5bn
threshold: 373 rows, $47.0bn, spread over 196 holder-country-quarters.
Indonesia 2017Q4 alone accounts for 65.9% of its country-quarter book.

We reran the pipeline under four filter versions: V0 drops nothing; V1 is the
shipped filter (3 rows, $452.6bn); V2 removes the same impossibility test at
any size (304 rows, $483.0bn); V3 adds a broader plausibility screen (30,586
rows, $496.8bn). Across the 19,340-row deltas table, 1,408 country-quarter
cells move at all. Only 8 move by more than 1% of the V1 global book. The
movers are the ones the filter was built around: under V0 the giant dropped
row returns (Poland 2017Q4, +1479.7%, +$239.7bn; Canada 2020Q4, +9.74%), and
under V2/V3 the sub-threshold residual comes out (Indonesia 2017Q4, -65.9%,
-$4.20bn, falling from $6.38bn to $2.18bn; Slovenia 2015Q2, -4.7%; Mauritius
2004Q4, -1.7%). US-held EU country rankings at 2018Q4 and 2022Q4 show zero
rank moves across all four versions; the GB/CH/FR/IE/DE top five is stable.
In the figure buckets the largest share movement against V1 is 0.0010pp on
the global denominator and 0.0022pp on the EU denominator. The filter choice
is irrelevant for the figures and rankings, apart from the named add-backs
and removals above.

## (f) Country-panel results (Step 4)

The country-quarter panel regressions use CRVE p-values and a score bootstrap
(Kline-Santos, Webb weights, 9999 replications) as the inference of record at
27 to 28 clusters. The bootstrap engine was switched at run time: boottest
does not run after reghdfe with two absorbed fixed-effect sets, so the
do-file's documented fallback was used for all cells.

The us_dlog block is cleanly null. M1: b = -0.025, p_crve = 0.77,
p_wild = 0.76, N = 1,940. M2: b = -0.076, p_crve = 0.44, p_wild = 0.45,
N = 2,016. M3: b = -0.008, p_crve = 0.98, p_wild = 0.98, N = 2,036.

The us_minus_nonus block, the DDD analogue, is uniformly negative with
marginal inference. M1: b = -0.695, p_crve = 0.024, p_wild = 0.060,
N = 1,938. M2: b = -0.459, p_crve = 0.104, p_wild = 0.025, N = 2,016.
M3: b = -1.310, p_crve = 0.092, p_wild = 0.080, N = 2,034. Only the M2
difference crosses 0.05 on the bootstrap. With 27 clusters we treat this
pattern as suggestive, not as a headline result.

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

In the estimation universe the extensive margin dominates. For M1 the shares
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
