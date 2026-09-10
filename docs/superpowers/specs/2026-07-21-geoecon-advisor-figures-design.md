# Geoecon Advisor Figures Redesign

## Objective

Replace the two level-only descriptive figures with normalized measures that
answer the advisor's July 2026 comments while preserving the project's local
FactSet Revere and LionShares definitions.

## Figure 1: China-link fraction

- Numerator: distinct firms with at least one active CUSTOMER or SUPPLIER
  relationship to a Chinese company at quarter-end, unchanged from the current
  figure.
- Denominator: all distinct firms in the same Revere universe whose
  time-versioned `home_region` places them in that country at the same
  quarter-end.
- Statistic: `100 * n_firms_cn_link / n_firms_universe`.
- Main display: the `all_revere` universe for the United States and the 28
  European countries. The CSV also retains the holdings-observed regression
  universe for European countries as an auditable alternative.
- Plot: country small multiples with percentage axes. The ordering follows the
  peak fraction, not the numerator count.

This denominator reduces the mechanical influence of country size and much of
Revere's expanding company coverage. It does not eliminate changes in Revere's
relationship-disclosure coverage, which remains a captioned limitation.

## Figure 2: real year-over-year holdings growth

- Nominal series: sum of FactSet LionShares `ADJ_MV` for US-domiciled funds,
  unchanged from the current figure.
- Currency: `ADJ_MV` is already represented in USD. FactSet's local Ownership
  V5 guide states that all prices are displayed in USD and defines `Adj_MV` as
  the adjusted position market value represented in USD.
- Deflator: US CPI-U, all items, not seasonally adjusted (`CPIAUCNS`), monthly.
  Quarter-end observations use the CPI for March, June, September, or December.
- Base: 2020 annual-average CPI. Real value is
  `nominal_usd * mean(CPI_2020) / CPI_quarter_end`.
- Displayed statistic: year-over-year percentage growth,
  `100 * (real_usd_t / real_usd_t-4 - 1)`. A four-quarter lag is valid only
  when observations are exactly four calendar quarters apart; gaps remain
  missing rather than being bridged.
- Output CSV: nominal USD, CPI, real 2020 USD, and real year-over-year growth.
- Plot: Total Europe plus all country panels, with a zero reference line.

The caption will state that market-value growth combines net trading,
valuation changes, exchange-rate effects already incorporated by FactSet, and
coverage changes. It will not describe the series as investment flows.

## Deliverables

- Revised build scripts and unit tests for fraction, deflation, and exact-lag
  growth logic.
- Revised CSVs and figure files in `julia_descriptive/output` and
  `julia_descriptive/plots`.
- Revised `Essay2_descriptive_figures.tex` and a compiled three-page PDF in the
  project directory.
- Updated notes documenting denominators, CPI source, currency, and caveats.

## Validation

- Assert numerator never exceeds denominator and fractions remain in [0, 1].
- Assert the regression universe denominator and numerator do not exceed the
  corresponding all-Revere cells.
- Preserve the existing independent holdings aggregation cross-check.
- Test that 2020 annual-average CPI leaves a 2020-average nominal amount
  unchanged, and that missing calendar quarters do not create false year-over-
  year growth observations.
- Compile the LaTeX document and visually inspect both revised figure pages.

