# NOTES — fig_china_link_fraction (Figure 1: share of firms with China supply-chain links)

Built by `julia_descriptive/build_desc_trend_china_links.py` (rerunnable, no arguments).
Data: `julia_descriptive/output/desc_trend_china_link_fraction.csv`
(columns: country, quarter_end, n_firms_cn_link, n_firms_universe, universe,
fraction_cn_link).

## Numerator (unchanged from the count version)

Distinct firms with >=1 ACTIVE China supply-chain relationship at quarter-end:

- Relationship types: `CUSTOMER` and `SUPPLIER` only (3,057,294 of 5,556,608
  raw edges). `COMPETITOR` and all 13 `PARTNER-*` types EXCLUDED.
- China edge: one endpoint `home_region='CN'`, the other in C_SET
  (US + 28 European countries), both classified AS-OF the edge's `rel_start`.
- Active at quarter-end q: `rel_start <= q AND (rel_end IS NULL OR rel_end >= q)`.
- Firm country: `home_region` AS-OF q (time-versioned master, NULLS FIRST
  dedup tiebreaker — see the dedup note below).

## Denominators — three universes in the CSV

1. **supply_chain (MAIN, plotted)**: firms with >=1 active CUSTOMER/SUPPLIER
   relationship (any counterparty) at q, valid as-of home_region in the
   country. Numerator and denominator are built from the same edge data, so
   Revere's analyst-coverage expansion nets out to first order. This is the
   figure's universe and the recommended one: the all-Revere company master
   expands in batches (US master: 9,104 -> 17,421 in 2011, -> 46,996 in 2015)
   that mechanically crash the ratio while the numerator is smooth.
2. **all_revere**: every company in the Revere master with a valid as-of
   home_region. Kept in the CSV for comparison; NOT plotted.
3. **regression** (EU only): Revere firms crosswalked to the estimation panel
   (ever institutionally held; selection-on-outcome caveat). CSV only.

Nesting asserted in-script: supply_chain ⊆ all_revere on both counts;
regression ⊆ all_revere; numerator <= denominator everywhere (`add_fraction`
raises otherwise).

## Plot masking

Quarters with `n_firms_universe < 50` (supply-chain universe) are hidden in
the FIGURE (values retained in the CSV). Fractions over a handful of firms are
noise (Luxembourg 2003: 1 linked firm / 9 firms = 11%).

## How to read the shapes (IMPORTANT — composition drift)

Several European panels start high when they first clear the 50-firm
threshold and then decline (DE 25% (2009) -> ~7%; NL 18% -> ~6%; SE/DK/GR
similar). This is mostly COMPOSITION, not de-linking: the first firms Revere
covers in a country are its largest multinationals (highest China-link
propensity); as coverage broadens to mid/small caps, the ratio dilutes. The
US step-down around 2015 coincides with a large Revere coverage batch. The
fraction nets out the LEVEL of coverage growth, not its composition. Read
within-panel trends over the mature period (post-~2015) and cross-country
levels with this in mind; do not read the early peaks as "peak China
exposure".

## Coverage window

Quarter-ends 2003-06-30 .. 2025-03-31 (88 quarters). Edges start 2003-04-03;
vintage ends 2025-05-03 (2025Q2 right-censored, excluded).

## Dedup tiebreaker (deliberate deviation from 02)

Company-master dedup keeps the open row on same-start_d ties
(`end_d DESC NULLS FIRST`); 02_china_exposure.jl §1c-pre used NULLS LAST,
which drops live firms (independent audit of all 11 affected firms; tail-end
undercount <=0.36%). 02 (both china and russia variants) has been fixed
in-code on 2026-07-21 but its downstream parquets are not yet rebuilt.

## Other caveats

- Un-listed/private counterparties under-captured by Revere.
- Firms with NULL as-of region at q drop out of both numerator and
  denominator for that quarter.
- China-specific recording intensity: the fraction cancels overall coverage
  effort, not a possible post-2018 shift in analysts' attention to China
  links specifically. Same limitation applies to the regression exposure.
- "European" = 28 countries incl. GB/CH/NO.
