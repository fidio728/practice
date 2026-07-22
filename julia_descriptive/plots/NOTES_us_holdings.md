# NOTES — fig_us_holdings_real_yoy_growth (Figure 2: real growth in US holdings)

Built by `julia_descriptive/build_desc_trend_us_holdings.py` (rerunnable).
Data: `julia_descriptive/output/desc_trend_us_holdings_real_growth.csv`
(sec_country incl. TOTAL_EUROPE, quarter_end, usd_value, n_firms_held,
cpi_index, cpi_2020_annual_avg, real_usd_2020, real_usd_2020_lag4q,
real_yoy_pct, value_currency, real_base_year).

## Underlying nominal series

Quarter-end USD market value of US-domiciled institutional funds' holdings of
firms listed in each of the 28 European countries (FactSet LionShares):

- `I_ict = SUM(adj_mv)` per (sec_entity_id, sec_country, investor_country,
  report_date); `issue_type IN ('EQ','AD')` (ordinary shares + depositary
  receipts); US holder = fund-level `investor_country='US'`.
- **Denomination: USD.** Verified two ways: `adj_mv = adj_holding x adj_price`
  holds row-by-row, and prices are USD (Japanese stocks $12-27, not
  hundreds of JPY; UK stocks in dollars, not pence). FactSet converts at spot
  FX, so dollar-value growth embeds FX movements (EUR depreciation lowers the
  USD value of unchanged holdings).
- Double-sourced: the `ownership_ict.parquet` aggregation is recomputed
  directly from `holdings_eom.parquet` (186.8M rows); the script HARD-FAILS
  on any cell difference. Last run: 0 difference on all 2,542 cells.
- No `ownership_share BETWEEN 0 AND 1` filter (deliberate deviation from
  04's own time-series CSV; that filter needs market_cap and drops legitimate
  dollar cells).

## Inflation adjustment

CPI-U all items NSA (FRED CPIAUCNS; cached at output/cpiaucns_monthly.csv).
`real_usd_2020 = usd_value * (2020 annual-average CPI) / (CPI of the
quarter-end month)`. 2020 annual average = 258.811.

## Growth definition

`real_yoy_pct` = percent change of real_usd_2020 vs the SAME quarter one year
earlier (exact calendar match, validated one-to-one; missing prior year ->
missing growth). YoY (not QoQ) smooths semi-annual reporting sawtooth.

## Figure conventions

- Plotted from 2005Q1 (`PLOT_START`); 1999-2004 kept in the CSV but dominated
  by LionShares coverage ramp-up (YoY in the thousands of percent).
- Headline stats printed by the script cover the shown period only.
- Common y-axis on ALL panels, clipped at [-100%, +200%]: keeps countries
  comparable and pushes residual coverage-artifact spikes (2014/2019/2020
  reporting expansions; late-starting CEE countries) off-scale instead of
  letting them squash their own panels. Off-scale excursions are artifacts,
  not data to interpret.
- 'Total Europe' panel = 28-country aggregate (distinct-firm count computed
  at the aggregate level, not summed over countries).
- Missing quarters break the line; isolated observations drawn as dots.

## Interpretation caveats

1. US holder = fund domicile, not ultimate manager: US-managed Luxembourg/
   Irish UCITS count as non-US; the US total is a lower bound.
2. Coverage expansions (2014, 2019, 2020) appear as spurious growth spikes;
   they are reporting-regime changes, not investment.
3. Market-value growth mixes net purchases, local valuation changes, and FX.
   Real deflation removes the US price level only. Stripping FX or valuation
   requires a flow decomposition (possible extension, not in this figure).
