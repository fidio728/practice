# Geoecon Advisor Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the two advisor figures as a China-link fraction and CPI-deflated year-over-year holdings growth, with auditable data outputs and a revised PDF.

**Architecture:** Put reusable, testable transformations in `desc_trend_metrics.py`; keep raw FactSet aggregation and plotting in the two existing build scripts. The holdings script downloads or reuses the official FRED/BLS CPIAUCNS CSV, then writes every intermediate metric needed to audit the figure.

**Tech Stack:** Python, pandas, DuckDB, Matplotlib, pytest, LaTeX.

---

### Task 1: Test and implement reusable transformations

**Files:**
- Create: `julia_descriptive/tests/test_desc_trend_metrics.py`
- Create: `julia_descriptive/desc_trend_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
def monthly_cpi_for_2019_to_2021_with_2020_average_100():
    dates = pd.date_range("2019-01-01", "2021-12-01", freq="MS")
    values = np.full(len(dates), 100.0)
    values[dates == pd.Timestamp("2020-12-01")] = 106.0
    values[(dates.year == 2020) & (dates.month != 12)] = 94.0 + 6.0 / 11.0
    return pd.DataFrame({"DATE": dates, "CPIAUCNS": values})

def test_add_fraction_rejects_numerator_above_denominator():
    frame = pd.DataFrame({"linked": [3], "universe": [2]})
    with pytest.raises(ValueError, match="exceeds"):
        add_fraction(frame, "linked", "universe", "share")

def test_add_fraction_computes_bounded_share():
    frame = pd.DataFrame({"linked": [2, 0], "universe": [10, 0]})
    out = add_fraction(frame, "linked", "universe", "share")
    assert out.loc[0, "share"] == pytest.approx(0.2)
    assert pd.isna(out.loc[1, "share"])

def test_deflate_and_add_yoy_uses_2020_annual_average():
    cpi = monthly_cpi_for_2019_to_2021_with_2020_average_100()
    frame = pd.DataFrame({"country": ["GB"], "quarter_end": ["2020-12-31"],
                          "usd_value": [200.0]})
    out = deflate_and_add_yoy(frame, cpi, "country")
    assert out.loc[0, "real_usd_2020"] == pytest.approx(200.0 * 100 / 106)

def test_deflate_and_add_yoy_requires_exact_four_quarter_gap():
    cpi = monthly_cpi_for_2019_to_2021_with_2020_average_100()
    frame = pd.DataFrame({"country": ["GB", "GB"],
                          "quarter_end": ["2020-03-31", "2021-06-30"],
                          "usd_value": [100.0, 120.0]})
    out = deflate_and_add_yoy(frame, cpi, "country")
    assert out["real_yoy_pct"].isna().all()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest julia_descriptive/tests/test_desc_trend_metrics.py -v`
Expected: collection fails because `desc_trend_metrics` does not exist.

- [ ] **Step 3: Implement the minimal API**

```python
def add_fraction(df, numerator, denominator, output):
    out = df.copy()
    if (out[[numerator, denominator]] < 0).any().any():
        raise ValueError("counts must be nonnegative")
    if (out[numerator] > out[denominator]).any():
        raise ValueError("numerator exceeds denominator")
    out[output] = out[numerator] / out[denominator].where(out[denominator] > 0)
    return out

def deflate_and_add_yoy(df, cpi, group_col, date_col="quarter_end",
                        nominal_col="usd_value", base_year=2020):
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    prices = cpi.rename(columns={"DATE": "cpi_date", "CPIAUCNS": "cpi_index"}).copy()
    prices["cpi_date"] = pd.to_datetime(prices["cpi_date"])
    prices["month"] = prices["cpi_date"].dt.to_period("M")
    out["month"] = out[date_col].dt.to_period("M")
    out = out.merge(prices[["month", "cpi_index"]], on="month", how="left", validate="many_to_one")
    base_cpi = prices.loc[prices.cpi_date.dt.year.eq(base_year), "cpi_index"].mean()
    out["real_usd_2020"] = out[nominal_col] * base_cpi / out["cpi_index"]
    lag = out[[group_col, date_col, "real_usd_2020"]].copy()
    lag[date_col] = lag[date_col] + pd.DateOffset(years=1)
    lag = lag.rename(columns={"real_usd_2020": "real_usd_2020_lag4q"})
    out = out.merge(lag, on=[group_col, date_col], how="left", validate="one_to_one")
    out["real_yoy_pct"] = 100 * (
        out["real_usd_2020"] / out["real_usd_2020_lag4q"] - 1)
    return out.drop(columns="month")
```

The growth function merges CPI by calendar month, uses the annual average CPI
for the base year, and computes growth only after an exact four-quarter date
match within each group.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest julia_descriptive/tests/test_desc_trend_metrics.py -v`
Expected: all tests pass.

### Task 2: Add the Revere denominator and fraction figure

**Files:**
- Modify: `julia_descriptive/build_desc_trend_china_links.py`
- Modify: `julia_descriptive/plots/NOTES_china_links.md`

- [ ] **Step 1: Extend the tests with an output-schema assertion**

```python
result = add_fraction(
    pd.DataFrame({"n_firms_cn_link": [4], "n_firms_universe": [20]}),
    "n_firms_cn_link", "n_firms_universe", "fraction_cn_link")
assert {"n_firms_cn_link", "n_firms_universe", "fraction_cn_link"} <= set(result)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run the focused pytest node; expect the new schema requirement to fail.

- [ ] **Step 3: Add denominator SQL and fraction plotting**

Create quarter-end country counts for all valid Revere firms and for the
European regression-universe subset; join them to the existing numerator grid,
call `add_fraction`, and plot `100 * fraction_cn_link`.

- [ ] **Step 4: Run the build and its assertions**

Run: `python julia_descriptive/build_desc_trend_china_links.py`
Expected: regenerated CSV/PNG/PDF, fractions in [0,1], and no universe-ordering
assertion failures.

### Task 3: Deflate holdings and plot year-over-year growth

**Files:**
- Modify: `julia_descriptive/build_desc_trend_us_holdings.py`
- Modify: `julia_descriptive/plots/NOTES_us_holdings.md`

- [ ] **Step 1: Add tests for CPI parsing and exact-quarter growth**

```python
march = out.loc[out.quarter_end.eq(pd.Timestamp("2021-03-31")), "real_yoy_pct"]
june = out.loc[out.quarter_end.eq(pd.Timestamp("2021-06-30")), "real_yoy_pct"]
assert march.notna().all()
assert june.isna().all()
```

- [ ] **Step 2: Run tests and verify RED**

Expected: failure until CPI parsing and transformed columns exist.

- [ ] **Step 3: Implement CPI cache, deflation, and plot**

Use `https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS`, retain the
existing nominal aggregation cross-check, write nominal and real columns, and
plot `real_yoy_pct` with a zero reference line.

- [ ] **Step 4: Run the build and inspect output diagnostics**

Run: `python julia_descriptive/build_desc_trend_us_holdings.py`
Expected: exact nominal aggregation match and regenerated CSV/PNG/PDF.

### Task 4: Revise and compile the research handout

**Files:**
- Modify: `julia_descriptive/Essay2_descriptive_figures.tex`
- Create: `julia_descriptive/geoecon_advisor_revised.pdf`

- [ ] **Step 1: Update prose and self-contained captions**

State the fraction denominator, CPI series/base, exact year-over-year formula,
FactSet USD denomination, and the market-value/coverage caveats.

- [ ] **Step 2: Compile twice**

Run LaTeX twice so references resolve. Expected: exit code 0 and a three-page
PDF.

- [ ] **Step 3: Render and visually inspect all pages**

Expected: readable labels, no clipping, no blank panels caused by a processing
error, and captions that match the plotted units.

### Task 5: Final verification

- [ ] **Step 1: Run the full focused test suite**

Run: `python -m pytest julia_descriptive/tests -v`
Expected: zero failures.

- [ ] **Step 2: Re-run both builders and LaTeX from a clean process**

Expected: zero exit codes and fresh timestamps on all outputs.

- [ ] **Step 3: Review the diff and deliver exact file paths**

Confirm only the scoped scripts, tests, notes, TeX, plan/spec, and generated
outputs changed; do not stage or commit without explicit user authorization.
