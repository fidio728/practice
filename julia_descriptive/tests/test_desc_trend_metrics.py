import numpy as np
import pandas as pd
import pytest

from julia_descriptive.desc_trend_metrics import (
    add_fraction,
    assert_nested_universe,
    deflate_and_add_yoy,
    normalize_fred_cpi,
)


def cpi_fixture() -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", "2021-12-01", freq="MS")
    values = np.full(len(dates), 100.0)
    other_2020 = (1200.0 - 106.0) / 11.0
    values[dates.year == 2020] = other_2020
    values[dates == pd.Timestamp("2020-12-01")] = 106.0
    return pd.DataFrame({"DATE": dates, "CPIAUCNS": values})


def test_add_fraction_rejects_numerator_above_denominator() -> None:
    frame = pd.DataFrame({"linked": [3], "universe": [2]})

    with pytest.raises(ValueError, match="exceeds"):
        add_fraction(frame, "linked", "universe", "share")


def test_add_fraction_computes_bounded_share_and_missing_zero_denominator() -> None:
    frame = pd.DataFrame({"linked": [2, 0], "universe": [10, 0]})

    out = add_fraction(frame, "linked", "universe", "share")

    assert out.loc[0, "share"] == pytest.approx(0.2)
    assert pd.isna(out.loc[1, "share"])
    assert {"linked", "universe", "share"} <= set(out.columns)


def test_deflate_uses_2020_annual_average_as_base() -> None:
    frame = pd.DataFrame(
        {"country": ["GB"], "quarter_end": ["2020-12-31"], "usd_value": [212.0]}
    )

    out = deflate_and_add_yoy(frame, cpi_fixture(), "country")

    assert out.loc[0, "cpi_index"] == pytest.approx(106.0)
    assert out.loc[0, "cpi_2020_annual_avg"] == pytest.approx(100.0)
    assert out.loc[0, "real_usd_2020"] == pytest.approx(200.0)


def test_yoy_growth_uses_exact_same_quarter_one_year_earlier() -> None:
    cpi = cpi_fixture()
    cpi.loc[cpi["DATE"].eq(pd.Timestamp("2020-03-01")), "CPIAUCNS"] = 100.0
    frame = pd.DataFrame(
        {
            "country": ["GB", "GB", "GB"],
            "quarter_end": ["2020-03-31", "2021-03-31", "2021-06-30"],
            "usd_value": [100.0, 110.0, 120.0],
        }
    )

    out = deflate_and_add_yoy(frame, cpi, "country")

    march = out.loc[out["quarter_end"].eq(pd.Timestamp("2021-03-31"))].iloc[0]
    june = out.loc[out["quarter_end"].eq(pd.Timestamp("2021-06-30"))].iloc[0]
    assert march["real_yoy_pct"] == pytest.approx(10.0)
    assert pd.isna(june["real_yoy_pct"])


def test_deflate_rejects_missing_quarter_end_cpi() -> None:
    frame = pd.DataFrame(
        {"country": ["GB"], "quarter_end": ["2022-03-31"], "usd_value": [100.0]}
    )

    with pytest.raises(ValueError, match="Missing CPI"):
        deflate_and_add_yoy(frame, cpi_fixture(), "country")


def test_nested_universe_rejects_inner_counts_above_outer_counts() -> None:
    frame = pd.DataFrame(
        {
            "country": ["GB", "GB"],
            "quarter_end": ["2020-03-31", "2020-03-31"],
            "universe": ["all_revere", "regression"],
            "n_firms_cn_link": [3, 4],
            "n_firms_universe": [10, 8],
        }
    )

    with pytest.raises(ValueError, match="not nested"):
        assert_nested_universe(
            frame,
            outer_label="all_revere",
            inner_label="regression",
            key_cols=["country", "quarter_end"],
            metric_cols=["n_firms_cn_link", "n_firms_universe"],
        )


def test_normalize_fred_cpi_accepts_observation_date_and_drops_missing_values() -> None:
    raw = pd.DataFrame(
        {
            "observation_date": ["2020-01-01", "2020-02-01"],
            "CPIAUCNS": ["258.7", "."],
        }
    )

    out = normalize_fred_cpi(raw)

    assert out.to_dict("records") == [
        {"DATE": pd.Timestamp("2020-01-01"), "CPIAUCNS": 258.7}
    ]
