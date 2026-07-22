"""Reusable transformations for the geoecon advisor trend figures."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def normalize_fred_cpi(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a FRED CPIAUCNS download to DATE and numeric CPIAUCNS."""
    date_candidates = [name for name in ("DATE", "observation_date") if name in raw.columns]
    if len(date_candidates) != 1 or "CPIAUCNS" not in raw.columns:
        raise KeyError("FRED CPI data must contain DATE/observation_date and CPIAUCNS")
    out = raw[[date_candidates[0], "CPIAUCNS"]].rename(columns={date_candidates[0]: "DATE"})
    out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
    out["CPIAUCNS"] = pd.to_numeric(out["CPIAUCNS"], errors="coerce")
    return out.dropna(subset=["DATE", "CPIAUCNS"]).sort_values("DATE").reset_index(drop=True)


def add_fraction(
    frame: pd.DataFrame,
    numerator: str,
    denominator: str,
    output: str,
) -> pd.DataFrame:
    """Return a copy with a bounded numerator/denominator fraction column."""
    missing = {numerator, denominator}.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing count columns: {sorted(missing)}")

    out = frame.copy()
    counts = out[[numerator, denominator]].apply(pd.to_numeric, errors="coerce")
    if counts.isna().any().any():
        raise ValueError("Fraction counts must be non-missing numeric values")
    if (counts < 0).any().any():
        raise ValueError("Fraction counts must be nonnegative")
    if (counts[numerator] > counts[denominator]).any():
        raise ValueError("Fraction numerator exceeds denominator")

    out[numerator] = counts[numerator]
    out[denominator] = counts[denominator]
    positive_denominator = counts[denominator].where(counts[denominator] > 0)
    out[output] = counts[numerator] / positive_denominator
    return out


def assert_nested_universe(
    frame: pd.DataFrame,
    outer_label: str,
    inner_label: str,
    key_cols: Sequence[str],
    metric_cols: Sequence[str],
    universe_col: str = "universe",
) -> None:
    """Raise when an inner-universe count exceeds its matching outer count."""
    required = set(key_cols) | set(metric_cols) | {universe_col}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing universe comparison columns: {sorted(missing)}")

    outer = frame.loc[frame[universe_col].eq(outer_label), list(key_cols) + list(metric_cols)]
    inner = frame.loc[frame[universe_col].eq(inner_label), list(key_cols) + list(metric_cols)]
    paired = inner.merge(
        outer,
        on=list(key_cols),
        how="inner",
        suffixes=("_inner", "_outer"),
        validate="one_to_one",
    )
    violations = pd.Series(False, index=paired.index)
    for metric in metric_cols:
        violations |= paired[f"{metric}_inner"] > paired[f"{metric}_outer"]
    if violations.any():
        sample = paired.loc[violations, list(key_cols)].head(5).to_dict("records")
        raise ValueError(
            f"Universe '{inner_label}' is not nested within '{outer_label}' at {sample}"
        )


def deflate_and_add_yoy(
    frame: pd.DataFrame,
    cpi: pd.DataFrame,
    group_col: str | Sequence[str],
    date_col: str = "quarter_end",
    nominal_col: str = "usd_value",
    base_year: int = 2020,
) -> pd.DataFrame:
    """Add CPI-U, real base-year USD, and exact-calendar-year growth columns."""
    groups = [group_col] if isinstance(group_col, str) else list(group_col)
    required = set(groups + [date_col, nominal_col])
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing holdings columns: {sorted(missing)}")
    if not {"DATE", "CPIAUCNS"}.issubset(cpi.columns):
        raise KeyError("CPI data must contain DATE and CPIAUCNS columns")

    prices = cpi[["DATE", "CPIAUCNS"]].copy()
    prices["cpi_date"] = pd.to_datetime(prices.pop("DATE"), errors="coerce")
    prices["cpi_index"] = pd.to_numeric(prices.pop("CPIAUCNS"), errors="coerce")
    prices = prices.dropna(subset=["cpi_date", "cpi_index"])
    prices["month"] = prices["cpi_date"].dt.to_period("M")
    if prices["month"].duplicated().any():
        raise ValueError("CPI data contain duplicate monthly observations")

    base = prices.loc[prices["cpi_date"].dt.year.eq(base_year), "cpi_index"]
    if len(base) != 12:
        raise ValueError(f"CPI data must contain all 12 months of base year {base_year}")
    base_cpi = float(base.mean())

    out = frame.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out[nominal_col] = pd.to_numeric(out[nominal_col], errors="coerce")
    if out[[date_col, nominal_col]].isna().any().any():
        raise ValueError("Holdings dates and nominal values must be non-missing")
    if (out[nominal_col] < 0).any():
        raise ValueError("Nominal holdings values must be nonnegative")
    if out.duplicated(groups + [date_col]).any():
        raise ValueError("Holdings data contain duplicate group-date observations")

    out["month"] = out[date_col].dt.to_period("M")
    out = out.merge(
        prices[["month", "cpi_index"]],
        on="month",
        how="left",
        validate="many_to_one",
    )
    if out["cpi_index"].isna().any():
        missing_months = sorted(out.loc[out["cpi_index"].isna(), "month"].astype(str).unique())
        raise ValueError(f"Missing CPI for holdings months: {missing_months}")

    out["cpi_2020_annual_avg"] = base_cpi
    out["real_usd_2020"] = out[nominal_col] * base_cpi / out["cpi_index"]

    lag = out[groups + [date_col, "real_usd_2020"]].copy()
    lag[date_col] = lag[date_col] + pd.DateOffset(years=1)
    lag = lag.rename(columns={"real_usd_2020": "real_usd_2020_lag4q"})
    out = out.merge(
        lag,
        on=groups + [date_col],
        how="left",
        validate="one_to_one",
    )
    lagged = out["real_usd_2020_lag4q"].where(out["real_usd_2020_lag4q"] > 0)
    out["real_yoy_pct"] = 100.0 * (out["real_usd_2020"] / lagged - 1.0)
    return out.drop(columns="month").sort_values(groups + [date_col]).reset_index(drop=True)
