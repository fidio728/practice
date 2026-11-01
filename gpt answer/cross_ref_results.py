from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


MANUAL_BENCHMARKS: List[Dict[str, Any]] = [
    {"section": "clean", "metric": "total_rows", "manual_value": 10489},
    {"section": "clean", "metric": "missing_weburl", "manual_value": 427},
    {"section": "clean", "metric": "parse_errors", "manual_value": 0},
    {"section": "clean", "metric": "valid_hosts", "manual_value": 10062},
    {"section": "clean", "metric": "unique_hosts", "manual_value": 4350},
    {"section": "clean", "metric": "shared_hosts", "manual_value": 911},
    {"section": "clean", "metric": "firms_on_shared_hosts", "manual_value": 6623},
    {"section": "scan_host", "metric": "ok_200", "manual_value": 3579},
    {"section": "scan_host", "metric": "missing_404", "manual_value": 352},
    {"section": "scan_host", "metric": "http_error_4xx", "manual_value": 229},
    {"section": "scan_host", "metric": "timeout", "manual_value": 125},
    {"section": "scan_host", "metric": "connection_error", "manual_value": 40},
    {"section": "scan_host", "metric": "ssl_error", "manual_value": 13},
    {"section": "scan_host", "metric": "http_error_5xx", "manual_value": 9},
    {"section": "scan_host", "metric": "too_many_redirects", "manual_value": 3},
    {"section": "firm", "metric": "allow", "manual_value": 5083},
    {"section": "firm", "metric": "partial_disallow", "manual_value": 4161},
    {"section": "firm", "metric": "fetch_error", "manual_value": 786},
    {"section": "firm", "metric": "full_disallow", "manual_value": 32},
    {"section": "firm", "metric": "googlebot_blocked_firms", "manual_value": 17},
    {"section": "firm", "metric": "any_ai_specific_rule_firms", "manual_value": 373},
]


def read_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, keep_default_na=True)


def compute_metrics(base_dir: Path) -> Dict[tuple[str, str], Any]:
    metrics: Dict[tuple[str, str], Any] = {}
    clean_df = read_if_exists(base_dir / "compustat_2025_clean.csv")
    host_df = read_if_exists(base_dir / "robots_host_scan.csv")
    bot_df = read_if_exists(base_dir / "robots_bot_panel.csv")
    firm_df = read_if_exists(base_dir / "robots_firm_summary.csv")
    intensity_df = read_if_exists(base_dir / "robots_sitemap_intensity_firm.csv")

    if clean_df is not None:
        metrics[("clean", "total_rows")] = len(clean_df)
        metrics[("clean", "missing_weburl")] = int(clean_df["weburl_missing"].fillna(False).sum())
        metrics[("clean", "parse_errors")] = int(clean_df["url_parse_error"].fillna(False).sum())
        valid_mask = (~clean_df["weburl_missing"].fillna(False)) & (~clean_df["url_parse_error"].fillna(False))
        valid_df = clean_df.loc[valid_mask & clean_df["host_clean"].notna()].copy()
        metrics[("clean", "valid_hosts")] = len(valid_df)
        metrics[("clean", "unique_hosts")] = valid_df["host_clean"].nunique()
        metrics[("clean", "shared_hosts")] = int(valid_df.loc[valid_df["shared_host_flag"], "host_clean"].nunique())
        metrics[("clean", "firms_on_shared_hosts")] = int(valid_df["shared_host_flag"].fillna(False).sum())

    if host_df is not None and not host_df.empty:
        for status, count in host_df["robots_status_type"].value_counts(dropna=False).items():
            metrics[("scan_host", str(status))] = int(count)

    if firm_df is not None and not firm_df.empty:
        for treatment, count in firm_df["treatment_category"].value_counts(dropna=False).items():
            metrics[("firm", str(treatment))] = int(count)
        metrics[("firm", "googlebot_blocked_firms")] = int((firm_df["googlebot_block"] == 1).sum())
        observable = firm_df["robots_status_type"].isin(["ok_200", "missing_404"])
        metrics[("firm", "any_ai_specific_rule_firms")] = int(
            ((firm_df["any_ai_specific_rule"] == 1) & observable).sum()
        )
        metrics[("firm", "firm_rows")] = len(firm_df)

    if bot_df is not None and not bot_df.empty:
        metrics[("bot_panel", "rows")] = len(bot_df)

    if intensity_df is not None and not intensity_df.empty:
        for source, count in intensity_df["intensity_source"].value_counts(dropna=False).items():
            metrics[("intensity", str(source))] = int(count)

    return metrics


def build_comparison(metrics: Dict[tuple[str, str], Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in MANUAL_BENCHMARKS:
        key = (item["section"], item["metric"])
        our_value = metrics.get(key)
        difference = None
        if isinstance(our_value, (int, float)) and isinstance(item["manual_value"], (int, float)):
            difference = our_value - item["manual_value"]
        rows.append(
            {
                "section": item["section"],
                "metric": item["metric"],
                "manual_value": item["manual_value"],
                "our_value": our_value,
                "difference": difference,
            }
        )

    if ("firm", "firm_rows") in metrics:
        firm_rows = metrics[("firm", "firm_rows")]
        rows.append(
            {
                "section": "consistency",
                "metric": "manual_bot_panel_row_claim_vs_12x_firms",
                "manual_value": 120900,
                "our_value": firm_rows * 12,
                "difference": (firm_rows * 12) - 120900,
            }
        )

    if ("bot_panel", "rows") in metrics:
        rows.append(
            {
                "section": "consistency",
                "metric": "actual_bot_panel_rows",
                "manual_value": None,
                "our_value": metrics[("bot_panel", "rows")],
                "difference": None,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-reference fresh outputs against manual benchmarks.")
    parser.add_argument(
        "--base-dir", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parent / "cross_ref_manual.csv"
    )
    args = parser.parse_args()

    metrics = compute_metrics(args.base_dir)
    comparison = build_comparison(metrics)
    comparison.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
