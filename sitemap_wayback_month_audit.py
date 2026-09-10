import argparse
import csv
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import requests


DEFAULT_JSONL = Path(r"D:/anoconda/bot_paper_data/sitemap_panel_results_eom_primary.jsonl")
DEFAULT_SUMMARY_CSV = Path("sitemap_wayback_month_audit_summary.csv")
DEFAULT_DETAIL_CSV = Path("sitemap_wayback_month_audit_detail.csv")
USER_AGENT = "Mozilla/5.0 (compatible; AcademicResearchBot/1.0; sitemap-audit)"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Audit whether Step 6 sitemap rows are exact-month Wayback snapshots "
            "or nearest-available fallback snapshots."
        )
    )
    p.add_argument("--input-jsonl", type=Path, default=DEFAULT_JSONL)
    p.add_argument("--start-ym", default="2021-01")
    p.add_argument("--end-ym", default="2025-08")
    p.add_argument("--sample-per-month", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sleep", type=float, default=0.5,
                   help="Seconds to sleep between live Wayback checks.")
    p.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    p.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAIL_CSV)
    p.add_argument("--include-non-strict-clean", action="store_true",
                   help="By default only audit rows that Step 6.5 would treat as strict clean.")
    p.add_argument("--months", default="",
                   help="Optional comma-separated explicit month list, e.g. 2024-01,2024-02.")
    return p.parse_args()


def month_diff(requested_ym, actual_ym):
    ry, rm = int(requested_ym[:4]), int(requested_ym[5:7])
    ay, am = int(actual_ym[:4]), int(actual_ym[5:7])
    return (ay - ry) * 12 + (am - rm)


def classify(requested_ym, actual_ym):
    if not actual_ym:
        return "missing", ""
    diff = abs(month_diff(requested_ym, actual_ym))
    if diff == 0:
        return "exact", diff
    if diff <= 3:
        return "near_le_3m", diff
    return "stale_gt_3m", diff


def is_strict_clean(row):
    err = row.get("error", "") or ""
    if err:
        return False
    if bool(row.get("had_child_errors")):
        return False
    if bool(row.get("url_cap_hit")):
        return False
    if bool(row.get("child_truncated")):
        return False
    try:
        return int(row.get("n_urls", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def parse_boolish(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def load_candidates(path, start_ym, end_ym, strict_clean_only):
    rows_by_ym = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            ym = row.get("year_month", "")
            if not ym or ym < start_ym or ym > end_ym:
                continue

            row["had_child_errors"] = parse_boolish(row.get("had_child_errors"))
            row["url_cap_hit"] = parse_boolish(row.get("url_cap_hit"))
            row["child_truncated"] = parse_boolish(row.get("child_truncated"))

            if strict_clean_only:
                if not is_strict_clean(row):
                    continue
            else:
                err = row.get("error", "") or ""
                try:
                    n_urls = int(row.get("n_urls", 0) or 0)
                except (TypeError, ValueError):
                    n_urls = 0
                if err or n_urls <= 0:
                    continue

            if not row.get("sitemap_url") or not row.get("timestamp"):
                continue
            rows_by_ym[ym].append(row)
    return rows_by_ym


def extract_actual_ts(resp):
    m = re.search(r"/web/(\d{14})", getattr(resp, "url", "") or "")
    if m:
        return m.group(1)
    memento = resp.headers.get("Memento-Datetime", "")
    m2 = re.search(r"(\d{8})", memento.replace("-", "").replace(":", "").replace(" ", ""))
    if m2:
        return m2.group(1) + "000000"
    return ""


def live_lookup_actual_ts(session, requested_ts, sitemap_url, sleep_s):
    req_url = f"https://web.archive.org/web/{requested_ts}id_/{sitemap_url}"
    try:
        resp = session.get(req_url, timeout=20, allow_redirects=True)
        actual_ts = extract_actual_ts(resp)
        http_status = resp.status_code
    except requests.RequestException:
        actual_ts = ""
        http_status = ""
    time.sleep(sleep_s)
    return actual_ts, http_status


def sample_rows(rows_by_ym, sample_per_month, seed):
    rng = random.Random(seed)
    sampled = {}
    for ym, rows in sorted(rows_by_ym.items()):
        if not rows:
            sampled[ym] = []
        elif len(rows) <= sample_per_month:
            sampled[ym] = list(rows)
        else:
            sampled[ym] = rng.sample(rows, sample_per_month)
    return sampled


def select_months(rows_by_ym, start_ym, end_ym, months_arg):
    if months_arg:
        wanted = [m.strip() for m in months_arg.split(",") if m.strip()]
        return {m: rows_by_ym.get(m, []) for m in wanted}
    return {ym: rows for ym, rows in rows_by_ym.items() if start_ym <= ym <= end_ym}


def write_detail(rows, path):
    fields = [
        "year_month",
        "host",
        "sitemap_url",
        "requested_timestamp",
        "actual_timestamp",
        "actual_year_month",
        "classification",
        "abs_month_diff",
        "http_status",
        "source",
        "norm_key",
        "digest",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_summary(summary_rows, path):
    fields = [
        "year_month",
        "population_rows",
        "sample_n",
        "exact_n",
        "near_le_3m_n",
        "stale_gt_3m_n",
        "missing_n",
        "exact_share",
        "near_le_3m_share",
        "stale_gt_3m_share",
        "missing_share",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)


def main():
    args = parse_args()
    strict_clean_only = not args.include_non_strict_clean

    print("=== sitemap_wayback_month_audit.py ===")
    print(f"  input_jsonl:       {args.input_jsonl}")
    print(f"  month window:      {args.start_ym} to {args.end_ym}")
    print(f"  sample_per_month:  {args.sample_per_month}")
    print(f"  strict_clean_only: {strict_clean_only}")

    rows_by_ym = load_candidates(
        args.input_jsonl,
        args.start_ym,
        args.end_ym,
        strict_clean_only,
    )
    rows_by_ym = select_months(rows_by_ym, args.start_ym, args.end_ym, args.months)
    sampled_by_ym = sample_rows(rows_by_ym, args.sample_per_month, args.seed)

    total_population = sum(len(v) for v in rows_by_ym.values())
    total_sample = sum(len(v) for v in sampled_by_ym.values())
    print(f"  population rows:   {total_population:,}")
    print(f"  sampled rows:      {total_sample:,}")

    detail_rows = []
    summary_rows = []
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT

    for ym in sorted(sampled_by_ym):
        sample = sampled_by_ym[ym]
        print(f"  auditing {ym}: sample_n={len(sample)}")
        for row in sample:
            requested_ts = row["timestamp"]
            recorded_actual_ts = row.get("actual_timestamp", "") or ""
            if recorded_actual_ts:
                actual_ts = recorded_actual_ts
                http_status = 200
                source = "row.actual_timestamp"
            else:
                actual_ts, http_status = live_lookup_actual_ts(
                    sess,
                    requested_ts,
                    row["sitemap_url"],
                    args.sleep,
                )
                source = "live_wayback_lookup"

            actual_ym = f"{actual_ts[:4]}-{actual_ts[4:6]}" if len(actual_ts) >= 6 else ""
            label, abs_diff = classify(row["year_month"], actual_ym)
            detail_rows.append({
                "year_month": row["year_month"],
                "host": row.get("host", ""),
                "sitemap_url": row["sitemap_url"],
                "requested_timestamp": requested_ts,
                "actual_timestamp": actual_ts,
                "actual_year_month": actual_ym,
                "classification": label,
                "abs_month_diff": abs_diff,
                "http_status": http_status,
                "source": source,
                "norm_key": row.get("norm_key", ""),
                "digest": row.get("digest", ""),
            })
        ym_rows = [r for r in detail_rows if r["year_month"] == ym]
        sample_n = len(ym_rows)
        counts = defaultdict(int)
        for r in ym_rows:
            counts[r["classification"]] += 1
        summary_row = {
            "year_month": ym,
            "population_rows": len(rows_by_ym[ym]),
            "sample_n": sample_n,
            "exact_n": counts["exact"],
            "near_le_3m_n": counts["near_le_3m"],
            "stale_gt_3m_n": counts["stale_gt_3m"],
            "missing_n": counts["missing"],
            "exact_share": round(counts["exact"] / sample_n, 4) if sample_n else "",
            "near_le_3m_share": round(counts["near_le_3m"] / sample_n, 4) if sample_n else "",
            "stale_gt_3m_share": round(counts["stale_gt_3m"] / sample_n, 4) if sample_n else "",
            "missing_share": round(counts["missing"] / sample_n, 4) if sample_n else "",
        }
        summary_rows.append(summary_row)
        print(
            f"    exact={summary_row['exact_n']} "
            f"near<=3m={summary_row['near_le_3m_n']} "
            f"stale>3m={summary_row['stale_gt_3m_n']} "
            f"missing={summary_row['missing_n']}"
        )
        write_detail(detail_rows, args.detail_csv)
        write_summary(summary_rows, args.summary_csv)

    print(f"\nWrote detail:  {args.detail_csv.resolve()}")
    print(f"Wrote summary: {args.summary_csv.resolve()}")


if __name__ == "__main__":
    main()
