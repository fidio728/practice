from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from clean_weburl import clean_compustat, find_default_input
from fetch_utils import fetch_robots
from robots_parser import classify_bot, parse_robots


BOT_CONFIG: List[Dict[str, Any]] = [
    {"bot": "GPTBot", "group": "training", "aliases": ["GPTBot"]},
    {"bot": "ClaudeBot", "group": "training", "aliases": ["ClaudeBot"]},
    {"bot": "Google-Extended", "group": "training", "aliases": ["Google-Extended"]},
    {"bot": "CCBot", "group": "training", "aliases": ["CCBot"]},
    {"bot": "Meta-ExternalAgent", "group": "training", "aliases": ["Meta-ExternalAgent"]},
    {"bot": "OAI-SearchBot", "group": "search", "aliases": ["OAI-SearchBot"]},
    {"bot": "Claude-SearchBot", "group": "search", "aliases": ["Claude-SearchBot"]},
    {"bot": "PerplexityBot", "group": "search", "aliases": ["PerplexityBot"]},
    {"bot": "ChatGPT-User", "group": "user", "aliases": ["ChatGPT-User"]},
    {"bot": "Perplexity-User", "group": "user", "aliases": ["Perplexity-User"]},
    {
        "bot": "Meta-ExternalFetcher",
        "group": "user",
        "aliases": ["Meta-ExternalFetcher"],
    },
    {"bot": "Googlebot", "group": "control", "aliases": ["Googlebot"]},
]
BOT_ORDER = [item["bot"] for item in BOT_CONFIG]
BOT_LOOKUP = {item["bot"]: item for item in BOT_CONFIG}
TRAINING_BOTS = [item["bot"] for item in BOT_CONFIG if item["group"] == "training"]
SEARCH_BOTS = [item["bot"] for item in BOT_CONFIG if item["group"] == "search"]
USER_BOTS = [item["bot"] for item in BOT_CONFIG if item["group"] == "user"]
TREATMENT_BOTS = TRAINING_BOTS + SEARCH_BOTS
AI_BOTS = [item["bot"] for item in BOT_CONFIG if item["group"] != "control"]


def now_utc() -> tuple[str, str]:
    stamp = datetime.now(timezone.utc)
    return stamp.isoformat().replace("+00:00", "Z"), stamp.date().isoformat()


def classify_status_type(fetch_result: Dict[str, Any]) -> str:
    status_code = fetch_result.get("status_code")
    if status_code == 200:
        return "ok_200"
    if status_code == 404:
        return "missing_404"
    if isinstance(status_code, int) and 400 <= status_code < 500:
        return "http_error_4xx"
    if isinstance(status_code, int) and 500 <= status_code < 600:
        return "http_error_5xx"

    fetch_error = str(fetch_result.get("fetch_error") or "")
    if fetch_error.startswith("timeout"):
        return "timeout"
    if fetch_error.startswith("ssl_error"):
        return "ssl_error"
    if fetch_error.startswith("connection_error"):
        return "connection_error"
    if fetch_error.startswith("too_many_redirects"):
        return "too_many_redirects"
    return "request_error"


def _counts_or_none(bot_results: Dict[str, Dict[str, Any]], bots: Iterable[str], key: str) -> int | None:
    values = [bot_results[bot][key] for bot in bots]
    if any(value is None for value in values):
        return None
    return int(sum(int(value) for value in values))


def assign_treatment_v1(bot_results: Dict[str, Dict[str, Any]], robots_status_type: str) -> str:
    if robots_status_type not in {"ok_200", "missing_404"}:
        return "fetch_error"

    root_blocked = _counts_or_none(bot_results, TREATMENT_BOTS, "root_block") or 0
    nonroot = _counts_or_none(bot_results, TREATMENT_BOTS, "has_nonroot_disallow") or 0

    if root_blocked == len(TREATMENT_BOTS):
        return "full_disallow"
    if root_blocked == 0 and nonroot == 0:
        return "allow"
    return "partial_disallow"


def summarize_host_result(host_result: Dict[str, Any]) -> Dict[str, Any]:
    bots = host_result["bots"]
    robots_status_type = host_result["robots_status_type"]
    host_result["googlebot_block"] = bots["Googlebot"]["root_block"]
    host_result["any_ai_specific_rule"] = (
        None
        if robots_status_type not in {"ok_200", "missing_404"}
        else int(any(bots[bot]["specific_rule"] == 1 for bot in AI_BOTS))
    )
    host_result["n_training_bots_root_blocked"] = _counts_or_none(bots, TRAINING_BOTS, "root_block")
    host_result["n_training_bots_has_nonroot_disallow"] = _counts_or_none(
        bots, TRAINING_BOTS, "has_nonroot_disallow"
    )
    host_result["n_search_bots_root_blocked"] = _counts_or_none(bots, SEARCH_BOTS, "root_block")
    host_result["n_search_bots_has_nonroot_disallow"] = _counts_or_none(
        bots, SEARCH_BOTS, "has_nonroot_disallow"
    )
    host_result["n_treatment_bots_root_blocked"] = _counts_or_none(
        bots, TREATMENT_BOTS, "root_block"
    )
    host_result["n_treatment_bots_has_nonroot_disallow"] = _counts_or_none(
        bots, TREATMENT_BOTS, "has_nonroot_disallow"
    )
    host_result["n_user_bots_root_blocked"] = _counts_or_none(bots, USER_BOTS, "root_block")
    host_result["n_user_bots_has_nonroot_disallow"] = _counts_or_none(
        bots, USER_BOTS, "has_nonroot_disallow"
    )
    host_result["treatment_category"] = assign_treatment_v1(bots, robots_status_type)
    return host_result


def scan_host(host: str) -> Dict[str, Any]:
    scan_utc, scan_date = now_utc()
    fetch_result = fetch_robots(host)
    robots_status_type = classify_status_type(fetch_result)
    bot_results: Dict[str, Dict[str, Any]] = {}

    if robots_status_type == "ok_200":
        groups = parse_robots(fetch_result.get("content"))
        for item in BOT_CONFIG:
            bot_results[item["bot"]] = classify_bot(groups, item["bot"], item["aliases"])
    elif robots_status_type == "missing_404":
        for item in BOT_CONFIG:
            bot_results[item["bot"]] = {
                "bot": item["bot"],
                "specific_rule": 0,
                "effective_source": "no_robots_file",
                "root_block": 0,
                "has_nonroot_disallow": 0,
            }
    else:
        for item in BOT_CONFIG:
            bot_results[item["bot"]] = {
                "bot": item["bot"],
                "specific_rule": None,
                "effective_source": "fetch_error",
                "root_block": None,
                "has_nonroot_disallow": None,
            }

    result = {
        "scan_utc": scan_utc,
        "scan_date": scan_date,
        "host": host,
        "status_code": fetch_result.get("status_code"),
        "robots_status_type": robots_status_type,
        "fetch_error": fetch_result.get("fetch_error") or "",
        "robots_final_url": fetch_result.get("final_url") or "",
        "http_fallback": int(bool(fetch_result.get("http_fallback"))),
        "ua_used": fetch_result.get("ua_used") or "",
        "bots": bot_results,
    }
    return summarize_host_result(result)


def load_progress(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    host_results = payload.get("host_results", [])
    return {entry["host"]: entry for entry in host_results}


def write_progress(path: Path, host_map: Dict[str, Dict[str, Any]]) -> None:
    payload = {
        "completed_hosts": sorted(host_map.keys()),
        "host_results": [host_map[key] for key in sorted(host_map.keys())],
    }
    for attempt in range(20):
        temp_path = path.with_name(f"{path.stem}.{os.getpid()}.{attempt}.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True)
            temp_path.replace(path)
            return
        except PermissionError:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt == 19:
                raise
            time.sleep(min(5.0, 0.5 * (attempt + 1)))


def clean_and_load(clean_output: Path, input_path: Path) -> pd.DataFrame:
    clean_compustat(input_path, clean_output)
    return pd.read_csv(
        clean_output,
        dtype={"gvkey": str, "cik": str, "cusip": str, "tic": str},
        keep_default_na=True,
    )


def host_rows_from_map(host_map: Dict[str, Dict[str, Any]], hosts: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for host in hosts:
        entry = host_map[host]
        rows.append(
            {
                "scan_utc": entry["scan_utc"],
                "scan_date": entry["scan_date"],
                "host": entry["host"],
                "status_code": entry["status_code"],
                "robots_status_type": entry["robots_status_type"],
                "fetch_error": entry["fetch_error"],
                "robots_final_url": entry["robots_final_url"],
                "http_fallback": entry["http_fallback"],
                "ua_used": entry["ua_used"],
                "googlebot_block": entry["googlebot_block"],
                "any_ai_specific_rule": entry["any_ai_specific_rule"],
                "n_training_bots_root_blocked": entry["n_training_bots_root_blocked"],
                "n_training_bots_has_nonroot_disallow": entry[
                    "n_training_bots_has_nonroot_disallow"
                ],
                "n_search_bots_root_blocked": entry["n_search_bots_root_blocked"],
                "n_search_bots_has_nonroot_disallow": entry[
                    "n_search_bots_has_nonroot_disallow"
                ],
                "n_treatment_bots_root_blocked": entry["n_treatment_bots_root_blocked"],
                "n_treatment_bots_has_nonroot_disallow": entry[
                    "n_treatment_bots_has_nonroot_disallow"
                ],
                "n_user_bots_root_blocked": entry["n_user_bots_root_blocked"],
                "n_user_bots_has_nonroot_disallow": entry[
                    "n_user_bots_has_nonroot_disallow"
                ],
                "treatment_category": entry["treatment_category"],
            }
        )
    return rows


def build_bot_panel(valid_df: pd.DataFrame, host_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in valid_df.itertuples(index=False):
        host_result = host_map[row.host_clean]
        googlebot_block = host_result["googlebot_block"]
        for bot_name in BOT_ORDER:
            bot_result = host_result["bots"][bot_name]
            rows.append(
                {
                    "scan_utc": host_result["scan_utc"],
                    "scan_date": host_result["scan_date"],
                    "firm": row.conm,
                    "ticker": row.tic,
                    "host": row.host_clean,
                    "gvkey": row.gvkey,
                    "bot": bot_name,
                    "bot_group": BOT_LOOKUP[bot_name]["group"],
                    "root_block": bot_result["root_block"],
                    "has_nonroot_disallow": bot_result["has_nonroot_disallow"],
                    "specific_rule": bot_result["specific_rule"],
                    "effective_source": bot_result["effective_source"],
                    "googlebot_block": googlebot_block,
                    "status_code": host_result["status_code"],
                    "robots_status_type": host_result["robots_status_type"],
                    "fetch_error": host_result["fetch_error"],
                    "robots_final_url": host_result["robots_final_url"],
                }
            )
    return pd.DataFrame(rows)


def build_firm_summary(valid_df: pd.DataFrame, host_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in valid_df.itertuples(index=False):
        host_result = host_map[row.host_clean]
        rows.append(
            {
                "scan_date": host_result["scan_date"],
                "firm": row.conm,
                "ticker": row.tic,
                "gvkey": row.gvkey,
                "weburl_original": row.weburl,
                "host": row.host_clean,
                "shared_host_flag": bool(row.shared_host_flag),
                "host_duplicate_count": int(row.host_duplicate_count),
                "status_code": host_result["status_code"],
                "robots_status_type": host_result["robots_status_type"],
                "googlebot_block": host_result["googlebot_block"],
                "any_ai_specific_rule": host_result["any_ai_specific_rule"],
                "n_training_bots_root_blocked": host_result["n_training_bots_root_blocked"],
                "n_training_bots_has_nonroot_disallow": host_result[
                    "n_training_bots_has_nonroot_disallow"
                ],
                "n_search_bots_root_blocked": host_result["n_search_bots_root_blocked"],
                "n_search_bots_has_nonroot_disallow": host_result[
                    "n_search_bots_has_nonroot_disallow"
                ],
                "n_treatment_bots_root_blocked": host_result[
                    "n_treatment_bots_root_blocked"
                ],
                "n_treatment_bots_has_nonroot_disallow": host_result[
                    "n_treatment_bots_has_nonroot_disallow"
                ],
                "n_user_bots_root_blocked": host_result["n_user_bots_root_blocked"],
                "n_user_bots_has_nonroot_disallow": host_result[
                    "n_user_bots_has_nonroot_disallow"
                ],
                "treatment_category": host_result["treatment_category"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Scan robots.txt policies for Compustat firms.")
    parser.add_argument("--input", type=Path, default=find_default_input())
    parser.add_argument("--clean-output", type=Path, default=script_dir / "compustat_2025_clean.csv")
    parser.add_argument("--host-output", type=Path, default=script_dir / "robots_host_scan.csv")
    parser.add_argument("--bot-output", type=Path, default=script_dir / "robots_bot_panel.csv")
    parser.add_argument("--firm-output", type=Path, default=script_dir / "robots_firm_summary.csv")
    parser.add_argument("--progress", type=Path, default=script_dir / "scan_progress.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-sleep", action="store_true")
    args = parser.parse_args()

    clean_df = clean_and_load(args.clean_output, args.input)
    valid_df = clean_df.loc[
        (~clean_df["weburl_missing"]) & (~clean_df["url_parse_error"]) & clean_df["host_clean"].notna()
    ].copy()

    unique_hosts = list(dict.fromkeys(valid_df["host_clean"].astype(str).tolist()))
    if args.limit is not None:
        unique_hosts = unique_hosts[: args.limit]
        valid_df = valid_df.loc[valid_df["host_clean"].isin(unique_hosts)].copy()

    host_map = load_progress(args.progress)

    for host in unique_hosts:
        if host in host_map:
            continue
        host_map[host] = scan_host(host)
        write_progress(args.progress, host_map)
        if not args.no_sleep:
            time.sleep(random.uniform(0.3, 0.5))

    host_rows = host_rows_from_map(host_map, unique_hosts)
    host_df = pd.DataFrame(host_rows)
    host_df.to_csv(args.host_output, index=False)

    bot_panel = build_bot_panel(valid_df, host_map)
    bot_panel.to_csv(args.bot_output, index=False)

    firm_summary = build_firm_summary(valid_df, host_map)
    firm_summary.to_csv(args.firm_output, index=False)


if __name__ == "__main__":
    main()
