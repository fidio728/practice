#!/usr/bin/env python3
"""
Scan robots.txt for a list of firm domains and output a bot-level panel.

Input CSV required columns:
- ticker
- domain
Optional columns:
- firm (or company/name)

Primary output (long panel): one row per firm-domain-bot-date.
Core fields requested by user:
- firm, domain, date, bot, root_block, path_block, googlebot_block, status_code

Usage:
  python scan_robots_bot_panel.py --input domains_input.csv --output robots_bot_panel.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import requests


BOT_DEFS: Dict[str, Dict[str, List[str] | str]] = {
    "GPTBot": {"group": "train", "aliases": ["gptbot"]},
    "ClaudeBot": {"group": "train", "aliases": ["claudebot"]},
    "Google-Extended": {"group": "train", "aliases": ["google-extended"]},
    "CCBot": {"group": "train", "aliases": ["ccbot"]},
    "OAI-SearchBot": {"group": "retrieval", "aliases": ["oai-searchbot", "oai-searchbot"]},
    "Claude-SearchBot": {"group": "retrieval", "aliases": ["claude-searchbot", "claude-search"]},
    "ChatGPT-User": {"group": "retrieval", "aliases": ["chatgpt-user"]},
    "Claude-User": {"group": "user", "aliases": ["claude-user"]},
    "Googlebot": {"group": "search_engine", "aliases": ["googlebot", "googlebot smartphone"]},
    "Bingbot": {"group": "search_engine", "aliases": ["bingbot"]},
}

TRAIN_BOTS = ["GPTBot", "ClaudeBot", "Google-Extended", "CCBot"]
RETRIEVAL_BOTS = ["OAI-SearchBot", "Claude-SearchBot", "ChatGPT-User"]


@dataclass
class FetchResult:
    status_code: int | None
    final_url: str
    content: str
    error: str


@dataclass
class RuleStatus:
    specific_rule: int
    effective_source: str  # specific / wildcard / none
    root_block: int
    path_block: int


def normalize_agent_token(token: str) -> List[str]:
    """Return one or more normalized user-agent tokens.

    Handles lines that list multiple agents separated by commas.
    """
    raw = token.strip().lower()
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts if parts else [raw]


def parse_robots(robots_text: str) -> List[Dict[str, List]]:
    """Parse robots.txt into groups of agents + rules.

    Group format: {"agents": [..], "rules": [(directive, path), ...]}
    where directive in {"allow","disallow"}
    """
    groups: List[Dict[str, List]] = []
    curr_agents: List[str] = []
    curr_rules: List[Tuple[str, str]] = []

    for raw_line in robots_text.splitlines():
        # Strip comments and whitespace
        line = raw_line.split("#", 1)[0].strip()

        # Blank line: close current group if non-empty
        if not line:
            if curr_agents or curr_rules:
                groups.append({"agents": curr_agents[:], "rules": curr_rules[:]})
                curr_agents = []
                curr_rules = []
            continue

        ua_m = re.match(r"(?i)^user-agent\s*:\s*(.+)$", line)
        if ua_m:
            # If rules already started, new UA starts a new group
            if curr_rules:
                groups.append({"agents": curr_agents[:], "rules": curr_rules[:]})
                curr_agents = []
                curr_rules = []
            for token in normalize_agent_token(ua_m.group(1)):
                curr_agents.append(token)
            continue

        rule_m = re.match(r"(?i)^(allow|disallow)\s*:\s*(.*)$", line)
        if rule_m and curr_agents:
            directive = rule_m.group(1).lower()
            path = rule_m.group(2).strip()
            curr_rules.append((directive, path))
            continue

    if curr_agents or curr_rules:
        groups.append({"agents": curr_agents[:], "rules": curr_rules[:]})

    return groups


def alias_to_canonical() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for canonical, meta in BOT_DEFS.items():
        for alias in meta["aliases"]:
            m[str(alias).lower()] = canonical
    return m


def applicable_rules(groups: List[Dict[str, List]], canonical_bot: str) -> Tuple[List[Tuple[str, str]], int, str]:
    """Get applicable rules for a bot.

    Returns: (rules, specific_rule, effective_source)
    - If specific UA exists, use only specific rules (even if empty).
    - Else fallback to wildcard rules.
    - Else no rules.
    """
    alias_map = alias_to_canonical()
    specific: List[Tuple[str, str]] = []
    specific_found = False

    for g in groups:
        agents = g["agents"]
        matched = False
        for a in agents:
            if alias_map.get(a.lower()) == canonical_bot:
                matched = True
                break
        if matched:
            specific_found = True
            specific.extend(g["rules"])

    if specific_found:
        return specific, 1, "specific"

    wildcard: List[Tuple[str, str]] = []
    for g in groups:
        if any(a.strip() == "*" for a in g["agents"]):
            wildcard.extend(g["rules"])

    if wildcard:
        return wildcard, 0, "wildcard"

    return [], 0, "none"


def classify_rules(rules: List[Tuple[str, str]], specific_rule: int, source: str) -> RuleStatus:
    root_block = 0
    path_block = 0

    for directive, path in rules:
        p = path.strip()
        if directive == "disallow" and p == "/":
            root_block = 1
        elif directive == "disallow" and p != "":
            path_block = 1

    return RuleStatus(
        specific_rule=specific_rule,
        effective_source=source,
        root_block=root_block,
        path_block=path_block,
    )


def fetch_robots(domain: str, timeout: int) -> FetchResult:
    url = f"https://{domain}/robots.txt"
    headers = {"User-Agent": "Mozilla/5.0 (Academic robots policy scanner)"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        text = resp.text if resp.status_code == 200 else ""
        return FetchResult(status_code=resp.status_code, final_url=resp.url, content=text, error="")
    except requests.exceptions.Timeout:
        return FetchResult(status_code=None, final_url=url, content="", error="timeout")
    except requests.exceptions.ConnectionError:
        return FetchResult(status_code=None, final_url=url, content="", error="connection_error")
    except requests.exceptions.RequestException as e:
        return FetchResult(status_code=None, final_url=url, content="", error=f"request_error:{e}")


def read_input(path: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        headers = {h.lower(): h for h in (rdr.fieldnames or [])}

        if "ticker" not in headers or "domain" not in headers:
            raise ValueError("Input CSV must include columns: ticker, domain")

        firm_col = headers.get("firm") or headers.get("company") or headers.get("name")
        ticker_col = headers["ticker"]
        domain_col = headers["domain"]

        for row in rdr:
            ticker = (row.get(ticker_col) or "").strip()
            domain = (row.get(domain_col) or "").strip().lower()
            firm = (row.get(firm_col) or ticker) if firm_col else ticker
            firm = firm.strip()
            if not ticker or not domain:
                continue
            out.append({"firm": firm, "ticker": ticker, "domain": domain})
    return out


def write_outputs(panel_path: str, summary_path: str, panel_rows: List[Dict[str, object]]) -> None:
    panel_cols = [
        "scan_utc",
        "scan_date",
        "firm",
        "ticker",
        "domain",
        "bot",
        "bot_group",
        "root_block",
        "path_block",
        "googlebot_block",
        "status_code",
        "fetch_error",
        "specific_rule",
        "effective_source",
        "retrieval_block_root",
        "train_block_root",
        "robots_final_url",
    ]

    with open(panel_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=panel_cols)
        w.writeheader()
        for r in panel_rows:
            w.writerow({k: r.get(k, "") for k in panel_cols})

    # Domain-level summary derived from panel
    key_order: List[Tuple[str, str, str, str, int | None]] = []
    seen = set()
    for r in panel_rows:
        k = (str(r["scan_date"]), str(r["firm"]), str(r["ticker"]), str(r["domain"]), r.get("status_code"))
        if k not in seen:
            seen.add(k)
            key_order.append(k)

    summary_cols = [
        "scan_date",
        "firm",
        "ticker",
        "domain",
        "status_code",
        "googlebot_block",
        "retrieval_block_root",
        "train_block_root",
        "n_bots_root_blocked",
        "n_bots_path_blocked",
    ]

    by_key: Dict[Tuple[str, str, str, str, int | None], List[Dict[str, object]]] = {}
    for r in panel_rows:
        k = (str(r["scan_date"]), str(r["firm"]), str(r["ticker"]), str(r["domain"]), r.get("status_code"))
        by_key.setdefault(k, []).append(r)

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        for k in key_order:
            rows = by_key[k]
            any_row = rows[0]
            n_root = sum(int(x.get("root_block", 0)) for x in rows)
            n_path = sum(int(x.get("path_block", 0)) for x in rows)
            w.writerow(
                {
                    "scan_date": k[0],
                    "firm": k[1],
                    "ticker": k[2],
                    "domain": k[3],
                    "status_code": k[4],
                    "googlebot_block": any_row.get("googlebot_block", 0),
                    "retrieval_block_root": any_row.get("retrieval_block_root", 0),
                    "train_block_root": any_row.get("train_block_root", 0),
                    "n_bots_root_blocked": n_root,
                    "n_bots_path_blocked": n_path,
                }
            )


def build_panel(input_rows: List[Dict[str, str]], timeout: int, sleep_sec: float) -> List[Dict[str, object]]:
    utc_now = dt.datetime.utcnow()
    scan_utc = utc_now.replace(microsecond=0).isoformat() + "Z"
    scan_date = utc_now.date().isoformat()

    panel_rows: List[Dict[str, object]] = []

    print(f"Scanning {len(input_rows)} domains...")
    for i, item in enumerate(input_rows, start=1):
        firm, ticker, domain = item["firm"], item["ticker"], item["domain"]
        print(f"[{i}/{len(input_rows)}] {ticker} {domain} ...", end=" ", flush=True)

        fetch = fetch_robots(domain=domain, timeout=timeout)

        # Default empty states for failures/non-200
        per_bot: Dict[str, RuleStatus] = {}
        if fetch.status_code == 200:
            groups = parse_robots(fetch.content)
            for canonical in BOT_DEFS:
                rules, specific, source = applicable_rules(groups, canonical)
                per_bot[canonical] = classify_rules(rules, specific, source)
            print("200")
        else:
            print(fetch.status_code if fetch.status_code is not None else fetch.error)
            for canonical in BOT_DEFS:
                per_bot[canonical] = RuleStatus(
                    specific_rule=0,
                    effective_source="none",
                    root_block=0,
                    path_block=0,
                )

        googlebot_block = per_bot["Googlebot"].root_block
        retrieval_block_root = int(any(per_bot[b].root_block for b in RETRIEVAL_BOTS))
        train_block_root = int(any(per_bot[b].root_block for b in TRAIN_BOTS))

        for bot, meta in BOT_DEFS.items():
            status = per_bot[bot]
            row = {
                "scan_utc": scan_utc,
                "scan_date": scan_date,
                "firm": firm,
                "ticker": ticker,
                "domain": domain,
                "bot": bot,
                "bot_group": meta["group"],
                "root_block": status.root_block,
                "path_block": status.path_block,
                "googlebot_block": googlebot_block,
                "status_code": fetch.status_code,
                "fetch_error": fetch.error,
                "specific_rule": status.specific_rule,
                "effective_source": status.effective_source,
                "retrieval_block_root": retrieval_block_root,
                "train_block_root": train_block_root,
                "robots_final_url": fetch.final_url,
            }
            panel_rows.append(row)

        time.sleep(sleep_sec)

    return panel_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build robots bot-level panel from ticker-domain CSV")
    p.add_argument("--input", required=True, help="Input CSV with at least ticker,domain")
    p.add_argument("--output", default="robots_bot_panel.csv", help="Output bot-level panel CSV")
    p.add_argument("--summary-output", default="robots_domain_summary.csv", help="Output domain summary CSV")
    p.add_argument("--timeout", type=int, default=15, help="HTTP timeout seconds")
    p.add_argument("--sleep", type=float, default=0.3, help="Sleep between domain requests")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_input(args.input)
    panel = build_panel(rows, timeout=args.timeout, sleep_sec=args.sleep)
    write_outputs(args.output, args.summary_output, panel)
    print(f"Saved panel: {args.output}")
    print(f"Saved summary: {args.summary_output}")


if __name__ == "__main__":
    main()
