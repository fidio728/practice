"""Parse the 2026-08 live robots.txt vintage and answer the go/no-go question.

The question
------------
The paper cannot separate AI crawlers from traditional ones because in
2021-2025 88% of restricting firms blocked everything through a bare
`User-agent: *` wildcard (paper Appendix A.3). Identification of an AI-specific
effect needs firms that restrict AI crawlers BY NAME while leaving the
incumbent search crawlers allowed. This script counts them, on the current
vintage, and compares against the 2026-03-27 scan on the overlapping hosts.

Reads the raw bodies written by `fetch_robots_live_2026_08.py`, so the crawler
taxonomy below can be revised and re-run at zero fetch cost. Where a host has
more than one record (from `--retry-errors`), the last one wins.

Parsing is delegated to `gpt answer/robots_parser.py` unchanged, so the
Disallow semantics (RFC 9309 longest-match, allow-on-tie) match the paper.

Usage
-----
    python parse_robots_live_2026_08.py
    python parse_robots_live_2026_08.py --outdir E:\\robots_live_2026_08
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_DIR = Path(r"c:\Users\xl\OneDrive - Universitat Ramón Llull\git\practice")
sys.path.insert(0, str(PROJECT_DIR / "gpt answer"))
from robots_parser import classify_bot, parse_robots  # noqa: E402

DEFAULT_OUTDIR = Path(r"E:\robots_live_2026_08")
TAXONOMY_PATH = Path(r"E:\robots_y_rerun_2026_06_10\data\bot_arm_taxonomy.csv")
MARCH_FIRM_SUMMARY = PROJECT_DIR / "gpt answer" / "robots_firm_summary.csv"

# --- The twelve crawlers of the 2026-03-27 scan, kept verbatim so the March
# --- vintage and this one are compared on an identical definition.
MARCH_TRAINING = ["GPTBot", "ClaudeBot", "Google-Extended", "CCBot", "Meta-ExternalAgent"]
MARCH_SEARCH = ["OAI-SearchBot", "Claude-SearchBot", "PerplexityBot"]
MARCH_USER = ["ChatGPT-User", "Perplexity-User", "Meta-ExternalFetcher"]
MARCH_AI = MARCH_TRAINING + MARCH_SEARCH + MARCH_USER

# --- AI training / dataset crawlers. Not in the paper's search taxonomy (the
# --- treatment there is search and discovery), but they are where AI-specific
# --- naming actually shows up, so they are what makes the variation visible.
AI_TRAINING = [
    "GPTBot", "ClaudeBot", "anthropic-ai", "CCBot", "Google-Extended",
    "Applebot-Extended", "Meta-ExternalAgent", "FacebookBot", "Bytespider",
    "Amazonbot", "cohere-ai", "Diffbot", "ImagesiftBot", "omgili", "omgilibot",
    "Timpibot", "Webzio-Extended", "AI2Bot", "PanguBot", "SemrushBot-OCOB",
    "VelenPublicWebCrawler", "img2dataset", "Kangaroo Bot", "TikTokSpider",
]
# --- On-demand fetchers triggered by a user prompt rather than by crawling.
AI_USER = ["ChatGPT-User", "Perplexity-User", "Claude-User", "Meta-ExternalFetcher"]
# --- The incumbents. "Still allowed?" is judged on these.
INCUMBENT_SEARCH = ["Googlebot", "bingbot"]

ALIASES = {"Sogou": ["Sogou", "Sogou web spider", "Sogou inst spider"]}


def load_taxonomy(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"taxonomy not found: {path}")
    return pd.read_csv(path)


def build_bot_config(taxonomy: pd.DataFrame) -> List[Dict[str, Any]]:
    """The paper's search crawlers plus the AI training/user crawlers."""
    config: List[Dict[str, Any]] = []
    seen = set()

    for row in taxonomy.itertuples(index=False):
        bot = str(row.bot)
        if bot in seen:
            continue
        seen.add(bot)
        config.append({"bot": bot, "arm": str(row.primary_arm), "family": "paper_search",
                       "aliases": ALIASES.get(bot, [bot])})

    for bot in AI_TRAINING:
        if bot in seen:
            continue
        seen.add(bot)
        config.append({"bot": bot, "arm": "ai_training", "family": "ai_training",
                       "aliases": ALIASES.get(bot, [bot])})

    for bot in AI_USER:
        if bot in seen:
            continue
        seen.add(bot)
        config.append({"bot": bot, "arm": "ai_user", "family": "ai_user",
                       "aliases": ALIASES.get(bot, [bot])})

    for bot in INCUMBENT_SEARCH + MARCH_AI:
        if bot in seen:
            continue
        seen.add(bot)
        config.append({"bot": bot, "arm": "other", "family": "other",
                       "aliases": ALIASES.get(bot, [bot])})

    return config


def load_last_per_host(raw_path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            host = row.get("host")
            if host:
                records[host] = row  # last wins
    return records


def classify_host(record: Dict[str, Any], config: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    status = record.get("robots_status_type")

    if status == "ok_200":
        groups = parse_robots(record.get("content"))
        return {c["bot"]: classify_bot(groups, c["bot"], c["aliases"]) for c in config}

    if status == "missing_404":
        return {
            c["bot"]: {"bot": c["bot"], "specific_rule": 0, "effective_source": "no_robots_file",
                       "root_block": 0, "has_nonroot_disallow": 0}
            for c in config
        }

    return {
        c["bot"]: {"bot": c["bot"], "specific_rule": None, "effective_source": "fetch_error",
                   "root_block": None, "has_nonroot_disallow": None}
        for c in config
    }


def _named_blocked(bots: Dict[str, Dict[str, Any]], names: List[str]) -> int:
    """Crawlers blocked at root through a rule that NAMES them, not a wildcard."""
    return sum(
        1 for b in names
        if b in bots and bots[b]["specific_rule"] == 1 and bots[b]["root_block"] == 1
    )


def _root_blocked(bots: Dict[str, Dict[str, Any]], names: List[str]) -> int:
    return sum(1 for b in names if b in bots and bots[b]["root_block"] == 1)


def summarize_host(host: str, record: Dict[str, Any], bots: Dict[str, Dict[str, Any]],
                   config: List[Dict[str, Any]]) -> Dict[str, Any]:
    status = record.get("robots_status_type")
    usable = status in {"ok_200", "missing_404"}

    ai_all = AI_TRAINING + AI_USER + [c["bot"] for c in config if c["arm"] in {"ai_strict", "ai_expanded"}]
    ai_all = list(dict.fromkeys(ai_all))

    out: Dict[str, Any] = {
        "host": host,
        "scan_date": record.get("scan_date"),
        "status_code": record.get("status_code"),
        "robots_status_type": status,
        "http_fallback": record.get("http_fallback"),
        "n_bytes": len(record.get("content") or "") if record.get("content") else 0,
    }

    if not usable:
        for key in ("n_ai_named_blocked", "n_ai_root_blocked", "n_incumbent_root_blocked",
                    "any_ai_specific_rule_march12", "googlebot_block", "bingbot_block",
                    "wildcard_root_block"):
            out[key] = None
        out["ai_specific_variation"] = None
        out["block_pattern"] = "fetch_error"
        return out

    out["n_ai_named_blocked"] = _named_blocked(bots, ai_all)
    out["n_ai_root_blocked"] = _root_blocked(bots, ai_all)
    out["n_incumbent_root_blocked"] = _root_blocked(bots, INCUMBENT_SEARCH)
    out["googlebot_block"] = bots["Googlebot"]["root_block"]
    out["bingbot_block"] = bots["bingbot"]["root_block"] if "bingbot" in bots else None
    # March-comparable measure: any of the twelve AI crawlers carries its own rule.
    out["any_ai_specific_rule_march12"] = int(
        any(bots[b]["specific_rule"] == 1 for b in MARCH_AI if b in bots)
    )
    # Was the root block reached through the wildcard group rather than a name?
    out["wildcard_root_block"] = int(
        any(bots[b]["root_block"] == 1 and bots[b]["effective_source"] == "wildcard"
            for b in ai_all if b in bots)
    )

    # THE measure this whole exercise exists for: AI crawlers named and blocked,
    # incumbents left alone. That is the differential the paper does not have.
    out["ai_specific_variation"] = int(
        out["n_ai_named_blocked"] > 0 and out["n_incumbent_root_blocked"] == 0
    )

    if out["n_ai_root_blocked"] == 0 and out["n_incumbent_root_blocked"] == 0:
        pattern = "allow_all"
    elif out["ai_specific_variation"] == 1:
        pattern = "ai_named_incumbent_allowed"
    elif out["n_incumbent_root_blocked"] > 0 and out["n_ai_root_blocked"] > 0:
        pattern = "blanket_block"
    elif out["n_incumbent_root_blocked"] > 0:
        pattern = "incumbent_blocked_ai_allowed"
    else:
        pattern = "ai_blocked_via_wildcard_only"
    out["block_pattern"] = pattern
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse the 2026-08 live robots.txt vintage.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--taxonomy", type=Path, default=TAXONOMY_PATH)
    args = parser.parse_args()

    raw_path = args.outdir / "raw_robots.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"no raw scan at {raw_path}")

    taxonomy = load_taxonomy(args.taxonomy)
    config = build_bot_config(taxonomy)
    records = load_last_per_host(raw_path)
    print(f"hosts with a record: {len(records)} | crawlers classified: {len(config)}")

    long_rows: List[Dict[str, Any]] = []
    host_rows: List[Dict[str, Any]] = []
    arm_of = {c["bot"]: c["arm"] for c in config}
    family_of = {c["bot"]: c["family"] for c in config}

    for host, record in records.items():
        bots = classify_host(record, config)
        host_rows.append(summarize_host(host, record, bots, config))
        for bot, res in bots.items():
            long_rows.append({
                "host": host, "bot": bot, "arm": arm_of[bot], "family": family_of[bot],
                "root_block": res["root_block"], "specific_rule": res["specific_rule"],
                "effective_source": res["effective_source"],
                "has_nonroot_disallow": res["has_nonroot_disallow"],
            })

    host_df = pd.DataFrame(host_rows)
    long_df = pd.DataFrame(long_rows)
    host_df.to_csv(args.outdir / "host_summary.csv", index=False)
    long_df.to_csv(args.outdir / "host_bot_long.csv", index=False)

    universe_path = args.outdir / "host_universe.csv"
    firm_df = None
    if universe_path.exists():
        universe = pd.read_csv(universe_path, dtype={"gvkey": str, "cik": str})
        firm_df = universe.merge(host_df, left_on="host_clean", right_on="host", how="left")
        firm_df.to_csv(args.outdir / "firm_summary.csv", index=False)

    lines: List[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    emit("=" * 66)
    emit("LIVE robots.txt VINTAGE 2026-08 - AI-specific variation probe")
    emit("=" * 66)
    emit()
    emit("Fetch outcomes")
    for name, count in host_df["robots_status_type"].value_counts().items():
        emit(f"  {name:<22} {count:>6}")
    usable = host_df.loc[host_df["robots_status_type"].isin(["ok_200", "missing_404"])].copy()
    emit(f"  {'USABLE':<22} {len(usable):>6} / {len(host_df)}")
    emit()

    emit("Block pattern (usable hosts)")
    for name, count in usable["block_pattern"].value_counts().items():
        emit(f"  {name:<32} {count:>6}  ({100*count/len(usable):5.1f}%)")
    emit()

    n_var = int(usable["ai_specific_variation"].sum())
    emit("-" * 66)
    emit(f"AI-SPECIFIC VARIATION: {n_var} hosts  ({100*n_var/len(usable):.1f}% of usable)")
    emit("  = at least one AI crawler blocked at root BY NAME,")
    emit("    with Googlebot and bingbot both still allowed.")
    emit("-" * 66)
    emit()

    named = long_df.loc[
        (long_df["specific_rule"] == 1) & (long_df["root_block"] == 1)
        & (long_df["family"].isin(["ai_training", "ai_user"])
           | long_df["arm"].isin(["ai_strict", "ai_expanded"]))
    ]
    emit("Most-named AI crawlers (own rule AND blocked at root)")
    for bot, count in named["bot"].value_counts().head(15).items():
        emit(f"  {bot:<26} {count:>6}")
    emit()

    if MARCH_FIRM_SUMMARY.exists():
        march = pd.read_csv(MARCH_FIRM_SUMMARY, dtype={"gvkey": str})
        march_host = march.drop_duplicates("host")[
            ["host", "robots_status_type", "any_ai_specific_rule", "googlebot_block"]
        ].rename(columns={
            "robots_status_type": "status_mar",
            "any_ai_specific_rule": "ai_named_mar",
            "googlebot_block": "googlebot_block_mar",
        })
        both = usable.merge(march_host, on="host", how="inner")
        both = both.loc[both["status_mar"].isin(["ok_200", "missing_404"])]
        if len(both):
            emit("-" * 66)
            emit(f"CHANGE 2026-03-27 -> 2026-08  (hosts usable in both: {len(both)})")
            emit("-" * 66)
            mar = both["ai_named_mar"].astype(float).mean()
            aug = both["any_ai_specific_rule_march12"].astype(float).mean()
            emit(f"  Names >=1 of the twelve AI crawlers   March {100*mar:5.1f}%  ->  Aug {100*aug:5.1f}%"
                 f"   ({100*(aug-mar):+.1f} pp)")
            gm = both["googlebot_block_mar"].astype(float).mean()
            ga = both["googlebot_block"].astype(float).mean()
            emit(f"  Googlebot blocked at root             March {100*gm:5.1f}%  ->  Aug {100*ga:5.1f}%"
                 f"   ({100*(ga-gm):+.1f} pp)")
            moved = both.loc[(both["ai_named_mar"] == 0) & (both["any_ai_specific_rule_march12"] == 1)]
            emit(f"  Hosts that ADDED AI-specific naming in five months: {len(moved)}")
            emit()

    if firm_df is not None:
        fu = firm_df.loc[firm_df["robots_status_type"].isin(["ok_200", "missing_404"])]
        n_firm_var = int(fu["ai_specific_variation"].sum())
        emit(f"Firm level: {fu['gvkey'].nunique()} gvkeys on usable hosts, "
             f"{n_firm_var} firm-rows with AI-specific variation")
        emit()

    emit("VERDICT INPUT: the paper's Appendix A.3 AI arm had 82 named firms and was")
    emit("underpowered. Compare the firm count above against that bar.")

    (args.outdir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten -> {args.outdir}")


if __name__ == "__main__":
    main()
