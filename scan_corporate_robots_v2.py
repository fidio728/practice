"""
Corporate Domain robots.txt Scanner v2
- Strict RFC 9309 parsing: User-agent: * rules apply to unlisted bots
- Distinguishes root_block (Disallow: /) from path_block (Disallow: /path)
- Adds Googlebot check
- Output: firm-level CSV with RetrievalBlockRoot, TrainBlockRoot, GooglebotBlock
- Multiple User-Agent strings to handle WAF/CDN 403s
"""

import requests
import csv
import time
import re
from datetime import datetime
from collections import defaultdict

# ============================================================
# Bot definitions
# ============================================================
RETRIEVAL_BOTS = ["OAI-SearchBot", "Claude-SearchBot", "ChatGPT-User"]
TRAINING_BOTS = ["GPTBot", "ClaudeBot", "Google-Extended", "CCBot"]
CONTROL_BOTS = ["Googlebot"]
ALL_BOTS = RETRIEVAL_BOTS + TRAINING_BOTS + CONTROL_BOTS

# ============================================================
# Company list
# ============================================================
COMPANIES = [
    # MEGA-CAP (>$500B)
    ("Apple", "AAPL", "apple.com", "mega", "Tech"),
    ("Microsoft", "MSFT", "microsoft.com", "mega", "Tech"),
    ("NVIDIA", "NVDA", "nvidianews.nvidia.com", "mega", "Tech"),
    ("Amazon", "AMZN", "aboutamazon.com", "mega", "Tech"),
    ("Alphabet", "GOOGL", "blog.google", "mega", "Tech"),
    ("Meta", "META", "about.meta.com", "mega", "Tech"),
    ("Tesla", "TSLA", "ir.tesla.com", "mega", "Auto"),
    ("Berkshire Hathaway", "BRK.B", "berkshirehathaway.com", "mega", "Finance"),
    ("Broadcom", "AVGO", "broadcom.com", "mega", "Tech"),
    ("JPMorgan", "JPM", "jpmorganchase.com", "mega", "Finance"),

    # LARGE-CAP ($100B-$500B)
    ("Walmart", "WMT", "corporate.walmart.com", "large", "Retail"),
    ("Visa", "V", "visa.com", "large", "Finance"),
    ("Johnson & Johnson", "JNJ", "jnj.com", "large", "Healthcare"),
    ("Procter & Gamble", "PG", "us.pg.com", "large", "Consumer"),
    ("Mastercard", "MA", "mastercard.com", "large", "Finance"),
    ("ExxonMobil", "XOM", "corporate.exxonmobil.com", "large", "Energy"),
    ("Costco", "COST", "costco.com", "large", "Retail"),
    ("Pfizer", "PFE", "pfizer.com", "large", "Healthcare"),
    ("Goldman Sachs", "GS", "goldmansachs.com", "large", "Finance"),
    ("Netflix", "NFLX", "about.netflix.com", "large", "Media"),
    ("Adobe", "ADBE", "adobe.com", "large", "Tech"),
    ("Salesforce", "CRM", "salesforce.com", "large", "Tech"),
    ("Chevron", "CVX", "chevron.com", "large", "Energy"),
    ("Merck", "MRK", "merck.com", "large", "Healthcare"),
    ("Bank of America", "BAC", "newsroom.bankofamerica.com", "large", "Finance"),

    # MID-CAP ($10B-$100B)
    ("Dominos Pizza", "DPZ", "dominos.com", "mid", "Consumer"),
    ("Zillow", "Z", "zillow.com", "mid", "Tech"),
    ("HubSpot", "HUBS", "hubspot.com", "mid", "Tech"),
    ("Etsy", "ETSY", "etsy.com", "mid", "Retail"),
    ("Roku", "ROKU", "roku.com", "mid", "Tech"),
    ("Carvana", "CVNA", "carvana.com", "mid", "Auto"),
    ("Cloudflare", "NET", "cloudflare.com", "mid", "Tech"),
    ("Ralph Lauren", "RL", "ralphlauren.com", "mid", "Consumer"),
    ("Hershey", "HSY", "thehersheycompany.com", "mid", "Consumer"),
    ("Williams-Sonoma", "WSM", "williams-sonomainc.com", "mid", "Retail"),
    ("Cleveland-Cliffs", "CLF", "clevelandcliffs.com", "mid", "Materials"),
    ("Textron", "TXT", "textron.com", "mid", "Industrial"),
    ("Leidos", "LDOS", "leidos.com", "mid", "Defense"),
    ("Expedia", "EXPE", "expediagroup.com", "mid", "Travel"),
    ("Tractor Supply", "TSCO", "tractorsupply.com", "mid", "Retail"),

    # SMALL-CAP ($1B-$10B)
    ("Chewy", "CHWY", "chewy.com", "small", "Retail"),
    ("Shutterstock", "SSTK", "shutterstock.com", "small", "Media"),
    ("Coursera", "COUR", "coursera.org", "small", "Education"),
    ("Lemonade", "LMND", "lemonade.com", "small", "Insurance"),
    ("BigCommerce", "BIGC", "bigcommerce.com", "small", "Tech"),
    ("Weave Communications", "WEAV", "getweave.com", "small", "Tech"),
    ("Couchbase", "BASE", "couchbase.com", "small", "Tech"),
    ("Dine Brands", "DIN", "dinebrands.com", "small", "Consumer"),
    ("Quaker Chemical", "KWR", "quakerhoughton.com", "small", "Materials"),
    ("Donnelley Financial", "DFIN", "dfinsolutions.com", "small", "Finance"),
    ("ScanSource", "SCSC", "scansource.com", "small", "Tech"),
    ("Xperi", "XPER", "xperi.com", "small", "Tech"),
    ("Duolingo", "DUOL", "duolingo.com", "small", "Education"),
    ("BioAtla", "BCAB", "bioatla.com", "small", "Healthcare"),
    ("Veritone", "VERI", "veritone.com", "small", "Tech"),
]

# User-Agent rotation to handle WAF/CDN 403s
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (compatible; AcademicResearchBot/1.0)",
    "Python-requests/2.31.0",
]


def fetch_robots_txt(domain, timeout=15):
    """Fetch robots.txt, trying multiple User-Agents if 403."""
    url = f"https://{domain}/robots.txt"

    for ua in USER_AGENTS:
        headers = {"User-Agent": ua}
        try:
            resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if resp.status_code == 200:
                return {
                    "status": 200,
                    "content": resp.text,
                    "url": resp.url,
                    "error": None,
                    "ua_used": ua
                }
            elif resp.status_code == 403:
                continue  # try next UA
            else:
                return {
                    "status": resp.status_code,
                    "content": resp.text if resp.status_code < 400 else "",
                    "url": resp.url,
                    "error": None,
                    "ua_used": ua
                }
        except requests.exceptions.Timeout:
            continue
        except requests.exceptions.ConnectionError:
            return {"status": None, "content": "", "url": url, "error": "connection_error", "ua_used": ua}
        except Exception as e:
            return {"status": None, "content": "", "url": url, "error": str(e), "ua_used": ua}

    # All UAs returned 403 or timed out
    return {"status": 403, "content": "", "url": url, "error": "all_ua_403", "ua_used": "all_failed"}


def parse_robots_txt(content):
    """
    Strict RFC 9309 parser.
    Returns dict: {user_agent_lowercase: [(directive, path), ...]}
    Groups consecutive User-agent lines.
    """
    rules = {}
    current_agents = []

    for line in content.split("\n"):
        line = line.split("#")[0].strip()  # remove comments
        if not line:
            continue

        ua_match = re.match(r"^[Uu]ser-[Aa]gent:\s*(.+)$", line)
        if ua_match:
            agent = ua_match.group(1).strip()
            # If previous agents had no rules yet, they're part of same group
            if current_agents and all(a.lower() not in rules or not rules[a.lower()] for a in current_agents):
                current_agents.append(agent)
            else:
                current_agents = [agent]
            for a in current_agents:
                if a.lower() not in rules:
                    rules[a.lower()] = []
            continue

        rule_match = re.match(r"^(Allow|Disallow):\s*(.*)$", line, re.IGNORECASE)
        if rule_match and current_agents:
            directive = rule_match.group(1).lower()
            path = rule_match.group(2).strip()
            for a in current_agents:
                rules[a.lower()].append((directive, path))

    return rules


def get_bot_status(rules, bot_name):
    """
    RFC 9309 compliant: if bot has specific rules, use those.
    Otherwise, fall back to User-agent: * rules.
    Returns: (root_block, path_block, has_specific_rules, effective_rules)
    - root_block: True if Disallow: / (full site block)
    - path_block: True if Disallow on specific paths (not root)
    - has_specific_rules: True if bot is explicitly listed
    """
    bot_lower = bot_name.lower()

    # Check for specific rules
    if bot_lower in rules and rules[bot_lower]:
        effective = rules[bot_lower]
        specific = True
    elif "*" in rules:
        effective = rules["*"]
        specific = False
    else:
        # No rules at all = fully open
        return False, False, False, []

    root_block = False
    path_block = False

    for directive, path in effective:
        if directive == "disallow" and path == "/":
            root_block = True
        elif directive == "disallow" and path and path != "/":
            path_block = True
        elif directive == "allow" and path == "/":
            # Explicit allow overrides disallow in some interpretations
            # But per RFC 9309, most specific rule wins
            pass

    # Check for explicit Allow: / that might override Disallow: /
    # RFC 9309: longer path wins; if same length, Allow wins
    has_root_allow = any(d == "allow" and p == "/" for d, p in effective)
    if root_block and has_root_allow:
        # Same length paths, Allow wins per RFC 9309
        root_block = False

    return root_block, path_block, specific, effective


def main():
    scan_date = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"{'=' * 80}")
    print(f"Corporate Domain robots.txt AI Bot Scanner v2 (RFC 9309 strict)")
    print(f"Scan date: {timestamp}")
    print(f"Companies: {len(COMPANIES)}")
    print(f"{'=' * 80}\n")

    # CSV output: one row per firm x bot
    bot_csv_path = "corporate_robots_v2_botlevel.csv"
    firm_csv_path = "corporate_robots_v2_firmlevel.csv"

    bot_rows = []
    firm_rows = []

    for i, (name, ticker, domain, cap_group, sector) in enumerate(COMPANIES):
        print(f"[{i+1}/{len(COMPANIES)}] {name} ({ticker}) -- {domain} ...", end=" ", flush=True)

        fetch = fetch_robots_txt(domain)

        if fetch["error"] and fetch["error"] != "all_ua_403":
            print(f"ERROR: {fetch['error']}")
            status_code = fetch["status"] or "error"
            for bot in ALL_BOTS:
                bot_rows.append({
                    "firm": name, "ticker": ticker, "domain": domain,
                    "date": scan_date, "cap_group": cap_group, "sector": sector,
                    "bot": bot, "bot_type": "retrieval" if bot in RETRIEVAL_BOTS else ("training" if bot in TRAINING_BOTS else "control"),
                    "root_block": "NA", "path_block": "NA",
                    "has_specific_rule": "NA", "status_code": status_code
                })
            firm_rows.append({
                "firm": name, "ticker": ticker, "domain": domain,
                "date": scan_date, "cap_group": cap_group, "sector": sector,
                "status_code": status_code,
                "RetrievalBlockRoot": "NA", "TrainBlockRoot": "NA",
                "GooglebotBlock": "NA",
                "any_ai_rule": "NA", "notes": fetch["error"]
            })
            time.sleep(0.5)
            continue

        if fetch["status"] == 403:
            print(f"403 (all UAs blocked)")
            status_code = 403
            for bot in ALL_BOTS:
                bot_rows.append({
                    "firm": name, "ticker": ticker, "domain": domain,
                    "date": scan_date, "cap_group": cap_group, "sector": sector,
                    "bot": bot, "bot_type": "retrieval" if bot in RETRIEVAL_BOTS else ("training" if bot in TRAINING_BOTS else "control"),
                    "root_block": "NA", "path_block": "NA",
                    "has_specific_rule": "NA", "status_code": 403
                })
            firm_rows.append({
                "firm": name, "ticker": ticker, "domain": domain,
                "date": scan_date, "cap_group": cap_group, "sector": sector,
                "status_code": 403,
                "RetrievalBlockRoot": "NA", "TrainBlockRoot": "NA",
                "GooglebotBlock": "NA",
                "any_ai_rule": "NA", "notes": "robots.txt returns 403 for all User-Agents tried"
            })
            time.sleep(0.5)
            continue

        if fetch["status"] == 404:
            print(f"404 (no robots.txt -> all bots allowed)")
            status_code = 404
            for bot in ALL_BOTS:
                bot_rows.append({
                    "firm": name, "ticker": ticker, "domain": domain,
                    "date": scan_date, "cap_group": cap_group, "sector": sector,
                    "bot": bot, "bot_type": "retrieval" if bot in RETRIEVAL_BOTS else ("training" if bot in TRAINING_BOTS else "control"),
                    "root_block": 0, "path_block": 0,
                    "has_specific_rule": 0, "status_code": 404
                })
            firm_rows.append({
                "firm": name, "ticker": ticker, "domain": domain,
                "date": scan_date, "cap_group": cap_group, "sector": sector,
                "status_code": 404,
                "RetrievalBlockRoot": 0, "TrainBlockRoot": 0,
                "GooglebotBlock": 0,
                "any_ai_rule": 0, "notes": "no robots.txt"
            })
            time.sleep(0.5)
            continue

        # Parse robots.txt
        rules = parse_robots_txt(fetch["content"])
        status_code = 200

        any_ai_rule = 0
        retrieval_root_blocks = 0
        training_root_blocks = 0
        googlebot_root_block = 0

        for bot in ALL_BOTS:
            root_block, path_block, has_specific, _ = get_bot_status(rules, bot)

            if has_specific:
                any_ai_rule = 1

            bot_type = "retrieval" if bot in RETRIEVAL_BOTS else ("training" if bot in TRAINING_BOTS else "control")
            bot_rows.append({
                "firm": name, "ticker": ticker, "domain": domain,
                "date": scan_date, "cap_group": cap_group, "sector": sector,
                "bot": bot, "bot_type": bot_type,
                "root_block": int(root_block), "path_block": int(path_block),
                "has_specific_rule": int(has_specific), "status_code": 200
            })

            if bot in RETRIEVAL_BOTS and root_block:
                retrieval_root_blocks += 1
            if bot in TRAINING_BOTS and root_block:
                training_root_blocks += 1
            if bot == "Googlebot" and root_block:
                googlebot_root_block = 1

        # Firm-level: RetrievalBlockRoot = 1 if ANY retrieval bot is root-blocked
        retrieval_block_root = 1 if retrieval_root_blocks > 0 else 0
        train_block_root = 1 if training_root_blocks > 0 else 0

        # Determine notes
        notes = ""
        if retrieval_block_root and train_block_root:
            notes = "blocks both retrieval and training"
        elif retrieval_block_root:
            notes = "blocks retrieval only"
        elif train_block_root:
            notes = "blocks training only"
        elif any_ai_rule:
            notes = "has AI rules but no root blocks (path-only or allow)"

        classification = "OPEN"
        if retrieval_block_root and train_block_root:
            classification = "BLOCK_ALL"
        elif retrieval_block_root:
            classification = "BLOCK_RETRIEVAL"
        elif train_block_root:
            classification = "BLOCK_TRAINING"
        elif any_ai_rule:
            classification = "HAS_RULES"
        else:
            classification = "OPEN"

        print(f"OK -> {classification}" + (f" ({notes})" if notes else ""))

        firm_rows.append({
            "firm": name, "ticker": ticker, "domain": domain,
            "date": scan_date, "cap_group": cap_group, "sector": sector,
            "status_code": 200,
            "RetrievalBlockRoot": retrieval_block_root,
            "TrainBlockRoot": train_block_root,
            "GooglebotBlock": googlebot_root_block,
            "any_ai_rule": any_ai_rule,
            "notes": notes
        })

        time.sleep(0.5)

    # ============================================================
    # Write bot-level CSV
    # ============================================================
    with open(bot_csv_path, "w", newline="", encoding="utf-8") as f:
        cols = ["firm", "ticker", "domain", "date", "cap_group", "sector",
                "bot", "bot_type", "root_block", "path_block",
                "has_specific_rule", "status_code"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(bot_rows)
    print(f"\nBot-level CSV: {bot_csv_path} ({len(bot_rows)} rows)")

    # ============================================================
    # Write firm-level CSV
    # ============================================================
    with open(firm_csv_path, "w", newline="", encoding="utf-8") as f:
        cols = ["firm", "ticker", "domain", "date", "cap_group", "sector",
                "status_code", "RetrievalBlockRoot", "TrainBlockRoot",
                "GooglebotBlock", "any_ai_rule", "notes"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(firm_rows)
    print(f"Firm-level CSV: {firm_csv_path} ({len(firm_rows)} rows)")

    # ============================================================
    # Summary statistics
    # ============================================================
    # Exclude firms where we couldn't get robots.txt
    valid = [r for r in firm_rows if r["status_code"] in (200, 404)]
    na_firms = [r for r in firm_rows if r["RetrievalBlockRoot"] == "NA"]

    print(f"\n{'=' * 80}")
    print(f"PREVALENCE SUMMARY (valid firms only, n={len(valid)})")
    print(f"Excluded: {len(na_firms)} firms (403/error: {', '.join(r['ticker'] for r in na_firms)})")
    print(f"{'=' * 80}")

    n = len(valid)
    ret_block = sum(1 for r in valid if r["RetrievalBlockRoot"] == 1)
    train_block = sum(1 for r in valid if r["TrainBlockRoot"] == 1)
    google_block = sum(1 for r in valid if r["GooglebotBlock"] == 1)
    any_rule = sum(1 for r in valid if r["any_ai_rule"] == 1)

    print(f"\n  RetrievalBlockRoot = 1:  {ret_block:3d} / {n} ({ret_block/n*100:5.1f}%)")
    print(f"  TrainBlockRoot = 1:     {train_block:3d} / {n} ({train_block/n*100:5.1f}%)")
    print(f"  GooglebotBlock = 1:     {google_block:3d} / {n} ({google_block/n*100:5.1f}%)")
    print(f"  Any AI-specific rule:   {any_rule:3d} / {n} ({any_rule/n*100:5.1f}%)")

    # By market cap
    print(f"\n{'=' * 80}")
    print(f"PREVALENCE BY MARKET CAP GROUP")
    print(f"{'=' * 80}")
    print(f"\n  {'Group':<8} {'N':>4} {'N_valid':>7} {'RetBlock':>10} {'TrainBlock':>12} {'GoogleBlock':>12} {'AnyRule':>10}")
    print(f"  {'-'*8} {'-'*4} {'-'*7} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    for cap in ["mega", "large", "mid", "small"]:
        cap_all = [r for r in firm_rows if r["cap_group"] == cap]
        cap_valid = [r for r in valid if r["cap_group"] == cap]
        nv = len(cap_valid)
        if nv == 0:
            continue
        rb = sum(1 for r in cap_valid if r["RetrievalBlockRoot"] == 1)
        tb = sum(1 for r in cap_valid if r["TrainBlockRoot"] == 1)
        gb = sum(1 for r in cap_valid if r["GooglebotBlock"] == 1)
        ar = sum(1 for r in cap_valid if r["any_ai_rule"] == 1)
        print(f"  {cap:<8} {len(cap_all):>4} {nv:>7} {rb:>4} ({rb/nv*100:4.1f}%) {tb:>5} ({tb/nv*100:5.1f}%) {gb:>5} ({gb/nv*100:5.1f}%) {ar:>4} ({ar/nv*100:4.1f}%)")

    # List all blocking firms
    print(f"\n{'=' * 80}")
    print(f"FIRMS WITH ROOT BLOCKS (RetrievalBlockRoot=1 or TrainBlockRoot=1)")
    print(f"{'=' * 80}")
    blockers = [r for r in valid if r["RetrievalBlockRoot"] == 1 or r["TrainBlockRoot"] == 1]
    if not blockers:
        print("  NONE")
    else:
        for r in blockers:
            bot_details = [b for b in bot_rows
                          if b["ticker"] == r["ticker"] and b["root_block"] == 1]
            blocked_bots = ", ".join(f"{b['bot']}({b['bot_type']})" for b in bot_details)
            print(f"  {r['firm']:25s} ({r['ticker']:6s}) {r['cap_group']:6s} {r['sector']:12s} | {r['domain']}")
            print(f"    RetrievalBlockRoot={r['RetrievalBlockRoot']}  TrainBlockRoot={r['TrainBlockRoot']}  GooglebotBlock={r['GooglebotBlock']}")
            print(f"    Root-blocked bots: {blocked_bots}")
            print()

    # List firms with path-only blocks
    print(f"{'=' * 80}")
    print(f"FIRMS WITH PATH-ONLY BLOCKS (no root block, but path Disallow)")
    print(f"{'=' * 80}")
    path_only = [r for r in valid
                 if r["RetrievalBlockRoot"] == 0 and r["TrainBlockRoot"] == 0
                 and r["any_ai_rule"] == 1]
    if not path_only:
        print("  NONE")
    else:
        for r in path_only:
            bot_details = [b for b in bot_rows
                          if b["ticker"] == r["ticker"] and b["path_block"] == 1]
            path_bots = ", ".join(f"{b['bot']}({b['bot_type']})" for b in bot_details)
            print(f"  {r['firm']:25s} ({r['ticker']:6s}) | Path-blocked bots: {path_bots}")


if __name__ == "__main__":
    main()
