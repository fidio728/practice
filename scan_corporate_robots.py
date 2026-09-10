"""
Corporate Domain robots.txt Scanner for AI Bot Policies
Scans corporate/IR domains (NOT retail) for AI crawler blocking rules.
Stratified by market cap quintile.
"""

import requests
import csv
import time
import re
from datetime import datetime
from collections import defaultdict

# AI bots to check
AI_BOTS = {
    # Training bots
    "GPTBot": "training",
    "ClaudeBot": "training",
    "Google-Extended": "training",
    "CCBot": "training",
    "Amazonbot": "training",
    "PerplexityBot": "training",
    # Search/retrieval bots
    "OAI-SearchBot": "search",
    "Claude-SearchBot": "search",
    "ChatGPT-User": "search",
    "Perplexity-User": "search",
    "Applebot-Extended": "training",
}

# Company list: (name, ticker, corporate_domain, market_cap_group, sector)
# Corporate domain = the site hosting press releases, IR, ESG, strategy info
# NOT the retail/product site
COMPANIES = [
    # === MEGA-CAP (>$500B) ===
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

    # === LARGE-CAP ($100B-$500B) ===
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

    # === MID-CAP ($10B-$100B) ===
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

    # === SMALL-CAP ($1B-$10B) ===
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


def fetch_robots_txt(domain, timeout=15):
    """Fetch robots.txt from a domain. Try https first, then http."""
    url = f"https://{domain}/robots.txt"
    headers = {
        "User-Agent": "Mozilla/5.0 (academic research bot scanner)"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        return {
            "status": resp.status_code,
            "content": resp.text if resp.status_code == 200 else "",
            "url": resp.url,
            "error": None
        }
    except requests.exceptions.Timeout:
        return {"status": None, "content": "", "url": url, "error": "timeout"}
    except requests.exceptions.ConnectionError:
        return {"status": None, "content": "", "url": url, "error": "connection_error"}
    except Exception as e:
        return {"status": None, "content": "", "url": url, "error": str(e)}


def parse_robots_txt(content):
    """Parse robots.txt and extract per-user-agent rules."""
    rules = defaultdict(list)
    current_agents = []

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Match user-agent line
        ua_match = re.match(r"^[Uu]ser-[Aa]gent:\s*(.+)$", line)
        if ua_match:
            agent = ua_match.group(1).strip()
            # If we were building rules for previous agents, finalize them
            if not current_agents or (current_agents and rules[current_agents[-1]]):
                current_agents = [agent]
            else:
                current_agents.append(agent)
            if agent not in rules:
                rules[agent] = []
            continue

        # Match allow/disallow
        rule_match = re.match(r"^(Allow|Disallow):\s*(.*)$", line, re.IGNORECASE)
        if rule_match and current_agents:
            directive = rule_match.group(1).capitalize()
            path = rule_match.group(2).strip()
            for agent in current_agents:
                rules[agent].append((directive, path))

    return dict(rules)


def classify_bot_status(rules, bot_name):
    """Classify a specific bot's status based on parsed rules."""
    # Check if bot has specific rules
    # Case-insensitive matching
    bot_rules = None
    for agent_name, agent_rules in rules.items():
        if agent_name.lower() == bot_name.lower():
            bot_rules = agent_rules
            break

    if bot_rules is not None:
        # Bot has specific rules
        for directive, path in bot_rules:
            if directive == "Disallow" and path == "/":
                return "BLOCK"
            if directive == "Disallow" and path and path != "/":
                return "PARTIAL"
            if directive == "Allow" and path == "/":
                return "ALLOW"
            if directive == "Allow" and path:
                return "RESTRICTED"
        # Has rules but no clear block/allow
        if bot_rules:
            return "LISTED"
        return "LISTED_EMPTY"

    # Bot not specifically listed -- falls under wildcard
    wildcard_rules = rules.get("*", [])
    for directive, path in wildcard_rules:
        if directive == "Disallow" and path == "/":
            return "BLOCK_VIA_WILDCARD"
    return "NOT_MENTIONED"


def classify_firm(bot_statuses):
    """Classify overall firm status based on all bot statuses."""
    search_bots = ["OAI-SearchBot", "Claude-SearchBot", "ChatGPT-User"]
    training_bots = ["GPTBot", "ClaudeBot", "Google-Extended", "CCBot"]

    search_blocked = sum(1 for b in search_bots if bot_statuses.get(b, "NOT_MENTIONED") in ("BLOCK", "BLOCK_VIA_WILDCARD"))
    training_blocked = sum(1 for b in training_bots if bot_statuses.get(b, "NOT_MENTIONED") in ("BLOCK", "BLOCK_VIA_WILDCARD"))
    search_partial = sum(1 for b in search_bots if bot_statuses.get(b, "NOT_MENTIONED") == "PARTIAL")
    training_partial = sum(1 for b in training_bots if bot_statuses.get(b, "NOT_MENTIONED") == "PARTIAL")

    any_explicit_allow = any(bot_statuses.get(b) == "ALLOW" for b in search_bots + training_bots)

    if search_blocked >= 2 and training_blocked >= 2:
        return "BLOCK_ALL"
    elif search_blocked >= 1:
        return "BLOCK_SEARCH"
    elif training_blocked >= 2:
        return "BLOCK_TRAINING"
    elif training_blocked >= 1 or search_partial >= 1 or training_partial >= 1:
        return "PARTIAL"
    elif any_explicit_allow:
        return "EXPLICIT_ALLOW"
    else:
        return "OPEN_DEFAULT"


def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"=" * 80)
    print(f"Corporate Domain robots.txt AI Bot Scanner")
    print(f"Scan date: {timestamp}")
    print(f"Companies: {len(COMPANIES)}")
    print(f"AI bots checked: {len(AI_BOTS)}")
    print(f"=" * 80)
    print()

    results = []

    for i, (name, ticker, domain, cap_group, sector) in enumerate(COMPANIES):
        print(f"[{i+1}/{len(COMPANIES)}] {name} ({ticker}) -- {domain} ...", end=" ", flush=True)

        fetch = fetch_robots_txt(domain)

        if fetch["error"]:
            print(f"ERROR: {fetch['error']}")
            bot_statuses = {bot: "FETCH_ERROR" for bot in AI_BOTS}
            classification = "FETCH_ERROR"
        elif fetch["status"] == 403:
            print(f"403 FORBIDDEN")
            bot_statuses = {bot: "SERVER_403" for bot in AI_BOTS}
            classification = "SERVER_BLOCK"
        elif fetch["status"] == 404:
            print(f"404 (no robots.txt = fully open)")
            bot_statuses = {bot: "NO_ROBOTS_TXT" for bot in AI_BOTS}
            classification = "NO_ROBOTS_TXT"
        elif fetch["status"] == 200:
            rules = parse_robots_txt(fetch["content"])
            bot_statuses = {}
            for bot in AI_BOTS:
                bot_statuses[bot] = classify_bot_status(rules, bot)
            classification = classify_firm(bot_statuses)
            print(f"OK -> {classification}")
        else:
            print(f"HTTP {fetch['status']}")
            bot_statuses = {bot: f"HTTP_{fetch['status']}" for bot in AI_BOTS}
            classification = f"HTTP_{fetch['status']}"

        results.append({
            "name": name,
            "ticker": ticker,
            "domain": domain,
            "cap_group": cap_group,
            "sector": sector,
            "http_status": fetch["status"],
            "error": fetch["error"],
            "classification": classification,
            "bot_statuses": bot_statuses,
        })

        time.sleep(0.5)  # polite delay

    # === OUTPUT: CSV ===
    csv_path = "corporate_robots_scan_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Company", "Ticker", "Corporate Domain", "Market Cap Group", "Sector",
                  "HTTP Status", "Classification"]
        for bot in AI_BOTS:
            header.append(f"{bot} ({AI_BOTS[bot]})")
        writer.writerow(header)

        for r in results:
            row = [r["name"], r["ticker"], r["domain"], r["cap_group"], r["sector"],
                   r["http_status"] or r["error"], r["classification"]]
            for bot in AI_BOTS:
                row.append(r["bot_statuses"].get(bot, "N/A"))
            writer.writerow(row)

    print(f"\nCSV saved: {csv_path}")

    # === OUTPUT: Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY BY CLASSIFICATION")
    print("=" * 80)

    from collections import Counter
    class_counts = Counter(r["classification"] for r in results)
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = count / len(results) * 100
        firms = [r["ticker"] for r in results if r["classification"] == cls]
        print(f"  {cls:20s}: {count:3d} ({pct:5.1f}%) -- {', '.join(firms)}")

    # By market cap group
    print("\n" + "=" * 80)
    print("SUMMARY BY MARKET CAP GROUP")
    print("=" * 80)
    for cap in ["mega", "large", "mid", "small"]:
        cap_results = [r for r in results if r["cap_group"] == cap]
        if not cap_results:
            continue
        n = len(cap_results)
        blocked = sum(1 for r in cap_results if r["classification"] in
                      ("BLOCK_ALL", "BLOCK_SEARCH", "BLOCK_TRAINING", "SERVER_BLOCK"))
        partial = sum(1 for r in cap_results if r["classification"] == "PARTIAL")
        explicit_allow = sum(1 for r in cap_results if r["classification"] == "EXPLICIT_ALLOW")
        open_default = sum(1 for r in cap_results if r["classification"] in ("OPEN_DEFAULT", "NO_ROBOTS_TXT"))
        errors = sum(1 for r in cap_results if "ERROR" in r["classification"])

        print(f"\n  {cap.upper()} (n={n}):")
        print(f"    Block (any):     {blocked:3d} ({blocked/n*100:5.1f}%)")
        print(f"    Partial:         {partial:3d} ({partial/n*100:5.1f}%)")
        print(f"    Explicit Allow:  {explicit_allow:3d} ({explicit_allow/n*100:5.1f}%)")
        print(f"    Open (default):  {open_default:3d} ({open_default/n*100:5.1f}%)")
        if errors:
            print(f"    Fetch Error:     {errors:3d} ({errors/n*100:5.1f}%)")

    # Detailed blocking info
    print("\n" + "=" * 80)
    print("FIRMS WITH ANY AI BOT BLOCKING ON CORPORATE DOMAIN")
    print("=" * 80)
    blocked_firms = [r for r in results if r["classification"] in
                     ("BLOCK_ALL", "BLOCK_SEARCH", "BLOCK_TRAINING", "PARTIAL", "SERVER_BLOCK")]
    if not blocked_firms:
        print("  NONE -- all corporate domains are open to AI bots")
    else:
        for r in blocked_firms:
            print(f"\n  {r['name']} ({r['ticker']}) -- {r['domain']} -- {r['classification']}")
            for bot, status in r["bot_statuses"].items():
                if status not in ("NOT_MENTIONED", "NO_ROBOTS_TXT"):
                    print(f"    {bot:20s}: {status}")


if __name__ == "__main__":
    main()
