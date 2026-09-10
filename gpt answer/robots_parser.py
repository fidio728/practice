import re
from typing import Dict, Iterable, List, Sequence, Tuple


Rule = Tuple[str, str]
Group = Dict[str, List]

_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
_COMMENT_RE = re.compile(r"#.*$")
_PERCENT_RE = re.compile(r"%([0-9A-Fa-f]{2})")
_PATTERN_CACHE: Dict[str, re.Pattern[str]] = {}


def _strip_comment(line: str) -> str:
    return _COMMENT_RE.sub("", line).strip()


def parse_robots(content: str | None) -> List[Group]:
    """Parse robots.txt content into groups of agents and rules."""
    if not content:
        return []

    text = str(content).lstrip("\ufeff")
    groups: List[Group] = []
    current_agents: List[str] = []
    current_rules: List[Rule] = []

    def flush_group() -> None:
        nonlocal current_agents, current_rules
        if current_agents:
            groups.append({"agents": current_agents[:], "rules": current_rules[:]})
        current_agents = []
        current_rules = []

    for raw_line in text.splitlines():
        line = _strip_comment(raw_line)
        if not line:
            flush_group()
            continue

        if ":" not in line:
            continue

        field, value = line.split(":", 1)
        directive = field.strip().lower()
        value = value.strip()

        if directive == "user-agent":
            agent = value.lower()
            if current_agents and current_rules:
                flush_group()
            current_agents.append(agent)
            continue

        if directive in {"allow", "disallow"}:
            if not current_agents:
                continue
            current_rules.append((directive, value))
            continue

    flush_group()
    return groups


def get_applicable_rules(
    groups: Sequence[Group], bot_aliases: Iterable[str]
) -> Tuple[List[Rule], bool, str]:
    """Merge specific or wildcard rules for the supplied bot aliases."""
    aliases = {alias.lower() for alias in bot_aliases}
    specific_rules: List[Rule] = []
    wildcard_rules: List[Rule] = []
    matched_specific = False
    matched_wildcard = False

    for group in groups:
        agents = [str(agent).lower() for agent in group.get("agents", [])]
        rules = list(group.get("rules", []))

        if any(agent != "*" and agent in aliases for agent in agents):
            matched_specific = True
            specific_rules.extend(rules)
            continue

        if "*" in agents:
            matched_wildcard = True
            wildcard_rules.extend(rules)

    if matched_specific:
        return specific_rules, True, "specific"
    if matched_wildcard:
        return wildcard_rules, False, "wildcard"
    return [], False, "none"


def _normalize_percent_encoding(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        decoded = chr(int(match.group(1), 16))
        if decoded in _UNRESERVED:
            return decoded
        return "%" + match.group(1).upper()

    return _PERCENT_RE.sub(replace, value)


def _compile_rule_pattern(rule_path: str) -> re.Pattern[str]:
    cached = _PATTERN_CACHE.get(rule_path)
    if cached is not None:
        return cached

    normalized = _normalize_percent_encoding(rule_path)
    anchored = normalized.endswith("$")
    core = normalized[:-1] if anchored else normalized

    parts: List[str] = []
    for char in core:
        if char == "*":
            parts.append(".*")
        else:
            parts.append(re.escape(char))

    regex = "^" + "".join(parts)
    if anchored:
        regex += "$"

    compiled = re.compile(regex)
    _PATTERN_CACHE[rule_path] = compiled
    return compiled


def _rule_matches(rule_path: str, path: str) -> bool:
    if rule_path == "":
        return False
    pattern = _compile_rule_pattern(rule_path)
    normalized_path = _normalize_percent_encoding(path)
    return bool(pattern.match(normalized_path))


def is_path_blocked(rules: Sequence[Rule], path: str = "/") -> bool:
    """Apply RFC 9309 longest-match semantics with allow-on-tie."""
    target_path = path or "/"
    best_length = -1
    best_directive = "allow"

    for directive, rule_path in rules:
        if not _rule_matches(rule_path, target_path):
            continue

        specificity = len(_normalize_percent_encoding(rule_path))
        directive = directive.lower()

        if specificity > best_length:
            best_length = specificity
            best_directive = directive
            continue

        if specificity == best_length and directive == "allow":
            best_directive = "allow"

    return best_length >= 0 and best_directive == "disallow"


def extract_sitemap_urls(content: str | None) -> List[str]:
    if not content:
        return []

    sitemap_urls: List[str] = []
    for raw_line in str(content).splitlines():
        line = _strip_comment(raw_line)
        if not line or ":" not in line:
            continue
        field, value = line.split(":", 1)
        if field.strip().lower() == "sitemap":
            sitemap = value.strip()
            if sitemap:
                sitemap_urls.append(sitemap)
    return sitemap_urls


def classify_bot(
    groups: Sequence[Group], bot_name: str, bot_aliases: Iterable[str]
) -> Dict[str, object]:
    rules, has_specific_rule, effective_source = get_applicable_rules(groups, bot_aliases)
    has_nonroot_disallow = any(
        directive == "disallow" and path.strip() not in {"", "/"}
        for directive, path in rules
    )

    return {
        "bot": bot_name,
        "specific_rule": int(has_specific_rule),
        "effective_source": effective_source,
        "root_block": int(is_path_blocked(rules, "/")),
        "has_nonroot_disallow": int(has_nonroot_disallow),
    }
