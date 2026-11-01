import unittest

from robots_parser import (
    classify_bot,
    extract_sitemap_urls,
    get_applicable_rules,
    is_path_blocked,
    parse_robots,
)


class RobotsParserTests(unittest.TestCase):
    def test_01_wildcard_root_block(self) -> None:
        groups = parse_robots("User-agent: *\nDisallow: /\n")
        result = classify_bot(groups, "GPTBot", ["GPTBot"])
        self.assertEqual(result["root_block"], 1)

    def test_02_specific_overrides_wildcard(self) -> None:
        content = "User-agent: *\nDisallow: /\n\nUser-agent: GPTBot\nAllow: /\n"
        groups = parse_robots(content)
        result = classify_bot(groups, "GPTBot", ["GPTBot"])
        self.assertEqual(result["effective_source"], "specific")
        self.assertEqual(result["root_block"], 0)

    def test_03_allow_wins_on_tie(self) -> None:
        rules = [("disallow", "/"), ("allow", "/")]
        self.assertFalse(is_path_blocked(rules, "/"))

    def test_04_longer_match_wins(self) -> None:
        rules = [("disallow", "/"), ("allow", "/public/")]
        self.assertFalse(is_path_blocked(rules, "/public/page"))
        self.assertTrue(is_path_blocked(rules, "/private/page"))

    def test_05_empty_specific_group_does_not_fallback(self) -> None:
        content = "User-agent: GPTBot\n\nUser-agent: *\nDisallow: /\n"
        groups = parse_robots(content)
        result = classify_bot(groups, "GPTBot", ["GPTBot"])
        self.assertEqual(result["effective_source"], "specific")
        self.assertEqual(result["root_block"], 0)

    def test_06_empty_disallow_allows(self) -> None:
        groups = parse_robots("User-agent: *\nDisallow:\n")
        result = classify_bot(groups, "GPTBot", ["GPTBot"])
        self.assertEqual(result["root_block"], 0)

    def test_07_user_agent_matching_is_case_insensitive(self) -> None:
        groups = parse_robots("User-agent: GPTBOT\nDisallow: /\n")
        result = classify_bot(groups, "GPTBot", ["gptbot"])
        self.assertEqual(result["root_block"], 1)

    def test_08_blank_lines_split_groups(self) -> None:
        content = "User-agent: GPTBot\nDisallow: /\n\nUser-agent: Googlebot\nAllow: /\n"
        groups = parse_robots(content)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["agents"], ["gptbot"])
        self.assertEqual(groups[1]["agents"], ["googlebot"])

    def test_09_no_match_and_no_wildcard_means_allow(self) -> None:
        groups = parse_robots("User-agent: Googlebot\nDisallow: /\n")
        result = classify_bot(groups, "GPTBot", ["GPTBot"])
        self.assertEqual(result["effective_source"], "none")
        self.assertEqual(result["root_block"], 0)

    def test_10_walmart_style_content(self) -> None:
        content = (
            "User-agent: GPTBot\n"
            "Disallow: /\n"
            "Allow: /\n\n"
            "User-agent: *\n"
            "Disallow: /private\n"
        )
        groups = parse_robots(content)
        result = classify_bot(groups, "GPTBot", ["GPTBot"])
        self.assertEqual(result["root_block"], 0)

    def test_11_specific_rules_are_merged(self) -> None:
        content = (
            "User-agent: GPTBot\n"
            "Disallow: /private\n\n"
            "User-agent: GPTBot\n"
            "Disallow: /secret\n"
        )
        groups = parse_robots(content)
        rules, has_specific, source = get_applicable_rules(groups, ["GPTBot"])
        self.assertTrue(has_specific)
        self.assertEqual(source, "specific")
        self.assertEqual(len(rules), 2)

    def test_12_wildcard_rules_are_merged(self) -> None:
        content = (
            "User-agent: *\n"
            "Disallow: /private\n\n"
            "User-agent: *\n"
            "Disallow: /secret\n"
        )
        groups = parse_robots(content)
        rules, has_specific, source = get_applicable_rules(groups, ["GPTBot"])
        self.assertFalse(has_specific)
        self.assertEqual(source, "wildcard")
        self.assertEqual(len(rules), 2)

    def test_13_nonroot_disallow_flag(self) -> None:
        root_only = parse_robots("User-agent: GPTBot\nDisallow: /\n")
        root_result = classify_bot(root_only, "GPTBot", ["GPTBot"])
        self.assertEqual(root_result["has_nonroot_disallow"], 0)

        nonroot = parse_robots("User-agent: GPTBot\nDisallow: /private\n")
        nonroot_result = classify_bot(nonroot, "GPTBot", ["GPTBot"])
        self.assertEqual(nonroot_result["has_nonroot_disallow"], 1)

    def test_14_wildcard_matching(self) -> None:
        rules = [("disallow", "/*.gif"), ("disallow", "/private*")]
        self.assertTrue(is_path_blocked(rules, "/image.gif"))
        self.assertFalse(is_path_blocked(rules, "/image.png"))
        self.assertTrue(is_path_blocked(rules, "/private/sub"))

    def test_15_dollar_anchor(self) -> None:
        rules = [("disallow", "/private/$")]
        self.assertTrue(is_path_blocked(rules, "/private/"))
        self.assertFalse(is_path_blocked(rules, "/private/subfolder"))

    def test_16_combined_wildcard_and_anchor(self) -> None:
        rules = [("disallow", "/*.pdf$")]
        self.assertTrue(is_path_blocked(rules, "/report.pdf"))
        self.assertFalse(is_path_blocked(rules, "/report.pdf.bak"))

    def test_17_extract_sitemap_urls(self) -> None:
        content = (
            "Sitemap: https://example.com/a.xml\n"
            "sitemap: https://example.com/b.xml # comment\n"
            "Allow: /\n"
        )
        self.assertEqual(
            extract_sitemap_urls(content),
            ["https://example.com/a.xml", "https://example.com/b.xml"],
        )

    def test_18_disallow_star_blocks_root(self) -> None:
        rules = [("disallow", "/*")]
        self.assertTrue(is_path_blocked(rules, "/"))
        self.assertTrue(is_path_blocked(rules, "/abc"))

    def test_19_percent_encoding_normalization(self) -> None:
        tilde_rule = [("disallow", "/%7Eprivate/")]
        self.assertTrue(is_path_blocked(tilde_rule, "/~private/page"))

        reserved_rule = [("disallow", "/public%2Fpage")]
        self.assertFalse(is_path_blocked(reserved_rule, "/public/page"))


if __name__ == "__main__":
    unittest.main()
