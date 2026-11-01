# Cross-Reference Notes

Benchmark source: `C:\Users\xl\OneDrive - Universitat Ramón Llull\bot_paper\DATA_CONSTRUCTION_MANUAL.md` dated 2026-03-26.

Fresh run date:
- URL cleaning and parser tests run on 2026-03-27.
- Live robots and sitemap fetching run on 2026-03-27.

What matched exactly:
- Step 1 cleaning counts matched the manual exactly.
- `full_disallow` firms matched exactly at 32.
- Googlebot-blocked firms matched exactly at 17.
- Host-level `http_error_4xx`, `ssl_error`, and `http_error_5xx` matched exactly.

What differed:
- Host-level `ok_200` was 3561 vs. 3579 in the manual.
- Host-level `timeout` was 145 vs. 125 in the manual.
- Firm-level `fetch_error` was 917 vs. 786 in the manual.
- Firm-level `allow` was 4976 vs. 5083 in the manual.
- Firm-level `partial_disallow` was 4137 vs. 4161 in the manual.
- `any_ai_specific_rule_firms` was 374 vs. 373 in the manual.

Likely reason for the differences:
- This was a fresh live web fetch one calendar day later, on 2026-03-27, not a replay of the earlier snapshot.
- Small changes in robots.txt files, server availability, timeout behavior, WAF responses, or redirect behavior can move firms between `ok_200`, `missing_404`, and `fetch_error`.

Internal consistency note:
- The manual says `robots_bot_panel.csv` has 120,900 rows and also says it equals `10,062 firms x 12 bots`.
- `10,062 x 12 = 120,744`, which matches the fresh run's actual bot-panel row count.
