# GPT Answer

Fresh pipeline built from `CLAUDE.md`, `DATA_CONSTRUCTION_MANUAL.md`, and `compustat_2025.csv` only.

Files:
- `robots_parser.py`: RFC-aware parser with wildcard, dollar-anchor, and percent-encoding handling.
- `test_robots_parser.py`: unit tests for parser behavior.
- `clean_weburl.py`: Compustat URL cleaning and host extraction.
- `fetch_utils.py`: shared HTTP fetch helpers.
- `scan_robots.py`: robots.txt scanning plus firm-level classification outputs.
- `sitemap_intensity.py`: sitemap-based blocking-intensity step.
- `cross_ref_results.py`: compares fresh outputs to the manual's reported benchmarks.

Default outputs:
- `compustat_2025_clean.csv`
- `robots_host_scan.csv`
- `robots_bot_panel.csv`
- `robots_firm_summary.csv`
- `robots_sitemap_intensity_host.csv`
- `robots_sitemap_intensity_firm.csv`
- `cross_ref_manual.csv`

Typical run order:

```powershell
cd "c:\Users\xl\OneDrive - Universitat Ramón Llull\git\practice\gpt answer"
python test_robots_parser.py
python clean_weburl.py
python scan_robots.py
python sitemap_intensity.py
python cross_ref_results.py
```
