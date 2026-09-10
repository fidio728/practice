from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

import pandas as pd

from fetch_utils import fetch_robots, fetch_url
from robots_parser import extract_sitemap_urls, get_applicable_rules, is_path_blocked, parse_robots
from scan_robots import BOT_LOOKUP, TREATMENT_BOTS, classify_status_type


MAX_SITEMAP_DEPTH = 2
MAX_CHILD_SITEMAPS = 20
MAX_COMPARISON_URLS = 200_000


@dataclass
class SitemapState:
    host: str
    visited_sitemaps: set[str] = field(default_factory=set)
    successful_sitemaps: int = 0
    failed_sitemaps: int = 0
    child_sitemaps_seen: int = 0
    sitemaps_truncated: int = 0
    n_urls_before_host_filter: int = 0
    n_urls_after_host_filter: int = 0
    n_urls_dropped_by_host_filter: int = 0
    url_cap_hit: int = 0
    n_urls_dropped_by_cap: int = 0
    full_urls_seen: set[str] = field(default_factory=set)
    comparison_seen: set[str] = field(default_factory=set)
    comparison_urls: List[str] = field(default_factory=list)

    def record_url(self, url: str) -> None:
        self.n_urls_before_host_filter += 1
        parsed = urlparse(url)
        candidate_host = (parsed.hostname or "").rstrip(".").lower()
        target_host = self.host.rstrip(".").lower()

        if candidate_host != target_host:
            self.n_urls_dropped_by_host_filter += 1
            return

        self.n_urls_after_host_filter += 1

        if url in self.full_urls_seen:
            return
        self.full_urls_seen.add(url)

        comparison = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")
        if comparison in self.comparison_seen:
            return

        if len(self.comparison_seen) >= MAX_COMPARISON_URLS:
            self.url_cap_hit = 1
            self.n_urls_dropped_by_cap += 1
            return

        self.comparison_seen.add(comparison)
        self.comparison_urls.append(comparison)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def maybe_decompress(url: str, content: bytes) -> bytes:
    if url.lower().endswith(".gz") or content[:2] == b"\x1f\x8b":
        return gzip.decompress(content)
    return content


def load_progress(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {entry["host"]: entry for entry in payload.get("host_results", [])}


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


def collect_locs(root: ET.Element, wanted_parent: str) -> List[str]:
    values: List[str] = []
    for elem in root.iter():
        if local_name(elem.tag) != "loc":
            continue
        parent = elem
        text = (parent.text or "").strip()
        if text:
            values.append(text)
    return values


def fetch_sitemap_recursive(url: str, depth: int, state: SitemapState) -> None:
    if state.url_cap_hit:
        return
    if url in state.visited_sitemaps:
        return
    if depth > MAX_SITEMAP_DEPTH:
        state.sitemaps_truncated += 1
        return

    state.visited_sitemaps.add(url)
    result = fetch_url(url, timeout=30, as_bytes=True)
    if result.get("status_code") != 200 or result.get("content") is None:
        state.failed_sitemaps += 1
        return

    try:
        content = maybe_decompress(url, bytes(result["content"]))
        root = ET.fromstring(content)
    except (OSError, ET.ParseError):
        state.failed_sitemaps += 1
        return

    state.successful_sitemaps += 1
    root_kind = local_name(root.tag)

    if root_kind == "urlset":
        for elem in root.iter():
            if local_name(elem.tag) == "loc":
                value = (elem.text or "").strip()
                if value:
                    state.record_url(value)
                if state.url_cap_hit:
                    return
        return

    if root_kind == "sitemapindex":
        child_urls = []
        for elem in root.iter():
            if local_name(elem.tag) == "loc":
                value = (elem.text or "").strip()
                if value:
                    child_urls.append(value)

        for child_url in child_urls:
            if state.url_cap_hit:
                return
            if state.child_sitemaps_seen >= MAX_CHILD_SITEMAPS:
                state.sitemaps_truncated += 1
                return
            state.child_sitemaps_seen += 1
            fetch_sitemap_recursive(child_url, depth + 1, state)
        return

    state.failed_sitemaps += 1


def build_host_intensity(host: str) -> Dict[str, Any]:
    result = fetch_robots(host)
    robots_refetch_status = classify_status_type(result)
    output: Dict[str, Any] = {
        "host": host,
        "step4_scan_utc": pd.Timestamp.utcnow().isoformat(),
        "robots_refetch_status": robots_refetch_status,
        "sitemap_url": "",
        "sitemap_fetch_status": "",
        "n_urls_before_host_filter": 0,
        "n_urls_after_host_filter": 0,
        "n_urls_dropped_by_host_filter": 0,
        "n_sitemap_urls": 0,
        "url_cap_hit": 0,
        "n_urls_dropped_by_cap": 0,
        "sitemaps_truncated": 0,
        "mean_treatment_intensity": None,
        "max_treatment_intensity": None,
        "intensity_source": "partial_refetch_error_na",
    }

    for bot in TREATMENT_BOTS:
        output[f"{bot}_n_blocked"] = None
        output[f"{bot}_intensity"] = None

    if robots_refetch_status != "ok_200":
        return output

    content = result.get("content") or ""
    sitemap_urls = extract_sitemap_urls(content)
    output["sitemap_url"] = "|".join(sitemap_urls)

    if not sitemap_urls:
        output["intensity_source"] = "partial_no_sitemap_na"
        return output

    state = SitemapState(host=host)
    for sitemap_url in sitemap_urls:
        if state.url_cap_hit:
            break
        fetch_sitemap_recursive(sitemap_url, depth=0, state=state)

    if state.successful_sitemaps == 0:
        output["sitemap_fetch_status"] = "all_failed"
        output["intensity_source"] = "partial_sitemap_error_na"
        output["sitemaps_truncated"] = state.sitemaps_truncated
        return output

    output["sitemap_fetch_status"] = (
        "partial_ok" if state.failed_sitemaps > 0 or state.sitemaps_truncated > 0 else "ok"
    )
    output["n_urls_before_host_filter"] = state.n_urls_before_host_filter
    output["n_urls_after_host_filter"] = state.n_urls_after_host_filter
    output["n_urls_dropped_by_host_filter"] = state.n_urls_dropped_by_host_filter
    output["n_sitemap_urls"] = len(state.comparison_urls)
    output["url_cap_hit"] = state.url_cap_hit
    output["n_urls_dropped_by_cap"] = state.n_urls_dropped_by_cap
    output["sitemaps_truncated"] = state.sitemaps_truncated

    if not state.comparison_urls:
        output["intensity_source"] = "partial_no_urls_na"
        return output

    groups = parse_robots(content)
    intensities: List[float] = []
    for bot in TREATMENT_BOTS:
        rules, _, _ = get_applicable_rules(groups, BOT_LOOKUP[bot]["aliases"])
        blocked = sum(1 for comparison in state.comparison_urls if is_path_blocked(rules, comparison))
        intensity = blocked / len(state.comparison_urls)
        output[f"{bot}_n_blocked"] = blocked
        output[f"{bot}_intensity"] = intensity
        intensities.append(intensity)

    output["mean_treatment_intensity"] = sum(intensities) / len(intensities)
    output["max_treatment_intensity"] = max(intensities)
    output["intensity_source"] = "partial_sitemap"
    return output


def error_host_result(host: str, label: str) -> Dict[str, Any]:
    output = {
        "host": host,
        "step4_scan_utc": pd.Timestamp.utcnow().isoformat(),
        "robots_refetch_status": "request_error",
        "sitemap_url": "",
        "sitemap_fetch_status": "",
        "n_urls_before_host_filter": 0,
        "n_urls_after_host_filter": 0,
        "n_urls_dropped_by_host_filter": 0,
        "n_sitemap_urls": 0,
        "url_cap_hit": 0,
        "n_urls_dropped_by_cap": 0,
        "sitemaps_truncated": 0,
        "mean_treatment_intensity": None,
        "max_treatment_intensity": None,
        "intensity_source": label,
    }
    for bot in TREATMENT_BOTS:
        output[f"{bot}_n_blocked"] = None
        output[f"{bot}_intensity"] = None
    return output


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compute sitemap-based blocking intensity.")
    parser.add_argument("--firm-input", type=Path, default=script_dir / "robots_firm_summary.csv")
    parser.add_argument(
        "--host-output", type=Path, default=script_dir / "robots_sitemap_intensity_host.csv"
    )
    parser.add_argument(
        "--firm-output", type=Path, default=script_dir / "robots_sitemap_intensity_firm.csv"
    )
    parser.add_argument("--progress", type=Path, default=script_dir / "sitemap_progress.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--no-sleep", action="store_true")
    args = parser.parse_args()

    firm_df = pd.read_csv(
        args.firm_input,
        dtype={"gvkey": str, "ticker": str},
        keep_default_na=True,
    )
    partial_hosts = list(
        dict.fromkeys(
            firm_df.loc[firm_df["treatment_category"] == "partial_disallow", "host"].astype(str)
        )
    )
    if args.limit is not None:
        partial_hosts = partial_hosts[: args.limit]

    host_map = load_progress(args.progress)
    pending_hosts = [host for host in partial_hosts if host not in host_map]

    if args.workers <= 1:
        for idx, host in enumerate(pending_hosts, start=1):
            try:
                host_map[host] = build_host_intensity(host)
            except Exception:
                host_map[host] = error_host_result(host, "partial_refetch_error_na")
            write_progress(args.progress, host_map)
            if not args.no_sleep:
                time.sleep(0.4)
    else:
        completed_since_save = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(build_host_intensity, host): host for host in pending_hosts}
            for future in as_completed(futures):
                host = futures[future]
                try:
                    host_map[host] = future.result()
                except Exception:
                    host_map[host] = error_host_result(host, "partial_refetch_error_na")
                completed_since_save += 1
                if completed_since_save >= max(1, args.save_every):
                    write_progress(args.progress, host_map)
                    completed_since_save = 0
        if completed_since_save > 0:
            write_progress(args.progress, host_map)

    host_rows = [host_map[host] for host in partial_hosts if host in host_map]
    host_df = pd.DataFrame(host_rows)
    host_df.to_csv(args.host_output, index=False)

    merged = firm_df.merge(host_df, how="left", on="host")
    intensity_final = []
    intensity_source = []
    for row in merged.itertuples(index=False):
        if row.treatment_category == "allow":
            intensity_final.append(0.0)
            intensity_source.append("allow_zero")
        elif row.treatment_category == "full_disallow":
            intensity_final.append(1.0)
            intensity_source.append("full_one")
        elif row.treatment_category == "fetch_error":
            intensity_final.append(pd.NA)
            intensity_source.append("fetch_error_na")
        else:
            intensity_final.append(
                row.mean_treatment_intensity
                if row.intensity_source == "partial_sitemap"
                else pd.NA
            )
            intensity_source.append(
                row.intensity_source if pd.notna(row.intensity_source) else "partial_refetch_error_na"
            )

    merged["intensity_final"] = intensity_final
    merged["intensity_source"] = intensity_source
    merged.to_csv(args.firm_output, index=False)


if __name__ == "__main__":
    main()
