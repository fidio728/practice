"""
archive_preP0.py — vintage bookkeeping for the P0 holdings-snapshot rebuild.

WHAT THIS DOES
    Renames (does NOT copy) every artifact that the P0 rebuild chain
    (03 -> 04 -> 05 -> 06 -> build_c6_panel -> build_audit_panel_f1f2f7 ->
    the two desc_trend figure scripts) will overwrite, so the pre-P0 vintage
    survives for old-vs-new comparison by diag_p0_compare.py.

    Rename, not copy, is deliberate: holdings_eom.parquet alone is 4.65 GB and
    the C: volume had ~12.6 GB free when this script was written. A copy-based
    archive would not fit alongside the rebuilt panel.

SAFETY MODEL
    * DEFAULT IS DRY RUN. Nothing moves unless you pass --apply.
    * Two-phase: the full plan is computed and validated FIRST. If ANY target
      is in a refuse state, the script aborts before renaming anything, so you
      never end up half-archived.
    * Idempotent: a target whose live file is already gone and whose _preP0
      twin already exists is reported SKIP (already archived) and is not an
      error.
    * REFUSE (hard abort) whenever a _preP0 twin exists AND the live file also
      exists. That state is ambiguous — renaming would clobber the archive.
      The spec's refusal condition is "twin exists AND live is newer"; this
      script is deliberately stricter and refuses in both directions, because
      the twin-older case is equally unexplained and the archive is
      irreplaceable. Resolve by hand, then re-run.

VINTAGE REGISTER (--apply only)
    Writes VINTAGE_P0.md at the julia_descriptive ROOT (committed; output/ and
    *.csv are gitignored) plus output/_preP0_stale_manifest.csv, recording the
    rule change, W, the archived twins, the rebuild chain and every PRE-P0
    stale result family. Without this the vintage status lived only in a Python
    constant printed to stdout and died with the terminal scrollback.

NAMING RULE
    stem.ext                -> stem_preP0.ext
    foo.parquet.meta.json   -> <archived foo.parquet name>.meta.json
    SPECIAL CASE (per spec):
        holdings_eom.parquet -> holdings_eom_exactEOM_preP0.parquet

USAGE
    python archive_preP0.py              # dry run: print the plan, touch nothing
    python archive_preP0.py --apply      # perform the renames
    python archive_preP0.py --apply --tier1-only
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
# (EM-FIX-5/7, 2026-08-06) honour DPN_OUT_DIR so the archive step operates on the
# SAME directory the Julia chain writes to. Archiving C: while rebuilding E: would
# leave the live vintage un-archived, which is the one thing this script exists
# to prevent.
_env_out = os.environ.get("DPN_OUT_DIR", "").strip()
OUT = Path(_env_out).resolve() if _env_out else BASE / "output"
PLOTS = BASE / "plots"

# ---------------------------------------------------------------------------
# ARCHIVE SUFFIX — parameterised (EM-FIX-8, 2026-08-06).
#
# This script originally hard-coded "_preP0". The 2026-08-06 advisor package
# (quarter snapshot rule + zero/missing recode) overwrites the SAME artifacts a
# second time, so the P0 vintage now needs a second, differently-named archive
# generation: "_preEM". With the suffix hard-coded there was no tool that could
# produce _preEM archives at all, and build_plan() hard-ABORTED on every Tier-1
# artifact because a _preP0 twin AND a live file both exist — which is the normal
# post-P0 state.
#
# Set with --suffix preEM. Because archived_path() derives every target name from
# SUFFIX, a run with --suffix preEM can never name, read, or overwrite a _preP0
# file: the refuse-check compares the live file against the _preEM twin only.
# ---------------------------------------------------------------------------
SUFFIX = "preP0"


def _suffixed(name: str) -> str:
    return name


def log_csv_path() -> Path:
    return OUT / f"_{SUFFIX}_archive_log.csv"


def stale_manifest_path() -> Path:
    return OUT / f"_{SUFFIX}_stale_manifest.csv"


def vintage_doc_path() -> Path:
    # Durable vintage bookkeeping. Deliberately lives at the julia_descriptive
    # ROOT, not under output/: .gitignore excludes `output/`, `*.csv`, `*.parquet`
    # and `*.dta`, so an output-only marker would never be committed and the
    # vintage status would die with the terminal scrollback. The CSV twin is
    # machine-readable and is allowed to be gitignored.
    return BASE / ("VINTAGE_P0.md" if SUFFIX == "preP0" else f"VINTAGE_{SUFFIX.upper()}.md")


# W actually used by the rebuild, so the doc records the real value rather than
# a hard-coded one that can drift from 03_eom_etl.jl.
#
# (EM-FIX-8) The default is NO LONGER "10". 03_eom_etl.jl now treats an UNSET
# DPN_ASOF_WINDOW_DAYS as the advisor-directed QUARTER RULE, and a numeric value
# as the legacy window override. Defaulting to "10" here made the vintage doc
# record a quarter-rule panel as "W = 10 days" — a fabricated provenance record
# produced precisely when the operator did the right thing and left the variable
# unset. Mirror 03's logic instead.
_asof_env = os.environ.get("DPN_ASOF_WINDOW_DAYS", "").strip()
ASOF_IS_QUARTER_RULE = (_asof_env == "")
ASOF_W = _asof_env if _asof_env else "n/a (quarter rule)"
ASOF_RULE_DESC = (
    "QUARTER rule — per (fund_id, fsym_id), the LATEST REPORT_DATE inside "
    "[quarter_start, qend] (advisor-directed, Emanuele 2026-08-04 24:17)"
    if ASOF_IS_QUARTER_RULE else
    f"WINDOW rule — per (fund_id, fsym_id), the LATEST REPORT_DATE in "
    f"[max(quarter_start, qend - {ASOF_W}), qend] (DPN_ASOF_WINDOW_DAYS override)"
)

# The rebuilt chain, in execution order (spec item 4).
REBUILD_CHAIN = [
    "03_eom_etl.jl                 -> holdings_eom.parquet  (THE rule change)",
    "04_us_ownership_european.jl   -> I_ict / marketcap_it / country_total_ct / ownership_ict",
    "05_combine_visualize.jl       -> merged_us_eu_matched, us_vs_nonus_diff, GPR series",
    "06_cartesian_grid.jl          -> merged_us_eu_zero_filled.parquet",
    "build_c6_panel.py             -> c6_panel.dta",
    "build_audit_panel_f1f2f7.py   -> audit_c6_panel.dta",
    "build_desc_trend_us_holdings.py / build_desc_trend_china_links.py -> advisor figures",
]

# Free space below this on the output volume is worth a loud warning: the P0
# rebuild writes a fresh ~4.7 GB holdings_eom.parquet (plus a .tmp of the same
# size mid-write, since atomic_copy_to writes .tmp then mv) on top of the
# archived 4.65 GB twin.
MIN_FREE_GB_WARN = 11.0

# ---------------------------------------------------------------------------
# TARGET ENUMERATION
# Each entry: (relative path from BASE, tier, written_by)
# Tiers:
#   1  = holdings-derived binary panel; content WILL change under P0.
#   1i = rebuilt by 05 but built from the GPR series only, NOT from holdings.
#        Archived as a free invariance check: post-rebuild these must be
#        numerically identical (SHOCK UNTOUCHED discipline).
#   2  = small CSV diagnostic overwritten by 03/04/05/06.
#   3  = advisor-facing figure inputs / figure files.
# ---------------------------------------------------------------------------
TARGETS: list[tuple[str, str, str]] = [
    # ---------------- Tier 1: binary panels (the real archive) -------------
    ("output/holdings_eom.parquet",                  "1",  "03_eom_etl.jl:80"),
    ("output/I_ict_panel.parquet",                   "1",  "04_us_ownership_european.jl:139"),
    ("output/marketcap_it.parquet",                  "1",  "04_us_ownership_european.jl:168"),
    ("output/country_total_ct.parquet",              "1",  "04_us_ownership_european.jl:223"),
    ("output/ownership_ict.parquet",                 "1",  "04_us_ownership_european.jl:244"),
    ("output/merged_us_eu_matched.parquet",          "1",  "05_combine_visualize.jl:359"),
    ("output/us_vs_nonus_diff.parquet",              "1",  "05_combine_visualize.jl:656"),
    ("output/merged_us_eu_zero_filled.parquet",      "1",  "06_cartesian_grid.jl:293"),
    ("output/c6_panel.dta",                          "1",  "build_c6_panel.py:37"),
    ("output/audit_c6_panel.dta",                    "1",  "build_audit_panel_f1f2f7.py:30"),
    # audit_c6_panel.parquet is a .dta->.parquet transcode written by
    # run_cum4_inference.py:97 / run_randomization_inference.py:102 /
    # run_ri_3pairwise.py:34. It is derived from audit_c6_panel.dta, so it is
    # stale the moment the .dta is rebuilt.
    ("output/audit_c6_panel.parquet",                "1",  "run_cum4_inference.py:97 (transcode)"),

    # ------- Tier 1i: rebuilt by 05 but holdings-independent (invariance) ---
    ("output/gpr_monthly_with_shock.parquet",        "1i", "05_combine_visualize.jl:142"),
    ("output/gpr_quarterly_with_shock.parquet",      "1i", "05_combine_visualize.jl:158"),
    # Tier 1i, NOT tier 2. Like the two GPR parquets it is built from the GPR
    # series alone and is holdings-independent, so post-rebuild it must be
    # byte-identical — a free "SHOCK UNTOUCHED" invariance check. It was missed
    # in the first enumeration because it is the ONLY artifact in the whole chain
    # written with a bare open(path, "w") instead of atomic_copy_to / CSV.write /
    # to_parquet / to_stata, so a grep on the write helpers does not surface it.
    ("output/gpr_ar1_coefficients.csv",              "1i", "05_combine_visualize.jl:129"),

    # ---------------- Tier 2: CSV diagnostics from 03 ----------------------
    ("output/03_eom_dup_key_count.csv",              "2",  "03_eom_etl.jl:193"),
    ("output/03_eom_weekend_audit.csv",              "2",  "03_eom_etl.jl:207"),
    ("output/03_eom_issue_type_breakdown.csv",       "2",  "03_eom_etl.jl:221"),
    ("output/03_eom_issue_type_breakdown_europe.csv", "2", "03_eom_etl.jl:230"),
    ("output/03_eom_issue_type_breakdown_us_investors.csv", "2", "03_eom_etl.jl:239"),
    ("output/03_eom_coverage_by_year.csv",           "2",  "03_eom_etl.jl:249"),
    ("output/03_eom_investor_country_2018_12.csv",   "2",  "03_eom_etl.jl:259"),
    ("output/03_eom_company_country_2018_12.csv",    "2",  "03_eom_etl.jl:267"),
    ("output/03_us_x_eu_cells_by_month.csv",         "2",  "03_eom_etl.jl:279"),
    # Phase A only (DPN_SKIP_AUDIT=false). Usually absent/stale; harmless.
    ("output/03_audit_issue_type_holdings.csv",      "2",  "03_eom_etl.jl:111 (Phase A only)"),
    ("output/03_audit_dom_distribution.csv",         "2",  "03_eom_etl.jl:116 (Phase A only)"),

    # ---------------- Tier 2: CSV diagnostics from 04 ----------------------
    ("output/04_us_ownership_eu_snapshot.csv",       "2",  "04_us_ownership_european.jl:360"),
    ("output/04_us_own_by_eu_country_snapshot.csv",  "2",  "04_us_ownership_european.jl:378"),
    ("output/04_us_ownership_eu_timeseries.csv",     "2",  "04_us_ownership_european.jl:395"),
    ("output/04_top30_us_owned_eu_firms.csv",        "2",  "04_us_ownership_european.jl:409"),

    # ---------------- Tier 2: CSV diagnostics from 05 ----------------------
    ("output/05_unmatched_profile_by_country.csv",   "2",  "05_combine_visualize.jl:251"),
    ("output/05_match_type_distribution.csv",        "2",  "05_combine_visualize.jl:261"),
    ("output/05_multi_match_per_sec_entity.csv",     "2",  "05_combine_visualize.jl:271"),
    ("output/05_coverage_cascade.csv",               "2",  "05_combine_visualize.jl:347"),
    ("output/05_merged_panel_composition.csv",       "2",  "05_combine_visualize.jl:453"),
    ("output/05_scatter_own_vs_cn_data.csv",         "2",  "05_combine_visualize.jl:562"),
    ("output/05_within_europe_share_by_group.csv",   "2",  "05_combine_visualize.jl:632"),
    ("output/05_us_vs_nonus_high_share_data.csv",    "2",  "05_combine_visualize.jl:645"),
    ("output/05_gap_months_diagnostic.csv",          "2",  "05_combine_visualize.jl:721"),
    ("output/05_diff_us_vs_nonus_high.csv",          "2",  "05_combine_visualize.jl:766"),

    # ---------------- Tier 2: CSV diagnostics from 06 ----------------------
    ("output/06_panel_composition_c6.csv",           "2",  "06_cartesian_grid.jl:361"),
    ("output/05_scatter_own_vs_cn_data_c6.csv",      "2",  "06_cartesian_grid.jl:397"),
    ("output/05_within_europe_share_by_group_c6.csv", "2", "06_cartesian_grid.jl:437"),
    ("output/05_diff_us_vs_nonus_high_c6.csv",       "2",  "06_cartesian_grid.jl:487"),
    ("output/05_us_vs_nonus_high_share_data_c6.csv", "2",  "06_cartesian_grid.jl:540"),

    # ---------------- Tier 3: advisor figures + their data ------------------
    ("output/desc_trend_us_holdings_real_growth.csv", "3", "build_desc_trend_us_holdings.py:218"),
    ("plots/fig_us_holdings_real_yoy_growth.png",     "3", "build_desc_trend_us_holdings.py:321"),
    ("plots/fig_us_holdings_real_yoy_growth.pdf",     "3", "build_desc_trend_us_holdings.py:322"),
    ("output/desc_trend_china_link_fraction.csv",     "3", "build_desc_trend_china_links.py:447"),
    ("plots/fig_china_link_fraction.png",             "3", "build_desc_trend_china_links.py:547"),
    ("plots/fig_china_link_fraction.pdf",             "3", "build_desc_trend_china_links.py:548"),

    # ===================== EM-CHANGE-2 additions (2026-08-06) ===============
    # The original TARGETS list predates EM-CHANGE-2 and enumerated only what
    # the P0 holdings rebuild overwrote. The 2026-08-06 package ALSO re-runs
    # 02_china_exposure.jl, which overwrites everything below. Two of these are
    # load-bearing, not bookkeeping:
    #
    #   firm_quarter_china_exposure.parquet — 02_china_exposure.jl reads
    #       firm_quarter_china_exposure_preEM.parquet for the EM-CHANGE-2 P0
    #       bit-identity gate ("exposure changes ONLY where the recode applies").
    #       If it is missing the gate is SKIPPED, so a missed archive silently
    #       disables the only automated proof that positive-denominator cells
    #       survived the recode unchanged.
    #   eu_revere_universe_qend.parquet — the time-versioned EU universe 05 and
    #       06 join on; the PIT presence spine is rebuilt with it.
    ("output/firm_quarter_china_exposure.parquet",    "1",  "02_china_exposure.jl:1113"),
    ("output/eu_revere_universe.parquet",             "1",  "02_china_exposure.jl (latest snapshot, doc only)"),
    ("output/eu_revere_universe_qend.parquet",        "1",  "02_china_exposure.jl (time-versioned; required by 05/06)"),
    ("output/02_china_exposure_timeseries.csv",       "2",  "02_china_exposure.jl:~1200"),
    ("output/02_china_exposure_percentiles_snapshot.csv", "2", "02_china_exposure.jl:~1177"),
    ("output/02_dist_cn_total_2018.csv",              "2",  "02_china_exposure.jl:~1141"),
    ("output/02_china_edge_path_summary.csv",         "2",  "02_china_exposure.jl"),
    ("output/02_rev_co_asof_drift_diag.csv",          "2",  "02_china_exposure.jl"),
    ("output/02_revere_co_variation_diag.csv",        "2",  "02_china_exposure.jl"),
    ("output/02_revere_company_country_top25.csv",    "2",  "02_china_exposure.jl"),
]

# Deliberately NOT archived:
#   output/cpiaucns_monthly.csv — CPI-U cache from FRED, rewritten byte-identical
#       by build_desc_trend_us_holdings.py. Holdings-independent; archiving it
#       only risks a needless FRED re-download.

# Special archive names (spec-mandated). ONLY valid for the preP0 generation:
# "exactEOM" names the RULE the pre-P0 panel was built under. The preEM
# generation archives the P0 as-of panel, whose name must be the plain
# holdings_eom_preEM.parquet that 03_eom_etl.jl's PHASE B0 guard looks for.
SPECIAL_NAME_BY_SUFFIX = {
    "preP0": {"output/holdings_eom.parquet": "holdings_eom_exactEOM_preP0.parquet"},
}

# Artifacts that consume holdings_eom / merged_us_eu_zero_filled but are NOT
# rebuilt by this package. They are NOT renamed (nothing overwrites them) —
# they simply become PRE-P0-VINTAGE and must not be mixed with post-P0 results.
STALE_VINTAGE = [
    ("output/fourgroup_panel.dta",              "build_fourgroup_panel.py"),
    ("output/fund_label.parquet",               "build_fourgroup_panel.py"),
    ("output/extensive_margin_panel.dta",       "build_extensive_margin_panel.py"),
    ("output/extensive_margin_panel.parquet",   "build_extensive_margin_panel.py"),
    ("output/extensive_margin_panel_smoke.dta", "build_extensive_margin_panel.py"),
    ("output/extensive_margin_panel_smoke.parquet", "build_extensive_margin_panel.py"),
    ("output/flow_decomposition_panel.dta",     "flow decomposition build"),
    ("output/flow_decomposition_panel.parquet", "flow decomposition build"),
    ("output/ownership_share_observed.parquet", "build_ownership_share_panel.py"),
    ("output/ownership_share_float.parquet",    "build_ownership_share_panel.py"),
    ("output/ownership_c6_panel.dta",           "build_ownership_share_c6_panel.py"),
    ("output/sagg_panel.dta",                   "build_sagg_panel.py"),
    ("output/sagg_panel.parquet",               "build_sagg_panel.py"),
    ("output/shocklag_panel.dta",               "build_shocklag_panel.py"),
    ("output/shocklag_panel.parquet",           "build_shocklag_panel.py"),
    ("output/c6_panel_riskset.dta",             "build_spell_riskset.py"),
    ("output/c6_panel_riskset_lagonly.dta",     "build_riskset_lagonly.py"),
    ("output/c6_panel_spell.dta",               "build_spell_boundary.py"),
    ("output/c6_panel_country_pair.dta",        "build_country_pair_shock.py"),
    ("output/country_panel.dta",                "build_country_panel.py"),
    ("output/firm_ladder_panel.dta",            "build_firm_ladder_panel.py"),
    ("output/merged_us_ru_zero_filled.parquet", "06_russia_grid.jl"),
    ("output/c6_panel_russia.dta",              "build_russia_c6_panel.py"),
    ("output/c6_panel_russia.parquet",          "build_russia_c6_panel.py"),
    ("output/russia_lp_panel.parquet",          "build_russia_lp_panel.py"),
]


def live_path(rel: str) -> Path:
    """Resolve a TARGETS-relative path, honouring DPN_OUT_DIR for output/*."""
    if rel.startswith("output/"):
        return OUT / rel[len("output/"):]
    return BASE / rel


def archived_path(rel: str) -> Path:
    """Map a live artifact path to its archive path under the ACTIVE SUFFIX.

    Every archive name is derived from the module-level SUFFIX, so a run with
    --suffix preEM can never name (and therefore never overwrite or refuse on)
    an existing _preP0 file.
    """
    live = live_path(rel)
    special = SPECIAL_NAME_BY_SUFFIX.get(SUFFIX, {})
    if rel in special:
        return live.parent / special[rel]
    name = live.name
    if name.endswith(".parquet.meta.json"):
        stem = name[: -len(".parquet.meta.json")]
        return live.parent / f"{stem}_{SUFFIX}.parquet.meta.json"
    stem, dot, ext = name.rpartition(".")
    if not dot:
        return live.parent / f"{name}_{SUFFIX}"
    return live.parent / f"{stem}_{SUFFIX}.{ext}"


def sidecar_of(rel: str) -> str | None:
    """Manifest sidecar written by 00_setup.jl write_manifest(), if applicable."""
    if rel.endswith(".parquet"):
        return rel + ".meta.json"
    return None


def sidecar_archive_path(rel: str) -> Path:
    """Sidecar archive rides along with its parquet's archived name."""
    return Path(str(archived_path(rel)) + ".meta.json")


def fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f}{unit}" if unit != "B" else f"{n:,}B"
        n /= 1024.0
    return f"{n:.1f}GB"


def fmt_mtime(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def build_plan(rels: list[tuple[str, str, str]]):
    """Return (plan_rows, n_refuse). Pure inspection; touches nothing."""
    plan = []
    n_refuse = 0
    for rel, tier, writer in rels:
        pairs = [(rel, archived_path(rel), "primary")]
        sc = sidecar_of(rel)
        if sc is not None:
            pairs.append((sc, sidecar_archive_path(rel), "sidecar"))

        for r, dst, kind in pairs:
            src = live_path(r)
            s_ex, d_ex = src.is_file(), dst.is_file()
            if s_ex and not d_ex:
                action, note = "RENAME", ""
            elif not s_ex and d_ex:
                action, note = "SKIP", "already archived"
            elif not s_ex and not d_ex:
                action = "MISSING"
                note = ("expected — sidecar not written for this artifact"
                        if kind == "sidecar" else "artifact was never built")
            else:  # both exist -> ambiguous, refuse
                action = "REFUSE"
                if src.stat().st_mtime > dst.stat().st_mtime:
                    note = (f"CLOBBER RISK: live file is NEWER than the _{SUFFIX} "
                            "twin — renaming would destroy the archive")
                else:
                    note = (f"AMBIGUOUS: _{SUFFIX} twin exists and is newer than "
                            "the live file — unexplained state")
                n_refuse += 1
            plan.append({
                "tier": tier, "kind": kind, "writer": writer, "action": action,
                "src": r, "src_abs": str(src),
                "dst": dst.name, "dst_abs": str(dst),
                "size": src.stat().st_size if s_ex else 0,
                "src_mtime": fmt_mtime(src) if s_ex else "",
                "dst_mtime": fmt_mtime(dst) if d_ex else "",
                "note": note,
            })
    return plan, n_refuse


def write_vintage_doc(plan: list[dict]) -> None:
    """Persist the vintage status to a COMMITTED markdown file plus a
    machine-readable CSV, so it survives the terminal session.

    The 25-file PRE-P0-VINTAGE list used to exist only as a Python constant
    printed to stdout; _preP0_archive_log.csv records the renames but not the
    stale families, and none of the 15 downstream consumers of holdings_eom
    carries a header banner. Once the terminal scrolled, nothing on disk said
    that fourgroup / direction / tercile / flow / extmargin / russia / riskset /
    sagg / shocklag results describe the OLD snapshot — exactly the mixed-vintage
    hazard the archive step exists to prevent.
    """
    stamp = datetime.now().isoformat(timespec="seconds")

    archived = [r for r in plan
                if r["action"] in ("RENAME", "SKIP") and r["kind"] == "primary"]
    never_built = [r for r in plan if r["action"] == "MISSING" and r["kind"] == "primary"]

    lines: list[str] = []
    A = lines.append
    A("# VINTAGE_P0 — holdings-snapshot rebuild vintage register")
    A("")
    A(f"Generated by `archive_preP0.py --apply` at **{stamp}**.")
    A("Regenerated on every `--apply`. Do not hand-edit; edit the script.")
    A("")
    A("## 1. The rule change")
    A("")
    A("| | |")
    A("|---|---|")
    A("| Date | 2026-08-04 |")
    A("| Package | P0 — holdings snapshot rebuild |")
    A("| OLD rule (`03_eom_etl.jl`) | `REPORT_DATE = LAST_DAY(REPORT_DATE)` AND "
      "`MONTH(REPORT_DATE) IN (3,6,9,12)` — an EXACT calendar quarter-end match |")
    A("| NEW rule | per `(fund_id, fsym_id, quarter)`, the LATEST `REPORT_DATE` "
      "inside the selection window, stamped with `report_date = qend`; true date "
      "kept in `report_date_actual`, `asof_gap_days = qend - actual` |")
    A(f"| **Selection window in force at archive time** | {ASOF_RULE_DESC} |")
    A(f"| **`DPN_ASOF_WINDOW_DAYS`** | **{ASOF_W}** "
      + ("(UNSET at archive time -> 03_eom_etl.jl uses the advisor quarter rule; "
         "this is NOT W = 10)" if ASOF_IS_QUARTER_RULE else "days") + " |")
    A(f"| Archive generation | `_{SUFFIX}` |")
    A("| Why | on weekend quarter-ends a large part of the fund universe stamps "
      "the prior business day and was dropped wholesale; the loss is NOT "
      "US/NONUS symmetric, so group x quarter FE cannot absorb it |")
    A("")
    A("**Universe change, not a weekend-only repair.** The as-of rule also admits "
      "EARLY REPORTERS on weekday quarter-ends, so coverage rises on *every* "
      "quarter (full sample weekday mean 81.82% -> 83.94%). Any post-P0 level "
      "compared against a pre-P0 level is a comparison across two universes.")
    A("")
    A("## 2. Archived pre-P0 twins (renamed, not copied)")
    A("")
    A("| tier | live artifact | archived as | written by |")
    A("|---|---|---|---|")
    for r in archived:
        A(f"| {r['tier']} | `{r['src']}` | `{r['dst']}` | {r['writer']} |")
    A("")
    A("Tier key: **1** holdings-derived binary panel (content WILL change); "
      "**1i** rebuilt by 05 but holdings-INDEPENDENT, so post-rebuild it must be "
      "byte-identical (the \"SHOCK UNTOUCHED\" invariance check); "
      "**2** CSV diagnostic; **3** advisor figure / figure data.")
    if never_built:
        A("")
        A("Enumerated but never built (nothing to archive): "
          + ", ".join(f"`{r['src']}`" for r in never_built))
    A("")
    A("## 3. Rebuild chain (run in this order)")
    A("")
    for i, step in enumerate(REBUILD_CHAIN, 1):
        A(f"{i}. `{step}`")
    A("")
    A("Then validate with `diag_p0_compare.py` (old vs new gates).")
    A("")
    A("## 4. PRE-P0 VINTAGE — stale result families")
    A("")
    A("These are **NOT** renamed (nothing overwrites them) and are **NOT** rebuilt "
      "in this package. After the rebuild they describe the **OLD** snapshot.")
    A("")
    A("> **PRE-P0, pending re-run, do not mix.** Every file below was built from "
      "the exact-EOM `holdings_eom.parquet` (or from `merged_us_eu_zero_filled."
      "parquet` built from it). Do not report any number derived from them "
      "alongside a post-P0 number, and do not re-run them as part of this "
      "package.")
    A("")
    A("| status | artifact | built by |")
    A("|---|---|---|")
    for rel, writer in STALE_VINTAGE:
        p = live_path(rel)
        A(f"| {'present' if p.is_file() else 'absent'} | `{rel}` | `{writer}` |")
    A("")
    A("Affected result families: fourgroup, direction, tercile, flow decomposition, "
      "extensive margin, russia, riskset/spell, sagg, shocklag, ownership-share, "
      "country panel, firm ladder.")
    A("")
    vintage_doc_path().write_text("\n".join(lines) + "\n", encoding="utf-8")

    with stale_manifest_path().open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["generated_at", "vintage", "status", "artifact", "built_by",
                    "asof_window_days", "note"])
        for rel, writer in STALE_VINTAGE:
            p = live_path(rel)
            w.writerow([stamp, "PRE-P0", "present" if p.is_file() else "absent",
                        rel, writer, ASOF_W,
                        "PRE-P0, pending re-run, do not mix with post-P0 results"])

    print(f"\nvintage register written:\n  {vintage_doc_path()}\n  {stale_manifest_path()}")
    print("  (VINTAGE_P0.md is OUTSIDE output/ on purpose — output/ and *.csv are "
          "gitignored, so only the .md gets committed.)")


def main() -> int:
    global SUFFIX
    ap = argparse.ArgumentParser(
        description="Archive a superseded artifact vintage by rename.")
    ap.add_argument("--apply", action="store_true",
                    help="actually rename (default is a dry-run plan)")
    ap.add_argument("--tier1-only", action="store_true",
                    help="restrict to tier 1 / 1i binary panels")
    ap.add_argument("--suffix", default="preP0",
                    help="archive generation suffix (default preP0; use preEM "
                         "for the 2026-08-06 advisor package). Archive names are "
                         "derived from it, so a preEM run never touches a preP0 twin.")
    args = ap.parse_args()

    SUFFIX = args.suffix.strip().lstrip("_")
    if not SUFFIX or not SUFFIX.replace("_", "").isalnum():
        print(f"FATAL: --suffix must be alphanumeric, got {args.suffix!r}")
        return 2

    if not OUT.is_dir():
        print(f"FATAL: output dir not found: {OUT}")
        return 2

    rels = TARGETS
    if args.tier1_only:
        rels = [t for t in TARGETS if t[1].startswith("1")]

    plan, n_refuse = build_plan(rels)

    width = max(len(r["src"]) for r in plan) + 2
    print("=" * 100)
    print(f"archive_preP0.py — {'APPLY' if args.apply else 'DRY RUN (nothing will move)'}")
    print(f"base       : {BASE}")
    print(f"output dir : {OUT}" + ("  [DPN_OUT_DIR override]" if _env_out else ""))
    print(f"suffix     : _{SUFFIX}")
    print(f"snapshot   : {ASOF_RULE_DESC}")
    print("=" * 100)
    cur_tier = None
    for r in plan:
        if r["tier"] != cur_tier:
            cur_tier = r["tier"]
            label = {"1": "TIER 1  holdings-derived binary panels",
                     "1i": "TIER 1i holdings-INDEPENDENT (invariance check)",
                     "2": "TIER 2  CSV diagnostics",
                     "3": "TIER 3  advisor figures + figure data"}[cur_tier]
            print(f"\n--- {label} " + "-" * max(0, 60 - len(label)))
        size = f"{fmt_size(r['size']):>10}" if r["size"] else " " * 10
        print(f"  {r['action']:<8} {r['src']:<{width}} -> {r['dst']:<46}"
              f" {size}  {r['note']}")

    counts = {}
    for r in plan:
        counts[r["action"]] = counts.get(r["action"], 0) + 1
    total_bytes = sum(r["size"] for r in plan if r["action"] == "RENAME")
    print("\n" + "-" * 100)
    print("summary: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"bytes to rename (no copy, same volume): {fmt_size(total_bytes)}")

    du = shutil.disk_usage(str(OUT))
    free_gb = du.free / 1024 ** 3
    print(f"free space on the output volume: {free_gb:.1f} GB")
    if free_gb < MIN_FREE_GB_WARN:
        print(f"  WARNING: below {MIN_FREE_GB_WARN} GB. The rebuild writes a fresh")
        print("  ~4.7 GB holdings_eom.parquet AND a same-size .tmp mid-write "
              "(atomic_copy_to),")
        print("  on top of the archived 4.65 GB twin. Free space before rebuilding,")
        print("  or point OUT_DIR at a roomier volume.")

    print("\nPRE-P0-VINTAGE (NOT renamed — not overwritten by this package, but stale):")
    for rel, writer in STALE_VINTAGE:
        p = live_path(rel)
        mark = "present" if p.is_file() else "absent "
        print(f"  [{mark}] {rel:<48} ({writer})")
    print("  These consume holdings_eom / merged_us_eu_zero_filled. After the")
    print("  rebuild they describe the OLD snapshot. Do NOT mix them with post-P0")
    print("  results and do NOT re-run them in this package.")
    print(f"  --apply persists this list to {vintage_doc_path().name} (committed) and")
    print(f"  {stale_manifest_path().name} (machine-readable), so it survives the session.")

    if n_refuse:
        print("\n" + "!" * 100)
        print(f"ABORT: {n_refuse} target(s) in a REFUSE state. Nothing was renamed.")
        print("Resolve each by hand (the _preP0 twin is the irreplaceable copy),")
        print("then re-run.")
        print("!" * 100)
        return 1

    if not args.apply:
        print("\nDRY RUN complete. Nothing moved. Re-run with --apply to execute.")
        return 0

    moved = []
    for r in plan:
        if r["action"] != "RENAME":
            continue
        src, dst = Path(r["src_abs"]), Path(r["dst_abs"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)          # same-volume rename; O(1), no data copy
        moved.append(r)
        print(f"  renamed {r['src']} -> {r['dst']}")

    stamp = datetime.now().isoformat(timespec="seconds")
    _log = log_csv_path()
    new_log = not _log.is_file()
    with _log.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new_log:
            w.writerow(["archived_at", "tier", "kind", "writer", "src", "dst",
                        "size_bytes", "src_mtime"])
        for r in moved:
            w.writerow([stamp, r["tier"], r["kind"], r["writer"], r["src"],
                        r["dst"], r["size"], r["src_mtime"]])

    print(f"\nAPPLY complete: {len(moved)} file(s) renamed. Log: {_log}")

    write_vintage_doc(plan)

    print("Next: rebuild 03 with the as-of selection rule, then 04, 05, 06,")
    print("build_c6_panel.py, build_audit_panel_f1f2f7.py, then run")
    print("diag_p0_compare.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
