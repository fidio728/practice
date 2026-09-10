"""
audit_eom_gap.py - Compare day-15 baseline vs new EOM-primary monthly
snapshots. Reports differ rates at the 3-tuple (digest) level for policy
comparison and the 4-tuple (digest + timestamp) level for Step 6.5 refetch
estimation, with sentinel-aware denominators.

Two key levels, two distinct denominators:

  - POLICY_TIERS = {ok_200, ok_revisit_resolved, ok_404_allow_all}
    3-tuple digest differ rate is computed over POLICY_TIERS. Captures
    Tier 0/1 treatment-state change (robots.txt content semantics
    changed between day-15 and EOM picks).

  - SITEMAP_TIERS = {ok_200, ok_revisit_resolved}
    4-tuple (digest OR timestamp) differ rate is computed only over
    SITEMAP_TIERS, because ok_404_allow_all cells have no robots.txt
    sitemap declarations - Step 6 skips them and Step 6.5 never refetches
    them, so they cannot contribute to refetch pressure even when their
    timestamp changes. Including them would inflate the reported refetch
    rate.

Inputs (BOTH must be present and BOTH must be Step 2 v3 schema):
  D:/anoconda/bot_paper_data/monthly_snapshots_day15_backup.csv
        - backup of the legacy day-15 selection. Must be re-emitted under
          v3 schema (raw_statuscode / tier_status / etc.) before running
          this audit. The script will refuse to run if the day-15 file is
          legacy schema, because legacy "-" rows are silently mapped to
          uninterpretable, which would understate revisit-resolved
          comparability.
  D:/anoconda/bot_paper_data/monthly_snapshots.csv
        - new EOM-primary output of Step 2 v3.

Outputs (printed to stdout):
  - Cell coverage by tier_status under each spec
  - 3-tuple differ rate over POLICY_TIERS (Tier 0/1 policy change)
  - 4-tuple differ rate over SITEMAP_TIERS (Step 6.5 refetch pressure;
    excludes ok_404_allow_all)
  - Tier-status flow matrix (day-15 tier x EOM tier)
  - Forward-resolved revisit count (revisit_source_after_timestamp == 1)

Note on the Step 6.5 refetch number: this is an upper bound on refetch
pressure. The true refetch count further requires that the digest
contains at least one Sitemap: declaration (extracted by Step 6 from the
robots body). That filter is not applied here because it requires loading
robots_content.json. Treat the number as "at most this many cells need
sitemap refetch under EOM."

Does not modify any files.
"""

from collections import Counter, defaultdict
from pathlib import Path
import csv
import sys

DATA = Path("D:/anoconda/bot_paper_data")
DAY15_PATH = DATA / "monthly_snapshots_day15_backup.csv"
EOM_PATH = DATA / "monthly_snapshots.csv"

# Tier statuses for which a digest+timestamp comparison is meaningful as a
# policy change measure (3-tuple).
POLICY_TIERS = {"ok_200", "ok_revisit_resolved", "ok_404_allow_all"}

# Subset of POLICY_TIERS that can actually trigger Step 6 / 6.5 sitemap
# fetching. ok_404_allow_all has no sitemap declarations -> never refetched
# -> excluded from refetch denominator.
SITEMAP_TIERS = {"ok_200", "ok_revisit_resolved"}

# Required v3 schema columns. If any are missing in either file, refuse
# to run (legacy schema would silently understate revisit-resolved
# comparability).
REQUIRED_V3_COLS = {
    "tier_status", "raw_statuscode", "digest", "snapshot_timestamp",
    "revisit_source_after_timestamp",
}


def load_snapshot_csv(path):
    """Load a v3 monthly_snapshots*.csv keyed by (host, year_month). Returns
    (rows_by_key, fieldnames). Refuses to interpret legacy schema."""
    if not path.exists():
        return {}, []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = {}
        for row in reader:
            key = (row["host"], row["year_month"])
            rows[key] = row
    return rows, fieldnames


def assert_v3_schema(fieldnames, label):
    cols = set(fieldnames or [])
    missing = REQUIRED_V3_COLS - cols
    if missing:
        print(f"ABORT: {label} is not Step 2 v3 schema.")
        print(f"  Missing columns: {sorted(missing)}")
        print(f"  Re-emit the file under Step 2 v3 (which produces")
        print(f"  raw_statuscode / tier_status / revisit_* fields) and re-run.")
        sys.exit(1)


def tier_breakdown(rows_by_key, label):
    cnt = Counter(r["tier_status"] for r in rows_by_key.values())
    n = len(rows_by_key)
    print(f"  [{label}] {n:,} cells")
    for t in ("ok_200", "ok_revisit_resolved", "ok_404_allow_all",
              "uninterpretable", "gap"):
        c = cnt.get(t, 0)
        pct = (100 * c / n) if n else 0.0
        print(f"      {t:<28} {c:>10,}  ({pct:5.1f}%)")


def policy_pair(row):
    """For 3-tuple policy comparison: (digest, timestamp) if tier in
    POLICY_TIERS and digest is non-empty, else None."""
    if row["tier_status"] not in POLICY_TIERS:
        return None
    digest = row.get("digest", "")
    ts = row.get("snapshot_timestamp", "")
    if digest in ("", "NA"):
        return None
    return (digest, ts)


def sitemap_pair(row):
    """For 4-tuple Step-6.5 refetch comparison: (digest, timestamp) if tier
    in SITEMAP_TIERS (excludes ok_404_allow_all), digest non-empty, else
    None."""
    if row["tier_status"] not in SITEMAP_TIERS:
        return None
    digest = row.get("digest", "")
    ts = row.get("snapshot_timestamp", "")
    if digest in ("", "NA"):
        return None
    return (digest, ts)


def main():
    print("=== audit_eom_gap (v3 schema, sentinel-aware denominators) ===\n")

    print(f"Loading day-15 baseline from {DAY15_PATH.name}...")
    day15, day15_cols = load_snapshot_csv(DAY15_PATH)
    print(f"  {len(day15):,} (host, ym) cells\n")

    print(f"Loading EOM-primary from {EOM_PATH.name}...")
    eom, eom_cols = load_snapshot_csv(EOM_PATH)
    print(f"  {len(eom):,} (host, ym) cells\n")

    if not day15 or not eom:
        print("ABORT: missing one or both inputs.")
        print(f"  Backup the legacy day-15 monthly_snapshots.csv to")
        print(f"  monthly_snapshots_day15_backup.csv (under v3 schema)")
        print(f"  before running new Step 2, then re-run this audit.")
        return 1

    assert_v3_schema(day15_cols, DAY15_PATH.name)
    assert_v3_schema(eom_cols, EOM_PATH.name)

    print("=== Tier breakdown by spec ===")
    tier_breakdown(day15, "day-15 baseline")
    print()
    tier_breakdown(eom, "EOM-primary    ")
    print()

    common_keys = set(day15) & set(eom)
    only_d = set(day15) - set(eom)
    only_e = set(eom) - set(day15)
    print(f"Cells in both files:        {len(common_keys):,}")
    print(f"Only in day-15 (orphan):    {len(only_d):,}")
    print(f"Only in EOM (orphan):       {len(only_e):,}")
    print()

    # === 3-tuple policy differ over POLICY_TIERS ===
    n_policy = 0
    n_3tuple_diff = 0
    flow = defaultdict(int)

    # === 4-tuple sitemap-refetch differ over SITEMAP_TIERS ===
    n_sitemap = 0
    n_sitemap_4_diff = 0
    n_sitemap_ts_only = 0  # same digest, different timestamp

    for k in common_keys:
        d_row = day15[k]
        e_row = eom[k]
        flow[(d_row["tier_status"], e_row["tier_status"])] += 1

        # Policy comparison (POLICY_TIERS, includes 404)
        d_pp = policy_pair(d_row)
        e_pp = policy_pair(e_row)
        if d_pp is not None and e_pp is not None:
            n_policy += 1
            if d_pp[0] != e_pp[0]:
                n_3tuple_diff += 1

        # Sitemap-refetch comparison (SITEMAP_TIERS, excludes 404)
        d_sp = sitemap_pair(d_row)
        e_sp = sitemap_pair(e_row)
        if d_sp is not None and e_sp is not None:
            n_sitemap += 1
            d_dg, d_ts = d_sp
            e_dg, e_ts = e_sp
            if d_dg != e_dg:
                n_sitemap_4_diff += 1
            elif d_ts != e_ts:
                n_sitemap_ts_only += 1
                n_sitemap_4_diff += 1

    print("=== 3-tuple policy differ (Tier 0/1 treatment change) ===")
    print(f"  denominator: cells with tier in POLICY_TIERS on both sides")
    print(f"               ({sorted(POLICY_TIERS)})")
    print(f"  comparable cells:        {n_policy:,}")
    if n_policy:
        print(f"  digest changed:          "
              f"{n_3tuple_diff:,}  ({100*n_3tuple_diff/n_policy:.2f}%)")
    print()

    print("=== 4-tuple sitemap-refetch differ (Step 6.5 refetch pressure) ===")
    print(f"  denominator: cells with tier in SITEMAP_TIERS on both sides")
    print(f"               ({sorted(SITEMAP_TIERS)})")
    print(f"               (ok_404_allow_all excluded: no sitemap to refetch)")
    print(f"  comparable cells:                  {n_sitemap:,}")
    if n_sitemap:
        print(f"  ts-only differ (same digest):    "
              f"{n_sitemap_ts_only:,}  ({100*n_sitemap_ts_only/n_sitemap:.2f}%)")
        print(f"  4-tuple differ (digest OR ts):   "
              f"{n_sitemap_4_diff:,}  ({100*n_sitemap_4_diff/n_sitemap:.2f}%)")
        print(f"  NOTE: this is an UPPER BOUND on refetch pressure. The true")
        print(f"        refetch number further requires the digest to contain")
        print(f"        at least one Sitemap: declaration (extracted by Step 6")
        print(f"        from robots_content.json). Filter not applied here.")
    print()

    # === Tier status flow matrix ===
    print("=== Tier-status flow (day-15 -> EOM, top 15) ===")
    sorted_flow = sorted(flow.items(), key=lambda kv: -kv[1])
    for (d_t, e_t), c in sorted_flow[:15]:
        marker = " (same)" if d_t == e_t else ""
        print(f"  {d_t:<24} -> {e_t:<24} {c:>10,}{marker}")
    if len(sorted_flow) > 15:
        rest = sum(c for _, c in sorted_flow[15:])
        print(f"  ... ({len(sorted_flow) - 15} more transitions, total {rest:,})")
    print()

    # === Revisit forward-resolved (EOM only) ===
    n_eom_revisit = 0
    n_eom_forward = 0
    for r in eom.values():
        if r["tier_status"] == "ok_revisit_resolved":
            n_eom_revisit += 1
            try:
                if int(r.get("revisit_source_after_timestamp", "0") or "0") == 1:
                    n_eom_forward += 1
            except ValueError:
                pass
    print("=== EOM revisit diagnostics ===")
    print(f"  ok_revisit_resolved cells:                       {n_eom_revisit:,}")
    print(f"  forward-resolved (revisit_source > revisit ts):  {n_eom_forward:,}")
    if n_eom_revisit:
        print(f"    forward-resolved share:                        "
              f"{100*n_eom_forward/n_eom_revisit:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
