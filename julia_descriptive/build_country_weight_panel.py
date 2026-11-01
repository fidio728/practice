# ============================================================================
# build_country_weight_panel.py — TASK e1: the country-level PORTFOLIO-WEIGHT
# panel + the identity hard gate. Advisor realignment after the 2026-08-04
# meeting. CODE ONLY at authoring time; this file runs nothing on import.
#
# v3.1 PROVENANCE. Reads the CURRENT canonical v3.1 artifacts:
#   output/merged_us_eu_zero_filled.parquet   06_cartesian_grid.jl  (2026-08-09 18:20)
#   output/I_ict_panel.parquet                04_us_ownership_european.jl, post-DQ
#   output/country_total_ct.parquet           04, both denominators
#   output/holdings_eom.parquet               03_eom_etl.jl, the row-level source
#   output/country_measures_m1m2m3.csv        build_country_measures.py, v3.1
# Freshness gates refuse pre-v3.1 vintages, and an ORDERING gate refuses a
# holdings_eom that is newer than the I_ict panel it is supposed to feed (in
# that case re-deriving I_ict from it could not reproduce canonical values).
#
# ============================== WHY THIS EXISTS =============================
# DO NOT RE-LITIGATE. A first-pass country panel (run_country_panel.do,
# country_panel_step4.csv) used dlog(US dollar holdings) and its US-minus-NONUS
# difference and produced M1 diff b = -0.695, p_crve = .024 / p_wild = .060, and
# M2 diff p_crve = .104 / p_wild = .025. Those are NOT the estimand the advisors
# asked for. Stefano, 2026-08-04 meeting at 16:00: "these are WEIGHTS of
# portfolios of US institutions". A dlog dollar outcome mixes price, quantity
# and composition changes, and the US-minus-NONUS difference does NOT net prices
# out, because the two groups hold different firms with different weights. This
# file realigns the country outcome onto the portfolio-weight estimand.
#
# ============================ LOCKED SPECIFICATION ==========================
# (decided by the researcher; implemented exactly, nothing improvised)
#
# OUTCOME.  W_{c,g,t} = sum_{i in c} w_{i,g,t} = (sum_{i in c} I_{i,g,t}) / T_{g,t}
#           = country c's share of investor group g's GLOBAL equity book.
#           dW_{c,g,t} = W_{c,g,t} - W_{c,g,t-1}.
#   MAIN          : global denominator (portfolio_weight_global)  -> dW_global
#   DIAGNOSTIC    : EU denominator (portfolio_weight_eu) -> dW_eu, labelled the
#                   WITHIN-EUROPE REALLOCATION diagnostic, never the headline
#   SUPPLEMENTARY : dlog of dollar holdings -> dlog_usd. A DIFFERENT ESTIMAND.
#                   It is NOT a net flow and must never be described as one.
#
# IDENTITY HARD GATE. dW must equal sum_{i in c} delta_w_global_{i,g,t} cell by
# cell. Built BOTH ways and gated — see section [6].
#
# REGRESSION (run later, not in this file).
#   dW_{c,g,t} = beta * US_g x M_{k,c,t-1} x S_{t-1} + US_g x M_{k,c,t-1}
#                + alpha_{cg} + alpha_{ct} + alpha_{gt} + e
#   two-way cluster (country, quarter). The three pairwise FE are the exact
#   country-level analogue of the firm-level fq / gq / ig. Absorption: alpha_ct
#   kills anything varying only by (c,t), INCLUDING the M_{k,c,t-1} main effect;
#   alpha_gt kills S_{t-1}; US x M and the triple survive. k in {M1, M2, M3}.
#
#   MAIN HYPOTHESIS FAMILY = EXACTLY THREE: {M1, M2, M3} on global dW under this
#   DDD. CRVE, score bootstrap and randomization inference are three INFERENCE
#   METHODS for those same three hypotheses, NOT additional hypotheses. The EU
#   denominator, the US-only (non-differenced) outcome and the dlog outcome are
#   SECONDARY / diagnostic and must not be folded into the main family.
#
# FIRM-LEVEL REFERENCE DESIGN this must mirror
# (output/headline_3pairwise_canonical.csv, cell spec=primary/global/slag/fq_gq_ig):
#   dw = delta_w_global on us_cn = us*cn_lag and us_cn_slag = us*cn_lag*s_lag,
#   absorb(fq gq ig) = firm x quarter, group x quarter, firm x group;
#   vce(cluster firm_n rd_m). b3 = -1.394461e-07, se = 1.032294e-06,
#   p = .892881, N = 909,638. MDE(80%, 5%) = 2.89e-06.
#
# =========================== WHAT THIS FILE DOES ============================
#   [1] freshness, ordering and rotation gates
#   [1b] PROVENANCE: SHA256 of this script and of every input artifact, printed
#       and censused into country_weight_gate_summary.csv
#   [2] firm-level grid integrity (complete cartesian panel => LAG == calendar lag)
#   [3] exact reconstruction of the canonical DQ filter from holdings_eom, and of
#       the outlier-excluded proxy-free twin                    [sensitivity (b)]
#   [4] firm-level twin weights w', w'_prev, delta w'
#   [5] country aggregation: W, dW BOTH ways, dollars, dlog, decomposition
#   [5b] COMMON-FIRM-SET M1/M3 for sensitivity (c): M1 and M3 RE-COMPUTED on the
#       market-cap-covered firms inside each country-quarter — the firm set M2 is
#       already built on — so the three measures can be compared on one FIRM set
#       rather than merely on one CELL set
#   [6] THE IDENTITY HARD GATE     -> output/country_weight_identity_gate.csv
#       (FAIL-CLOSED on the main sample: any partial-NULL cell aborts the build)
#   [7] M1/M2/M3 stamped at t-1 + shock / s_lag; m{1,2,3}cov_lag stamped the same
#   [8] write output/country_weight_panel.dta (columns sec_country / holder_group
#       / rdate / dw_global / dw_eu / ... — ALL LOWER CASE, the contract every e2
#       consumer reads), output/firm_wls_weights.dta (the named dependency of the
#       firm-level WLS diagnostic) and the audit CSVs
#
# ============================ DELIBERATELY NOT HERE =========================
#   * the regression, the score bootstrap, and the CIRCULAR-SHIFT RANDOMIZATION
#     INFERENCE ARBITER (max-|t| FWER across the three M statistics). That
#     arbiter must RE-ESTIMATE THE FULL FE MODEL on every draw — no collapse
#     shortcut unless algebraic equivalence is proved and self-checked to 1e-8 —
#     and its output must DISCLOSE that circular shift assumes the shock series
#     is exchangeable / approximately stationary under rotation.
#   * SENSITIVITY (a) leave-one-country-out: a reporting-side loop over this
#     panel. Report the coefficient range and the largest single-country change
#     for each M. NEVER drop a country and report the remainder as the headline.
#   * SENSITIVITY (c) M2 common coverage: M2's market-cap coverage is
#     incomplete, so M1 and M3 must ALSO be judged on the SAME FIRM SET M2 uses.
#     Section [7b] RE-COMPUTES M1 and M3 on the market-cap-covered firms inside
#     each country-quarter (m1cov_lag / m2cov_lag / m3cov_lag). The old
#     country-quarter flag m2_cov is still written, but it is a CELL-set
#     restriction only and is labelled as such — see the long note in [7b].
#   * the FIRM-LEVEL WLS run itself, which comes LAST and is a HETEROGENEITY
#     DIAGNOSTIC ONLY — "do larger positions respond more?" (its WEIGHTS are
#     emitted here, in firm_wls_weights.dta, because the audit panel does not
#     carry a lagged level column). It is NOT a reconciliation
#     test and must not be described as reproducing or adjudicating the country
#     result. The withdrawn claim "WLS reproduces the country result" is FALSE:
#     country aggregation is a SUM of firm deltas while WLS minimises a weighted
#     SSR, and equivalence needs conditions this design does not satisfy.
#     Lagged-holdings weights additionally (i) DOUBLE-COUNT SIZE, because w is
#     already size-scaled, and (ii) assign ZERO WEIGHT TO NEW POSITIONS,
#     discarding the entry margin. That is precisely why the DECOMPOSITION in
#     section [5] replaces it as the answer to "what carries the movement".
#
# RUNTIME. Two selective passes plus one quarter-end pass over the 9.3 GB
# holdings_eom.parquet (186.8M rows), then small in-memory work. ~8-15 min,
# plus the PROVENANCE HASH pass added 2026-08-10 (SHA256 of this file and of
# every input, including the 9.3 GB parquet — streamed, add a few minutes on a
# cold cache). duckdb is pinned to temp_directory='E:/duckdb_tmp',
# memory_limit=12GB, threads=4 (C: has ~15 GB free only). No network. No retries.
# ============================================================================

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# [0] Paths, constants, thresholds
# ---------------------------------------------------------------------------
PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
_env_out = os.environ.get("DPN_OUT_DIR", "").strip()
OUT = Path(_env_out).resolve() if _env_out else PROJ / "output"

GRID_P = OUT / "merged_us_eu_zero_filled.parquet"
ICT_P  = OUT / "I_ict_panel.parquet"
ICT_M  = OUT / "I_ict_panel.parquet.meta.json"
CTOT_P = OUT / "country_total_ct.parquet"
EOM_P  = OUT / "holdings_eom.parquet"
MEAS_P = OUT / "country_measures_m1m2m3.csv"
# marketcap_it.parquet is 04's PIT market cap. It is the ONLY thing that decides
# which firms M2 can be computed on, so it is also what defines the COMMON FIRM
# SET used by sensitivity (c) — see [7b].
MCAP_P = OUT / "marketcap_it.parquet"
# LIVING ANCHOR for the DQ residual census (see DQ_RESID note below): the
# per-version drop census written by robustness_dq_filter_variants.py.
DQ_CENSUS = OUT / "dq_variants" / "dq_census.csv"

DTA       = OUT / "country_weight_panel.dta"
F_WLSW    = OUT / "firm_wls_weights.dta"
F_GATE    = OUT / "country_weight_identity_gate.csv"
F_SUMMARY = OUT / "country_weight_gate_summary.csv"
F_OUTLIER = OUT / "country_weight_outlier_rows.csv"
F_DECOMP  = OUT / "country_weight_decomposition.csv"
TARGETS   = (DTA, F_WLSW, F_GATE, F_SUMMARY, F_OUTLIER, F_DECOMP)

# v3.1 canonical rebuild window opened 2026-08-09 17:00 (the same constant
# build_country_panel.py uses). holdings_eom is the 03 output and legitimately
# PREDATES that; it gets its own floor plus the ordering gate in [1].
VMIN     = datetime(2026, 8, 9, 17, 0, 0)
VMIN_EOM = datetime(2026, 8, 9, 0, 0, 0)

# 28 European jurisdictions, verbatim from 00_setup.jl / build_country_panel.py.
EU_COUNTRIES = ("GB", "DE", "FR", "NL", "CH", "IT", "ES", "SE", "DK", "NO",
                "FI", "BE", "AT", "IE", "LU", "PT", "PL", "CZ", "HU", "GR",
                "RO", "SK", "SI", "BG", "HR", "EE", "LV", "LT")
EU_SQL = "(" + ",".join(f"'{c}'" for c in EU_COUNTRIES) + ")"

Q_FIRST, Q_LAST, N_Q = "1999Q1", "2023Q4", 100      # frozen v3.1 span
N_GROUPS = 2
COUNTRY_UNIVERSE = "ownership_matched"              # is_primary = 1 in the measures CSV

# ---- DQ census constants (vintage-drift gates) -----------------------------
# 04_us_ownership_european.jl "DQ-FIX" census, 2026-08-09: exactly 3 rows,
# $452.6bn, each failing BOTH impossibility tests.
DQ_V1_N_EXPECTED   = 3
DQ_V1_USD_EXPECTED = 452.6e9
#
# THE RESIDUAL R = D2 \ D1 IS READ AT RUN TIME, NOT HARDCODED (fix, 2026-08-10).
# An earlier vintage of this file carried DQ_RESID_N_EXPECTED = 373 and enforced
# it as a hard gate. That constant is STALE: against the current v3.1 data the
# census in output/dq_variants/dq_census.csv gives V2 = 304 rows / $483.0105bn
# and V1 = 3 rows / $452.626bn, hence R = 301 rows / $30.38bn. The old gate would
# have compared 301 against 373 and killed the run before anything was written.
# The rule this project already applies everywhere else is used instead: read the
# LIVING anchor and gate against it, so a genuine upstream drift still aborts
# while a stale memo cannot. Every downstream number that used to be quoted from
# the memo ($47.0bn, 196 holder-country-quarters, "Indonesia 2017Q4 = 65.9%") is
# likewise RE-DERIVED below from `outlier_rows`, never reprinted from prose.
DQ_CENSUS_V1 = "V1"      # shipped DQ-FIX leg in dq_census.csv
DQ_CENSUS_V2 = "V2"      # proxy-free leg (adj_holding > adj_shares_out, any size)


def read_dq_residual_anchor() -> tuple[int, float]:
    """R = V2 \\ V1 from the living census. Returns (n_rows, usd)."""
    if not DQ_CENSUS.is_file():
        die(f"missing DQ census anchor {DQ_CENSUS}. It is written by "
            f"robustness_dq_filter_variants.py and is the living anchor for the "
            f"residual gate; refusing to fall back to a hardcoded constant.")
    c = pd.read_csv(DQ_CENSUS)
    need = {"version", "n_rows_dropped", "usd_dropped"}
    if not need.issubset(c.columns):
        die(f"{DQ_CENSUS.name} lacks {sorted(need - set(c.columns))} — stale layout.")
    agg = c.groupby("version")[["n_rows_dropped", "usd_dropped"]].sum()
    for v in (DQ_CENSUS_V1, DQ_CENSUS_V2):
        if v not in agg.index:
            die(f"{DQ_CENSUS.name} has no {v} rows — cannot derive the residual.")
    n = int(agg.loc[DQ_CENSUS_V2, "n_rows_dropped"]
            - agg.loc[DQ_CENSUS_V1, "n_rows_dropped"])
    usd = float(agg.loc[DQ_CENSUS_V2, "usd_dropped"]
                - agg.loc[DQ_CENSUS_V1, "usd_dropped"])
    n_v1 = int(agg.loc[DQ_CENSUS_V1, "n_rows_dropped"])
    if n_v1 != DQ_V1_N_EXPECTED:
        die(f"{DQ_CENSUS.name} says V1 drops {n_v1} rows but the 04 DQ-FIX census "
            f"is {DQ_V1_N_EXPECTED} — the census and the shipped filter are from "
            f"different vintages. Resolve upstream before aggregating.")
    print(f"    DQ residual anchor (live, from {DQ_CENSUS.name}): "
          f"V2 {int(agg.loc[DQ_CENSUS_V2,'n_rows_dropped'])} rows / "
          f"${float(agg.loc[DQ_CENSUS_V2,'usd_dropped'])/1e9:,.4f}bn minus "
          f"V1 {n_v1} rows / ${float(agg.loc[DQ_CENSUS_V1,'usd_dropped'])/1e9:,.4f}bn "
          f"=> R = {n} rows / ${usd/1e9:,.4f}bn")
    return n, usd

# ---- numerical thresholds --------------------------------------------------
TWO_53 = 9007199254740992.0     # 2**53; integers below this are exact in float64

# Identity-gate tolerances. W is a sum of ~10^2-10^3 firm weights of magnitude
# <= ~0.3, so the naive-summation error bound n*eps*sum|w| is ~1e-13. The hard
# gate sits two orders above that and five orders BELOW the firm-level
# MDE(80%,5%) = 2.89e-06, so any discrepancy that passes cannot move a reported
# coefficient. Both the absolute and the scale-relative gate must pass.
TOL_IDENT_ABS  = 1e-11
TOL_IDENT_ADV  = 1e-13          # advisory only: printed, never aborts
TOL_IDENT_REL  = 1e-06          # max|disc| / p90(|dW|) within the family
TOL_DECOMP_ABS = 1e-11          # each partition must re-sum to dW
EPS_DIFFER     = 1e-15          # "cells where they differ" census threshold

# Gates on the dollar chain are BIT-exactness gates, not tolerance gates.
BITEXACT = 0

# ---- the ONE size-split tie convention (see the [5] doc block) -------------
# run_country_decomposition.py carries the SAME literal and must keep it in
# sync; a reader comparing the two artifacts reads this string, not the SQL.
SIZE_TIE_RULE = "large = wp > cell_median(wp | wp>0); small = 0 < wp <= median; new = wp == 0"


# ---------------------------------------------------------------------------
# gate plumbing
# ---------------------------------------------------------------------------
GATES: list[dict] = []


def die(msg: str) -> None:
    raise RuntimeError(msg)


def _record(sev: str, name: str, family: str, stat: str, value, threshold,
            ok: bool, note: str) -> None:
    GATES.append({"severity": sev, "gate": name, "family": family,
                  "statistic": stat, "value": value, "threshold": threshold,
                  "pass": int(bool(ok)), "note": note})


def gate(name: str, family: str, stat: str, value, threshold, ok: bool,
         note: str = "") -> None:
    """Hard gate: records the result and aborts immediately when it fails."""
    _record("gate", name, family, stat, value, threshold, ok, note)
    print(f"    [{'PASS' if ok else 'FAIL'}] {name} ({family}): {stat} = {value} "
          f"| threshold {threshold} | {note}")
    if not ok:
        die(f"GATE FAILED: {name} ({family}) — {stat} = {value}, threshold "
            f"{threshold}. {note}\nNothing has been written.")


def advisory(name: str, family: str, stat: str, value, threshold, ok: bool,
             note: str = "") -> None:
    """Soft gate: recorded and printed, never aborts."""
    _record("advisory", name, family, stat, value, threshold, ok, note)
    tag = "ok  " if ok else "WARN"
    print(f"    [{tag}] {name} ({family}): {stat} = {value} | advisory "
          f"threshold {threshold} | {note}")


def census(name: str, family: str, stat: str, value, note: str = "") -> None:
    """Reported fact, no threshold. On the record, never a pass/fail."""
    _record("census", name, family, stat, value, "reported", True, note)
    print(f"    [cens] {name} ({family}): {stat} = {value} | {note}")


# ---------------------------------------------------------------------------
# PROVENANCE HASHING (external-review item, 2026-08-10)
# ---------------------------------------------------------------------------
# A run must be tie-able to the EXACT BYTES that produced it. mtimes cannot do
# that here — OneDrive rewrites them on sync, and the junction to E: means the
# same artifact can carry two different timestamps. So every run prints the
# SHA256 of THIS SCRIPT and of every input artifact it reads, and the same
# hashes are censused into country_weight_gate_summary.csv, which travels with
# the numbers. Hashing is STREAMED (holdings_eom.parquet is 9.3 GB).
SELF_PATH = Path(__file__).resolve()
HASH_CHUNK = 1 << 22          # 4 MiB


def sha256_file(p: Path) -> tuple[str, int, float]:
    """(sha256_hex, size_bytes, seconds). 'MISSING' when the file is absent."""
    p = Path(p)
    if not p.is_file():
        return "MISSING", 0, 0.0
    t0 = time.time()
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(HASH_CHUNK), b""):
            h.update(blk)
            n += len(blk)
    return h.hexdigest(), n, time.time() - t0


def log_provenance(inputs: dict[str, Path]) -> None:
    """Print + census the script hash and every input hash."""
    print("[0b] PROVENANCE — SHA256 of this script and of every input artifact")
    dg, sz, el = sha256_file(SELF_PATH)
    print(f"    SCRIPT  {SELF_PATH.name:<38s} {dg}  ({sz:,} B, {el:.1f}s)")
    census("provenance_script_sha256", "provenance", SELF_PATH.name, dg,
           f"{sz:,} bytes; the code that produced every number below")
    for label, p in inputs.items():
        dg, sz, el = sha256_file(p)
        print(f"    INPUT   {label:<38s} {dg}  ({sz:,} B, {el:.1f}s)")
        census("provenance_input_sha256", "provenance", label, dg,
               f"{sz:,} bytes from {Path(p).as_posix()}")


def assert_fresh(p: Path, label: str, floor: datetime) -> datetime:
    if not p.is_file():
        raise FileNotFoundError(f"missing input {label}: {p}")
    mt = datetime.fromtimestamp(p.stat().st_mtime)
    if mt < floor:
        die(f"STALE INPUT {label}: mtime {mt:%Y-%m-%d %H:%M:%S} < "
            f"{floor:%Y-%m-%d %H:%M} — pre-v3.1 vintage. Rebuild upstream first.")
    return mt


def rotation_guard() -> None:
    """REFUSE to overwrite a canonical artifact in place.

    Project rule: rename the existing file to *_cwpre first; refuse outright if
    that rotation slot is already occupied. Mirrors build_country_panel.py's
    guard (which uses *_r2pre for its own family of outputs).
    """
    existing = [p for p in TARGETS if p.exists()]
    if not existing:
        return
    lines = []
    for p in existing:
        pre = p.with_name(f"{p.stem}_cwpre{p.suffix}")
        if pre.exists():
            lines.append(f"  {p.name}: EXISTS, and its rotation slot {pre.name} "
                         f"is ALSO occupied — archive {pre.name} out of output/ "
                         f"before rotating again")
        else:
            lines.append(f"  {p.name}: EXISTS — rename it to {pre.name} first")
    die("ROTATION GUARD: refusing to overwrite canonical output(s) in place.\n"
        + "\n".join(lines) + "\nRe-run after rotating. Nothing has been written.")


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET memory_limit='12GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    tmp = Path("E:/duckdb_tmp")
    if not tmp.parent.exists():
        die("E: is unavailable, but duckdb MUST spill to E:/duckdb_tmp "
            "(C: has ~15 GB free only). Refusing to run against a C: temp dir.")
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    return con


def one(con: duckdb.DuckDBPyConnection, sql: str) -> pd.Series:
    return con.sql(sql).df().iloc[0]


# ===========================================================================
# [3] SENSITIVITY (b): THE OUTLIER-EXCLUSION METHOD, AND WHY IT IS EXACT
# ===========================================================================
# THE PROBLEM. The merged panel carries only firm x group x quarter weights. The
# offending rows live at the FUND x SECURITY x MONTH grain of holdings_eom, and
# the merged panel holds no fund / security key to join back on. So the twin
# CANNOT be produced by filtering the panel.
#
# THE METHOD. Every aggregate in the chain is a SUM of the SAME atom, adj_mv,
# over a subset of the SAME row set:
#
#   base = holdings_eom rows with sec_entity_id IS NOT NULL
#          AND investor_country IS NOT NULL AND issue_type IN ('EQ','AD')
#          — the identified EQ/AD universe I_ict is built on. The NULL-entity /
#          non-EQ sentinel families never enter ANY version, exactly as they
#          never enter I_ict.
#   D1   = the shipped DQ-FIX drop set (04, "V1"):
#            adj_mv > $5bn AND (adj_mv > inline market cap
#                               OR adj_holding > adj_shares_out)
#   D2   = the PROXY-FREE leg ("V2"): adj_holding > adj_shares_out AND
#          adj_shares_out > 0, at ANY size. NULL-SAFE: a row with a NULL operand
#          is KEPT, identical to the shipped anti-join, where a NULL predicate
#          keeps the row out of dq_impossible.
#   R    = D2 \ D1 — the INCREMENTAL rows, i.e. the ones still inside the
#          canonical panel. The expected count is READ AT RUN TIME from
#          output/dq_variants/dq_census.csv (see read_dq_residual_anchor), NOT
#          hardcoded; on the current v3.1 data that anchor is 301 rows.
#
#   canonical : I(k)  = SUM_{r in base\D1,      key(r)=k} adj_mv
#   twin      : I'(k) = SUM_{r in base\(D1 u D2), key(r)=k} adj_mv
#   and because R is disjoint from D1 and contained in base\D1,
#               I'(k) = I(k) - dI(k),   dI(k) = SUM_{r in R, key(r)=k} adj_mv
#   identically for the group books,
#               T'(g,t) = T(g,t) - dT(g,t).
#   dI is taken over GRID-UNIVERSE firms only and mapped to the GRID's canonical
#   sec_country — mirroring 06's ict_grouped, which restricts on sec_entity_id
#   and takes the country from the grid, never from the row. dT is taken over
#   ALL entities and ALL sec_country, because T is the group's FULL book. The
#   twin weight is then RE-FORMED as w' = I'/T'. BOTH legs move: excluding a row
#   shrinks the numerator only when the firm is in c, but shrinks the
#   denominator always. Subtracting from the numerator alone would be WRONG.
#
# WHY IT IS EXACT (not "exact to tolerance").
#   (P1) adj_mv is BIGINT in holdings_eom, so every quantity above is an INTEGER,
#        and duckdb accumulates SUM(BIGINT) in HUGEINT (int128) — exact integer
#        arithmetic, before any float exists.
#   (P2) Every integer of magnitude < 2**53 is exactly representable in IEEE-754
#        binary64, and IEEE-754 + and - are correctly rounded: when the exact
#        result is representable the operation is EXACT, with zero error.
#   (P3) This script GATES AT RUNTIME that every quantity in the chain satisfies
#        SUM(ABS(adj_mv)) < 2**53 inside its aggregation group. Because that
#        bounds the absolute value of every PARTIAL sum whatever the summation
#        order, the sums are exact AND order-independent — which also makes them
#        reproducible across duckdb thread counts and across re-runs.
#   (P4) Hence I' = I - dI and T' = T - dT are EXACT identities in float64. The
#        only inexact operation in the entire twin is the final division
#        w' = I'/T': one correctly-rounded division, the same operation with the
#        same error the canonical pipeline performs for w.
#   (P5) EMPIRICAL CONFIRMATION, not merely algebra. The script
#        (i)   re-derives the canonical I_ict from the raw rows under its own
#              reconstruction of D1, and requires it to be BIT-IDENTICAL to
#              I_ict_panel.parquet on every cell (no extra cells, no missing
#              cells, zero differing values);
#        (ii)  requires its reconstructed group books to be BIT-IDENTICAL to the
#              ones country_total_ct.parquet implies under 06's grouping;
#        (iii) requires the canonical weights in the merged panel to satisfy
#              w == I/T BIT-IDENTICALLY;
#        (iv)  computes I' BOTH directly (SUM over base\(D1 u D2)) and as
#              I - dI, and requires the two to agree BIT-IDENTICALLY, and the
#              same for T'.
#        If (i)-(iii) hold, the reconstruction provably IS the canonical
#        pipeline; (iv) then makes the twin the canonical pipeline minus R.
#   The D1 reconstruction needs the inline market-cap proxy. It is replicated
#   VERBATIM from 04 (primary-EQ share classes, fsym_id = fsym_primary_id,
#   issue_type = 'EQ', adj_shares_out > 0, adj_price > 0, AVG within
#   (sec_entity_id, report_date, fsym_id), SUM across classes, restricted to the
#   candidate entities, and WITHOUT the EM-FIX-2 freshest-gap restriction) and is
#   then checked against the published census: exactly 3 rows, $452.6bn.
# ===========================================================================
def build_dq_sets(con: duckdb.DuckDBPyConnection, qdates_sql: str) -> None:
    print("[3] Reconstructing the canonical DQ filter from holdings_eom...")

    # D1 candidates: the > $5bn leg is extremely selective, so this scan prunes
    # on parquet row-group statistics.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cand AS
        SELECT fund_id, fsym_id, sec_entity_id, report_date,
               adj_mv, adj_holding, adj_shares_out
        FROM read_parquet('{EOM_P.as_posix()}')
        WHERE sec_entity_id IS NOT NULL AND investor_country IS NOT NULL
          AND issue_type IN ('EQ', 'AD') AND adj_mv > 5e9
    """)
    n_cand = int(one(con, "SELECT COUNT(*) AS n FROM cand")["n"])
    print(f"    D1 candidate rows (adj_mv > $5bn, identified EQ/AD): {n_cand:,}")

    # inline market-cap proxy, VERBATIM from 04 (candidate restriction included)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE mc AS
        SELECT cls.sec_entity_id, cls.report_date,
               SUM(cls.shares_out * cls.price) AS market_cap
        FROM (
            SELECT sec_entity_id, report_date, fsym_id,
                   AVG(adj_shares_out) AS shares_out, AVG(adj_price) AS price
            FROM read_parquet('{EOM_P.as_posix()}')
            WHERE fsym_id = fsym_primary_id AND issue_type = 'EQ'
              AND adj_shares_out > 0 AND adj_price > 0
              AND sec_entity_id IN (SELECT DISTINCT sec_entity_id FROM cand)
            GROUP BY 1, 2, 3
        ) cls
        GROUP BY 1, 2
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE dq_v1 AS
        SELECT c.fund_id, c.fsym_id, c.sec_entity_id, c.report_date, c.adj_mv
        FROM cand c LEFT JOIN mc USING (sec_entity_id, report_date)
        WHERE (mc.market_cap IS NOT NULL AND c.adj_mv > mc.market_cap)
           OR (c.adj_holding > c.adj_shares_out AND c.adj_shares_out > 0)
    """)
    v1 = one(con, "SELECT COUNT(*) AS n, "
                  "CAST(COALESCE(SUM(adj_mv), 0) AS DOUBLE) AS usd FROM dq_v1")
    n_v1, usd_v1 = int(v1["n"]), float(v1["usd"])
    print(f"    D1 (shipped DQ-FIX) reconstructed: {n_v1} rows, ${usd_v1/1e9:,.1f}bn")

    # DRIFT GATE. Every number below is a delta against the canonical panel, so a
    # drifted D1 would silently compare two different vintages.
    gate("dq_v1_row_census", "outlier", "n_rows_D1", n_v1, DQ_V1_N_EXPECTED,
         n_v1 == DQ_V1_N_EXPECTED,
         "must reproduce the 2026-08-09 04 DQ-FIX census exactly")
    gate("dq_v1_usd_census", "outlier", "usd_bn_D1", round(usd_v1 / 1e9, 1),
         round(DQ_V1_USD_EXPECTED / 1e9, 1),
         abs(usd_v1 - DQ_V1_USD_EXPECTED) < 0.05e9,
         "dollars dropped must match the 04 census at its printed 0.1bn rounding")

    # 04's anti-join matches on (fund_id, fsym_id, report_date). If that triple
    # were not unique, the anti-join would remove MORE rows than dq_v1 contains
    # and the row-level reconstruction below would be wrong. 04 hard-fails on
    # global duplicates; only the D1 keys can matter here, so check exactly those.
    dupk = int(one(con, f"""
        SELECT COUNT(*) AS n FROM (
            SELECT h.fund_id, h.fsym_id, h.report_date
            FROM read_parquet('{EOM_P.as_posix()}') h
            JOIN dq_v1 d ON d.fund_id = h.fund_id AND d.fsym_id = h.fsym_id
                        AND d.report_date = h.report_date
            GROUP BY 1, 2, 3 HAVING COUNT(*) > 1)
    """)["n"])
    gate("dq_v1_key_uniqueness", "outlier", "n_duplicated_D1_keys", dupk, 0,
         dupk == 0, "04's anti-join key must select exactly the D1 rows")

    # base view at the panel's quarter-end dates, carrying both drop flags
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW base AS
        SELECT h.sec_entity_id, h.sec_country, h.investor_country, h.report_date,
               h.adj_mv, h.adj_holding, h.adj_shares_out, h.fund_id, h.fsym_id,
               CASE WHEN h.investor_country = 'US' THEN 'US' ELSE 'NONUS' END AS hg,
               (d.fund_id IS NOT NULL) AS drop_d1,
               COALESCE(h.adj_holding > h.adj_shares_out
                        AND h.adj_shares_out > 0, FALSE) AS drop_d2
        FROM read_parquet('{EOM_P.as_posix()}') h
        LEFT JOIN dq_v1 d ON d.fund_id = h.fund_id AND d.fsym_id = h.fsym_id
                         AND d.report_date = h.report_date
        WHERE h.sec_entity_id    IS NOT NULL
          AND h.investor_country IS NOT NULL
          AND h.issue_type IN ('EQ', 'AD')
          AND h.report_date IN {qdates_sql}
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE outlier_rows AS
        SELECT fund_id, fsym_id, sec_entity_id, sec_country, investor_country,
               hg, report_date, adj_mv, adj_holding, adj_shares_out,
               CASE WHEN adj_shares_out > 0
                    THEN adj_holding::DOUBLE / adj_shares_out END AS holding_over_shares
        FROM base WHERE drop_d2 AND NOT drop_d1
    """)
    r = one(con, """SELECT COUNT(*) AS n,
                           CAST(COALESCE(SUM(adj_mv), 0) AS DOUBLE) AS usd,
                           COUNT(DISTINCT investor_country || '|'
                                 || report_date::VARCHAR) AS n_cq
                    FROM outlier_rows""")
    n_r, usd_r, n_cq = int(r["n"]), float(r["usd"]), int(r["n_cq"])
    print(f"    R = D2 \\ D1 (incremental, proxy-free leg): {n_r} rows, "
          f"${usd_r/1e9:,.4f}bn over {n_cq} holder-country-quarters")
    n_exp, usd_exp = read_dq_residual_anchor()
    gate("dq_residual_census", "outlier", "n_rows_R", n_r, n_exp, n_r == n_exp,
         "must reproduce the LIVE dq_census.csv residual V2 \\ V1 (not a "
         "hardcoded memo constant)")
    advisory("dq_residual_usd", "outlier", "usd_bn_R", round(usd_r / 1e9, 4),
             round(usd_exp / 1e9, 4), abs(usd_r - usd_exp) < 0.01e9,
             "dollars in R re-derived here vs the live census; a gap means this "
             "script's quarter-end restriction excludes rows the census counts")
    census("dq_residual_cells", "outlier", "n_holder_country_quarters_touched",
           n_cq, "RE-DERIVED from outlier_rows; supersedes the stale '196' memo")

    # The single most concentrated holder-country-quarter, RE-DERIVED. The old
    # prose quoted "Indonesia 2017Q4 = 65.9% of that holder-country cell" from a
    # memo; the share below is computed from this run's own rows.
    conc = con.sql("""
        WITH t AS (SELECT investor_country, report_date,
                          CAST(SUM(adj_mv) AS DOUBLE) AS usd
                   FROM outlier_rows GROUP BY 1, 2)
        SELECT investor_country, report_date, usd,
               usd / (SELECT SUM(usd) FROM t) AS share_of_R
        FROM t ORDER BY usd DESC LIMIT 1""").df()
    if len(conc):
        c0 = conc.iloc[0]
        census("dq_residual_concentration", "outlier",
               f"top_cell_{c0['investor_country']}_{c0['report_date']}",
               round(float(c0["share_of_R"]), 4),
               f"${float(c0['usd'])/1e9:,.4f}bn = this share of ALL residual "
               f"dollars; country aggregation amplifies it (sensitivity (b))")

    # D1 is expected to be NESTED inside D2 (all three shipped drops fail BOTH
    # tests). Not needed for correctness — R = D2 \ D1 is right either way — but
    # a violation is a fact the reader of sensitivity (b) must be told.
    n_d1_not_d2 = int(one(con, "SELECT COUNT(*) AS n FROM base "
                               "WHERE drop_d1 AND NOT drop_d2")["n"])
    census("dq_nesting_diagnostic", "outlier", "n_D1_rows_not_in_D2", n_d1_not_d2,
           "0 means the shipped rule is nested inside the proxy-free leg "
           "(counted at quarter-end dates only)")

    top = con.sql("""
        SELECT investor_country, report_date, COUNT(*) AS n_rows,
               CAST(SUM(adj_mv) AS DOUBLE) / 1e9 AS usd_bn
        FROM outlier_rows GROUP BY 1, 2 ORDER BY usd_bn DESC LIMIT 8""").df()
    print("    most affected holder-country-quarters (residual dollars, $bn):")
    print(top.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


def rebuild_ict(con: duckdb.DuckDBPyConnection) -> None:
    """One pass: canonical I_ict, the twin I', dI, and the exactness magnitudes."""
    print("[3b] Re-deriving I_ict (canonical + outlier-excluded twin)...")
    con.execute("""
        CREATE OR REPLACE TABLE ict AS
        SELECT sec_entity_id, sec_country, investor_country, hg, report_date,
               SUM(adj_mv)      FILTER (WHERE NOT drop_d1)                 AS i_can,
               SUM(adj_mv)      FILTER (WHERE NOT drop_d1 AND NOT drop_d2) AS i_x,
               SUM(adj_mv)      FILTER (WHERE drop_d2 AND NOT drop_d1)     AS d_i,
               SUM(ABS(adj_mv)) FILTER (WHERE NOT drop_d1)                 AS absmass,
               COUNT(*)         FILTER (WHERE drop_d2 AND NOT drop_d1)     AS n_r
        FROM base
        GROUP BY 1, 2, 3, 4, 5
        HAVING COUNT(*) FILTER (WHERE NOT drop_d1) > 0
    """)
    print(f"    reconstructed I_ict cells at quarter-ends: "
          f"{int(one(con, 'SELECT COUNT(*) AS n FROM ict')['n']):,}")

    # ---- (P3) magnitude gate: every partial sum stays below 2**53 ------------
    mx = float(one(con, "SELECT CAST(MAX(absmass) AS DOUBLE) AS m FROM ict")["m"])
    gate("f64_exactness_cell", "outlier", "max_cell_SUM_ABS(adj_mv)", mx, TWO_53,
         mx < TWO_53, "(P3) bounds every partial sum inside an I_ict cell")

    # ---- (P5)(i) BIT-IDENTITY against the canonical I_ict panel --------------
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE canon_ict AS
        SELECT sec_entity_id, sec_country, investor_country, report_date, I_ict
        FROM read_parquet('{ICT_P.as_posix()}')
        WHERE report_date IN (SELECT DISTINCT report_date FROM ict)
    """)
    cmpres = one(con, """
        WITH j AS (
            SELECT c.I_ict AS canon, CAST(r.i_can AS DOUBLE) AS recon
            FROM canon_ict c JOIN ict r
              ON  r.sec_entity_id    = c.sec_entity_id
              AND r.sec_country      = c.sec_country
              AND r.investor_country = c.investor_country
              AND r.report_date      = c.report_date
        )
        SELECT (SELECT COUNT(*) FROM canon_ict) AS n_canon,
               (SELECT COUNT(*) FROM ict)       AS n_recon,
               (SELECT COUNT(*) FROM j)         AS n_matched,
               (SELECT COUNT(*) FROM j WHERE canon <> recon)        AS n_value_diff,
               (SELECT COALESCE(MAX(ABS(canon - recon)), 0) FROM j) AS max_abs_diff
    """)
    n_canon, n_recon = int(cmpres["n_canon"]), int(cmpres["n_recon"])
    n_matched = int(cmpres["n_matched"])
    gate("ict_reconstruction_rowset", "outlier", "n_unmatched_cells",
         n_canon + n_recon - 2 * n_matched, 0,
         (n_canon == n_recon == n_matched),
         f"(P5)(i) recon {n_recon:,} vs canonical {n_canon:,} cells, "
         f"{n_matched:,} matched")
    gate("ict_reconstruction_bitexact", "outlier", "n_cells_differing",
         int(cmpres["n_value_diff"]), BITEXACT, int(cmpres["n_value_diff"]) == 0,
         f"(P5)(i) max abs diff {float(cmpres['max_abs_diff']):.3e}; the "
         f"row-level D1 predicate provably reproduces the shipped filter")

    n_bad_int = int(one(con, f"""
        SELECT COUNT(*) AS n FROM read_parquet('{ICT_P.as_posix()}')
        WHERE I_ict <> FLOOR(I_ict) OR ABS(I_ict) >= {TWO_53:.1f}""")["n"])
    gate("ict_integrality", "outlier", "n_noninteger_or_oversize_I_ict",
         n_bad_int, 0, n_bad_int == 0,
         "(P1)+(P2) every I_ict value is an integer strictly below 2**53")

    # ---- (P5)(iv) the numerator subtraction identity, bit for bit ------------
    sub = one(con, """
        SELECT COUNT(*) FILTER (
                 WHERE CAST(COALESCE(i_x, 0) AS DOUBLE)
                    <> CAST(i_can AS DOUBLE) - CAST(COALESCE(d_i, 0) AS DOUBLE)
               ) AS n_bad,
               CAST(COALESCE(MAX(ABS(CAST(COALESCE(i_x, 0) AS DOUBLE)
                    - (CAST(i_can AS DOUBLE)
                       - CAST(COALESCE(d_i, 0) AS DOUBLE)))), 0) AS DOUBLE) AS max_abs
        FROM ict""")
    gate("subtraction_identity_numerator", "outlier",
         "n_cells_where_I_x != I - dI", int(sub["n_bad"]), BITEXACT,
         int(sub["n_bad"]) == 0,
         f"(P4)+(P5)(iv) direct recomputation and subtraction agree bit for bit "
         f"(max abs {float(sub['max_abs']):.3e})")


def build_group_totals(con: duckdb.DuckDBPyConnection) -> None:
    """Group books T (canonical and twin), bit-checked against country_total_ct."""
    print("[3c] Group books T_{g,t} (global and EU-restricted), both regimes...")
    con.execute(f"""
        CREATE OR REPLACE TABLE gtot AS
        SELECT hg, report_date AS rd,
               CAST(SUM(i_can) AS DOUBLE)                                 AS t_g,
               CAST(SUM(COALESCE(i_x, 0)) AS DOUBLE)                      AS t_g_x,
               CAST(SUM(CASE WHEN sec_country IN {EU_SQL} THEN i_can ELSE 0 END)
                    AS DOUBLE)                                            AS t_e,
               CAST(SUM(CASE WHEN sec_country IN {EU_SQL}
                             THEN COALESCE(i_x, 0) ELSE 0 END) AS DOUBLE) AS t_e_x,
               CAST(SUM(ABS(i_can)) AS DOUBLE)                            AS t_absmass,
               CAST(SUM(COALESCE(d_i, 0)) AS DOUBLE)                      AS d_t_g,
               CAST(SUM(CASE WHEN sec_country IN {EU_SQL}
                             THEN COALESCE(d_i, 0) ELSE 0 END) AS DOUBLE) AS d_t_e
        FROM ict GROUP BY 1, 2
    """)
    m = float(one(con, "SELECT MAX(t_absmass) AS m FROM gtot")["m"])
    gate("f64_exactness_group_book", "outlier", "max_group_SUM_ABS(I_ict)", m,
         TWO_53, m < TWO_53, "(P3) bounds every partial sum inside a group book")

    # (P5)(ii) BIT-IDENTITY with 06's country_total_grouped construction. Note
    # the bracketing differs — canonical sums per investor_country and then over
    # the group, this file sums the group in one pass — and (P3) is exactly what
    # makes the two brackets identical rather than merely close.
    cmp2 = one(con, f"""
        WITH canon AS (
            SELECT CASE WHEN investor_country = 'US' THEN 'US' ELSE 'NONUS' END AS hg,
                   report_date AS rd,
                   SUM(COALESCE(country_total_holdings_global, 0)) AS t_g,
                   SUM(COALESCE(country_total_holdings_eu, 0))     AS t_e
            FROM read_parquet('{CTOT_P.as_posix()}') GROUP BY 1, 2)
        SELECT COUNT(*) FILTER (WHERE g.t_g <> c.t_g OR g.t_e <> c.t_e) AS n_bad,
               COALESCE(MAX(ABS(g.t_g - c.t_g)), 0) AS max_g,
               COALESCE(MAX(ABS(g.t_e - c.t_e)), 0) AS max_e,
               COUNT(*) AS n_cells
        FROM gtot g JOIN canon c USING (hg, rd)
    """)
    gate("group_total_bitexact", "outlier", "n_group_quarters_differing",
         int(cmp2["n_bad"]), BITEXACT, int(cmp2["n_bad"]) == 0,
         f"(P5)(ii) reconstruction == 06's country_total_grouped over "
         f"{int(cmp2['n_cells'])} cells (max |dT_g| {float(cmp2['max_g']):.3e}, "
         f"max |dT_eu| {float(cmp2['max_e']):.3e})")
    gate("group_total_coverage", "outlier", "n_matched_group_quarters",
         int(cmp2["n_cells"]), N_GROUPS * N_Q,
         int(cmp2["n_cells"]) == N_GROUPS * N_Q,
         "every grid group-quarter must have a canonical book to compare against")

    sub = one(con, """SELECT COUNT(*) FILTER (WHERE t_g_x <> t_g - d_t_g
                                                 OR t_e_x <> t_e - d_t_e) AS n_bad
                      FROM gtot""")
    gate("subtraction_identity_denominator", "outlier",
         "n_group_quarters_where_T_x != T - dT", int(sub["n_bad"]), BITEXACT,
         int(sub["n_bad"]) == 0,
         "(P4)+(P5)(iv) the denominators subtract exactly too — BOTH legs move")

    n_zero = int(one(con, "SELECT COUNT(*) AS n FROM gtot WHERE t_g_x <= 0")["n"])
    census("twin_zero_book", "global_outlier_excluded",
           "n_group_quarters_with_zero_twin_book", n_zero,
           "w' is NULL there by construction, exactly as w is when T = 0")


# ===========================================================================
# [4] FIRM-LEVEL TABLE: canonical weights (verbatim) + the twin
# ===========================================================================
def build_firm_table(con: duckdb.DuckDBPyConnection) -> None:
    print("[4] Firm-level table: canonical panel weights + the twin...")
    # Firm x group x quarter dollars under both regimes, restricted to the grid
    # universe. The country is deliberately NOT taken from the row: 06's
    # ict_grouped restricts on sec_entity_id and lets the GRID assign the
    # canonical primary-listing country, and a firm's non-EU-listed lines are
    # part of its I_ict. Mirrored exactly.
    con.execute("""
        CREATE OR REPLACE TABLE firm_usd AS
        SELECT sec_entity_id AS fid, hg, report_date AS rd,
               CAST(SUM(i_can) AS DOUBLE)            AS i_can,
               CAST(SUM(COALESCE(i_x, 0)) AS DOUBLE) AS i_x,
               CAST(SUM(ABS(i_can)) AS DOUBLE)       AS absmass
        FROM ict
        WHERE sec_entity_id IN (SELECT sec_entity_id FROM grid_univ)
        GROUP BY 1, 2, 3
    """)
    m = float(one(con, "SELECT MAX(absmass) AS m FROM firm_usd")["m"])
    gate("f64_exactness_firm_cell", "outlier", "max_firm_cell_SUM_ABS(I_ict)", m,
         TWO_53, m < TWO_53, "(P3) firm x group x quarter dollars")

    con.execute("""
        CREATE OR REPLACE TABLE firm AS
        SELECT g.fid, g.ctry, g.hg, g.rd, g.us, g.I_ict,
               g.w_g, g.wp_g, g.d_g, g.w_e, g.wp_e, g.d_e,
               COALESCE(u.i_can, 0.0) AS i_can,
               COALESCE(u.i_x,   0.0) AS i_x,
               t.t_g, t.t_g_x, t.t_e, t.t_e_x,
               CASE WHEN t.t_g_x > 0 THEN COALESCE(u.i_x, 0.0) / t.t_g_x END AS wx_g,
               CASE WHEN t.t_e_x > 0 THEN COALESCE(u.i_x, 0.0) / t.t_e_x END AS wx_e
        FROM grid g
        LEFT JOIN firm_usd u ON u.fid = g.fid AND u.hg = g.hg AND u.rd = g.rd
        LEFT JOIN gtot     t ON t.hg  = g.hg  AND t.rd = g.rd
    """)

    chk = one(con, """
        SELECT COUNT(*) FILTER (WHERE t_g IS NULL)                          AS n_no_book,
               COUNT(*) FILTER (WHERE i_can <> I_ict)                       AS n_usd_diff,
               COUNT(*) FILTER (WHERE w_g IS NOT NULL AND w_g <> i_can/t_g) AS n_wg_diff,
               COUNT(*) FILTER (WHERE w_e IS NOT NULL AND t_e > 0
                                  AND w_e <> i_can/t_e)                     AS n_we_diff,
               COUNT(*) FILTER (WHERE w_g IS NULL AND t_g > 0)              AS n_wg_null_bad
        FROM firm""")
    gate("firm_book_coverage", "outlier", "n_grid_rows_without_a_group_book",
         int(chk["n_no_book"]), 0, int(chk["n_no_book"]) == 0,
         "every grid (group, quarter) must join a reconstructed book")
    gate("firm_dollars_bitexact", "outlier", "n_rows_where_I_recon != panel_I_ict",
         int(chk["n_usd_diff"]), BITEXACT, int(chk["n_usd_diff"]) == 0,
         "(P5) firm-level dollars reproduce the merged panel exactly")
    gate("firm_weight_bitexact_global", "global", "n_rows_where_w != I/T",
         int(chk["n_wg_diff"]), BITEXACT, int(chk["n_wg_diff"]) == 0,
         "(P5)(iii) the canonical global weight IS I/T under this reconstruction")
    gate("firm_weight_bitexact_eu", "eu", "n_rows_where_w_eu != I/T_eu",
         int(chk["n_we_diff"]), BITEXACT, int(chk["n_we_diff"]) == 0,
         "(P5)(iii) same for the within-Europe reallocation diagnostic family")
    gate("firm_weight_null_semantics", "global", "n_rows_w_NULL_while_T>0",
         int(chk["n_wg_null_bad"]), 0, int(chk["n_wg_null_bad"]) == 0,
         "the global weight may be NULL only when the group's global book is empty")

    # Twin lags. The grid is a COMPLETE cartesian panel (gated at [2]), so
    # LAG(...,1) over (fid, hg) ORDER BY rd IS the previous CALENDAR quarter.
    con.execute("""
        CREATE OR REPLACE TABLE firmx AS
        SELECT *,
               LAG(wx_g, 1) OVER w         AS wpx_g,
               LAG(wx_e, 1) OVER w         AS wpx_e,
               wx_g - LAG(wx_g, 1) OVER w  AS dx_g,
               wx_e - LAG(wx_e, 1) OVER w  AS dx_e
        FROM firm
        WINDOW w AS (PARTITION BY fid, hg ORDER BY rd)
    """)
    lagchk = one(con, """
        WITH z AS (
            SELECT w_g, wp_g, d_g,
                   LAG(w_g, 1) OVER (PARTITION BY fid, hg ORDER BY rd) AS wp_chk
            FROM firm)
        SELECT COUNT(*) FILTER (WHERE wp_chk IS DISTINCT FROM wp_g)      AS n_wp_diff,
               COUNT(*) FILTER (WHERE (w_g - wp_g) IS DISTINCT FROM d_g) AS n_d_diff
        FROM z""")
    gate("lag_semantics_match_panel", "global", "n_rows_wprev_disagreeing",
         int(lagchk["n_wp_diff"]), 0, int(lagchk["n_wp_diff"]) == 0,
         "this file's LAG == 06_cartesian_grid.jl's w_prev_global, so the twin "
         "lag has the same calendar meaning as the canonical one")
    gate("delta_definition_match_panel", "global", "n_rows_delta_disagreeing",
         int(lagchk["n_d_diff"]), 0, int(lagchk["n_d_diff"]) == 0,
         "delta_w_global == w - w_prev on every row (backward difference)")


# ===========================================================================
# [5] COUNTRY AGGREGATION + DECOMPOSITION
# ===========================================================================
# DECOMPOSITION (LOCKED). Two exhaustive, mutually exclusive partitions of the
# firms in (c, g, t); each therefore re-sums to dW EXACTLY, up to float
# reassociation, which is gated at TOL_DECOMP_ABS.
#
#   MARGIN partition, on (w_{t-1}, w_t):
#     entry       w_{t-1} = 0, w_t > 0
#     exit        w_{t-1} > 0, w_t = 0
#     continuing  w_{t-1} > 0, w_t > 0
#     zero-zero   w_{t-1} = 0, w_t = 0     contributes EXACTLY 0 (gated)
#
#   SIZE partition, on the lagged weight WITHIN the country — the spec's
#   "large firms vs small firms (by lagged w within country)":
#     large  w_{t-1} > 0 and w_{t-1} >  median(w_{t-1} | w_{t-1} > 0) in (c,g,t)
#     small  w_{t-1} > 0 and w_{t-1} <= that median
#     new    w_{t-1} = 0
#   The two size branches are complementary by construction, so ties at the
#   median cannot lose a firm; "new" is the entry margin (plus the identically
#   zero zero-zero firms), and it is reported separately precisely because a
#   lagged-size split cannot classify a position that did not exist at t-1.
#   That is the same margin a lagged-holdings-weighted WLS silently discards.
#
#   MEDIAN-TIE CONVENTION (UNIFIED 2026-08-10 after an external review found the
#   two implementations of the size split disagreeing). THE PROJECT-WIDE RULE IS
#       large = wp >  median ,   small = 0 < wp <= median
#   i.e. STRICTLY ABOVE the cell median is "large" and a firm sitting EXACTLY ON
#   the median is "small". run_country_decomposition.py used the opposite
#   allocation (large = wp >= median, small = wp < median) and has been changed
#   to match THIS file. Both rules give an exhaustive, disjoint partition, so the
#   choice only moves median-tied firms between the two buckets — but median ties
#   are COMMON in small country cells, and the two artifacts were describing
#   "large" differently while being read side by side. Why this side of the tie:
#   with an EVEN number of positive lagged weights duckdb's MEDIAN interpolates
#   between the two central values, no firm equals it, and both rules split the
#   cell in half; with an ODD number the median IS a firm's own weight, and cells
#   here are frequently tiny (a handful of firms, sometimes ONE). Under `>=' a
#   one-firm cell reports large = 1 / small = 0 — the cell's only, and therefore
#   also smallest, firm is labelled large. Under `>' it reports large = 0 /
#   small = 1, which does not manufacture a "large firm" out of a singleton. The
#   rule is STAMPED into country_weight_decomposition.csv (column size_tie_rule)
#   so no reader has to infer it from code.
#
#   Concentration diagnostics (NOT a partition): the signed contribution of the
#   single firm with the largest |delta w|, and of the top 5 such firms.
#
# Rows with a NULL w, w_prev or delta fall into NO branch of either partition.
# That is why the identity gate asserts only on FULLY-NON-NULL cells and
# censuses the rest instead of silently dropping them.
#
# READING NOTE (deliberate, do not "fix" it silently). Each component is
# COALESCE(..., 0), so a branch with no firms reads 0 rather than missing — that
# is the honest value for an empty branch. The consequence is that on an
# ALL-NULL cell (the first grid quarter, or an empty group book) every component
# reads 0 while dW itself is MISSING. Such a cell is not a decomposition of
# anything: read the components only where dW is non-missing. Both the .dta
# (null_class, ident_ok) and country_weight_decomposition.csv (dW_total, which
# is missing there) make that visible without having to remember this note.
# ===========================================================================
AGG_TEMPLATE = """
CREATE OR REPLACE TABLE agg_{sfx} AS
WITH med AS (
    SELECT ctry, hg, rd, MEDIAN(CASE WHEN {wp} > 0 THEN {wp} END) AS med_wp
    FROM firmx GROUP BY 1, 2, 3
), rk AS (
    SELECT f.ctry, f.hg, f.rd, f.fid,
           f.{w} AS w, f.{wp} AS wp, f.{d} AS d, m.med_wp,
           ROW_NUMBER() OVER (PARTITION BY f.ctry, f.hg, f.rd
                              ORDER BY ABS(f.{d}) DESC NULLS LAST, f.fid) AS rn
    FROM firmx f JOIN med m USING (ctry, hg, rd)
)
SELECT ctry, hg, rd,
       COUNT(*)                                             AS n_firms,
       SUM(w)                                               AS W,
       SUM(wp)                                              AS W_sumwprev,
       SUM(d)                                               AS dW_sumdelta,
       COUNT(*) - COUNT(w)                                  AS n_w_null,
       COUNT(*) - COUNT(wp)                                 AS n_wprev_null,
       COUNT(*) - COUNT(d)                                  AS n_delta_null,
       COUNT(*) FILTER (WHERE w  > 0)                       AS n_held,
       COUNT(*) FILTER (WHERE wp > 0)                       AS n_held_lag,
       ANY_VALUE(med_wp)                                    AS med_wprev,
       COALESCE(SUM(d) FILTER (WHERE wp = 0 AND w > 0), 0)  AS dW_entry,
       COALESCE(SUM(d) FILTER (WHERE wp > 0 AND w = 0), 0)  AS dW_exit,
       COALESCE(SUM(d) FILTER (WHERE wp > 0 AND w > 0), 0)  AS dW_cont,
       COALESCE(SUM(d) FILTER (WHERE wp = 0 AND w = 0), 0)  AS dW_zz,
       COUNT(*) FILTER (WHERE wp = 0 AND w > 0)             AS n_entry,
       COUNT(*) FILTER (WHERE wp > 0 AND w = 0)             AS n_exit,
       COUNT(*) FILTER (WHERE wp > 0 AND w > 0)             AS n_cont,
       COUNT(*) FILTER (WHERE wp = 0 AND w = 0)             AS n_zz,
       COALESCE(SUM(d)  FILTER (WHERE wp > 0 AND wp >  med_wp), 0) AS dW_large,
       COALESCE(SUM(d)  FILTER (WHERE wp > 0 AND wp <= med_wp), 0) AS dW_small,
       COALESCE(SUM(d)  FILTER (WHERE wp = 0), 0)                  AS dW_new,
       COUNT(*)         FILTER (WHERE wp > 0 AND wp >  med_wp)     AS n_large,
       COUNT(*)         FILTER (WHERE wp > 0 AND wp <= med_wp)     AS n_small,
       COUNT(*)         FILTER (WHERE wp = 0)                      AS n_new,
       COALESCE(SUM(wp) FILTER (WHERE wp > 0 AND wp >  med_wp), 0) AS wprev_large,
       COALESCE(SUM(wp) FILTER (WHERE wp > 0), 0)                  AS wprev_tot,
       COALESCE(SUM(d) FILTER (WHERE rn  = 1), 0)                  AS dW_top1,
       COALESCE(SUM(d) FILTER (WHERE rn <= 5), 0)                  AS dW_top5,
       COALESCE(SUM(ABS(d)), 0)                                    AS dW_absmass
FROM rk GROUP BY 1, 2, 3
"""

# Own-lag leg: W_{c,g,t} - W_{c,g,t-1}, the lag taken on the COUNTRY series
# itself. Legitimate only because the country panel inherits completeness from
# the firm grid (gated at [2] and re-gated on the row count at [7]).
OWNLAG_TEMPLATE = """
CREATE OR REPLACE TABLE agg2_{sfx} AS
SELECT *,
       LAG(W, 1)     OVER w AS W_lag_ownlag,
       W - LAG(W, 1) OVER w AS dW_ownlag
FROM agg_{sfx}
WINDOW w AS (PARTITION BY ctry, hg ORDER BY rd)
"""

FAMILIES = (
    ("g",  "w_g",  "wp_g",  "d_g",  "global"),
    ("e",  "w_e",  "wp_e",  "d_e",  "eu"),
    ("gx", "wx_g", "wpx_g", "dx_g", "global_outlier_excluded"),
    ("ex", "wx_e", "wpx_e", "dx_e", "eu_outlier_excluded"),
)


def aggregate_country(con: duckdb.DuckDBPyConnection) -> None:
    print("[5] Country aggregation (4 weight families) + decomposition...")
    for sfx, w, wp, d, lab in FAMILIES:
        con.execute(AGG_TEMPLATE.format(sfx=sfx, w=w, wp=wp, d=d))
        con.execute(OWNLAG_TEMPLATE.format(sfx=sfx))
        n = int(one(con, f"SELECT COUNT(*) AS n FROM agg2_{sfx}")["n"])
        print(f"    {lab:<24s} -> agg2_{sfx}: {n:,} country x group x quarter cells")

    # Dollar block: SUPPLEMENTARY, a DIFFERENT ESTIMAND. dlog mixes price,
    # quantity and composition changes and is NEVER a net flow.
    con.execute("""
        CREATE OR REPLACE TABLE agg_usd AS
        WITH s AS (
            SELECT ctry, hg, rd, SUM(i_can) AS usd, SUM(i_x) AS usd_x,
                   SUM(ABS(i_can)) AS absmass
            FROM firmx GROUP BY 1, 2, 3
        )
        SELECT *,
               LAG(usd,   1) OVER w AS usd_lag,
               LAG(usd_x, 1) OVER w AS usd_x_lag,
               CASE WHEN usd   > 0 AND LAG(usd,   1) OVER w > 0
                    THEN LN(usd)   - LN(LAG(usd,   1) OVER w) END AS dlog_usd,
               CASE WHEN usd_x > 0 AND LAG(usd_x, 1) OVER w > 0
                    THEN LN(usd_x) - LN(LAG(usd_x, 1) OVER w) END AS dlog_usd_x
        FROM s
        WINDOW w AS (PARTITION BY ctry, hg ORDER BY rd)
    """)
    m = float(one(con, "SELECT MAX(absmass) AS m FROM agg_usd")["m"])
    gate("f64_exactness_country_cell", "outlier",
         "max_country_cell_SUM_ABS(I_ict)", m, TWO_53, m < TWO_53,
         "(P3) bounds the country-level dollar sums as well")


# ===========================================================================
# [5b] SENSITIVITY (c) DONE PROPERLY: A COMMON *FIRM* SET, NOT A COMMON CELL SET
# ===========================================================================
# WHAT WAS WRONG (external review, 2026-08-10 — confirmed, do not re-litigate).
# The old implementation of "M2 common coverage" was the country-quarter flag
#     m2_cov = (m2_lag is non-missing)
# and the sensitivity was `if m2_cov == 1'. That ALIGNS THE CELLS and nothing
# else. It does NOT put M1 and M3 on the firms M2 is actually computed on: a
# country-quarter in which only 10% of firms carry a market cap still has a
# non-missing M2 and therefore still gets m2_cov = 1, so M1 and M3 keep being
# averaged over the FULL firm set while M2 is averaged over a possibly tiny
# subset. The "common sample" was common in name only, and the comparison the
# advisors asked for — are the three measures telling the same story when they
# are built on the same firms? — was never run.
#
# WHAT IS DONE INSTEAD. Inside every country-quarter, M1 and M3 are RECOMPUTED
# from the firm-level grid over EXACTLY the firms M2 uses, i.e. the firms with a
# non-missing, strictly positive market cap in marketcap_it.parquet:
#     m1cov = SUM_{i in c, mcap_i>0} cn_links_i / SUM_{i in c, mcap_i>0} sc_links_i
#     m3cov = mean_{i in c, mcap_i>0} cn_share_i
#     m2cov = M2 itself (M2 is ALREADY defined only over covered firms, which is
#             the whole point — it needs no restriction, the other two do)
# so the (c) sensitivity compares three measures on ONE FIRM SET.
#
# TWO FACTS ABOUT COVERAGE, AT TWO DIFFERENT GRAINS. BOTH MUST BE STATED; each
# on its own is misleading, and the project has already told the story wrong
# once in each direction.
#   COUNTRY-QUARTER GRAIN. At v3.1 M2 has FEWER country-quarter NAs (27) than M1
#     (106). So the familiar framing "M2 is the sparse one" is FLATLY WRONG at
#     country grain: on the cell set it is M1 that goes missing more often (M1's
#     denominator SUM(sc_links) is 0 for a country-quarter in which no firm has
#     any supply-chain link, while M2 only needs one covered firm with a market
#     cap). This is exactly why the old cell-set restriction looked harmless.
#   FIRM GRAIN. M2's market-cap coverage is only 42.9% of firm-quarters in 2023
#     (declining over time as the Revere universe adds small unlisted firms).
#     So at FIRM grain M2 really is built on a minority of the firms M1 and M3
#     see — which is precisely the comparability problem sensitivity (c) exists
#     to measure, and which the cell-set restriction did nothing about.
#   The two numbers are NOT in conflict; they are different grains. Both are
#   RE-DERIVED live below and printed, never quoted from this comment.
#
# The reconstruction is gated: recomputing M1/M2/M3 over ALL firms here must
# reproduce country_measures_m1m2m3.csv (universe = ownership_matched) cell for
# cell, and n_firms / n_firms_m2 must match exactly. If that holds, the
# covered-subset legs are the same construction restricted to a subset — not a
# second, differently-built measure.
# ===========================================================================
def build_common_coverage_measures(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    print("[5b] Common-FIRM-SET M1/M3 (sensitivity (c) rebuilt properly)...")
    if not MCAP_P.is_file():
        die(f"DEPENDENCY: {MCAP_P} not found. Sensitivity (c) needs the PIT "
            f"market cap that DEFINES M2's firm set; without it M1/M3 cannot be "
            f"restricted to that set and the sensitivity would silently fall "
            f"back to the cell-set restriction the review rejected. "
            f"Run 04_us_ownership_european.jl.")

    # u_owned, VERBATIM from build_country_measures.py: firm attributes in the
    # grid are identical across holder_group, so holder_group = 'US' picks
    # exactly one row per firm-quarter; china_share IS NOT NULL == PIT-present.
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW u_owned AS
        SELECT g.sec_country                                     AS country,
               g.report_date                                     AS quarter_end,
               g.sec_entity_id                                   AS firm_id,
               CAST(g.n_cn_customer + g.n_cn_supplier AS BIGINT) AS cn_links,
               g.n_supplychain_links                             AS sc_links,
               g.china_share                                     AS cn_share,
               m.market_cap                                      AS mcap
        FROM read_parquet('{GRID_P.as_posix()}') g
        LEFT JOIN read_parquet('{MCAP_P.as_posix()}') m
               ON m.sec_entity_id = g.sec_entity_id
              AND m.sec_country   = g.sec_country
              AND m.report_date   = g.report_date
        WHERE g.holder_group = 'US'
          AND g.china_share IS NOT NULL
          AND g.sec_country IN {EU_SQL}
    """)
    dupf = int(one(con, """SELECT COUNT(*) AS n FROM (
        SELECT firm_id, quarter_end FROM u_owned GROUP BY 1, 2 HAVING COUNT(*) > 1)
    """)["n"])
    gate("coverage_universe_key_uniqueness", "coverage",
         "n_duplicate_firm_quarter", dupf, 0, dupf == 0,
         "a duplicated firm-quarter would double-count links into M1")

    cov = con.sql("""
        SELECT country, quarter_end,
               COUNT(*)                             AS n_firms_all,
               COUNT(*) FILTER (WHERE mcap > 0)     AS n_firms_cov,
               -- ALL-FIRM legs: recomputed ONLY to prove this reconstruction is
               -- build_country_measures.py, so the covered legs are the same
               -- construction restricted to a subset.
               CAST(SUM(cn_links) AS DOUBLE) / NULLIF(SUM(sc_links), 0) AS m1_all,
               SUM(cn_share * mcap) FILTER (WHERE mcap > 0)
                   / NULLIF(SUM(mcap) FILTER (WHERE mcap > 0), 0)       AS m2_all,
               AVG(cn_share)                                            AS m3_all,
               -- COMMON FIRM SET: exactly the firms M2 is computed on.
               CAST(SUM(cn_links) FILTER (WHERE mcap > 0) AS DOUBLE)
                   / NULLIF(SUM(sc_links) FILTER (WHERE mcap > 0), 0)   AS m1_cov,
               AVG(cn_share) FILTER (WHERE mcap > 0)                    AS m3_cov
        FROM u_owned GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    cov["quarter_end"] = pd.to_datetime(cov["quarter_end"])

    # ---- reconstruction gate against the canonical measures CSV -------------
    canon = pd.read_csv(MEAS_P, parse_dates=["quarter_end"])
    canon = canon.loc[canon["is_primary"].eq(1)
                      & canon["universe"].eq(COUNTRY_UNIVERSE)].copy()
    j = canon.merge(cov, on=["country", "quarter_end"], how="outer",
                    indicator=True, validate="one_to_one")
    n_unmatched = int((j["_merge"] != "both").sum())
    gate("coverage_cellset_matches_canonical", "coverage",
         "n_unmatched_country_quarters", n_unmatched, 0, n_unmatched == 0,
         f"this reconstruction must span exactly the canonical "
         f"{COUNTRY_UNIVERSE} cell set of {MEAS_P.name}")
    n_nf = int((j["n_firms"].astype("int64") != j["n_firms_all"].astype("int64")).sum())
    n_nm2 = int((j["n_firms_m2"].astype("int64") != j["n_firms_cov"].astype("int64")).sum())
    gate("coverage_n_firms_exact", "coverage", "n_cells_with_n_firms_mismatch",
         n_nf, 0, n_nf == 0, "the all-firm count must reproduce n_firms exactly")
    gate("coverage_n_firms_m2_exact", "coverage",
         "n_cells_with_n_firms_m2_mismatch", n_nm2, 0, n_nm2 == 0,
         "the covered-firm count must reproduce n_firms_m2 exactly — it IS the "
         "definition of the common firm set")
    worst = {}
    for k, a, b in (("M1", "M1", "m1_all"), ("M2", "M2", "m2_all"),
                    ("M3", "M3", "m3_all")):
        both = j[a].notna() & j[b].notna()
        d = (j.loc[both, a] - j.loc[both, b]).abs()
        scale = j.loc[both, a].abs().clip(lower=1e-12)
        worst[k] = float((d / scale).max()) if both.any() else 0.0
        n_na_mismatch = int((j[a].isna() != j[b].isna()).sum())
        gate(f"coverage_recon_{k}_nullset", "coverage",
             f"n_cells_where_{k}_missingness_differs", n_na_mismatch, 0,
             n_na_mismatch == 0,
             "the reconstruction must go missing on exactly the canonical cells")
    mx = max(worst.values())
    gate("coverage_recon_values", "coverage", "max_rel_diff_M1_M2_M3", mx, 1e-12,
         mx < 1e-12,
         f"per-measure max rel diff M1 {worst['M1']:.3e} / M2 {worst['M2']:.3e} "
         f"/ M3 {worst['M3']:.3e}; this proves the covered legs are the SAME "
         f"construction restricted to M2's firm set")

    # ---- the two coverage facts, at their two grains, RE-DERIVED live -------
    na_cell = {k: int(canon[k].isna().sum()) for k in ("M1", "M2", "M3")}
    census("coverage_country_quarter_NAs", "coverage",
           f"M1={na_cell['M1']} M2={na_cell['M2']} M3={na_cell['M3']}",
           na_cell["M2"] - na_cell["M1"],
           "COUNTRY-QUARTER grain. Memo value at v3.1: M2 = 27 NAs, M1 = 106. "
           "If M2 <= M1 here, the framing 'M2 is the sparse one' is WRONG at "
           "this grain (M1's denominator SUM(sc_links) is what goes to zero)")
    yr = cov.assign(year=cov["quarter_end"].dt.year).groupby("year")[
        ["n_firms_all", "n_firms_cov"]].sum()
    yr["mcap_covered_share"] = yr["n_firms_cov"] / yr["n_firms_all"]
    print("    FIRM-grain market-cap coverage by year (M2's actual firm set):")
    print(yr.tail(8).to_string(float_format=lambda x: f"{x:,.4f}"))
    if 2023 in yr.index:
        census("coverage_firm_grain_2023", "coverage", "mcap_covered_share_2023",
               round(float(yr.loc[2023, "mcap_covered_share"]), 4),
               "FIRM grain. Memo value at v3.1: 0.429. This and the "
               "country-quarter NA counts above are DIFFERENT GRAINS; both must "
               "be stated together or the coverage story is misleading")

    # how far the two legs move, before any regression sees them
    for k in ("m1", "m3"):
        both = cov[f"{k}_all"].notna() & cov[f"{k}_cov"].notna()
        d = (cov.loc[both, f"{k}_cov"] - cov.loc[both, f"{k}_all"]).abs()
        census(f"coverage_shift_{k}", "coverage",
               f"median_abs_change_{k}_cov_vs_all", round(float(d.median()), 6),
               f"{int(both.sum())} cells; max {float(d.max()):.6f}. This is the "
               f"size of the thing the OLD cell-set restriction could not see")
    n_cov_na = int(cov["m1_cov"].isna().sum())
    census("coverage_m1cov_NAs", "coverage", "n_cells_m1cov_missing", n_cov_na,
           "cells where NO covered firm carries a supply-chain link; the (c) "
           "sensitivity loses them for M1 and they are not silently zero-filled")
    return cov[["country", "quarter_end", "m1_cov", "m3_cov", "m2_all",
                "n_firms_cov", "n_firms_all"]]


# ===========================================================================
# [6] THE IDENTITY HARD GATE
# ===========================================================================
# dW_{c,g,t} must equal sum_{i in c} delta_w_{i,g,t} cell by cell. Three legs:
#   I1  W_lag (own lag of the COUNTRY series)  ==  sum_i w_prev
#   I2  dW_ownlag = W_t - W_lag                ==  sum_i delta_w      <- headline
#   I3  sum_i w - sum_i w_prev                 ==  sum_i delta_w
# I1 additionally proves that 06's firm-level LAG and the country-series lag
# refer to the SAME calendar quarter: had 06's lag ever skipped a quarter, I1
# would break while I2 and I3 could still pass.
#
# NULL SEMANTICS, handled explicitly and NEVER silently dropped. SQL SUM ignores
# NULLs, so a single firm with a NULL delta makes sum_i delta_w a sum over a
# SUBSET while W_t - W_{t-1} stays a full difference: the identity breaks. Cells
# are therefore classified as
#   fully_non_null  no NULL w, w_prev or delta among the cell's firms
#                   -> HARD ASSERT: |I1|, |I2|, |I3| below tolerance
#   all_null        every firm NULL — a legitimately empty group book, or the
#                   first grid quarter where no lag exists -> dW is NULL
#   partial_null    0 < n_null < n_firms. THE DANGEROUS CASE.
#
# FAIL-CLOSED ON THE MAIN SAMPLE (external review, 2026-08-10). An earlier
# vintage of this file merely PRINTED a warning for partial-null cells and
# carried them into the panel with null_class = 2, while run_country_weight_ddd.do
# does NOT exclude null_class == 2 from the main regression. So the advertised
# promise — "dW equals the sum of firm deltas on the main sample, or the build
# stops" — was not actually enforced: a partial-null cell would have entered the
# headline with dW and sum_i dw_i disagreeing by an unbounded amount. The gate
# below now ABORTS THE BUILD when the GLOBAL (main) family carries any
# partial-null cell. The alternative the review allows — keeping such cells but
# EXCLUDING them from the estimation sample — is a CHANGE OF ESTIMAND (the
# outcome would become "dW over country-quarters in which every firm delta is
# observed") and is therefore not taken silently here; the abort message names
# it so the decision is made in the open, upstream, by a human.
# The non-main families (eu, and the two outlier-excluded twins) are censused
# with their own counts rather than gated, because they are diagnostics and a
# diagnostic that vanishes is worse than a diagnostic with a documented hole.
# ===========================================================================
def identity_gate(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    print("[6] IDENTITY HARD GATE (dW own-lag vs sum of firm deltas)...")
    parts = []
    for sfx, _w, _wp, _d, lab in FAMILIES:
        df = con.sql(f"""
            SELECT '{lab}' AS family, ctry, hg, rd, n_firms,
                   n_w_null, n_wprev_null, n_delta_null, n_held, n_held_lag,
                   W, W_lag_ownlag, W_sumwprev, dW_ownlag, dW_sumdelta,
                   (W - W_sumwprev) AS dW_sumw_minus_sumwprev,
                   dW_entry, dW_exit, dW_cont, dW_zz,
                   dW_large, dW_small, dW_new
            FROM agg2_{sfx} ORDER BY ctry, hg, rd
        """).df()
        parts.append(df)
    g = pd.concat(parts, ignore_index=True)

    n_any = g[["n_w_null", "n_wprev_null", "n_delta_null"]].max(axis=1)
    g["null_class"] = np.where(
        n_any == 0, "fully_non_null",
        np.where((g["n_w_null"] == g["n_firms"])
                 | (g["n_delta_null"] == g["n_firms"]), "all_null",
                 "partial_null"))

    g["disc_I1_wlag"] = (g["W_lag_ownlag"] - g["W_sumwprev"]).abs()
    g["disc_I2_headline"] = (g["dW_ownlag"] - g["dW_sumdelta"]).abs()
    g["disc_I3_sums"] = (g["dW_sumw_minus_sumwprev"] - g["dW_sumdelta"]).abs()
    g["resid_margin_partition"] = (
        g[["dW_entry", "dW_exit", "dW_cont", "dW_zz"]].sum(axis=1)
        - g["dW_sumdelta"]).abs()
    g["resid_size_partition"] = (
        g[["dW_large", "dW_small", "dW_new"]].sum(axis=1) - g["dW_sumdelta"]).abs()

    # Scale for the relative gate: a robust dispersion of |dW| WITHIN the family,
    # not the cell's own |dW| — a near-zero cell would otherwise manufacture an
    # enormous ratio out of pure float noise.
    g["dW_scale_p90"] = np.nan
    for fam, sub in g.groupby("family"):
        v = sub["dW_sumdelta"].abs().dropna()
        g.loc[g["family"].eq(fam), "dW_scale_p90"] = (
            float(np.percentile(v, 90)) if len(v) else np.nan)

    print("    NULL census by family x class (cells):")
    print(g.groupby(["family", "null_class"], as_index=False)
           .agg(n_cells=("ctry", "size"), n_countries=("ctry", "nunique"),
                n_quarters=("rd", "nunique")).to_string(index=False))

    for fam in sorted(g["family"].unique()):
        f = g[g["family"].eq(fam)]
        fn = f[f["null_class"].eq("fully_non_null")]
        if fn.empty:
            die(f"family {fam}: ZERO fully-non-null cells — the identity gate "
                f"cannot be evaluated, which is itself a failure.")
        scale = float(fn["dW_scale_p90"].iloc[0])
        for leg, col in (("I1_wlag", "disc_I1_wlag"),
                         ("I2_headline", "disc_I2_headline"),
                         ("I3_sums", "disc_I3_sums")):
            v = fn[col].dropna()
            mxv = float(v.max()) if len(v) else 0.0
            gate(f"identity_{leg}_abs", fam, "max_abs_discrepancy", mxv,
                 TOL_IDENT_ABS, mxv < TOL_IDENT_ABS,
                 f"over {len(fn):,} fully-non-null cells ({len(v):,} evaluable)")
            rel = mxv / scale if scale and np.isfinite(scale) and scale > 0 else 0.0
            gate(f"identity_{leg}_rel", fam, "max_abs_disc / p90(|dW|)", rel,
                 TOL_IDENT_REL, rel < TOL_IDENT_REL, f"p90(|dW|) = {scale:.6e}")
            advisory(f"identity_{leg}_tight", fam, "max_abs_discrepancy", mxv,
                     TOL_IDENT_ADV, mxv <= TOL_IDENT_ADV,
                     "float reassociation alone should sit below this")
            census(f"identity_{leg}_differing_cells", fam,
                   f"n_cells_with_disc_gt_{EPS_DIFFER:g}",
                   int((v > EPS_DIFFER).sum()),
                   "the two constructions are bitwise identical on the rest")
        for leg, col in (("margin", "resid_margin_partition"),
                         ("size", "resid_size_partition")):
            v = fn[col].dropna()
            mxv = float(v.max()) if len(v) else 0.0
            gate(f"decomposition_{leg}_resums", fam, "max_abs_residual", mxv,
                 TOL_DECOMP_ABS, mxv < TOL_DECOMP_ABS,
                 "the partition must re-sum to dW")
        vz = fn["dW_zz"].dropna().abs()
        mzz = float(vz.max()) if len(vz) else 0.0
        gate("decomposition_zerozero_is_zero", fam, "max_abs_dW_zz", mzz, 0.0,
             mzz == 0.0, "firms with w_{t-1} = 0 and w_t = 0 contribute exactly 0")

    # the worst offenders, printed so a reviewer sees magnitudes not adjectives
    worst = (g[g["null_class"].eq("fully_non_null")]
             .nlargest(5, "disc_I2_headline")
             [["family", "ctry", "hg", "rd", "dW_sumdelta", "dW_ownlag",
               "disc_I2_headline"]])
    print("    largest I2 discrepancies among fully-non-null cells:")
    print(worst.to_string(index=False, float_format=lambda x: f"{x:.6e}"))

    bad = g[g["null_class"].ne("fully_non_null")]
    if len(bad):
        print("    cells carrying NULLs (documented, NOT dropped):")
        print(bad.groupby(["family", "null_class"], as_index=False)
                 .agg(n_cells=("ctry", "size"),
                      max_disc_I2=("disc_I2_headline", "max"),
                      max_n_delta_null=("n_delta_null", "max"),
                      first_rd=("rd", "min"), last_rd=("rd", "max"))
                 .to_string(index=False))
    n_partial = int((g["null_class"] == "partial_null").sum())
    census("identity_partial_null", "all", "n_partial_null_cells", n_partial,
           "cells where SOME firms are NULL; flagged null_class = 2 in the .dta")
    for fam in sorted(g["family"].unique()):
        n_p_fam = int(((g["family"] == fam)
                       & (g["null_class"] == "partial_null")).sum())
        if fam != "global":
            census("identity_partial_null_family", fam,
                   "n_partial_null_cells", n_p_fam,
                   "diagnostic family: censused, not gated")

    # ---- THE FAIL-CLOSED GATE ON THE MAIN SAMPLE ---------------------------
    n_partial_main = int(((g["family"] == "global")
                          & (g["null_class"] == "partial_null")).sum())
    if n_partial_main:
        bad_main = (g[(g["family"] == "global")
                      & (g["null_class"] == "partial_null")]
                    .nlargest(10, "disc_I2_headline")
                    [["ctry", "hg", "rd", "n_firms", "n_w_null", "n_wprev_null",
                      "n_delta_null", "dW_ownlag", "dW_sumdelta",
                      "disc_I2_headline"]])
        print("    partial-NULL cells on the MAIN (global) family — worst 10 by "
              "|dW_ownlag - dW_sumdelta|:")
        print(bad_main.to_string(index=False))
    gate("identity_partial_null_main_fail_closed", "global",
         "n_partial_null_cells_on_main_sample", n_partial_main, 0,
         n_partial_main == 0,
         "THE MAIN-SAMPLE PROMISE: dW = sum_i dw_i cell by cell, or the build "
         "stops. A partial-NULL cell breaks that identity by an UNBOUNDED "
         "amount (SQL SUM skips NULLs, so sum_i dw_i is a sum over a SUBSET "
         "while W_t - W_{t-1} is a full difference) and run_country_weight_ddd.do "
         "does NOT exclude null_class == 2, so such a cell would enter the "
         "headline. REMEDIES, both upstream and both explicit: (1) fix the "
         "source of the NULL firm-deltas in 06_cartesian_grid.jl / the group "
         "books, or (2) decide to EXCLUDE these cells from the estimation "
         "sample — which CHANGES THE ESTIMAND to 'dW over country-quarters in "
         "which every firm delta is observed' and must be stated in the "
         "write-up. This script refuses to make choice (2) on its own")
    return g


# ===========================================================================
def main() -> None:
    print("=" * 78)
    print("build_country_weight_panel.py — country-level PORTFOLIO-WEIGHT panel")
    print("MAIN = global denominator. EU = within-Europe reallocation diagnostic.")
    print("dlog(dollars) = SUPPLEMENTARY, a DIFFERENT ESTIMAND, never a net flow.")
    print("=" * 78)

    # ---------------- [1] gates ----------------------------------------------
    print("[1] Freshness, ordering and rotation gates...")
    mt_grid = assert_fresh(GRID_P, "merged_us_eu_zero_filled.parquet", VMIN)
    mt_ict = assert_fresh(ICT_P, "I_ict_panel.parquet", VMIN)
    assert_fresh(CTOT_P, "country_total_ct.parquet", VMIN)
    assert_fresh(MEAS_P, "country_measures_m1m2m3.csv", VMIN)
    mt_eom = assert_fresh(EOM_P, "holdings_eom.parquet", VMIN_EOM)
    if mt_eom > mt_ict:
        die(f"ORDERING GATE: holdings_eom.parquet ({mt_eom:%Y-%m-%d %H:%M}) is "
            f"NEWER than I_ict_panel.parquet ({mt_ict:%Y-%m-%d %H:%M}). The "
            f"row-level source has been rebuilt since the derived panel, so "
            f"re-deriving I_ict from it cannot reproduce canonical values. "
            f"Re-run 04 and 06 before this script.")
    meta = json.loads(ICT_M.read_text(encoding="utf-8"))
    if datetime.fromisoformat(meta["build_ts"]) < VMIN:
        die(f"STALE MANIFEST: I_ict build_ts {meta['build_ts']} < {VMIN}.")
    print(f"    freshness PASS (I_ict build_ts {meta['build_ts']}, row_count "
          f"{meta['row_count']:,}; grid {mt_grid:%Y-%m-%d %H:%M})")
    rotation_guard()
    print("    rotation guard PASS (no canonical output would be overwritten)")

    # ---------------- [1b] provenance hashes ---------------------------------
    log_provenance({
        "merged_us_eu_zero_filled.parquet": GRID_P,
        "I_ict_panel.parquet": ICT_P,
        "I_ict_panel.parquet.meta.json": ICT_M,
        "country_total_ct.parquet": CTOT_P,
        "holdings_eom.parquet": EOM_P,
        "country_measures_m1m2m3.csv": MEAS_P,
        "marketcap_it.parquet": MCAP_P,
        "dq_variants/dq_census.csv": DQ_CENSUS,
    })

    con = connect()

    # ---------------- [2] grid integrity -------------------------------------
    print("[2] Firm-level grid integrity...")
    con.execute(f"""
        CREATE OR REPLACE TABLE grid AS
        SELECT sec_entity_id AS fid, sec_country AS ctry, holder_group AS hg,
               report_date AS rd,
               CASE WHEN holder_group = 'US' THEN 1 ELSE 0 END AS us,
               I_ict,
               portfolio_weight_global AS w_g, w_prev_global AS wp_g,
               delta_w_global          AS d_g,
               portfolio_weight_eu     AS w_e, w_prev AS wp_e, delta_w AS d_e,
               shock_us_cn, shock_us_cn_lag1q
        FROM read_parquet('{GRID_P.as_posix()}')
    """)
    con.execute("CREATE OR REPLACE TABLE grid_univ AS "
                "SELECT DISTINCT fid AS sec_entity_id, ctry FROM grid")

    s = one(con, """SELECT COUNT(*) AS n, COUNT(DISTINCT fid) AS nf,
                           COUNT(DISTINCT rd) AS nq, COUNT(DISTINCT hg) AS ng,
                           COUNT(DISTINCT ctry) AS nc FROM grid""")
    n, nf, nq = int(s["n"]), int(s["nf"]), int(s["nq"])
    ng, nc = int(s["ng"]), int(s["nc"])
    gate("grid_complete_cartesian", "grid", "n_rows", n, nf * nq * ng,
         n == nf * nq * ng,
         f"{nf:,} firms x {nq} quarters x {ng} groups — LAG(...,1) means the "
         f"previous CALENDAR quarter ONLY if this holds")
    gate("grid_quarter_count", "grid", "n_quarters", nq, N_Q, nq == N_Q,
         f"frozen v3.1 span {Q_FIRST}..{Q_LAST}")
    gate("grid_group_count", "grid", "n_holder_groups", ng, N_GROUPS,
         ng == N_GROUPS, "US / NONUS")
    gate("grid_country_count", "grid", "n_sec_countries", nc, len(EU_COUNTRIES),
         nc == len(EU_COUNTRIES), "EU-28 set from 00_setup.jl")

    dupk = int(one(con, """SELECT COUNT(*) AS n FROM (
        SELECT fid, hg, rd FROM grid GROUP BY 1, 2, 3 HAVING COUNT(*) > 1)""")["n"])
    gate("grid_key_uniqueness", "grid", "n_duplicate_firm_group_quarter", dupk,
         0, dupk == 0, "")
    dupc = int(one(con, """SELECT COUNT(*) AS n FROM (
        SELECT sec_entity_id FROM grid_univ GROUP BY 1 HAVING COUNT(*) > 1)""")["n"])
    gate("grid_one_country_per_firm", "grid", "n_firms_with_multiple_countries",
         dupc, 0, dupc == 0,
         "the country must come from the grid's canonical primary-listing pick")

    q = con.sql("SELECT DISTINCT rd FROM grid ORDER BY rd").df()
    rds = pd.to_datetime(q["rd"])
    qq = pd.PeriodIndex(rds, freq="Q")
    exp_q = pd.period_range(Q_FIRST, Q_LAST, freq="Q")
    gate("grid_quarters_contiguous", "grid", "quarter_sequence",
         f"{qq[0]}..{qq[-1]}", f"{exp_q[0]}..{exp_q[-1]}",
         len(qq) == len(exp_q) and bool((qq.astype(str) == exp_q.astype(str)).all()),
         "no gap; a gap would silently change what LAG means")
    n_bad_qe = int((rds.dt.to_period("Q").dt.to_timestamp(how="end").dt.normalize()
                    != rds).sum())
    gate("grid_dates_are_quarter_ends", "grid", "n_non_quarter_end_dates",
         n_bad_qe, 0, n_bad_qe == 0, "v3.1 as-of rule")
    qdates_sql = "(" + ",".join(f"DATE '{d:%Y-%m-%d}'" for d in rds) + ")"
    print(f"    grid OK: {n:,} rows, {nf:,} firms, {nq} quarters "
          f"({qq[0]}..{qq[-1]}), {nc} countries")

    # ---------------- [3] outlier leg + exactness ----------------------------
    build_dq_sets(con, qdates_sql)
    rebuild_ict(con)
    build_group_totals(con)

    # ---------------- [4] firm table -----------------------------------------
    build_firm_table(con)

    # ---------------- [5] aggregation ----------------------------------------
    aggregate_country(con)

    # ---------------- [5b] common-FIRM-SET measures --------------------------
    cov = build_common_coverage_measures(con)

    # ---------------- [6] identity gate --------------------------------------
    gates_df = identity_gate(con)

    # ---------------- [7] assemble + merge regressors ------------------------
    print("[7] Assembling the country panel and merging regressors...")
    # NAMING CONTRACT. dW_global is the LOCKED definition W_t - W_{t-1} (the
    # own-lag leg). dW_global_sd is the sum-of-firm-deltas leg, which is what the
    # decomposition partitions. Section [6] gates them equal on every
    # fully-non-null cell; both are written so the equality stays auditable in
    # Stata rather than being taken on trust.
    base = con.sql("""
        SELECT g.ctry, g.hg, g.rd,
               CASE WHEN g.hg = 'US' THEN 1 ELSE 0 END AS us,
               g.n_firms, g.n_held, g.n_held_lag,
               g.n_w_null, g.n_wprev_null, g.n_delta_null,
               g.W            AS W_global,
               g.W_lag_ownlag AS W_global_lag,
               g.dW_ownlag    AS dW_global,
               g.dW_sumdelta  AS dW_global_sd,
               g.dW_entry, g.dW_exit, g.dW_cont, g.dW_zz,
               g.n_entry, g.n_exit, g.n_cont, g.n_zz,
               g.dW_large, g.dW_small, g.dW_new,
               g.n_large, g.n_small, g.n_new, g.med_wprev,
               g.wprev_large, g.wprev_tot,
               g.dW_top1, g.dW_top5, g.dW_absmass,
               e.W           AS W_eu,
               e.dW_ownlag   AS dW_eu,
               e.dW_sumdelta AS dW_eu_sd,
               gx.W            AS W_global_x,
               gx.dW_ownlag    AS dW_global_x,
               gx.dW_sumdelta  AS dW_global_x_sd,
               gx.dW_entry AS dW_entry_x, gx.dW_exit  AS dW_exit_x,
               gx.dW_cont  AS dW_cont_x,  gx.dW_zz    AS dW_zz_x,
               gx.dW_large AS dW_large_x, gx.dW_small AS dW_small_x,
               gx.dW_new   AS dW_new_x,   gx.dW_absmass AS dW_absmass_x,
               ex.dW_ownlag   AS dW_eu_x,
               ex.dW_sumdelta AS dW_eu_x_sd,
               u.usd, u.usd_x, u.dlog_usd, u.dlog_usd_x
        FROM agg2_g g
        JOIN agg2_e  e  USING (ctry, hg, rd)
        JOIN agg2_gx gx USING (ctry, hg, rd)
        JOIN agg2_ex ex USING (ctry, hg, rd)
        JOIN agg_usd u  USING (ctry, hg, rd)
        ORDER BY ctry, hg, rd
    """).df()
    base["rd"] = pd.to_datetime(base["rd"])
    exp_rows = len(EU_COUNTRIES) * N_GROUPS * N_Q
    gate("panel_row_count", "panel", "n_rows", len(base), exp_rows,
         len(base) == exp_rows, "28 countries x 2 groups x 100 quarters")
    dup = int(base.duplicated(["ctry", "hg", "rd"]).sum())
    gate("panel_key_uniqueness", "panel", "n_duplicate_keys", dup, 0, dup == 0, "")

    # ---- shock, taken from the SAME panel the firm-level headline uses, so the
    # ---- country and firm designs cannot drift apart.
    shk = con.sql("""SELECT DISTINCT rd, shock_us_cn AS shock,
                            shock_us_cn_lag1q AS s_lag FROM grid ORDER BY rd""").df()
    shk["rd"] = pd.to_datetime(shk["rd"])
    gate("shock_quarter_constant", "panel", "n_distinct_shock_rows", len(shk),
         N_Q, len(shk) == N_Q,
         "shock and s_lag must each be one common value per quarter")
    sk = shk.sort_values("rd").reset_index(drop=True)
    prev = sk["shock"].shift(1)
    both = prev.notna() & sk["s_lag"].notna()
    mxs = float((sk.loc[both, "s_lag"] - prev[both]).abs().max()) if both.any() else 0.0
    gate("shock_lag_is_previous_quarter", "panel", "max_abs(s_lag - shock[t-1])",
         mxs, 1e-12, mxs < 1e-12,
         "S_{t-1} is the PRIMARY timing (advisor-directed); alpha_gt absorbs it, "
         "so it enters only through the triple")
    base = base.merge(shk, on="rd", how="left", validate="many_to_one")

    # ---- M1/M2/M3 at t-1. SAME convention as build_fig_ab_data.py: the measure
    # ---- is classified at t-1 and STAMPED FORWARD one quarter onto t.
    m = pd.read_csv(MEAS_P, parse_dates=["quarter_end"])
    m = m.loc[m["is_primary"].eq(1) & m["universe"].eq(COUNTRY_UNIVERSE)].copy()
    if m.empty:
        die(f"no rows with is_primary = 1 and universe = '{COUNTRY_UNIVERSE}' in "
            f"{MEAS_P.name}")
    if m.duplicated(["country", "quarter_end"]).any():
        die(f"{MEAS_P.name} is not unique on (country, quarter_end) within the "
            f"primary universe — the t-1 stamp would fan the panel out.")
    extra = sorted(set(m["country"]) - set(EU_COUNTRIES))
    if extra:
        die(f"measures carry non-EU-28 countries: {extra}")
    m["rd"] = m["quarter_end"] + pd.offsets.QuarterEnd(1)
    m = m.rename(columns={"country": "ctry", "M1": "m1_lag", "M2": "m2_lag",
                          "M3": "m3_lag", "n_firms": "nfirm_m_lag",
                          "n_firms_m2": "nfirm_m2_lag",
                          "mcap_covered_share": "mcapcov_lag"})
    keep = ["ctry", "rd", "m1_lag", "m2_lag", "m3_lag", "nfirm_m_lag",
            "nfirm_m2_lag", "mcapcov_lag"]
    before = len(base)
    base = base.merge(m[keep], on=["ctry", "rd"], how="left",
                      validate="many_to_one")
    gate("measure_merge_no_fanout", "panel", "n_rows_after_merge", len(base),
         before, len(base) == before, "the M1/M2/M3 merge must not duplicate rows")

    # ---- SENSITIVITY (c), TWO DISTINCT OBJECTS. Do not confuse them. --------
    # m2_cov      = the OLD, WEAK restriction: the country-quarter CELLS where
    #               M2 exists. It aligns cells and NOTHING ELSE — inside a kept
    #               cell, M1 and M3 are still averaged over ALL firms while M2 is
    #               averaged over the market-cap-covered ones. Kept in the panel
    #               only so the two restrictions can be REPORTED SIDE BY SIDE and
    #               the difference between them is visible; it is NOT the (c)
    #               sensitivity any more. Renamed nowhere, relabelled everywhere.
    # m{1,2,3}cov_lag = the REAL restriction: all three measures recomputed on
    #               the SAME FIRM SET (the market-cap-covered firms inside each
    #               country-quarter), built in [5b] and stamped forward to t here
    #               with the SAME t-1 convention as m*_lag.
    base["m2_cov"] = base["m2_lag"].notna().astype("int8")
    print("    measure coverage on the panel (rows with a non-missing t-1 value):")
    for c in ("m1_lag", "m2_lag", "m3_lag"):
        print(f"      {c}: {int(base[c].notna().sum()):,} / {len(base):,} rows, "
              f"{base.loc[base[c].notna(), 'ctry'].nunique()} countries")
    census("m2_cellset_coverage", "panel", "n_rows_with_m2_cov_eq_1",
           int(base["m2_cov"].sum()),
           "CELL-set restriction only (the weak one). NOT the (c) sensitivity")

    covm = cov.rename(columns={"country": "ctry", "m1_cov": "m1cov_lag",
                               "m3_cov": "m3cov_lag", "m2_all": "m2cov_lag",
                               "n_firms_cov": "nfirm_cov_lag",
                               "n_firms_all": "nfirm_all_lag"}).copy()
    covm["rd"] = covm["quarter_end"] + pd.offsets.QuarterEnd(1)
    if covm.duplicated(["ctry", "rd"]).any():
        die("the common-coverage measures are not unique on (country, quarter) "
            "after the t-1 stamp — the merge would fan the panel out.")
    keep_cov = ["ctry", "rd", "m1cov_lag", "m2cov_lag", "m3cov_lag",
                "nfirm_cov_lag", "nfirm_all_lag"]
    before = len(base)
    base = base.merge(covm[keep_cov], on=["ctry", "rd"], how="left",
                      validate="many_to_one")
    gate("coverage_merge_no_fanout", "panel", "n_rows_after_merge", len(base),
         before, len(base) == before,
         "the common-FIRM-SET measure merge must not duplicate rows")
    # m2cov_lag IS m2_lag: M2 is already defined only over covered firms, which
    # is exactly why it needs no restriction while M1 and M3 do. Gated rather
    # than asserted, because if it ever fails the two stamps have drifted apart
    # and the (c) triple would be comparing three different vintages.
    both = base["m2_lag"].notna() & base["m2cov_lag"].notna()
    d2 = float((base.loc[both, "m2_lag"] - base.loc[both, "m2cov_lag"]).abs().max()) \
        if bool(both.any()) else 0.0
    n_na2 = int((base["m2_lag"].isna() != base["m2cov_lag"].isna()).sum())
    gate("m2cov_equals_m2", "panel", "max_abs(m2cov_lag - m2_lag)", d2, 1e-12,
         d2 < 1e-12 and n_na2 == 0,
         f"M2 needs no coverage restriction — it IS the covered-firm measure "
         f"({n_na2} missingness mismatches)")

    print("    estimation-sample census (dW_global, s_lag and the M lag present):")
    print("      [own]   = each measure on its own sample")
    print("      [cell]  = the OLD cell-set restriction (weak; aligns cells only)")
    print("      [firm]  = the COMMON FIRM SET restriction (the (c) sensitivity)")
    for mm, mc in (("m1_lag", "m1cov_lag"), ("m2_lag", "m2cov_lag"),
                   ("m3_lag", "m3cov_lag")):
        a = base.dropna(subset=["dW_global", "s_lag", mm])
        b = a[a["m2_cov"].eq(1)]
        c3 = base.dropna(subset=["dW_global", "s_lag", mc])
        print(f"      {mm}: [own] N = {len(a):,} ({a['ctry'].nunique()} c, "
              f"{a['rd'].nunique()} q) | [cell] N = {len(b):,} "
              f"({b['ctry'].nunique()} c) | [firm] N = {len(c3):,} "
              f"({c3['ctry'].nunique()} c, {c3['rd'].nunique()} q)")

    # ---- null-class flag + the per-cell identity residual, carried to Stata --
    cls = gates_df.loc[gates_df["family"].eq("global"),
                       ["ctry", "hg", "rd", "null_class"]].copy()
    cls["rd"] = pd.to_datetime(cls["rd"])
    cls["null_class"] = cls["null_class"].map(
        {"fully_non_null": 0, "all_null": 1, "partial_null": 2})
    base = base.merge(cls, on=["ctry", "hg", "rd"], how="left",
                      validate="one_to_one")
    if base["null_class"].isna().any():
        die("null_class failed to merge onto every panel row — key mismatch "
            "between the gate table and the assembled panel.")
    base["null_class"] = base["null_class"].astype("int8")
    # ident_ok = 1 when the identity HOLDS, and also when it is VACUOUS (both
    # legs missing on an all-NULL cell). It is 0 only where the two
    # constructions genuinely disagree — which section [6] has already gated to
    # be impossible on fully-non-null cells, so a 0 here can only be a
    # partial-NULL cell.
    base["ident_abs"] = (base["dW_global"] - base["dW_global_sd"]).abs()
    base["ident_ok"] = ((base["ident_abs"] < TOL_IDENT_ABS)
                        | base["ident_abs"].isna()).astype("int8")
    base["wprev_large_shr"] = (base["wprev_large"]
                               / base["wprev_tot"].where(base["wprev_tot"] > 0))

    # ---------------- [8] write ----------------------------------------------
    print("[8] Writing outputs...")
    # COLUMN CONTRACT (fixed 2026-08-10). Every e2 consumer — _country_weight_lib.do
    # (cw_prep), run_ri_country_ddd.py (REQUIRED_COLS) and, through the import,
    # run_country_decomposition.py — reads sec_country / holder_group / rdate /
    # dw_global / dw_eu, all lower case. Stata and pyreadstat are both
    # case-sensitive, so an earlier vintage that emitted ctry / hgroup / rd /
    # dW_global made every consumer die on first contact. e1 moves, because three
    # consumer files plus the library document the contract. Lowercasing the whole
    # frame is collision-free here: W_global -> w_global and dW_global ->
    # dw_global stay distinct, and the same for the _eu / _x twins (asserted below).
    out = base.rename(columns={"ctry": "sec_country", "hg": "holder_group",
                               "rd": "rdate"}).copy()
    lowered = [c.lower() for c in out.columns]
    if len(set(lowered)) != len(lowered):
        dupes = sorted({c for c in lowered if lowered.count(c) > 1})
        die(f"lowercasing the panel would collide on {dupes} — rename in the "
            f"[7] assembly block instead of masking it here.")
    out.columns = lowered
    KEYS = ("sec_country", "holder_group", "rdate")
    for c in KEYS:
        if c not in out.columns:
            die(f"column contract broken: {c} absent after the rename.")
    int8_cols = ("us", "m2_cov", "null_class", "ident_ok")
    int32_cols = ("n_firms", "n_held", "n_held_lag", "n_w_null", "n_wprev_null",
                  "n_delta_null", "n_entry", "n_exit", "n_cont", "n_zz",
                  "n_large", "n_small", "n_new")
    for c in out.columns:
        if c in KEYS:
            continue
        out[c] = pd.to_numeric(out[c], errors="raise")
        if c in int8_cols:
            out[c] = out[c].astype("int8")
        elif c in int32_cols:
            out[c] = out[c].astype("int32")
        else:
            out[c] = out[c].astype("float64")
    bad_names = [c for c in out.columns if len(c) > 32]
    if bad_names:
        die(f"Stata variable names longer than 32 characters: {bad_names}")
    out.to_stata(DTA, write_index=False, convert_dates={"rdate": "tc"},
                 version=118)
    print(f"    wrote {DTA.name}: {len(out):,} rows x {len(out.columns)} columns")

    # ---- firm-level WLS weights (dependency of run_firm_wls_heterogeneity.do) --
    # NAMED DEPENDENCY, now delivered rather than left for the consumer to fake.
    # audit_c6_panel.dta carries no level or lagged-level column (
    # build_audit_panel_f1f2f7.py projects w / w_prev only to form cum0..cum4 and
    # drops them), so the lagged-holdings weight has to come from here, where the
    # firm table already exists. Two flavours, because "lagged holdings" is
    # ambiguous: the lagged PORTFOLIO WEIGHT and the lagged DOLLAR position.
    # Both are the CANONICAL (non-twin) quantities, matching the firm-level
    # primary spec the diagnostic weights.
    wls = con.sql("""
        SELECT fid AS firm_str, hg AS hgroup, rd AS rdate,
               wp_g AS w_prev,
               LAG(i_can, 1) OVER (PARTITION BY fid, hg ORDER BY rd) AS i_prev_usd
        FROM firmx ORDER BY fid, hg, rd
    """).df()
    wls["rdate"] = pd.to_datetime(wls["rdate"])
    dupw = int(wls.duplicated(["firm_str", "hgroup", "rdate"]).sum())
    gate("wls_weight_key_uniqueness", "panel", "n_duplicate_weight_keys", dupw,
         0, dupw == 0,
         "run_firm_wls_heterogeneity.do merges m:1 on (firm_str, hgroup, rdate)")
    gate("wls_weight_row_count", "panel", "n_rows", len(wls), n, len(wls) == n,
         "one weight row per firm x group x quarter of the canonical grid")
    for c in ("w_prev", "i_prev_usd"):
        wls[c] = pd.to_numeric(wls[c], errors="raise").astype("float64")
    wls.to_stata(F_WLSW, write_index=False, convert_dates={"rdate": "tc"},
                 version=118)
    n_zero_w = int((wls["w_prev"] == 0).sum())
    n_null_w = int(wls["w_prev"].isna().sum())
    print(f"    wrote {F_WLSW.name}: {len(wls):,} rows "
          f"(w_prev == 0 on {n_zero_w:,} rows = the ENTRY margin an aweight "
          f"deletes; w_prev NULL on {n_null_w:,} = the first grid quarter)")

    gates_df.to_csv(F_GATE, index=False)
    print(f"    wrote {F_GATE.name}: {len(gates_df):,} rows "
          f"({len(FAMILIES)} weight families x {exp_rows:,} cells)")

    gsum = pd.DataFrame(GATES)
    gsum.to_csv(F_SUMMARY, index=False)
    n_hard = int((gsum["severity"] == "gate").sum())
    print(f"    wrote {F_SUMMARY.name}: {len(gsum)} rows "
          f"({n_hard} hard gates, all passing — the script would have aborted "
          f"otherwise)")

    orow = con.sql("SELECT * FROM outlier_rows ORDER BY adj_mv DESC").df()
    orow.to_csv(F_OUTLIER, index=False)
    print(f"    wrote {F_OUTLIER.name}: {len(orow):,} excluded rows "
          f"(sensitivity (b), the proxy-free leg)")

    dparts = []
    for fam, tot, absc, cols in (
            ("global", "dW_global_sd", "dW_absmass",
             ("dW_entry", "dW_exit", "dW_cont", "dW_zz",
              "dW_large", "dW_small", "dW_new")),
            ("global_outlier_excluded", "dW_global_x_sd", "dW_absmass_x",
             ("dW_entry_x", "dW_exit_x", "dW_cont_x", "dW_zz_x",
              "dW_large_x", "dW_small_x", "dW_new_x"))):
        for c in cols:
            comp = c.split("_")[1]
            part = base[["ctry", "hg", "rd"]].copy()
            part["family"] = fam
            part["partition"] = ("margin" if comp in ("entry", "exit", "cont", "zz")
                                 else "size")
            part["component"] = comp
            part["contribution"] = base[c].to_numpy()
            part["dW_total"] = base[tot].to_numpy()
            part["dW_absmass"] = base[absc].to_numpy()
            part["share_of_absmass"] = (part["contribution"]
                                        / part["dW_absmass"].where(
                                            part["dW_absmass"] > 0))
            # the tie rule travels WITH the numbers, so a reader comparing this
            # artifact to run_country_decomposition.py's does not have to infer
            # which side of the median "large" sits on.
            part["size_tie_rule"] = (SIZE_TIE_RULE if part["partition"].iloc[0]
                                     == "size" else "n/a (margin partition)")
            dparts.append(part)
    dec = pd.concat(dparts, ignore_index=True)
    dec.to_csv(F_DECOMP, index=False)
    print(f"    wrote {F_DECOMP.name}: {len(dec):,} rows "
          f"(size_tie_rule = {SIZE_TIE_RULE})")

    print("\n    DECOMPOSITION — mean |contribution| by component, global family, "
          "by holder group (which component carries the country movement):")
    d0 = dec[dec["family"].eq("global")]
    piv = (d0.assign(a=lambda x: x["contribution"].abs())
             .pivot_table(index=["partition", "component"], columns="hg",
                          values="a", aggfunc="mean"))
    print(piv.to_string(float_format=lambda x: f"{x:.4e}"))

    con.close()
    print("\nDONE. Consumers: the country DDD runner (three pairwise FE cg / ct / "
          "gt, two-way cluster country x quarter) and the circular-shift RI "
          "arbiter both read country_weight_panel.dta.")


if __name__ == "__main__":
    sys.exit(main())
