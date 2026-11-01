"""
run_country_decomposition.py — margin and size decomposition of the country
portfolio-weight change (task e2 item 4, 2026-08-10).  CODE ONLY at time of
writing: nothing here has been executed.

================================================================================
WHY THIS EXISTS, AND WHAT IT REPLACES
================================================================================
It replaces the WITHDRAWN claim that "a firm-level WLS reproduces the country
result".  That claim is FALSE and must not reappear: the country outcome is a
SUM of firm deltas,

    dW_{c,g,t} = SUM_{i in c} dw_{i,g,t},

while WLS minimises a weighted sum of squared residuals.  A weighted regression
equals an aggregated regression only under conditions this design does not
satisfy (it would need, among other things, the within-cell design matrix to be
identical across cells and the weights to be exactly the aggregation weights of
a linear aggregator with no residual covariance structure).  The honest object
is the one below: an EXACT ADDITIVE DECOMPOSITION of the country outcome, and
the coefficient each piece carries.

================================================================================
TWO EXHAUSTIVE PARTITIONS OF dW
================================================================================
Both are computed at the (country, group, quarter) cell and both sum EXACTLY to
dW (hard-gated, not asserted in prose):

 P1 MARGIN
   entry       w_{i,t-1} = 0 and w_{i,t} > 0     contribution =  w_{i,t}
   exit        w_{i,t-1} > 0 and w_{i,t} = 0     contribution = -w_{i,t-1}
   continuing  w_{i,t-1} > 0 and w_{i,t} > 0     contribution =  w_{i,t} - w_{i,t-1}
   inactive    both zero                          contribution =  0  (must be exactly 0)

 P2 SIZE (by LAGGED weight within the country-group-quarter cell)
   large       w_{i,t-1} >  median of the cell's POSITIVE lagged weights
   small       0 < w_{i,t-1} <= that median
   entry       w_{i,t-1} = 0            (an entrant has no lagged size; it gets
                                         its own bucket rather than being
                                         forced into "small", which would
                                         silently merge the entry margin into
                                         the size story)

Median split rather than a top-decile split: cells are small (a country-quarter
can hold a handful of firms) and a decile cut would be undefined or a single
firm in many of them.  The cell median is reported alongside so the cut is
auditable.

MEDIAN-TIE CONVENTION (CHANGED 2026-08-10 — external review item 9).  This file
used to allocate ties the OTHER way (large = wp >= median, small = wp < median)
while build_country_weight_panel.py used large = wp > median.  Both rules are
exhaustive and disjoint, so the only firms that move are those sitting EXACTLY
on the cell median — but median ties are COMMON in small country cells (with an
ODD number of positive lagged weights the median IS one firm's own weight, and
cells here are frequently tiny), so the two artifacts were labelling different
firms "large" while being read side by side.  The PROJECT-WIDE rule is now the
one in build_country_weight_panel.py:

    large = wp > median ,   small = 0 < wp <= median

carried here as SIZE_TIE_RULE, IMPORTED from that file and gated equal at run
time (`_gate_size_tie_rule`), and stamped into both output CSVs so no reader has
to infer it.  Why this side: under `>=' a ONE-FIRM cell reports large = 1,
small = 0 — the cell's only, and therefore also smallest, firm is called large.
Under `>' it reports large = 0, small = 1, which invents no large firm.
WHAT IT CHANGES IN THE NUMBERS: c_large loses, and c_small gains, exactly the
median-tied firms' deltas; the two buckets still sum to dW, so no identity and
no total moves.

================================================================================
COEFFICIENT ATTRIBUTION — THE METHOD, STATED
================================================================================
The DDD is re-run with the outcome REPLACED BY EACH COMPONENT, holding the
regressors, the FE and the sample fixed.  OLS is linear in the outcome and the
components sum to dW, so

    beta(dW) = SUM_j beta(component_j)     EXACTLY.

That identity is CHECKED at 1e-8 (`beta_sum_gate`), and the "share" of the
coefficient carried by component j is beta_j / beta(dW).

SHARE-DENOMINATOR WARNING, ENFORCED IN THE OUTPUT.  The headline coefficient is
a deep null.  A share whose denominator is statistically indistinguishable from
zero is not interpretable — it can be 300% or -1200% from noise alone.  So every
share is emitted together with `share_interpretable`, which is 0 whenever
|beta(dW)| < 2 * se(beta(dW)).  When it is 0 the ABSOLUTE component betas are
the reportable objects and the share column must not be quoted.

================================================================================
IDENTITY GATE AND NULL SEMANTICS
================================================================================
A firm whose delta is NULL breaks the aggregation identity.  Cells are therefore
split:
  * fully-non-null cells — SUM(components) must equal the e1 panel's dw_global
    to float tolerance (hard gate, aborts);
  * cells containing at least one NULL-delta firm — counted, listed by quarter,
    and EXCLUDED from the gate but NOT silently dropped from the report.
This is an independent re-derivation of e1's identity gate: it is built here
from the firm-level parquet, not read from e1's output.

INPUT : output/merged_us_eu_zero_filled.parquet   (firm x group x quarter)
        output/country_weight_panel.dta           (e1; regressors + dw_global)
OUTPUT: output/country_weight_decomp_shares.csv   (descriptive, per country + pooled)
        output/country_weight_decomp_beta.csv     (component betas + shares)
Rotation discipline: refuses to overwrite; rotate to *_cwpre first.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The estimator is DEFINED ONCE, in the RI script, and imported here so the
# decomposition regressions are bit-for-bit the same FE/vce as the arbiter.
from run_ri_country_ddd import (  # noqa: E402
    MEASURES,
    estimation_sample,
    fit_from_raw,
    load_country_panel,
    rotation_guard,
)

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")
GRID = OUT / "merged_us_eu_zero_filled.parquet"
PANEL_DTA = OUT / "country_weight_panel.dta"
SHARES_CSV = OUT / "country_weight_decomp_shares.csv"
BETA_CSV = OUT / "country_weight_decomp_beta.csv"

IDENTITY_ATOL = 1e-12
BETA_SUM_RELTOL = 1e-8

MARGIN_COMPS = ["entry", "exit", "continuing", "inactive"]
SIZE_COMPS = ["large", "small", "entry_size"]

# THE ONE SIZE-SPLIT TIE CONVENTION.  Must be character-for-character the literal
# in build_country_weight_panel.py; `_gate_size_tie_rule` imports that file and
# aborts on any difference, so the two artifacts can never again describe
# "large" differently while being read side by side.
SIZE_TIE_RULE = "large = wp > cell_median(wp | wp>0); small = 0 < wp <= median; new = wp == 0"

# PROVENANCE: this script's own bytes and every input artifact's bytes.  mtimes
# cannot serve here (OneDrive rewrites them; output/ is a junction to E:).
SELF_PATH = Path(__file__).resolve()
HASH_CHUNK = 1 << 22


def sha256_file(p: Path) -> tuple[str, int]:
    p = Path(p)
    if not p.is_file():
        return "MISSING", 0
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(HASH_CHUNK), b""):
            h.update(blk)
            n += len(blk)
    return h.hexdigest(), n


def log_provenance(inputs: dict[str, Path]) -> dict[str, str]:
    print("PROVENANCE — SHA256 of this script and of every input artifact")
    out = {}
    dg, sz = sha256_file(SELF_PATH)
    out["script_sha256"] = dg
    print(f"  SCRIPT  {SELF_PATH.name:<38s} {dg}  ({sz:,} B)")
    for label, p in inputs.items():
        dg, sz = sha256_file(p)
        out[label] = dg
        print(f"  INPUT   {Path(p).name:<38s} {dg}  ({sz:,} B)")
    # the estimator is imported, so ITS bytes are part of this run too
    est = SELF_PATH.with_name("run_ri_country_ddd.py")
    dg, sz = sha256_file(est)
    out["estimator_sha256"] = dg
    print(f"  IMPORT  {est.name:<38s} {dg}  ({sz:,} B)")
    return out


def _gate_size_tie_rule() -> None:
    """The tie convention must be IDENTICAL to e1's, not merely similar."""
    try:
        from build_country_weight_panel import SIZE_TIE_RULE as E1_RULE
    except Exception as exc:                       # noqa: BLE001
        raise SystemExit(
            f"cannot import SIZE_TIE_RULE from build_country_weight_panel.py "
            f"({exc!r}).  That literal is the single definition of the size "
            "split; without it this script cannot prove it is bucketing firms "
            "the same way the panel does.  Refusing to decompose.") from exc
    if E1_RULE != SIZE_TIE_RULE:
        raise SystemExit(
            "SIZE-TIE RULE DRIFT — the two implementations of the size split "
            f"disagree.\n  build_country_weight_panel.py: {E1_RULE}\n"
            f"  run_country_decomposition.py : {SIZE_TIE_RULE}\n"
            "Median ties are common in small country cells, so this is not "
            "cosmetic: the two artifacts would call different firms 'large'. "
            "Fix the SQL in BOTH files, not just the literal.")
    print(f"size-split tie rule (gated equal to e1's): {SIZE_TIE_RULE}")


# =============================================================================
# firm-level component aggregation
# =============================================================================
def build_components() -> pd.DataFrame:
    """Aggregate firm deltas into the two partitions at (country, group, quarter).

    NULL semantics: a firm with a NULL delta_w_global cannot be assigned to any
    component, so it is counted (n_null_firms) and the cell is flagged rather
    than being treated as a zero.
    """
    import duckdb

    if not GRID.exists():
        raise SystemExit(f"missing firm-level grid: {GRID}")

    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    tmp = Path("E:/duckdb_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")

    uri = GRID.as_posix()
    # cell median of POSITIVE lagged weights -> the large/small cut
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cut AS
        SELECT sec_country, holder_group, report_date,
               median(w_prev_global) FILTER (WHERE w_prev_global > 0) AS w_prev_med
        FROM read_parquet('{uri}')
        GROUP BY 1, 2, 3
    """)

    comp = con.execute(f"""
        WITH f AS (
            SELECT m.sec_country, m.holder_group, m.report_date,
                   m.delta_w_global      AS d,
                   m.portfolio_weight_global AS w,
                   m.w_prev_global       AS wp,
                   c.w_prev_med          AS med
            FROM read_parquet('{uri}') m
            JOIN cut c USING (sec_country, holder_group, report_date)
        )
        SELECT sec_country, holder_group, report_date,
               COUNT(*)                                    AS n_firms,
               COUNT(*) FILTER (WHERE d IS NULL)           AS n_null_firms,
               any_value(med)                              AS w_prev_median,
               -- ---------- P1 margin ----------
               COALESCE(SUM(d) FILTER (WHERE d IS NOT NULL
                    AND COALESCE(wp,0) = 0 AND w > 0), 0)                AS c_entry,
               COALESCE(SUM(d) FILTER (WHERE d IS NOT NULL
                    AND wp > 0 AND COALESCE(w,0) = 0), 0)                AS c_exit,
               COALESCE(SUM(d) FILTER (WHERE d IS NOT NULL
                    AND wp > 0 AND w > 0), 0)                            AS c_continuing,
               COALESCE(SUM(d) FILTER (WHERE d IS NOT NULL
                    AND COALESCE(wp,0) = 0 AND COALESCE(w,0) = 0), 0)    AS c_inactive,
               -- ---------- P2 size (by LAGGED weight) ----------
               -- TIE RULE, unified with build_country_weight_panel.py:
               --   large = wp >  median ,  small = 0 < wp <= median
               -- (see SIZE_TIE_RULE and the docstring block; changed 2026-08-10)
               COALESCE(SUM(d) FILTER (WHERE d IS NOT NULL
                    AND wp > 0 AND wp >  med), 0)                        AS c_large,
               COALESCE(SUM(d) FILTER (WHERE d IS NOT NULL
                    AND wp > 0 AND wp <= med), 0)                        AS c_small,
               COALESCE(SUM(d) FILTER (WHERE d IS NOT NULL
                    AND COALESCE(wp,0) = 0), 0)                          AS c_entry_size,
               -- ---------- counts, for the descriptive table ----------
               COUNT(*) FILTER (WHERE COALESCE(wp,0) = 0 AND w > 0)      AS n_entry,
               COUNT(*) FILTER (WHERE wp > 0 AND COALESCE(w,0) = 0)      AS n_exit,
               COUNT(*) FILTER (WHERE wp > 0 AND w > 0)                  AS n_continuing,
               COALESCE(SUM(d) FILTER (WHERE d IS NOT NULL), 0)          AS dw_sum_firmdelta
        FROM f
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
    """).df()
    con.close()

    comp["report_date"] = pd.to_datetime(comp["report_date"])
    comp["qper"] = comp["report_date"].dt.to_period("Q")
    comp = comp.drop(columns=["report_date"])
    # rename anything that could collide with an e1 panel column on the merge —
    # a silent _x/_y suffix would make the descriptive block read the wrong
    # column, which is precisely the class of bug this project keeps hitting.
    comp = comp.rename(columns={"n_firms": "n_firms_cell"})

    # ---- within-partition additivity (arithmetic, must hold everywhere) ----
    m_sum = comp[["c_entry", "c_exit", "c_continuing", "c_inactive"]].sum(axis=1)
    s_sum = comp[["c_large", "c_small", "c_entry_size"]].sum(axis=1)
    for lbl, v in (("margin", m_sum), ("size", s_sum)):
        bad = (v - comp["dw_sum_firmdelta"]).abs() > IDENTITY_ATOL
        if bad.any():
            raise SystemExit(
                f"{lbl} partition does not sum to the firm-delta total in "
                f"{int(bad.sum())} cells (max gap "
                f"{float((v - comp['dw_sum_firmdelta']).abs().max()):.3e}). "
                "The buckets are not exhaustive/disjoint — fix the SQL, do not "
                "widen the tolerance.")
    if (comp["c_inactive"].abs() > IDENTITY_ATOL).any():
        raise SystemExit(
            "the `inactive' bucket (w_{t-1}=0 and w_t=0) is not exactly zero — "
            "a firm with no position on either side is contributing to dW, "
            "which means the zero-fill or the delta column is wrong.")
    return comp


# =============================================================================
# identity gate against e1's panel
# =============================================================================
def identity_gate(panel: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    """SUM_i dw_i must equal the panel's dW cell by cell on fully-non-null cells.

    Independent re-derivation of e1's gate: built here from the firm grid.
    """
    key = ["sec_country", "holder_group", "qper"]
    p = panel[key + ["dw_global"]].copy()
    overlap = (set(comp.columns) & set(p.columns)) - set(key)
    if overlap:
        raise SystemExit(
            f"column name collision on the identity merge: {sorted(overlap)} — "
            "rename in build_components() rather than letting pandas append "
            "_x/_y suffixes.")
    merged = p.merge(comp, on=key, how="left", validate="1:1", indicator=True)
    n_unmatched = int((merged["_merge"] != "both").sum())
    if n_unmatched:
        raise SystemExit(
            f"{n_unmatched} panel cells have no firm-level counterpart — the e1 "
            "panel and the firm grid disagree on the (country, group, quarter) "
            "universe.  Refusing to decompose across a mismatched key.")
    merged = merged.drop(columns="_merge")

    clean = merged["n_null_firms"] == 0
    d = (merged["dw_sum_firmdelta"] - merged["dw_global"]).abs()
    d_clean = d[clean & merged["dw_global"].notna()]
    rel = (d / merged["dw_global"].abs().replace(0, np.nan))[clean & merged["dw_global"].notna()]
    print("\n--- IDENTITY GATE: dW (panel) vs SUM_i dw_i (firm grid) ---")
    print(f"  cells                         : {len(merged):,}")
    print(f"  fully-non-null cells          : {int(clean.sum()):,}")
    print(f"  cells with >=1 NULL-delta firm: {int((~clean).sum()):,} "
          f"(EXCLUDED from the gate, NOT dropped from the report)")
    if len(d_clean):
        print(f"  max abs discrepancy (clean)   : {float(d_clean.max()):.3e}")
        print(f"  max rel discrepancy (clean)   : {float(rel.max()):.3e}")
        if float(d_clean.max()) > IDENTITY_ATOL:
            worst = merged.loc[d_clean.idxmax(), ["sec_country", "holder_group", "qper"]]
            raise SystemExit(
                f"IDENTITY GATE FAILED: max abs discrepancy "
                f"{float(d_clean.max()):.3e} > {IDENTITY_ATOL:.0e} on a "
                f"fully-non-null cell (worst: {worst.to_dict()}).  dW is not the "
                "sum of firm deltas — the decomposition would be attributing "
                "movement that the outcome does not contain.")
    if (~clean).any():
        byq = (merged.loc[~clean].groupby("qper")["n_null_firms"]
               .agg(["size", "sum"]).rename(columns={"size": "cells",
                                                     "sum": "null_firms"}))
        print("  NULL-firm cells by quarter (first 10):")
        print(byq.head(10).to_string())
    return merged


# =============================================================================
# main
# =============================================================================
def main():
    rotation_guard(SHARES_CSV)
    rotation_guard(BETA_CSV)
    prov = log_provenance({"grid_sha256": GRID, "panel_sha256": PANEL_DTA})
    _gate_size_tie_rule()

    panel = load_country_panel()
    comp = build_components()
    merged = identity_gate(panel, comp)

    comp_cols = {"entry": "c_entry", "exit": "c_exit",
                 "continuing": "c_continuing", "inactive": "c_inactive",
                 "large": "c_large", "small": "c_small",
                 "entry_size": "c_entry_size"}

    # ------------------------------------------------------------------
    # [1] DESCRIPTIVE: which component carries the movement?
    #     Two scales, because they answer different questions:
    #       share_of_gross = SUM|comp| / SUM_j SUM|comp_j|  -> who MOVES
    #       net_sum        = SUM comp                       -> who DRIFTS
    # ------------------------------------------------------------------
    rows = []
    for scope, frame in [("pooled", merged)] + [
            (c, g) for c, g in merged.groupby("sec_country")]:
        for partition, comps in (("margin", MARGIN_COMPS), ("size", SIZE_COMPS)):
            gross = {c: float(frame[comp_cols[c]].abs().sum()) for c in comps}
            tot_gross = sum(gross.values())
            for c in comps:
                rows.append({
                    "scope": scope,
                    "partition": partition,
                    "component": c,
                    "gross_abs_sum": gross[c],
                    "share_of_gross": gross[c] / tot_gross if tot_gross > 0 else np.nan,
                    "net_sum": float(frame[comp_cols[c]].sum()),
                    "mean": float(frame[comp_cols[c]].mean()),
                    "sd": float(frame[comp_cols[c]].std(ddof=1)),
                    "n_cells": int(len(frame)),
                    "n_firms_mean": float(frame["n_firms_cell"].mean()),
                    "n_entry_mean": float(frame["n_entry"].mean()),
                    "n_exit_mean": float(frame["n_exit"].mean()),
                    # the tie rule travels WITH the numbers
                    "size_tie_rule": (SIZE_TIE_RULE if partition == "size"
                                      else "n/a (margin partition)"),
                    "script_sha256": prov["script_sha256"],
                    "grid_sha256": prov["grid_sha256"],
                    "panel_sha256": prov["panel_sha256"],
                })
    shares = pd.DataFrame(rows)
    shares.to_csv(SHARES_CSV, index=False)
    print(f"\nwrote {SHARES_CSV.name}")
    print(shares[shares["scope"] == "pooled"][
        ["partition", "component", "share_of_gross", "net_sum"]].to_string(index=False))

    # ------------------------------------------------------------------
    # [2] COEFFICIENT ATTRIBUTION: re-run the DDD with y = each component.
    #     Additive because the components sum to dW and OLS is linear in y.
    # ------------------------------------------------------------------
    # attach the components to the regressor panel, keyed identically
    key = ["sec_country", "holder_group", "qper"]
    panel2 = panel.merge(merged[key + list(comp_cols.values())], on=key,
                         how="left", validate="1:1")

    beta_rows = []
    print("\n--- COEFFICIENT ATTRIBUTION (outcome replaced by each component) ---")
    for m in MEASURES:
        sub = estimation_sample(panel2, m, extra=list(comp_cols.values()))
        q = sub["q_code"].to_numpy()
        S = panel.attrs["S_by_quarter"][np.sort(sub["q_orig"].unique())]
        x2 = sub["us"].to_numpy(float) * sub["m_lag"].to_numpy(float)
        x3 = x2 * S[q]

        tot = fit_from_raw(sub["dw_global"].to_numpy(float), x2, x3, sub)
        b_tot, se_tot = tot["b"], tot["se"]
        interpretable = int(np.isfinite(se_tot) and abs(b_tot) >= 2.0 * se_tot)
        if not interpretable:
            print(f"  [{m}] |b(dW)|={abs(b_tot):.3e} < 2*se={2*se_tot:.3e} — SHARES "
                  "ARE NOT INTERPRETABLE for this measure; report the absolute "
                  "component betas instead.")

        for partition, comps in (("margin", MARGIN_COMPS), ("size", SIZE_COMPS)):
            bsum = 0.0
            for c in comps:
                y = sub[comp_cols[c]].to_numpy(float)
                f = fit_from_raw(y, x2, x3, sub)
                bsum += f["b"]
                beta_rows.append({
                    "measure": m, "partition": partition, "component": c,
                    "b_component": f["b"], "se_component": f["se"],
                    "t_component": f["t"],
                    "b_total": b_tot, "se_total": se_tot,
                    "share_of_beta": f["b"] / b_tot if b_tot != 0 else np.nan,
                    "share_interpretable": interpretable,
                    "N": f["n"],
                    "method": ("DDD re-run with y = component; additive because "
                               "the components sum to dW and OLS is linear in y"),
                    "size_tie_rule": (SIZE_TIE_RULE if partition == "size"
                                      else "n/a (margin partition)"),
                    # se/t here are on the CONSERVATIVE small-sample convention
                    # (SE_CONVENTION_DEFAULT = "all", the raw parameter count).
                    # The arbiter pins its convention against Stata per measure;
                    # this file only uses se for the share-interpretability rule,
                    # where the larger SE is the cautious choice.
                    "se_convention": f.get("se_convention", "all"),
                    "script_sha256": prov["script_sha256"],
                    "estimator_sha256": prov["estimator_sha256"],
                    "panel_sha256": prov["panel_sha256"],
                })
            rel = abs(bsum / b_tot - 1.0) if b_tot != 0 else abs(bsum)
            print(f"  [{m}/{partition}] SUM_j beta_j = {bsum:+.6e} vs beta(dW) = "
                  f"{b_tot:+.6e}  (rel {rel:.2e})")
            if not rel < BETA_SUM_RELTOL:
                raise SystemExit(
                    f"ADDITIVITY GATE FAILED for {m}/{partition}: component betas "
                    f"sum to {bsum:.6e} but beta(dW) = {b_tot:.6e} (rel "
                    f"{rel:.2e} >= {BETA_SUM_RELTOL:.0e}).  Since OLS is linear "
                    "in y this can only mean the components do not sum to the "
                    "outcome ON THE ESTIMATION SAMPLE — check the merge and the "
                    "listwise deletion.")
            beta_rows.append({
                "measure": m, "partition": partition, "component": "__SUM_CHECK__",
                "b_component": bsum, "se_component": np.nan, "t_component": np.nan,
                "b_total": b_tot, "se_total": se_tot,
                "share_of_beta": bsum / b_tot if b_tot != 0 else np.nan,
                "share_interpretable": interpretable, "N": tot["n"],
                "method": f"additivity gate: rel diff {rel:.3e} < {BETA_SUM_RELTOL:.0e}",
                "size_tie_rule": (SIZE_TIE_RULE if partition == "size"
                                  else "n/a (margin partition)"),
                "se_convention": tot.get("se_convention", "all"),
                "script_sha256": prov["script_sha256"],
                "estimator_sha256": prov["estimator_sha256"],
                "panel_sha256": prov["panel_sha256"],
            })

    beta = pd.DataFrame(beta_rows)
    beta.to_csv(BETA_CSV, index=False)
    print(f"\nwrote {BETA_CSV.name}")

    print("\n" + "=" * 92)
    print("READING RULES stamped with these artifacts")
    print("=" * 92)
    print(" * The decomposition is EXACT: both partitions sum to dW cell by cell,")
    print("   and the component betas sum to beta(dW) by linearity of OLS.  Both")
    print("   identities are gated, not asserted.")
    print(" * share_of_beta is only meaningful when share_interpretable == 1.  A")
    print("   share whose denominator is a null coefficient is noise divided by")
    print("   noise; quote the absolute component betas instead.")
    print(" * This file does NOT reconcile anything with a firm-level WLS.  The")
    print("   country outcome is a SUM of firm deltas; a WLS minimises a weighted")
    print("   SSR.  They are different objects and the withdrawn 'WLS reproduces")
    print("   the country result' claim stays withdrawn.")
    print(" * Cells containing a NULL-delta firm are excluded from the identity")
    print("   gate and counted in the log; they are not silently dropped.")
    print(f" * SIZE TIE RULE: {SIZE_TIE_RULE}.  Unified 2026-08-10 with")
    print("   build_country_weight_panel.py and gated equal at run time; it is")
    print("   stamped in the size_tie_rule column of both CSVs.  Median-tied")
    print("   firms count as SMALL, in both artifacts.")
    print(" * PROVENANCE: script / estimator / input SHA256 are printed above and")
    print("   written into every row, so these numbers name the bytes that made")
    print("   them.")


if __name__ == "__main__":
    main()
