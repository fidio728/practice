"""
run_ri_shockmenu.py — randomization inference for the SHOCK MENU (P1a/P1b/P1c),
the design-based ARBITER for run_shock_menu.do.

TIMING NOTE (updated 2026-08-10; SUPERSEDES the 2026-08-09 "stays on S_t" note).
The menu was pre-registered at S_t, but the paper's PRIMARY timing is S_{t-1}
(run_headline_3pairwise.do, 2026-08-08 decision), and construction robustness
must be established AT the primary timing. This file therefore runs BOTH arms
and tags every output row with timing in {st, slag}:
  st    the pre-registered contemporaneous arm — UNCHANGED. Permutes the S_t
        variant vectors; its baseline anchors the canonical diag/global/st cell.
  slag  the primary-timing arm: for every variant the shock vector is the TRUE
        CALENDAR LAG L.S on the CONTIGUOUS quarter grid (menu sorted by
        quarter, contiguity asserted, positional shift(1) — the same
        derivation as the s_lag map in run_headline_3pairwise.do Part B, and
        the same ordering as run_shock_menu.do: lag FIRST on the full menu
        grid, THEN restrict to panel quarters). RI permutes / circular-shifts
        the LAGGED vector on its covered quarters. The slag BASELINE is the
        PRIMARY regression itself (us x cn_lag x S_{t-1}), so its drift-gate
        anchor is the canonical primary/global/slag cell — NOT diag/global/st;
        that wiring is explicit in CANON_ANCHORS below.
This closes the "menu not re-adjudicated at the primary timing" gap. The
primary-spec RI arbiters proper remain run_ri_3pairwise.py /
run_cum4_inference.py / run_ri_direction.py / run_ri_fourgroup.py /
run_ri_tercile.py; the slag arm here is the menu's construction-robustness leg
AT that timing, not a replacement for them. Family size doubles relative to the
pre-registered S_t battery, so the pre-registered reading's false-positive
arithmetic (item e) applies WITHIN each timing arm.

WHAT THIS IS. The headline shock S_t is the monthly level-AR(1) residual of the
Iacoviello-Tong bilateral USA|China GPR sampled at the quarter-end month. Three
documented defects motivate a menu of alternative constructions: (P1a) the
quarterly S_t is serially correlated (acf1=+0.271, LB(1) p=0.013, measured), so
free permutation of the quarter shocks is ANTI-CONSERVATIVE and the circular-shift
p is the serial-robust arbiter; (P1b) USA|China is DIRECTIONAL while the
hypothesis is about US-China tension broadly; (P1c) the AR is fit on the FULL
sample, i.e. look-ahead in a generated regressor. build_shock_menu.py builds the
variants and standardizes each over the 82 panel quarters; run_shock_menu.do runs
the CRVE battery; this file runs the design-based test for every variant.

COLLAPSE (mirrors run_ri_3pairwise.py / run_ri_extensive.py). On the balanced
2-per-(firm,quarter) grid the three pairwise FE (it firm x quarter + gt group x
quarter + ig firm x group) collapse, on the US-minus-NONUS within-firm-quarter
difference Delta y, to a FIRM + QUARTER two-way FE regression on the difference
panel:

    Delta y_it = b2 * cn_it + b3 * (cn_it * S_t) + phi_i + delta_t + e_it

There is no closed-form sufficient statistic (firm-demeaning of cn*S mixes S
across a firm's quarters), so cn*S is two-way-demeaned via fast bincount
alternating projections; b3 via FWL against the (once-)demeaned Delta y and cn.

REUSE ACROSS VARIANTS. The collapse is IDENTICAL for every variant that shares a
row set — Delta y and cn are fixed, only the quarter vector S changes. Two
consequences are exploited:
  (1) demean(Delta y) and demean(cn) are computed ONCE per row set;
  (2) the demeaner (fixed firm/quarter codes, fixed iteration count) is a LINEAR
      operator, and cn*S = sum_t S_t * (cn . 1[q=t]), so
          demean(cn*S) = B @ S,  B[:, t] = demean(cn . 1[q=t]).
      B is built once per row set. Every b3 evaluation then reduces to
          b   = (u . S) / cn_dot                      u = B' cn
          b3  = (v . S - b * y_cn) / (S' G S - 2 b (u . S) + b^2 cn_dot)
                                                      v = B' y,  G = B' B
      which is algebraically IDENTICAL to the row-level route, not an
      approximation. A SELF-CHECK re-derives the observed b3 the slow row-level
      way and aborts if the two disagree beyond 1e-8 relative — so the speedup
      can never silently change a number.
The slag VARIANT rows drop the first covered quarter (their in-file lag is
missing by construction), so they form their own row set with its own Collapsed
instance — handled by the existing missing-pattern grouping, disclosed, never
hidden. The slag BASELINE row does NOT drop it (first-run fix 2026-08-10): the
panel's own s_lag column carries the Q1 value from the pre-panel contiguous GPR
series, and the canonical primary cell it anchors against was estimated on that
FULL panel — a positional in-file lag can never reproduce it (measured: relerr
3.5e-2, drift-gate abort). The panel column is cross-checked against the
positional lag on every derivable quarter before use.

TESTS PER (VARIANT, TIMING) (headline 3-pairwise only, DV = dw):
  free permutation  : N_PERM draws of the covered quarter shocks (identical draw
                      sequence across variants — fresh rng(SEED) per row — so
                      rows are compared on the same reference distribution). For
                      slag rows the permuted object IS the lagged vector.
  circular shift    : the nq-1 non-trivial rolls, which PRESERVE the serial
                      structure of S. This is the arbiter when the two diverge.

Also run: the panel's own raw (unstandardized) shock — at st as the
headline-continuity anchor, at slag as the PRIMARY-continuity anchor (it is the
primary regressor itself). p-values of the raw baselines must match the menu's
standardized baseline_repro column within their timing arm, because a positive
affine rescaling of S changes b3 by a constant factor and leaves every
permutation comparison invariant.

DRIFT ANCHORS (read from the LIVING canonical artifact at runtime, NEVER
hardcoded — REBUILD v3, 2026-08-08; slag cell added 2026-08-10). The
b3_stata_anchor column only compares THIS run's Stata leg to THIS run's Python
leg, which cannot detect a panel-vintage swap — both legs would move together,
and stale panel vintages sit in the same directory as c6_panel.dta. So the two
baseline_panel_shock_raw rows are additionally hard-gated against
output/headline_3pairwise_canonical.csv (LOCKED 2026-08-08 layout
spec,denom,timing,fe,b3,se,p,N):
  st   row -> spec=="diag"    & denom=="global" & timing=="st"   & fe=="fq_gq_ig"
  slag row -> spec=="primary" & denom=="global" & timing=="slag" & fe=="fq_gq_ig"
       (the PRIMARY cell: the slag baseline IS the primary regression, so the
       diag/st cell would be the WRONG anchor for it).
Python-vs-canonical gate at 1e-3 (singleton-drop-sized only); Stata-leg gate at
1e-5 (identical regression). Missing file / stale layout => abort (fail-closed).

ANCHOR HYGIENE. shockmenu_results.csv is ingested as b3_stata_anchor, keyed by
(variant, timing) since the .do's 2026-08-10 timing column. A CSV predating the
timing column is treated as st-only (printed, not silent), so slag rows simply
lack a Stata anchor rather than silently matching the wrong arm. The .do
suffixes smoke output filenames with `_SMOKE' and stamps a `smoke' column, so a
200-firm syntax pass cannot become the anchor of record unnoticed; this script
additionally prints the anchor file's mtime and row count, and flags every row
whose |b3_python/b3_stata - 1| exceeds ANCHOR_RELTOL.

================================================================================
PRE-REGISTERED READING  (header note, NOT a conclusion)
================================================================================
Written BEFORE any menu RI was run; the AUTHORITATIVE copy is the identically
worded block in build_shock_menu.py. ~12 variant labels x 2 p-flavours means the
output is otherwise fully open to post-hoc narration.

(a) ALL variants null, including the pre-registered preferred variant and the
    bidirectional E_C_i_bidir column => the P0 deep null (b3 = -5.28e-7, CRVE
    p = 0.763, RI p = 0.801) is NOT an artifact of shock construction, and
    P1a/P1b/P1c are DISCLOSED-AND-CLOSED defects rather than open threats.
(b) The PREFERRED variant REJECTS while the baseline does not => a
    construction-sensitivity result on the pre-registered column ONLY, with the
    caveat that it correlates just ~0.25 with the baseline shock, so it is close
    to an INDEPENDENT test rather than a perturbation of the headline.
(c) The BASELINE rejects while no no-look-ahead variant does => the headline is a
    look-ahead / serial-correlation artifact (P1a+P1c); the NLA column GOVERNS.
(d) An E direction column rejects while its USA|China twin does not => a
    direction-axis result, reported in parallel and NEVER promoted to headline,
    because E was excluded from the selection rule by design.
(e) Any SINGLE one of ~12 columns crossing p < .05 with the rest null is roughly
    ONE EXPECTED FALSE POSITIVE at this family size. Only the pre-registered
    preferred variant and the continuity baseline carry weight, and p_circ
    governs over p_free wherever they diverge.
================================================================================
TIMING-EXTENSION NOTE on the reading (2026-08-10, added AFTER the verbatim block
above): the reading was pre-registered for the S_t family. The slag arm re-uses
(a)-(e) verbatim at the primary timing; a slag-only rejection with the st twin
null is a TIMING-sensitivity observation, not a construction result.

VINTAGE NOTE on the reading (2026-08-10, r1 advisory A1 — appended, block above
left verbatim): item (a)'s cited cells (b3=-5.28e-7, CRVE p=0.763, RI p=0.801)
are the 2026-08-04 P0 348,156-row vintage's numbers. The LIVING canonical cells
this file hard-gates via CANON_ANCHORS are the v3.1 (909,724-row) values:
diag/global/st b3=+7.578581e-7 p=0.420 and primary/global/slag b3=-1.394461e-7
p=0.893 (RI circ arbiter 0.878). Read "the P0 deep null" in (a) as "the
canonical deep null at the current vintage" — the null carries over; the cited
magnitudes do not.
================================================================================

OUTPUT: output/ri_shockmenu.csv  (variant, timing, b3, p_free, p_circ,
+ provenance; timing in {st, slag}).

HONESTY: every number verbatim from this run; failures reported as failures; no
prior external measurement is trusted without re-derivation.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyreadstat

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")

# --- Smoke flag default OFF (fast syntax pass: far fewer permutations) ---
SMOKE = False
N_PERM = 200 if SMOKE else 5000
DEMEAN_ITERS = 30
SEED = 20260702

PANEL_DTA = OUT / "c6_panel.dta"
PANEL_PQ = OUT / "shockmenu_c6_panel.parquet"     # regenerated every run (freshness)
MENU_PQ = OUT / "shock_menu_quarterly.parquet"
ANCHOR_CSV = OUT / "shockmenu_results.csv"        # written by run_shock_menu.do
RESULT_CSV = OUT / "ri_shockmenu.csv"

BASELINE_LBL = "baseline_panel_shock_raw"

# -----------------------------------------------------------------------------
# DRIFT ANCHORS -- read at runtime from the LIVING canonical artifact, never
# hardcoded (REBUILD v3, 2026-08-08; slag cell 2026-08-10). Two cells, one per
# timing arm:
#   "st"   -> diag/global/st/fq_gq_ig      (the S_t continuity cell)
#   "slag" -> primary/global/slag/fq_gq_ig (the PRIMARY cell — the slag
#             baseline IS the primary regression us x cn_lag x S_{t-1}, so it
#             anchors the primary cell, explicitly NOT the diag/st cell)
# Each is a cross-artifact gate, not an aspiration: it catches a panel-vintage
# swap that the per-run b3_stata_anchor comparison structurally cannot (that
# column compares this run's Stata leg to this run's Python leg, and a vintage
# swap moves both together).
# -----------------------------------------------------------------------------
CANONICAL_CSV = OUT / "headline_3pairwise_canonical.csv"

ANCHOR_CELLS = {
    # timing -> (spec, denom, timing, fe) in the LOCKED 2026-08-08 layout
    "st":   ("diag",    "global", "st",   "fq_gq_ig"),
    "slag": ("primary", "global", "slag", "fq_gq_ig"),
}


def load_canonical_anchor():
    """{timing: (b3, N)} for the two drift-gate cells (see ANCHOR_CELLS).
    Fail-closed: missing file, stale layout, or a missing cell aborts."""
    if not CANONICAL_CSV.exists():
        raise SystemExit(
            f"{CANONICAL_CSV.name} not found — run run_headline_3pairwise.do "
            "(or run_attribution_em.do) first; refusing to run the RI menu "
            "without vintage protection.")
    can = pd.read_csv(CANONICAL_CSV)
    need = {"spec", "denom", "timing", "fe", "b3", "N"}
    if not need.issubset(can.columns):
        raise SystemExit(
            f"{CANONICAL_CSV.name} lacks {sorted(need - set(can.columns))} — "
            "stale pre-2026-08-08 layout; re-run run_headline_3pairwise.do.")
    out = {}
    for timing, (spec, denom, tim, fe) in ANCHOR_CELLS.items():
        r = can[(can["spec"] == spec) & (can["denom"] == denom)
                & (can["timing"] == tim) & (can["fe"] == fe)]
        if len(r) != 1:
            raise SystemExit(
                f"{CANONICAL_CSV.name}: no unique {spec}/{denom}/{tim}/{fe} row "
                f"(found {len(r)}) — stale vintage; re-run run_headline_3pairwise.do.")
        out[timing] = (float(r["b3"].iloc[0]), int(r["N"].iloc[0]))
    return out


CANON_ANCHORS = load_canonical_anchor()

# Python-vs-canonical tolerance: reghdfe drops singleton firm-quarters while
# this collapse keeps them (coefficient moves at ~1e-4 relative order at most,
# measured at the st cell), so the cross-artifact gate on the PYTHON leg uses
# the singleton-sized tolerance. NOTE for the slag gate: the primary cell's
# |b3| (~1.4e-7) is ~4x smaller than the st cell's (~5.3e-7), so the SAME
# absolute singleton gap is ~4x the RELATIVE error; the measured ~1e-4-order
# st gap still clears 1e-3 with margin at slag, but a slag-only near-miss with
# an absolute gap of order 1e-10 is singleton geometry, not vintage drift —
# investigate before ever widening, never widen silently.
B3_ANCHOR_RELTOL = 1e-3
STATA_LEG_RELTOL = 1e-5      # Stata leg vs canonical = identical regression
# b3 tolerance for the WITHIN-RUN Stata-vs-Python cross-check. Generous, because
# reghdfe drops singleton firm-quarters while this collapse keeps them, so a small
# gap is expected and legitimate; anything larger is a real disagreement. The
# same slag-relative-error caveat as B3_ANCHOR_RELTOL applies.
ANCHOR_RELTOL = 1e-3

# Column-name exclusions, mirrored EXACTLY in run_shock_menu.do's resolver so the
# Stata and Python legs always estimate the same variant set.
EXCL_EXACT = {
    "qtr", "_qe_day", "quarter_end", "quarter", "qdate", "q", "date", "year",
    "yr", "n", "obs", "nobs", "index", "_merge", "_mm",
}
EXCL_PFX = ("gpr", "raw_", "sd_", "acf", "lb_", "ljung", "nq_", "n_", "p_")
EXCL_SFX = ("_raw", "_sd", "_unstd", "_pre", "_prestd", "_nostd", "_level")


# -----------------------------------------------------------------------------
# menu variant resolver
# -----------------------------------------------------------------------------
def resolve_variants(menu: pd.DataFrame):
    """Enumerate variant columns GENERICALLY (the contract is the interface; w1's
    exact names are not hard-coded). Preference: S_* -> s_* -> every numeric
    column; then the documented exclusions. Returns (names, resolver_label)."""
    numeric = [c for c in menu.columns if pd.api.types.is_numeric_dtype(menu[c])]
    pref = [c for c in numeric if c.startswith("S_")]
    label = "S_* prefix"
    if not pref:
        pref = [c for c in numeric if c.startswith("s_")]
        label = "s_* prefix"
    if not pref:
        pref = numeric
        label = "all numeric columns minus exclusions"
    keep = []
    for c in pref:
        lc = c.lower()
        if lc in EXCL_EXACT:
            continue
        if lc.startswith(EXCL_PFX):
            continue
        if lc.endswith(EXCL_SFX):
            continue
        keep.append(c)
    return keep, label


# -----------------------------------------------------------------------------
# two-way demeaner (identical to run_ri_3pairwise.py / run_ri_extensive.py)
# -----------------------------------------------------------------------------
def make_demeaner(firm_c, q_c, iters):
    """Two-way (firm, quarter) alternating-projection demeaner via bincount."""
    nf = int(firm_c.max()) + 1
    nq = int(q_c.max()) + 1
    fcount = np.bincount(firm_c, minlength=nf).astype(float)
    qcount = np.bincount(q_c, minlength=nq).astype(float)

    def demean(x):
        x = x.copy()
        for _ in range(iters):
            fm = np.bincount(firm_c, weights=x, minlength=nf) / fcount
            x = x - fm[firm_c]
            qm = np.bincount(q_c, weights=x, minlength=nq) / qcount
            x = x - qm[q_c]
        return x

    return demean


class Collapsed:
    """Pre-demeaned pieces of one difference-panel sample. Everything that does
    not depend on S is built here, once, and reused by every (variant, timing)
    row that shares this row set."""

    def __init__(self, sub, ycol, iters=DEMEAN_ITERS):
        self.n = sub.shape[0]
        self.firm_c = pd.factorize(sub["firm_str"])[0]
        # chronological quarter code (sort=True) so np.roll shifts along calendar
        # time, not along an arbitrary hash order
        q_c, uq = pd.factorize(sub["rdate"], sort=True)
        self.q_c = np.asarray(q_c)
        self.uq = uq
        self.nq = len(uq)
        self.n_firm = int(self.firm_c.max()) + 1 if self.n else 0
        # firm + quarter FE absorb (n_firm + nq - 1) params; guard non-positive DoF
        self.degenerate = (self.n == 0) or ((self.n - self.n_firm - self.nq + 1) <= 0)
        if self.degenerate:
            return
        self.demean = make_demeaner(self.firm_c, self.q_c, iters)
        self.y = self.demean(sub[ycol].to_numpy(float))
        self.cn = self.demean(sub["cn"].to_numpy(float))
        self.cn_raw = sub["cn"].to_numpy(float)
        self.cn_dot = float(self.cn @ self.cn)
        self.y_cn = float(self.y @ self.cn)
        if self.cn_dot <= 0:
            self.degenerate = True
            return
        # --- linear basis: B[:, t] = demean(cn . 1[q == t]) ---
        B = np.empty((self.n, self.nq), dtype=float)
        for t in range(self.nq):
            e = np.zeros(self.n)
            m = self.q_c == t
            e[m] = self.cn_raw[m]
            B[:, t] = self.demean(e)
        self.u = B.T @ self.cn
        self.v = B.T @ self.y
        self.G = B.T @ B
        del B

    # -- fast route (used for every observed / permuted / shifted S) -----------
    def beta3(self, S):
        uS = float(self.u @ S)
        b = uS / self.cn_dot
        num = float(self.v @ S) - b * self.y_cn
        den = float(S @ (self.G @ S)) - 2.0 * b * uS + b * b * self.cn_dot
        return num / den if den > 0 else np.nan

    # -- slow row-level route, kept ONLY as the self-check --------------------
    def beta3_rowlevel(self, S):
        x = self.cn_raw * S[self.q_c]
        xt = self.demean(x)
        b = (self.cn @ xt) / self.cn_dot
        xr = xt - b * self.cn
        den = float(xr @ xr)
        return float(self.y @ xr) / den if den > 0 else np.nan


def ri_one(col: Collapsed, S, n_perm=N_PERM, seed=SEED):
    """Free-permutation AND circular-shift RI for the sharp null b3 = 0.
    S is the spec's own vector on the covered quarters — for slag rows this IS
    the lagged vector, so the permuted/rolled object is the lagged vector.
    Returns (b_obs, p_free, p_circ, n_shift)."""
    b_obs = col.beta3(S)
    if not np.isfinite(b_obs):
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)     # fresh per row -> identical draws
    cnt = 0
    for _ in range(n_perm):
        bp = col.beta3(rng.permutation(S))
        if np.isfinite(bp) and abs(bp) >= abs(b_obs) - 1e-300:
            cnt += 1
    p_free = (cnt + 1) / (n_perm + 1)
    shift_b = np.array([col.beta3(np.roll(S, k)) for k in range(1, col.nq)])
    if shift_b.size:
        p_circ = (np.nansum(np.abs(shift_b) >= abs(b_obs) - 1e-300) + 1) / (shift_b.size + 1)
    else:
        p_circ = np.nan
    return b_obs, p_free, p_circ, int(shift_b.size)


def collapse(panel_parquet):
    """US-minus-NONUS firm-quarter difference for dw, plus the fixed cn and the
    panel's own raw shock. Sorted canonically (firm_str, rdate) so factorize codes
    and every float sum are reproducible regardless of duckdb's row order."""
    con = duckdb.connect()
    cols = {r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{panel_parquet}') LIMIT 0").fetchall()}
    slag_sel = ("any_value(s_lag) AS s_lag_panel," if "s_lag" in cols
                else "CAST(NULL AS DOUBLE) AS s_lag_panel,")
    d = con.execute(f"""
    SELECT firm_str,
           CAST(rdate AS TIMESTAMP) AS rdate,
           any_value(cn_lag) AS cn,
           any_value(shock)  AS s_base,
           {slag_sel}
           MAX(CASE WHEN us=1 THEN dw END) - MAX(CASE WHEN us=0 THEN dw END) AS d_dw
    FROM read_parquet('{panel_parquet}')
    GROUP BY firm_str, rdate
    ORDER BY firm_str, rdate
    """).df()
    con.close()
    return d.sort_values(["firm_str", "rdate"], kind="mergesort").reset_index(drop=True)


def load_anchor():
    """{(variant, timing): stata_b3} from the .do's 3-pairwise dw rows, if present.

    PROVENANCE IS PRINTED, not assumed: file mtime, row count, and the .do's
    `smoke'/`n_firms' markers. A SMOKE run of run_shock_menu.do writes to
    shockmenu_results_SMOKE.csv (different path) AND stamps smoke=1 on every row,
    so both a wrong-file and a wrong-content anchor are visible here rather than
    silently becoming the anchor of record. A CSV that predates the 2026-08-10
    `timing' column is treated as st-only (printed): slag rows then have NO
    Stata anchor, which is disclosed as NaN rather than mismatched to the wrong
    arm.
    """
    if not ANCHOR_CSV.exists():
        return {}
    try:
        # na_values="." : Stata's missing marker, in case the .do ever emits one
        a = pd.read_csv(ANCHOR_CSV, na_values=["."], keep_default_na=True)
    except Exception as exc:                                    # noqa: BLE001
        print(f"anchor: could not read {ANCHOR_CSV.name} ({exc}) — continuing without it")
        return {}

    mtime = pd.Timestamp(ANCHOR_CSV.stat().st_mtime, unit="s", tz="UTC").tz_convert(None)
    print(f"anchor file : {ANCHOR_CSV.name}  mtime={mtime:%Y-%m-%d %H:%M:%S} (UTC)  "
          f"rows={len(a)}")

    if "smoke" in a.columns:
        sm = pd.to_numeric(a["smoke"], errors="coerce").fillna(0)
        nf = (a["n_firms"].iloc[0] if "n_firms" in a.columns and len(a) else "?")
        if (sm > 0).any():
            print("  *** ANCHOR IS A SMOKE ARTIFACT *** "
                  f"{int((sm > 0).sum())}/{len(a)} rows carry smoke=1 (n_firms={nf}). "
                  "Its b3 values are a 200-firm SYNTAX PASS and are NOT valid "
                  "inference. Re-run run_shock_menu.do with SMOKE=0.")
        else:
            print(f"  smoke=0 on all rows (n_firms={nf}) — full-panel anchor.")
    else:
        print("  NOTE: anchor CSV has no `smoke' column — it predates the SMOKE "
              "marker fix, so a syntax-pass CSV would be indistinguishable here.")

    need = {"variant", "dv", "fe", "b3_triple"}
    if not need.issubset(a.columns):
        print(f"anchor: {ANCHOR_CSV.name} lacks {sorted(need - set(a.columns))} — skipped")
        return {}
    if "timing" not in a.columns:
        print("  NOTE: anchor CSV has no `timing' column — it predates the slag arm "
              "(2026-08-10). All rows treated as timing=st; slag rows will carry NO "
              "Stata anchor until run_shock_menu.do is re-run.")
        a = a.assign(timing="st")
    a = a[(a["dv"] == "dw") & (a["fe"] == "3pw_fq_gq_ig")]
    out = {}
    for _, r in a.iterrows():
        try:
            out[(str(r["variant"]), str(r["timing"]))] = float(r["b3_triple"])
        except (TypeError, ValueError):
            pass
    return out


def main():
    # ROTATION DISCIPLINE (r1 must-fix M2, 2026-08-10): REFUSE to clobber the
    # canonical RI artifact in place — rename it to *_r2pre first. Mirrors the
    # guard in run_shock_menu.do on its two anchors.
    if RESULT_CSV.exists():
        raise SystemExit(
            f"target {RESULT_CSV} already exists — rename it to "
            "ri_shockmenu_r2pre.csv (rotation rule) before re-running; "
            "refusing to overwrite the canonical RI artifact in place.")

    print("=" * 92)
    print(f"run_ri_shockmenu.py  SMOKE={SMOKE}  N_PERM={N_PERM}  iters={DEMEAN_ITERS}  SEED={SEED}")
    print("timing arms: st (pre-registered S_t diagnostic, UNCHANGED) + slag "
          "(S_{t-1}, the menu re-adjudicated AT THE PRIMARY TIMING)")
    print("=" * 92)

    if not MENU_PQ.exists():
        raise SystemExit(
            f"MENU NOT FOUND: {MENU_PQ}\n"
            f"build_shock_menu.py must write shock_menu_quarterly.parquet first."
        )

    # R3-F6 freshness: regenerate the parquet from the CURRENT .dta so this can
    # never silently run on a stale panel left over from an earlier build.
    df, _ = pyreadstat.read_dta(PANEL_DTA.as_posix())
    df.to_parquet(PANEL_PQ, index=False)
    d = collapse(PANEL_PQ.as_posix())
    d = d.dropna(subset=["d_dw", "cn"]).reset_index(drop=True)
    d["qper"] = pd.to_datetime(d["rdate"]).dt.to_period("Q")
    panel_q = pd.PeriodIndex(sorted(d["qper"].unique()))
    print(f"panel collapse: {len(d):,} firm-quarter differences over {len(panel_q)} quarters "
          f"({panel_q[0]} -> {panel_q[-1]})")

    # CONTIGUITY of the panel quarter grid — required before ANY positional lag
    # can claim to be the true calendar lag (Part-B map discipline).
    if not (np.diff(panel_q.asi8) == 1).all():
        raise SystemExit(
            "panel quarter grid is NOT contiguous — a positional lag would not be "
            "the true calendar lag; refusing to build the slag arm.")

    # ---- menu ----
    menu = pd.read_parquet(MENU_PQ)
    menu["qper"] = pd.to_datetime(menu["quarter_end"]).dt.to_period("Q")
    if menu["qper"].duplicated().any():
        raise SystemExit("menu quarter key is not unique — aborting.")
    menu = menu.sort_values("qper").reset_index(drop=True)
    # CONTIGUITY of the MENU grid: lags are built on the menu grid FIRST and
    # restricted to panel quarters SECOND (mirroring run_shock_menu.do, where
    # lagv is built before the panel merge — a menu that starts before the panel
    # legitimately gives the first panel quarter a non-missing lag).
    if not (np.diff(pd.PeriodIndex(menu["qper"]).asi8) == 1).all():
        raise SystemExit(
            "menu quarter grid is NOT contiguous — a positional shift would not "
            "be the true calendar lag; refusing to build the slag arm.")
    variants, resolver = resolve_variants(menu)
    if not variants:
        raise SystemExit(f"no variant columns resolved from {MENU_PQ.name} — check w1's naming.")
    print(f"menu: {MENU_PQ.name}  ({len(menu)} quarter rows)  resolver = {resolver}")
    for i, v in enumerate(variants, 1):
        s = menu[v]
        print(f"  [{i}] {v:<34s} n={s.notna().sum():>3d}  mean={s.mean():+.3e}  sd={s.std(ddof=0):.6f}")

    missing_q = sorted(set(panel_q) - set(menu["qper"]))
    if missing_q:
        raise SystemExit(
            f"MENU DOES NOT COVER THE PANEL: {len(missing_q)} panel quarter(s) absent "
            f"from the menu, e.g. {missing_q[:5]} — aborting rather than estimating on a "
            f"silently reduced sample."
        )
    mq = menu.set_index("qper")
    # TRUE CALENDAR LAG of every variant on the contiguous menu grid: menu is
    # sorted by quarter and contiguity is asserted above, so a positional
    # shift(1) IS the calendar lag (the Part-B map derivation, vectorized).
    mql = mq[variants].shift(1)

    # ---- shock vectors, aligned to the chronological quarter code ----
    # baseline (raw, unstandardized) comes off the panel itself; every menu
    # variant is looked up by quarter period. SLAG BASELINE (first-run fix
    # 2026-08-10): the canonical primary cell was estimated on the panel's OWN
    # s_lag column, whose Q1 value comes from the pre-panel contiguous GPR
    # series — so the slag baseline MUST use that column (a positional lag on
    # the panel grid is missing at Q1 and shrinks the sample; measured drift
    # vs the primary cell: relerr 3.5e-2, gate abort). The positional lag is
    # still derived and CROSS-CHECKED against the panel column on every
    # derivable quarter (Part-B map discipline), then discarded.
    base_by_q = d.groupby("qper")["s_base"].agg(["nunique", "first"])
    if (base_by_q["nunique"] > 1).any():
        raise SystemExit("panel shock varies within a quarter — expected one common S_t.")

    base_vec = base_by_q["first"].reindex(panel_q).to_numpy(float)
    base_lag = np.empty_like(base_vec)
    base_lag[0] = np.nan
    base_lag[1:] = base_vec[:-1]

    if d["s_lag_panel"].notna().any():
        slag_by_q = d.groupby("qper")["s_lag_panel"].agg(["nunique", "first"])
        if (slag_by_q["nunique"] > 1).any():
            raise SystemExit("panel s_lag varies within a quarter — expected one common S_{t-1}.")
        slag_vec = slag_by_q["first"].reindex(panel_q).to_numpy(float)
        if not np.isfinite(slag_vec).all():
            raise SystemExit(
                "panel s_lag column is INCOMPLETE on the panel grid — the canonical "
                "primary cell was estimated on a complete column; refusing a "
                "silently smaller slag baseline sample.")
        m = np.isfinite(base_lag)
        rel = np.abs(slag_vec[m] - base_lag[m]) / np.maximum(np.abs(base_lag[m]), 1e-12)
        if rel.max() >= 1e-6:
            raise SystemExit(
                f"panel s_lag diverges from the positional Part-B lag on a derivable "
                f"quarter (max relerr {rel.max():.3e} >= 1e-6) — wrong grid alignment; "
                f"refusing to build the slag arm.")
        base_lag = slag_vec
        print("slag baseline vector: panel s_lag column (complete; Q1 from the pre-panel "
              f"GPR series; matches the positional lag on all {int(m.sum())} derivable "
              f"quarters, max relerr {rel.max():.1e})")
    else:
        print("slag baseline vector: positional Part-B lag (panel carries no s_lag "
              "column; Q1 missing by construction)")

    specs = [(BASELINE_LBL, "st", base_vec)]
    for v in variants:
        specs.append((v, "st", mq[v].reindex(panel_q).to_numpy(float)))
    specs.append((BASELINE_LBL, "slag", base_lag))
    for v in variants:
        specs.append((v, "slag", mql[v].reindex(panel_q).to_numpy(float)))

    # ---- group rows by their missing-quarter pattern; one Collapsed each ----
    # (with a complete menu on the panel grid there are exactly TWO groups: the
    # full row set for the st arm and the first-quarter-dropped set for the
    # slag arm — the lag is missing at the first covered quarter by
    # construction; a menu extending before the panel collapses the two)
    groups = {}
    for lbl, tim, S in specs:
        key = tuple(np.flatnonzero(~np.isfinite(np.asarray(S, float))).tolist())
        groups.setdefault(key, []).append((lbl, tim, S))
    if len(groups) > 2:
        print(f"\nNOTE: {len(groups)} distinct missing-quarter patterns across rows — "
              f"each gets its own collapse (samples are NOT identical; disclosed, not hidden).")

    anchor = load_anchor()
    print(f"\nanchor: {len(anchor)} Stata 3-pairwise b3 value(s) keyed (variant, timing) "
          f"from {ANCHOR_CSV.name}"
          if anchor else f"\nanchor: {ANCHOR_CSV.name} not found — RI b3 reported alone")

    rows = []
    mismatches = []
    print()
    print(f"{'variant':<34s} {'tim':>4s} {'b3':>13s} {'p_free':>8s} {'p_circ':>8s} "
          f"{'b3_stata':>13s} {'relerr':>9s} {'n_fq':>9s} {'nq':>4s} {'flag':>7s}")
    for key, members in groups.items():
        drop_q = set(int(k) for k in key)
        if drop_q:
            keep_per = [panel_q[i] for i in range(len(panel_q)) if i not in drop_q]
            sub = d[d["qper"].isin(keep_per)].reset_index(drop=True)
            idx_keep = [i for i in range(len(panel_q)) if i not in drop_q]
        else:
            sub = d
            idx_keep = list(range(len(panel_q)))
        col = Collapsed(sub, "d_dw")
        if col.degenerate:
            for lbl, tim, _S in members:
                print(f"{lbl:<34s} {tim:>4s} {'':>13s} {'':>8s} {'':>8s} {'':>13s} {'':>9s} "
                      f"{col.n:>9,d} {col.nq:>4d} {'DEGEN':>7s}")
                rows.append({"variant": lbl, "timing": tim, "b3": np.nan,
                             "p_free": np.nan, "p_circ": np.nan,
                             "fe": "it+gt+ig (3-pairwise)", "dv": "dw",
                             "b3_stata_anchor": anchor.get((lbl, tim), np.nan), "relerr": np.nan,
                             "n_firmquarters": col.n, "n_quarters": col.nq, "n_circ_shifts": 0,
                             "n_perm": N_PERM, "seed": SEED, "arbiter_p": np.nan,
                             "anchor_mismatch": False,
                             "ri_degenerate": True, "n_quarters_dropped": len(drop_q)})
            continue

        # ---- SELF-CHECK: the fast basis route must reproduce the row-level b3 ----
        S0 = np.asarray(members[0][2], float)[idx_keep]
        b_fast = col.beta3(S0)
        b_slow = col.beta3_rowlevel(S0)
        rel = abs(b_fast / b_slow - 1.0) if np.isfinite(b_slow) and b_slow != 0 else np.nan
        if not (np.isfinite(rel) and rel < 1e-8):
            raise SystemExit(
                f"SELF-CHECK FAILED on the linear-basis speedup: fast={b_fast!r} "
                f"slow={b_slow!r} relerr={rel!r}. Refusing to report accelerated numbers."
            )
        print(f"[self-check] basis vs row-level b3: {b_fast:.6e} vs {b_slow:.6e} "
              f"(relerr {rel:.2e}) OK")

        for lbl, tim, S_full in members:
            S = np.asarray(S_full, float)[idx_keep]
            b_obs, p_free, p_circ, n_sh = ri_one(col, S)
            b_st = anchor.get((lbl, tim), np.nan)
            relerr = (abs(b_obs / b_st - 1.0)
                      if np.isfinite(b_st) and np.isfinite(b_obs) and b_st != 0 else np.nan)
            bad = bool(np.isfinite(relerr) and relerr > ANCHOR_RELTOL)
            if bad:
                mismatches.append((lbl, tim, relerr))
            flag = "MISMATCH" if bad else ""
            print(f"{lbl:<34s} {tim:>4s} {b_obs:13.5e} {p_free:8.4f} {p_circ:8.4f} "
                  f"{b_st:13.5e} {relerr:9.2e} {col.n:>9,d} {col.nq:>4d} {flag:>8s}")
            if bad:
                print(f"    *** ANCHOR MISMATCH on {lbl} (timing={tim}): python "
                      f"b3={b_obs:.6e} vs stata b3={b_st:.6e}, relerr={relerr:.3e} > "
                      f"{ANCHOR_RELTOL:.0e}. Expected gap is singleton-drop-sized only. "
                      f"Check that {ANCHOR_CSV.name} is the CURRENT full-panel run "
                      f"(not a SMOKE artifact, not stale, and carrying the timing "
                      f"column).")
            rows.append({"variant": lbl, "timing": tim, "b3": b_obs,
                         "p_free": p_free, "p_circ": p_circ,
                         "fe": "it+gt+ig (3-pairwise)", "dv": "dw",
                         "b3_stata_anchor": b_st, "relerr": relerr,
                         "anchor_mismatch": bad,
                         "n_firmquarters": col.n, "n_quarters": col.nq, "n_circ_shifts": n_sh,
                         "n_perm": N_PERM, "seed": SEED, "arbiter_p": p_circ,
                         "ri_degenerate": False, "n_quarters_dropped": len(drop_q)})

            # ------- canonical drift gate (baseline rows only; living source) --
            # per-timing anchor: st -> diag/global/st, slag -> primary/global/slag
            # (the slag baseline IS the primary regression, so the primary cell —
            # not the diag/st cell — is its anchor; wired via CANON_ANCHORS).
            if lbl == BASELINE_LBL:
                b3_anc, n_anc = CANON_ANCHORS[tim]
                cell = "/".join(ANCHOR_CELLS[tim])
                print(f"\n[drift gate — timing={tim}] canonical {cell} cell vs this run "
                      "(b3 HARD-GATED on both legs; anchors read at runtime, "
                      "never hardcoded)")
                print(f"    b3 anchor       = {b3_anc:.12e}   "
                      f"(living source: {CANONICAL_CSV.name}, {cell}, N={n_anc:,d})")
                print(f"    b3 observed     = {b_obs:.12e}   "
                      f"(n_fq={col.n:,d}; reghdfe N differs by singleton drops)")
                fails = []
                if np.isfinite(b_st):
                    st_rel = abs(b_st / b3_anc - 1.0)
                    st_ok = st_rel <= STATA_LEG_RELTOL
                    print(f"    stata-leg b3    = {b_st:.6e} vs canonical "
                          f"{b3_anc:.6e}  relerr={st_rel:.2e}  "
                          f"{'OK' if st_ok else '<-- STATA LEG DRIFTED'}")
                    if not st_ok:
                        fails.append(f"stata-leg relerr={st_rel:.3e} > "
                                     f"{STATA_LEG_RELTOL:.0e} (identical regression "
                                     "-- shockmenu_results.csv is a stale/foreign "
                                     "vintage)")
                elif tim == "slag":
                    print("    stata-leg b3    = (absent — shockmenu_results.csv "
                          "predates the slag arm or lacks the timing column; "
                          "re-run run_shock_menu.do to restore the cross-check)")
                b_rel = (abs(b_obs / b3_anc - 1.0)
                         if np.isfinite(b_obs) and b3_anc != 0 else np.inf)
                if not (np.isfinite(b_rel) and b_rel <= B3_ANCHOR_RELTOL):
                    fails.append(f"python-leg relerr={b_rel:.3e} > "
                                 f"{B3_ANCHOR_RELTOL:.0e} (singleton-drop-sized "
                                 "gap only is legitimate)")
                if fails:
                    raise SystemExit(
                        f"DRIFT GATE FAILED on {BASELINE_LBL} (timing={tim}, cell "
                        f"{cell}): " + "; ".join(fails) +
                        ".\nThis row IS that canonical cell, so a mismatch means a "
                        "STALE PANEL OR WRONG VINTAGE (stale panel vintages are "
                        "siblings of c6_panel.dta). Refusing to report a menu built "
                        "on the wrong data."
                    )
                print(f"    python-leg rel  = {b_rel:.3e}  -> PASS")
                if tim == "st":
                    print(f"    p_free          = {p_free:.4f}   p_circ = {p_circ:.4f}   "
                          "(no living RI-p anchor for the S_t cell; p_circ is the "
                          "serial-robust arbiter)")
                else:
                    print(f"    p_free          = {p_free:.4f}   p_circ = {p_circ:.4f}   "
                          "(the primary cell's own RI-p lives in run_ri_3pairwise.py's "
                          "output and is NOT re-read here; this arm is the menu's "
                          "construction-robustness re-derivation — p_circ is the "
                          "serial-robust arbiter)")
                if SMOKE:
                    print("    (SMOKE run: N_PERM tiny -> p not comparable across runs; "
                          "the b3 gates are what this proves.)")
                print()

    res = pd.DataFrame(rows)
    # required columns first, provenance after
    lead = ["variant", "timing", "b3", "p_free", "p_circ"]
    res = res[lead + [c for c in res.columns if c not in lead]]
    res.to_csv(RESULT_CSV, index=False)
    print(f"\nwrote {RESULT_CSV.name}  ({len(res)} rows; timing in {{st, slag}})")
    print("  timing arms: st = the pre-registered contemporaneous construction")
    print("  diagnostic (UNCHANGED); slag = the SAME menu re-adjudicated at the")
    print("  PRIMARY timing S_{t-1} — lagged vectors permuted/rolled, baseline")
    print("  gated on the canonical primary/global/slag cell.")
    print("  p_circ (circular shift) is the SERIAL-ROBUST ARBITER: S_t is serially")
    print("  correlated under the baseline construction (P1a), which makes the free")
    print("  permutation p anti-conservative. Where the two diverge, read p_circ.")
    print("  b3 magnitudes are comparable ACROSS MENU VARIANTS within a timing arm")
    print("  because every menu column is standardized on the panel quarters;")
    print(f"  {BASELINE_LBL} is on the RAW shock scale and is a continuity")
    print("  anchor only — compare its p, not its b3.")
    print("  Variant SELECTION is governed solely by the pre-registered whiteness rule")
    print("  on shock_menu_diagnostics.csv. Nothing here may be used to choose it.")

    if mismatches:
        print("\n  *** ANCHOR MISMATCHES (anchor_mismatch=True in the CSV) ***")
        for lbl, tim, r in mismatches:
            print(f"    {lbl:<34s} [{tim}] relerr={r:.3e}")
        print(f"    Tolerance is {ANCHOR_RELTOL:.0e}; only a singleton-drop-sized gap is")
        print(f"    expected. Verify {ANCHOR_CSV.name} is the CURRENT full-panel .do run.")
    else:
        print(f"\n  anchor cross-check: all anchored rows within {ANCHOR_RELTOL:.0e} of the Stata leg.")

    print("\n" + "=" * 92)
    print("PRE-REGISTERED READING (fixed in this file's header BEFORE any RI ran)")
    print("=" * 92)
    print("  (a) ALL variants null, incl. the preferred variant and the bidirectional")
    print("      E_C_i_bidir column => the P0 deep null (b3=-5.28e-7, CRVE p=0.763,")
    print("      RI p=0.801) is NOT an artifact of shock construction; P1a/P1b/P1c are")
    print("      DISCLOSED-AND-CLOSED defects, not open threats.")
    print("  (b) PREFERRED rejects while baseline does not => construction-sensitivity")
    print("      result on the pre-registered column ONLY; it correlates just ~0.25 with")
    print("      the baseline shock, so it is close to an INDEPENDENT test, not a")
    print("      perturbation of the headline.")
    print("  (c) BASELINE rejects while no no-look-ahead variant does => the headline is")
    print("      a look-ahead / serial-correlation artifact; the NLA column GOVERNS.")
    print("  (d) An E direction column rejects while its USA|China twin does not => a")
    print("      direction-axis result, reported in parallel, NEVER promoted to headline")
    print("      (E was excluded from the selection rule by design).")
    print("  (e) Any SINGLE one of ~12 columns crossing p<.05 with the rest null is")
    print("      roughly ONE EXPECTED FALSE POSITIVE at this family size. Only the")
    print("      pre-registered preferred variant and the continuity baseline carry")
    print("      weight, and p_circ governs over p_free wherever they diverge.")
    print("  TIMING-EXTENSION NOTE (2026-08-10): the reading above was pre-registered")
    print("  for the S_t family; the slag arm re-uses it verbatim at the primary")
    print("  timing. (e)'s arithmetic applies WITHIN each timing arm; a slag-only")
    print("  rejection with the st twin null is a TIMING-sensitivity observation,")
    print("  not a construction result.")
    print("  VINTAGE NOTE (2026-08-10, appended — reading above frozen verbatim):")
    print("  item (a)'s cited cells (b3=-5.28e-7, CRVE p=0.763, RI p=0.801) are the")
    print("  2026-08-04 P0 348,156-row vintage. The living canonical cells hard-gated")
    print("  via CANON_ANCHORS are the v3.1 values: diag/global/st b3=+7.578581e-7")
    print("  p=0.420 and primary/global/slag b3=-1.394461e-7 p=0.893 (RI circ 0.878).")
    print("  Read 'the P0 deep null' as 'the canonical deep null at the current")
    print("  vintage'.")
    print("=" * 92)


if __name__ == "__main__":
    main()
