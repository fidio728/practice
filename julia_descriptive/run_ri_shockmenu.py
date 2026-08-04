"""
run_ri_shockmenu.py — randomization inference for the SHOCK MENU (P1a/P1b/P1c),
the design-based ARBITER for run_shock_menu.do.

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

REUSE ACROSS VARIANTS. The collapse is IDENTICAL for every variant — Delta y and
cn are fixed, only the 82-vector S changes. Two consequences are exploited:
  (1) demean(Delta y) and demean(cn) are computed ONCE;
  (2) the demeaner (fixed firm/quarter codes, fixed iteration count) is a LINEAR
      operator, and cn*S = sum_t S_t * (cn . 1[q=t]), so
          demean(cn*S) = B @ S,  B[:, t] = demean(cn . 1[q=t]).
      B is built once (82 demeans). Every b3 evaluation then reduces to
          b   = (u . S) / cn_dot                      u = B' cn
          b3  = (v . S - b * y_cn) / (S' G S - 2 b (u . S) + b^2 cn_dot)
                                                      v = B' y,  G = B' B
      which is algebraically IDENTICAL to the row-level route, not an
      approximation. A SELF-CHECK re-derives the observed b3 the slow row-level
      way and aborts if the two disagree beyond 1e-8 relative — so the speedup
      can never silently change a number.

TESTS PER VARIANT (headline 3-pairwise only, DV = dw):
  free permutation  : N_PERM draws of the 82 quarter shocks (identical draw
                      sequence across variants — fresh rng(SEED) per variant — so
                      variants are compared on the same reference distribution).
  circular shift    : the nq-1 non-trivial rolls, which PRESERVE the serial
                      structure of S. This is the arbiter when the two diverge.

Also run: the panel's own raw (unstandardized) shock, as the headline-continuity
anchor. Its p-values must match the menu's standardized baseline_repro column,
because a positive affine rescaling of S changes b3 by a constant factor and
leaves every permutation comparison invariant.

DRIFT ANCHORS (module constants below, each tagged with its living source). The
b3_stata_anchor column only compares THIS run's Stata leg to THIS run's Python
leg, which cannot detect a panel-vintage swap — both legs would move together,
and c6_panel_preP0.dta sits in the same directory as c6_panel.dta. So the
baseline_panel_shock_raw row is additionally hard-gated against the canonical P0
artifacts (b3 and n_firmquarters), and its free-permutation p is printed with the
same "<-- DRIFT > 0.02" flag run_cum4_inference.py uses.

ANCHOR HYGIENE. shockmenu_results.csv is ingested as b3_stata_anchor. It now
carries a `smoke' column and the .do suffixes smoke output filenames with
`_SMOKE', so a 200-firm syntax pass cannot become the anchor of record unnoticed;
this script additionally prints the anchor file's mtime and row count, and flags
every row whose |b3_python/b3_stata - 1| exceeds ANCHOR_RELTOL.

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

OUTPUT: output/ri_shockmenu.csv  (variant, b3, p_free, p_circ, + provenance).

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
# DRIFT ANCHORS -- canonical P0 headline, each tagged with its LIVING SOURCE.
# The baseline_panel_shock_raw row of this script IS the canonical headline (same
# panel, same collapse, same raw shock), so these are cross-run gates, not
# aspirations. They catch a panel-vintage swap (c6_panel_preP0.dta is a sibling
# of c6_panel.dta) that the per-run b3_stata_anchor comparison structurally
# cannot: that column compares this run's Stata leg to this run's Python leg, and
# a vintage swap moves both together.
# -----------------------------------------------------------------------------
# source: output/audit_ri_3pairwise.csv, row "headline dw"
B3_ANCHOR = -5.279916082656558e-07
N_FQ_ANCHOR = 174078
RI_P_FREE_ANCHOR = 0.8014397
# source: output/headline_3pairwise_canonical.csv, row "3pairwise_fq_gq_ig"
B3_ANCHOR_STATA = -5.279916e-07

B3_ANCHOR_RELTOL = 1e-6      # b3 is order-invariant -> a 6-sig-fig gate is legitimate
P_DRIFT_TOL = 0.02           # same flag threshold as run_cum4_inference.py L504
# b3 tolerance for the WITHIN-RUN Stata-vs-Python cross-check. Generous, because
# reghdfe drops singleton firm-quarters while this collapse keeps them, so a small
# gap is expected and legitimate; anything larger is a real disagreement.
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
    not depend on S is built here, once, and reused by every variant that shares
    this row set."""

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
    Returns (b_obs, p_free, p_circ, n_shift)."""
    b_obs = col.beta3(S)
    if not np.isfinite(b_obs):
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)     # fresh per variant -> identical draws
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
    d = con.execute(f"""
    SELECT firm_str,
           CAST(rdate AS TIMESTAMP) AS rdate,
           any_value(cn_lag) AS cn,
           any_value(shock)  AS s_base,
           MAX(CASE WHEN us=1 THEN dw END) - MAX(CASE WHEN us=0 THEN dw END) AS d_dw
    FROM read_parquet('{panel_parquet}')
    GROUP BY firm_str, rdate
    ORDER BY firm_str, rdate
    """).df()
    con.close()
    return d.sort_values(["firm_str", "rdate"], kind="mergesort").reset_index(drop=True)


def load_anchor():
    """{variant: stata_b3} from the .do's 3-pairwise dw rows, if present.

    PROVENANCE IS PRINTED, not assumed: file mtime, row count, and the .do's
    `smoke'/`n_firms' markers. A SMOKE run of run_shock_menu.do writes to
    shockmenu_results_SMOKE.csv (different path) AND stamps smoke=1 on every row,
    so both a wrong-file and a wrong-content anchor are visible here rather than
    silently becoming the anchor of record.
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
    a = a[(a["dv"] == "dw") & (a["fe"] == "3pw_fq_gq_ig")]
    out = {}
    for _, r in a.iterrows():
        try:
            out[str(r["variant"])] = float(r["b3_triple"])
        except (TypeError, ValueError):
            pass
    return out


def main():
    print("=" * 92)
    print(f"run_ri_shockmenu.py  SMOKE={SMOKE}  N_PERM={N_PERM}  iters={DEMEAN_ITERS}  SEED={SEED}")
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

    # ---- menu ----
    menu = pd.read_parquet(MENU_PQ)
    menu["qper"] = pd.to_datetime(menu["quarter_end"]).dt.to_period("Q")
    if menu["qper"].duplicated().any():
        raise SystemExit("menu quarter key is not unique — aborting.")
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

    # ---- shock vectors, aligned to the chronological quarter code ----
    # baseline (raw, unstandardized) comes off the panel itself; every menu
    # variant is looked up by quarter period.
    base_by_q = d.groupby("qper")["s_base"].agg(["nunique", "first"])
    if (base_by_q["nunique"] > 1).any():
        raise SystemExit("panel shock varies within a quarter — expected one common S_t.")

    specs = [(BASELINE_LBL, base_by_q["first"].reindex(panel_q).to_numpy(float))]
    for v in variants:
        specs.append((v, mq[v].reindex(panel_q).to_numpy(float)))

    # ---- group variants by their missing-quarter pattern; one Collapsed each ----
    # (with a complete menu there is exactly ONE group = the full row set)
    groups = {}
    for lbl, S in specs:
        key = tuple(np.flatnonzero(~np.isfinite(S)).tolist())
        groups.setdefault(key, []).append((lbl, S))
    if len(groups) > 1:
        print(f"\nNOTE: {len(groups)} distinct missing-quarter patterns across variants — "
              f"each gets its own collapse (samples are NOT identical; disclosed, not hidden).")

    anchor = load_anchor()
    print(f"\nanchor: {len(anchor)} Stata 3-pairwise b3 value(s) from {ANCHOR_CSV.name}"
          if anchor else f"\nanchor: {ANCHOR_CSV.name} not found — RI b3 reported alone")

    rows = []
    mismatches = []
    print()
    print(f"{'variant':<34s} {'b3':>13s} {'p_free':>8s} {'p_circ':>8s} "
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
            for lbl, _S in members:
                print(f"{lbl:<34s} {'':>13s} {'':>8s} {'':>8s} {'':>13s} {'':>9s} "
                      f"{col.n:>9,d} {col.nq:>4d} {'DEGEN':>7s}")
                rows.append({"variant": lbl, "b3": np.nan, "p_free": np.nan, "p_circ": np.nan,
                             "fe": "it+gt+ig (3-pairwise)", "dv": "dw",
                             "b3_stata_anchor": anchor.get(lbl, np.nan), "relerr": np.nan,
                             "n_firmquarters": col.n, "n_quarters": col.nq, "n_circ_shifts": 0,
                             "n_perm": N_PERM, "seed": SEED, "arbiter_p": np.nan,
                             "anchor_mismatch": False,
                             "ri_degenerate": True, "n_quarters_dropped": len(drop_q)})
            continue

        # ---- SELF-CHECK: the fast basis route must reproduce the row-level b3 ----
        S0 = np.asarray(members[0][1], float)[idx_keep]
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

        for lbl, S_full in members:
            S = np.asarray(S_full, float)[idx_keep]
            b_obs, p_free, p_circ, n_sh = ri_one(col, S)
            b_st = anchor.get(lbl, np.nan)
            relerr = (abs(b_obs / b_st - 1.0)
                      if np.isfinite(b_st) and np.isfinite(b_obs) and b_st != 0 else np.nan)
            bad = bool(np.isfinite(relerr) and relerr > ANCHOR_RELTOL)
            if bad:
                mismatches.append((lbl, relerr))
            flag = "MISMATCH" if bad else ""
            print(f"{lbl:<34s} {b_obs:13.5e} {p_free:8.4f} {p_circ:8.4f} "
                  f"{b_st:13.5e} {relerr:9.2e} {col.n:>9,d} {col.nq:>4d} {flag:>8s}")
            if bad:
                print(f"    *** ANCHOR MISMATCH on {lbl}: python b3={b_obs:.6e} vs "
                      f"stata b3={b_st:.6e}, relerr={relerr:.3e} > {ANCHOR_RELTOL:.0e}. "
                      f"Expected gap is singleton-drop-sized only. Check that "
                      f"{ANCHOR_CSV.name} is the CURRENT full-panel run (not a SMOKE "
                      f"artifact and not stale).")
            rows.append({"variant": lbl, "b3": b_obs, "p_free": p_free, "p_circ": p_circ,
                         "fe": "it+gt+ig (3-pairwise)", "dv": "dw",
                         "b3_stata_anchor": b_st, "relerr": relerr,
                         "anchor_mismatch": bad,
                         "n_firmquarters": col.n, "n_quarters": col.nq, "n_circ_shifts": n_sh,
                         "n_perm": N_PERM, "seed": SEED, "arbiter_p": p_circ,
                         "ri_degenerate": False, "n_quarters_dropped": len(drop_q)})

            # ---------------- canonical P0 drift gate (baseline row only) -----
            if lbl == BASELINE_LBL:
                print("\n[drift gate] canonical P0 headline vs this run "
                      "(b3 + n_firmquarters HARD-GATED; free-perm p = MC cross-check)")
                print(f"    b3 anchor       = {B3_ANCHOR:.12e}   "
                      f"(source: audit_ri_3pairwise.csv, row 'headline dw')")
                print(f"    b3 observed     = {b_obs:.12e}")
                print(f"    n_fq anchor     = {N_FQ_ANCHOR:,d}   observed = {col.n:,d}")
                if np.isfinite(b_st):
                    st_rel = abs(b_st / B3_ANCHOR_STATA - 1.0)
                    st_flag = "OK" if st_rel <= 1e-5 else "<-- STATA LEG DRIFTED"
                    print(f"    stata-leg b3    = {b_st:.6e} vs canonical "
                          f"{B3_ANCHOR_STATA:.6e}  relerr={st_rel:.2e}  {st_flag}   "
                          f"(source: headline_3pairwise_canonical.csv)")
                b_rel = (abs(b_obs / B3_ANCHOR - 1.0)
                         if np.isfinite(b_obs) and B3_ANCHOR != 0 else np.inf)
                fails = []
                if not (np.isfinite(b_rel) and b_rel <= B3_ANCHOR_RELTOL):
                    fails.append(f"b3 relerr={b_rel:.3e} > {B3_ANCHOR_RELTOL:.0e}")
                if col.n != N_FQ_ANCHOR:
                    fails.append(f"n_firmquarters {col.n} != {N_FQ_ANCHOR}")
                if fails:
                    raise SystemExit(
                        "DRIFT GATE FAILED on " + BASELINE_LBL + ": " + "; ".join(fails) +
                        ".\nThis row IS the canonical P0 headline, so a mismatch means a "
                        "STALE PANEL OR WRONG VINTAGE (c6_panel_preP0.dta is a sibling of "
                        "c6_panel.dta). Refusing to report a menu built on the wrong data."
                    )
                print(f"    b3 relerr       = {b_rel:.3e}  -> PASS (6 sig figs)")
                # p is an MC CROSS-CHECK ONLY, never a gate: this script sorts the
                # collapse canonically BEFORE pd.factorize while run_ri_3pairwise.py
                # (which produced the 0.8014 anchor) did not, so the permutation
                # reference distribution is not draw-for-draw identical. The observed
                # ~0.8154 vs 0.8014 gap is that documented ordering effect, not drift.
                d_p = p_free - RI_P_FREE_ANCHOR
                pflag = ("" if abs(d_p) <= P_DRIFT_TOL or SMOKE
                         else "  <-- DRIFT > 0.02 (investigate)")
                print(f"    p_free anchor={RI_P_FREE_ANCHOR:.4f}  got={p_free:.4f}  "
                      f"d={d_p:+.4f}{pflag}")
                if SMOKE:
                    print("    (SMOKE run: N_PERM tiny -> p not comparable to the anchor; "
                          "the b3 / n_fq gates are what this proves.)")
                print()

    res = pd.DataFrame(rows)
    # required columns first, provenance after
    lead = ["variant", "b3", "p_free", "p_circ"]
    res = res[lead + [c for c in res.columns if c not in lead]]
    res.to_csv(RESULT_CSV, index=False)
    print(f"\nwrote {RESULT_CSV.name}  ({len(res)} rows)")
    print("  p_circ (circular shift) is the SERIAL-ROBUST ARBITER: S_t is serially")
    print("  correlated under the baseline construction (P1a), which makes the free")
    print("  permutation p anti-conservative. Where the two diverge, read p_circ.")
    print("  b3 magnitudes are comparable ACROSS MENU VARIANTS because every menu")
    print(f"  column is standardized on the panel quarters; {BASELINE_LBL} is on the")
    print("  RAW shock scale and is a continuity anchor only — compare its p, not its b3.")
    print("  Variant SELECTION is governed solely by the pre-registered whiteness rule")
    print("  on shock_menu_diagnostics.csv. Nothing here may be used to choose it.")

    if mismatches:
        print("\n  *** ANCHOR MISMATCHES (anchor_mismatch=True in the CSV) ***")
        for lbl, r in mismatches:
            print(f"    {lbl:<34s} relerr={r:.3e}")
        print(f"    Tolerance is {ANCHOR_RELTOL:.0e}; only a singleton-drop-sized gap is")
        print(f"    expected. Verify {ANCHOR_CSV.name} is the CURRENT full-panel .do run.")
    else:
        print(f"\n  anchor cross-check: all rows within {ANCHOR_RELTOL:.0e} of the Stata leg.")

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
    print("=" * 92)


if __name__ == "__main__":
    main()
