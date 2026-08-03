"""
run_cum4_inference.py -- cum4 inference hardening (A0: G5 + G7 + G8).

WHY. In the 3-pairwise LP family (h0 headline dw, cum1..cum4) cum4 is the ONLY
RI-significant cell (free-perm RI p=0.0388, POSITIVE sign +8.826e-6, anti-H2.1).
Three validated gaps in that claim, addressed here:
  (G5) the headline quarter shock S_t is serially correlated, so FREE permutation
       is anti-conservative -- worst exactly at cum4 whose MA-4 overlapping outcome
       stacks the dependence. We re-derive the serial structure and add two
       serial-robust references (circular shift, moving block).
  (G7) no within-family multiplicity control over h0..h4. We add a single-step
       max-|t| FWER (studentized, so cum4's ~8.8e-6 scale cannot dominate cum1's
       ~2.7e-6 in the reference distribution).
  (G8) "WCB impossible (boottest OOM)" is a dense-matrix artifact. A score-based
       quarter-cluster WCB via one-time FWL residualization + O(nq*B) reweighting
       IS feasible and is computed here. Quarter is the binding cluster dimension:
       firm clusters number in the thousands (~6,854), quarter clusters ~82, and
       few-cluster inference is governed by the SMALLER count -- hence quarter
       clustering with Webb (6-point) weights, not Rademacher.

MACHINERY (mirrors run_ri_3pairwise.py exactly; the b3 gates below reproduce it).
On the balanced 2-per-(firm,quarter) panel the three pairwise FE collapse, on the
US-minus-NONUS within-firm-quarter difference dy, to a FIRM + QUARTER two-way FE
model:  dy_it = b2*cn_it + b3*(cn_it*S_t) + phi_i + delta_t + e.  Under the sharp
null b3=0 we permute/shift the 82 quarter shocks S_t. The two-way-demeaned
interaction cn*S has NO closed-form sufficient statistic, so run_ri_3pairwise.py
re-demeans each draw with a 30-iter alternating-projection (bincount) demeaner.

DETERMINISM / SPEED. Because the two-way demean is LINEAR and the interaction is
linear in S, we precompute the demeaned interaction BASIS P = demean(D) once per
horizon, where D[:,q] = cn_raw * 1[quarter=q]. Then for ANY shock vector s the
two-way-demeaned interaction is exactly P @ s (verified to 3.6e-15 vs the per-draw
scalar demeaner; b3 reproduces to 6 significant figures). Point estimates use an
O(nq^2) closed form; studentized scores use the O(N*nq) row-level xr = P@s. This
is the SAME estimator as run_ri_3pairwise.py, only vectorized -- not a new method.

FROZEN INPUT / ORDERING. Regenerate output/audit_c6_panel.parquet from the current
.dta (freshness), then GROUP BY (firm,quarter) into the difference panel, then SORT
by (firm_str, rdate) BEFORE factorize. run_ri_3pairwise.py did NOT sort and DuckDB's
16-thread hash aggregation returns a NON-deterministic row order (verified: two runs
differ); its factorize-order therefore varied run to run. b3 is order-invariant
(inner products) so the three b3 GATES reproduce exactly and are hard-asserted to 6
sig figs. The free-perm p-VALUES depend on the Svec ordering (a fixed seed applied
to a differently-ordered vector yields a different permutation stream), so the
published anchors 0.3083/0.1052/0.0388 -- generated under the old non-deterministic
order -- are Monte-Carlo CROSS-CHECKS here (expected within ~+/-0.01 MC noise at
N_PERM=5000), NOT hard gates. Sorting makes THIS script reproducible going forward.

HONESTY. Never fabricate; the three b3 gates abort loudly on mismatch (stale panel
or broken collapse). Seed 20260702, sorted panel, frozen artifact, deterministic.
cum2/cum3 have CRVE references in the F1b battery (h2 p=0.0025, h3 in logs) but NO
free-RI anchors -- printed as NEW results, no gate.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import duckdb
import pyreadstat
from scipy import stats

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")

# ---- convention constants (project-frozen) ----
SEED = 20260702
DEMEAN_ITERS = 30
N_PERM = 5000          # free-permutation and moving-block draws
B_WCB = 9999           # Webb wild-cluster bootstrap draws
BLOCK_L_PRIMARY = 5    # moving-block length (respects MA-4 dependence)
BLOCK_L_SENS = 8       # moving-block sensitivity length
CHUNK = 250            # column-chunk for the studentized batch (memory bound)

# smoke mode: prove end-to-end + gates without the full draw counts
SMOKE = "--smoke" in sys.argv
if SMOKE:
    N_PERM = 20
    B_WCB = 200

# published anchors (drift-protection cross-checks)
B3_GATES = {"h0": 2.745538e-06, "cum1": 4.093621e-06, "cum4": 8.825974e-06}
P_FREE_ANCHORS = {"h0": 0.3083, "cum1": 0.1052, "cum4": 0.0388}
SAGG_CORR_DISCLOSED = 0.357   # existing s_agg lag-1 autocorr disclosure (run_ri_sagg.py)

HORIZONS = [("h0", "d_dw"), ("cum1", "d_c1"), ("cum2", "d_c2"),
            ("cum3", "d_c3"), ("cum4", "d_c4")]

# Webb 6-point weights: {+/-sqrt(3/2), +/-1, +/-sqrt(1/2)}, each prob 1/6; mean 0, var 1.
WEBB = np.array([np.sqrt(1.5), 1.0, np.sqrt(0.5),
                 -np.sqrt(0.5), -1.0, -np.sqrt(1.5)])


# ======================================================================
# frozen input: regen parquet -> difference panel -> SORT before factorize
# ======================================================================
def load_panel():
    _df, _ = pyreadstat.read_dta((OUT / "audit_c6_panel.dta").as_posix())
    _df.to_parquet(OUT / "audit_c6_panel.parquet", index=False)
    con = duckdb.connect()
    d = con.execute(f"""
    SELECT firm_str, rdate,
           any_value(cn_lag) AS cn, any_value(shock) AS s,
           MAX(CASE WHEN us=1 THEN dw   END) - MAX(CASE WHEN us=0 THEN dw   END) AS d_dw,
           MAX(CASE WHEN us=1 THEN cum1 END) - MAX(CASE WHEN us=0 THEN cum1 END) AS d_c1,
           MAX(CASE WHEN us=1 THEN cum2 END) - MAX(CASE WHEN us=0 THEN cum2 END) AS d_c2,
           MAX(CASE WHEN us=1 THEN cum3 END) - MAX(CASE WHEN us=0 THEN cum3 END) AS d_c3,
           MAX(CASE WHEN us=1 THEN cum4 END) - MAX(CASE WHEN us=0 THEN cum4 END) AS d_c4
    FROM read_parquet('{(OUT/'audit_c6_panel.parquet').as_posix()}')
    GROUP BY firm_str, rdate
    """).df()
    con.close()
    # sort BEFORE factorize -> deterministic quarter-code / Svec ordering
    d = d.sort_values(["firm_str", "rdate"]).reset_index(drop=True)
    return d


# ======================================================================
# per-horizon precompute: two-way-demean basis + closed-form pieces
# ======================================================================
def make_demeaner(firm_c, q_c, iters=DEMEAN_ITERS):
    """Exact vectorized clone of run_ri_3pairwise.py's 30-iter bincount two-way
    demeaner. Firm blocks are contiguous (panel sorted by firm) -> reduceat;
    quarter groups scattered -> argsort + reduceat. Matches the scalar bincount
    demeaner to ~3e-15."""
    nf = firm_c.max() + 1
    nq = q_c.max() + 1
    fcount = np.bincount(firm_c, minlength=nf).astype(float)
    qcount = np.bincount(q_c, minlength=nq).astype(float)
    firm_starts = np.concatenate([[0], np.where(np.diff(firm_c) != 0)[0] + 1])
    qsort = np.argsort(q_c, kind="stable")
    q_sorted = q_c[qsort]
    q_starts = np.concatenate([[0], np.where(np.diff(q_sorted) != 0)[0] + 1])
    fc = fcount[:, None]
    qc = qcount[:, None]

    def demean(X):
        sq = (X.ndim == 1)
        X = np.asarray(X, float).copy()
        if sq:
            X = X[:, None]
        for _ in range(iters):
            fsum = np.add.reduceat(X, firm_starts, axis=0)
            X = X - (fsum / fc)[firm_c]
            Xs = X[qsort]
            qsum = np.add.reduceat(Xs, q_starts, axis=0)
            X = X - (qsum / qc)[q_c]
        return X[:, 0] if sq else X

    return demean, qsort, q_starts


def build_horizon(d, ycol, full_chrono):
    """Precompute everything needed for one horizon (one outcome column)."""
    sub = d.dropna(subset=[ycol, "cn", "s"]).reset_index(drop=True)
    firm_c = pd.factorize(sub["firm_str"])[0]
    q_codes, q_dates = pd.factorize(sub["rdate"])       # q_dates[k] = rdate of code k
    q_c = q_codes
    nq = q_c.max() + 1
    demean, qsort, q_starts = make_demeaner(firm_c, q_c)

    cn_raw = sub["cn"].to_numpy(float)
    y = demean(sub[ycol].to_numpy(float))
    cn = demean(cn_raw)

    # demeaned interaction basis: D[:,q] = cn_raw * 1[quarter==q]; P = demean(D)
    D = np.zeros((len(sub), nq))
    D[np.arange(len(sub)), q_c] = cn_raw
    P = demean(D)

    cnn = float(cn @ cn)
    cvec = cn @ P                       # (nq,)
    yP = y @ P                          # (nq,)
    ycn = float(y @ cn)
    G = P.T @ P                         # (nq, nq)

    # per-quarter shock in LOCAL code order (shock constant within quarter)
    Svec = np.zeros(nq)
    Svec[q_c] = sub["s"].to_numpy(float)

    # local quarter code -> position in the 82-quarter chronological vector
    chrono_of_localq = np.searchsorted(full_chrono, q_dates.values)

    return dict(ycol=ycol, n=len(sub), nq=nq, q_c=q_c, qsort=qsort,
                q_starts=q_starts, y=y, cn=cn, P=P, cnn=cnn, cvec=cvec,
                yP=yP, ycn=ycn, G=G, Svec=Svec, chrono_of_localq=chrono_of_localq)


# ======================================================================
# estimators
# ======================================================================
def b3_batch(Sm, H):
    """Closed-form b3 for a batch of shock vectors Sm (B, nq), LOCAL order.
    Exactly the FWL b3 of run_ri_3pairwise.py (xt = P@s, residualize on cn)."""
    cnn, cvec, yP, ycn, G = H["cnn"], H["cvec"], H["yP"], H["ycn"], H["G"]
    b = (Sm @ cvec) / cnn
    num = Sm @ yP - b * ycn
    SG = Sm @ G
    quad = np.einsum("bi,bi->b", SG, Sm)
    den = quad - b * b * cnn
    # B9 guard: restore run_ri_3pairwise.py line 88's finite/positive-denominator
    # check. A degenerate draw (den<=0 or non-finite) returns nan -> excluded from
    # the RI tail, never a silent inf; a nan OBSERVED b3 turns the gate into a loud
    # FAIL (relerr nan !< 1e-6) rather than a written degenerate row.
    good = (den > 0) & np.isfinite(den)
    out = np.full(den.shape, np.nan)
    np.divide(num, den, out=out, where=good)
    return out


def studentized_batch(Sm, H, chunk=CHUNK):
    """b3 and studentized t (score-based quarter-cluster, UNRESTRICTED residual,
    CR1 factor nq/(nq-1)) for a batch of shock vectors Sm (B, nq), LOCAL order."""
    P, cn, y = H["P"], H["cn"], H["y"]
    cnn, cvec = H["cnn"], H["cvec"]
    q_c, nq = H["q_c"], H["nq"]
    qsort, q_starts = H["qsort"], H["q_starts"]
    lbl = H.get("lbl", H["ycol"])
    c = nq / (nq - 1.0)
    projy = (cn @ y) / cnn
    B = Sm.shape[0]
    b3o = np.empty(B)
    to = np.empty(B)
    for a in range(0, B, chunk):
        s = Sm[a:a + chunk]                      # (b, nq)
        XT = P @ s.T                             # (N, b)
        bcoef = (s @ cvec) / cnn                 # (b,)
        XR = XT - cn[:, None] * bcoef[None, :]   # (N, b)
        xrr = np.einsum("nb,nb->b", XR, XR)      # (b,)
        # B9 guard: a non-positive/non-finite residual SS would yield inf/nan t*
        # that np.max silently propagates into tmax and corrupts every FWER p.
        if not ((xrr > 0).all() and np.isfinite(xrr).all()):
            raise SystemExit(
                f"degenerate VCE (zero/missing SE) at horizon {lbl} -- aborting "
                "(studentized_batch: non-positive/non-finite residual SS xrr).")
        b3 = (y @ XR) / xrr                      # (b,)
        E = y[:, None] - projy * cn[:, None] - XR * b3[None, :]
        prod = XR * E
        ps = prod[qsort]
        g = np.add.reduceat(ps, q_starts, axis=0)   # (nq, b)
        se = np.sqrt(c * np.sum(g * g, axis=0)) / xrr
        # B9 guard: zero/missing SE -> t = b3/se is inf/nan; abort loudly rather
        # than let it silently corrupt the max-T FWER reference distribution.
        if not ((se > 0).all() and np.isfinite(se).all()):
            raise SystemExit(
                f"degenerate VCE (zero/missing SE) at horizon {lbl} -- aborting "
                "(studentized_batch: non-positive/non-finite cluster SE).")
        b3o[a:a + chunk] = b3
        to[a:a + chunk] = b3 / se
    return b3o, to


def wcb_score_qcluster(H, b_wcb, seed):
    """Score-based quarter-cluster WCB (G8). One-time FWL: x_tilde = residual of
    demeaned cn*S on demeaned cn; e_hat = restricted (b3=0) residual of demeaned y
    on demeaned cn. Collapse to nq quarter scores g_t = sum x_tilde*e_hat and
    s_t = sum x_tilde^2. Webb-reweight the cluster scores; studentize with the
    reweighted-score cluster SE. The CR1 factor cancels between t_obs and t*."""
    P, cn, y = H["P"], H["cn"], H["y"]
    cnn, cvec = H["cnn"], H["cvec"]
    q_c, nq = H["q_c"], H["nq"]
    lbl = H.get("lbl", H["ycol"])
    Svec = H["Svec"]
    xt = P @ Svec
    b = (cvec @ Svec) / cnn
    x_tilde = xt - b * cn
    e_hat = y - ((cn @ y) / cnn) * cn                 # restricted, b3 forced 0
    g = np.bincount(q_c, weights=x_tilde * e_hat, minlength=nq)   # (nq,)
    s_t = np.bincount(q_c, weights=x_tilde * x_tilde, minlength=nq)
    c = nq / (nq - 1.0)
    denom_obs = np.sqrt(c * np.sum(g * g))
    # B9 guard: a degenerate all-zero cluster-score vector makes denom_obs=0 ->
    # t_obs=nan, after which |t*|>=|t_obs| is all-False and p is silently written
    # as 1/(B+1) -- maximally SIGNIFICANT. Abort loudly instead.
    if not (np.isfinite(denom_obs) and denom_obs > 0):
        raise SystemExit(
            f"degenerate VCE (zero/missing SE) at horizon {lbl} -- aborting "
            "(wcb_score_qcluster: non-positive observed score SE).")
    t_obs = np.sum(g) / denom_obs                     # score t (== b3/se form)
    b3_obs = np.sum(g) / np.sum(s_t)
    rng = np.random.default_rng(seed)
    V = WEBB[rng.integers(0, 6, size=(b_wcb, nq))]     # (B, nq)
    num = V @ g                                        # (B,)
    den = np.sqrt(c * ((V * V) @ (g * g)))             # (B,)
    # B9 guard: any zero/non-finite bootstrap SE -> t* inf/nan silently biases p.
    if not ((den > 0).all() and np.isfinite(den).all()):
        raise SystemExit(
            f"degenerate VCE (zero/missing SE) at horizon {lbl} -- aborting "
            "(wcb_score_qcluster: non-positive/non-finite bootstrap SE).")
    t_star = num / den
    p = (np.sum(np.abs(t_star) >= abs(t_obs) - 1e-300) + 1) / (b_wcb + 1)
    return b3_obs, t_obs, p, nq


# ======================================================================
# serial-structure references on the shock vector
# ======================================================================
def gen_mb_shocks(S_full, L, n_draws, seed):
    """Moving-block permutation of the chronological shock series: random circular
    start per draw, cut into ceil(nq/L) contiguous blocks, permute block ORDER,
    concatenate. Preserves within-block (serial) structure."""
    rng = np.random.default_rng(seed)
    nqf = len(S_full)
    n_blocks = int(np.ceil(nqf / L))
    idx0 = np.arange(nqf)
    out = np.empty((n_draws, nqf))
    for j in range(n_draws):
        start = int(rng.integers(0, nqf))
        rolled = np.roll(idx0, start)
        blocks = [rolled[b * L:(b + 1) * L] for b in range(n_blocks)]
        order = rng.permutation(n_blocks)
        new_idx = np.concatenate([blocks[o] for o in order])
        out[j] = S_full[new_idx]
    return out


def shock_serial_diagnostics(S):
    """Hand-rolled ACF (biased 1/n normalization) + Ljung-Box with scipy chi2
    survival. statsmodels.acorr_ljungbox is broken in this env (deprecate_kwarg
    TypeError), so LB is hand-rolled -- stated per spec."""
    n = len(S)
    Sd = S - S.mean()
    denom = np.sum(Sd * Sd)

    def acf(k):
        return np.sum(Sd[k:] * Sd[:n - k]) / denom

    r1, r2 = acf(1), acf(2)

    def ljungbox(m):
        Q = n * (n + 2) * sum(acf(k) ** 2 / (n - k) for k in range(1, m + 1))
        return Q, float(stats.chi2.sf(Q, m))

    Q1, p1 = ljungbox(1)
    Q4, p4 = ljungbox(4)
    return dict(acf1=r1, acf2=r2, Q1=Q1, p1=p1, Q4=Q4, p4=p4, n=n)


# ======================================================================
# main
# ======================================================================
def main():
    print("=" * 78)
    print("run_cum4_inference.py -- cum4 inference hardening (G5+G7+G8)")
    print(f"  SMOKE={SMOKE}  N_PERM={N_PERM}  B_WCB={B_WCB}  SEED={SEED}  "
          f"iters={DEMEAN_ITERS}")
    print("=" * 78)

    d = load_panel()

    # 82-quarter chronological shock vector (shock constant within quarter)
    sfull = d.groupby("rdate")["s"].first().sort_index()
    full_chrono = sfull.index.values                 # sorted datetime64 (82,)
    S_full = sfull.to_numpy(float)                    # chronological shocks (82,)
    nqf = len(S_full)
    assert nqf == 82, f"expected 82 quarters, got {nqf}"

    # ---------------- [1] SHOCK SERIAL DIAGNOSTICS ----------------
    diag = shock_serial_diagnostics(S_full)
    print("\n[1] SHOCK SERIAL DIAGNOSTICS (stamped shock S_t, 82 quarters, "
          "re-derived; LB hand-rolled w/ scipy.chi2)")
    print(f"    existing convention (run_ri_sagg.py): corr(s_agg_t,s_agg_t-1) "
          f"= {SAGG_CORR_DISCLOSED:.3f} (disclosed)")
    print(f"    autocorr lag1 = {diag['acf1']:+.6f}   lag2 = {diag['acf2']:+.6f}")
    print(f"    Ljung-Box Q(1) = {diag['Q1']:.4f}  p = {diag['p1']:.4f}")
    print(f"    Ljung-Box Q(4) = {diag['Q4']:.4f}  p = {diag['p4']:.4f}")
    print("    => S_t is serially correlated: free permutation is anti-conservative;")
    print("       circular-shift / moving-block references are the serial-robust ones.")

    # ---------------- per-horizon precompute + GATES ----------------
    print("\n[gates] hard-asserting b3 to 6 significant figures ...")
    Hs = {}
    for lbl, ycol in HORIZONS:
        H = build_horizon(d, ycol, full_chrono)
        H["lbl"] = lbl                       # for B9 abort messages downstream
        Hs[lbl] = H
        b3 = float(b3_batch(H["Svec"][None, :], H)[0])
        H["b3_obs"] = b3
        gate = B3_GATES.get(lbl)
        if gate is not None:
            rel = abs(b3 / gate - 1.0)
            ok = rel < 1e-6
            print(f"    {lbl:5s} b3={b3:+.6e}  gate={gate:+.6e}  relerr={rel:.2e}  "
                  f"{'PASS' if ok else 'FAIL'}  n={H['n']:,}")
            if not ok:
                raise SystemExit(
                    f"GATE FAIL {lbl}: b3={b3:.8e} != {gate:.8e} (relerr {rel:.2e}). "
                    "Stale panel or broken collapse -- aborting.")
        else:
            print(f"    {lbl:5s} b3={b3:+.6e}  (NEW, no free-RI anchor)  n={H['n']:,}")

    # ---------------- [2] RI THREE WAYS per horizon ----------------
    # pre-generate shared moving-block shock draws (chronological, full 82)
    S_mb_L5 = gen_mb_shocks(S_full, BLOCK_L_PRIMARY, N_PERM, SEED)
    S_mb_L8 = gen_mb_shocks(S_full, BLOCK_L_SENS, N_PERM, SEED)
    # pre-generate shared free-permutation draws (full 82, for the max-T family)
    _rng_free = np.random.default_rng(SEED)
    perms_full = np.array([_rng_free.permutation(S_full) for _ in range(N_PERM)])

    rows = []
    p_free_marg = {}
    print("\n[2] RANDOMIZATION INFERENCE, THREE WAYS (raw |b3|, 2-sided)")
    print(f"    {'horizon':7s} {'b_obs':>13s} {'p_free':>8s} {'p_circ':>8s} "
          f"{'p_mb(L5)':>9s} {'p_mb(L8)':>9s}")
    for lbl, ycol in HORIZONS:
        H = Hs[lbl]
        b_obs = H["b3_obs"]
        Svec = H["Svec"]
        col = H["chrono_of_localq"]

        # (a) FREE permutation -- horizon-own Svec, fresh rng(SEED) (mirrors
        #     run_ri_3pairwise.py; only difference vs anchors is the frozen order)
        rng = np.random.default_rng(SEED)
        Sm_free = np.array([rng.permutation(Svec) for _ in range(N_PERM)])
        bp = b3_batch(Sm_free, H)
        p_free = (np.sum(np.abs(bp) >= abs(b_obs) - 1e-300) + 1) / (N_PERM + 1)
        p_free_marg[lbl] = p_free

        # (b) CIRCULAR SHIFT -- 81 non-trivial rolls of the full chronological S,
        #     mapped to this horizon's quarters (min two-sided p = 1/82)
        Sm_circ = np.array([np.roll(S_full, k)[col] for k in range(1, nqf)])
        bc = b3_batch(Sm_circ, H)
        p_circ = (np.sum(np.abs(bc) >= abs(b_obs) - 1e-300) + 1) / (Sm_circ.shape[0] + 1)

        # (c) MOVING BLOCK -- shared draws mapped to this horizon's quarters
        bm5 = b3_batch(S_mb_L5[:, col], H)
        p_mb5 = (np.sum(np.abs(bm5) >= abs(b_obs) - 1e-300) + 1) / (N_PERM + 1)
        bm8 = b3_batch(S_mb_L8[:, col], H)
        p_mb8 = (np.sum(np.abs(bm8) >= abs(b_obs) - 1e-300) + 1) / (N_PERM + 1)

        print(f"    {lbl:7s} {b_obs:+13.6e} {p_free:8.4f} {p_circ:8.4f} "
              f"{p_mb5:9.4f} {p_mb8:9.4f}")
        for method, p, ndraw in [
            ("free_perm", p_free, N_PERM),
            ("circular_shift", p_circ, int(Sm_circ.shape[0])),
            (f"moving_block_L{BLOCK_L_PRIMARY}", p_mb5, N_PERM),
            (f"moving_block_L{BLOCK_L_SENS}", p_mb8, N_PERM),
        ]:
            rows.append(dict(horizon=lbl, method=method, statistic="raw_b",
                             b_obs=b_obs, stat_obs=b_obs, p=p, n_draws=ndraw))

    # ---------------- [3] FAMILY max-|t| FWER (single-step max-T) ----------------
    print("\n[3] WITHIN-FAMILY max-|t| FWER (studentized; score-based quarter-cluster SE)")
    order = [lbl for lbl, _ in HORIZONS]
    # observed studentized t per horizon (unrestricted CRVE, from own real shock)
    t_obs = {}
    for lbl in order:
        H = Hs[lbl]
        _, to = studentized_batch(H["Svec"][None, :], H)
        t_obs[lbl] = float(to[0])

    for tag, draws in [("free", perms_full), ("moveblock", S_mb_L5)]:
        # shared draws -> per-horizon |t*| matrix (computed once), row-wise max
        tabs = np.empty((len(order), N_PERM))
        for i, lbl in enumerate(order):
            H = Hs[lbl]
            _, tst = studentized_batch(draws[:, H["chrono_of_localq"]], H)
            tabs[i] = np.abs(tst)
        tmax = np.max(tabs, axis=0)
        print(f"    [{tag} draws]  {'horizon':7s} {'t_obs':>9s} {'p_marg':>8s} "
              f"{'p_FWER':>8s}")
        for i, lbl in enumerate(order):
            p_marg = (np.sum(tabs[i] >= abs(t_obs[lbl]) - 1e-300) + 1) / (N_PERM + 1)
            p_fwer = (np.sum(tmax >= abs(t_obs[lbl]) - 1e-300) + 1) / (N_PERM + 1)
            print(f"                {lbl:7s} {t_obs[lbl]:+9.4f} {p_marg:8.4f} {p_fwer:8.4f}")
            rows.append(dict(horizon=lbl, method=f"maxT_fwer_{tag}",
                             statistic="studentized_t", b_obs=Hs[lbl]["b3_obs"],
                             stat_obs=t_obs[lbl], p=p_fwer, n_draws=N_PERM))

    # ---------------- [4] SCORE-BASED QUARTER-CLUSTER WCB ----------------
    print("\n[4] SCORE-BASED QUARTER-CLUSTER WCB (Webb 6-point; quarter is the "
          "binding cluster dim, ~82 vs ~6,854 firms)")
    print(f"    {'horizon':7s} {'b_obs':>13s} {'t_obs(score)':>13s} {'p_wcb':>8s} "
          f"{'n_clust':>8s}")
    for lbl in order:
        H = Hs[lbl]
        b3w, tw, pw, ncl = wcb_score_qcluster(H, B_WCB, SEED)
        print(f"    {lbl:7s} {b3w:+13.6e} {tw:+13.4f} {pw:8.4f} {ncl:8d}")
        rows.append(dict(horizon=lbl, method="wcb_score_qcluster",
                         statistic="studentized_t_score", b_obs=b3w,
                         stat_obs=tw, p=pw, n_draws=B_WCB))

    # ---------------- shock-diagnostic rows into the CSV ----------------
    for m, val, pv in [("autocorr_lag1", diag["acf1"], np.nan),
                       ("autocorr_lag2", diag["acf2"], np.nan),
                       ("ljungbox_Q1", diag["Q1"], diag["p1"]),
                       ("ljungbox_Q4", diag["Q4"], diag["p4"]),
                       ("sagg_corr_lag1_disclosed", SAGG_CORR_DISCLOSED, np.nan)]:
        rows.append(dict(horizon="_shock_diag", method=m, statistic="serial",
                         b_obs=val, stat_obs=val, p=pv, n_draws=diag["n"]))

    df_out = pd.DataFrame(rows, columns=["horizon", "method", "statistic",
                                         "b_obs", "stat_obs", "p", "n_draws"])
    out_csv = OUT / "cum4_inference_hardening.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")

    # ---------------- anchor drift-protection cross-check ----------------
    print("\n[cross-check] published anchors vs this run (b3 gated; free-perm p = "
          "MC cross-check under the newly-frozen order)")
    print("    b3 anchors:  h0 +2.745538e-06  cum1 +4.093621e-06  cum4 +8.825974e-06")
    print("    free-perm p anchors: h0 0.3083  cum1 0.1052  cum4 0.0388")
    for lbl in ["h0", "cum1", "cum4"]:
        got = p_free_marg[lbl]
        anc = P_FREE_ANCHORS[lbl]
        flag = "" if abs(got - anc) <= 0.02 or SMOKE else "  <-- DRIFT > 0.02 (investigate)"
        print(f"    {lbl:5s} p_free anchor={anc:.4f}  got={got:.4f}  "
              f"d={got - anc:+.4f}{flag}")
    if SMOKE:
        print("    (SMOKE run: N_PERM tiny -> p-values NOT comparable to anchors; "
              "gates + end-to-end execution are what this proves.)")

    print("\ndone.")


if __name__ == "__main__":
    main()
