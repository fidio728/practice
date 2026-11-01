"""
run_ri_tercile.py — randomization inference for the SHOCK-TERCILE dose menu
under the 3-pairwise FE (fq firm×quarter + gq group×quarter + ig firm×group),
the tercile analogue of the retired tail menu (run_tail_3pairwise.do) / run_ri_3pairwise.py.

Design (mirrors the retired tail menu, run_tail_3pairwise.do, exactly, but replaces the single 2σ tail
dummy with three shock terciles cut on the 82 DISTINCT quarters):

    dw_it = β2·cn_it
          + βT1·(cn_it·D_T1(t)) + βT3·(cn_it·D_T3(t))
          + fq + gq + ig + error                       (T2 = middle = base)

On the balanced 2-per-(firm,quarter) panel the three pairwise FE collapse, on the
US−NONUS within-firm-quarter difference Δy = dw(US) − dw(NONUS), to a FIRM + QUARTER
two-way-FE regression of Δy on cn, cn·D_T1, cn·D_T3 (ig → firm intercept; gq → quarter
intercept; fq absorbed by the pairwise difference itself). cn_lag is constant within a
(firm,quarter) cell (verified: same on US and NONUS rows), so cn = cn_lag survives.

TERCILE CUT (critical gotcha): terciles are cut on the 82 DISTINCT quarterly
S_{t-1} values (s_lag, PRIMARY lagged-shock timing 2026-08-08, exactly the
variable run_tercile_3pairwise.do:89 cuts), never on the full rows — cutting on
rows would let uneven cell counts bias the quantiles. Bottom third → D_T1, top third →
D_T3, middle third → T2 (base). Realized bins: 28 / 27 / 27 (the boundary quarter
falls in T1 because D_T1 uses <=), matching run_tercile_3pairwise.do exactly
(verified: Stata _pctile and np.quantile agree here because (82-1)/3 is an
integer, so no interpolation occurs; an assert below guards this).

Two reported statistics (both on the headline outcome dw):
    (1) β₃(T3)              — top-tercile triple coefficient (dose at the strong end)
    (2) β₃(T3) − β₃(T1)     — top-minus-bottom dose contrast (monotonicity test)
Decoupling predicts BOTH negative (US pulls away from high-CN firms more under
top-tercile shocks); positive values run AGAINST decoupling.

RI (sharp null of no differential dose response): permute the 82 quarter shock values,
re-cut terciles on the permuted values (bin sizes stay ~27/28/27 automatically, since a
permutation only reshuffles the same 82 values across quarters), rebuild cn·D_T1 and
cn·D_T3, two-way-demean them, and re-solve the SAME 3-variable system. Because D_T1 and
D_T3 partition the quarters and both interact cn, the two triple regressors are
correlated: we solve the full 3×3 normal equations (cn, x1, x3) jointly and read βT1,
βT3 off the solution — NOT a single-variable FWL of x3 on cn alone.

Δy, cn, and the demeaner are fixed across permutations; only the tercile assignment
(hence x1, x3) moves. Two-sided p for each statistic.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")
N_PERM = 5000
DEMEAN_ITERS = 30
SEED = 20260702
N_TERCILES = 3

# ----------------------------------------------------------------------------
# 1. Read the CURRENT panel .dta directly (reading the live .dta is inherently
#    fresh — no stale intermediate can slip in) and collapse to the firm-quarter
#    US−NONUS difference panel.
# ----------------------------------------------------------------------------
df, _ = pyreadstat.read_dta((OUT / "c6_panel.dta").as_posix())

# US−NONUS difference of dw within each (firm, quarter) cell
piv = df.pivot_table(index=["firm_str", "rdate"], columns="us", values="dw",
                     aggfunc="first")
if not {0, 1}.issubset(piv.columns):
    raise SystemExit("panel is not US/NONUS paired: us column must take 0 and 1")
dy = (piv[1] - piv[0]).rename("dy")

# cn_lag and s_lag are constant within a (firm, quarter) cell -> take first.
# S_{t-1} PRIMARY (2026-08-08): run_tercile_3pairwise.do cuts its terciles on
# the DISTINCT quarterly s_lag values; this RI twin MUST cut on the same
# variable or it tests a different design.
meta = (df.groupby(["firm_str", "rdate"])
          .agg(cn=("cn_lag", "first"), s=("s_lag", "first")))

d = pd.concat([dy, meta], axis=1).reset_index()
_nq_all = d["rdate"].nunique()
d = d.dropna(subset=["dy", "cn", "s"]).reset_index(drop=True)
if d["rdate"].nunique() != _nq_all:
    raise SystemExit(
        f"s_lag missing on {_nq_all - d['rdate'].nunique()} whole quarter(s) — "
        "the .do requires S_{t-1} on EVERY panel quarter (exit 459 there); "
        "lag-propagation bug upstream, refusing to cut terciles.")

# integer codes for firm / quarter (the two FE dimensions of the diff panel)
d["firm_c"] = pd.factorize(d["firm_str"])[0]
d["q_c"] = pd.factorize(d["rdate"])[0]
firm_c = d["firm_c"].to_numpy()
qcode = d["q_c"].to_numpy()
nq = int(qcode.max()) + 1
n_fq = d.shape[0]

# per-quarter S_{t-1} vector (s_lag is constant within quarter -> 82 distinct values)
Svec = np.zeros(nq)
Svec[qcode] = d["s"].to_numpy(float)
# sanity: s_lag must be constant within quarter
if d.groupby("q_c")["s"].nunique().max() != 1:
    raise SystemExit("s_lag is not constant within quarter — tercile cut would be ill-defined")
print(f"panel: {n_fq:,d} firm-quarters, {nq} distinct quarters, {d['firm_c'].nunique():,d} firms")

# ----------------------------------------------------------------------------
# 2. Two-way (firm × quarter) demeaner via fast bincount alternating projections.
# ----------------------------------------------------------------------------
def make_demeaner(firm_c, q_c, iters):
    nf = firm_c.max() + 1
    nqq = q_c.max() + 1
    fcount = np.bincount(firm_c, minlength=nf).astype(float)
    qcount = np.bincount(q_c, minlength=nqq).astype(float)
    def demean(x):
        x = x.copy()
        for _ in range(iters):
            fm = np.bincount(firm_c, weights=x, minlength=nf) / fcount
            x = x - fm[firm_c]
            qm = np.bincount(q_c, weights=x, minlength=nqq) / qcount
            x = x - qm[q_c]
        return x
    return demean

demean = make_demeaner(firm_c, qcode, DEMEAN_ITERS)

# fixed pieces (do not depend on the shock permutation)
y_dm = demean(d["dy"].to_numpy(float))     # demeaned outcome
cn_raw = d["cn"].to_numpy(float)
cn_dm = demean(cn_raw)                       # demeaned cn (the β2 regressor)

# ----------------------------------------------------------------------------
# 3. Tercile cut on the 82 DISTINCT quarter shock values (quarter layer).
#    Cutpoints from the sorted 82 values; a permutation only reshuffles the same
#    82 values across quarters, so the value-space cutpoints — and hence the bin
#    sizes (~27/28/27) — are invariant. We still recut every permutation for
#    clarity (it is the same operation applied to the permuted quarter->value map).
# ----------------------------------------------------------------------------
def tercile_dummies(Sq):
    """Return (D_T1, D_T3) length-nq boolean arrays: bottom / top third of the
    82 quarter shock values (middle third = base). Cut via np.quantile thirds."""
    c1, c2 = np.quantile(Sq, [1.0 / 3.0, 2.0 / 3.0])
    D_T1 = Sq <= c1
    D_T3 = Sq > c2
    return D_T1, D_T3

# report observed bin sizes for cross-checking against the Stata _pctile cut
_d1, _d3 = tercile_dummies(Svec)
_n1, _n3 = int(_d1.sum()), int(_d3.sum())
_n2 = nq - _n1 - _n3
print(f"tercile bin sizes on {nq} quarters: T1={_n1}  T2={_n2}  T3={_n3}")
# balance guard (mirrors the .do): a cut bug still sums to nq, so bound each bin
for _lbl, _n in [("T1", _n1), ("T2", _n2), ("T3", _n3)]:
    if not (25 <= _n <= 29):
        raise SystemExit(f"tercile bin {_lbl}={_n} badly imbalanced — cutpoint bug "
                         f"(np.quantile/_pctile agreement only guaranteed at nq=82)")

# ----------------------------------------------------------------------------
# 4. Joint 3×3 solve. Statistics: βT3 and βT3 − βT1.
# ----------------------------------------------------------------------------
def stats_from_shock(Sq):
    D_T1, D_T3 = tercile_dummies(Sq)
    x1 = demean(cn_raw * D_T1[qcode].astype(float))   # cn · D_T1, two-way demeaned
    x3 = demean(cn_raw * D_T3[qcode].astype(float))   # cn · D_T3, two-way demeaned
    X = np.column_stack([cn_dm, x1, x3])              # [β2, βT1, βT3]
    XtX = X.T @ X
    Xty = X.T @ y_dm
    try:
        b = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        b = np.linalg.lstsq(X, y_dm, rcond=None)[0]
    bT1, bT3 = b[1], b[2]
    return bT3, (bT3 - bT1)

b_T3_obs, b_diff_obs = stats_from_shock(Svec)
print(f"observed b3(T3)        = {b_T3_obs:.6e}")
print(f"observed b3(T3)-b3(T1) = {b_diff_obs:.6e}")

# ---- living Stata anchors (never hardcoded): tercile_vce_diag.csv -----------
_diag_path = OUT / "tercile_vce_diag.csv"
if _diag_path.exists():
    _dg = pd.read_csv(_diag_path)
    if not {"spec", "coef", "b"} <= set(_dg.columns):
        raise SystemExit(f"{_diag_path.name} lacks spec/coef/b — stale layout; "
                         "re-run run_tercile_3pairwise.do.")
    _r3 = _dg[(_dg["spec"] == "m_main") & (_dg["coef"] == "us_cn_t3")]
    _r1 = _dg[(_dg["spec"] == "m_main") & (_dg["coef"] == "us_cn_t1")]
    if len(_r3) != 1 or len(_r1) != 1:
        raise SystemExit(f"{_diag_path.name}: no unique m_main us_cn_t3/us_cn_t1 "
                         "rows — re-run run_tercile_3pairwise.do.")
    _st3 = float(_r3["b"].iloc[0])
    _stdiff = _st3 - float(_r1["b"].iloc[0])
    print(f"expected (Stata MAIN, living {_diag_path.name}): b3(T3) {_st3:+.6e}, "
          f"T3-T1 {_stdiff:+.6e}")
    for _lbl, _py, _st in [("T3", b_T3_obs, _st3), ("T3-T1", b_diff_obs, _stdiff)]:
        _rel = abs(_py - _st) / max(abs(_st), 1e-300)
        if _rel > 1e-3:
            print(f"    *** ANCHOR MISMATCH on {_lbl}: python {_py:.6e} vs stata "
                  f"{_st:.6e} (rel {_rel:.2e} > 1e-3). Expected gap is "
                  "singleton-drop-sized only — stale panel or broken collapse; "
                  "do NOT cite this RI run.")
else:
    print(f"NOTE: {_diag_path.name} not found — Stata cross-check UNAVAILABLE. "
          "Run run_tercile_3pairwise.do on the current panel before citing.")

# ----------------------------------------------------------------------------
# 5. Permutation loop — two-sided p for each statistic.
# ----------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
cnt_T3 = 0
cnt_diff = 0
for _ in range(N_PERM):
    Sp = rng.permutation(Svec)          # permute the 82 quarter shock values
    bT3_p, bdiff_p = stats_from_shock(Sp)
    if abs(bT3_p) >= abs(b_T3_obs) - 1e-300:
        cnt_T3 += 1
    if abs(bdiff_p) >= abs(b_diff_obs) - 1e-300:
        cnt_diff += 1

p_T3 = (cnt_T3 + 1) / (N_PERM + 1)
p_diff = (cnt_diff + 1) / (N_PERM + 1)

# ----------------------------------------------------------------------------
# 6. Output.
# ----------------------------------------------------------------------------
rows = [
    {"statistic": "b3_T3",        "b_obs": b_T3_obs,  "ri_p": p_T3,   "n_fq": n_fq},
    {"statistic": "b3_T3_minus_T1", "b_obs": b_diff_obs, "ri_p": p_diff, "n_fq": n_fq},
]
res = pd.DataFrame(rows)
res.to_csv(OUT / "ri_tercile_results.csv", index=False)

print(f"\n(N_PERM={N_PERM}, iters={DEMEAN_ITERS}, seed={SEED})")
print(f"{'statistic':18s} {'b_obs':>14s} {'ri_p_2side':>12s} {'n_fq':>10s}")
for r in rows:
    print(f"{r['statistic']:18s} {r['b_obs']:14.6e} {r['ri_p']:12.4f} {r['n_fq']:10,d}")
print(f"\nwrote {(OUT / 'ri_tercile_results.csv').as_posix()}")
