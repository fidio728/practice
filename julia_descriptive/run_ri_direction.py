"""
run_ri_direction.py — randomization inference for the DIRECTION SPLIT
(sell-to-China vs buy-from-China link-count shares) under the 3-pairwise FE.
Companion to run_direction_split.do; RI machinery mirrors run_ri_tercile.py.

Collapse to the US-NONUS firm-quarter difference dy; two-way (firm x quarter)
demean; regressors [sell, buy, sell*S, buy*S] with S = S_{t-1} (s_lag, the
PRIMARY lagged-shock timing of 2026-08-08 that run_direction_split.do runs;
firm-quarters with missing S_{t-1} are dropped before demeaning, matching
reghdfe); permute the quarter S_{t-1} values (only the two shock interactions
move; sell/buy themselves are fixed firm-quarter attributes). Because the four
regressors are correlated (sell_lag + buy_lag = cn_lag additively), each
permutation re-solves the FULL 4x4 normal equations — never single-variable FWL.

DRIFT ANCHORS (never hardcoded): the Stata MAIN triples are read at runtime
from the living output/direction_vce_diag.csv (spec=="d_main", coef in
{us_sell_slag, us_buy_slag}), written by run_direction_split.do on the same
panel vintage. Missing file => cross-check reported unavailable; stale layout
=> abort.

Reported statistics (decoupling predicts all three negative):
    (1) b3_sell             — US x sell x S triple coefficient
    (2) b3_buy              — US x buy  x S triple coefficient
    (3) b3_sell - b3_buy    — directional contrast (H0 of the pooled spec)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")
N_PERM = 5000
DEMEAN_ITERS = 30
SEED = 20260702

df, _ = pyreadstat.read_dta((OUT / "c6_panel.dta").as_posix())

piv = df.pivot_table(index=["firm_str", "rdate"], columns="us", values="dw",
                     aggfunc="first")
if not {0, 1}.issubset(piv.columns):
    raise SystemExit("panel is not US/NONUS paired")
dy = (piv[1] - piv[0]).rename("dy")

meta = (df.groupby(["firm_str", "rdate"])
          .agg(sell=("sell_lag", "first"), buy=("buy_lag", "first"),
               cn=("cn_lag", "first"), s=("s_lag", "first")))  # S_{t-1} PRIMARY
# group-invariance guard: sell/buy are firm-level attributes and must be
# identical on the US and NONUS rows of a firm-quarter (checked, not assumed)
for _c in ["sell_lag", "buy_lag"]:
    if df.groupby(["firm_str", "rdate"])[_c].nunique(dropna=False).max() > 1:
        raise SystemExit(f"{_c} varies within a firm-quarter — collapse by 'first' unsafe")

d = pd.concat([dy, meta], axis=1).reset_index()
d = d.dropna(subset=["dy", "sell", "buy", "cn", "s"]).reset_index(drop=True)

# additive identity check (pipeline-level guard already exists; belt here)
if (d["sell"] + d["buy"] - d["cn"]).abs().max() > 1e-9:
    raise SystemExit("sell + buy != cn on the collapsed panel — upstream bug")

firm_c = pd.factorize(d["firm_str"])[0]
qcode = pd.factorize(d["rdate"])[0]
nq = int(qcode.max()) + 1
n_fq = len(d)
Svec = np.zeros(nq)
Svec[qcode] = d["s"].to_numpy(float)
if d.groupby(qcode)["s"].nunique().max() != 1:
    raise SystemExit("shock not constant within quarter")
print(f"panel: {n_fq:,d} firm-quarters, {nq} quarters, {pd.Series(firm_c).nunique():,d} firms")

def make_demeaner(firm_c, q_c, iters):
    nf, nqq = firm_c.max() + 1, q_c.max() + 1
    fcount = np.bincount(firm_c, minlength=nf).astype(float)
    qcount = np.bincount(q_c, minlength=nqq).astype(float)
    def demean(x):
        x = x.copy()
        for _ in range(iters):
            x -= (np.bincount(firm_c, weights=x, minlength=nf) / fcount)[firm_c]
            x -= (np.bincount(q_c, weights=x, minlength=nqq) / qcount)[q_c]
        return x
    return demean

demean = make_demeaner(firm_c, qcode, DEMEAN_ITERS)

y_dm = demean(d["dy"].to_numpy(float))
sell_raw = d["sell"].to_numpy(float)
buy_raw = d["buy"].to_numpy(float)
sell_dm = demean(sell_raw)
buy_dm = demean(buy_raw)
print(f"corr(sell, buy) raw = {np.corrcoef(sell_raw, buy_raw)[0,1]:.3f}")

def stats_from_shock(Sq):
    srow = Sq[qcode]
    xs = demean(sell_raw * srow)
    xb = demean(buy_raw * srow)
    X = np.column_stack([sell_dm, buy_dm, xs, xb])   # [b2s, b2b, b3s, b3b]
    XtX = X.T @ X
    Xty = X.T @ y_dm
    try:
        b = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        b = np.linalg.lstsq(X, y_dm, rcond=None)[0]
    return b[2], b[3], b[2] - b[3]

# conditioning check on the observed design matrix: solve() only raises on
# exact singularity; warn on mere ill-conditioning (sell+buy=cn correlation)
_srow = Svec[qcode]
_X = np.column_stack([sell_dm, buy_dm, demean(sell_raw * _srow), demean(buy_raw * _srow)])
_cond = np.linalg.cond(_X.T @ _X)
print(f"cond(XtX) at observed shock = {_cond:.2e}")
if _cond > 1e8:
    print("WARNING: XtX ill-conditioned (>1e8) — point estimates computed via lstsq "
          "in stats_from_shock's fallback may differ; interpret with care")

b3s_obs, b3b_obs, bdiff_obs = stats_from_shock(Svec)
print(f"observed b3_sell        = {b3s_obs:.6e}")
print(f"observed b3_buy         = {b3b_obs:.6e}")
print(f"observed b3_sell-b3_buy = {bdiff_obs:.6e}")

# ---- living Stata anchors (never hardcoded): direction_vce_diag.csv ---------
_diag_path = OUT / "direction_vce_diag.csv"
if _diag_path.exists():
    _dg = pd.read_csv(_diag_path)
    if not {"spec", "coef", "b"} <= set(_dg.columns):
        raise SystemExit(f"{_diag_path.name} lacks spec/coef/b — stale layout; "
                         "re-run run_direction_split.do.")
    _row_s = _dg[(_dg["spec"] == "d_main") & (_dg["coef"] == "us_sell_slag")]
    _row_b = _dg[(_dg["spec"] == "d_main") & (_dg["coef"] == "us_buy_slag")]
    if len(_row_s) != 1 or len(_row_b) != 1:
        raise SystemExit(f"{_diag_path.name}: no unique d_main us_sell_slag/"
                         "us_buy_slag rows — it predates the S_{t-1} switch; "
                         "re-run run_direction_split.do.")
    _st_s, _st_b = float(_row_s["b"].iloc[0]), float(_row_b["b"].iloc[0])
    print(f"expected (Stata MAIN, living {_diag_path.name}): sell {_st_s:+.6e}, "
          f"buy {_st_b:+.6e}, sell-buy {_st_s - _st_b:+.6e}")
    for _lbl, _py, _st in [("sell", b3s_obs, _st_s), ("buy", b3b_obs, _st_b)]:
        _rel = abs(_py - _st) / max(abs(_st), 1e-300)
        if _rel > 1e-3:
            print(f"    *** ANCHOR MISMATCH on b3_{_lbl}: python {_py:.6e} vs "
                  f"stata {_st:.6e} (rel {_rel:.2e} > 1e-3). Expected gap is "
                  "singleton-drop-sized only — stale panel or broken collapse; "
                  "do NOT cite this RI run.")
else:
    print(f"NOTE: {_diag_path.name} not found — Stata cross-check UNAVAILABLE. "
          "Run run_direction_split.do on the current panel before citing.")

rng = np.random.default_rng(SEED)
cnt = np.zeros(3, dtype=int)
obs = np.array([b3s_obs, b3b_obs, bdiff_obs])
for _ in range(N_PERM):
    Sp = rng.permutation(Svec)
    v = np.array(stats_from_shock(Sp))
    cnt += (np.abs(v) >= np.abs(obs) - 1e-300)

ps = (cnt + 1) / (N_PERM + 1)
rows = [
    {"statistic": "b3_sell",           "b_obs": b3s_obs,  "ri_p": ps[0], "n_fq": n_fq},
    {"statistic": "b3_buy",            "b_obs": b3b_obs,  "ri_p": ps[1], "n_fq": n_fq},
    {"statistic": "b3_sell_minus_buy", "b_obs": bdiff_obs, "ri_p": ps[2], "n_fq": n_fq},
]
pd.DataFrame(rows).to_csv(OUT / "ri_direction_results.csv", index=False)
print(f"\n(N_PERM={N_PERM}, iters={DEMEAN_ITERS}, seed={SEED})")
print(f"{'statistic':20s} {'b_obs':>14s} {'ri_p_2side':>12s} {'n_fq':>10s}")
for r in rows:
    print(f"{r['statistic']:20s} {r['b_obs']:14.6e} {r['ri_p']:12.4f} {r['n_fq']:10,d}")
print(f"\nwrote {(OUT / 'ri_direction_results.csv').as_posix()}")
