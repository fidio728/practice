"""
run_ri_fourgroup.py — randomization inference for the ACTIVE-ONLY four-group MAIN
test (US_ACTIVE vs NONUS_ACTIVE) under the 3-pairwise FE. Companion to
run_fourgroup.do; RI machinery mirrors run_ri_direction.py / run_ri_tercile.py.

Design. Restrict the four-group panel to the ACTIVE side {US_ACTIVE, NONUS_ACTIVE}
and collapse to the within-firm-quarter US-minus-NONUS difference
    dy_it = dw(US_ACTIVE)_it - dw(NONUS_ACTIVE)_it.
On the balanced 2-per-(firm,quarter) active panel the three pairwise FE (fq firm x
quarter, gq grp x quarter, ig firm x grp) collapse, on this difference, to a
FIRM + QUARTER two-way-FE regression of dy on [cn, cn x S] (ig -> firm intercept,
gq -> quarter intercept, fq absorbed by the pairwise difference itself). cn_lag is
constant within a (firm,quarter) cell (same on the US and NONUS active rows,
checked) so cn survives the collapse; S is constant within quarter.

    dy_it = b2 * cn_it + b3 * (cn_it * S_t) + firm + quarter + e_it

Reported statistic:
    b3 — the US_ACTIVE-vs-NONUS_ACTIVE differential exposure response to the shock.
Dilution reading (stated with restraint): b3 more negative than the pooled c6
headline is CONSISTENT with passive dilution of the pooled coefficient.

RI (sharp null of no differential response): permute the quarter shock values,
rebuild cn x S, two-way demean, and RE-SOLVE the full 2x2 normal equations for
[cn, cn x S] jointly (cn and cn x S are correlated, so this is NOT a single-variable
FWL of cn x S on nothing). dy, cn, and the demeaner are fixed across permutations;
only the shock interaction moves. Two-sided p.

Two samples, each its own RI p:
    (1) FULL period            — Funds snapshot ~2018-08 gives look-ahead labels
                                  before it; disclosed, not primary.
    (2) POST-2018 (>= 2018-08) — labels PREDETERMINED; the PRIMARY report.
Quarter count and firm count are re-derived per sample; the permutation reshuffles
that sample's own quarter shock vector (re-factorized within the sample). CAVEAT:
the post-2018 sample has only ~22 quarters, so its RI rests on ~22 quarter-level
shock draws — coarser randomization granularity than the 82-quarter full sample.

COUNT RECONCILIATION vs run_fourgroup.do: Python n_fq counts balanced firm-quarter
DIFFERENCE cells, so n_fq ~ (Stata N)/2; reghdfe additionally drops single-quarter
firms as FE singletons while the RI keeps them (they carry ~zero weight after the
firm demean), so the two files' firm/quarter counts differ by exactly those drops.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")
N_PERM = 5000
DEMEAN_ITERS = 30
SEED = 20260702
POST_CUTOFF = pd.Timestamp("2018-08-01")   # matches the .do's rd_m >= tm(2018m8)
ACTIVE_GRPS = ("US_ACTIVE", "NONUS_ACTIVE")

df, _ = pyreadstat.read_dta((OUT / "fourgroup_panel.dta").as_posix())

# ---- restrict to the ACTIVE side and build the US(1)/NONUS(0) indicator --------
act = df[df["grp"].isin(ACTIVE_GRPS)].copy()
if act.empty:
    raise SystemExit("no ACTIVE rows: expected grp in " + repr(ACTIVE_GRPS))
act["us"] = (act["grp"] == "US_ACTIVE").astype(int)
act["rdate"] = pd.to_datetime(act["rdate"])


def collapse(frame):
    """Collapse to the US-minus-NONUS firm-quarter difference panel with cn / S."""
    # key-uniqueness guard: pivot_table(aggfunc='first') would silently mask a
    # duplicate; mirror the builder's (firm, grp, rdate) uniqueness assertion
    if frame.duplicated(["firm_str", "rdate", "us"]).any():
        raise SystemExit("duplicate (firm, quarter, us) rows — builder key regression")
    piv = frame.pivot_table(index=["firm_str", "rdate"], columns="us", values="dw",
                            aggfunc="first")
    if not {0, 1}.issubset(piv.columns):
        raise SystemExit("active panel is not US/NONUS paired: us must take 0 and 1")
    dy = (piv[1] - piv[0]).rename("dy")

    # cn_lag and shock must be constant within a (firm, quarter) cell -> take first
    for _c in ["cn_lag", "shock"]:
        if frame.groupby(["firm_str", "rdate"])[_c].nunique(dropna=False).max() > 1:
            raise SystemExit(f"{_c} varies within a firm-quarter — collapse by 'first' unsafe")
    meta = (frame.groupby(["firm_str", "rdate"])
                 .agg(cn=("cn_lag", "first"), s=("shock", "first")))
    d = pd.concat([dy, meta], axis=1).reset_index()
    d = d.dropna(subset=["dy", "cn", "s"]).reset_index(drop=True)
    return d


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


def run_ri(d, label):
    """Full RI for one sample. Returns (b3_obs, ri_p, n_fq, nq, nfirm)."""
    firm_c = pd.factorize(d["firm_str"])[0]
    qcode = pd.factorize(d["rdate"])[0]
    nq = int(qcode.max()) + 1
    n_fq = len(d)
    nfirm = pd.Series(firm_c).nunique()

    # per-quarter shock vector; shock must be constant within quarter
    Svec = np.zeros(nq)
    Svec[qcode] = d["s"].to_numpy(float)
    if d.groupby(qcode)["s"].nunique().max() != 1:
        raise SystemExit(f"[{label}] shock not constant within quarter")

    demean = make_demeaner(firm_c, qcode, DEMEAN_ITERS)
    y_dm = demean(d["dy"].to_numpy(float))
    cn_raw = d["cn"].to_numpy(float)
    cn_dm = demean(cn_raw)

    def stats_from_shock(Sq):
        xs = demean(cn_raw * Sq[qcode])          # cn x S, two-way demeaned
        X = np.column_stack([cn_dm, xs])         # [b2, b3]
        XtX = X.T @ X
        Xty = X.T @ y_dm
        try:
            b = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            b = np.linalg.lstsq(X, y_dm, rcond=None)[0]
        return b[1]                              # b3 = coefficient on cn x S

    # conditioning check on the observed design (solve() only raises on exact
    # singularity; warn on mere ill-conditioning)
    _X = np.column_stack([cn_dm, demean(cn_raw * Svec[qcode])])
    _cond = np.linalg.cond(_X.T @ _X)
    if _cond > 1e8:
        print(f"[{label}] WARNING: XtX ill-conditioned ({_cond:.2e}>1e8) — "
              "lstsq fallback estimates may differ; interpret with care")

    b3_obs = stats_from_shock(Svec)

    # demean-convergence check on the OBSERVED statistic: recompute with double
    # the alternating-projection iterations; a material drift means 30 iters
    # under-converged and the Stata cross-check would be meaningless.
    demean2 = make_demeaner(firm_c, qcode, 2 * DEMEAN_ITERS)
    y2 = demean2(d["dy"].to_numpy(float))
    cn2 = demean2(cn_raw)
    x2 = demean2(cn_raw * Svec[qcode])
    X2 = np.column_stack([cn2, x2])
    b2v = np.linalg.solve(X2.T @ X2, X2.T @ y2)
    drift = abs(b2v[1] - b3_obs) / max(abs(b3_obs), 1e-300)
    if drift > 1e-6:
        print(f"[{label}] WARNING: demean under-convergence (b3 drift {drift:.2e} "
              f"at 2x iters) — bump DEMEAN_ITERS before citing the cross-check")

    rng = np.random.default_rng(SEED)
    cnt = 0
    for _ in range(N_PERM):
        Sp = rng.permutation(Svec)               # permute this sample's quarter shocks
        if abs(stats_from_shock(Sp)) >= abs(b3_obs) - 1e-300:
            cnt += 1
    ri_p = (cnt + 1) / (N_PERM + 1)
    return b3_obs, ri_p, n_fq, nq, nfirm


rows = []
for label, frame in [
    ("full",      act),
    ("post2018",  act[act["rdate"] >= POST_CUTOFF]),
]:
    d = collapse(frame)
    if d.empty:
        raise SystemExit(f"[{label}] empty collapsed panel")
    b3_obs, ri_p, n_fq, nq, nfirm = run_ri(d, label)
    print(f"[{label:9s}] firm-quarters={n_fq:,d}  firms={nfirm:,d}  quarters={nq}  "
          f"b3={b3_obs:.6e}  ri_p={ri_p:.4f}")
    rows.append({"sample": label, "statistic": "b3_us_active",
                 "b_obs": b3_obs, "ri_p": ri_p,
                 "n_fq": n_fq, "n_firms": nfirm, "n_quarters": nq})

pd.DataFrame(rows).to_csv(OUT / "ri_fourgroup_results.csv", index=False)

print(f"\n(N_PERM={N_PERM}, iters={DEMEAN_ITERS}, seed={SEED})")
print(f"{'sample':10s} {'b3_obs':>14s} {'ri_p_2side':>12s} {'n_fq':>10s} {'firms':>8s} {'quarters':>9s}")
for r in rows:
    print(f"{r['sample']:10s} {r['b_obs']:14.6e} {r['ri_p']:12.4f} "
          f"{r['n_fq']:10,d} {r['n_firms']:8,d} {r['n_quarters']:9d}")
# hardcoded Stata anchors for drift protection (run_fourgroup.do MAIN t_cn_s,
# reghdfe absorb(fq gq ig); 2026-08-02): full +2.789446e-06, post2018 +1.159757e-06
print("\nexpected (Stata MAIN)  = full +2.789446e-06, post2018 +1.159757e-06 "
      "— investigate if far off")
print("cross-check each b3_obs against run_fourgroup.do MAIN t_cn_s "
      "(reghdfe absorb(fq gq ig)): full column m_act_full, post column m_act_post. "
      "A material mismatch means a stale panel or broken collapse.")
print("post-2018 is the PRIMARY report (predetermined labels); full is disclosed "
      "with the ~2018-08 snapshot look-ahead caveat.")
print(f"wrote {(OUT / 'ri_fourgroup_results.csv').as_posix()}")
