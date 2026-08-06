"""
verify_attribution_em.py — INDEPENDENT re-derivation of the three attribution
coefficients, outside Stata, on the differenced panel.

On the balanced 2-rows-per-(firm, quarter) grid the three pairwise FE
(fq firm x quarter, gq group x quarter, ig firm x group) collapse, on the
US - NONUS within-firm-quarter difference, to

    d_dw_it = b2 * cn_it + b3 * (cn_it * S_t) + phi_i + delta_t + e

(fq is annihilated by the difference itself; gq -> a pure quarter intercept;
ig -> a pure firm intercept). b3 is then FWL: two-way demean d_dw, cn and
cn*S over (firm, quarter); residualize the demeaned cn*S on the demeaned cn;
regress. This is the same identity run_ri_3pairwise.py relies on, and it
reproduced the pre-change Stata coefficient to 7 significant digits.

Nothing here reads a Stata result — the numbers are recomputed from
audit_c6_panel.parquet, and the preEM firm universe is read straight from the
archived c6_panel_preEM.dta.

Note on N: reghdfe drops singleton groups, this does not. Singletons carry no
residual variation, so the coefficient is unaffected; only the row count differs.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import pyreadstat

OUT = Path(r"e:/geoecon_output")
PQ = (OUT / "audit_c6_panel.parquet").as_posix()

con = duckdb.connect()
d = con.execute(f"""
SELECT firm_str, rdate,
       any_value(cn_lag) AS cn, any_value(shock) AS s,
       any_value(zr_lag) AS zr,
       MAX(CASE WHEN us=1 THEN dw END) - MAX(CASE WHEN us=0 THEN dw END) AS d_dw,
       COUNT(*) AS nrow
FROM read_parquet('{PQ}')
GROUP BY firm_str, rdate
""").df()
con.close()
assert (d["nrow"] == 2).all(), "panel is not exactly 2 rows per (firm, quarter)"
print(f"differenced panel: {len(d):,} firm-quarters, {d['firm_str'].nunique():,} firms")

# preEM (pre-change) firm universe, read from the archived panel
old, _ = pyreadstat.read_dta((OUT / "c6_panel_preEM.dta").as_posix(),
                             usecols=["firm_str"])
old_firms = set(old["firm_str"].astype(str).unique())
print(f"preEM firm universe: {len(old_firms):,} firms")
d["old_univ"] = d["firm_str"].astype(str).isin(old_firms)


def demean2(cols, firm_c, q_c, tol=1e-14, maxit=2000):
    """alternating projections onto firm and quarter means, to convergence"""
    nf, nq = firm_c.max() + 1, q_c.max() + 1
    fc = np.bincount(firm_c, minlength=nf).astype(float)
    qc = np.bincount(q_c, minlength=nq).astype(float)
    out, iters = [], []
    for x in cols:
        x = np.asarray(x, dtype=float).copy()
        it = 0
        for it in range(1, maxit + 1):
            fm = np.bincount(firm_c, weights=x, minlength=nf) / fc
            x -= fm[firm_c]
            qm = np.bincount(q_c, weights=x, minlength=nq) / qc
            x -= qm[q_c]
            if max(np.abs(fm).max(), np.abs(qm).max()) < tol:
                break
        out.append(x)
        iters.append(it)
    return out, iters


def b3(sub, label):
    sub = sub.dropna(subset=["d_dw", "cn", "s"]).reset_index(drop=True)
    firm_c = pd.factorize(sub["firm_str"])[0]
    q_c = pd.factorize(sub["rdate"])[0]
    x_raw = sub["cn"].to_numpy(float) * sub["s"].to_numpy(float)
    (y, cn, x), iters = demean2(
        [sub["d_dw"].to_numpy(float), sub["cn"].to_numpy(float), x_raw],
        firm_c, q_c)
    # FWL: partial cn out of x, then regress y on the residual
    xr = x - (cn @ x) / (cn @ cn) * cn
    b = (y @ xr) / (xr @ xr)
    # b2 for completeness: partial x out of cn
    cr = cn - (x @ cn) / (x @ x) * x
    b2 = (y @ cr) / (cr @ cr)
    print(f"{label:52s} b3={b: .6e}  b2={b2: .6e}  "
          f"fq={len(sub):>9,}  rows={2*len(sub):>10,}  firms={sub['firm_str'].nunique():>7,}  "
          f"demean_iters={iters}")
    return b


print()
print("INDEPENDENT b3 (differenced panel, firm+quarter FE, FWL)")
b_B = b3(d, "B  new snapshot + zero recode (CANONICAL)")
b_A = b3(d[d["zr"] == 0], "A  new snapshot + OLD missing rule (zr==0)")
b_A2 = b3(d[(d["zr"] == 0) & d["old_univ"]], "A2 run A INTERSECT preEM firm universe")

stata = {"B": -6.923784e-07, "A": -4.379509e-07, "A2": -4.394850e-07}
print()
print(f"{'run':4s} {'python':>15s} {'stata(reghdfe)':>18s} {'rel_diff':>12s}")
for k, v in [("B", b_B), ("A", b_A), ("A2", b_A2)]:
    rel = abs(v - stata[k]) / abs(stata[k])
    print(f"{k:4s} {v:15.6e} {stata[k]:18.6e} {rel:12.2e}")
    assert rel < 1e-4, f"run {k}: python and Stata disagree (rel {rel:.2e})"
print("\nall three coefficients reproduced independently (rel diff < 1e-4)")
