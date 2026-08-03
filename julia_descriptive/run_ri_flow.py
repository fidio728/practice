"""
run_ri_flow.py — winsorize + randomization inference for the ownership FLOW
outcome on the B7-rebuilt ownership_c6_panel.dta.

Why: after the B7 rebuild the two-way-cluster CRVE is DEGENERATE on the primary
fq-gq flow spec (SE reported as missing) and gives p=0.0001 on the 3-pairwise
spec, but `flow` has pathological fat tails (kurtosis ~5000, max +983% of float
from tiny lagged denominators, un-winsorized). This script settles whether that
CRVE significance is a fat-tail / few-cluster artifact by (a) winsorizing flow
at p1/p99 and (b) running design-based randomization inference (permute the 82
quarter shocks), which is CRVE-independent and immune to fat tails and few
clusters. RI machinery mirrors run_ri_3pairwise.py exactly.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")
N_PERM = 5000
DEMEAN_ITERS = 30
SEED = 20260702

df = pd.read_stata(OUT / "ownership_c6_panel.dta")
for c in ["flow", "cn_lag", "shock", "us"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# winsorize flow at p1/p99 (two-sided) as a fat-tail robustness; dw is never
# winsorized, only flow is
lo, hi = df["flow"].quantile([0.01, 0.99])
df["flow_w"] = df["flow"].clip(lo, hi)
print(f"flow winsor bounds: p1={lo:.4e}  p99={hi:.4e}")
print(f"flow raw:  mean {df['flow'].mean():.3e}  sd {df['flow'].std():.3e}  "
      f"kurt {df['flow'].kurt():.0f}  max {df['flow'].max():.3f}")
print(f"flow wins: mean {df['flow_w'].mean():.3e}  sd {df['flow_w'].std():.3e}  "
      f"kurt {df['flow_w'].kurt():.1f}  max {df['flow_w'].max():.4f}")

# collapse to within-(firm,quarter) US - NONUS difference
def collapse(ycol):
    piv = df.pivot_table(index=["firm_str", "rdate"], columns="us", values=ycol, aggfunc="first")
    piv = piv.dropna(subset=[0.0, 1.0])
    meta = (df.groupby(["firm_str", "rdate"])
              .agg(cn=("cn_lag", "first"), s=("shock", "first")).reset_index())
    out = piv.reset_index().rename(columns={1.0: "y_us", 0.0: "y_nonus"})
    out["dy"] = out["y_us"] - out["y_nonus"]
    out = out.merge(meta, on=["firm_str", "rdate"])
    return out.dropna(subset=["dy", "cn", "s"]).reset_index(drop=True)

def make_demeaner(firm_c, q_c, iters):
    nf, nq = firm_c.max() + 1, q_c.max() + 1
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

def ri(sub, n_perm=N_PERM, seed=SEED):
    firm_c = pd.factorize(sub["firm_str"])[0]
    q_c = pd.factorize(sub["rdate"])[0]
    demean = make_demeaner(firm_c, q_c, DEMEAN_ITERS)
    y = demean(sub["dy"].to_numpy(float))
    cn = demean(sub["cn"].to_numpy(float))
    cn_raw = sub["cn"].to_numpy(float)
    nq = q_c.max() + 1
    Svec = np.zeros(nq)
    Svec[q_c] = sub["s"].to_numpy(float)
    cn_dot_c = float(cn @ cn)
    def beta3(Sq):
        x = demean(cn_raw * Sq[q_c])
        b = (cn @ x) / cn_dot_c
        xr = x - b * cn
        denom = xr @ xr
        return (y @ xr) / denom if denom > 0 else np.nan
    b_obs = beta3(Svec)
    rng = np.random.default_rng(seed)
    cnt = sum(abs(beta3(rng.permutation(Svec))) >= abs(b_obs) - 1e-300 for _ in range(n_perm))
    return b_obs, (cnt + 1) / (n_perm + 1), len(sub)

print(f"\n{'outcome':14s} {'b3':>12s} {'RI_p_2side':>12s} {'n_fq':>9s}  (N_PERM={N_PERM})")
rows = []
for lbl, col in [("flow (raw)", "flow"), ("flow (winsor)", "flow_w")]:
    sub = collapse(col)
    b, p, n = ri(sub)
    print(f"{lbl:14s} {b:12.3e} {p:12.4f} {n:9,d}")
    rows.append({"outcome": lbl, "b3": b, "ri_p_2sided": p, "n_firmquarters": n})
pd.DataFrame(rows).to_csv(OUT / "ri_flow_results.csv", index=False)
print("\nwrote ri_flow_results.csv")
