"""
run_ri_russia_3pairwise.py — two-way-FE (firm+quarter) randomization inference
for the Russia 3-pairwise headline (it+gt+ig), matching run_ri_3pairwise.py's
algebra exactly but on c6_panel_russia.parquet with ru_lag instead of cn_lag.
"""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")
N_PERM = 5000
DEMEAN_ITERS = 30
SEED = 20260702

con = duckdb.connect()
# R3-F6 freshness: regenerate the parquet from the CURRENT .dta so this script can
# never silently run on a stale panel left over from an earlier build.
import pyreadstat
_df, _ = pyreadstat.read_dta((OUT / "c6_panel_russia.dta").as_posix())
_df.to_parquet(OUT / "c6_panel_russia.parquet", index=False)
d = con.execute(f"""
SELECT firm_str, rdate,
       any_value(ru_lag) AS cn, any_value(shock) AS s,
       MAX(CASE WHEN us=1 THEN dw END) - MAX(CASE WHEN us=0 THEN dw END) AS d_dw
FROM read_parquet('{(OUT/'c6_panel_russia.parquet').as_posix()}')
GROUP BY firm_str, rdate
""").df()
con.close()

d["firm_c"] = pd.factorize(d["firm_str"])[0]
d["q_c"] = pd.factorize(d["rdate"])[0]

def make_demeaner(firm_c, q_c, iters):
    nf = firm_c.max() + 1
    nq = q_c.max() + 1
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

sub = d.dropna(subset=["d_dw", "cn", "s"]).reset_index(drop=True)
firm_c = pd.factorize(sub["firm_str"])[0]
q_c = pd.factorize(sub["rdate"])[0]
demean = make_demeaner(firm_c, q_c, DEMEAN_ITERS)
y = demean(sub["d_dw"].to_numpy(float))
cn = demean(sub["cn"].to_numpy(float))
cn_raw = sub["cn"].to_numpy(float)
qcode = q_c
nq = qcode.max() + 1
Svec = np.zeros(nq)
Svec[qcode] = sub["s"].to_numpy(float)
cn_dot_c = float(cn @ cn)

def beta3(Sq):
    x = cn_raw * Sq[qcode]
    xt = demean(x)
    b = (cn @ xt) / cn_dot_c
    xr = xt - b * cn
    denom = xr @ xr
    return (y @ xr) / denom if denom > 0 else np.nan

b_obs = beta3(Svec)
rng = np.random.default_rng(SEED)
cnt = 0
for _ in range(N_PERM):
    Sp = rng.permutation(Svec)
    bp = beta3(Sp)
    if abs(bp) >= abs(b_obs) - 1e-300:
        cnt += 1
p = (cnt + 1) / (N_PERM + 1)
print(f"Russia 3-pairwise RI: b3 = {b_obs:.4e}  RI p = {p:.4f}  n_fq={sub.shape[0]:,}  (N_PERM={N_PERM})")
print(f"  (Stata CRVE 3-pairwise reference: b3=-4.2812e-06, p=0.0358)")
pd.DataFrame([{"spec": "russia 3-pairwise", "b3": b_obs, "ri_p_2sided": p, "n_firmquarters": sub.shape[0]}]).to_csv(
    OUT / "russia_ri_3pairwise.csv", index=False)
