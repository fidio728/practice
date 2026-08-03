"""
run_ri_3pairwise.py — randomization inference for the 3-PAIRWISE headline
(it firm×quarter + gt group×quarter + ig firm×group).

On the balanced 2-per-(firm,quarter) panel the three pairwise FE collapse, on the
US−NONUS within-firm-quarter difference Δy, to

    Δy_it = β₂·cn_it + β₃·(cn_it·S_t) + φ_i (firm FE) + δ_t (quarter FE) + error

i.e. FIRM + QUARTER two-way FE on the difference panel (ig → the firm intercept φ_i;
gt → the quarter intercept δ_t; it absorbed by the pairwise difference itself).

RI (sharp null β₃=0): permute the 82 quarter shocks S_t. Unlike the quarter-FE-only
case there is no closed-form sufficient statistic (firm-demeaning of cn·S mixes S
across a firm's quarters), so we two-way-demean cn·S each permutation via fast
bincount alternating projections. β₃ via FWL against the (once-)demeaned Δy and cn.
Reported for the headline (h0) and the LP horizons that looked significant under CRVE.
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
_df, _ = pyreadstat.read_dta((OUT / "audit_c6_panel.dta").as_posix())
_df.to_parquet(OUT / "audit_c6_panel.parquet", index=False)
d = con.execute(f"""
SELECT firm_str, rdate,
       any_value(cn_lag) AS cn, any_value(shock) AS s,
       MAX(CASE WHEN us=1 THEN dw   END) - MAX(CASE WHEN us=0 THEN dw   END) AS d_dw,
       MAX(CASE WHEN us=1 THEN cum1 END) - MAX(CASE WHEN us=0 THEN cum1 END) AS d_c1,
       MAX(CASE WHEN us=1 THEN cum2 END) - MAX(CASE WHEN us=0 THEN cum2 END) AS d_c2,
       MAX(CASE WHEN us=1 THEN cum4 END) - MAX(CASE WHEN us=0 THEN cum4 END) AS d_c4
FROM read_parquet('{(OUT/'audit_c6_panel.parquet').as_posix()}')
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

def ri_twoway(df, ycol, n_perm=N_PERM, seed=SEED):
    sub = df.dropna(subset=[ycol, "cn", "s"]).reset_index(drop=True)
    firm_c = pd.factorize(sub["firm_str"])[0]
    q_c = pd.factorize(sub["rdate"])[0]
    demean = make_demeaner(firm_c, q_c, DEMEAN_ITERS)
    y = demean(sub[ycol].to_numpy(float))          # two-way demeaned outcome (once)
    cn = demean(sub["cn"].to_numpy(float))          # two-way demeaned cn (once)
    cn_raw = sub["cn"].to_numpy(float)
    # per-quarter shock lookup + row->quarter map
    qs = sub.groupby("q_c" if "q_c" in sub else q_c)  # noqa
    # map quarter code -> its shock
    qcode = q_c
    nq = qcode.max() + 1
    Svec = np.zeros(nq)
    Svec[qcode] = sub["s"].to_numpy(float)          # shock constant within quarter
    cn_dot_c = float(cn @ cn)                        # denom piece (fixed)
    def beta3(Sq):
        x = cn_raw * Sq[qcode]                       # cn * S (row level)
        xt = demean(x)                              # two-way demean
        # FWL: residualize xt on cn (already demeaned), then regress y on that
        b = (cn @ xt) / cn_dot_c
        xr = xt - b * cn
        denom = xr @ xr
        return (y @ xr) / denom if denom > 0 else np.nan
    b_obs = beta3(Svec)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        Sp = rng.permutation(Svec)                  # permute quarter shocks
        bp = beta3(Sp)
        if abs(bp) >= abs(b_obs) - 1e-300:
            cnt += 1
    return b_obs, (cnt + 1) / (n_perm + 1), sub.shape[0]

print(f"{'spec':16s} {'b3(chk vs Stata)':>18s} {'RI_p_2side':>12s} {'n_fq':>9s}   (N_PERM={N_PERM}, iters={DEMEAN_ITERS})")
print("  Stata 3-pairwise b3: headline +2.746e-6, cum1 +4.094e-6, cum4 +8.826e-6")
rows = []
for lbl, y in [("headline dw", "d_dw"), ("LP cum1", "d_c1"), ("LP cum4", "d_c4")]:
    b, p, n = ri_twoway(d, y)
    print(f"{lbl:16s} {b:12.3e} {p:12.4f} {n:9,d}")
    rows.append({"spec": lbl, "fe": "it+gt+ig (3-pairwise)", "b3": b, "ri_p_2sided": p, "n_firmquarters": n})
pd.DataFrame(rows).to_csv(OUT / "audit_ri_3pairwise.csv", index=False)
print(f"\nwrote audit_ri_3pairwise.csv")
