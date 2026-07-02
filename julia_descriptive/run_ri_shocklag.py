"""
run_ri_shocklag.py — randomization inference for the advisor's S_(t-1) spec
(it+gt), same exact pairwise-difference-collapse algebra used throughout this
project (proven to reproduce reghdfe's beta3 exactly). Checks whether the
sign flip (S_t: +1.28e-6 -> S_(t-1): -8.65e-7) is a real pattern or noise.
"""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")
N_PERM = 200000
SEED = 20260702

con = duckdb.connect()
d = con.execute(f"""
SELECT firm_str, rdate,
       any_value(cn_lag) AS cn, any_value(shock_tm1) AS s,
       MAX(CASE WHEN us=1 THEN dw END) - MAX(CASE WHEN us=0 THEN dw END) AS d_dw
FROM read_parquet('{(OUT / "shocklag_panel.parquet").as_posix()}')
GROUP BY firm_str, rdate
""").df()
con.close()

def beta3_and_ri(df, ycol, n_perm=N_PERM, seed=0):
    dd = df.dropna(subset=[ycol, "cn", "s"]).copy()
    dd["cn_c"] = dd["cn"] - dd.groupby("rdate")["cn"].transform("mean")
    g = dd.groupby("rdate")
    A = g.apply(lambda x: np.sum(x["cn_c"].values ** 2)).values.astype(float)
    C = g.apply(lambda x: np.sum(x["cn_c"].values * x[ycol].values)).values.astype(float)
    S = g["s"].first().values.astype(float)

    def solve_b3(Svec):
        sA = np.sum(A); sSA = np.sum(Svec * A); sS2A = np.sum(Svec * Svec * A)
        sC = np.sum(C); sSC = np.sum(Svec * C)
        det = sA * sS2A - sSA * sSA
        if abs(det) < 1e-300:
            return np.nan
        return (sA * sSC - sSA * sC) / det

    b3_obs = solve_b3(S)
    rng = np.random.default_rng(seed)
    cnt = 0
    done = 0
    block = 20000
    while done < n_perm:
        b = min(block, n_perm - done)
        perms = np.array([rng.permutation(S) for _ in range(b)])
        sA = np.sum(A)
        sSA = perms @ A
        sS2A = (perms * perms) @ A
        sC = np.sum(C)
        sSC = perms @ C
        det = sA * sS2A - sSA * sSA
        b3p = np.where(np.abs(det) < 1e-300, np.nan, (sA * sSC - sSA * sC) / det)
        cnt += np.nansum(np.abs(b3p) >= abs(b3_obs) - 1e-300)
        done += b
    ri_p = (cnt + 1) / (n_perm + 1)
    return b3_obs, ri_p, len(S), len(dd)

b3, p, nq, nfq = beta3_and_ri(d, "d_dw", seed=SEED)
print(f"S_(t-1) spec, it+gt:  b3 = {b3:.4e}   RI p (2-sided) = {p:.4f}   n_quarters={nq}   n_firmquarters={nfq:,}")
print(f"  (Stata CRVE reference: b3=-8.65e-07, p=0.679)")
pd.DataFrame([{"spec": "S_(t-1) it+gt", "b3": b3, "ri_p_2sided": p,
               "n_quarters": nq, "n_firmquarters": nfq}]).to_csv(
    OUT / "shocklag_ri_results.csv", index=False)
