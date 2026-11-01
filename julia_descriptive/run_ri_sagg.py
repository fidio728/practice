"""
run_ri_sagg.py — internal permutation check for the AGGREGATED-shock h=0 spec
(negative beta_3 across specs; on the B7 2026-08-03 rebuild it is a clean null:
CRVE p=0.348 two-way / 0.411 three-pairwise, RI free-perm p=0.512 — the pre-B7
near-marginal CRVE p=0.086 did not survive the rebuild). Same exact pairwise-collapse algebra as
run_randomization_inference.py (independently verified in review), with s_agg
in place of the stamped shock. Permutes the 82 quarterly s_agg values.

CAVEAT reported alongside: corr(s_agg_t, s_agg_{t-1}) = 0.357, so free
permutation is approximate (the innovations are not exactly white within the
panel window); a block/circular-shift variant is the stricter version.
Both free-permutation and circular-shift p-values are reported.
"""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pyreadstat

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")
N_PERM = 200000
SEED = 20260702

df, _ = pyreadstat.read_dta((OUT / "sagg_panel.dta").as_posix())
df.to_parquet(OUT / "sagg_panel.parquet", index=False)

con = duckdb.connect()
d = con.execute(f"""
SELECT firm_str, rdate,
       any_value(cn_lag) AS cn, any_value(s_agg) AS s,
       MAX(CASE WHEN us=1 THEN dw END) - MAX(CASE WHEN us=0 THEN dw END) AS d_dw
FROM read_parquet('{(OUT/'sagg_panel.parquet').as_posix()}')
GROUP BY firm_str, rdate
""").df()
con.close()

dd = d.dropna(subset=["d_dw", "cn", "s"]).copy()
dd["cn_c"] = dd["cn"] - dd.groupby("rdate")["cn"].transform("mean")
g = dd.groupby("rdate")
A = g.apply(lambda x: np.sum(x["cn_c"].values ** 2)).values.astype(float)
C = g.apply(lambda x: np.sum(x["cn_c"].values * x["d_dw"].values)).values.astype(float)
S = g["s"].first().values.astype(float)
nq = len(S)

def solve_b3(Svec):
    sA = np.sum(A); sSA = np.sum(Svec * A); sS2A = np.sum(Svec * Svec * A)
    sC = np.sum(C); sSC = np.sum(Svec * C)
    det = sA * sS2A - sSA * sSA
    return np.nan if abs(det) < 1e-300 else (sA * sSC - sSA * sC) / det

b_obs = solve_b3(S)
print(f"observed b3 (it+gt collapse) = {b_obs:.4e}   (Stata 2-way CRVE: -1.06e-06, p=0.3478; B7 2026-08-03)")

# --- free permutation ---
rng = np.random.default_rng(SEED)
cnt = 0
done = 0
while done < N_PERM:
    b = min(20000, N_PERM - done)
    perms = np.array([rng.permutation(S) for _ in range(b)])
    sA = np.sum(A); sSA = perms @ A; sS2A = (perms * perms) @ A
    sC = np.sum(C); sSC = perms @ C
    det = sA * sS2A - sSA * sSA
    b3p = np.where(np.abs(det) < 1e-300, np.nan, (sA * sSC - sSA * sC) / det)
    cnt += np.nansum(np.abs(b3p) >= abs(b_obs) - 1e-300)
    done += b
p_free = (cnt + 1) / (N_PERM + 1)
print(f"free-permutation p (2-sided, {N_PERM:,}) = {p_free:.4f}")

# --- circular shift (preserves the serial structure of S; 81 non-trivial shifts) ---
shift_b3 = np.array([solve_b3(np.roll(S, k)) for k in range(1, nq)])
p_shift = (np.sum(np.abs(shift_b3) >= abs(b_obs) - 1e-300) + 1) / (len(shift_b3) + 1)
print(f"circular-shift p (2-sided, {len(shift_b3)} shifts) = {p_shift:.4f}")

pd.DataFrame([{"spec": "s_agg h=0 (it+gt collapse)", "b3": b_obs,
               "p_free_perm": p_free, "p_circular_shift": p_shift,
               "n_quarters": nq, "n_firmquarters": len(dd)}]).to_csv(
    OUT / "sagg_ri_check.csv", index=False)
print("wrote sagg_ri_check.csv")
