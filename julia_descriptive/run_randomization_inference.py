"""
run_randomization_inference.py — design-based inference for the triple-difference
β₃, robust to the few-quarter-cluster + overlapping-window problems that make the
CRVE unreliable (review findings F1b, F3, F9).

Exact algebra. The panel is balanced 2-per-(firm, quarter) {US, NONUS}. The De Haas
spec  y_g = β₂·(us_g·cn) + β₃·(us_g·cn·S) + α_{firm×qtr} + γ_{group×qtr}
collapses, on the US−NONUS within-firm-quarter difference Δy, to

    Δy_it = β₂·cn_it + β₃·(cn_it·S_t) + δ_t + Δε_it        (δ_t = quarter FE)

which reproduces the two-way-FE β₂, β₃ EXACTLY. After within-quarter demeaning of cn
(→ c̃n; note Σ_i c̃n_it = 0), and because S_t is constant within a quarter,

    β₃ = solve the 2×2 OLS of Δỹ on [c̃n, S·c̃n],

whose sufficient statistics are, per quarter t,  A_t = Σ_i c̃n²,  C_t = Σ_i c̃n·Δỹ.
For ANY assignment of the 82 quarter shocks S_t:
    X'X = [[ΣA,      Σ S_t A_t],
           [Σ S_t A_t, Σ S_t² A_t]],   X'y = [ΣC,  Σ S_t C_t].
So a randomization test that PERMUTES the 82 shocks across quarters is just weighted
sums over 82 numbers → millions of permutations are instant.

RI p-value (two-sided) = share of permutations with |β₃_perm| ≥ |β₃_obs|.
Reported for the headline (h=0), the lead-flow (Δw_{t+1}), and the LP horizons h=1..4,
on the full grid and the in-span subset.
"""

from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
OUT = PROJ / "output"
DTA = (OUT / "audit_c6_panel.dta").as_posix()
N_PERM = 200000
SEED_STREAM = 20260702  # deterministic; project-convention seed (2026-07-02)

con = duckdb.connect()

def load_diffs(where=""):
    # collapse to firm-quarter US−NONUS differences for each outcome
    q = f"""
    WITH p AS (SELECT * FROM read_parquet('{OUT.as_posix()}/audit_c6_panel.parquet') {where})
    SELECT firm_str, rdate,
           any_value(cn_lag) AS cn, any_value(shock) AS s,
           MAX(CASE WHEN us=1 THEN dw       END) - MAX(CASE WHEN us=0 THEN dw       END) AS d_dw,
           MAX(CASE WHEN us=1 THEN dw_lead1 END) - MAX(CASE WHEN us=0 THEN dw_lead1 END) AS d_lead1,
           MAX(CASE WHEN us=1 THEN cum0 END) - MAX(CASE WHEN us=0 THEN cum0 END) AS d_c0,
           MAX(CASE WHEN us=1 THEN cum1 END) - MAX(CASE WHEN us=0 THEN cum1 END) AS d_c1,
           MAX(CASE WHEN us=1 THEN cum2 END) - MAX(CASE WHEN us=0 THEN cum2 END) AS d_c2,
           MAX(CASE WHEN us=1 THEN cum3 END) - MAX(CASE WHEN us=0 THEN cum3 END) AS d_c3,
           MAX(CASE WHEN us=1 THEN cum4 END) - MAX(CASE WHEN us=0 THEN cum4 END) AS d_c4
    FROM p GROUP BY firm_str, rdate
    """
    return con.execute(q).df()

def beta3_and_ri(df, ycol, n_perm=N_PERM, seed=0):
    d = df.dropna(subset=[ycol, "cn", "s"]).copy()
    # within-quarter demean of cn
    d["cn_c"] = d["cn"] - d.groupby("rdate")["cn"].transform("mean")
    # per-quarter sufficient stats
    g = d.groupby("rdate")
    A = g.apply(lambda x: np.sum(x["cn_c"].values**2)).values.astype(float)          # A_t
    C = g.apply(lambda x: np.sum(x["cn_c"].values * x[ycol].values)).values.astype(float)  # C_t
    S = g["s"].first().values.astype(float)
    def solve_b3(Svec):
        sA = np.sum(A); sSA = np.sum(Svec*A); sS2A = np.sum(Svec*Svec*A)
        sC = np.sum(C); sSC = np.sum(Svec*C)
        # X'X = [[sA, sSA],[sSA, sS2A]] ; X'y = [sC, sSC] ; want second coef
        det = sA*sS2A - sSA*sSA
        if abs(det) < 1e-300: return np.nan
        b3 = (sA*sSC - sSA*sC)/det
        return b3
    b3_obs = solve_b3(S)
    rng = np.random.default_rng(seed)
    nq = len(S)
    cnt = 0
    # vectorize permutations in blocks
    block = 20000
    done = 0
    while done < n_perm:
        b = min(block, n_perm - done)
        # generate b permutations of S
        perms = np.array([rng.permutation(S) for _ in range(b)])  # b x nq
        sA = np.sum(A);
        sSA = perms @ A
        sS2A = (perms*perms) @ A
        sC = np.sum(C)
        sSC = perms @ C
        det = sA*sS2A - sSA*sSA
        b3p = np.where(np.abs(det) < 1e-300, np.nan, (sA*sSC - sSA*sC)/det)
        cnt += np.sum(np.abs(b3p) >= abs(b3_obs) - 1e-300)
        done += b
    ri_p = (cnt + 1) / (n_perm + 1)
    return b3_obs, ri_p, len(S), len(d)

# write a parquet copy (duckdb reads dta poorly; use pandas->parquet once)
import pyreadstat  # noqa
_df, _ = pyreadstat.read_dta(DTA)
_df.to_parquet(OUT / "audit_c6_panel.parquet", index=False)

full = load_diffs("")
span = load_diffs("WHERE in_span=1")

print(f"{'spec':22s} {'b3':>12s} {'RI_p(2side)':>12s} {'nq':>4s} {'n_fq':>9s}")
rows = []
for label, df, y in [
    ("headline dw (h0)",       full, "d_dw"),
    ("lead-flow dw_{t+1}",     full, "d_lead1"),
    ("LP cum h=1",             full, "d_c1"),
    ("LP cum h=2",             full, "d_c2"),
    ("LP cum h=3",             full, "d_c3"),
    ("LP cum h=4",             full, "d_c4"),
    ("headline dw IN-SPAN",    span, "d_dw"),
    ("LP cum h=1 IN-SPAN",     span, "d_c1"),
    ("LP cum h=4 IN-SPAN",     span, "d_c4"),
]:
    b3, p, nq, nfq = beta3_and_ri(df, y, seed=SEED_STREAM)
    print(f"{label:22s} {b3:12.3e} {p:12.4f} {nq:4d} {nfq:9,d}")
    rows.append({"spec": label, "b3": b3, "ri_p_2sided": p, "n_quarters": nq, "n_firmquarters": nfq})

pd.DataFrame(rows).to_csv(OUT / "audit_randomization_inference.csv", index=False)
print(f"\nwrote audit_randomization_inference.csv  ({N_PERM:,} permutations each)")
