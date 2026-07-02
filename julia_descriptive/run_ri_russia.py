"""
run_ri_russia.py — randomization inference for the Russia positive control.

Same exact algebra as run_randomization_inference.py (US-NONUS pairwise
difference on the firm-quarter panel collapses the it+gt FE exactly; per-quarter
sufficient statistics let a permutation of the 82 quarter shocks be evaluated as
an O(1) weighted sum, so 200k permutations are instant). Only the input columns
change: ru_lag instead of cn_lag, on c6_panel_russia.parquet.

Also runs a SEPARATE placebo test for the event-window (2022 Q1-Q2 invasion)
dummy spec: since a specific calendar event is not a random draw, the natural
placebo is "how extreme is the actual invasion-window coefficient relative to
EVERY OTHER possible 2-consecutive-quarter window in the sample" (an event-study-
style permutation, not a shock-relabeling permutation).
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
       any_value(ru_lag) AS cn, any_value(shock) AS s,
       MAX(CASE WHEN us=1 THEN dw END) - MAX(CASE WHEN us=0 THEN dw END) AS d_dw
FROM read_parquet('{(OUT/'c6_panel_russia.parquet').as_posix()}')
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

print("===== Russia positive control: continuous-shock RI =====")
b3, p, nq, nfq = beta3_and_ri(d, "d_dw", seed=SEED)
print(f"  b3 = {b3:.4e}   RI p (2-sided) = {p:.4f}   n_quarters={nq}   n_firmquarters={nfq:,}")
print(f"  (Stata CRVE 3-pairwise reference: b3=-7.327e-06, p=0.085)")

pd.DataFrame([{"spec": "russia headline continuous-shock", "b3": b3, "ri_p_2sided": p,
               "n_quarters": nq, "n_firmquarters": nfq}]).to_csv(
    OUT / "russia_ri_results.csv", index=False)

print("\n===== Event-window placebo: is 2022Q1-Q2 extreme vs EVERY other 2-quarter window? =====")
dd = d.dropna(subset=["d_dw", "cn"]).copy()
dd["cn_c"] = dd["cn"] - dd.groupby("rdate")["cn"].transform("mean")
quarters = sorted(dd["rdate"].unique())
qidx = {q: i for i, q in enumerate(quarters)}
dd["qi"] = dd["rdate"].map(qidx)

# per-quarter A_t, C_t are NOT what we need here (post_invasion is a window dummy,
# not a shock level) -- build per-quarter sums of cn_c and cn_c*dw once, then any
# 2-quarter window's beta(us_ru_post) coefficient can be built from the SAME
# quarter-FE-collapsed regression: post-dummy interacted with (demeaned) cn.
# Sufficient stats per quarter: A_t = sum(cn_c^2) [same as above], and we need
# the coefficient of a REGRESSOR that is cn_c * 1{quarter in window}. Since the
# window regressor is a 0/1 mask over quarters, its own within-quarter demean is
# itself when included alongside cn_c (both already summed to zero within qtr for
# cn_c). We fit the two-regressor OLS (cn_c, cn_c*window) exactly via the closed
# form used above with S replaced by the window indicator vector.
nq_tot = len(quarters)
window_size = 2  # matches the 2022 Q1-Q2 dummy
g = dd.groupby("qi")
A = g.apply(lambda x: np.sum(x["cn_c"].values ** 2)).values.astype(float)
C = g.apply(lambda x: np.sum(x["cn_c"].values * x["d_dw"].values)).values.astype(float)
assert len(A) == nq_tot

def solve_b3_window(window_vec):
    sA = np.sum(A); sSA = np.sum(window_vec * A); sS2A = np.sum(window_vec * window_vec * A)
    sC = np.sum(C); sSC = np.sum(window_vec * C)
    det = sA * sS2A - sSA * sSA
    if abs(det) < 1e-300:
        return np.nan
    return (sA * sSC - sSA * sC) / det

# actual 2022Q1-Q2 window
inv_start = qidx[pd.Timestamp("2022-03-31")] if pd.Timestamp("2022-03-31") in qidx else None
# find quarters in [2022-01, 2022-06]
inv_mask = np.array([1.0 if pd.Timestamp("2022-01-01") <= q <= pd.Timestamp("2022-06-30") else 0.0 for q in quarters])
b3_actual = solve_b3_window(inv_mask)
print(f"  actual 2022Q1-Q2 window: b3 = {b3_actual:.4e}  (n_quarters_in_window={int(inv_mask.sum())})")

all_windows = []
for start in range(nq_tot - window_size + 1):
    wv = np.zeros(nq_tot); wv[start:start + window_size] = 1.0
    all_windows.append(solve_b3_window(wv))
all_windows = np.array(all_windows)
rank = np.mean(np.abs(all_windows) >= abs(b3_actual) - 1e-300)
print(f"  placebo: {len(all_windows)} possible {window_size}-quarter windows; "
      f"share with |b3| >= |actual invasion b3|: {rank:.4f}")
print(f"  (this is the fraction of ALL rolling 2-quarter windows in the sample "
      f"at least as extreme as the actual 2022 invasion window)")

pd.DataFrame({"start_quarter": quarters[:len(all_windows)], "b3_window": all_windows}).to_csv(
    OUT / "russia_event_window_placebo.csv", index=False)
print("\nwrote russia_ri_results.csv, russia_event_window_placebo.csv")
