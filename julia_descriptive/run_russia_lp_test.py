"""
run_russia_lp_test.py — cumulative event-study test for the Russia positive
control: for each horizon h (2022Q1..2023Q4), collapse to the US-NONUS
cross-firm difference in cumulative Delta-w since 2021Q4, and test whether it
is more negative for higher pre-invasion Russia exposure (ru_lag, as of
2021Q4). Single cross-section per horizon (no within-horizon time variation),
so plain heteroskedasticity-robust OLS is the natural estimator; inference is
corroborated with a firm-level permutation test (shuffle ru_lag across firms,
sharp null of no relationship) since the ru_lag distribution is extremely
right-skewed (94% zeros) and a normal-theory SE could be unreliable.
"""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")
N_PERM = 20000
SEED = 20260702

con = duckdb.connect()
d = con.execute(f"""
SELECT sec_entity_id, report_date,
       any_value(ru_lag) AS ru_lag,
       MAX(CASE WHEN us=1 THEN cum_dw END) - MAX(CASE WHEN us=0 THEN cum_dw END) AS d_cum
FROM read_parquet('{(OUT / "russia_lp_panel.parquet").as_posix()}')
GROUP BY sec_entity_id, report_date
""").df()
con.close()

def ols_robust(x, y):
    X = np.column_stack([np.ones_like(x), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    # HC1 robust covariance
    meat = (X * resid[:, None]).T @ (X * resid[:, None])
    n, k = X.shape
    vcov = XtX_inv @ meat @ XtX_inv * (n / (n - k))
    se = np.sqrt(np.diag(vcov))
    return beta[1], se[1]

rng = np.random.default_rng(SEED)
rows = []
for h, (q, sub) in enumerate(d.groupby("report_date")):
    sub = sub.dropna(subset=["d_cum", "ru_lag"])
    x = sub["ru_lag"].to_numpy(float)
    y = sub["d_cum"].to_numpy(float)
    b, se = ols_robust(x, y)
    t = b / se
    # permutation: shuffle ru_lag across firms within this horizon
    cnt = 0
    for _ in range(N_PERM):
        xp = rng.permutation(x)
        bp, _ = ols_robust(xp, y)
        if abs(bp) >= abs(b) - 1e-300:
            cnt += 1
    perm_p = (cnt + 1) / (N_PERM + 1)
    rows.append({"quarter": q, "h": h, "beta": b, "se_hc1": se, "t": t,
                 "n": len(sub), "perm_p": perm_p})
    print(f"h={h}  {pd.Timestamp(q).date()}  beta={b:.4e}  se={se:.4e}  "
          f"t={t:.2f}  perm_p={perm_p:.4f}  n={len(sub):,}")

pd.DataFrame(rows).to_csv(OUT / "russia_lp_test_results.csv", index=False)
print("\nwrote russia_lp_test_results.csv")
