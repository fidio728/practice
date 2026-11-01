"""
verify_ladder_L3_fwl.py — INDEPENDENT re-derivation (outside Stata) of the
NEW-PRIMARY ladder cell L3 (fund grain, GLOBAL denom, S_{t-1}) on the v3
canonical panel, to < 1e-6 relative. Machinery is verbatim
verify_attribution_em.py (two-way demean to 1e-14 + FWL on the US−NONUS
difference panel); anchors are read from the LIVING canonical CSV at runtime,
never hardcoded. Also re-derives the L3 itgt cell via the exact quarter-FE
collapse (fq gq -> quarter FE only on the difference panel).

R3-F6 freshness: audit_c6_panel.parquet is re-transcoded from the CURRENT
audit_c6_panel.dta on every run.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import pyreadstat

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")
PQ = (OUT / "audit_c6_panel.parquet").as_posix()

# ---- living anchors (never hardcoded) ---------------------------------------
_can_path = OUT / "headline_3pairwise_canonical.csv"
can = pd.read_csv(_can_path)
assert {"spec", "denom", "timing", "fe", "b3"} <= set(can.columns), (
    f"{_can_path.name}: stale layout — re-run run_headline_3pairwise.do")

def _cell(spec, fe):
    r = can[(can["spec"] == spec) & (can["fe"] == fe)]
    assert len(r) == 1, f"{_can_path.name}: no unique {spec}/{fe} row"
    return float(r["b3"].iloc[0])

B3_L3_3PW = _cell("primary", "fq_gq_ig")
B3_L3_ITGT = _cell("primary_itgt", "fq_gq")
print(f"living anchors: L3 3pw b3={B3_L3_3PW:+.6e}   L3 itgt b3={B3_L3_ITGT:+.6e}")

# ---- freshness: re-transcode parquet from the CURRENT .dta ------------------
_df, _ = pyreadstat.read_dta((OUT / "audit_c6_panel.dta").as_posix())
_df.to_parquet(OUT / "audit_c6_panel.parquet", index=False)
del _df

con = duckdb.connect()
d = con.execute(f"""
SELECT firm_str, rdate,
       any_value(cn_lag) AS cn,
       any_value(s_lag)  AS s,
       MAX(CASE WHEN us=1 THEN dw END) - MAX(CASE WHEN us=0 THEN dw END) AS d_dw,
       COUNT(*) AS nrow
FROM read_parquet('{PQ}')
GROUP BY firm_str, rdate
""").df()
con.close()
assert (d["nrow"] == 2).all(), "panel is not exactly 2 rows per (firm, quarter)"
d = d.dropna(subset=["d_dw", "cn", "s"]).reset_index(drop=True)
print(f"differenced panel: {len(d):,} firm-quarters, {d['firm_str'].nunique():,} firms")


def demean2(cols, firm_c, q_c, tol=1e-14, maxit=2000):
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


firm_c = pd.factorize(d["firm_str"])[0]
q_c = pd.factorize(d["rdate"])[0]
x_raw = d["cn"].to_numpy(float) * d["s"].to_numpy(float)

# ---- L3 3pw: firm + quarter FE on the difference panel (exact collapse) -----
(y, cn, x), iters = demean2(
    [d["d_dw"].to_numpy(float), d["cn"].to_numpy(float), x_raw], firm_c, q_c)
xr = x - (cn @ x) / (cn @ cn) * cn
b3_3pw = (y @ xr) / (xr @ xr)
rel_3pw = abs(b3_3pw - B3_L3_3PW) / abs(B3_L3_3PW)
print(f"L3 3pw  FWL b3={b3_3pw:+.9e}  stata={B3_L3_3PW:+.6e}  rel={rel_3pw:.2e}  "
      f"demean_iters={iters}")

# ---- L3 itgt: quarter FE only on the difference panel (exact collapse) ------
nq = q_c.max() + 1
qc = np.bincount(q_c, minlength=nq).astype(float)


def qdm(v):
    return v - (np.bincount(q_c, weights=v, minlength=nq) / qc)[q_c]


yq = qdm(d["d_dw"].to_numpy(float))
cq = qdm(d["cn"].to_numpy(float))
xq = qdm(x_raw)
xrq = xq - (cq @ xq) / (cq @ cq) * cq
b3_itgt = (yq @ xrq) / (xrq @ xrq)
rel_itgt = abs(b3_itgt - B3_L3_ITGT) / abs(B3_L3_ITGT)
print(f"L3 itgt FWL b3={b3_itgt:+.9e}  stata={B3_L3_ITGT:+.6e}  rel={rel_itgt:.2e}")

assert rel_3pw < 1e-6, f"L3 3pw FWL does not reproduce Stata (rel {rel_3pw:.2e})"
assert rel_itgt < 1e-6, f"L3 itgt FWL does not reproduce Stata (rel {rel_itgt:.2e})"
print("\nL3 reproduced independently outside Stata: 3pw and itgt both < 1e-6 rel. PASS")
