"""
verify_attribution_em.py — INDEPENDENT re-derivation of the three attribution
coefficients, outside Stata, on the differenced panel.

GLOBAL-MAIN + S_{t-1} PRIMARY (2026-08-08): dw in the audit panel is now the
FULL-portfolio (global-denominator) Δw and the primary triple uses the LAGGED
shock S_{t-1} (s_lag), so this script differences dw and builds cn * S_{t-1}.
That matches run_attribution_em.do's primary spec verbatim:
    reghdfe dw us_cn us_cn_slag, absorb(fq gq ig) vce(cluster firm_n rd_m)

On the balanced 2-rows-per-(firm, quarter) grid the three pairwise FE
(fq firm x quarter, gq group x quarter, ig firm x group) collapse, on the
US - NONUS within-firm-quarter difference, to

    d_dw_it = b2 * cn_it + b3 * (cn_it * S_{t-1}) + phi_i + delta_t + e

(fq is annihilated by the difference itself; gq -> a pure quarter intercept;
ig -> a pure firm intercept). b3 is then FWL: two-way demean d_dw, cn and
cn*S_{t-1} over (firm, quarter); residualize the demeaned cn*S_{t-1} on the
demeaned cn; regress. This is the same identity run_ri_3pairwise.py relies on,
and it reproduced the pre-change Stata coefficient to 7 significant digits.

The RE-DERIVATION never reads a Stata result — the coefficients are recomputed
from the panel data. The COMPARISON TARGETS, however, are read from the
freshly generated Stata artifacts, NEVER hardcoded (the retired premm
hardcoded dict was exactly the stale-vintage failure mode):
  - attribution_em_snapshot_vs_zerorecode.csv (run_attribution_em.do):
      rows run in {B, A, A2} at fe == "fq gq ig" -> the three b3 targets;
  - headline_3pairwise_canonical.csv (run_headline_3pairwise.do or the
      refresh block in run_attribution_em.do; LOCKED 2026-08-08 layout
      spec,denom,timing,fe,b3,se,p,N): the row spec=="primary" fe=="fq_gq_ig"
      must agree with the attribution run-B cell — same regression on the same
      panel — which gates VINTAGE COHERENCE between the two living artifacts.
Run this AFTER run_attribution_em.do in the chain.

R3-F6 freshness: audit_c6_panel.parquet is re-transcoded from the CURRENT
audit_c6_panel.dta on every run (same pattern as run_ri_3pairwise.py), so this
script can never silently run on a stale parquet vintage.

Note on N: reghdfe drops singleton groups, this does not. Singletons carry no
residual variation, so the coefficient is unaffected; only the row count differs.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import pyreadstat

PROJ = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive")
# (EM-FIX-5/7) honour the same DPN_OUT_DIR override as 00_setup.jl and the
# panel builders, so targets, panel and transcode all resolve to ONE vintage.
_env_out = os.environ.get("DPN_OUT_DIR", "").strip()
OUT = Path(_env_out).resolve() if _env_out else PROJ / "output"
if _env_out:
    print(f"[DPN_OUT_DIR] reading {OUT} (override in force)")
PQ = (OUT / "audit_c6_panel.parquet").as_posix()

# -----------------------------------------------------------------------------
# Comparison targets from the LIVING Stata artifacts (never hardcoded).
# -----------------------------------------------------------------------------
_att_path = OUT / "attribution_em_snapshot_vs_zerorecode.csv"
att = pd.read_csv(_att_path)
assert {"run", "fe", "b3"} <= set(att.columns), (
    f"{_att_path.name} lacks run/fe/b3 columns — stale layout? "
    "Re-run run_attribution_em.do.")


def _stata_target(run_id: str) -> float:
    r = att[(att["run"] == run_id) & (att["fe"] == "fq gq ig")]
    assert len(r) == 1, (
        f"{_att_path.name}: expected exactly one run=='{run_id}' & "
        f"fe=='fq gq ig' row, found {len(r)} — re-run run_attribution_em.do")
    return float(r["b3"].iloc[0])


stata = {k: _stata_target(k) for k in ("B", "A", "A2")}

_can_path = OUT / "headline_3pairwise_canonical.csv"
can = pd.read_csv(_can_path)
assert {"spec", "fe", "b3"} <= set(can.columns), (
    f"{_can_path.name} lacks spec/fe/b3 columns — stale pre-2026-08-08 layout? "
    "Re-run run_headline_3pairwise.do.")
_prim = can[(can["spec"] == "primary") & (can["fe"] == "fq_gq_ig")]
assert len(_prim) == 1, (
    f"{_can_path.name}: no unique spec=='primary' & fe=='fq_gq_ig' row — "
    "stale layout? Re-run run_headline_3pairwise.do.")
_b_can = float(_prim["b3"].iloc[0])
_rel_can = abs(_b_can - stata["B"]) / max(abs(stata["B"]), 1e-300)
assert _rel_can < 1e-6, (
    f"VINTAGE DRIFT between living artifacts: canonical primary b3 {_b_can:.6e} "
    f"!= attribution run-B b3 {stata['B']:.6e} (rel {_rel_can:.2e}) — the two "
    "CSVs were not refreshed from the same panel build")
print("targets read from living artifacts (never hardcoded):")
print(f"  B={stata['B']: .6e}  A={stata['A']: .6e}  A2={stata['A2']: .6e}")
print(f"  canonical primary cross-check: b3={_b_can: .6e}  rel diff {_rel_can:.2e}")

# -----------------------------------------------------------------------------
# R3-F6 freshness: regenerate the parquet from the CURRENT .dta so this script
# can never silently run on a stale panel left over from an earlier build.
# -----------------------------------------------------------------------------
_df, _ = pyreadstat.read_dta((OUT / "audit_c6_panel.dta").as_posix())
_df.to_parquet(OUT / "audit_c6_panel.parquet", index=False)
del _df

con = duckdb.connect()
d = con.execute(f"""
SELECT firm_str, rdate,
       any_value(cn_lag) AS cn,
       any_value(s_lag)  AS s,    -- S_{{t-1}}, PRIMARY timing (2026-08-08)
       any_value(zr_lag) AS zr,
       MAX(CASE WHEN us=1 THEN dw END) - MAX(CASE WHEN us=0 THEN dw END) AS d_dw,
       COUNT(*) AS nrow
FROM read_parquet('{PQ}')
GROUP BY firm_str, rdate
""").df()
con.close()
assert (d["nrow"] == 2).all(), "panel is not exactly 2 rows per (firm, quarter)"
print(f"differenced panel: {len(d):,} firm-quarters, {d['firm_str'].nunique():,} firms")
_n_noslag = int(d["s"].isna().sum())
print(f"firm-quarters without S_(t-1) (dropped, matches reghdfe): {_n_noslag:,}")

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
print("INDEPENDENT b3 (differenced panel, firm+quarter FE, FWL, cn x S_(t-1))")
b_B = b3(d, "B  new snapshot + zero recode (CANONICAL)")
b_A = b3(d[d["zr"] == 0], "A  new snapshot + OLD missing rule (zr==0)")
b_A2 = b3(d[(d["zr"] == 0) & d["old_univ"]], "A2 run A INTERSECT preEM firm universe")

print()
print(f"{'run':4s} {'python':>15s} {'stata(reghdfe)':>18s} {'rel_diff':>12s}")
for k, v in [("B", b_B), ("A", b_A), ("A2", b_A2)]:
    rel = abs(v - stata[k]) / abs(stata[k])
    print(f"{k:4s} {v:15.6e} {stata[k]:18.6e} {rel:12.2e}")
    assert rel < 1e-4, f"run {k}: python and Stata disagree (rel {rel:.2e})"
print("\nall three coefficients reproduced independently (rel diff < 1e-4)")
