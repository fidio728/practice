"""
run_ri_extensive.py — randomization inference for the EXTENSIVE-MARGIN outcomes
(A1/G1), the design-based ARBITER for run_extensive_margin.do.

Mirrors run_ri_3pairwise.py's collapse machinery. On the balanced 2-per-
(firm,quarter) grid the three pairwise FE (it firm x quarter + gt group x quarter
+ ig firm x group) collapse, on the US-minus-NONUS within-firm-quarter difference
Delta y, to a FIRM + QUARTER two-way FE regression on the difference panel:

    Delta y_it = b2 * cn_it + b3 * (cn_it * S_t) + phi_i + delta_t + e_it

There is no closed-form sufficient statistic (firm-demeaning of cn*S mixes S
across a firm's quarters), so cn*S is two-way-demeaned each draw via fast bincount
alternating projections; b3 via FWL against the (once-)demeaned Delta y and cn.

DVs (one RI each; free-permutation AND circular-shift per the post-A0 convention,
both reported — if they disagree materially the circular-shift p is the arbiter):
  d_breadth (PRIMARY), d_nh          — paired difference over all firm-quarters
  exit, init                          — paired difference on the both-conditioning
                                        restricted cells only (exit: both group
                                        rows held_lag==1; init: both held_lag==0),
                                        matching the .do estimation samples exactly.
                                        exit/init condition on a LAGGED OUTCOME
                                        state, so their b3 is a conditional
                                        exit/initiation HAZARD among the
                                        conditioned-in set, NOT an unconditional
                                        divestment probability; and the both-held /
                                        neither restriction selects toward
                                        dual-held (larger, more-visible) firms,
                                        narrowing the estimand away from "US
                                        divestment" broadly.
Each DV is also run on the riskset-conditional (engaged) subsample (semi-join on
c6_panel_riskset.dta firm-quarter keys) when that file is present. Riskset
restricts the ROW set only; it partially mitigates coverage-driven false exits
but does NOT recover unfiled counts nor adjust the n_active denominator. d_breadth
(a ratio) largely cancels a uniform reporting lag (n_holders and n_active shrink
together); d_nh / exit / init are unnormalized COUNT outcomes and carry the
reporting-lag exposure — so the coverage threat is concentrated OFF the PRIMARY.

PRE-REGISTERED READING (header note, NOT a conclusion). The partial-divestment /
pro-H2.1 sign is DV-SPECIFIC, because `exit` is a held->0 (full-drop) indicator
and moves OPPOSITE to the count/breadth DVs. Divestment for the US on the DDD
triple predicts:
    b3 < 0  for d_breadth, d_nh, init   (breadth / holders / initiations fall)
    b3 > 0  for exit                    (held positions drop to zero MORE often)
A negative exit b3 = FEWER US exits = anti-divestment, NOT divestment (reading
exit with the breadth sign is the project's documented wrong-sign trap). null
across all four = completeness; the sign FLIPPED from divestment = anti-H2.1,
same direction as the weight margin.

DRIFT-ANCHOR gate: extmargin_b3_anchor.csv (written by the .do) carries the Stata
3-pairwise b3 per DV; this script prints its collapsed b3 next to it with the
relative error. Values are NEW, so this is a print-and-compare, NOT a hard abort.

HONESTY: every number verbatim from this run; failures reported as failures.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import pyreadstat

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")

# --- Smoke flag default OFF (fast syntax pass: fewer permutations) ---
SMOKE = False
N_PERM = 200 if SMOKE else 5000
DEMEAN_ITERS = 30
SEED = 20260702

PANEL_DTA = OUT / "extensive_margin_panel.dta"
RISKSET_DTA = OUT / "c6_panel_riskset.dta"
ANCHOR_CSV = OUT / "extmargin_b3_anchor.csv"


def make_demeaner(firm_c, q_c, iters):
    """Two-way (firm, quarter) alternating-projection demeaner via bincount."""
    nf = int(firm_c.max()) + 1
    nq = int(q_c.max()) + 1
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


def ri_twoway(sub, ycol, n_perm=N_PERM, seed=SEED, iters=DEMEAN_ITERS):
    """Free-permutation AND circular-shift RI for the sharp null b3=0 on the
    US-minus-NONUS difference panel. Returns (b_obs, p_free, p_circ, n, nq, degen).

    A subsample whose residual DoF under the firm+quarter two-way FE is <=0 is
    DEGENERATE (the Stata side fails such specs with r(2001)); we return NaN p's
    and degen=True rather than a garbage b3 or a spuriously tiny permutation p."""
    sub = sub.dropna(subset=[ycol, "cn", "s"]).reset_index(drop=True)
    n = sub.shape[0]
    if n == 0:
        return np.nan, np.nan, np.nan, 0, 0, True
    firm_c = pd.factorize(sub["firm_str"])[0]
    # chronological quarter code (sort=True): np.roll shifts along calendar time
    q_c, uq = pd.factorize(sub["rdate"], sort=True)
    q_c = np.asarray(q_c)
    nq = len(uq)
    n_firm = int(firm_c.max()) + 1
    # firm + quarter FE absorb (n_firm + nq - 1) params; guard non-positive DoF
    if (n - n_firm - nq + 1) <= 0:
        return np.nan, np.nan, np.nan, n, nq, True
    demean = make_demeaner(firm_c, q_c, iters)
    y = demean(sub[ycol].to_numpy(float))      # two-way demeaned outcome (once)
    cn = demean(sub["cn"].to_numpy(float))       # two-way demeaned cn (once)
    cn_raw = sub["cn"].to_numpy(float)
    cn_dot = float(cn @ cn)                       # FWL denom piece (fixed)
    S_chrono = np.zeros(nq)                       # per-quarter shock, chronological
    S_chrono[q_c] = sub["s"].to_numpy(float)      # shock constant within quarter

    def beta3(Svec):
        if cn_dot <= 0:
            return np.nan
        x = cn_raw * Svec[q_c]                    # cn * S at the row level
        xt = demean(x)                           # two-way demean
        b = (cn @ xt) / cn_dot                    # FWL: residualize xt on cn
        xr = xt - b * cn
        den = xr @ xr
        return (y @ xr) / den if den > 0 else np.nan

    b_obs = beta3(S_chrono)
    if not np.isfinite(b_obs):
        return np.nan, np.nan, np.nan, n, nq, True

    # free permutation of the quarter shocks (anti-conservative under serial S,
    # so the circular-shift p is reported alongside as the serial-robust arbiter)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        if abs(beta3(rng.permutation(S_chrono))) >= abs(b_obs) - 1e-300:
            cnt += 1
    p_free = (cnt + 1) / (n_perm + 1)

    # circular shift: nq-1 non-trivial rolls preserve the serial structure of S
    shift_b = np.array([beta3(np.roll(S_chrono, k)) for k in range(1, nq)])
    if len(shift_b) > 0:
        p_circ = (np.nansum(np.abs(shift_b) >= abs(b_obs) - 1e-300) + 1) / (len(shift_b) + 1)
    else:
        p_circ = np.nan
    return b_obs, p_free, p_circ, n, nq, False


def collapse(panel_parquet):
    """US-minus-NONUS firm-quarter differences for every DV, plus the per-group
    held_lag flags needed to reconstruct the exit/init both-conditioning cells.
    COUNT(DISTINCT) DVs are integer aggregation upstream — immune to the duckdb
    parallel float-sum nondeterminism; this collapse only differences them."""
    con = duckdb.connect()
    d = con.execute(f"""
    SELECT firm_str,
           CAST(rdate AS TIMESTAMP) AS rdate,
           any_value(cn_lag) AS cn,
           any_value(shock)  AS s,
           MAX(CASE WHEN us=1 THEN d_breadth END) - MAX(CASE WHEN us=0 THEN d_breadth END) AS d_db,
           MAX(CASE WHEN us=1 THEN d_nh END)      - MAX(CASE WHEN us=0 THEN d_nh END)      AS d_dnh,
           -- ::DOUBLE forces NULL -> NaN in a plain float64 column. Without it,
           -- duckdb .df() returns these BIGINT-with-NULL aggregates as pandas
           -- NULLABLE Int64, and the both_held/neither masks below become a
           -- nullable BooleanArray carrying pd.NA at every series-start cell,
           -- which makes np.where raise "boolean value of NA is ambiguous" and
           -- crashes the whole RI before any p-value. NaN==1 -> False, so the
           -- masks and np.where run unchanged on the float64 columns.
           MAX(CASE WHEN us=1 THEN held_lag END)::DOUBLE AS hl_us,
           MAX(CASE WHEN us=0 THEN held_lag END)::DOUBLE AS hl_nonus,
           MAX(CASE WHEN us=1 THEN "exit" END)::DOUBLE AS exit_us,
           MAX(CASE WHEN us=0 THEN "exit" END)::DOUBLE AS exit_nonus,
           MAX(CASE WHEN us=1 THEN "init" END)::DOUBLE AS init_us,
           MAX(CASE WHEN us=0 THEN "init" END)::DOUBLE AS init_nonus
    FROM read_parquet('{panel_parquet}')
    GROUP BY firm_str, rdate
    """).df()
    con.close()
    # exit/init differences defined ONLY on the both-conditioning paired cells.
    # hl_/exit_/init_ are float64 (::DOUBLE above) so NaN==1 -> False: a
    # series-start (NaN) cell falls out of both masks, never raising on pd.NA.
    both_held = (d["hl_us"] == 1) & (d["hl_nonus"] == 1)
    neither = (d["hl_us"] == 0) & (d["hl_nonus"] == 0)
    d["d_exit"] = np.where(both_held, d["exit_us"] - d["exit_nonus"], np.nan)
    d["d_init"] = np.where(neither, d["init_us"] - d["init_nonus"], np.nan)
    return d


def load_anchor():
    """{(dv, sample): stata_b3} from the .do's drift-anchor CSV, if present."""
    if not ANCHOR_CSV.exists():
        return {}
    a = pd.read_csv(ANCHOR_CSV)
    out = {}
    for _, r in a.iterrows():
        out[(str(r["dv"]), str(r["sample"]))] = float(r["b3"])
    return out


def main():
    print("=" * 84)
    print(f"run_ri_extensive.py  SMOKE={SMOKE}  N_PERM={N_PERM}  iters={DEMEAN_ITERS}  SEED={SEED}")
    print("=" * 84)

    # R3-F6 freshness: regenerate the parquet from the CURRENT .dta so this can
    # never silently run on a stale panel left over from an earlier build.
    df, _ = pyreadstat.read_dta(PANEL_DTA.as_posix())
    panel_pq = OUT / "extensive_margin_panel.parquet"
    df.to_parquet(panel_pq, index=False)
    d = collapse(panel_pq.as_posix())

    # riskset (engaged) semi-join key set — string key avoids datetime dtype drift
    rs_keys = None
    if RISKSET_DTA.exists():
        rs, _ = pyreadstat.read_dta(RISKSET_DTA.as_posix())
        rs_keys = set(
            rs["firm_str"].astype(str) + "|" +
            pd.to_datetime(rs["rdate"]).dt.strftime("%Y-%m-%d")
        )
        print(f"riskset semi-join: {len(rs_keys):,} engaged firm-quarter keys loaded")
    else:
        print("NOTE: c6_panel_riskset.dta not found — riskset RI SKIPPED.")

    d["_key"] = (d["firm_str"].astype(str) + "|" +
                 pd.to_datetime(d["rdate"]).dt.strftime("%Y-%m-%d"))

    anchor = load_anchor()
    if anchor:
        print(f"drift-anchor: loaded {len(anchor)} Stata b3 value(s) from {ANCHOR_CSV.name}")
    else:
        print(f"drift-anchor: {ANCHOR_CSV.name} not found — printing RI b3 only (no comparison)")

    # (dv-label, difference column)
    dvspecs = [("d_breadth", "d_db"), ("d_nh", "d_dnh"), ("exit", "d_exit"), ("init", "d_init")]
    samples = [("full", d)]
    if rs_keys is not None:
        samples.append(("riskset", d[d["_key"].isin(rs_keys)].copy()))

    print()
    print(f"{'dv':10s} {'sample':8s} {'b3_RI':>13s} {'b3_Stata':>13s} {'relerr':>9s} "
          f"{'p_free':>8s} {'p_circ':>8s} {'n_fq':>9s} {'nq':>4s} {'flag':>6s}")
    rows = []
    for samp_lbl, dsamp in samples:
        for dv_lbl, ycol in dvspecs:
            b_obs, p_free, p_circ, n, nq, degen = ri_twoway(dsamp, ycol)
            b_stata = anchor.get((dv_lbl, samp_lbl), np.nan)
            if np.isfinite(b_stata) and np.isfinite(b_obs) and b_stata != 0:
                relerr = abs(b_obs / b_stata - 1.0)
            else:
                relerr = np.nan
            flag = "DEGEN" if degen else ""
            print(f"{dv_lbl:10s} {samp_lbl:8s} {b_obs:13.5e} {b_stata:13.5e} "
                  f"{relerr:9.2e} {p_free:8.4f} {p_circ:8.4f} {n:9,d} {nq:4d} {flag:>6s}")
            rows.append({
                "dv": dv_lbl, "sample": samp_lbl, "fe": "it+gt+ig (3-pairwise)",
                "b3_ri": b_obs, "b3_stata_anchor": b_stata, "relerr": relerr,
                "p_free_perm": p_free, "p_circular_shift": p_circ,
                "n_firmquarters": n, "n_quarters": nq,
                "arbiter_p": p_circ, "ri_degenerate": degen,
            })

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "ri_extensive_results.csv", index=False)
    print(f"\nwrote ri_extensive_results.csv  ({len(res)} rows)")
    print("  reading (DV-SPECIFIC sign; exit is a held->0 indicator, opposite the "
          "count/breadth DVs):")
    print("    partial divestment for US  =>  b3<0 for d_breadth, d_nh, init ; "
          "b3>0 for exit")
    print("    null across all four = completeness ; sign flipped from the above = "
          "anti-H2.1 (weight-margin direction).")
    print("  exit/init b3 are conditional exit/initiation HAZARDS on the both-held / "
          "neither cells (dual-held-selected), not unconditional probabilities.")
    print("  circular-shift p is the serial-robust arbiter when it diverges from the free-perm p.")


if __name__ == "__main__":
    main()
