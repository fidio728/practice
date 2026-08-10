"""
run_ri_country_ddd.py — the REPORTING ARBITER for the country-level
portfolio-weight DDD (task e2 item 2, 2026-08-10).  CODE ONLY at time of
writing: nothing here has been executed.

================================================================================
WHAT IT DOES
================================================================================
Circular-shift randomization inference for the LOCKED specification

    dW_{c,g,t} = beta * US_g x M_{k,c,t-1} x S_{t-1}
               +  gam * US_g x M_{k,c,t-1}
               + alpha_{cg} + alpha_{ct} + alpha_{gt} + e_{c,g,t}

for k in {M1, M2, M3}, with a single-step max-|t| FWER across those three
statistics.  A FREE PERMUTATION p is reported as a cross-check.

--------------------------------------------------------------------------------
ONE CALENDAR DRAW PER REPLICATION  (fixed 2026-08-10 — external review item 6)
--------------------------------------------------------------------------------
A single-step max-|t| FWER is only valid if, on every replication, the three
statistics come from THE SAME draw of the randomisation device.  The previous
vintage did not do that.  It rotated each measure's OWN compressed shock vector
S[q_keep_k] by r and called "rotation r" a common draw; when the three measures
survive different quarter sets, S[q_keep_1] rotated by r and S[q_keep_2] rotated
by r are DIFFERENT calendar re-assignments, so max-|t| was taken across three
statistics drawn from three different devices — which does not control the FWER.
The free-permutation arm was worse: `perm[perm < nq_k]' produced a different
permutation per measure from one parent vector.

The device is now the CALENDAR, drawn ONCE per replication:
  * the rotation base is the FULL shock calendar S_full restricted to the grid
    quarters on which S_{t-1} is DEFINED (`cal' below, n_cal quarters), not any
    measure's compressed vector.  S_{t-1} is missing on the panel's first
    quarter by construction, and a rotation base carrying a NaN would inject
    NaN into some measure's design on some rotations;
  * per draw, that one vector is rotated (or permuted) ONCE and scattered back
    onto the calendar, giving ONE re-assignment of the shock to calendar
    quarters;
  * each measure then READS that one rotated calendar at ITS OWN quarters via
    the ORIGINAL quarter index (q_orig), exactly as the observed design reads
    the unrotated calendar.
So all three statistics share one draw by construction, whatever their samples,
and the rotation count n_cal - 1 is a property of the CALENDAR, not of a sample.

A DRAW IS ALL-OR-NOTHING.  A two-way cluster meat is not guaranteed PSD in
finite samples, so a single draw can return a NaN SE — and therefore a NaN t —
for one measure.  Such a draw is dropped from ALL THREE, never just from the one
that failed: a nan-aware max over the row would compute "max over three
statistics" from two of them and bias the FWER null downward, and it would give
the three measures different draw sets, undoing the common draw.  The dropped
count is reported (n_rotations_incomplete / n_perm_incomplete) and more than 10%
aborts, because a null built on a selected subset of rotations is not the test
the output claims to be.
In addition — and this is a separate requirement, not a consequence — the three
measures' SAMPLE QUARTER SETS are HARD-GATED to be identical.  The old code only
printed a NOTE when they differed and then silently used `min(nq) - 1' rotations.

--------------------------------------------------------------------------------
THE STATA ANCHOR IS GATED ON N, SE AND t — NOT ONLY ON b
--------------------------------------------------------------------------------
max-|t| is computed from the PYTHON t, so gating only b left the statistic that
actually decides the verdict unchecked.  Every measure now hard-gates N (exact),
se and t (relative) against run_country_weight_ddd.do's CSV.

The one wrinkle is the small-sample factor.  The two-way CGM factor is
    q_j = G_j/(G_j-1) * (N-1)/(N-K),   K = #regressors + absorbed parameters,
and reghdfe's default `dofadjustments(... clusters ...)' declares an absorbed FE
group REDUNDANT when it is NESTED within a clustering variable, so reghdfe's K
can be far smaller than the raw parameter count.  Here cg and ct are nested in
the country cluster and gt is nested in the quarter cluster, so the two
conventions differ by a factor of roughly sqrt((N-K_all)/(N-K_nested)) ~ 1.5 —
big enough that gating against the wrong one would abort every run.  So this
file computes the SE under BOTH conventions (nestedness is CHECKED, never
assumed) and requires Stata's se_crve to match ONE of them to SE_GATE_RELTOL.
Whichever matches is then used for t_obs AND for every RI draw, so the reported
t is on reghdfe's scale.  A mismatch against BOTH conventions is a real defect
(wrong sample, wrong regressor, wrong cluster dimensions) and aborts.
NOTE: the factor is constant across draws, so it cannot move a p-value within a
measure; it can only matter across measures — where N and the cluster counts can
differ — which is why it is pinned per measure rather than assumed common.
WHY SE_GATE_RELTOL IS 1e-3 AND NOT 1e-8.  The two conventions differ by ~50%, so
1e-3 separates them with five orders of margin, while still absorbing the one
ambiguity this reimplementation cannot resolve from outside reghdfe: whether the
absorbed intercept is counted inside K.  That off-by-one moves the SE by about
1/(2(N-K)) ~ 1e-4 here.  A tighter gate would abort on that alone; a looser one
would stop discriminating between the conventions.

N is gated EXACTLY (listwise deletion must agree row for row), b and t are gated
at B_GATE_RELTOL.  t is gated even though b and se already are, because t is the
statistic max-|t| ranks and an agreeing b with an agreeing se but a disagreeing
t would mean the Stata CSV's t column was formed from something else.

--------------------------------------------------------------------------------
FULL FE RE-ESTIMATION ON THE ARBITER ARM (the spec's hard requirement)
--------------------------------------------------------------------------------
Every CIRCULAR draw goes through `fit_from_raw`: y, US x M and US x M x S_shift
are each re-projected onto the three pairwise FE FROM RAW by alternating
projections, and the 2-regressor OLS plus its two-way cluster vcov are re-solved.
Nothing is cached, nothing is collapsed, on that arm.

The FREE-PERMUTATION arm (5,000 draws x 3 measures) uses an EXACT linear
identity rather than re-demeaning:
    x3 = x2 . S[q]  and  demean(.) is linear, so
    demean(x3) = SUM_t S_t * demean(x2 . 1[q = t]) = B @ S,
    B[:, t] = demean(x2 . 1[q = t])   (built once per measure, nq columns).
B @ S is the SAME ROW-LEVEL demeaned regressor the from-raw route produces, so
the OLS, the residuals and the cluster-robust t are all still computed at row
level from the full model — this is an algebraic identity, not a collapse to a
sufficient statistic.  It is nevertheless treated as a proof obligation:
`selfcheck_basis_route` compares BOTH b AND t against the from-raw route on the
observed draw and on two random permutations and aborts unless they agree to
1e-8.

FE ABSORPTION is alternating projections over cg / ct / gt with a hard
CONVERGENCE CERTIFICATE: after the sweeps, the within-cell mean of the result is
recomputed for every cell of all three groups and must be below 1e-11.  That
condition IS the FWL orthogonality condition; failing it aborts rather than
returning a partially absorbed vector.

--------------------------------------------------------------------------------
INDEPENDENT CROSS-CHECK OF THE ESTIMATOR (not used to produce any number)
--------------------------------------------------------------------------------
With exactly two holder groups on a balanced (c,t) grid the three pairwise FE
collapse, on the US-minus-NONUS within-(c,t) difference D_ct, to
    D_ct = gam * M_{c,t-1} + beta * M_{c,t-1} S_{t-1} + phi_c + delta_t + u
(alpha_ct cancels in the difference; alpha_cg -> phi_c; alpha_gt -> delta_t).
The observed beta is re-derived that way and compared at 1e-8.  Unbalanced grid
=> reported as unavailable, never silently skipped.

--------------------------------------------------------------------------------
t STATISTIC AND WHY max-|t| RATHER THAN max-|b|
--------------------------------------------------------------------------------
t uses the two-way cluster (country, quarter) vcov — the same vce as the Stata
leg.  M1, M2 and M3 are on DIFFERENT SCALES, so a max-|b| FWER statistic would
mechanically select the largest-scaled measure; max-|t| is scale-free.  The
small-sample factor is constant across draws (same N, K and cluster counts), so
it cannot move any RI p-value; the CRVE inference of record stays Stata's.

--------------------------------------------------------------------------------
DRIFT ANCHOR (never hardcoded)
--------------------------------------------------------------------------------
b for each measure is hard-gated at 1e-4 relative against the LIVING artifact
output/country_weight_ddd.csv (family=="main", outcome=="dw_global",
spec=="ddd_3pairwise").  Missing file or stale layout => abort.  Run
run_country_weight_ddd.do first.

================================================================================
CAVEAT — PRINTED AND WRITTEN INTO THE OUTPUT (required by the spec)
================================================================================
CIRCULAR SHIFT ASSUMES THE SHOCK SERIES IS EXCHANGEABLE / APPROXIMATELY
STATIONARY UNDER ROTATION.  Rotating S_{t-1} PRESERVES its serial-correlation
structure — which free permutation destroys, making the free p anti-conservative
for a serially correlated shock — but validity still requires the shock's
distribution to be invariant to a calendar rotation.  It is not exactly: the
US-China series has a level regime shift.  p_circ is therefore the SERIAL-ROBUST
arbiter, not an exact test.  It also carries a granularity floor of
1/(n_rotations+1): with ~89 quarters no rotation p can fall below ~0.011,
however large the effect.

OUTPUT: output/ri_country_ddd.csv
  measure, b_obs, t_obs, p_circ, p_free, p_fwer_max_t, n_rotations, + provenance
  (including script_sha256 / panel_sha256 / anchor_sha256 — see PROVENANCE)
Rotation discipline: refuses to overwrite; rotate to *_cwpre first.

--------------------------------------------------------------------------------
PROVENANCE
--------------------------------------------------------------------------------
The SHA256 of THIS FILE and of every input artifact is printed at the top of the
log and written into every row of the output CSV, so a set of p-values can be
tied to the exact bytes that produced them.  mtimes cannot do that here: OneDrive
rewrites them on sync and output/ is a junction to E:.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(r"c:/Users/xl/OneDrive - Universitat Ramón Llull/git/practice/julia_descriptive/output")

PANEL_DTA = OUT / "country_weight_panel.dta"
STATA_CSV = OUT / "country_weight_ddd.csv"
RESULT_CSV = OUT / "ri_country_ddd.csv"

N_PERM = 5000
SEED = 20260810
DEMEAN_MAXIT = 5000
DEMEAN_TOL = 1e-13
CERT_TOL = 1e-11          # FE-cell mean of the demeaned vector must be below this
B_GATE_RELTOL = 1e-4      # python b vs Stata b, and python t vs Stata t
SE_GATE_RELTOL = 1e-3     # python se vs Stata se_crve (see the docstring block)
COLLAPSE_RELTOL = 1e-8    # difference-collapse cross-check
BASIS_RELTOL = 1e-8       # basis route vs from-raw route (b AND t)

MEASURES = ("M1", "M2", "M3")
MCOL = {"M1": "m1_lag", "M2": "m2_lag", "M3": "m3_lag"}

# Small-sample K conventions for the two-way cluster factor
#   q_j = G_j/(G_j-1) * (N-1)/(N-K)
# "all"    : K counts every absorbed FE parameter (the raw parameter count)
# "nested" : K drops the FE groups that reghdfe declares REDUNDANT because they
#            are nested inside a clustering variable (its dofadjustments default)
# The adopted convention is decided PER MEASURE at run time by matching Stata;
# it is never assumed.  SE_CONVENTION_DEFAULT is what an IMPORTING script (e.g.
# run_country_decomposition.py) gets when it calls fit_from_raw without an
# anchor to adopt from: "all" is the CONSERVATIVE choice (larger K -> larger SE).
SE_CONVENTIONS = ("all", "nested")
SE_CONVENTION_DEFAULT = "all"

CAVEAT = ("circular-shift RI assumes S_{t-1} is exchangeable / approximately "
          "stationary under calendar rotation; granularity floor = "
          "1/(n_rotations+1)")


# =============================================================================
# PROVENANCE (external-review item, 2026-08-10)
# =============================================================================
# A set of p-values must be tie-able to the EXACT BYTES that produced it: this
# script, the panel it read and the Stata anchor it gated against.  mtimes
# cannot do that here (OneDrive rewrites them on sync; output/ is a junction to
# E:), so every run prints the SHA256 of itself and of every input, and stamps
# the same hashes into every row of the output CSV.
SELF_PATH = Path(__file__).resolve()
HASH_CHUNK = 1 << 22          # 4 MiB


def sha256_file(p: Path) -> tuple[str, int]:
    """(sha256_hex, size_bytes); 'MISSING' when the file is absent."""
    p = Path(p)
    if not p.is_file():
        return "MISSING", 0
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(HASH_CHUNK), b""):
            h.update(blk)
            n += len(blk)
    return h.hexdigest(), n


def log_provenance(inputs: dict[str, Path]) -> dict[str, str]:
    """Print the script hash and every input hash; return them for the CSV."""
    print("PROVENANCE — SHA256 of this script and of every input artifact")
    out = {}
    dg, sz = sha256_file(SELF_PATH)
    out["script_sha256"] = dg
    print(f"  SCRIPT  {SELF_PATH.name:<34s} {dg}  ({sz:,} B)")
    for label, p in inputs.items():
        dg, sz = sha256_file(p)
        out[label] = dg
        print(f"  INPUT   {Path(p).name:<34s} {dg}  ({sz:,} B)")
    return out


# =============================================================================
# rotation discipline
# =============================================================================
def rotation_guard(target: Path) -> None:
    """Never overwrite a canonical artifact in place: rotate to *_cwpre first,
    and REFUSE outright if that slot is already occupied (two live vintages)."""
    pre = target.with_name(target.stem + "_cwpre" + target.suffix)
    if target.exists():
        if pre.exists():
            raise SystemExit(
                f"REFUSING: {target.name} exists AND the rotation slot "
                f"{pre.name} is already occupied — two live vintages; resolve "
                "by hand.")
        raise SystemExit(
            f"{target.name} already exists — rename it to {pre.name} "
            "(rotation rule) before re-running.")


# =============================================================================
# panel loading + validation
# (importable — run_country_decomposition.py reuses this and the estimator)
# =============================================================================
REQUIRED_COLS = ("sec_country", "holder_group", "rdate", "dw_global",
                 "m1_lag", "m2_lag", "m3_lag", "s_lag")


def load_country_panel(dta_path: Path = PANEL_DTA) -> pd.DataFrame:
    """Read e1's country weight panel, validate it, attach FE codes.

    Fails BY NAME on a missing column so an e1 contract gap surfaces as a named
    dependency rather than being silently worked around.
    """
    import pyreadstat

    if not dta_path.exists():
        raise SystemExit(
            f"DEPENDENCY: {dta_path} not found — it is task e1's output. "
            "This script estimates only; it builds no panel.")
    df, _ = pyreadstat.read_dta(dta_path.as_posix())
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"DEPENDENCY: {dta_path.name} is missing column(s) {missing}. "
            "See the contract block in _country_weight_lib.do.")

    df = df.copy()
    df["rdate"] = pd.to_datetime(df["rdate"])
    df["qper"] = df["rdate"].dt.to_period("Q")
    groups = set(df["holder_group"].unique())
    if not groups <= {"US", "NONUS"}:
        raise SystemExit(f"holder_group outside {{US, NONUS}}: {sorted(groups)}")
    df["us"] = (df["holder_group"] == "US").astype(float)

    if df.duplicated(["sec_country", "holder_group", "qper"]).any():
        raise SystemExit("duplicate (country, group, quarter) rows — panel key broken.")

    df = df.sort_values(["sec_country", "holder_group", "qper"],
                        kind="mergesort").reset_index(drop=True)
    df["c_code"] = pd.factorize(df["sec_country"], sort=True)[0]
    df["g_code"] = pd.factorize(df["holder_group"], sort=True)[0]
    # chronological quarter code so np.roll shifts along CALENDAR time
    df["q_code"] = pd.factorize(df["qper"], sort=True)[0]

    quarters = pd.PeriodIndex(sorted(df["qper"].unique()))
    if not (np.diff(quarters.asi8) == 1).all():
        raise SystemExit(
            "quarter grid is NOT contiguous — a circular shift would not respect "
            "calendar time; refusing to run RI.")

    sq = df.groupby("q_code")["s_lag"].agg(["nunique", "first"])
    if (sq["nunique"] > 1).any():
        raise SystemExit("s_lag varies within a quarter — expected one common S_{t-1}.")

    df.attrs["quarters"] = quarters
    df.attrs["S_by_quarter"] = sq["first"].reindex(range(len(quarters))).to_numpy(float)
    return df


def estimation_sample(df: pd.DataFrame, measure: str, ycol: str = "dw_global",
                      extra: list[str] | None = None) -> pd.DataFrame:
    """The rows reghdfe would keep for one measure, with DENSE FE codes.

    Mirrors reghdfe's listwise deletion on the model variables, then
    re-factorizes so cg/ct/gt are dense on the estimation sample.
    """
    mcol = MCOL[measure]
    keep = ["sec_country", "holder_group", "qper", "us", ycol, mcol, "s_lag",
            "c_code", "g_code", "q_code"] + list(extra or [])
    sub = df[keep].rename(columns={mcol: "m_lag"}).copy()
    sub = sub.dropna(subset=[ycol, "m_lag"]).reset_index(drop=True)
    sub["c_code"] = pd.factorize(sub["c_code"], sort=True)[0]
    sub["g_code"] = pd.factorize(sub["g_code"], sort=True)[0]
    # NOTE: q_code is deliberately re-factorized against the FULL quarter grid
    # only if no quarter was lost; otherwise the rotation vector must be
    # restricted to the surviving quarters (handled by the caller via q_keep).
    sub["q_orig"] = sub["q_code"]
    sub["q_code"] = pd.factorize(sub["q_code"], sort=True)[0]
    # PAIRWISE FE CODES. pandas >= 3 rejects a bare list in pd.factorize
    # ("factorize requires a Series, Index, ExtensionArray, np.ndarray or
    # NumpyExtensionArray got list"), so the pairs are built as INTEGER
    # COMPOSITES on an ndarray instead. c_code/g_code/q_code are already dense
    # 0-based codes on this sample, so c * n_g + g is a bijection onto the
    # observed pairs and factorizing it re-densifies exactly as the tuple route
    # did — with no Python-object hashing.
    c = sub["c_code"].to_numpy(np.int64)
    g = sub["g_code"].to_numpy(np.int64)
    q = sub["q_code"].to_numpy(np.int64)
    n_g = int(g.max()) + 1
    n_q = int(q.max()) + 1
    sub["cg"] = pd.factorize(c * n_g + g, sort=True)[0]
    sub["ct"] = pd.factorize(c * n_q + q, sort=True)[0]
    sub["gt"] = pd.factorize(g * n_q + q, sort=True)[0]
    return sub


# =============================================================================
# FE absorption
# =============================================================================
def make_demeaner(codes, maxit: int = DEMEAN_MAXIT, tol: float = DEMEAN_TOL):
    """Absorb a list of FE groups by alternating projections, with a hard
    convergence certificate (see module docstring)."""
    codes = [np.asarray(c, dtype=np.int64) for c in codes]
    counts = [np.bincount(c).astype(float) for c in codes]
    for cnt in counts:
        if (cnt == 0).any():
            raise SystemExit("FE codes are not dense — re-factorize on the sample.")

    def demean(x):
        x = np.asarray(x, dtype=float).copy()
        for _ in range(maxit):
            worst = 0.0
            for c, cnt in zip(codes, counts):
                m = np.bincount(c, weights=x, minlength=cnt.size) / cnt
                worst = max(worst, float(np.abs(m).max()))
                x -= m[c]
            if worst < tol:
                break
        worst_final = 0.0
        for c, cnt in zip(codes, counts):
            m = np.bincount(c, weights=x, minlength=cnt.size) / cnt
            worst_final = max(worst_final, float(np.abs(m).max()))
        if not worst_final < CERT_TOL:
            raise SystemExit(
                f"FE ABSORPTION DID NOT CONVERGE: residual FE-cell mean "
                f"{worst_final:.3e} >= {CERT_TOL:.0e} after {maxit} sweeps. "
                "Refusing to report a partially absorbed estimate.")
        return x

    return demean


def twoway_cluster_se(Xd: np.ndarray, u: np.ndarray, c1, c2, c12, K: float):
    """Cameron-Gelbach-Miller two-way cluster vcov on demeaned data.

    V = B (q1 M1 + q2 M2 - q12 M12) B,   B = (X'X)^-1,
    q_j = G_j/(G_j-1) * (N-1)/(N-K),  K = the small-sample parameter count.
    The correction is IDENTICAL across RI draws, so it cannot move an RI p-value;
    it exists so t_obs is comparable in magnitude to reghdfe's.
    """
    n, k = Xd.shape
    B = np.linalg.inv(Xd.T @ Xd)

    def meat(codes):
        codes = np.asarray(codes, dtype=np.int64)
        G = int(codes.max()) + 1
        S = np.empty((G, k))
        for j in range(k):
            S[:, j] = np.bincount(codes, weights=Xd[:, j] * u, minlength=G)
        q = (G / (G - 1.0)) * ((n - 1.0) / max(n - K, 1.0))
        return q * (S.T @ S), G

    M1, G1 = meat(c1)
    M2, G2 = meat(c2)
    M12, _ = meat(c12)
    V = B @ (M1 + M2 - M12) @ B
    d = np.diag(V)
    # a two-way meat is not guaranteed PSD in finite samples: surface it as NaN
    # rather than as a silently truncated SE.
    d = np.where(d > 0, d, np.nan)
    return np.sqrt(d), min(G1, G2)


def _is_nested(fe_codes, cluster_codes) -> bool:
    """True iff every level of the FE group sits inside ONE cluster.

    This is the condition reghdfe's `dofadjustments(clusters)' tests before
    declaring an absorbed FE group redundant.  CHECKED, never assumed: cg and ct
    are nested in the country cluster and gt in the quarter cluster ONLY as long
    as the panel keys mean what they are supposed to mean.
    """
    fe = np.asarray(fe_codes, dtype=np.int64)
    cl = np.asarray(cluster_codes, dtype=np.int64)
    order = np.lexsort((cl, fe))          # by FE level, then by cluster
    fe_s, cl_s = fe[order], cl[order]
    # within a block the cluster codes are sorted, so the level sits inside one
    # cluster iff its first and last cluster code coincide
    first = np.r_[True, fe_s[1:] != fe_s[:-1]]
    last = np.r_[fe_s[1:] != fe_s[:-1], True]
    return bool((cl_s[first] == cl_s[last]).all())


def k_conventions(sub) -> dict[str, float]:
    """The two small-sample K's, with the nestedness of each FE group CHECKED.

    K_all    = 2 regressors + every absorbed FE parameter (net of the shared
               intercepts): the raw parameter count.
    K_nested = 2 regressors + 1 intercept + the parameters of only those FE
               groups that are NOT nested in a clustering variable — reghdfe's
               default, which zeroes a nested group's contribution to e(df_a).
    """
    c = sub["c_code"].to_numpy()
    q = sub["q_code"].to_numpy()
    n_lev = {g: int(sub[g].max()) + 1 for g in ("cg", "ct", "gt")}
    k_all = 2.0 + (n_lev["cg"] + n_lev["ct"] + n_lev["gt"] - 2)
    nested = {g: (_is_nested(sub[g].to_numpy(), c)
                  or _is_nested(sub[g].to_numpy(), q))
              for g in ("cg", "ct", "gt")}
    k_nested = 2.0 + 1.0 + sum(0 if nested[g] else (n_lev[g] - 1)
                               for g in ("cg", "ct", "gt"))
    return {"all": k_all, "nested": k_nested, "nested_flags": nested,
            "levels": n_lev}


def _solve(y_d, x2_d, x3_d, sub, convention: str = SE_CONVENTION_DEFAULT,
           kconv: dict | None = None):
    """OLS of the demeaned outcome on [x2, x3] + two-way cluster SE on beta3.

    The SE is returned under BOTH small-sample conventions (se_all / se_nested,
    and the matching t's); `se' and `t' are aliases for the convention passed in,
    which the caller pins against Stata before any number is reported.
    """
    X = np.column_stack([x2_d, x3_d])
    XX = X.T @ X
    kc0 = kconv if kconv is not None else k_conventions(sub)
    if np.linalg.matrix_rank(XX) < 2:
        return {"b": np.nan, "se": np.nan, "t": np.nan, "n": int(len(y_d)),
                "se_all": np.nan, "se_nested": np.nan,
                "t_all": np.nan, "t_nested": np.nan, "df": np.nan,
                "k_all": kc0["all"], "k_nested": kc0["nested"],
                "se_convention": convention}
    coef = np.linalg.solve(XX, X.T @ y_d)
    b = float(coef[1])
    u = y_d - X @ coef
    kc = kc0
    out = {"b": b, "n": int(len(y_d)), "k_all": kc["all"],
           "k_nested": kc["nested"]}
    gmin = np.nan
    for conv in SE_CONVENTIONS:
        se_vec, gmin = twoway_cluster_se(
            X, u, sub["c_code"].to_numpy(), sub["q_code"].to_numpy(),
            sub["ct"].to_numpy(), kc[conv])
        se = float(se_vec[1])
        out[f"se_{conv}"] = se
        out[f"t_{conv}"] = b / se if np.isfinite(se) and se > 0 else np.nan
    out["df"] = gmin - 1
    if convention not in SE_CONVENTIONS:
        raise SystemExit(f"unknown SE convention {convention!r}")
    out["se"] = out[f"se_{convention}"]
    out["t"] = out[f"t_{convention}"]
    out["se_convention"] = convention
    return out


def fit_from_raw(y, x2, x3, sub, convention: str = SE_CONVENTION_DEFAULT,
                 kconv: dict | None = None):
    """FULL FE re-estimation: demean all three vectors from raw, then solve.
    This is the ONLY route used on the circular-shift (arbiter) arm."""
    demean = make_demeaner([sub["cg"].to_numpy(), sub["ct"].to_numpy(),
                            sub["gt"].to_numpy()])
    return _solve(demean(y), demean(x2), demean(x3), sub, convention, kconv)


class Absorbed:
    """Draw-invariant projections + the exact linear basis for x3.

    B[:, t] = demean(x2 . 1[q == t]) so that demean(x2 . S[q]) = B @ S EXACTLY.
    Used only on the free-permutation cross-check arm, and only after
    selfcheck_basis_route has verified b and t against fit_from_raw.
    """

    def __init__(self, sub, y, x2, convention: str = SE_CONVENTION_DEFAULT,
                 kconv: dict | None = None):
        self.sub = sub
        self.convention = convention
        self.kconv = kconv if kconv is not None else k_conventions(sub)
        self.demean = make_demeaner([sub["cg"].to_numpy(), sub["ct"].to_numpy(),
                                     sub["gt"].to_numpy()])
        self.y_d = self.demean(np.asarray(y, float))
        self.x2 = np.asarray(x2, float)
        self.x2_d = self.demean(self.x2)
        self.q = sub["q_code"].to_numpy()
        self.nq = int(self.q.max()) + 1
        n = len(self.q)
        self.B = np.empty((n, self.nq))
        for t in range(self.nq):
            e = np.zeros(n)
            m = self.q == t
            e[m] = self.x2[m]
            self.B[:, t] = self.demean(e)

    def fit(self, S):
        """S is the DENSE per-measure shock vector: S[t] is the shock assigned
        to this sample's t-th surviving quarter."""
        S = np.asarray(S, float)
        if S.shape != (self.nq,):
            raise SystemExit(
                f"basis route got a shock vector of shape {S.shape}, expected "
                f"({self.nq},) — the calendar draw was not mapped onto this "
                "measure's own quarter index.")
        return _solve(self.y_d, self.x2_d, self.B @ S, self.sub,
                      self.convention, self.kconv)


# =============================================================================
# proof obligations
# =============================================================================
def _reldiff(a: float, b: float) -> float:
    """Relative difference, falling back to the absolute one when a == 0."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.inf
    if a == 0.0:
        return abs(b)
    return abs(b / a - 1.0)


def selfcheck_basis_route(absorbed: Absorbed, sub, y, x2, S_by_q, rng):
    """The basis identity must reproduce fit_from_raw's b AND t to BASIS_RELTOL
    on the observed draw and two random permutations, before any accelerated
    number is reported."""
    q = sub["q_code"].to_numpy()
    for i, S in enumerate([S_by_q, rng.permutation(S_by_q), rng.permutation(S_by_q)]):
        raw = fit_from_raw(y, x2, x2 * S[q], sub, absorbed.convention,
                           absorbed.kconv)
        fast = absorbed.fit(S)
        for key in ("b", "t"):
            rel = _reldiff(raw[key], fast[key])
            if not rel < BASIS_RELTOL:
                raise SystemExit(
                    f"BASIS-ROUTE SELF-CHECK FAILED on draw {i}, statistic "
                    f"{key}: from-raw {raw[key]!r} vs basis {fast[key]!r} "
                    f"(rel {rel!r} >= {BASIS_RELTOL:.0e}).  Refusing to report "
                    "accelerated free-permutation numbers.")
    return True


def selfcheck_difference_collapse(sub, S_by_q, b_full, measure):
    """Independent re-derivation of beta via the two-group difference collapse.
    Valid only on a balanced (c,t) grid; otherwise reported as unavailable."""
    us_rows = sub[sub["us"] == 1]
    non_rows = sub[sub["us"] == 0]
    if len(us_rows) == 0 or len(non_rows) == 0:
        return None, "unavailable (a holder group is absent from the sample)"
    key = ["c_code", "q_code"]
    u = us_rows.set_index(key)[["dw_global", "m_lag"]]
    v = non_rows.set_index(key)[["dw_global", "m_lag"]]
    common = u.index.intersection(v.index)
    if len(common) != len(u) or len(common) != len(v):
        return None, (f"unavailable (grid not balanced: {len(u)} US vs {len(v)} "
                      f"NONUS (c,t) cells, {len(common)} matched)")
    u = u.loc[common]
    v = v.loc[common]
    if not np.allclose(u["m_lag"].to_numpy(float), v["m_lag"].to_numpy(float),
                       rtol=0, atol=1e-12):
        return None, "unavailable (M_lag differs across groups within (c,t))"
    idx = common.to_frame(index=False)
    q = idx["q_code"].to_numpy()
    m = u["m_lag"].to_numpy(float)
    dy = (u["dw_global"].to_numpy(float) - v["dw_global"].to_numpy(float))
    demean = make_demeaner([idx["c_code"].to_numpy(), q])   # phi_c + delta_t
    X = np.column_stack([demean(m), demean(m * S_by_q[q])])
    coef = np.linalg.solve(X.T @ X, X.T @ demean(dy))
    b_col = float(coef[1])
    rel = abs(b_col / b_full - 1.0) if b_full != 0 else abs(b_col)
    if not rel < COLLAPSE_RELTOL:
        raise SystemExit(
            f"COLLAPSE CROSS-CHECK FAILED for {measure}: full-FE b={b_full:.6e} "
            f"vs difference-collapse b={b_col:.6e} (rel {rel:.2e} >= "
            f"{COLLAPSE_RELTOL:.0e}).  The two routes are algebraically identical "
            "on a balanced two-group grid, so a gap means the FE absorption or "
            "the sample construction is wrong.  Refusing to report.")
    return b_col, f"PASS (rel {rel:.2e})"


def load_stata_anchor():
    """Living drift anchor: the three MAIN rows of run_country_weight_ddd.do.

    b, se_crve, t AND N are all carried, because all four are hard-gated: the
    FWER statistic is max-|t|, so gating only b would leave the number that
    actually decides the verdict unchecked.
    """
    if not STATA_CSV.exists():
        raise SystemExit(
            f"{STATA_CSV.name} not found — run run_country_weight_ddd.do first; "
            "refusing to arbitrate without a living drift anchor.")
    a = pd.read_csv(STATA_CSV)
    need = {"family", "measure", "outcome", "spec", "b", "se_crve", "p_crve",
            "t", "N"}
    if not need.issubset(a.columns):
        raise SystemExit(
            f"{STATA_CSV.name} lacks {sorted(need - set(a.columns))} — stale "
            "layout; re-run run_country_weight_ddd.do.")
    a = a[(a["family"] == "main") & (a["outcome"] == "dw_global")
          & (a["spec"] == "ddd_3pairwise")]
    out = {}
    for m in MEASURES:
        r = a[a["measure"] == m]
        if len(r) != 1:
            raise SystemExit(
                f"{STATA_CSV.name}: no unique main/dw_global/ddd_3pairwise row "
                f"for {m} (found {len(r)}) — stale layout.")
        out[m] = {"b": float(r["b"].iloc[0]), "N": int(r["N"].iloc[0]),
                  "se": float(r["se_crve"].iloc[0]),
                  "t": float(r["t"].iloc[0]),
                  "p_crve": float(r["p_crve"].iloc[0])}
    return out


def adopt_se_convention(fit: dict, se_stata: float, measure: str) -> str:
    """Pick the small-sample convention that reproduces reghdfe's se_crve.

    Both candidates are computed; the one Stata matches is ADOPTED for t_obs and
    for every RI draw of this measure.  Matching NEITHER is a real defect (wrong
    sample, wrong regressor, wrong cluster dimensions), not a tolerance problem.
    """
    rels = {c: _reldiff(se_stata, fit[f"se_{c}"]) for c in SE_CONVENTIONS}
    ok = [c for c in SE_CONVENTIONS if rels[c] < SE_GATE_RELTOL]
    detail = "  ".join(
        f"se_{c}={fit[f'se_{c}']:.6e} (K={fit['k_' + c]:.0f}, rel={rels[c]:.2e})"
        for c in SE_CONVENTIONS)
    if not ok:
        raise SystemExit(
            f"SE GATE FAILED for {measure}: Stata se_crve={se_stata:.6e} matches "
            f"NEITHER small-sample convention at {SE_GATE_RELTOL:.0e}.  {detail}. "
            "A gap of this size is not a dof convention — it is a different "
            "sample, a different regressor or different cluster dimensions. "
            "Refusing to arbitrate with a t on an unverified scale.")
    if len(ok) == 2:
        # only possible if the two K's coincide, i.e. no FE group was nested
        ok = [min(SE_CONVENTIONS, key=lambda c: rels[c])]
    return ok[0]


# =============================================================================
# main
# =============================================================================
def main():
    rotation_guard(RESULT_CSV)
    t_start = time.time()

    print("=" * 92)
    print(f"run_ri_country_ddd.py  N_PERM={N_PERM}  SEED={SEED}  "
          f"demean tol={DEMEAN_TOL:.0e}  cert={CERT_TOL:.0e}")
    print("ARBITER = circular shift, FULL FE re-estimation per draw, "
          "max-|t| FWER over {M1, M2, M3}")
    print("=" * 92)
    prov = log_provenance({"panel_sha256": PANEL_DTA,
                           "anchor_sha256": STATA_CSV})

    anchor = load_stata_anchor()
    df = load_country_panel()
    S_full = df.attrs["S_by_quarter"]
    quarters = df.attrs["quarters"]
    # NOTE (fixed 2026-08-10): S_{t-1} is missing for the panel's FIRST quarter by
    # construction — there is no t-1 before it — and that quarter can never enter
    # any regression anyway (dw_global and every m*_lag are missing there too).
    # An earlier vintage checked np.isfinite(S_full).all() over the FULL grid and
    # aborted unconditionally. The check now lives inside the per-measure loop,
    # AFTER the sample restriction, where it means what it should: abort only if a
    # SURVIVING quarter has no shock.
    n_s_missing = int((~np.isfinite(S_full)).sum())
    if n_s_missing:
        miss_q = [str(quarters[i]) for i in np.flatnonzero(~np.isfinite(S_full))]
        print(f"note: S_{{t-1}} missing on {n_s_missing} grid quarter(s) "
              f"{miss_q} — checked again per measure on the SURVIVING quarters.")
    print(f"panel: {len(df):,} rows | {df['sec_country'].nunique()} countries | "
          f"{len(quarters)} quarters ({quarters[0]} -> {quarters[-1]})")

    # ---- THE RANDOMISATION DEVICE: ONE CALENDAR ----------------------------
    # `cal' is the shock calendar — the grid quarters on which S_{t-1} exists.
    # It is the ONLY thing that gets rotated or permuted, ONCE per replication,
    # and every measure then reads the drawn calendar at its OWN quarters.
    cal = np.flatnonzero(np.isfinite(S_full))
    S_cal = S_full[cal]
    n_cal = int(len(cal))
    if n_cal < 8:
        raise SystemExit(
            f"only {n_cal} quarters carry S_{{t-1}} — a circular shift over so "
            "few quarters has no resolution; refusing to arbitrate.")
    n_rot = n_cal - 1
    print(f"randomisation device: the CALENDAR — {n_cal} quarters carry "
          f"S_{{t-1}} ({quarters[cal[0]]} -> {quarters[cal[-1]]}), "
          f"{n_rot} non-identity rotations")

    def draw_calendar(values: np.ndarray) -> np.ndarray:
        """Scatter one drawn shock vector back onto the full calendar.

        Returns a length-len(quarters) vector, NaN off the shock calendar, so a
        measure reading it at a quarter without a shock would produce NaN rather
        than a silently wrong number.
        """
        full = np.full(len(S_full), np.nan)
        full[cal] = values
        return full

    # ---- per-measure setup, observed statistics, proof obligations ----------
    setups = {}
    q_keep_ref = None
    ref_measure = None
    for m in MEASURES:
        sub = estimation_sample(df, m)
        # q_keep = this measure's ORIGINAL (calendar) quarter indices. The dense
        # code q_code is its position inside q_keep, which is how a drawn
        # calendar is mapped onto this measure: S_draw_dense = drawn[q_keep].
        q_keep = np.sort(sub["q_orig"].unique())
        S = S_full[q_keep]
        nq = len(S)
        if not np.isfinite(S).all():
            bad = [str(quarters[q_keep[i]])
                   for i in np.flatnonzero(~np.isfinite(S))]
            raise SystemExit(
                f"[{m}] S_{{t-1}} is missing on SURVIVING quarter(s) {bad} — a "
                "rotation would mix missing values into the design, and the "
                "observed fit would already be NaN; refusing to arbitrate.")
        if nq != int(sub["q_code"].max()) + 1:
            raise SystemExit("quarter re-coding mismatch — aborting.")

        # HARD GATE (external review item 6): the three measures must survive on
        # the SAME quarters. Under the calendar device the max-|t| draws are
        # already common, but a differing quarter set means the three statistics
        # describe three different estimation windows, and a single-step FWER
        # over three different windows is not the object it claims to be. The
        # old code only printed a NOTE here.
        if q_keep_ref is None:
            q_keep_ref = q_keep
            ref_measure = m
        elif not np.array_equal(q_keep, q_keep_ref):
            only_here = [str(quarters[i]) for i in np.setdiff1d(q_keep, q_keep_ref)]
            only_ref = [str(quarters[i]) for i in np.setdiff1d(q_keep_ref, q_keep)]
            raise SystemExit(
                f"SAMPLE-QUARTER GATE FAILED: {m} survives on "
                f"{len(q_keep)} quarters but {ref_measure} on "
                f"{len(q_keep_ref)}.  Only in {m}: {only_here}.  Only in "
                f"{ref_measure}: {only_ref}.  The max-|t| FWER is a statement "
                "about three hypotheses measured over ONE window; with "
                "different windows it is not.  Fix upstream (the M lags come "
                "from the same country_measures build) or re-declare the "
                "family — this script will not paper over it.")
        if not (np.diff(q_keep) == 1).all():
            print(f"  [{m}] NOTE: the surviving quarter set is not contiguous "
                  f"({len(q_keep)} of {n_cal} shock quarters). The device is "
                  "still the CALENDAR, so the rotation remains a calendar "
                  "rotation; some drawn values simply land outside the sample. "
                  "Disclosed, not hidden.")
        y = sub["dw_global"].to_numpy(float)
        x2 = sub["us"].to_numpy(float) * sub["m_lag"].to_numpy(float)
        q = sub["q_code"].to_numpy()

        # observed fit, both SE conventions; the one Stata matches is ADOPTED
        kconv = k_conventions(sub)
        fit0 = fit_from_raw(y, x2, x2 * S[q], sub, SE_CONVENTION_DEFAULT, kconv)
        conv = adopt_se_convention(fit0, anchor[m]["se"], m)
        fit = dict(fit0)
        fit["se"], fit["t"] = fit[f"se_{conv}"], fit[f"t_{conv}"]
        fit["se_convention"] = conv
        b_obs, t_obs = fit["b"], fit["t"]

        b_st, se_st, t_st, n_st = (anchor[m]["b"], anchor[m]["se"],
                                   anchor[m]["t"], anchor[m]["N"])
        rel_b = _reldiff(b_st, b_obs)
        rel_se = _reldiff(se_st, fit["se"])
        rel_t = _reldiff(t_st, t_obs)
        print(f"\n[{m}] N={fit['n']:,} (Stata N={n_st:,})  nq={nq}  "
              f"b={b_obs:+.6e} (Stata {b_st:+.6e}, rel {rel_b:.2e})  "
              f"se={fit['se']:.6e} (Stata {se_st:.6e}, rel {rel_se:.2e})  "
              f"t={t_obs:+.4f} (Stata {t_st:+.4f}, rel {rel_t:.2e})")
        print(f"     SE convention ADOPTED: {conv} "
              f"(K_all={kconv['all']:.0f}, K_nested={kconv['nested']:.0f}, "
              f"nested={kconv['nested_flags']}) — used for t_obs AND every draw")
        # ---- the four hard gates against the Stata leg ---------------------
        if not rel_b < B_GATE_RELTOL:
            raise SystemExit(
                f"DRIFT GATE FAILED for {m}: python b={b_obs:.6e} vs canonical "
                f"{b_st:.6e} (rel {rel_b:.2e} >= {B_GATE_RELTOL:.0e}). Stale panel, "
                "stale CSV, or a specification mismatch — refusing to arbitrate.")
        if fit["n"] != n_st:
            raise SystemExit(
                f"N GATE FAILED for {m}: python N={fit['n']:,} vs Stata "
                f"N={n_st:,}.  The two legs are not estimating on the same rows, "
                "so neither the coefficient agreement nor the t is meaningful. "
                "Listwise deletion differs — check the panel's missingness.")
        if not rel_t < B_GATE_RELTOL:
            raise SystemExit(
                f"t GATE FAILED for {m}: python t={t_obs:.6f} vs Stata "
                f"t={t_st:.6f} (rel {rel_t:.2e} >= {B_GATE_RELTOL:.0e}) even "
                f"though b and se agree.  max-|t| RANKS this statistic, so an "
                "unexplained t is fatal, not cosmetic.")

        _, coll_status = selfcheck_difference_collapse(sub, S, b_obs, m)
        print(f"     difference-collapse cross-check: {coll_status}")

        absorbed = Absorbed(sub, y, x2, conv, kconv)
        selfcheck_basis_route(absorbed, sub, y, x2, S,
                              np.random.default_rng(SEED + 7))
        print(f"     basis-route self-check (b and t, 3 draws): PASS "
              f"(< {BASIS_RELTOL:.0e})")

        setups[m] = {"sub": sub, "y": y, "x2": x2, "q": q, "S": S, "nq": nq,
                     "q_keep": q_keep, "absorbed": absorbed, "kconv": kconv,
                     "conv": conv, "b_obs": b_obs, "t_obs": t_obs,
                     "se_obs": fit["se"], "n": fit["n"]}

    # ---- CIRCULAR SHIFT: full FE re-estimation, no caching, no collapse -----
    # ONE rotation of the CALENDAR per replication; the three measures read that
    # same rotated calendar at their own quarters.  This is what makes "rotation
    # r" a single draw of the randomisation device and the single-step max-|t|
    # FWER a valid one.
    print(f"\ncircular shift: {n_rot} calendar rotations x {len(MEASURES)} "
          "measures, ONE calendar draw per rotation, FULL FE re-estimation")
    t_circ = {m: np.full(n_rot, np.nan) for m in MEASURES}
    for i, r in enumerate(range(1, n_rot + 1)):
        drawn = draw_calendar(np.roll(S_cal, r))
        for m in MEASURES:
            s = setups[m]
            S_r = drawn[s["q_keep"]]          # map onto THIS measure's quarters
            f = fit_from_raw(s["y"], s["x2"], s["x2"] * S_r[s["q"]], s["sub"],
                             s["conv"], s["kconv"])
            t_circ[m][i] = f["t"]
        if (i + 1) % 20 == 0:
            print(f"    ... {i+1}/{n_rot} rotations "
                  f"({time.time()-t_start:.0f}s)")

    # ---- FREE PERMUTATION: exact basis route (self-checked above) -----------
    # Same device, same discipline: ONE permutation of the calendar per draw,
    # read by all three measures.  The old `perm[perm < nq_k]' produced a
    # DIFFERENT permutation per measure out of one parent vector.
    print(f"free permutation: {N_PERM} draws x {len(MEASURES)} measures "
          "(exact linear-basis route; equality to the from-raw route verified "
          "above on b and t)")
    rng = np.random.default_rng(SEED)
    t_free = {m: np.full(N_PERM, np.nan) for m in MEASURES}
    for d in range(N_PERM):
        drawn = draw_calendar(S_cal[rng.permutation(n_cal)])
        for m in MEASURES:
            s = setups[m]
            t_free[m][d] = s["absorbed"].fit(drawn[s["q_keep"]])["t"]
        if (d + 1) % 1000 == 0:
            print(f"    ... {d+1}/{N_PERM} permutations "
                  f"({time.time()-t_start:.0f}s)")

    # ---- p-values, incl. single-step max-|t| FWER --------------------------
    # A DRAW IS ALL-OR-NOTHING.  A two-way cluster meat is not guaranteed PSD in
    # finite samples, so an individual draw can return a NaN SE (and therefore a
    # NaN t) for ONE measure.  Taking a nan-aware max over the row would then
    # compute "max over three statistics" from two of them, which biases the FWER
    # null DOWNWARD and makes p_fwer anti-conservative.  It would also give the
    # three measures different draw sets, undoing the whole point of drawing the
    # calendar once.  So a draw in which ANY measure fails is dropped from ALL
    # of them, and every p below — per-measure and FWER — is computed on the SAME
    # set of complete draws.  The drop count is reported, and a large one aborts:
    # a null distribution built from a materially selected subset of rotations is
    # not the null the arbiter claims to use.
    def _complete(tdict, n):
        ok = np.ones(n, dtype=bool)
        for m in MEASURES:
            ok &= np.isfinite(tdict[m])
        return ok

    ok_circ = _complete(t_circ, n_rot)
    ok_free = _complete(t_free, N_PERM)
    n_bad_circ, n_bad_free = int((~ok_circ).sum()), int((~ok_free).sum())
    if n_bad_circ or n_bad_free:
        print(f"\nINCOMPLETE DRAWS DROPPED (a NaN two-way SE on at least one "
              f"measure): circular {n_bad_circ}/{n_rot}, free {n_bad_free}/"
              f"{N_PERM}.  Dropped from ALL measures and from max-|t|, so every "
              "p below rests on the same complete draws.")
    for lbl, n_bad, n_tot in (("circular", n_bad_circ, n_rot),
                              ("free", n_bad_free, N_PERM)):
        if n_bad > 0.10 * n_tot:
            raise SystemExit(
                f"{n_bad} of {n_tot} {lbl} draws returned a non-finite t on some "
                "measure (>10%).  The surviving draws are a SELECTED subset of "
                "the randomisation distribution, so the p-values would not be "
                "the test they are labelled as.  This is a small-sample vcov "
                "problem (a two-way cluster meat that is not PSD), not a bug to "
                "tolerate: reduce the cluster dimensions or change the "
                "inference, and re-run.")
    if int(ok_circ.sum()) < 8:
        raise SystemExit(
            f"only {int(ok_circ.sum())} complete rotations survive — no usable "
            "randomisation distribution; refusing to report p_circ.")

    absmat_circ = np.column_stack([np.abs(t_circ[m]) for m in MEASURES])
    absmat_free = np.column_stack([np.abs(t_free[m]) for m in MEASURES])
    maxt_circ = absmat_circ[ok_circ].max(axis=1)
    maxt_free = absmat_free[ok_free].max(axis=1)
    n_draw_circ, n_draw_free = int(ok_circ.sum()), int(ok_free.sum())

    rows = []
    for m in MEASURES:
        s = setups[m]
        tabs = abs(s["t_obs"])
        tc = np.abs(t_circ[m])[ok_circ]
        tf = np.abs(t_free[m])[ok_free]
        p_circ = (int((tc >= tabs - 1e-300).sum()) + 1) / (n_draw_circ + 1)
        p_free = (int((tf >= tabs - 1e-300).sum()) + 1) / (n_draw_free + 1)
        p_fwer = (int((maxt_circ >= tabs - 1e-300).sum()) + 1) / (n_draw_circ + 1)
        p_fwer_free = (int((maxt_free >= tabs - 1e-300).sum()) + 1) / (n_draw_free + 1)
        rows.append({
            "measure": m,
            "b_obs": s["b_obs"],
            "t_obs": s["t_obs"],
            "p_circ": p_circ,
            "p_free": p_free,
            "p_fwer_max_t": p_fwer,
            # COMMON to all three measures by construction: incomplete draws are
            # dropped from every measure, not just from the one that failed.
            "n_rotations": n_draw_circ,
            "n_rotations_attempted": n_rot,
            "n_rotations_incomplete": n_bad_circ,
            "p_fwer_max_t_free": p_fwer_free,
            "n_perm": n_draw_free,
            "n_perm_incomplete": n_bad_free,
            "se_obs_2way": s["se_obs"],
            "N": s["n"],
            "n_countries": int(pd.Series(s["sub"]["c_code"]).nunique()),
            "n_quarters": s["nq"],
            "n_calendar_quarters": n_cal,
            "b_stata_anchor": anchor[m]["b"],
            "se_stata_anchor": anchor[m]["se"],
            "t_stata_anchor": anchor[m]["t"],
            "N_stata_anchor": anchor[m]["N"],
            "p_crve_stata": anchor[m]["p_crve"],
            "se_convention": s["conv"],
            "k_small_sample": s["kconv"][s["conv"]],
            "fe": "cg + ct + gt (country analogue of fq/gq/ig)",
            "vce": "two-way cluster (country, quarter)",
            "circ_estimation": "full FE re-estimation from raw per draw",
            "free_estimation": "exact linear basis, self-checked vs from-raw on b and t",
            "draw_device": ("ONE rotation/permutation of the shock CALENDAR per "
                            "replication, read by all three measures at their "
                            "own quarter index (sample-quarter sets hard-gated "
                            "identical)"),
            "anchor_gates": ("b, N, se and t all hard-gated against "
                             "country_weight_ddd.csv"),
            "seed": SEED,
            "assumption": CAVEAT,
            "script_sha256": prov["script_sha256"],
            "panel_sha256": prov["panel_sha256"],
            "anchor_sha256": prov["anchor_sha256"],
        })

    res = pd.DataFrame(rows)
    lead = ["measure", "b_obs", "t_obs", "p_circ", "p_free", "p_fwer_max_t",
            "n_rotations"]
    res = res[lead + [c for c in res.columns if c not in lead]]
    res.to_csv(RESULT_CSV, index=False)

    print("\n" + "=" * 92)
    print(res[lead].to_string(index=False))
    print("=" * 92)
    print("EXCHANGEABILITY / STATIONARITY CAVEAT (also in the CSV's `assumption'"
          " column):")
    print("  The circular shift assumes S_{t-1} is exchangeable / approximately")
    print("  stationary under calendar rotation.  Rotation PRESERVES the shock's")
    print("  serial correlation, which free permutation destroys — that is why")
    print("  p_circ, not p_free, is the arbiter for a serially correlated shock.")
    print("  Invariance to rotation is still an ASSUMPTION, and the US-China")
    print("  series has a level regime shift, so p_circ is serial-ROBUST, not")
    print("  exact.")
    print(f"  Granularity floor: no p_circ can fall below 1/(n_rotations+1) = "
          f"{1.0/(n_draw_circ+1):.4f} ({n_draw_circ} complete rotations of "
          f"{n_rot} attempted).")
    print("  p_fwer_max_t is single-step max-|t| across the THREE main")
    print("  hypotheses {M1, M2, M3} on a COMMON draw sequence.  |t| not |b|:")
    print("  the measures are on different scales, so max-|b| would")
    print("  mechanically select the largest-scaled one.")
    print("  THE DRAW IS THE CALENDAR: one rotation (or permutation) of the")
    print("  shock calendar per replication, read by all three measures at")
    print("  their own quarter index, with the three sample-quarter sets")
    print("  hard-gated identical.  That is what makes a single-step max-|t|")
    print("  FWER over the three statistics well defined.")
    print(f"  Anchor gates passed: b, N, se and t vs {STATA_CSV.name} "
          f"(se convention adopted per measure: "
          f"{ {m: setups[m]['conv'] for m in MEASURES} }).")
    print(f"\nwrote {RESULT_CSV.name}  ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
