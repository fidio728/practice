# build_fig_a_chained.py
# ===========================================================================
# Composition control for Figure A, rebuilt WITHOUT a freeze date (2026-08-21).
#
# The old control froze group membership at BASE_QUARTER (2017Q4). A sweep over
# all 52 candidate freeze quarters showed the residual gap it leaves is not
# sign-stable (-19.5 to +18.4 index points, positive in 24 of 52), so any
# statement about the residual was a function of the chosen date.
#
# This replacement is a CHAIN-LINKED MATCHED-SAMPLE index and involves no date:
# for each quarter t and each arm, growth is computed only over firms present
# in t-1 and t AND classified in the same arm at both dates; the quarterly
# growth rates are then chained. A firm whose links enter the data, or that
# switches arms, contributes nothing in the quarter it moves, so roster changes
# cannot move this index by construction. BASE_QUARTER survives only as the
# quarter where both lines are drawn at 100, which is presentation, not
# construction.
#
# Standalone on purpose: build_fig_ab_data.py refuses to overwrite its
# canonical CSVs in place (rotation rule), and this script adds a new artifact
# rather than changing an existing one.
#
# Output: output/fig_A_firm_chained.csv
#   bucket        zero | positive (classification on cn_lag, as everywhere)
#   quarter_end
#   growth        matched same-arm quarterly growth factor (real, CPI-deflated)
#   chained_idx   cumulative product, rebased to BASE_QUARTER = 100
#   matched_firms firms entering that quarter's growth computation
# ===========================================================================

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ))
_spec = importlib.util.spec_from_file_location("bf", PROJ / "build_fig_ab_data.py")
bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bf)

BASE = bf.BASE_QUARTER
OUT_CSV = PROJ / "output" / "fig_A_firm_chained.csv"


def main() -> None:
    con = duckdb.connect()
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    tmp = Path("E:/duckdb_tmp")
    if tmp.parent.exists():
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{tmp.as_posix()}'")

    df = bf.build_firm_quarter_table(con)
    cpi = bf.load_cpi()
    c = cpi.set_index("DATE")["CPIAUCNS"]
    q = c.groupby(c.index.to_period("Q").to_timestamp("Q")).last()
    defl = (q.loc[BASE] / q).to_dict()

    d = df.loc[df["cn_lag"].notna()].copy()
    d["bucket"] = np.where(d["cn_lag"] > 0, "positive", "zero")
    d["real"] = d["usd_us"].fillna(0.0) * d["quarter_end"].map(defl)
    d = d.sort_values(["fid", "quarter_end"], kind="mergesort")

    d["bucket_prev"] = d.groupby("fid")["bucket"].shift(1)
    d["real_prev"] = d.groupby("fid")["real"].shift(1)
    d["q_prev"] = d.groupby("fid")["quarter_end"].shift(1)
    step = d.loc[d["bucket_prev"].eq(d["bucket"])
                 & d["q_prev"].notna()
                 & (d["quarter_end"] - d["q_prev"]).le(pd.Timedelta(days=100))]

    g = (step.groupby(["bucket", "quarter_end"], as_index=False)
             .agg(now=("real", "sum"), prev=("real_prev", "sum"),
                  matched_firms=("fid", "size")))
    g["growth"] = g["now"] / g["prev"].replace(0.0, np.nan)
    g = g.sort_values(["bucket", "quarter_end"], kind="mergesort")
    g["chain_raw"] = g.groupby("bucket")["growth"].cumprod()

    base = (g.loc[g["quarter_end"].eq(BASE), ["bucket", "chain_raw"]]
              .rename(columns={"chain_raw": "b"}))
    if len(base) != 2 or base["b"].isna().any():
        raise RuntimeError(f"chain undefined at BASE_QUARTER {BASE.date()}")
    g = g.merge(base, on="bucket", validate="many_to_one")
    g["chained_idx"] = 100.0 * g["chain_raw"] / g["b"]
    if not np.isfinite(g["chained_idx"]).all():
        bad = g.loc[~np.isfinite(g["chained_idx"]), "quarter_end"]
        raise RuntimeError(f"non-finite chain at {sorted(bad.dt.date.unique())[:4]}")

    out = g[["bucket", "quarter_end", "growth", "chained_idx", "matched_firms"]]
    out.to_csv(OUT_CSV, index=False)

    end = out.loc[out["quarter_end"].eq(out["quarter_end"].max())].set_index("bucket")
    print(f"wrote {OUT_CSV.name} ({len(out):,} rows, "
          f"{out['quarter_end'].min().date()}..{out['quarter_end'].max().date()})")
    for arm in ("zero", "positive"):
        print(f"  {arm:9s} end idx {end.loc[arm, 'chained_idx']:6.1f}  "
              f"matched {int(end.loc[arm, 'matched_firms']):,}")
    print(f"  end gap (zero - positive): "
          f"{end.loc['zero', 'chained_idx'] - end.loc['positive', 'chained_idx']:+.1f}")


if __name__ == "__main__":
    main()
