"""Descriptive fact-find (read-only): is robots_spec a mixed construct?
For robots_spec CANON firms, split onset into:
  genuine   = first ANY block coincides with the specific-rule block (any_fb == spec_fb)
              -> bot was fully allowed before, named rule = real access flip
  naming    = firm already blocked the bot via wildcard/nonroot before the named rule
              (any_fb < spec_fb) -> adding the named rule did not change 'blocked-at-all' status
Also: of robots_spec CLEAN_NEVER controls, how many restrict the bots via wildcard/nonroot anyway.
NOTE: firm-level cohort (NOT in_sample-filtered); the in-sample (n=217) split comes with the run.
"""
import pandas as pd

P = ("C:/Users/xl/OneDrive - Universitat Ramón Llull/"
     "bot_paper/panel_construction/firm_robots_treatment.csv")
df = pd.read_csv(P)

def num(c):
    return pd.to_numeric(df[c], errors="coerce")

spec_canon = num("robots_spec_canon_treated") == 1
spec_fb = num("robots_spec_first_block")
any_fb  = num("robots_any_first_block")
root_fb = num("robots_root_first_block")

t = spec_canon
n = int(t.sum())
genuine = int((t & (any_fb == spec_fb)).sum())
naming  = int((t & (any_fb <  spec_fb)).sum())
anomaly = int((t & (any_fb >  spec_fb)).sum())   # should be 0 (bk_spec implies bk_any)
nanany  = int((t & any_fb.isna()).sum())          # should be 0

print(f"=== robots_spec CANON (firm-level cohort) N = {n} ===")
print(f"  genuine open->named (any_fb == spec_fb)        : {genuine:5d}  ({100*genuine/n:5.1f}%)")
print(f"  naming-only (already blocked, any_fb < spec_fb): {naming:5d}  ({100*naming/n:5.1f}%)")
print(f"  [anomaly any_fb>spec_fb = {anomaly}; any_fb NaN = {nanany}]")

# Of the naming-only firms, how many were already ROOT-locked (full wildcard lockdown) pre-onset?
nm = t & (any_fb < spec_fb)
root_pre = int((nm & root_fb.notna() & (root_fb < spec_fb)).sum())
print(f"  of naming-only, already ROOT-locked before named onset (root_fb<spec_fb): "
      f"{root_pre:5d}  ({100*root_pre/max(naming,1):5.1f}% of naming-only)")

# Controls: robots_spec clean_never, but do they restrict the bots via wildcard/nonroot?
spec_clean = num("robots_spec_clean_never") == 1
cn = int(spec_clean.sum())
c_anyfb = any_fb
restricting = int((spec_clean & c_anyfb.notna()).sum())
print(f"\n=== robots_spec CLEAN_NEVER controls N = {cn} ===")
print(f"  also restrict bots via wildcard/nonroot (robots_any blocked ever): "
      f"{restricting:5d}  ({100*restricting/cn:5.1f}%)")
