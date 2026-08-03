# GLOBAL CONSISTENCY AUDIT (2026-08-03)

4视角(dataflow/doc/spec/design)+综合。级联失同步全面盘点。

## STALENESS MATRIX

STALENESS MATRIX (B7 grid = merged_us_eu_zero_filled.parquet 2026-08-02; headline +2.75e-6 / N=347,490 / 6,854 firms)

产物/组件 | 最后更新 | 依赖版本 | 状态 | 消费者
--------------------------------------------------------------------------------------------------
audit_c6_panel.dta | 07-02 | parquet 08-02 (BOTH pre-B7 & pre-dir-split) | STALE | run_headline_3pairwise.do:24 (CANONICAL headline), run_ri_3pairwise.py:33, run_audit_f1f2f7.do:12, run_audit_f4f8.do, run_audit_f1b_robust.do, run_randomization_inference.py — emits OLD +1.80e-6/N=462,096
shocklag_panel.dta | 07-02 | parquet 08-02 | STALE (no gate) | run_shocklag.do, run_ri_shocklag.py (§7.3b)
c6_panel_riskset.dta | 07-02 | parquet 08-02 | STALE (no gate) | 07g_spell_riskset.do (§7.4, N=342,262)
c6_panel_riskset_lagonly.dta | 07-02 | c6_panel_riskset.dta (itself stale) | STALE (DOUBLE — inherits #riskset) | run_audit_f4f8.do (F8 β₃=+2.6e-6)
sagg_panel.dta | 07-02 | parquet 08-02 | STALE (ORPHAN; NEG β₃ contradicts "all-positive") | run_sagg_distlag.do, run_ri_sagg.py
c6_panel_russia.dta / merged_us_ru | 07-02 | 02_russia all-rel-type numerator | STALE+SUPERSEDED-numerator | run_russia_headline.do, run_ri_russia.py:30, run_ri_russia_3pairwise.py:21, build_russia_lp_panel.py
07d_three_spec_results.txt | 06-09 | c6_panel.dta current (.do OK, artifact frozen) | STALE (artifact only) | §7.1 (N=462,564)
c6_panel.dta | 08-02 | parquet 08-02 | CURRENT | run_tercile_3pairwise.do, run_direction_split.do, run_tail_3pairwise.do:9
firm_ladder_panel.dta | 07-22 | parquet 08-02 (additive-only, byte-verified) | VALUE-EQUIV | run_firm_ladder.do, run_country_ladder.do, run_ladder_2x2_cells.do, run_ddd_nofe_bil.do:17
country_panel.dta | 07-22 | parquet 08-02 (additive) | VALUE-EQUIV | run_country_ladder.do
ownership_c6_panel.dta | 07-22 | parquet 08-02 (additive ASSERTED, not bit-verified) | VALUE-EQUIV(pending cn_lag/shock bit-check) | run_ownership_share.do, run_flow_heldonly.do, run_flow_winsor.do, run_ri_flow.py
tercile/direction/fourgroup panels | 08-02/08-03 | c6_panel.dta current | CURRENT | run_tercile/run_direction/run_fourgroup (+ RI)
run_ddd_nofe_bil.do Stage B/C | frozen | gate hardcodes N=462096/1.80e-6 → HALTS | QUEUED (fail-safe, but no-FE/bilateral cols never regen on B7) | slide add-ons
run_tail_3pairwise.do (role) | — | reads CURRENT c6 (data OK) | CURRENT-data / SUPERSEDED-role by tercile | §7.3
Essay2_methodology_full.md | partial | mixed pre/post-B7 | STALE (partial-refresh = more dangerous than untouched) | paper

## FIXES (ranked)

### 1. [MUST-FIX] Rebuild audit_c6_panel.dta (linchpin) + re-run its 6 consumers
**Action**: Run build_audit_panel_f1f2f7.py against the B7 parquet (already fresh), then re-run run_headline_3pairwise.do, run_ri_3pairwise.py, run_audit_f1f2f7.do, run_audit_f4f8.do, run_audit_f1b_robust.do, run_randomization_inference.py. Confirm regenerated 3-pairwise = +2.75e-6 / N=347,490. Also align run_randomization_inference.py seed 12345 -> 20260702 and refresh its hardcoded Stata constants + run_ri_3pairwise.py:100 anchors (+1.800e-6/+2.373e-6/+6.621e-6).
**Blocking**: BLOCKS doc refresh (§7.1/§7.2/§13/§15) and every headline+F1/F2/F7+LP RI p; run_ddd_nofe_bil gate update (fix 9) and canonical-artifact (fix 6) depend on this.

### 2. [MUST-FIX] Rebuild spell-riskset chain IN ORDER (fixes double-stale F8)
**Action**: build_spell_riskset.py -> 07g_spell_riskset.do FIRST; THEN build_riskset_lagonly.py -> run_audit_f4f8.do. Verify firm count/N moves off 7,928/342,262 to the B7 universe before citing §7.4/§F8.
**Blocking**: riskset_lagonly inherits the stale riskset — must not run lagonly before the base rebuild. Blocks §7.4 doc numbers.

### 3. [MUST-FIX] Rebuild shock-lag panel (§7.3b)
**Action**: build_shocklag_panel.py -> run_shocklag.do -> run_ri_shocklag.py on B7. Confirm N drops to the 347k family before updating §7.3b; keep sigma_S=2.578 (shock-only, B7-invariant).
**Blocking**: No gate exists (unlike run_ddd) so a bare re-run silently yields pre-B7 numbers pasted next to the 347k headline. Blocks §7.3b.

### 4. [MUST-FIX] Re-run sagg AND immediately scope the false 'every estimate positive' claim
**Action**: Re-run build_sagg_panel.py -> run_sagg_distlag.do -> run_ri_sagg.py on B7; adjudicate the near-marginal NEG β₃ by RI. UNTIL then, carve the aggregated-shock spec out of the universal 'all point estimates positive' sentences (§0 line5, §D line41, line67, §13 line560) — as written the claim is FALSE (sagg CRVE p=0.086 negative exists).
**Blocking**: Correctness: a live load-bearing doc claim is currently false. Do the carve-out NOW; the re-run gates whether the spec integrates or stays retired.

### 5. [SHOULD-FIX] Russia rel_type rebuild + RI regen (positive control)
**Action**: Apply B7 CUSTOMER+SUPPLIER filter to 02_russia_exposure.jl (line 571 uses all-rel-type COUNT(*)); rebuild 06_russia -> build_russia_c6_panel -> merged_us_ru; re-run run_russia_headline / run_ri_russia / build_russia_lp_panel. Add read_dta->to_parquet regen at top of run_ri_russia.py:30 and run_ri_russia_3pairwise.py:21 (mirror run_ri_3pairwise) so the rebuild propagates. Update their hardcoded refs (b3=-7.3265e-6, p=0.0853).
**Blocking**: BLOCKS citing Russia LP h=0 perm_p=0.040 as design validation. Bundle with B9 VCE work.

### 6. [SHOULD-FIX] Produce a canonical living B7 headline artifact
**Action**: Emit and store a canonical 3-pairwise headline (+2.75e-6/347,490) from c6_panel.dta or the refreshed audit panel; repoint tercile/direction/fourgroup cross-references (run_fourgroup.do:39-40 etc.) at that artifact instead of a hardcoded constant.
**Blocking**: Depends on fix 1. The +2.75e-6 anchor currently has no living producer; tercile/direction/fourgroup compare against an un-reproduced constant.

### 7. [SHOULD-FIX] Add B9 degenerate-VCE guard to flow specs
**Action**: Insert the standard B9 block (missing/zero-SE detection + *_vce_diag.csv + loud warning pointing to run_ri_flow.py) into run_ownership_share.do, run_flow_heldonly.do, run_flow_winsor.do. These are the exact specs where run_ri_flow.py documents the two-way CRVE is degenerate, yet R1 writes a missing/zero SE with no flag.
**Blocking**: Non-blocking but protects flow SE integrity — the precise failure B9 was built to prevent.

### 8. [SHOULD-FIX] Verify ownership flow panel vintage or rebuild
**Action**: Confirm china_share_lag1q and shock_us_cn are bit-identical between the Jul-22 and Aug-2 parquet grids; if not, regenerate ownership_c6_panel.dta from the current parquet before citing flow numbers or building run_flow_decomposition.py.
**Blocking**: BLOCKS flow-decomposition build and flow citation. Flow point est is B7-panel but the additive-preservation is asserted, not bit-verified.

### 9. [SHOULD-FIX] Update run_ddd_nofe_bil.do gate, then re-run Stage B/C
**Action**: After the canonical B7 headline lands, change the Stage-A gate (line 42) target b3->+2.75e-6 and e(N)->347,952 (or post-singleton N) so Stage B (no-FE full-factorial) and Stage C (bilateral footnote) regenerate on B7.
**Blocking**: Depends on fixes 1/6. Until updated the gate fail-safe-halts and silently freezes the no-FE and bilateral doc columns pre-B7.

### 10. [SHOULD-FIX] Refresh 07d artifact + hardcoded header constants
**Action**: Re-run 07d_three_spec_table.do to refresh 07d_three_spec_results.txt; update N=462,564 / β₃=+1.28 references baked into headers of 07d:16,21, 07e_firmgroup_tail.do:9, 07f_spell_boundary.do:7, 07g_spell_riskset.do:7 to the B7 headline (N~347,490).
**Blocking**: None.

### 11. [SHOULD-FIX] Fix stale ownership reference constants
**Action**: verify_ownership_regression.do:77 display 300,866/5,704 -> 240,246/6,854; build_ownership_share_panel.py:195 ref_main_panel_firms 7928 -> 6,854; regenerate ownership_share_diagnostics.csv from the B7 panel.
**Blocking**: None (panel itself is B7-current; only baked refs are stale).

### 12. [SHOULD-FIX] Mark tail menu SUPERSEDED + reword cross-refs
**Action**: Add 'SUPERSEDED by run_tercile_3pairwise.do (advisor 2026-08-02); retained for provenance/few-cluster-invalidity (F3) only' banner to run_tail_3pairwise.do and 07e_firmgroup_tail.do; reword run_tercile_3pairwise.do:3 and run_ri_tercile.py:4/7/23 'isomorphic to / mirrors run_tail_3pairwise.do' to 'the retired tail menu'. In doc §7.3/§13 mark tail rows superseded.
**Blocking**: None (tail .do reads CURRENT c6 — not data-stale, positioning only).

### 13. [SHOULD-FIX] Add the three new B7 components + estimand caveats to the doc
**Action**: Add tercile dose menu as the §7.3 replacement; direction split as new §7.7 (pooling valid, offset alternative rejected); four-group design near §6.4/§10 (passive-dilution fails), STATING fourgroup PRIMARY = post-2018 subsample (not the full 347,490). Add sell_lag/buy_lag to the §5.4 SQL SELECT (lines 256-266); add Russia B10 note ('LP anchored; only h=0 marginal; h=1 null p=0.052; do not cite as validation') + B9 VCE-guard note to §10/§15; append 'CUSTOMER+SUPPLIER only (B7)' to the §5.3 C4 row. Register all three in §12/§13/§15.
**Blocking**: None on the additions; but §13/§15 row values for the new components should carry RI inference (see safe_to_cite).

### 14. [MUST-FIX] ATOMIC doc number refresh — strictly LAST
**Action**: After fixes 1-5 land, one atomic pass: swap 7,928->6,854 (lines 18,127,643), 462,564->347,952 (268,484), 0.0476->0.0625 & '~4%'->'~6%' (440 and everywhere), refresh §7.1/§7.2 headline (+2.75e-6, se1.70e-6, p0.110, N347,490; MDE recompute), §7.4 (366-371,486), §7.5 (379-387,639), §7.3b (347-356,644), §5.5 gap on 0.0625, §8 centered/backward (424-425), §7.6 flow (399-408: N + inference RI p=0.37 not CRVE 0.44), whole §13 master table (538-556), §15 map (632-646), second-round table (57-65). Re-derive EVERY RI p from the regenerated audit panel, not by editing individual constants.
**Blocking**: BLOCKED-BY fixes 1,2,3,4,5. Refreshing the doc before the data re-runs locks in numbers the re-runs will move (queue-ordering hazard Q4). run_ddd gate constants also last.

### 15. [NIT] Nit cleanups on RI scripts
**Action**: run_ri_flow.py:27 — delete the false 'same spirit as the Delta-w main-spec convention' clause (dw is never winsorized; only flow p1/p99 as fat-tail robustness). run_ri_tercile.py:165 — update date tag 2026-07-22 -> 2026-08-02 (values already verified matching). Add hardcoded expected Stata constants to run_ri_direction.py:110 (sell +2.53e-7, buy +5.50e-6, sell-buy -5.25e-6) and run_ri_fourgroup.py:187 for symmetric drift protection.
**Blocking**: None.

### 16. [NIT] DESIGN_AGENDA snapshot hygiene
**Action**: When next touched, mark rank1/rank2 DONE (already completed in B7_REBUILD) and soften 'aggregate china_share carries a path/direction aggregation BUG' to 'the path split was a decomposition test; pooling confirmed valid (F p=0.424)'. No action if treated as a frozen snapshot.
**Blocking**: None.

## SAFE TO CITE
SAFE TO CITE NOW (all on the current B7 c6_panel.dta, 6,854 firms; note inference method):
- Ladder family (firm/country/2x2 cells): run_firm_ladder.do, run_country_ladder.do, run_ladder_2x2_cells.do — VALUE-EQUIV B7 panel (347,952 rows / 6,854 firms), verified byte-identical additive rebuild (git d14a6e8..b503338 = 0 deletions). CRVE inference OK.
- Shock-tercile dose menu (run_tercile_3pairwise.do / run_ri_tercile.py, 08-02, N=347,490, bins 28/27/27): us_cn×T3 +2.05e-5 (CRVE 0.264, RI 0.233); T3-T1 +6.76e-6 (RI 0.670). Cite by RI. Values re-verified identical on the current panel.
- Direction split (run_direction_split.do / run_ri_direction.py, 08-02, N=347,490): β₃sell +2.53e-7 (RI 0.931), β₃buy +5.50e-6 (RI 0.136), sell-buy (RI 0.205), pooling F p=0.424. Cite by RI. Load-bearing: rebuts the 'directional offset masks the null' alternative.
- Four-group active-only (build_fourgroup_panel.py / run_fourgroup.do / run_ri_fourgroup.py, 08-03, 695,904 rows): US_ACTIVE vs NONUS_ACTIVE FULL-period +2.79e-6 (CRVE 0.069, RI 0.289); passive-vs-passive ~0. Cite by RI. MUST label the post-2018 PRIMARY figure (+1.16e-6, RI 0.561) as a post-2018 SUBSAMPLE, not the full-sample number.
- Flow spec (§7.6): the POINT estimate +0.00089 raw (held-only +0.00135, B8) — but PRIMARY inference is RI p=0.37, NOT CRVE p=0.44 (two-way CRVE is degenerate/fat-tailed under B7). Cite only with the RI qualifier; pending the fix-8 bit-check on the flow panel vintage.

NEVER CITE (pre-B7 / stale / superseded):
- Any run_headline_3pairwise.do PART-A output or audit_c6_panel.dta number: OLD headline +1.80e-6 / N=462,096, it+gt +1.28e-6 / N=462,564. This is the documented 'headline pipeline' but it does NOT reproduce the number the project now calls the headline.
- All F1/F2/F7 + LP RI p from the pre-B7 audit panel: headline RI 0.51/0.62, LP cum1 0.23, cum4 0.07, h1 0.47, h4 0.23.
- §7.4 risk-set N=342,262 / β₃ +2.46e-6 and F8 β₃=+2.6e-6 (double-stale).
- §7.5 country-pair N=227,310 / β₁ +17.7 / β₃ +3.48 / F4 clustering p 0.098->0.227.
- §7.3b shock-lag anchors +1.28e-6/+1.80e-6 and S_{t-1} -8.65e-7/-6.51e-7.
- §7.1 tail-dummy menu +1.07 / +14.0 / +8.59 (N=462,564) — superseded by tercile AND stale.
- Entire §13 master table (any row), 07d three-spec N=462,564 (§7.1), §5.5 extensive-margin gap +11.26pp (computed on 0.0476 HIGH set).
- Russia LP h=0 perm_p=0.040 as design validation — sits on the all-rel-type (non-CUST+SUPP) numerator; and h=1 is now null (p=0.052, B10). Not comparable to the China measure until fix 5.
- sagg NEG β₃ (CRVE p=0.086 / three-pairwise 0.105) — pre-B7 and unadjudicated; do not cite in EITHER direction until re-run.
- run_ddd_nofe_bil Stage B/C no-FE and bilateral footnote columns — frozen pre-B7 (gate halts).
- The universal claim 'every point estimate is positive, opposite to disengagement' — FALSE as written while the sagg negative exists; do not repeat until scoped (fix 4).

## DROPPED (dup with known queue)
Dropped as duplicates of the known queue (a=partial-refresh, b=run_ddd gate, c=audit panel, d=Russia, e=tail supersession) — their content was MERGED into the ranked fixes above, adding only line-level anchors / extra consumers, so they are not carried as separate items:
- 'Supplement to known item (c)' entries naming the 5-consumer fan-out of audit_c6_panel.dta (run_headline_3pairwise.do:24, run_audit_f1f2f7.do:12, run_audit_f4f8.do, run_audit_f1b_robust.do, run_randomization_inference.py) and the design-lens 'SYNC BREAK' entry -> folded into fix 1.
- 'Supplement to known item (b)' entries on run_ddd_nofe_bil.do:42 (both spec and design lenses) -> folded into fix 9; confirmed the gate fail-stops safely (no silent wrong number), so no urgency beyond Stage B/C regen.
- 'Supplement to known item (d)' entries on 02_russia_exposure.jl:571 / build_russia_lp_panel.py and run_ri_russia parquet-regen gap -> folded into fix 5.
- 'Supplement to known item (e)' entries on run_tail_3pairwise.do supersession -> folded into fix 12.
- 'Details known item (a)' partial-refresh entry -> folded into fix 14 (atomic-pass requirement).

Dropped as NO-ACTION / verified-clean (moved to matrix as CURRENT / VALUE-EQUIV, removed from the fix list):
- firm_ladder_panel.dta, country_panel.dta, ownership_c6_panel.dta 'RESOLVED / value-equivalent, do NOT rebuild needlessly' — verified additive-only 08-02 rebuild; only ownership_c6 keeps a residual bit-check (kept as fix 8), the ladder panels dropped entirely.
- run_tail_3pairwise.do DATA-staleness concern — confirmed it reads the CURRENT c6_panel.dta, so the data-stale worry is dropped; only the role-supersession survives (fix 12).
- run_ri_tercile.py:165 staleness — the hardcoded constants were verified to still match the 08-02 panel exactly; the guard is correct, so only the cosmetic date-tag update survives (fix 15).
- Essay2_descriptive_figures.tex — cross-check found NO contradiction (Figure-1 CUST+SUPP universe matches §4.2; country-average incidence rates are a distinct construct from the 0.0625 HIGH edge-share cutoff). Dropped; optional one-line clarifying pointer only.
- DESIGN_AGENDA rank1/rank2 'aggregate-BUG' framing — the 08-02 direction split already proved pooling valid (F p=0.424), so rank1/rank2 are DONE; dropped to a nit (fix 16) with 'no action if treated as frozen snapshot.'