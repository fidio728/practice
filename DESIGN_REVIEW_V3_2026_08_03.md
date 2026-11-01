# DESIGN REVIEW V3 — Essay 2 (US 机构对华供应链暴露欧股去风险)

- **日期 / Date**: 2026-08-03
- **对象 / Object**: Essay 2 triple-difference design（`dw ~ β₂ US·CN_{t-1} + β₃ US·CN_{t-1}·S_t`，3-pairwise FE，two-way cluster firm×quarter，design-based RI arbiter）
- **Ground-truth docs**: `julia_descriptive/Essay2_methodology_full.md`（~747 行）; `DESIGN_AGENDA_2026_08_02.md`（19 ranks）; `B7_REBUILD_RESULTS_2026_07_22.md`; `CONSISTENCY_AUDIT_2026_08_03.md`
- **Headline（不变）**: 3pw `β₃ = +2.746e-6`，`SE = 1.697e-6`，`p = 0.110`，`RI p = 0.308`，`N = 347,490`，6,854 firms，HIGH cutoff 0.0625

## 一、方法与计数 / Method & Counts

**Method**: 6 个审查 lens（inference / referee-credibility / mechanism / literature / measurement / completeness）→ 候选去重（dedup）→ **逐候选对抗式验证**（对每个候选跑三条反驳线：INVALID 代码/事实误读、DUPLICATE 已被 doc §9 或 agenda 覆盖、IMMATERIAL 不改结论/可信度）。只有三条反驳全部未能击杀的候选才升级为 NEW-VALID，并按验证结论重新校准 severity 与 fix。

**Counts**:
- NEW-VALID gaps: **8**（MAJOR 5 + MINOR 3）
- Killed candidates: **17**（INVALID 5 + DUPLICATE 7 + IMMATERIAL 5）
- Agenda v2 status audited: 19 ranks（DONE 3 / NOT-STARTED 非顾问 9 / ADVISOR-GATED 6 / 低可行 deferred 1；另 2 个 registration gap）

**排序规则**: 先按 severity（MAJOR > MINOR），severity 内按 `changes-conclusion > referee-credibility > polish`。因此每个 severity 层内，**有可能产生新发现或改写叙事的排在前面**，纯可信度/内部一致性其次。

---

## 二、NEW VALIDATED GAPS（ranked）

### 【MAJOR】

#### G1 — 缺 extensive-margin（holder-count / breadth / position-termination）结果变量
- **Lens**: mechanism, literature
- **Gap**: 所有 outcome 都是 value/weight（§4.1 的 Δw USD portfolio weight、§7.6 的 shares-of-float flow `β₃=+0.00089`、六格 flow decomposition），**没有一个测持有广度（breadth）或退出（exit）**。`04_us_ownership_european.jl:141-153` 用 `SUM(adj_mv) AS I_ict ... GROUP BY sec_entity_id,...` 在形成 outcome 前就把 `fund_id` 丢掉，把上千只 US 基金塌成一个 group——10 只 US 基金里 5 只清仓，group Δw 几乎不动，完全不可见。divestment/exclusion 文献（Hong-Kacperczyk 2009 机构数、Chen-Hong-Stein 2002 breadth）里 holder count 是**首要且最先出信号**的边际。关键：doc §5.5 自己的 extensive-margin 描述性 gap 是 **+11.26pp 朝 H2.1（去风险）方向**（US zero-fill 33.67% = 5,733/17,027 vs NONUS 22.41% = 3,815/17,027），与 value-weighted β₃（+2.746e-6，正号）**符号相反**——两个边际发散，breadth 绝非冗余。`COUNT(DISTINCT fund_id)` 目前只作整册诊断（`n_funds`），从未作 firm×group×quarter 结果变量。
- **Refined fix**: 用同一张 `holdings_eom.parquet`（186,800,295 行，`fund_id` 已在）+ 06 的 firm×group×quarter 网格，聚合出 `n_holders = COUNT(DISTINCT fund_id)`。跑两个 DV，spec 完全同 headline（3pw FE + two-way cluster + RI 用 `run_ri_3pairwise.py` 机制做 arbiter）：(a) **breadth** — 主形式为 Chen-Hong-Stein 归一化 `n_holders/(该 group 当季活跃基金数)`（净掉 13F filer 长期上升），配 Δ(raw count)；(b) **exit/termination hazard** — §5.5 只算过描述性的那个 binary：`exit=1` if 上季持有本季为 0（LPM 保 RI 线性，cloglog robustness），加对称 initiation。**预登记方向**：因 §5.5 描述性同类 gap 已 +11.26pp 朝 H2.1，这是唯一带"有信号先验"的边际——breadth 出信号 + intensive weight null = **真正的 partial-divestment 发现**（extensive 有、intensive 无）；breadth null 则加固 null 的完备性。**新增披露**：count/exit 比 Δw 更受 US-13F-vs-NONUS 覆盖不对称影响（覆盖驱动的消失会被读成退出），故在 risk-set/engaged 子样本复跑。
- **Effort**: 1-2 天，数据全在盘上（复用 06 grid + RI 代码）。
- **为何前几轮漏掉**: 前审把 value-weighted outcome 家族（Δw、flow、decomposition）当成穷尽，从没问 divestment 文献的**首要边际**（持有广度）是否被构造出来——基金→group 的塌缩在 outcome 形成前就把它藏掉了。

#### G2 — 组内 simplex 归一化 + group×quarter FE 使 β₃ 对"US 专属、非 CN-graded 的整册欧洲/中国区收缩"完全失明
- **Lens**: mechanism, referee
- **Gap**: `w = H/T`，`T` = group g 的**欧洲总册**（§4.1）；组内权重是 simplex（§5.5：全 firm Δw 均值机械 ≈0）。`γ_{g,t}` 吸收 US 每季均值（β₀(US)、β₁(US·S)），§6.2 纯当"识别优点"呈现。后果：US 若**等比例**收缩整个欧洲册（转入美股/现金），Δw≡0——β₃ 只能识别组内再配置的 **firm-level CN 梯度**，看不到宽面去风险。§7.6 flow 同一 C6 grid + 3pw FE，`γ_{g,t}` 同样吸收 US 每季均值 flow，所以 flow 也是 per-firm US−NONUS differential。仓库里**没有任何非归一化的整册/水平 outcome 进过回归**（`portfolio_weight_global` 仅诊断，`I_ict` 不进任何回归，change-log A9 证实）。**与已有限制不同**：§9 limit 15 是 COMMON de-risking（两组同动），advisor Q1 是 market-clearing（NONUS 套利吸收 US 抛售）；本机制在**部分均衡下也咬**（NONUS 不动也成立，NONUS 全动也成立），是第三条轴。
- **Refined fix**: (1) **PROSE（~1-2h）**：在 §6.2、§9 补一条作为**独立于 limit 15 与 advisor Q1 的第三条轴**的限制，明说 β₃ 只识别组内 CN 梯度、对 US 专属整册收缩失明，并据此改写 §0/§1 estimand 措辞（"reallocate away from ... firms"）。(2) **TEST（~半天，仅用已有数据）**：从 `I_ict`（已在 `06_cartesian_grid.jl:149` 聚到 holder-group×quarter）+ `country_total_holdings_eu`（04）建 group×quarter 序列，回归 `S_t × US`：(i) US-vs-NONUS 投向 HIGH-CN bucket（CN>0.0625）占总欧洲册的份额；(ii) 非归一化的美元（及 summed share-of-float）HIGH-CN 暴露。用 design-based RI 锚定。**必须明标为描述性/confounded**（它必然丢掉 firm×quarter FE，是 bound 不是因果，类比 §5.5 extensive 描述）。读法：null → "HIGH-CN bucket 也无 gross US 抛出"（加固 differential null）；US 专属下降 → differential null 是归一化 artifact，headline 变有条件。**不要**把该 aggregate 回归当因果卖。
- **Effort**: 半天 + 1-2h prose，仅用已有对象（`I_ict`、`country_total_holdings_eu`、`portfolio_weight_global`）。
- **为何前几轮漏掉**: 前审用 COMMON de-risking（limit 15）和 NONUS market-clearing（advisor Q1）框住吸收；本条的"组内**分母** + `γ_{g,t}`"机制（部分均衡也咬）被和那两条混为一谈，从未单独隔离。

#### G3 — CN treatment 是 disclosure-timed 且 death-censored：customer/supplier 边只增不减，暴露"下不去"、无法登记 decoupling
- **Lens**: measurement
- **Gap**: `02_china_exposure.jl:592-594` 激活边用 `q.qend >= e.rel_start AND (e.rel_end IS NULL OR q.qend <= e.rel_end)`；`76-77` 与 `279-280` 把哨兵 `4000-01-01` 映射为 NULL——开口边（NULL rel_end）从 rel_start 到样本末**每季都活着**。`rel_start/rel_end` 是 FactSet Revere **披露日**，非经济现实日。后果 1（成立）：边**死亡登记不足**，公司真正对华 decoupling 时 CN **不降**——treatment 向下刚性，测不到紧张本应引发的供应链调整，古典测量误差把本已 null 的 β₃ 向 0 衰减，使 null 更不 informative。经验佐证 `02_china_exposure_timeseries.csv`：`avg_cn_customer` 0.69(2003Q2)→1.98(2025Q2)，`avg_china_share_among_exposed` 0.0202→0.363，近单调无实质下降（但 `avg_cn_jv` 0.50→0.11 会降——按 rel_type 分特征，不是绝对刚性）。**唯一存在的 NULL-end 诊断**在公司主表（`_run_02.log`: `n_null_end_post_norm = 428,786`/3,897,161 ≈11%），**边级（rev_rel）诊断完全没有**。后果 2（birth 与新闻 GPR 冲击共动）**被驳回**：CN·S 在 firm×quarter cell 内无变差，`α_{i,t}` 吸收任何共同 news-attention 驱动，不偏 β₃——**丢弃**。**与已有不同**：§9 item 12 是 match/coverage 率、F12 只是 counterparty **国别** as-of edge-start，都不谈生死日期；agenda rank12 前提**相反**（假设 CN 会降、FE 藏之），本条恰恰揭示 CN 在**数据里下不去**，反过来威胁 rank12 那个"高-CN 公司紧张期 CN 是否系统下降"的适应性描述——审查下平坦 CN 路径**不能**当作反适应证据。
- **Refined fix**: (1) **诊断（0.5 天，无 gate）** on China edge set（`eu_china_edge`/`rev_rel`）：**按 rel_type**（CUSTOMER/SUPPLIER/JV）报 (a) NULL rel_end 占比（never-closed）、(b) 已关闭边 duration 分布、(c) rel_start-to-quarter 分布（披露滞后）。(2) **Robustness**：首次观测（或 pre-sample）CN 冻结暴露，复跑 3pw headline + RI——**注意这与 rank12 的 optional DO-NOW 是同一个 spec**，不重复计数，但要**并列报告**死亡登记诊断，说明 censoring 下的解释差异。(3) **PROSE（§9 一条，区别于 item 12）**：CN 源自 Revere 供应链边，start/end 为披露日，customer/supplier 只增鲜减，暴露"下不去"、古典测量误差把 β₃ 向 0 衰减；绑到 TOST/credible-null 框架（rank13）；**删掉** birth/shock 共动那一支。
- **Effort**: 诊断 0.5 天；冻结 rerun 与 rank12 合并。
- **为何前几轮漏掉**: rank12 抱着**相反**前提（CN 会降、FE 藏之），没人去核 CN 在**数据里**到底能不能降——边的死亡登记从未被刻画。

#### G4 — §7.6 "make-or-break" 持股份额检验静默丢掉 26% headline 公司（选择性），且把"总股本"分母误标为 "float"
- **Lens**: measurement
- **Gap**: (A) w-headline 跑 347,490 行/6,854 firms；§7.6 flow panel 跑 239,792 行/**5,061 firms with a primary-EQ float**（doc line 426）——`build_ownership_share_c6_panel.py:65` 用 `CASE WHEN f.shares_out IS NULL THEN NULL` 把无 primary-EQ float 的公司整个丢掉，**1,793 firms（26%）样本内流失，dropped set 从未 profile**。doc 自称 §7.6 为 "make-or-break"（line 25），是 abstract 那句 "US investors do not reduce their ownership **stake**"（line 5）的**唯一支撑**（w-headline 只 license "weight"）。**与 caveat 13 不同**：caveat 13 披露 ADR **公司内**渠道，本条是**公司构成**选择（只经 ADR/非主类持有的公司被整体剔除）。(B) 分母是 `adj_shares_out` = **总股本**（`build_ownership_share_panel.py:85`；与 `04:173,186` market_cap 同源），非 free float，却在 lines 424/426/442/446/448 及 abstract 标 "float"/"bps of float"——库藏/国有/战略/内部人锁仓都在分母里，~1.5 bps MDE 其实是 bps-of-shares-outstanding。**被驳回不做**：(C) "LEVEL 用全类分子/主类分母" 是**误读**（`BASE_WHERE` 对分子分母同施 primary-EQ，INVALID）；(D) "ADR 误标 domestic dual-class" **已在 Caveat 1** 的 "ADR/GDR OR non-primary classes" 覆盖。且 >1 浓度过滤只删 25 cells/32 firm-quarters（`ownership_share_diagnostics.csv`），非主要驱动——26% 几乎全由无 primary-EQ float 覆盖造成。
- **Refined fix**: (1) **Profile 流失（核心，~2-4h）**：对 6,854-grid 里不在 5,061 样本的 1,793 firms 做 anti-join，按 (a) **CN bucket**（cutoff 0.0625，**决定性数字**：多少 HIGH-CN 公司无 primary-EQ float）、(b) 上市国、(c) 规模（market_cap）、(d) 单季 vs 多季 列表。CN/规模中性 → 明写 "flow null 非覆盖选择 artifact"；若 CN/规模倾斜 → 升级为真限制，指出 §7.6 null 可能剔除了最可能去风险的公司。(2) **术语（~30min）**：把 "float"/"bps of float" 全改 "shares outstanding"/"bps of shares outstanding"，与 line 422 对齐；若 FactSet 有 free-float 字段，脚注报 float-based level，否则注明当前份额是 fraction-of-tradable-float 的下界。**不碰** C、D。
- **Effort**: ≈0.5 天；预期不动点估计或 RI null，价值在可辩护的流失表 + 正确分母语言。
- **为何前几轮漏掉**: 26% 流失只以一个裸计数（5,061 firms）藏在 "make-or-break" 检验的 headline 行后；前审读成覆盖脚注、非 selection-on-outcome，从未 anti-join dropped set。

#### G5 — design-based RI 的更严 block/circular-shift permutation 被从 headline 与唯一 RI-显著的 cum4 上撤下；而 headline 冲击经验上序列相关（corr=0.271，Ljung-Box p=0.013），free permutation 对 arbiter 不满足可交换性
- **Lens**: inference
- **Gap**: circular-shift（`np.roll`）只在**唯一一个** engine `run_ri_sagg.py:70-73`，其 docstring 自称 free permutation "approximate"（因 `corr(s_agg_t,s_agg_{t-1})=0.357`）。其余全部 RI engine 只用 `rng.permutation` free perm：`run_ri_3pairwise.py:93`（含 headline + cum1/cum4）、`run_randomization_inference.py:86`、`run_ri_flow.py:81`、direction/fourgroup/tercile/russia/shocklag。**实测**（`output/audit_c6_panel.parquet` 上 82 个季度冲击）：`corr(S_t,S_{t-1})=0.271`，lag2=0.163；Ljung-Box(1) Q=6.23 **p=0.013**，LB(4) Q=11.36 **p=0.023**——headline 冲击**显著序列相关，非白噪**（AR(1) 在**月度**数据拟合、在季末月取残差，`05_combine_visualize.jl:96-97`；月度白噪不因季末抽样而保持）。项目自称 "innovations serially uncorrelated by construction" 是**月度**性质，从未在季末 stamped headline 冲击上验证。因 RI 统计量 `Σ_t S_t·C_t`，free perm 忽略 S 与 C 的序列对齐、低估 null 方差 → 反保守；cum4（outcome `w_{t+4}-w_{t-1}`，5 季重叠 → C_t 最长 MA-4 依赖）处最严重——正好是唯一 RI(0.039) 比 CRVE(0.053) **更**显著、与常规（flow RI 0.37；Russia RI 0.24）方向相反的 spec。RI 是 doc 明定的 "arbiter"（line 77），识别/推断构造是 stated contribution——把更严的 serial-robust 检验只跑在 null spec、never 验证 permuted 冲击可交换、把唯一越线的 horizon 留在最反保守的方案上，是对 arbiter 本身的尖锐、有据的攻击。
- **Refined fix**（~2-4h，无新数据；panel + `np.roll` 已存在）: (1) **REPORT** headline 冲击序列结构，紧挨已有 s_agg=0.357 披露：`corr(S_t,S_{t-1})=0.271`，LB(1) p=0.013，LB(4) p=0.023，纠正隐含的"S_t 是白 AR(1) 残差"假设。(2) **ADD** circular-shift（`run_ri_sagg.py:71` 的 81 个非平凡移位）**+ moving-block permutation（block ≥5，尊重 cum4 的 MA-4）** 到 `run_ri_3pairwise.py`（h0, cum1..cum4）与 `run_randomization_inference.py`（it+gt horizons），三个 p 值（free/circular-shift/moving-block）并列，正如 `run_ri_sagg.py` 已对 s_agg 所做。(3) 把 circular-shift（或三者更保守者）设为 LP horizons 的**报告 arbiter**。解释两向：cum4 block/circular p 升过 0.05 → 唯一 RI-显著 horizon 消失，null 更干净；仍成立 → 正号（反 H2.1）效应须解释，不能只用 overlapping-window caveat 挥手。
- **Effort**: 2-4h，无新数据。
- **为何前几轮漏掉**: circular-shift 变体确实存在，但只在 s_agg 脚本（一个 null spec）；前审看到"block variant is run"就放过，没注意它被从 headline 撤下，也从未测 headline 冲击自身的 autocorrelation。

### 【MINOR】

#### G6 — β₃ 无 US×(非中国 firm 特征)×S 判别效度 placebo：无法区分 China-specific 与泛 US-vs-NONUS risk-off/factor-demand tilt
- **Lens**: literature
- **Gap**: β₃ 识别 US−NONUS 对 `CN_{t-1}·S_t` 的**差异**响应；`α_{i,t}` 吸收任意特征的**水平**，`γ_{g,t}` 吸收 US×S，但**都不吸收 US×X×S**——β₃ 在 CN 单独项上可载入相关特征（规模/beta/foreign-rev）的差异响应。现有 placebo（Russia §7.9、IN/VN/MX rank9、shock-permutation RI）**全在变 exposure，无一在同一冲击轴上变 firm 特征**。`.do` 里 grep 无 beta/volatility/momentum/foreign-revenue control；唯一 `us_bil` 是国别 bilateral **水平**控制、不与 S 交互。**校准两点**（故 minor 非 major）：候选把 masking **方向**说反了——"高-CN 更小/高-beta→US 更抛"是**同号**（放大而非掩盖），观测 β₃ 又是**正**；合理的掩盖故事需高-CN 公司**大**（US flight-to-large 正 confound 抵消负 China channel），与描述性 "switchers larger (+1.36 log_at)" 一致。可行性也被高估：realized-vol/beta 需未建的欧股收益（§10 H2.2），non-China foreign-rev 不可行（rank8 覆盖 CUSTOMER 12.8%/SUPPLIER ~0%）。
- **Refined fix**（改写为判别效度/冲击特异性，纠正掩盖方向）: (A) **首选、新颖、~1-2h** — 错误双边冲击 placebo：把 `S^Russia`（`russia_shock_monthly.csv`，季末 stamp）并到 `c6_panel.dta`，估 `dw ~ US·CN_{t-1}·S^Russia`（3pw FE + RI）。null（预期）→ β₃ 非泛 risk-off 载于 China-exposed 公司；显著 → contamination。**与 §7.9 的 US×RU×S^Russia 不同**（这里固定 CN 暴露、换冲击）；注意 S^China/S^Russia 在 2022 共动，配 (B)。(B) **次选、~半天** — 规模×冲击 placebo：并入滞后 `ln(mktcap)`（rank7 层已有），估 `dw ~ US·CN_{t-1}·S_t + US·ln_mktcap_{t-1}·S_t`（正交化控制）+ 独立 `US·ln_mktcap_{t-1}·S_t` placebo 列，3pw + RI，报 β₃ 是否对该控制稳健、规模×冲击差异是否本身 null。明标为 confound/正交化角色（US×X×S），区别于 rank7 的 CN-moderator。(C) **DROP** vol/beta（需未建欧股收益）与 non-China foreign-rev（覆盖 ~0-12%），§9 披露。若错误冲击或规模 placebo 回显著，headline null 须重述为"非干净 China-specific"。
- **Effort**: A 单跑 ~2h，A+B ~半天，纯 Stata，已有 panel（`c6_panel_russia.dta`、`russia_shock_monthly.csv` 已建）。
- **为何前几轮漏掉**: placebo 阵（Russia、IN/VN/MX）全在变 exposure；没人在同一冲击上变 firm 特征，CN 相对泛 risk-off tilt 的判别效度从未测。

#### G7 — F10 把唯一 RI-显著的 cum4 horizon 误标为 "RI-null"，且对 LP horizon 族（唯一含 RI-显著成员的族）不施族内多重比较（max-|t|）校正
- **Lens**: inference, referee
- **Gap**: F10（line 83）称被标记的 nominal-CRVE-p<0.10 集（**明确含 "the cumulative-LP horizons"**）"each adjudicated **null under valid design-based RI**"；但 §0(line 5)、F1(line 65)、§13(line 626) 均报 "LP cum h4 ... **RI 0.039**（3-pairwise, **positive** sign）"。RI p=0.039 在 5%（乃至 10%）下**非 null**——paper 专讲多重检验的那一段**说错了自己唯一的 RI-显著结果**（doc 别处对 cum4 都诚实：正号、反 H2.1、overlapping-window caveat、WCB OOM）。且 `run_ri_3pairwise.py:102-105` 对 "headline dw"、"LP cum1"、"LP cum4" **分别**调 `ri_twoway`（各自 `abs(bp)>=abs(b_obs)` 两侧 p，lines 92-97），**族内无 MAX|t|/Romano-Wolf** step-down——对唯一产出 RI-显著成员（cum4 0.039）的 spec 族（h0..h4）报未校正的族内 RI p 是教科书式多重性漏洞，且校正现成可得（permutation 已 seed 对齐，`SEED=20260702`）。
- **Refined fix**（两处编辑，复用现成 O(82) RI 机制，~半天）: (1) 在 `run_ri_3pairwise.py` 单 permutation 循环内，逐 draw 记录 horizons 上 **studentized** 统计 `|t_h|=|β₃_h|/se_h` 的 **MAX**（**不用 raw |β₃|**，因 cum4 系数尺度 ≈8.8e-6 会机械压过 headline ≈2.7e-6），报"任一 LP horizon 显著"的族内 FWER p，替换 F10 的 "count is consistent with expected false-positive rate" 断言；理想扩展到全 h0..h4。(2) 改 F10 句：把 "cumulative-LP horizons ... each adjudicated null under valid design-based RI" 换成 §0/F1/§13 已用的诚实表述（cum4 达 RI 0.039 但**正号**、反 H2.1、WCB 无法跑、族内 max-|t| 校正后 p=[填]），保留 F10 真点（所有 CRVE-only 正号在 RI 下死）但停止称 cum4 是 RI-null。**不发**候选更宽的 spec β₃/RI 相关矩阵与"有效独立检验数"——建立在假的"same shock vector"前提上（s_agg corr 0.425、S_{t-1}、raw GPR、Russia、country-pair 都是不同冲击），且与 Russia positive control + limitation 1 + rank14/15 冗余；至多加一句：FE/样本扰动 robustness 列 correlated-by-design（稳定性检验，非独立佐证）。
- **Effort**: ~半天，复用 O(82) 机制。
- **为何前几轮漏掉**: F10 写在 cum4 RI 0.039 结果**之前/独立**且从未 reconcile；前审在 §0/§13 处核对 cum4（那里诚实），没交叉读多重检验段。

#### G8 — "WCB 不可能（boottest OOM）" 是稠密矩阵工具伪限制；score-based quarter-cluster WCB（或已编好的 circular-shift block-RI）是唯一 rejection 上现缺的、可行的独立检验
- **Lens**: inference
- **Gap**: doc 5-6 处称 wild-cluster bootstrap "could not be completed（boottest ran out of memory at 10k reps on the ~155–169k×10k allocation）"（§6.3 line 301；§0、F1 line 75、§10 line 530、§13 line 642、table line 65），并列为硬 infra 约束，cum4 "anchored by RI alone"。但 "~155–169k×10k allocation" 正是稠密（diff-rows × B）矩阵伪限制；score-based 手写可绕开。**校准**（故 minor）：候选说 "A_t,C_t collapse 使 WCB O(82)" 是**误读**——那对充分统计只存在于 **it+gt/quarter-FE-only** 差分回归；唯一 rejection（cum4 RI 0.039）是 **3-pairwise** spec，`run_ri_3pairwise.py:13-16` 明说无闭式充分统计、每次 permutation 两向去均值全 panel。但**结论仍成立**：score-based quarter-cluster WCB 只需一次全 panel FWL residualization（已编好的 30-iter demean）再 O(82·B) 聚合，boottest 的 OOM 仍可绕。DESIGN_AGENDA 对 boottest/wild/WCB **零命中**，无 inference-robustness 项排队，故非 DUPLICATE——它**推翻**一个项目当前相信的硬约束。conclusion-invariant（cum4 正号、反 H2.1）故 minor，但 paper 自己把 WCB 框成 wanted-but-blocked、在唯一 rejection 上，用作者自己的 O(82) 机制即可完成——对以推断纪律为卖点的 null paper 是真可信度负债。
- **Refined fix**（~半天，无新数据）: **不要**卖成 "A_t,C_t collapse 扩到 WCB"（那是 it+gt-only）。正确路线：(1) it+gt LP horizons 的 per-quarter `(A_t,C_t)` 直接给 quarter-cluster scores → trivial WCB。(2) **3-pairwise cum4（唯一 rejection）**：复用 `run_ri_3pairwise.py` 的 30-iter 两向去均值，一次残差化 Δy、cn、cn·S；FWL 得 x̃ 与 restricted（β₃=0）残差 ê；聚成 82 个 quarter-cluster scores `g_t=Σ_{i∈t} x̃_i·ê_i` 及 `x̃_t'x̃_t`；每 rep 抽 82 个 **Webb 6-point** 权重（82 clusters 适用），`β₃*=(Σ_t v_t g_t)/(x̃'x̃)`，用重加权 scores 重算 cluster-robust SE 做 studentize——O(82·B)，无稠密 N×B，无 boottest。对 headline、cum1..cum4、flow 报 WCB-p 并列 RI-p。因是共同时序冲击，**quarter-cluster** WCB 是对的维度（firm 有数千 cluster，渐近无碍），注明以免审稿人期待 two-way bootstrap。**并提候选漏掉的更便宜首步**：把 `run_ri_sagg.py:71` 已编 circular-shift block-RI 直接跑在 cum4 outcome（`d_c4`）上——近零边际成本直接治 overlapping-window serial-dependence caveat。然后改写 §0/§6.3/F1/§10/§13 及 memory 里 "WCB impossible/boottest OOM" 句，说检验已完成（或 serial-robust RI 变体替代）。
- **Effort**: ~半天，无新数据。
- **为何前几轮漏掉**: "boottest OOM" 被当硬 infra 事实、传播到 5-6 处 + memory；没人质疑 score-based 手写可绕开稠密矩阵。

> **推断簇提示**: G5（major）、G7、G8（minor）都围绕 cum4/RI，共享 permutation 与 cum4 机制。建议作为一个工作包 "cum4 inference hardening" 一次做完（见第五节）。

---

## 三、KILLED CANDIDATES（review trail）

**INVALID（5）— 代码/事实误读**
1. "RI 比 raw β₃、非 studentized，故只在强可交换下 exact 并重继承 few-cluster fragility" → **INVALID** → 代码事实真（三 engine 都比 raw 估计），但 design-based RI 用 raw β 是标准做法，"重继承 few-cluster fragility" 未被证成。
2. "Outcome 是 raw within-book weight，混淆机械 benchmark drift 与主动 tilt" → **INVALID** → 描述性真，但推断性主张与 fix 都误读 FE 结构（firm×quarter FE 已处理）。
3. "旗舰 3-pairwise headline 坐在 F2 谴责的全 grid 上，in-span fix 只施于 it+gt 表亲" → **INVALID** → `run_headline_3pairwise.do:72` 恰好跑 3-pairwise in-span spec。
4. "仓库跑 level-w ladder + bilateral no-FE，有 β₃=+9.0e-5(p=0.048) 单元格且不在 doc" → **INVALID** → 把 double-difference 误标为 β₃；ladder 无冲击项。
5. "唯一 motivating 描述性(+11.26pp zero-fill gap) 算在 superseded pre-B7 集、标 NEVER CITE，故无 live motivating fact" → **INVALID** → "无 live motivating fact" 的推论建立在误读上，为假。

**DUPLICATE（7）— 已被 doc §9 或 agenda 覆盖**
6. "从未确立随连续 China supply-share 分级的 US-specific friction" → **DUPLICATE** → agenda rank14 + advisor Q2 + §9 limit 15/16。
7. "MDE 只报统计单位、从未译成 divestment 情景/基准/条件化" → **DUPLICATE** → 已排队 rank13（TOST 经济锚定）。
8. "DiD 无 EXECUTED pre-trend/anticipation placebo，只是披露限制；负 horizon LP 近乎免费" → **DUPLICATE** → 已知已披露；LP 只建前向 horizon。
9. "唯一 positive control（Russia）在 RI arbiter 下失败，无 first-stage 证 S_t 动任何量" → **DUPLICATE** → 两半都在 doc/agenda，"新"半亦部分 INVALID。
10. "如此 scope 是干净 panel + 薄 null，不可独立发表；需 H2.2 bundle 或 re-scope" → **DUPLICATE** → fix(a)=rank18，核心是 doc 自陈立场。
11. "关键 retraction（centered→backward Δw）把 3× SE 跳归因 look-ahead 靠断言，从未分解" → **DUPLICATE** → §8 + §10 deferred + §0 已披露。
12. "F7 全样本 AR(1) look-ahead robustness 只 APPROXIMATE，exact-nesting 变体 deferred" → **DUPLICATE** → doc line 81 逐字已述。

**IMMATERIAL（5）— 不改结论/可信度**
13. "持续正 β₃ 从未对 friend-shoring/China-substitute channel 检验" → **IMMATERIAL** → 真缺，但 §9 limit 16 已披露 substitution 概念，建 competitor 回归不改结论。
14. "季末 report-date 过滤注入 NONUS-only in-out-in Δw 噪声（off-cycle UCITS vs 季度 13F）" → **IMMATERIAL** → 机制真但不改结论。
15. "w 是多币种册的 USD value weight，FX 经 group 币种构成载于 CN×S 并存活双 FE" → **IMMATERIAL** → 机制有效但 numerator-FX 在 US−NONUS 差分抵消，残余存活 FE 但不 material。
16. "识别来处从未量化：无 leave-one-quarter-out/leverage/influence；un-winsorized CRVE primary" → **IMMATERIAL** → 两 prong 都不改结论；winsorize prong INVALID（headline 是 RI-primary 非 CRVE）。
17. "HIGH/LOW 表对照的 LOW 控制自身带 1-6% China 暴露、用数据依赖 cutoff" → **IMMATERIAL** → 载重子主张 INVALID；cutoff robustness 已存在。

---

## 四、AGENDA v2 状态表（rank 1-19）

| Rank | 项目 | 状态 | 一行证据 |
|---|---|---|---|
| 1 | direction-split（sell/buy） | **DONE** | §7.7 已注册；pooling F p=0.424，offset rejected |
| 2 | four-group active/passive | **DONE** | §7.8；active +2.79e-6≈pooled，passive-dilution rejected，post-2018 primary RI 0.561 |
| 3 | flow accounting decomposition | **DONE** | §7.6；六格 null，corr(flow_US,flow_NONUS)=+0.20 锚 |
| 4 | NONUS→Continental-EU main control（bloc 分解、GPFG share） | **NOT STARTED** | doc 仅 EU-domiciled-only deferred §10；无 bloc 分解、无 GPFG-share 数字 |
| 5 | post-2018 regime split + long-difference stock adj | **NOT STARTED**（standalone） | 仅 §7.8 内有 post-2018 window；无 us×cn×S×post2018 四交互、无 long-difference/年度聚合 |
| 6 | S_pos/S_neg 非对称交互 | **NOT STARTED** | 缺席 |
| 7 | absolute exposure ln(1+links) + mcap-visibility | **NOT STARTED** | 缺席 |
| 8 | CN deciles/splines dose form | **NOT STARTED** | §7.3 tercile 切的是**冲击 S_t**、非 CN 暴露（勿混） |
| 9 | IN/VN/MX exposure-specificity placebos | **NOT STARTED** | Russia positive-control 是不同 construct；无 nearshoring placebos |
| 10 | Greater-China {CN,HK,TW} | **NOT STARTED** | 缺席 |
| 11 | intensive-only margin on main w | **NOT STARTED** | 仅 FLOW outcome §7.6 有 held-only（+0.00135）；主 Δw 未做 |
| 12 | firm endogenous adaptation as boundary | **NOT STARTED + 未披露** | §9 十六条无 "adaptation absorbed by firm×quarter FE" 项（registration gap） |
| 13 | TOST equivalence | **NOT STARTED / ADVISOR-GATED** | 已报 MDE，无正式 TOST；δ 顾问决定 |
| 14 | direct-China ADR benchmark + weak-prior scope | **NOT STARTED / ADVISOR-GATED** | email 草稿在 agenda |
| 15 | discrete policy events（GSDB/entity-list/tariff） | **NOT STARTED / ADVISOR-GATED** | 同上 |
| 16 | monthly EOM event studies | **NOT STARTED / ADVISOR-GATED** | monthly 可行但 advisor-gated |
| 17 | pre-specified responder subsets（ESG/pension/active-ex-index） | **NOT STARTED / ADVISOR-GATED** | 同上 |
| 18 | two-essay joint design（H2.2 price channel） | **NOT STARTED / ADVISOR-GATED** | 同上 |
| 19 | state divestment-law DDD | **NOT STARTED（deferred，低可行）** | deferred，低 feasibility |

**Registration gaps（DONE-without-doc-registration）**：(a) rank12 完全未在 §9 披露；(b) level-w LADDER 家族 + bilateral no-FE specs（`run_firm_ladder`/`country_ladder`/`ladder_2x2`/`ddd_nofe_bil`）已跑、标 "safe to cite"，但**不在 doc**，且含边际单元格 p=0.048/0.060/0.090——须么带上下文注册、么明确排除（注意这些是 double-difference 非 β₃，见 killed #4）。

---

## 五、RECOMMENDED SEQUENCE（合并 new gaps + NOT-STARTED agenda）

### 队列 A — DO-NOW（无 advisor gate，按 性价比 × 可能改结论 排序）

**A0 · cum4 inference hardening 工作包（G5+G7+G8 一次做完，~1 天）** — 依赖关系最强、共享 permutation/cum4 机制。
- 先 G5：报 headline 冲击序列结构 + 加 circular-shift & moving-block(≥5) 到 `run_ri_3pairwise.py`/`run_randomization_inference.py`。
- 顺带 G8：把 circular-shift block-RI 跑在 `d_c4`（近零成本）+ score-based quarter-cluster WCB（cum4/headline/cum1/flow），改写 "WCB impossible" 句。
- 收尾 G7：族内 studentized max-|t| RI over h0..h4，改 F10 句。
- **产出**：cum4 用更严且一致的 arbiter 报告；唯一 rejection 的可信度问题一次性关闭。

**A1 · G1 extensive-margin 结果家族（1-2 天）** — **最可能产生新发现**（breadth 有信号 + weight null = partial-divestment）。独立，复用 06 grid + RI。

**A2 · G2 aggregate/level 互补检验（半天）** — 可能把 headline 变有条件（若 US 专属整册收缩）。独立，用 `I_ict`/`country_total_holdings_eu`。

**A3 · G3 CN death-censoring 诊断（0.5 天）+ 冻结暴露 rerun（= rank12 spec，合并）** — 测量误差衰减 + 威胁 rank12 适应性描述解释。冻结 rerun 与 rank12 是同一 spec，一次跑、两种解读并报。

**A4 · G6 判别效度 placebo（~2h 起）** — A：错误双边冲击 US×CN×S^Russia（用已建 `c6_panel_russia.dta`/`russia_shock_monthly.csv`）；B：规模×冲击 placebo。可与 rank9（IN/VN/MX）合成一个"冲击/暴露特异性"电池。

**A5 · G4 §7.6 流失 profiling + 分母术语（0.5 天）** — 可信度；决定性数字 = HIGH-CN 无 primary-EQ float 占比。独立。

**A6 · 便宜 Stata agenda 项（各 2h-半天，纯 `c6_panel.dta`/`.jl` 网格，无新数据）**：rank6（S_pos/S_neg 非对称）、rank8（CN deciles/splines dose，**注意与 rank7 剂量形式区分**）、rank7（absolute exposure ln(1+links) + mcap-visibility）、rank10（Greater-China {CN,HK}）、rank11（intensive-only on 主 Δw）、rank5（post-2018 四交互 + long-difference）、rank9（IN/VN/MX placebos，与 A4 合并）、rank4（Continental-EU control + bloc 分解 + GPFG share）。

### 队列 B — WRITING-ONLY（prose/registration，可与 A 并行）
- **rank12 披露**：§9 补 "firm adaptation absorbed by firm×quarter FE" boundary 限制（registration gap；与 G3 的死亡登记诊断互相强化——censoring 下平坦 CN 非反适应证据）。
- **G2/G3/G4/G6 prose**：各自 §9/§6.2 的一条限制 + G4 分母术语 float→shares outstanding。
- **Ladder/bilateral no-FE 注册**（status audit flag b）：把 `run_firm_ladder` 等及 p=0.048/0.060/0.090 边际单元格带上下文注册或明确排除，注明是 double-difference 非 β₃。
- **rank13 credible-null 框架 prose**（δ 待 advisor，但 TOST 叙事骨架可先写）。

### 队列 C — ADVISOR-GATED（先备 email，勿自行启动）
rank13（TOST δ）、rank14（direct-China ADR benchmark + weak-prior scope）、rank15（discrete policy events primary treatment）、rank16（monthly EOM event study）、rank17（pre-specified responder subsets）、rank18（two-essay joint H2.2 price channel）、rank19（state DDD，低可行、最后）。

### 依赖与合并要点
1. **A0 内部有序**：G5 的 permutation 机制是 G7（max-|t|）与 G8（block-RI on d_c4）的前置，务必同一工作包。
2. **A3 ≡ rank12 frozen-exposure**：同一 spec，**不重复计数**；解读不同（rank12 假设 CN 会降、G3 揭示 CN 下不去），两读并报。
3. **A4 ⊂ rank9 电池**：错误冲击 + IN/VN/MX 合成一个冲击/暴露特异性电池，一次性报。
4. **A1/A2/A5 相互独立**，可并行，均只用盘上数据。
5. **所有 A 队列 RI 都用 `run_ri_3pairwise.py` 的 O(82) 机制做 arbiter**，与 headline 一致（SEED=20260702）。

---

*本文件为自足综述，供未来 session 直接续作。数字均逐字引自经对抗式验证的 doc/code 证据块；未验证处不写数字。*
