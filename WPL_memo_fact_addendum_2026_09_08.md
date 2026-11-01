# 底稿补正与审阅误报驳回（2026-09-08，优先级高于 GROUNDING_BRIEF.md §A）

## 0. 为什么有这份补正

GROUNDING_BRIEF.md 的 §A 是 Essay 1 审计记录的**摘要**，不是全集。sec02 的起草 agent 按其指令读了完整的内部审计记录（tool-results/b8s53y351.txt 与 bhonr8h0t.txt，以及仓库内 REGRESSION_SPEC_AND_ROBUSTNESS.md），因此写入了若干 §A 未收录但**真实存在**的结果。引用审阅者据 §A 判定这些为「编造」，属于误报。以下逐条驳回，并把这些事实正式补入 §A。

**规则更新：sec02 的事实基准是完整审计记录（含本补正 §1），不是 GROUNDING_BRIEF §A 的摘要。其余各节仍以 §A + §1 为准。**

## 1. 经复核为真、正式补入 §A 的 Essay 1 事实

以下每条均在审计记录原文中逐字核实：

- **A5 事件特异性分解（2026-06-13，WHY-#1）**：把 headline `yv_l1p = ln(1+AIM_fil[-1,+1]) − ln(1+AIM_base[-60,-10])` 拆成两个 level 分别对 sub_post 回归。BJS noctrl，nTbjs=580：`fil_l1p` β=−0.008 p=.20 且 **pretrend 失败**（pre12=0.004）；`base_l1p` β=−0.0248 p=.010 pretrend 干净（pre12=.239）；恒等式精确（残差 1e-9）。**读法（记录锁定）**：headline 的 abnormal 上升在算术上是**基准窗口 AIM 下降**，申报窗口 level 持平，不是申报窗口非流动性的绝对上升。禁止写「申报窗口原始非流动性上升」；应写「相对于下降中的局部基准，申报日恶化」。术语：`[-60,-10]` 称 **local baseline**，不称「普通日」。用户裁定（2026-06-13）：这是**识别/稳健性诊断**，不是机制展品，不威胁机制叙事。
- **A6 EDGAR 替代（abnormal 口径重跑）**：`run_edgar_substitution_td.do`，口径 `ln(1+edgar[-1,+1])−ln(1+edgar[-60,-10])`，gate exact，27,691 obs：**所有访问类型（periodic/10k/10q/8k）全部 null**。旧的 form-10K access −11% 至 −18% 属水平/窗口/cohort 口径。记录表述：「EDGAR reading → no substitution channel (NULL, cleaner exclusion than the old negative)」。**两条并存，abnormal 口径是与 AIM 同构的当前口径。**
- **A7 unblock 降级（X5）**：816 个 reverter 中 **48% 之后再次封锁**（非吸收性处理）；in-sample 734 个处理公司中 359 个（48.9%）为 ever-reverter，约 27% 的吸收式 post 公司月实际已重新开放。`did_imputation` 不支持开关式处理。结论：标准 staggered DiD 用于 unblock 无效，只剩描述性对称图；若要做需 censor-at-reblock 或 dCDH。
- **A8 切换稳健性 gate（2026-06-05）**：RAny filing-window 吸收式 BJS +0.0179 p=.038；剔除 359 个 reverter 后 **+0.0251 p=0.137（点估计更大，精度下降）**；dCDH filing-window 季度网格 placebo 干净（joint p=0.323），Av_tot=−0.016（95% CI [−0.045,+0.012] **包含 +0.018**）→ 失败拒绝，不是反号推翻。显著性杠杆是 onset-month 申报的纳入/剔除（excl +0.0179 p=.038；incl +0.0142 p=.106）。锁定表述：「not robustly identified once treatment reversibility is accounted for; underpowered to distinguish a small positive from zero, NOT evidence of no effect.」
- **A9 中介 horse race 与复杂度渠道（2026-06-12）**：`px_age`（公司年龄）在**每一个** horse race 中显著为负（l1p −0.0189 p=.031；lev −0.056 p=.024），在控制 IO、retail float、分析师、规模、申报复杂度之后仍存活；`px_cplx`（申报包字节数）处处为 null。结论：机器处理成本叙事得不到支持，年龄梯度不是复杂度的伪装。
- **A10 uncovered-204 profile 与 kill-test**：covered/uncovered 在机制上是**上市年份分割**（IO moderator 需 2020 年 13F → uncovered 中 89.7% 于 2021-01 后进入 CRSP；covered 全为 incumbent）。204 家 uncovered = 80 家 SPAC/低价壳 + 103 家真实新上市 + 21 家 uncovered incumbent。kill-test（处理与对照对称过滤）：uncovered 全样本 +0.3596（p=.005，n=204）→ 剔除 SPAC/penny +0.2280（p=.017，n=124）→ 仅真实新上市 +0.3681（p=.028，n=103）；covered 参照 +0.0005 null。**效应不是 SPAC/penny 产物。** uncovered 并非分析师荒漠（74.5% 有 IBES 覆盖）。
- **A11 CLT 过滤瀑布（2026-06-11）**：headline 对 CLT（Chang–Ljungqvist–Tseng RFS 2023）式股份代码宇宙与**公司层面** 5 美元价格过滤稳健（+0.0335 p=.003）；per-filing 5 美元过滤是向下偏误的产物；**剔除金融业（SIC 6000-6999）是约束性敏感点**，移走 43 家 uncovered SIC-6799 投资/控股壳公司，系数变弱且 pretrend 走坏。金融业敏感性必须披露。
- **A12 EOM 构建（REGRESSION_SPEC_AND_ROBUSTNESS.md，2026-04-28）**：天真的 latest-any 规则会把 **41.5%** 的 primary-cohort host-month 判为不可解释；这是采用 latest-interpretable 的理由。稳健性口径 r2/r4/r5 如该文件所列。tier 分布：ok_200 25.8%、ok_revisit_resolved 29.9%、ok_404_allow_all 5.1%、uninterpretable 3.6%、gap 35.5%。
- **A13 WHY-#2 同步性与事件时间（2026-06-14）**：sync 指标（SyncCorr、ThinImpactShare、PeakMismatch）在两个估计量下**全部 null**；事件时间剖面是 **[−9,+10] 上的宽而平的抬升**（pre-side 均值 ≈+0.0069 ≈ post-side ≈+0.0080），不是申报日尖峰；远端安慰剂 [+29,+31] null。结论：wedge 不表现为可测的收益—成交量去同步化，**不得主张同步化机制**；效应局限于申报邻域且在 +30 前衰减。
- **A14 Channel D robots 层解剖（2026-06-13）**：root_new 472（22%）、path-only 1,699（78%）。`θ_root>0` **失败**（px_root −0.016 p=.08-.09，nArm=91）；`θ_ops≈0` 确认；**`θ_qs≈0` 被违反**：querystring/CMS 模板卫生型屏蔽 px_qs −0.054 p=.037-.052（nArm=73），即模板波事件**不承载** AIM 上升，base 效应存在于 path-only 非模板屏蔽者。内容臂 MDE@80% ≈ 0.053 ≈ 3 倍 base 效应 → 「无内容差异」是功效陈述，不是等价性。
- **A15 WHY-#3 Common Crawl 可发现性 first stage**：2026-06-13 启动（单 IP 单 worker，rate 1.0s；CC 禁止同 IP 多线程）；**2026-06-15 判定 FAILED**（记录原文：「WHY-#3 CC DISCOVERABILITY FIRST-STAGE — RAN 2026-06-15, FAILED (the last positive candidate)」）。规模：4,366 个 canonical host（1,431 处理 + 2,935 clean_never）。
- **A16 AI 臂**：论文不做 search-vs-training 对比；AI 臂安慰剂 null 且 underpowered（2026-06-15 汇总）。与 2026-08-18 live scan 的「AI-search 具名仅约 82–85 host」一致。

## 2. 驳回的审阅意见（误报，**不得据此删改**）

以下编号对应 ISSUES.md。这些条目的共同错误是「§A 未收录 ⇒ 编造」。经原始记录核实为真，**保留原文，不删不改**：

- **[5] [7] [14]（部分）** EDGAR 06-14 abnormal 重跑全 null：真实，见 §1 A6。**但**：sec02 与 sec12 必须口径一致（见 §3 第 1 条，这部分的批评成立）。
- **[6]** 事件特异性 2026-06-13：真实，见 §1 A5。
- **[8]** unblock 近半数 reverter：真实，见 §1 A7。
- **[9]** Li–Ramesh 并发/非并发拆分：真实（记录中有该拆分且方向对「申报文档被发现」叙事不利）。保留，保留其「事件期分组、不可作因果读」的限定。
- **[10]** 中介 horse race 公司年龄显著为负：真实，见 §1 A9。
- **[12]** 41.5% 与 r2/r4/r5：真实，见 §1 A12。
- **[13]** Sun–Abraham 动态系数 null 或略负：审阅者称 §A 只提 BJS/TWFE/csdid。该条**未在本轮复核中定位到原文**，处理方式：保留句子但**降级表述**为「另有内部记录显示替代估计量结果不一致」，删去「Sun–Abraham」具名与方向描述，或标 \textbf{[需核验]}。这是唯一未证实的一条。
- **[16]（CLT 部分）** SIC 6799 与 CLT 过滤瀑布：真实，见 §1 A11。**但** CLT 的引用语境须改：CLT 2023 是 EDGAR 分批上线 → 分析师覆盖/准确度的论文，本处只是借用其**样本过滤口径**，行文须写明「借用其股份代码宇宙口径」而非暗示 CLT 做了价格过滤研究。
- **[19]** AI 臂安慰剂：真实，见 §1 A16。
- **[20]（部分）** egen cohort 撤回、排除规则、旧 headline 退休：真实（2026-05-17 记录）。
- **[21]（部分）** uncovered 九成为 2021 后新上市、剔除 SPAC 后仍全幅承载：真实，见 §1 A10。**但** 数字须按 A10 精确化（89.7%、+0.2280/n=124、+0.3681/n=103）。
- **[33]** Common Crawl first stage「脚本已写但未跑」：**表述错误但方向相反**——它跑了，且于 2026-06-15 判定 FAILED（§1 A15）。按 A15 改写，不是按审阅者的建议改写。

## 3. 成立的审阅意见（必须修）

1. **口径一致性**：sec02 §2.6 说 06-14 abnormal 重跑全 null，sec12 D1/D2 仍按 05-19 的 −11%~−18% 负值引用为当前证据。两处必须统一为 A6 的双口径表述：「事件 TWFE 的 form-10K 访问为 −11% 至 −18%；与 AIM 同构的 abnormal 口径下所有访问类型 null；当前表述是无补偿性替代渠道」。
2. **sec02 台账内部矛盾**：第 47 行自定 effect present ≡ BJS p<0.05 且 TWFE p<0.10，clean pretrend ≡ pre_p ≥ 0.10；第 83 行却把「TWFE n.s.、pretrend 5% 被拒」的 robots_spec 记为「成立，pretrend 存疑」。按自己的规则，状态应为「未通过预注册规则；诊断性」，且 headline 锚定 broad23 +0.0193。
3. **`\S A` 泄漏**：第 32、63、110 行出现「\S A」，那是底稿分节名，编译后不存在。改为「内部审计记录」或具体节引用。
4. **sec13 与 sec11 路径编号错位**：sec13 推荐的「路径 (ii)」描述的是 sec11 的路径 (iii)（平台侧到达冲击的反向运用）。sec11 的 (ii) 是券商关闭型冲击。sec13、sec01 全部改为 (iii)，或反过来统一 sec11 的编号。**以 sec11 的定义为准，改 sec13/sec01。**
5. **sec07 估计量描述矛盾**：年度面板不可能做 12 期 lead（sec06 规定年度面板 L=3）；`did_imputation` 只接受单个聚类变量。按 sec06 的规定改写。
6. **sec11 拥挤度 moderator 违规**：按事件当日实测的同日申报家数是事件期分组，违反预定 moderator 规则，且与 sec06 自相矛盾。改用屏蔽前均值。
7. **sec11 `did_imputation` 两向聚类**：该命令不支持；且交互斜率不是它的输出。按两步法改写（先在未处理观测上填补，再对处理观测的填补差分回归）。
8. **sec06 交互项 TWFE 偏误**：式 (06-dvp-reg)/(06-hlt) 在交错处理下仍是被本节自己否定的 TWFE。补两步 imputation 版本，原式降为对照。
9. **sec06 年度面板两向聚类**：2016–2026 只有 10–11 个年份簇，远低于可靠下限。改为按公司聚类 + wild cluster bootstrap（`boottest`）。
10. **B 组 primary 五处不一致**：sec06/09/11/12/05 各写一版。以 sec06 §6.4 为唯一登记处，其余改为引用。A 组、C 组同理。
11. **篇幅**：59 页对 30–40 页目标。全文修剪 25%–30%，删跨节重复（理论在 3/4/9/10/12 重复；Essay 1 事实在 sec02 之外重复；预测在 4/5/7 重复）。
12. **文风**：二元对比句式「不是 X，是 Y」全文 20 余处，只保留 sec13 末句与 sec03 §3 标题；半角标点、\S 泄漏、机械三段式一并清理。
