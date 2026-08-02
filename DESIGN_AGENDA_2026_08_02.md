# DESIGN REVIEW AGENDA — Essay 2 (2026-08-02)

4 视角(identification/measurement/groups/scope) Opus 设计评审+综合。已知项(tier-2、B15-B18、C6、MDE 等15项)已排除。

## 1. [DO-NOW] 重建 treatment:按 rel_type×path 拆 sell/buy(现有 china_share 方向混合 + path 聚合 BUG)
(measurement) 已核实的真实 BUG:firm_quarter_cn 按 rel_type 分组丢掉 path,n_cn_customer 把 Path1/Path2 反向边混在一起;方向抵消是 null 的一级机械解释
**How**: 改 02(约 548-570 行 CASE WHEN 同时用 rel_type+path):n_cn_sell=Path1-CUSTOMER∪Path2-SUPPLIER,n_cn_buy=Path1-SUPPLIER∪Path2-CUSTOMER,各除对应方向分母得 sell_share/buy_share。重跑 02→06→build_c6;回归 dw~us*sell_share*S + us*buy_share*S(fq gq ig 双向 cluster)+ RI。两者反号且各自可辨=合并掩盖机制;两者皆 null=抵消假说被排除、null 更硬。

## 2. [DO-NOW, 修订版 2026-08-02 v2] Active-only:四组对称拆分 + Funds.STYLE 三状态打标
(groups) 本地核实:US 欧洲持仓中 Funds.STYLE='Index' 占市值 39.78%(2021Q4);NULL entity_type 桶(34.37%)里 95.99% 是 Index。近四成指数型资金"预计响应较弱"(非恒等零:申赎/指数调整/公司行动/相对价格仍会动它),pooled β₃≈(1−p)·β_active 是**待检验的稀释假说**。
**打标(v2,取代旧三层合并)**:主标签 = Funds.STYLE=='Index'→PASSIVE;明确非 Index 风格→ACTIVE;缺失/未匹配→UNKNOWN(**绝不当 active**)。FUND_TYPE='ETF'(仅覆盖 15.5%,且 index 钱里 OEF 占 6 成)、MANAGER_STYLE(仅 3.7%,Vanguard 常标 Generalist)、名称只作交叉核验;名称补录按 fund_id 白名单,**禁止品牌整体硬编码**(这些集团也有主动产品)。
**时间维度**:Funds master ≈ 2018-08 snapshot——post-2018 冲击 predetermined(优点);回填 2003-2017 =前视分类(披露或截断);匹配 MV 覆盖率 2021Q4≈95%→2023Q4≈90% 衰减。每季度报 passive/active/unknown MV share + matched share;主检验限制到分类覆盖稳定期。
**对称设计**:四组 US_ACTIVE/US_PASSIVE/NONUS_ACTIVE/NONUS_PASSIVE,各组自分母归一化(组维度 2→4,06 grid 参数化,工作量 1-2 天)。主检验 US_ACTIVE vs NONUS_ACTIVE;辅:pooled、passive vs passive、US 内 active vs passive、active-only flow。
**前置检查**:NONUS 侧 UNKNOWN 份额可能远大于 US(STYLE 对欧洲 UCITS 覆盖差)——打标后先按组×季度报 unknown share;若 NONUS unknown>~40%,降级为 US_ACTIVE vs US_PASSIVE 组内对比 + US_ACTIVE vs pooled NONUS 附注。
**实现**:STYLE join 加在 04 层(holdings_eom 有 fund_id),不动 03 重 ETL。与方向拆分的 06 下游重建合并成一次。

## 3. [Rank 2, 解释性诊断, 修订版 2026-08-02 v2] §7.6 flow 的会计分解(非"会计恒等式威胁")
(identification) **只适用于 shares-based flow(§7.6),不适用于主回归的 w**——w 是组内归一化组合权重,US/NONUS 之间无相加恒等式;主回归 §6 只需一句均衡解释语言(β₃=净差异化再配置,含均衡吸收),不构成识别威胁。
**正确恒等式**:flow_US + flow_NONUS + flow_R = Δout/out_{t-1}(R=未观测剩余部门:散户+内部人+未申报机构+战略持股,**不叫散户**;右边≠0 除非 float 不变,residual flow 用同一滞后分母构造并显式保留 issuance/buyback 项)。
**本地初步事实(已跑,待正式化)**:高-CN 三分位 firm-quarter 的 corr(flow_US, flow_NONUS) ≈ **+0.20**(stable-float 子样本与 winsorize 后仍 +0.20~0.23)——两组**同向流动**,共同资金流主导,对手盘 margin 在 residual 部门;corr≈−1 的预设不成立。
**四步正式诊断**:(1) 构造 flow_US/flow_NONUS/flow_residual/float_growth;(2) 报 raw/winsorized/stable-float 的相关系数**与斜率**;(3) 检验 residual 与共同机构 flow 对 CN_{t-1}×S_t 的响应——注意 CN×S 在 firm-quarter 层,fq FE 会整吸收,只能 firm FE + quarter FE 的弱设计,定位=描述性边界;(4) §7.6 estimand 改写为 "the differential ownership flow of US institutions relative to non-US institutions, rather than the gross response of US investors in isolation"。命名:"accounting decomposition of observed institutional and residual ownership flows"(不用 interference lower bound)。
**边界(重要)**:世界 B("US 想卖但价格吸收、数量不动")**不可由数量数据识别**——数量 null 是数量问题的正确答案;区分"无需求变化"与"需求变化被价格吸收"只能靠 H2.2 的价格证据。删除"不管哪种世界都赢"的说法。

## 4. [DO-NOW] NONUS 控制去污染:主对照改 Continental-EU domicile
(groups) NONUS 混入反号 China/HK-bloc、同向 Asia-allied、巨型伦理主权(挪威 GPFG),pooled 残差是成分意外,β₃ 可能双向有偏;§9 limit15 只想到同向衰减
**How**: I_ict_panel 保留原始 investor_country(04),仅改 06 分组 SQL:(1)主规格 NONUS 限 Continental-EU(剔 CN/HK/JP/KR/TW/SG/GCC/主权),重跑 headline;(2)分解诊断 {US,EU-home,Asia-allied,China-bloc,Sovereign} 看哪个 bloc 驱动 pooled、China-bloc 是否反号;(3)报 GPFG 在高-CN 公司 NONUS book 的份额。复用 06→panel→headline。

## 5. [DO-NOW] 处理时间尺度:post-2018 regime 拆分 + long-difference 存量调整
(identification) AR(1) 残差=高频 surprise,被差掉 GPR 持久成分;机构去风险是朝目标持仓的存量/regime 调整,单一线性 β₃ 检验的是错误的处理时间尺度(合并 #20 十五年 regime 抵消)
**How**: c6_panel 加 post2018=(rd_m>=2018m1):(a)四项分解 us*cn*S*post2018;(b)子样本 2018Q1-2023Q4 单跑 3-pairwise+RI;(c)把 w/ownership_share 的水平对紧张水平做 firm-FE long-difference 估存量调整;(d)年度聚合重估。脱钩期转负且 RI 显著=null 是 regime 抵消假象,headline 改条件性发现;仍 null=更稳。纯 Stata + outcome 重构。

## 6. [DO-NOW] 冲击符号非对称:S_pos/S_neg 分别交互
(scope) 脱钩是'坏消息卖、好消息不急回补'的粘性反应;对称线性交互把升级卖出与缓和买回摊成 0,系统性衰减(全样本 well-populated,规避 §7.3 尾部 few-cluster)
**How**: 构造 S_pos=max(S,0)、S_neg=min(S,0),跑 us_cn×S_pos 与 us_cn×S_neg 两个三交互 + 3-pairwise + RI,检验 H2.1 的'升级期 β₃<0、缓和期≈0'非对称。纯 Stata + 两次 RI,半天。

## 7. [DO-NOW] treatment 比率的规模/可见性混淆:补绝对暴露 + market_cap 分位交互
(measurement) 分母=Revere 边数使 china_share 对大/被密集覆盖公司系统性偏低,把投资者最会反应的显著公司推入 LOW;可见性(perceived≠potential)同理稀释(合并 #9)
**How**: (a)诊断 corr(china_share, ln n_supplychain_links) 与 corr(china_share, ln mcap)(预计强负);(b)替代 treatment:绝对暴露 ln(1+n_cn_customer+n_cn_supplier)、规模标度暴露,各跑 headline 3-pairwise+RI;(c)market_cap 三分位(lag 一期)作可见度,估 us*cn*S*High_vis 四重交互或高/低子样本各跑。高可见度仍 null=正文强化'无脱钩';仅高可见度显著=主 null 需重述为可见性稀释。边数与 mcap 皆在 parquet。

## 8. [DO-NOW] 剂量函数形式:CN 换 deciles/样条,并诚实披露 revenue 加权不可行
(measurement) count-share 线性三重交互假定单调剂量,但边数≠关键性/价值;若真实暴露是阈值/非单调函数,线性 β₃ 朝零平均(合并 #10 revenue 加权数据约束)
**How**: 把 CN 换 deciles 或阈值样条重估 β₃(1-2 小时),检验剂量是否非单调、高分位是否有信号。LIMITATIONS:实测 revenue_percent 覆盖 CUSTOMER 12.8%/SUPPLIER ~0% 且均为估计值→经济强度加权在现有 Revere 抽取下不可行,故用边数份额;可选在 CUSTOMER-only 有值子集做 rev-weighted vs count rank 相关佐证一致性。

## 9. [DO-NOW] 暴露特异性 placebo:对 IN/VN/MX 跑同一条 β₃ 谱
(scope) 把 null 从'设计测不出任何东西'升级为'可发表的中国特异性 null';若替代国也平则 China null 无信息,若近岸受益国出现正向 tilt 则 China null 立刻获对照式解释力
**How**: 复用 isomorphic 的 02_russia/06_russia 模板,'RU'→'IN'/'VN'/'MX',同 rel_type、同 bilateral-union、同 point-in-time 分类算 share_c;06 同 grid/zero-fill,HIGH=各自正值中位数;有对应 bilateral GPR 用之,否则退成 share_c×S_us_cn 或 share_c×post2018,跑 3-pairwise+RI。四国 β₃ 并排成 exposure-specificity 表(点估计/CRVE/RI p/σ-MDE)。每国一次 02→06→RI。

## 10. [DO-NOW] '中国'构念边界:{CN,HK,(TW)} 大中华稳健性
(measurement) 暴露只认 home_region='CN',漏计经港转口/港资实体签约、把中国集团德国子公司记为 EU-EU,系统性低估;投资者语境的对华暴露常是大中华
**How**: 诊断先行:统计对手方 home_region 分布,量化 EU-HK/EU-TW 边占潜在大中华边比例(单次 duckdb 扫描)。非平凡则把 CN 扩 {CN,HK}(及第三列 {CN,HK,TW}),重跑 02→06→headline+RI。子公司口径/ultimate-parent 归属 FUTURE。

## 11. [LIMITATIONS] zero-fill 覆盖非对称:改 intensive-only 重跑作界
(groups) US 13F 强制→zero-fill 多为真零;NONUS 按辖区覆盖→zero-fill 多为未观测持仓;US-minus-NONUS 差分下 extensive-margin 测量误差直接进 β₃,方向取决于 NONUS 何处最薄
**How**: 正文明写两臂 extensive margin 载不同测量误差,故 extensive-driven β₃ 变异不可干净解释。诊断:在 intensive margin(w_{t-1}>0 且 w_t>0,两组皆是)重跑 headline 与全 grid β₃ 比;吻合=覆盖非对称不驱动结果。可选按 domicile 对 NONUS 覆盖做 benchmark。复用现面板 + w_prev>0 过滤,约 2 小时。

## 12. [LIMITATIONS] 公司内生适应被 firm×quarter FE 吸收:写成设计边界
(identification) CN 是公司选择,高对华公司紧张期主动换供应商(CN 降而非 US 抛售),CN 水平又被 α_{i,t} 吸收→'成功适应'公司对 β₃ 零贡献,null 机械相容于'适应普遍发生'
**How**: LIMITATIONS 明写'设计对公司内生适应盲'。可选 DO-NOW:用首次观测 CN 冻结暴露重估(与 B17 共用面板),描述性展示高-CN 公司 CN 是否在紧张期系统性下降(适应的直接证据)。写作为主。

## 13. [DO-NOW] 把 null 做成正式 TOST 等价检验(不止 MDE)
(scope) 顶刊对 null 的门槛已从'报 MDE'升到'正式拒绝经济上重要的效应';§9 limit1'无法拒绝任何一方'恰是承认没拒绝任何东西;有 RI 分布等价检验几乎免费
**How**: 与导师预注册式设定经济阈值 δ(文献资金流 episode 幅度或 float 阈值)。用现有 RI 置换分布做 design-based 等价检验 H0:|β₃|≥δ,看观测 β₃ 是否落等价区间并给 p。同时把 master 表升级成 specification-curve/multiverse 图配 F10 多重检验说明。注意:应在 rank1 修好 treatment 之后再 TOST 最终规格。复用 RI 输出。

## 14. [ADVISOR] in-pipeline 直接对华持仓 benchmark + 弱先验 scope 表述
(identification) 欧股对华供应链暴露是 US 去风险的三阶边际;不先证'同批 US 基金在直接中概/ADR 上确有 HFCAA 时代减持',欧股 null 只能解释为'看错了边际'而非'US 不减持'(合并 #25 外部效度)
**How**: 用同一 LionShares holdings_eom 拉这批 US 基金对 sec_country∈{CN,HK}/中概 ADR 持仓,做同构三重差分或时序减持检验作 benchmark。测到已知中概减持→欧股 null='truly absent';连直接持仓都测不到→问题在功效/边际选择。LIMITATIONS 兜底:若导师不做 benchmark,写 external-validity/scope-conditions 段(总体=被机构持有的欧洲上市发行人;within-Europe 归一化 β₃ 只测欧洲内部再配置;§7.6 share-of-float 弥补绝对 stake 维度)。ADVISOR 定 Essay 定位与 universe。

## 15. [ADVISOR] 离散 policy-event 作主处理(GSDB/entity-list/tariff)替换 S_t
(identification) GPR-AR(1) 残差=报道强度惊奇,与有约束力的离散政策(2018 关税/2019 华为/2020 SMIC/2022 出口管制)既非同步也非同尺度;处理变异与经济相关变异错位,null 可能只反映处理太糊
**How**: 用已在库的 GSDB(数据源#5)对华 sanction 起始季 + 手编 entity-list/tariff 日历构造离散 P_t,在同一 C6 三重差分替换 S_t 并做 event-time DiD。离散事件仍 null→null 大幅增强;出信号→S_t 构念稀释。与 H2.2 事件研究复用同一日历。GSDB 半天 + 日历手编约 1 天。ADVISOR 拍板是否升为主处理。

## 16. [ADVISOR] 月度化:EOM 持仓做高频事件研究 / 提升反应速度分辨率
(measurement) 季度 Δw 抹平季内往返、outcome 与季末月 shock 同对齐,系统性错过较快反应;holdings_eom 与 GPR 皆月度,月度化可行(合并 #21 离散事件研究)
**How**: 基金子样本按月构造 ownership-share-of-float 面板 × 月度 S_m,firm×month + group×month FE;或对 Pelosi(2022-08)/首轮关税(2018)/Entity List 扩容做 [-3,+3] 月窗式 (High_CN)×US×Post,事件日横截面置换 RI;或折中做冲击后 [0,+3] 月局部投影。须导师定:(i)月度只 mutual fund/N-PORT 支撑,与 rank3 被动稀释交叉,月度样本偏非受压主体须披露;(ii)月度自相关下 RI 可交换性重定(block/circular-shift)。多日工程,先定方向。

## 17. [ADVISOR] 预设 US responder 子集(active-ex-index / ESG-mandate / public-pension)
(groups) 即便去掉被动钱,pooling 全部 active 仍把有反应理由的(ESG/被审查/受托约束)与无关的(quant/retail-facing)平均;这是经济 responder 限制,区别于 rank3 的机械非反应
**How**: 骑 rank3 的 manager-style join。导师拍板哪个定义是理论驱动且 PRE-SPECIFIED(按预定属性切合法,事后挑显著子组不可接受):(a)active US ex-index by MANAGER_STYLE;(b)ESG/exclusion 分类;(c)public-pension FUND_TYPE='PLP'。作 labelled group split,框成 heterogeneity map 非 headline swap,披露多重检验(F10)。每切约半天,主成本是导师预设决策。

## 18. [ADVISOR] 两文联合设计:数量 null 锐化 H2.2 价格解释,共享 primitives
(scope) 若 H2.2 发现高-CN 欧股高紧张期异常负收益而本文证 US 相对 NONUS 持仓不差异化变动,则价格效应无法由 US 特异性抛压/downward-sloping demand 解释,只剩基本面重定价——数量 null 是价格识别的关键前提
**How**: 向导师提:(1)H2.2 复用 02 的 CN 暴露与 06 的 HIGH 切点不另起口径;(2)在 H2.2 identification 里显式把本文数量 null 作为'排除 US price-pressure、留基本面重定价'的先验;(3)若价格动数量不动,讨论边际投资者身份(非 US 机构/散户接盘)。定位/写作决策,非新代码。

## 19. [ADVISOR] 州反华撤资立法作准外生的 which-US-investor 变异(DDD)
(groups) 最可信的横截面外生变异:养老金是否受强制撤资取决于州辖区而非公司基本面,可把弥散的组级处理变成 investor 级 DDD;但机制间接、立法晚且窄(多 2021-23),功效与机制映射双重不确定
**How**: 复用 01_master_files.jl §9 已搭的 pension/PLP/州识别(现仅描述性 CSV)。若绿灯:手编州撤资立法生效日 + pension→州 crosswalk,标记受强制 US pension,建 holder-type 级面板(须放松 06 组折叠),估 Δ(stake)~mandate×CN×post。约 1-2 周。导师定:间接的欧洲公司渠道是否值得手工采集。

# DO-NOW TOP 3
现有数据、无需导师、且互为前置/决定 null 可解释性的三件,应按此顺序做:

1) 【重建 treatment,修 path/方向 BUG】(rank1,约 1 天)。已核实真实 BUG:02_china_exposure.jl 的 eu_china_edge(360-387 行)带 path,但 firm_quarter_cn(548-570 行)只按 rel_type 分组、丢掉 path,导致 n_cn_customer 把 Path1 与 Path2 的反向 CUSTOMER 边混在一起。改 CASE WHEN 同时用 rel_type+path,产 sell_share/buy_share,重跑 02→06→build_c6,回归加 us*sell_share*S 与 us*buy_share*S 两列 + RI。这是一切下游的前置:在方向混合的 treatment 上跑出的 null 不可信,且方向抵消本身是 null 的一级机械解释。

2) 【market-clearing / interference 诊断】(rank2,约半天)。在 §7.6 已建好的 ownership_share 面板上,对高-CN firm-quarter 算 within-fq 的 corr(ΔShare_US, ΔShare_NONUS) 与未覆盖 float 残差份额对 S_t 的反应,把 β₃ 的 estimand 在 §6 明写为 market-clearing net differential。这决定 null 到底是不是'US 没减持'。零新数据、零建管线。

3) 【active-only US book】(rank3,约 1 天)。给 fund 打 passive flag(Funds ETF + Institutions Index/Passive + Vanguard/BlackRock/State Street 硬编兜 NULL-type 壳),04 并行输出 I_ict_active,过 06→build_c6→run_headline + RI,并在 active-only book 重算 w。作者自评为现有数据支持的最高价值稳健性;它直接判定 pooled null 是否是被动稀释的成分假象。

紧随其后的两件同样廉价、决定 null 说服力(可并入首轮):rank4 NONUS→Continental-EU(0.5-1 天,只改 06 分组 SQL)与 rank13 TOST 等价检验(约 1 天,复用 RI;但须等 rank1 修好 treatment 后对最终规格做)。

# ADVISOR QUESTIONS (email-ready)
1) Estimand framing: Because the US and NON-US arms clear against each other within the same firm-quarter, our triple-difference β₃ identifies a market-clearing net differential (a general-equilibrium quantity), not a partial-equilibrium US behavioral elasticity — a large true US pull-out fully absorbed by NON-US arbitrageurs would still produce a near-zero β₃. Are you comfortable defining and reporting the estimand explicitly as a "market-clearing net differential," and framing the null accordingly?

2) Universe and scope: European-listed firms' China supply-chain exposure is a third-order margin; the strongest de-risking prior lives in the same US funds' direct Chinese ADR / mainland-HK holdings. Should we add an in-pipeline benchmark showing whether these same US funds cut their DIRECT China holdings over the HFCAA/decoupling era, as a precondition for interpreting the European null — and does the weak-prior nature of the European channel change how you want the essay positioned?

3) Treatment construct: Our shock S_t is the AR(1) surprise in a newspaper-based geopolitical-risk index, which is neither synchronized with nor scaled to the binding discrete policy actions (2018 tariff lists, 2019 Huawei, 2020 SMIC, 2022 export controls). Should discrete policy events (GSDB sanction episodes plus a hand-coded entity-list / tariff calendar) become the PRIMARY treatment in the ownership triple-difference, or stay as a robustness check alongside the continuous shock?

4) Responder set and pre-specification: After removing mechanically non-responding passive/index money, which US responder subset should be PRE-SPECIFIED as the theory-motivated group — active-ex-index, ESG/exclusion-mandated funds, or public-pension (PLP) money — so the split is admissible under the predetermined-moderator rule rather than an ex-post "significant-subgroup" selection?

5) Null as deliverable and cross-essay design: If we commit to a credible-null contribution, do you approve (a) setting an economically meaningful equivalence threshold δ and reporting a formal TOST/equivalence test against it, and (b) having the price-channel essay (H2.2) reuse this essay's exact China-exposure measure and HIGH cutoff, so the quantity null can be cited as prior evidence that any price effect is fundamental repricing rather than US-specific price pressure?

# DROPPED (dup with known)
合并(同根因,已并入对应议程条,非因重复【已知项】而删):
- #19 (scope: US 被动稀释切 active/公养老金) → 并入 rank3,与 groups-lens 的 passive-dilution 是同一机制、同一 join plumbing,完全重复。
- #21 (scope: 月度 EOM 离散事件研究) → 并入 rank16 月度化 ADVISOR;两者同根因(季度折叠丢高频反应),连续月度 + 离散事件窗合为一条。
- #20 (scope: 按 2018 拆融合期/脱钩期) → 并入 rank5;其'post-2018 dummy × US × CN'与 identification-lens 的 regime/level 错配是同一 estimand,sketch 已复用。
- #9 (measurement: 可见性缺口) → 并入 rank7;与'比率分母规模偏差'同用 market_cap、同判大/显著公司被误分,合为一条(绝对暴露列 + 可见度交互)。
- #10 (measurement: revenue_percent 加权) → 折入 rank8 的 LIMITATIONS 部分;实测覆盖 CUSTOMER 12.8%/SUPPLIER ~0% 且均估计值,加权数据不可行,作诚实局限 + customer 子集验证,与 count-share 函数形式同属'边数≠经济权重'。
- #25 (scope: 外部效度边界) → 折入 rank14 的 LIMITATIONS 兜底;若导师不做 direct-China benchmark,则以 scope-conditions/weak-prior 段落落地,同根因(欧洲上市第三国暴露是弱先验)。

因重复【已知项/已完成工作】而剔除:无整条剔除。各提案均自证区别于既有 limitations 1/4/11/15/16 及 B16/B17/B22/item16,经核对成立;§7.6 shares-based(已完成,β₃=+0.0006 p=0.61)与 Russia 正控制(已建 02/06_russia 模板)被 REUSE 而非重复提出(rank2 借用 share 面板做新诊断,rank9 借用 Russia 模板扩到 IN/VN/MX),故保留不删。

保留但降级(feasibility 低、导师门槛高,置于议程尾部而非删):#15 州撤资立法 DDD(rank19)机制间接、立法晚窄、需数周手工采集,identification 价值高但现阶段不可即做。