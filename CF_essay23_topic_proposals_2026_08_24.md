# Essay 2/3 选题设计文档（2026-08-24）

> ⚠️ **已被取代（2026-08-25）**：本文档在错误的「AI 爬虫」框架下写成（essay 1 实为搜索/发现型 bot 屏蔽，AI 子样本功效不足）。最新版见 `CF_essay23_v2_blind_rethink_2026_08_25.md`。本文档仍然有效的部分：§3 引用核验（KMN 两篇撤稿勿引等）、§6 工具安装清单、§1 拥挤地图的「不要去」方向。方案 A/B/C 全部作废或降级。

**目的**：回答「博士论文后两篇写什么」。基于四路独立调研的收敛结果：① 8 角度并行文献扫描（118 篇，SSRN/NBER/JF·JFE·RFS/econ top-5/JAR·JAE·TAR）；② deep-research 独立第二路系统检索（PRISMA 流程，33 篇入库）；③ 50 篇关键文献的对抗性引用核验（42 confirmed / 8 corrected / 0 fabricated）；④ 文献检索工具链调研。所有载重引文均经 OpenAlex/DOI/出版商页面核验。

**状态**：设计提案，待你与导师裁决。未启动任何实施。

---

## 0. 一句话结论

你的直觉对了一半：illiquidity / price informativeness 作为**因变量家族**应该保留，但它不是你的护城河——**真正的突破口是供给端处置（treatment）**：全市场只有你持有「公司层面、月度+日度、跨十年」的 robots.txt 面板。四路检索一致确认：**「公司主动切断 AI 对自身信息的访问 → 资本市场后果」这条赛道在 finance/accounting 里仍然无人占据**，而它两侧的相邻文献（需求端 GenAI 冲击、出版商屏蔽）已在 JAE/JAR 和 marketing 顶级 WP 层面快速收敛。结论：essays 2-3 应该沿着这条供给端赛道纵深展开，抢在相邻文献合流之前。

---

## 1. 文献地图：哪里已经挤满了（不要去）

以下方向经检索确认已被占据。每条列出占坑论文（全部核验过）。

**C1. 需求端 GenAI 可得性冲击 → 交易/流动性（市场级事件）**
- Cheng, Lin & Zhao, *JAE* 80(2-3) 2025, 101821：ChatGPT 宕机期间成交量约 -10%，短期 price impact 与收益方差下降。
- Bertomeu, Lin, Liu & Ni, *JAE* 80(1) 2025, 101782：意大利 ChatGPT 禁令 → spread 走阔、本国分析师产出与准确度下降；小盘低覆盖公司受损更重（与你 essay 1 的异质性同构）。
- Chang, Dong, Martin & **Changyun Zhou**, *JAR* 64(3) 2026, 1287-1331：ChatGPT 后散户交易与 AI 情绪信号对齐度大增，信息不对称下降。
- Ecker, Li, Li & Wu, *JAR* 64(3) 2026, 1375-1426：中国大型 GenAI 平台 170 万条股票查询；使用强度与 informed trading 上升、流动性下降相关。
- Blankespoor, Croom & Grant, *JAE* 82(2) 2026：券商 GenAI chatbot 40 万+散户查询 + 2000 人调查。
- Even-Tov, Lourie, Munevar & Nekrasov, SSRN 5778246（WP）：账户级数据 × 意大利禁令。
- Cao, Chen, Ma & Srinivasan, *JAR* 64(3) 2026, 1427-1450：2025 JAR conference 综述，三层框架（production / dissemination / processing）。
- 判决：在位者手握专有查询/账户数据，纯需求端设计已无空间。但这批论文的冲击（宕机、禁令、发布日）可以**变成你的工具**（见方案 A）。

**C2. AI 写披露 / AI-washing**：Blankespoor, deHaan & Li, *JAR* 64(3) 2026（新增披露文本至 4.5% 为 AI 所写）；Jia, Li, Ma & Xu, *RAST* 31(2) 2026（电话会议 GenAI 讨论 0.34%→9.25%）；Barrios-Campbell 团队 + 至少两个 AI-washing 团队并行 + SEC 执法。没空间。

**C3. LLM 读财报/预测收益**：Lopez-Lira & Tang, *JFE* 184 (2026) 104335（Sharpe 随采用扩散 6.54→1.22，价格效率改善）；de Kok, *Management Science* 71(9)（LLM 度量方法论）。Chicago Booth 系占据。⚠️ 重要例外见 §3。

**C4. AI 交易者理论/模拟**：Dou, Goldstein & Ji（NBER 34054 + 2026 续作 SSRN 5763222）；Colliard, Foucault & Lovo（*RFS* advance, hhag010）；Cont & Xiong（*Math Finance* 34(2)）。资深理论家锁场，学生模拟论文到场即被碾压。

**C5. 前 LLM 时代 EDGAR 机器阅读**：Cao, Jiang, Yang & Zhang, *RFS* 36(9) 2023（机器下载占比 2003 年 39% → 2016 年 78%，公司迎合算法改写披露）；Crane, Crotty & Umar, *MS* 69(6)；Sipeng Zeng & Kuo Zhou, "Textual Analysis by Hedge Funds", *JAR* 2026 online-first（678 家对冲基金 EDGAR 访问 × 13F）；Gao & Huang, *RFS* 33(4) 2020 与 Goldstein, Yang & Zuo, *JAR* 61(5) 2023（EDGAR 1993-96 staggered 实施的经典模板）。度量与结论都已被拿走；且你的 EDGAR log 无 IP/UA，无法复制投资者映射线。

**C6. 出版商/创作者对 GenAI 的内容供给反应（流量结果）**
- Zhao & Berman（Wharton），arXiv 2512.24968 v4 / SSRN 5992774：新闻出版商 robots.txt 屏蔽（staggered，Callaway-Sant'Anna）→ 六周内周流量约 **-7%**（⚠️ 引用注意：v1 的 23%/14% 已被 2026-04 v4 取代；v1 标题也不同）。**这是你最近的竞争者，但它做媒体行业和流量，不碰资本市场。**
- Peukert 等, SSRN 4807979：Unsplash 供图者在图片被开放商用 AI 后月上传量约 **-37%**（不是流传的 -20%）。
- Kim, Jin & Lee, NBER 34733：Pixiv 插画师上传被 AI 挤出。
- del Rio-Chanona 等, *PNAS Nexus* 3(9)：ChatGPT 后 Stack Overflow 活动 -25%。
- Agarwal & Sen, SSRN 6513059：现场实验，出现 Google AI Overview 时外链自然点击 **-39.8%**、零点击搜索 +34.5%（⚠️ 不是 -15%）。
- Longpre 等（NeurIPS 2024）：一年内 C4 最关键源 28%+ 被 robots.txt 限制。
- Zhou, Proserpio & Goli, SSRN 6858298：8,082 家新闻域名的屏蔽记录 × 五个 AI 平台引流——目前最接近「屏蔽的均衡后果」的实证，仍是媒体版。
- 判决：「屏蔽 → 流量损失」已被证明可发表。**资本市场版是 essay 1 的领地，必须尽快挂 SSRN 打时间戳。**

**C7. 另类数据/信息供给 → 流动性（前 AI 设计）**：Zhu, *RFS* 32(5) 2019；Katona, Painter, Patatoukas & Zeng, *JFQA* 60(2)（卫星数据加剧不对称、损害流动性——与 Zhu 相反的分配效应）；Dessaint, Foucault & Frésard, *JF* 79(3)（horizon effect）；Allee, Cating & Rawson, *RAST* 2023（地方报纸消亡 → spread/illiquidity 上升）；Sammon, *MS* 71(6)。用作 outcome battery 与设计模板，不作新 essay。

**C8. 数据经济学理论**：Jones & Tonetti, *AER* 110(9)；Farboodi & Veldkamp（*REStud* advance rdag074）；Liu, Ma & Veldkamp, *JFE* 169（2025 Jensen Prize 一等奖，data dilution）；Farboodi, Singal, Veldkamp & Venkateswaran, *RFS* 38(3)（数据估值与 illiquidity 直接相连）；Dugast & Foucault, *JF* 80(1)；Chan, NBER 35344（AI 与开放网络瓦解的市场设计模型）。借框架，不做理论。

---

## 2. 空白地带（检索确认无人回答，你的数据能答）

- **G1**：GenAI 交易渠道是否 firm-content-specific？所有已发表冲击都是市场级，无人有公司级 AI 可见性面板来做交互。只有你的日度面板能做。
- **G2**：企业 web 数据在资本市场值多少钱？AI 内容授权协议（News Corp-OpenAI >$250M/5yr、Reddit、Getty 等，Oderinwale & Kazlauskas, arXiv 2510.24990 已编目）**没有任何学术事件研究**。block-vs-license 边际在非媒体上市公司完全空白。
- **G3**：退出 AI 语料库的公司事后如何调整披露（网站披露量、8-K 节奏、filings 机器可读性）？Cao et al. (RFS 2023) 的 LLM 时代续集没人写；Boulland, Bourveau & Breuer (*JAR* 64(1) 2026) 已为 Wayback 网站度量提供了发表先例。
- **G4**：屏蔽训练爬虫后，AI 对该公司的表征是否停留在最后截止日（staleness）？Xu, Xuan & Zheng (SSRN 5651130) 记录了 stale-training-data 误导渠道，但从未连到公司自身的访问选择。
- **G5**：公司退出 AI 数据池时，哪类投资者退出股票？（fund-level 分配检验——注意低功效，见方案设计。）
- **G6**：training-blocking vs search-blocking 的差异动态（search 屏蔽即时生效，training 屏蔽在下个模型截止日才咬合）。你自己的 live-scan 发现（增长全部来自训练爬虫）是别人没有的变异。
- **G7**：屏蔽的决定因素（IP 密集度、诉讼暴露、机构压力）——单独成文只是 determinants paper，应折叠进其他设计做第一阶段。

---

## 3. 引用核验的重要发现（写作时必须遵守）

1. **两篇「文献锚点」已被作者撤稿，不能引用**：
   - Kim, Muhn & Nikolaev, "Bloated Disclosures"（arXiv 2306.10224）：v5（2025-10）撤稿声明——合作者复现失败，"分析未能支持所报告的发现"。从未发表于 JAE。
   - Kim, Muhn & Nikolaev, "Financial Statement Analysis with LLMs"（arXiv 2407.17866）：v3（2025-02）撤回修订中。
   - 论证「AI 处理披露有真实价值」时改用已发表且干净的 Cheng-Lin-Zhao (JAE 2025)、Bertomeu et al. (JAE 2025)、Ecker et al. (JAR 2026)。
2. **van Binsbergen, Han & Lopez-Lira (*RFS* 36(6) 2023) 被 RFS 挂 Expression of Concern（2026，调查未决）**——如引必须注明，勿作支柱。
3. Zhao & Berman 引 v4 的 ~7%，不引 v1 的 23%/14%；Agarwal & Sen 引 39.8% 外链点击下降，不引 "~15%"；Peukert et al. 引 ~37% 不引 ~20%；Chang et al. 第四作者 Changyun Zhou；Dávila & Parlatore 已落刊 *RFS* 39(5) 2026, 1267-1309。
4. 元教训：这个领域三年里已出 2 撤稿 + 1 EoC，全在最高引论文里。**每篇引文核验后再进正文**（与既有工作规矩一致）。

---

## 4. 收敛后的三个方案（按推荐顺序）

因变量家族统一升级为四件套：Amihud、Kyle-λ 型 price impact、Dávila-Parlatore (*RFS* 39(5)) informativeness 估计量、Sammon 式盘前信息份额。这是 RFS 审稿人现在的期望配置。

### 方案 A（推荐）：机制递进三部曲

**Essay 2 =「Invisible to the Machines」：公司级 AI 可见性与 GenAI 冲击的传导**
- RQ：GenAI 对市场的影响是否经由公司自己的机器可访问内容传导？
- 识别：2023-2026 十余次 ChatGPT/Claude 大宕机 + 模型发布日（公开时间戳）× **事前**（预定）屏蔽状态的高频 DiD 交互。屏蔽公司对宕机的敏感度应被削弱。JAE 已两次接受同款冲击，审稿风险集中在交互项而非冲击本身。
- 数据：日度 robots.txt 面板（月度面板做不了这个频率）+ CRSP 日度。新增数据仅为宕机时间戳，零专有数据。
- 差异化：把 C1 的全部在位论文从对手变成工具——他们无法构造这个 moderator。同时是 essay 1 price-impact 机制的样本外验证。
- 风险：交互项为 null（GenAI 交易可能走 filings/新闻商而非公司网站）。缓解：精确 null 本身是对 firm-content 渠道的定界结果，可发表但弱一档；用 G6 的 training/search 分层增加检验面。抢发风险低（面板独占）。
- 工期：数据已备，6-9 个月出初稿。

**Essay 3 =「How to Talk When the Machine Stops Listening」：屏蔽后的披露替代**
- RQ：屏蔽损害流动性（essay 1），公司在自己仍控制的渠道上是否补偿？
- 识别：围绕屏蔽日期的 staggered BJS；结果为网站披露量（Boulland-Bourveau-Breuer 方法）、8-K/press release 节奏、filings 机器可读性（Cao et al. 度量）；再检验披露反应是否调节流动性损害。
- 数据：treatment 面板即 essay 1 的；Wayback/Common Crawl 抓取是你的核心比较优势。需新抓 treated + matched control 公司的网站快照（工作量大但已被你的基础设施证明可行）。
- 差异化：Cao et al. (RFS 2023) 的 LLM 时代续集，机器的访问权本身成为 treatment；BBB (JAR 2026) 提供度量的发表先例。
- 风险：① 继承 essay 1 的内生性，可能被批 salami-slicing——缓解：结果变量完全不同（披露供给 vs 市场质量），把替代经济学（free-riding：AI 白拿就少发）放到前台；② Wayback 网站快照噪音——你自己 2026-06 的审计发现 sitemap/URL 层有漂移而 robots.txt 干净，度量必须吸收那次教训并对 BBB 基准年做验证。
- 工期：9-12 个月（抓取重）。
- 论文弧线：essay 1 后果 → essay 2 传导 → essay 3 企业反应。答辩叙事完整，全部复用同一套基础设施。

### 方案 B（进攻型）：把授权协议事件研究抢下来

**Essay 2 =「Paywalls for Robots」：AI 内容授权协议与企业数据的市场定价**
- RQ：2023-2026 授权协议浪潮（$5M-$250M+）是企业 web 数据第一批可观测价格；市场如何为数据资产定价？
- 设计：协议公告窗口的收益与流动性；同业 rival 溢出（Liu-Ma-Veldkamp dilution 逻辑预测同类数据持有者被重估）；独有部分——日度面板测每次协议公告后非当事公司的屏蔽行为迁移（block 作为谈判筹码的显示偏好：先屏蔽后授权 vs 从未屏蔽）。
- 数据：手工协议/判决目录（以 Oderinwale-Kazlauskas 编目起步，几周工作量）+ CRSP/Compustat + 日度面板。
- 风险：**全清单里抢发风险最高**——目录公开、设计快，资产定价组几个月就能跑完。要么第一个做，要么不做。当事上市公司只有几十家且偏媒体，own-firm 检验低 N；靠 rival 溢出与屏蔽反应两条腿撑论文。部分协议与诉讼新闻混杂。
- 工期：核心事件研究 6-8 周可挂 SSRN 抢时间戳，全文 6 个月。
- **Essay 3 = 方案 A 的「Invisible to the Machines」。**

### 方案 C（稳健型）：保留 geoecon

geoecon（当前 essay 2，主口径 p=.849）转型为 measurement + clean null 贡献保留，essay 3 取「Invisible to the Machines」。是否可行取决于导师对 null-result essay 的接受度（08-24 通话正好可议）。若 geoecon 保留，AI 赛道只占一篇，上面 G1-G4 的空白大概率被别人填掉。

### 不推荐单独成文
- 「Stale Minds」（G4）：度量新颖但 LLM-elicited data 审稿风险高、二阶段可能弱；可作为方案 A essay 2 的一节或后续期刊文。
- Fund-level 分配检验（G5）：效应集中在低机构持股公司，机械上没多少机构仓位可动，你在相邻设计里的 fund-level 检验也一直偏 null。做成某篇的一个 section，不做 essay。

---

## 5. 回答你原来的问题

「illiquidity 是不是突破口？」——是因变量家族的正确选择，但要升级度量（§4 四件套），并且认识到：**你的可辩护边界不在 outcome 端，在 treatment 端**。「严谨市场信息的变化」这个直觉，收敛成一句可答辩的 spine 就是：*firm-controlled machine access to corporate information*（公司对机器读取自身信息的控制权）。essay 1-3 全部挂在这根轴上，每篇的 treatment、结果、机制各自独立。

---

## 6. 立即行动清单（待你确认后执行）

1. **Essay 1 尽快挂 SSRN**。四路检索一致的最大风险是速度：Zhao-Berman 在活跃修订，NBER 开始收 crowd-out 论文，JAE 显示从流传到发表约两年。时间戳就是护城河。
2. 若选方案 B：先花 6-8 周做协议目录 + 核心事件研究，挂 SSRN，再回头补全。
3. 设置月度文献监测（C1/C6 两个最快的方向）。可用 schedule skill 建 cron 任务。
4. 工具安装（来自工具链调研，均已验证）：
   - `paper-search-mcp`（openags，2.5k stars，本周仍在维护）：一包覆盖 OpenAlex/Crossref/S2/arXiv/SSRN(best-effort)，可装成 Claude Code skill；
   - Ai2 官方 **Asta MCP**（`https://asta-tools.allen.ai/mcp/v1`，免费申请 key）：Semantic Scholar 唯一可靠通道（无 key 公共池实测连续 429）；
   - `zotero-mcp`（4.8k stars）：接你的 Zotero 库；
   - 可选：Wiley AI Gateway connector（claude.ai，JF 全文语义检索）；RFS/OUP 确认无 API；SSRN 确认无官方 API，最新 WP 用普通 WebSearch 最快。
   - 不要用任何 Google Scholar MCP（全是脆弱爬虫）。

---

## 7. 已核验文献总表（两路合并去重，60 条）

状态标记：[P]=published，[WP]=working paper，[F]=forthcoming（已核验），[撤稿]，[EoC]。仅列载重文献；每条经 OpenAlex/DOI/出版商页核验。

**需求端 GenAI 冲击**
1. [P] Cheng, Lin & Zhao (2025) ChatGPT outages, *JAE* 80(2-3), 101821. doi:10.1016/j.jacceco.2025.101821
2. [P] Bertomeu, Lin, Liu & Ni (2025) Italy ban, *JAE* 80(1), 101782. doi:10.1016/j.jacceco.2025.101782
3. [P] Chang, Dong, Martin & Zhou (2026) AI democratization, *JAR* 64(3), 1287-1331. doi:10.1111/1475-679x.70063
4. [P] Ecker, Li, Li & Wu (2026) GenAI user-platform data, *JAR* 64(3), 1375-1426. doi:10.1111/1475-679x.70051
5. [P] Blankespoor, Croom & Grant (2026) Investor processing, *JAE* 82(2). doi:10.1016/j.jacceco.2026.101908
6. [P] Bradshaw, Ma, Yost & Zou (2026) Seeking Alpha, *JAR* 64(3), 1233-1286. doi:10.1111/1475-679x.70053
7. [P] Cao, Chen, Ma & Srinivasan (2026) 综述, *JAR* 64(3), 1427-1450. doi:10.1111/1475-679x.70061
8. [WP] Even-Tov, Lourie, Munevar & Nekrasov (2025) SSRN 5778246
9. [WP] Stanco & Chung (2025) SSRN 5171287
10. [WP] Cho, Huang, Pacelli & Rennekamp (2025-26) SSRN 6479459
11. [WP] Huang, Hugon, Zhang & Zheng (2025-26) SSRN 6682498（对照:Xue-Zhang-Zhu SSRN 5821142 报告相反方向）
12. [WP] Sheng, Sun, Yang & Zhang (2024-25) GenAI 与资管, SSRN 4786575
13. [WP] Bhagwat, Cookson, Dim & Niessner (2025-26) LLM 分歧度量, SSRN 5375473
14. [WP] Chen, Green, Gulen & Zhou (2024-25) LLM 外推偏差, SSRN 4941906（R&R MS）

**机器阅读与披露**
15. [P] Cao, Jiang, Yang & Zhang (2023) *RFS* 36(9), 3603-3642. doi:10.1093/rfs/hhad021
16. [P] Crane, Crotty & Umar (2023) *MS* 69(6), 3241-3262. doi:10.1287/mnsc.2022.4466
17. [P] Blankespoor, deHaan & Marinovic (2020) *JAE* 70(2-3), 101344
18. [P] Gao & Huang (2020) *RFS* 33(4), 1367-1411（EDGAR lead article）
19. [P] Goldstein, Yang & Zuo (2023) *JAR* 61(5), 1699-1733. doi:10.1111/1475-679X.12496
20. [P] Boulland, Bourveau & Breuer (2026) Company Websites, *JAR* 64(1), 81-125. doi:10.1111/1475-679X.70007
21. [P] Blankespoor, deHaan & Li (2026) GenAI in Financial Reporting, *JAR* 64(3), 1189-1232. doi:10.1111/1475-679x.70050
22. [P] Zeng & Zhou (2026) Textual Analysis by Hedge Funds, *JAR* online-first. doi:10.1111/1475-679x.70074
23. [P] Jia, Li, Ma & Xu (2026) *RAST* 31(2), 649-703. doi:10.1007/s11142-025-09916-1
24. [P] de Kok (2025) *MS* 71(9), 7888-7906. doi:10.1287/mnsc.2023.03253
25. [WP] Cao, Liang & Moon (2023-25) Machine readership, SSRN 4650113
26. [WP] Jha, Qian, Weber & Yang (2024) ChatGPT and corporate policies, NBER 32161
27. [撤稿] Kim, Muhn & Nikolaev, Bloated Disclosures（arXiv 2306.10224 v5=撤稿声明）。**勿引。**
28. [撤稿] Kim, Muhn & Nikolaev, Financial Statement Analysis with LLMs（arXiv 2407.17866 v3=撤回）。**勿引。**
29. [P] Lopez-Lira & Tang (2026) *JFE* 184, 104335. doi:10.1016/j.jfineco.2026.104335
30. [P][EoC] van Binsbergen, Han & Lopez-Lira (2023) *RFS* 36(6), 2361-2396。RFS 2026 挂 Expression of Concern，引用须注明。
31. [P] Coleman, Merkley & Pacelli (2022) Robo-analysts, *TAR* 97(5), 221-244
32. [P] Grennan & Michaely (2021) *JFQA* 56(6), 1877-1907

**robots.txt / AI 爬虫 / 内容供给**
33. [WP] Zhao & Berman, arXiv 2512.24968 **v4**(2026-04) / SSRN 5992774。屏蔽 → 周流量约 -7%（v1 旧数字勿引）
34. [P] Longpre et al. (2024) Consent in Crisis, NeurIPS 2024 D&B, arXiv 2407.14933
35. [P] Liu, Luo, Shan, Voelker, Zhao & Savage (2025) ACM IMC. doi:10.1145/3730567.3732913
36. [P] Fan et al. (2025) Web crawling opt-outs, COLM 2025, arXiv 2504.06219
37. [WP] Bouchaud & Ramaciotti (2025) arXiv 2510.09031（GPTBot 屏蔽 34.2%，高事实性媒体 55%）
38. [WP] Chan (2026) AI and the Collapse of the www, NBER 35344
39. [WP] Agarwal & Sen (2026) AI Overviews 现场实验, SSRN 6513059（外链点击 -39.8%）
40. [WP] Zhou, Proserpio & Goli (2026) LLMs as Gatekeepers, SSRN 6858298
41. [WP] Oderinwale & Kazlauskas (2025) AI 训练数据经济学（协议目录）, arXiv 2510.24990
42. [WP] Xu, Xuan & Zheng (2025) AI agents mislead investors, SSRN 5651130
43. [WP] Fan, Liu, Song & Wang (2024) 金融市场错误信息度量, SSRN 4922648
44. [WP] Emett, Eulerich, Pickerd & Wood (2024) Deepfake 财经新闻实验, SSRN 4869830

**内容供给挤出（跨平台证据）**
45. [WP] Peukert, Abeillon, Haese, Kaiser & Staub, SSRN 4807979 / CESifo 11099（Unsplash 上传 -37%）
46. [WP] Kim, Jin & Lee (2026) Pixiv, NBER 34733
47. [P] del Rio-Chanona, Laurentsyeva & Wachs (2024) *PNAS Nexus* 3(9)（Stack Overflow -25%）

**数据经济学与流动性理论**
48. [P] Jones & Tonetti (2020) *AER* 110(9), 2819-2858
49. [F] Farboodi & Veldkamp, *REStud* advance. doi:10.1093/restud/rdag074
50. [P] Liu, Ma & Veldkamp (2025) Data Sales and Data Dilution, *JFE* 169, 104053（2025 Jensen Prize 一等）
51. [P] Farboodi, Singal, Veldkamp & Venkateswaran (2025) Valuing Financial Data, *RFS* 38(3), 938-980
52. [P] Farboodi, Matray, Veldkamp & Venkateswaran (2022) *RFS* 35(7), 3101-3138
53. [P] Dugast & Foucault (2025) *JF* 80(1), 211-258
54. [P] Dessaint, Foucault & Frésard (2024) *JF* 79(3), 2237-2287
55. [P] Zhu (2019) *RFS* 32(5), 2021-2061
56. [P] Katona, Painter, Patatoukas & Zeng (2025) *JFQA* 60(2), 551-579
57. [P] Sammon (2025) *MS* 71(6), 4582-4598
58. [P] Dávila & Parlatore (2026) *RFS* 39(5), 1267-1309. doi:10.1093/rfs/hhaf051
59. [P] Abis & Veldkamp (2024) *RFS* 37(1), 89-118；Veldkamp (2023) *RoF* 27(5)；Goldstein (2023) *RoF* 27(1)
60. [WP/P] AI 交易理论：Dou-Goldstein-Ji NBER 34054 + SSRN 5763222；Colliard-Foucault-Lovo *RFS* advance hhag010；Cont-Xiong *Math Finance* 34(2), 467-521；Bergemann-Bonatti-Smolin Cowles 2425/EC'25；[F] Eisfeldt, Schubert & (M.B.) Zhang, NBER 31222, forthcoming *JF*；[P] Babina, Fedyk, He & Hodson (2024) *JFE* 151, 103745

---

## 8a. 追加（2026-08-25）：DFF 期限结构模块的核验与占位扫描

背景：外部 AI 建议将 essay 1 重定位为"search bots"、essay 2 做"Search Frictions and the Term Structure of Financial Information"。评估结论：**重定位不成立**（传统搜索 bots 无处理变异；屏蔽 GPTBot 不影响 Google 索引，其提议的 search-visibility first stage 对真实处理机械为零；2026-08-18 live scan 确认 AI-search 命名持平、增长全在 training 爬虫）；**但 DFF 期限结构设计值得并入方案 A 的 essay 2**。两项核查结果：

**A. DFF (JF 79(3) 2024) 逐条核对原文**（Wiley 全文 + 两个 WP 版本）：
- 属实：R² informativeness 构造（realized earnings 对该分析师横截面预测回归）；样本 IBES 1983-2017、65,889,122 个 analyst-day-horizon 观测（StockTwits 检验用 2005-2017 子样本约 3160 万）；基线斜率每年约 -12pp（新样本 -16.66）；短端 (h≤1) +0.54~0.66pp；**期限结构绕约一年期旋转、无整体平移**；机制条件（低 β 或高 multitasking cost；workload 代理；高 earnings persistence 缓冲长端损失）。
- 修正四处：① **原文没有任何 2SLS/IV**（外部 AI 称"2SLS 结果类似"系捏造）——识别是 analyst/time FE 的 OLS + Capital IQ 新闻流证伪检验；② exposure 是覆盖公司的**平均**非加总；③ 不存在"覆盖组合固定在 2009 年"的版本——真实稳健性是 fixed-portfolio 分析师子样本（exposure 按当期覆盖）；④ 长端恶化从 **h>2** 起（2-3 年桶已显著为负），非"3 年以上"——功效门槛可从 FY3+ 放宽到 FY2+。引用具体列系数前人工核对 Table VI。

**B. 占位扫描**（16 组检索 + DFF 全部 81 篇施引文献逐条核查）：
- **槽位 (a)（爬虫限制/web 可见性 × 分析师预测）：OPEN**。四条路径（SSRN/NBER/OpenAlex/引文网络）均未检出。
- **槽位 (b)（access-reduction 冲击 × 预测期限结构）：PARTIALLY TAKEN**。占位者 = Abis, Tang & Bian (2025) "Breaking the Data Chain"（SSRN 5334566）：Apple ATT 使 app 流量类 alt data 失真 → 依赖该数据的分析师精度下降、价格效率变差；明确以 DFF multitasking 框架立论，**且 Dessaint 本人是讨论人**。但其分析师端只做了季度 EPS 精度（PMAFE）水平检验，**没有任何 FY2+/长期 informativeness 或 rotation-vs-level 分解**；其基金端结论是经理"跨股票"转移注意力而非同一公司内跨期限转移（对 within-firm rotation 预期是一个诚实的降温信号）。唯一做过期限构成 rotation 的准实验是 Bayes 的分析师流感论文（产能冲击，非信息获取冲击）。相邻：Li & Shi "The Real Cost of Machine Readability"（SSRN 4752770，machine readability 提高挤出价格中长期信息——同 horizon 逻辑、价格端）。
- **定位要求**：与 Abis-Tang-Bian 的区分点必须是 ① rotation 分解本身 + ② firm 自身选择的供给端边际（公司把自己移出语料 vs 平台级数据精度冲击）。此槽位受活跃威胁（ATT 团队随时可补 horizon 检验），若做需要快。

**并入方式**：essay 2（Invisible to the Machines）加 IBES FY1-FY5 informativeness 期限结构模块，按 DFF 真实装置构造（平均 exposure、fixed-portfolio 子样本、证伪检验），以 earnings persistence 异质性做机制区分。**先决条件 = 功效预检**：统计处理组公司在 IBES 中 FY2+/FY3+ 预测的覆盖率（一个下午的 Stata 工作）；注意 2026-05-20 的 adjacent-channel 电池中 broad23 口径 IBES 水平量已是 clean null——模块的生死在长端。

## 8. 检索局限（诚实披露）

- 「无直接竞争者」是两路共约 40+ 次查询的**未检出**，不是不存在证明；该空间移动极快，正式启动 essay 2 前应复查一轮（建议同时设月度监测）。
- SSRN 检索依赖 Google site-search，2025-26 最新会计圈 WP 可能有漏。
- Semantic Scholar 公共 API 全程限流，核验以 OpenAlex/DOI/出版商页替代。
- 付费全文未读，结论基于摘要与公开版本。
