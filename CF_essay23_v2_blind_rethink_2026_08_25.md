# Essay 2/3 选题 v2：盲重想与最终建议（2026-08-25）

**框架修正**：本文档取代 `CF_essay23_topic_proposals_2026_08_24.md`（v1）。v1 在错误的「AI 爬虫」框架下写成；essay 1 的真实处理是**搜索/发现型 bot 屏蔽**（十年面板），AI 子样本（~85 hosts）功效不足。v1 中仍然有效的部分：引用核验结果（含 KMN 两篇撤稿勿引）、工具链调研（§6）、拥挤地图中「不要去」的方向。v1 的方案 A（宕机×屏蔽交互）作废；方案 B（授权协议事件研究）与论文主线脱钩，降为可选独立项目。

**本轮方法**：先做 7 视角 JF/JFE/RFS 全扫描（认知理论 / 信息成本理论 / 媒体传播冲击 / 注意力 / 公司可见性行为 / 2020-26 前沿 / DV 菜单，约 90 篇经 URL 核验）；然后**盲重想**——3 个 ideation agent（微观结构、公司金融与披露、信息经济学-资产定价）只看 essay 1 正确事实 + 数据资产 + 文献地图，prompt 不含任何旧提案；最后由评审 agent 按五维打分（识别、数据复用、新颖性、独立成篇、功效）合并排名，并被明确要求敌视「在处理层掉数据」的设计。第二道引用对抗核验正在后台跑，结果回来在 §7 打补丁。

---

## 0. 一个重要的中间收获：essay 1 的理论 spine 排序

综合 agent 对 essay 1（非选题目标，但顺手产出）的裁决值得单独记录：

- **首选 spine：Kyle 深度/逆向选择构成**（Kyle 1985；Diamond-Verrecchia JF 1991 反向读；Easley-O'Hara JF 2004；Glosten-Milgrom JFE 1985 作为被拒绝的对照模型）。屏蔽只提高「依赖搜索的（散户/未知情）交易者」的发现成本 → 他们退出 → 知情流失去掩护 → **λ（price impact）上升而报价 spread 不必动**。这解释你的全部事实模式，包括 spread null——把 null 写成模型判别证据。你 geoecon 检验里「美国基金不减持股份额」的 null 也反过来支持构成/流量渠道而非减持渠道。
- **次选：Merton (1987) 认知集**——作为**异质性引擎**（shadow cost ∝ 相对规模×特质波动×(1-q)/q，直接预测小盘、低机构、窄基础集中），但它本身不预测 impact-up-spread-flat。Chen-Noronha-Singal (JF 2004) 的「认知向下粘性」（在位者不会忘记公司，损失作用于**新投资者流入**）恰好把你的 new/micro 集中解释成理论结果。
- **判别检验**（Kadlec-McConnell JF 1994 模板）：ΔAmihud 对 Δbreadth（Refinitiv 基金计数，Chen-Hong-Stein 构造）与 Δ成交构成回归——breadth 中介 = Merton；breadth 平而构成携带 = Kyle。数据全在手。
- 必引清单 20 篇与 4 个现成 spec（SVI first stage、Kadlec-McConnell horse race、信息含量期限结构、filing-window 吸收比率）见工作流输出存档，写 essay 1 修订时直接取用。

---

## 1. 盲重想的收敛信号（最有分量的结果）

三个想法被**不同视角独立提出**，细节到 DV 电池和识别方案都几乎重合：

| 收敛簇 | 提出者 | 收敛内容 |
|---|---|---|
| **价格发现期限结构**（delay、VR、R²、BPS/Dávila-Parlatore informativeness、filing-window 再配置） | microstructure + info-AP | 理论钩子都是 Dugast-Foucault 反向 + Gao-Huang 镜像 |
| **Merton 认知集 extensive margin**（新基金 initiation 而非持股份额；shadow cost 直测；SVI first stage） | microstructure + info-AP | 连「把已有 Δ-share null 重构为理论预测的非结果」都一致 |
| **披露供给响应**（8-K/guidance/IR 页；Balakrishnan 替代 vs Banerjee-Marinovic-Smith 流动性反馈赛马） | corp-disclosure + info-AP | 识别方案同为 vendor/template-wave 子样本 |

## 2. 评审排名（9 个合并后想法，五维各 1-10）

| # | 想法 | 视角 | 识别 | 数据 | 新颖 | 独立 | 功效 | 总分 |
|---|---|---|---|---|---|---|---|---|
| 1 | **Turning Off the Machines**（机器监督侵蚀，反向 Zhu） | corp-disc | 7 | 8 | 9 | 9 | 7 | **40** |
| 2 | **Where Prices Learn**（价格发现期限结构） | 收敛×2 | 7 | 9 | 8 | 6 | 7 | **37** |
| 3 | Substituting for Search（披露供给响应） | 收敛×2 | 5 | 8 | 8 | 8 | 6 | 35 |
| 4 | Brokerage Closures × Web Visibility | micro | 9 | 7 | 7 | 9 | 5 | 37* |
| 5 | Leaving the Awareness Set（投资者基础 extensive margin） | 收敛×2 | 7 | 9 | 8 | 7 | 5 | 36 |
| 6 | Findability and the Firm（融资/投资实际效应） | corp-disc | 6 | 9 | 6 | 8 | 3 | 32 |
| 7 | Managing the Machine Audience（战略择时） | corp-disc | 6 | 9 | 7 | 4 | 4 | 30 |
| 8 | The Price of Being Unfindable（预期收益/ICC） | info-AP | 5 | 9 | 6 | 5 | 3 | 28 |
| 9 | Who Trades the Invisible Firm（订单流构成） | micro | 7 | 4 | 8 | 4 | 3 | 26 |

\* 原始分并列第二，但排名被存在性风险压低（2016 后干净关闭事件可能太稀），评审要求先跑两周 kill-gate。

## 3. 推荐组合

**Essay 2 =「Turning Off the Machines: Machine Access and the Erosion of Algorithmic Monitoring」**
- RQ：公司屏蔽发现爬虫、机器读取其公开信息下降后，外部算法监督是否弱化到让内部人抽租（机会主义内幕交易更赚钱、盈余管理上升、申报纪律松弛）？
- 理论：**Zhu (RFS 2019) 反向**——另类数据覆盖到来降低内幕交易利润、提高投资效率；那么公司层面的机器访问**移除**应反向。Heese-Pérez-Cavazos-Peter (JFE 2022，地方报纸消亡→违规上升) 给「监督者消失」模板。你的 first stage（屏蔽后 EDGAR 机器访问 -11~-18%）是**已测量事实**，Zhu 当年只能假设。
- 为什么第一：它是唯一在你的处理层（小、低 IO、低覆盖）**不掉数据**的设计——Form 4、Compustat accruals、EDGAR 申报时效对微型股全覆盖，正面回应功效审查；结局家族属治理文献，与 essay 1 的流动性完全不同，salami 质疑不成立；文献空位干净（无 top-3 论文碰过「公司自己拔掉机器监督层」）。
- DV：Form 4 买卖后 BHAR（routine vs opportunistic 分类，Cohen-Malloy-Pomorski JF 2012）；盈余公告前交易倾向；修正 Jones 可操纵应计；NT/迟交申报与 8-K 时效；空头利息响应。全部免 TAQ。
- 识别：BJS + template/vendor-wave（Cloudflare 分类已建）的 firm-passive 子样本 + routine 交易和高 IO 公司双 placebo。
- 新工作量：EDGAR Form 4 批量抓取解析（约 4-6 周，基础设施现成）+ accruals 模块（2 周）。6-8 个月出稿。
- 风险与对冲：内幕收益 DV 噪声大 → 三个独立 DV 家族（内部人租金/应计/申报纪律）互为保险，按你的 MDE 规矩先用真实 reghdfe 聚类结构算最小可测效应，对照 Zhu 的效应量。

**Essay 3 =「Where Prices Learn: Discoverability and the Term Structure of Price Informativeness」**（两视角独立收敛）
- RQ：失去开放网络可发现性后，价格在申报之间学得更慢、价格发现向强制披露事件再配置吗？信息含量在哪个期限恶化？
- 理论：Grossman-Stiglitz 边际获取者退出 + **Dugast-Foucault (JFE 2018) raw-vs-processed 反向**：开放网络搜索是 raw/便宜渠道，终端/分析师是 processed 渠道；封 raw 渠道 → 短期限信息含量先掉、价格发现推迟到 scheduled releases（正好把 essay 1 的 filing-window 集中从「结果」升格为「研究问题」）。
- DV（全部 CRSP/Compustat/IBES 现货，零新采集）：Hou-Moskowitz delay、Lo-MacKinlay VR、Morck-Yeung-Yu 非同步性、Bai-Philippon-Savov 分期限 informativeness、Dávila-Parlatore (RFS 39(5) 2026) 口径、DellaVigna-Pollet 即时/总吸收比率×Hirshleifer-Lim-Teoh 拥挤度。
- 两个 essay 1 没有的独立性装置：**unblock 事件的不对称检验**（十年面板里有解封——认知粘性预测 block 与 unblock 效应不对称）+ Gao-Huang (RFS 2020) EDGAR 上线的镜像锚（审稿人必问，先写死）。
- 4-6 个月出结果，是 solo 两年约束下风险最低的一章。写作纪律：第一页就立住估计量差异（信息含量/效率 vs 交易成本）。
- 注意与外部 AI 提案的关系：期限结构直觉幸存，但对象从**分析师预测**换成**价格**——原因见 §4。DFF-analyst 模块（FY1-FY2 informativeness + 长端预测构成）作为本章一节，条件是先过 IBES FY2+ 覆盖率预检。

**Runner-up = Substituting for Search（披露供给响应）**：补全弧线且 standalone 干净，但识别最弱（披露与屏蔽同为公司选择，全押 template-wave 子样本）且 guidance 在处理组微型股大片为零。若 Essay 3 初稿读起来仍太像 essay 1，它顶上。启动前两个 gate：template-wave 子样本规模、8-K/IR-page 退路电池可行性。
- §7 核验补注：BMS (JF 2024) 的「披露随流动性递增」是条件性结论（仅在经理确定知情分支；不确定知情时可反号）。赛马设计据此升级：先用可观测代理（内部人交易活跃度、管理层持股、行业信息环境）划分「确定知情 vs 不确定知情」两组，再检验披露响应的符号是否按模型分支翻转——这比笼统的 Balakrishnan-vs-BMS 对撞更有判别力。

**两件无条件要做的便宜保险**：
1. Brokerage-closure 两周 kill-gate：数清 2016 后干净关闭事件数。若够，「Brokerage Closures × Web Visibility」凭 9 分识别+9 分独立性可临时升级顶掉任何一章——它是全家族唯一完全不依赖 essay 1 内生处理的设计。
2. Lou (RFS 2014) 式择时检验（block 日期前后 insider sales/SEO/坏消息聚集吗）：不做章，直接写进 Essays 2-3 的 identification 附录；日度面板是独家资产。

**论文总弧线**：同一冲击、三个文献的估计量——essay 1 流动性（微观结构）、essay 2 治理（监督）、essay 3 效率（信息含量）。评审语：比「同一冲击、两个市场质量 DV 加一个弱识别响应章」更站得住。

## 4. 与前两版提案的对照裁决（你要求的「如果还一样，说为啥」）

**殊途同归的（2.5/3）**：
- **披露供给响应**：v1 方案 A 的 essay 3、外部 AI 未提、盲重想两视角独立再现。同归原因：Balakrishnan (JF 2014) 模板 + 你的 Wayback 基础设施构成的组合太自然，任何认真读文献的人都会走到。但盲重想版本更诚实——识别分只有 5，被降为 runner-up 而非正选，主因是 v1 没充分正视的「guidance 在处理组为零」问题。
- **期限结构**：外部 AI 的核心直觉（horizon 结构判别机制）幸存并升为 essay 3 正选，但测量对象从 IBES 分析师预测换成价格。为什么换：① 处理组是低覆盖公司，分析师期限结构在处理层掉数据，价格度量（CRSP 周收益）不掉；② 实现值滞后问题（FY3+ 的 informativeness 要等 2026-27）价格度量没有；③ ATT 论文（Abis-Tang-Bian）占了分析师精度水平检验，价格空间更空。这是「同一直觉、更好的载体」。
- **投资者基础/Merton**：我 v1 排最后的 fund-rotation 直觉以更好的形式再现（extensive margin + shadow cost 直测），两视角收敛，但评审用与我当初相同的理由压到第 5：Refinitiv 只见基金，处理组低 IO 公司的 initiation 事件近乎不存在，理论说的暴露边际（散户）恰好不可观测。结论一致：做机制小节，不做章。
- **被淘汰且未复活的**：v1 方案 A essay 2（宕机交互，死于 AI 功效）；方案 B（授权协议，死于与 search 主线脱钩——作为独立 side project 仍可选）。

**真正的新东西**：**Turning Off the Machines（第 1 名）在此前三轮中从未出现**。为什么之前漏掉：前三轮全部锚定在「信息→流动性/信息含量」的结果家族里打转，没人跳到治理文献的槽位（反向 Zhu）。这正是盲重想+多视角设计的意义——corp-disclosure 视角不背流动性包袱，直接看见了「机器读取 = 监督技术」这层。我背书这个排名：它同时解决了新颖性（空位最干净）和功效（DV 在微型股全覆盖）两个此前所有提案的软肋。

## 5. 行动清单（按顺序）

1. 与导师确认组合（Essay 2 = 监督侵蚀；Essay 3 = 期限结构；runner-up = 披露响应）。
2. 跑三个 gate（合计约三周）：brokerage-closure 事件计数；template-wave 子样本规模；IBES FY2+ 覆盖率（决定 DFF-analyst 模块去留）。
3. Essay 1 修订顺手收编 §0 的四个 spec（SVI first stage 最优先——它同时是三章的机制地基）。
4. Essay 1 尽快挂 SSRN（不变的建议）。
5. Form 4 抓取管线立项（essay 2 的长杆项）。

## 6. 本轮扫描的关键文献（按用途，全部经检索 agent URL 核验；二道核验补丁见 §7）

**理论**：Merton JF 1987；Kyle 1985；Glosten-Milgrom JFE 1985；Diamond-Verrecchia JF 1991；Easley-O'Hara JF 2004；Grossman-Stiglitz AER 1980；Shapiro RFS 2002；Peng-Xiong JFE 2006；Dugast-Foucault JFE 2018 + JF 2025；Banerjee-Marinovic-Smith JF 2024。
**识别模板**：Gao-Huang RFS 2020（EDGAR 上线镜像）；Kelly-Ljungqvist RFS 2012；Balakrishnan et al. JF 2014；Derrien-Kecskés JF 2013；Peress JF 2014；Engelberg-Parsons JF 2011；Zhu RFS 2019；Heese et al. JFE 2022；Gao-Lee-Murphy JFE 2020；Eaton et al. JFE 2022。
**认知/可见性**：Kadlec-McConnell JF 1994；Foerster-Karolyi JF 1999；Amihud-Mendelson-Uno JF 1999；Chen-Noronha-Singal JF 2004；Grullon-Kanatas-Weston RFS 2004；Bodnaruk-Ostberg JFE 2009；Green-Jame JFE 2013；Lou RFS 2014；Solomon JF 2012；Focke et al. RFS 2020；Greenwood-Sammon JF 2025。
**注意力**：Da-Engelberg-Gao JF 2011；Ben-Rephael-Da-Israelsen RFS 2017；Barber-Odean RFS 2008；Huberman-Regev JF 2001；Fedyk JF 2024；Kempf-Manconi-Spalt RFS 2017。
**DV 工具箱**：Hou-Moskowitz RFS 2005；Goyenko-Holden-Trzcinka JFE 2009；Bai-Philippon-Savov JFE 2016；Dávila-Parlatore RFS 2026；Morck-Yeung-Yu JFE 2000；Chen-Hong-Stein JFE 2002；Boehmer et al. JF 2021（TAQ）；Back-Crotty-Li RFS 2018（TAQ）；Brogaard et al. RFS 2022（TAQ）；DellaVigna-Pollet JF 2009；Hirshleifer-Lim-Teoh JF 2009；Cohen-Malloy-Pomorski JF 2012。
**前沿定位**：Farboodi-Matray-Veldkamp-Venkateswaran RFS 2022；Goldstein-Spatt-Ye RFS 2021；Fang-Peress JF 2009（异质性指纹）。
**关键空位事实**：recent-frontier 视角确认 2020-2026 无任何 top-3 论文使用公司层面搜索引擎可见性/网站收录作为变量。

## 7. 二道引用核验补丁（2026-08-25 完成：44 篇，41 CONFIRMED / 3 CORRECTED / 0 捏造）

**实质性修正一处**：
- **Banerjee, Marinovic & Smith《Disclosing to Informed Traders》，JF 79(2) 2024, 1513-1578（doi:10.1111/jofi.13296）**。「披露概率随流动性递增」是**条件性结论，不是一般结论**：仅在「经理确定知情 + 披露有成本」分支成立；当投资者不确定经理是否知情时，披露可随流动性**递减**。不得引用为无条件命题。对 runner-up 提案的影响见 §3 补注——这其实让赛马设计更锋利：模型自身按可观测条件给出两个相反符号，实证要先判断小盘屏蔽公司落在哪个分支。

**次要修正两处**：
- Focke, **Ruenzi & Ungeheuer**（不是 Niessen-Ruenzi；三作者引全），RFS 33(10) 2020, 4676-4720。关键结论核验成立：广告拉动的短暂注意力对 turnover/流动性「几乎无影响」、短期收益不显著——essay 1 用它顶住「纯注意力故事」质疑的引法是安全的。
- Chen-Noronha-Singal JF 59(4) 2004：页码 1901-**1930**。

**核验确认的重要细节（引用时保真）**：
- Tetlock (RFS 2011) 陈旧信息：个体投资者对 stale news 交易、价格反应**随后数周反转**——是过度反应证据，不是有效吸收；引用措辞要按这个方向。
- Fedyk (JF 2024)：Bloomberg 头版位置（内容不变）→ 前 10 分钟成交量 +240%、|收益| +176%，1 小时内吸收完 vs 非头版 >2 天——「显著位置改变吸收速度」的量级锚。
- Merton (1987) shadow cost 公式、Kadlec-McConnell 全套归因、Gao-Huang staggered EDGAR、Balakrishnan et al. guidance 响应、Heese et al. 违规 +1.1%/罚款 +15.2%、Brogaard et al. 方差分解（noise 31%/私有 24%/公共 37%/市场 8%）等全部逐条属实。
- 本轮 44 篇 + 前两轮合计约 140 篇载重文献经双道核验，累计发现 14 处错误（2 撤稿、1 EoC、11 处归因/书目错误）。教训维持不变：**任何二手归因（包括本文档）写进 proposal 前对原文。**
