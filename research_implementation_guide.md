# Research Implementation Guide: Step-by-Step Data Pipeline

> **Paper**: "Do as I Say or Do as I Do? AI Disclosure, AI Accessibility, and the Corporate Information Environment"
> **Created**: 2026-03-11

---

## Overview: 7 Data Modules

| # | Module | Frequency | Automation | Timeline |
|---|--------|-----------|------------|----------|
| 1 | Sample universe (Russell 3000 firm list + URLs) | One-time + annual refresh | Semi-auto | Month 1 |
| 2 | robots.txt panel (AIBlock, AIAccessIndex) | Monthly scrape | **Fully auto** | Month 1→ongoing |
| 3 | AI Disclosure from 10-K/10-Q (AIDisclosure) | Annual/Quarterly | **Fully auto** | Month 2-3 |
| 4 | AI Hypocrisy construction | After modules 2+3 | **Fully auto** | Month 3-4 |
| 5 | LLM Accuracy test (LLMAccuracy) | Quarterly or semi-annual | Semi-auto | Month 5-7 |
| 6 | Financial/market data merge | Annual/Monthly | Semi-auto (WRDS) | Month 7-8 |
| 7 | Regression & analysis | One-time | **Fully auto** | Month 8-13 |

---

## Module 1: Sample Universe

### What to do
1. Get the Russell 3000 constituent list (as of a specific reconstitution date, e.g., June 2023, June 2024, June 2025).
2. For each firm, identify:
   - CIK number (for EDGAR)
   - Ticker / PERMNO / GVKEY (for CRSP/Compustat merge)
   - **Corporate website URL** (the main one, e.g., `apple.com`)
   - **Investor relations URL** (e.g., `investor.apple.com`)

### Data sources
| Data | Source | Cost | Notes |
|------|--------|------|-------|
| Russell 3000 list | FTSE Russell via WRDS, or Bloomberg | WRDS access | WRDS has historical index constituents |
| CIK numbers | SEC EDGAR company search (`efts.sec.gov/LATEST/search-index?q=...`) | Free | |
| Corporate URLs | Compustat (`weburl` field), or SEC EDGAR filing headers | Free/WRDS | Compustat has a `weburl` column but it's often stale |
| IR page URLs | Need manual/semi-auto collection | Free | Typically `investor.[domain].com` or `[domain].com/investors` |

### Can Claude Code automate this?
- **Yes, partially.** I can write a Python script that:
  - Queries EDGAR's full-text search API for CIK → ticker mapping
  - Scrapes SEC filing headers for company URLs
  - Generates candidate IR URLs by pattern (`investor.X.com`, `X.com/investors`, `X.com/investor-relations`)
- **Manual part**: Verifying ~3000 URLs is tedious. Strategy: auto-generate candidates, then batch-verify with HTTP HEAD requests. Flag failures for manual check (~10-15% might need manual correction).

### Existing work
- **Compustat** has a `weburl` field — start here, then validate.
- **EDGAR company file**: `https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=&CIK=...` — free, programmatic.
- **No existing academic dataset** maps Russell 3000 to corporate + IR URLs systematically. This would itself be a minor data contribution.

---

## Module 2: robots.txt Panel (AIBlock, AIAccessIndex)

### What to do

**Step 2a: Current snapshot (monthly going forward)**

Every month, for each firm $i$:
1. Fetch `https://[corporate-url]/robots.txt`
2. Fetch `https://[ir-url]/robots.txt` (if different domain)
3. Parse the file: for each AI crawler User-Agent, record Allow/Disallow
4. Record: date, firm_id, url, and status for EACH crawler individually
5. Compute: `AIRetrievalBlock_{i,t}`, `AITrainBlock_{i,t}`, `AIAccessIndex_{i,t}`, and blocking pattern (A/B/C/D)

**Crawlers to check** (organized by function):

**Training crawlers** (blocking = IP protection, NOT information management):
| Crawler | Operator | Launched | Official doc |
|---------|----------|----------|-------------|
| `GPTBot` | OpenAI | Aug 2023 | https://platform.openai.com/docs/bots/gptbot |
| `ClaudeBot` | Anthropic | 2024 | https://support.claude.com/en/articles/8896518 |
| `Google-Extended` | Google | Sep 2023 | https://developers.google.com/search/docs/crawling-indexing/overview-google-crawlers |
| `CCBot` | Common Crawl | pre-2023 | https://commoncrawl.org/ccbot |

**Retrieval/search crawlers** (blocking = information management, CORE variable):
| Crawler | Operator | Launched | Official doc |
|---------|----------|----------|-------------|
| `OAI-SearchBot` | OpenAI | Late 2024 | https://platform.openai.com/docs/bots/oai-searchbot |
| `Claude-SearchBot` | Anthropic | 2025 | https://support.claude.com/en/articles/8896518 |

**User-initiated crawlers** (blocking = prevent direct user access):
| Crawler | Operator | Official doc |
|---------|----------|-------------|
| `ChatGPT-User` | OpenAI | https://platform.openai.com/docs/bots/chatgpt-user |
| `Claude-User` | Anthropic | https://support.claude.com/en/articles/8896518 |

**Traditional search crawlers** (controls — blocking these = generic anti-bot):
| Crawler | Operator |
|---------|----------|
| `Googlebot` | Google |
| `Bingbot` | Microsoft |

**Blocking pattern classification** (key for the paper):

| Pattern | Training bots | Retrieval bots | Googlebot | Motive | Use in paper |
|---------|--------------|----------------|-----------|--------|-------------|
| A | Allow | **Block** | Allow | **Information management** | Core sample |
| B | **Block** | Allow | Allow | IP/training protection | Contrast group |
| C | **Block** | **Block** | Allow | Mixed motives | Core sample |
| D | **Block** | **Block** | **Block** | Generic anti-bot | Noise / control |

**Important temporal note**: OAI-SearchBot and Claude-SearchBot only exist from late 2024+. For Aug 2023 – late 2024, we can only observe GPTBot/ClaudeBot (which mixed training + retrieval). The paper handles this by using GPTBot as the primary variable for the early period, and the retrieval-specific crawlers for the later period.

**Step 2b: Historical data (Aug 2023 → present)**

Use the **Wayback Machine** (`web.archive.org`):
1. Query `https://web.archive.org/web/20230801*/https://[firm-url]/robots.txt`
2. Collect all archived snapshots between Aug 2023 and present
3. Parse each snapshot the same way
4. Fill gaps with linear interpolation or last-observation-carried-forward

### Can Claude Code automate this?
- **Yes, fully.** I can write:
  - A Python script using `requests` + `robotsparser` (stdlib `urllib.robotparser`) to fetch & parse robots.txt
  - A Wayback Machine CDX API query script for historical data
  - A monthly cron job / scheduled task
  - Output: clean CSV panel `(firm_id, date, gptbot_blocked, claudebot_blocked, ...)`

**Estimated code**: ~200 lines of Python. ~2-3 hours of scraping for 3000 firms per month.

### Existing datasets / papers
| Resource | What it has | Limitation |
|----------|-------------|------------|
| Paul Calvano (2025) blog post | Aggregate stats on robots.txt AI blocking for top-1000 websites | Not firm-level, not for public companies |
| Reitinger (2025) SSRN #5555040 | Large-scale robots.txt analysis | General websites, not public firms; no financial data merge |
| Dark Visitors (`darkvisitors.com`) | Database of known AI crawlers and their User-Agent strings | Useful reference for crawler names, but no firm panel |
| Wayback Machine CDX API | Historical robots.txt snapshots | Coverage varies; large firms ~90%+ coverage, small firms ~50-60% |

**No existing academic dataset provides a firm×month panel of AI crawler blocking for publicly traded companies.** This is your novel contribution.

### Practical concern: Wayback Machine coverage
- Large-cap firms (S&P 500): likely 80-90% of months covered
- Mid-cap: maybe 50-70%
- Small-cap Russell 3000: maybe 30-50%

**Mitigation**:
- Start with S&P 500 or Russell 1000 as primary sample
- Use Russell 3000 as extended sample with appropriate caveats about survivorship/coverage bias
- Begin your own monthly scraping ASAP to build forward-looking panel

---

## Module 3: AI Disclosure from 10-K/10-Q (AIDisclosure)

### What to do

For each firm-year (10-K) and firm-quarter (10-Q):
1. Download full-text filing from EDGAR
2. Split into sentences
3. Classify each sentence: does it mention AI?
4. Compute: `AIDisclosure_{i,t} = AI sentences / total sentences`

### Step 3a: Download filings from EDGAR

Use the **EDGAR Full-Text Search API** or **bulk download**:
- Bulk filings: `https://efts.sec.gov/LATEST/search-index?q=...` or EDGAR Archives
- Python packages: `sec-edgar-downloader`, `edgartools`, or direct HTTP

### Step 3b: Text extraction

10-K filings on EDGAR come in several formats:
- `.htm` / `.html` (most common post-2000)
- `.txt` (plain text, older filings)

You need to:
1. Strip HTML tags, tables, headers/footers
2. Extract clean body text
3. Split into sentences (using `nltk.sent_tokenize` or `spacy`)

### Step 3c: AI sentence classification

**Three approaches (in order of credibility for a top journal):**

| Method | Description | 审稿人怎么看 (Reviewer perception) | Your paper should... |
|--------|-------------|--------------------------------------|---------------------|
| **Keyword dictionary** | Regex match: "artificial intelligence", "machine learning", "deep learning", "neural network", "NLP", "LLM", etc. | Accepted, standard practice. Loughran-McDonald dictionary is the gold standard in accounting/finance. | **Use this as primary method** |
| **BERT/FinBERT classifier** | Fine-tune a BERT model to classify "AI-related sentence" vs not | More sophisticated, increasingly expected in top journals (2024+). Good for validation. | **Use as robustness / validation** |
| **LLM (GPT-4/Claude) classifier** | Prompt an LLM: "Is this sentence about AI?" | Cutting-edge but controversial. Reviewer may ask: "Is GPT-4 reliable? What about hallucination?" | **Use for labeling training data for BERT, not as primary method** |

### 你能在论文里说用LLM处理吗？

**可以，但要分层说：**

1. **Primary method**: keyword dictionary（传统方法，无争议）
2. **Validation**: LLM-based classification on a random subsample to check keyword approach的recall和precision
3. **Robustness**: FinBERT classifier trained on LLM-labeled data

这样说的好处：
- 审稿人不能说你方法不严谨（keyword是标准的）
- 你展示了方法论awareness（用LLM验证）
- 就算LLM做了大量工作，你可以说"we validate our keyword approach using LLM classification and find X% agreement"

### 至于代码是Claude Code生成的？

**这完全不是问题。** 几乎所有实证论文的代码都是用某种工具辅助写的。你不需要在论文里说"code generated by Claude"——就像你不会说"code written with GitHub Copilot autocomplete"一样。你说 "We implement the classifier in Python using [libraries]" 就够了。但是你要：
- 理解每一行代码在做什么
- 能在seminar上解释方法论细节
- 做好code review，确保没有bug

### Can Claude Code automate this?
- **Yes, fully.** I can write:
  - EDGAR download script (bulk or API-based)
  - HTML → clean text pipeline
  - Keyword dictionary classifier
  - FinBERT fine-tuning script (using `transformers` library)
  - LLM-based validation script (using OpenAI/Anthropic API)
  - Output: `(firm_id, fiscal_year, filing_date, total_sentences, ai_sentences, ai_disclosure_ratio)`

**Estimated effort**: ~500 lines of Python. Processing time: ~1-2 days for all Russell 3000 10-Ks (2020-2025).

### Existing work / datasets
| Resource | What | Useful? |
|----------|------|---------|
| **Cao, Liu, Qiu, Zhao (2024)** SSRN #4987085 | AI disclosure measures from 10-K | **Directly relevant** — they may share data or at least their keyword list. Email the authors. |
| **WRDS SEC Analytics Suite** | Pre-parsed 10-K text, readability, sentiment | Has full text; you still need to do AI keyword extraction yourself |
| **Loughran-McDonald Master Dictionary** | Finance-specific word lists | No AI-specific dictionary, but methodology is the template |
| **FinBERT** (Huang, Wang, Yang 2023) | Pre-trained BERT for finance text | Use for sentence-level classification |
| **10-K text from WRDS** | Clean, parsed 10-K sections | Saves you the HTML stripping step |

**Strong recommendation**: Email Cao, Liu, Qiu, Zhao (2024) and ask if they'll share their AI keyword dictionary and/or processed data. It's common practice and saves months of work.

---

## Module 4: AI Hypocrisy Construction

### What to do

1. Merge Module 2 (AIBlock panel, aggregated to fiscal-year) with Module 3 (AIDisclosure)
2. Compute:
   - `AIHypocrisy_{i,t} = AIDisclosure_{i,t} × AIBlock_{i,t}` (primary, continuous × binary)
   - `AIHypocrisy_C_{i,t} = AIDisclosure_{i,t} × (1 - AIAccessIndex_{i,t})` (fully continuous)
3. Generate descriptive stats: 2×2 table, time trends, industry breakdown

### Can Claude Code automate this?
- **Yes, trivially.** This is a `pandas` merge + column multiplication. ~20 lines of code.

### Frequency alignment (as in the proposal)
- For annual analysis: use fiscal-year-end AIBlock value
- Robustness: modal status, FracBlocked
- For quarterly: match 10-Q quarter with corresponding AIBlock months

---

## Module 5: LLM Accuracy Test (LLMAccuracy)

### What to do

This is the **most novel and labor-intensive** module. For each firm $i$ in the sample:

**Step 5a: Design question set**

Create $Q$ factual questions per firm (suggest $Q = 10-20$):

| Question type | Example | Ground truth source |
|--------------|---------|-------------------|
| Basic facts | "Who is the CEO of [Firm]?" | Proxy statement / 10-K |
| Revenue segments | "What are [Firm]'s main business segments?" | 10-K Item 1 |
| Recent events | "What acquisitions has [Firm] made in the past year?" | 8-K filings, press releases |
| Strategy | "What is [Firm]'s sustainability strategy?" | IR page, sustainability report |
| Financial metrics | "What was [Firm]'s revenue in FY2024?" | 10-K |
| Forward-looking | "What risks does [Firm] identify in its latest filing?" | 10-K Item 1A |

**Important**: Some questions should be answerable ONLY from IR/website content (not from EDGAR mandatory filings). This isolates the blocking channel.

**Step 5b: Query LLMs**

For each firm × question:
1. Query **with web access enabled** (ChatGPT Browse, Claude with web search)
2. Record the full response
3. Score against ground truth

**Step 5c: Scoring**

| Dimension | Scale | How to score |
|-----------|-------|-------------|
| Accuracy | 0/1 | Is the factual claim correct? |
| Completeness | 0-5 | How many relevant details included? |
| Recency | Date | How current is the information? |
| Confidence | 0-1 | Does the LLM hedge / say "I don't have access"? |

### Can Claude Code automate this?
- **Partially.** I can write:
  - API scripts to query OpenAI (GPT-4o with browsing) and Anthropic (Claude with web search)
  - Automated scoring for factual accuracy (name matching, number comparison)
  - **But**: ground truth collection needs human validation for a subsample
  - Cost estimate: ~$0.01-0.05 per query × 3000 firms × 15 questions × 2 LLMs = **$900-$4,500** in API costs per wave

### Key methodological choice

**你要控制LLM的web access设置：**

| Test condition | What it tests |
|---------------|---------------|
| LLM with web access ON | Real-world information environment (the thing investors actually use) |
| LLM with web access OFF | Baseline: what's in training data only |
| Difference (ON - OFF) for blocked vs. unblocked firms | **The key test**: does blocking reduce the INCREMENTAL value of web access? |

如果一个公司block了GPTBot：
- Web access ON: GPTBot不能爬该公司网站，但能爬新闻、EDGAR等
- Web access OFF: 只有训练数据

如果blocking真的有信息效果，那blocked firms的 (ON - OFF) gap应该比unblocked firms更小。

### Existing work
- **No existing academic paper** does exactly this kind of LLM accuracy test for corporate information. This is truly novel.
- Related: Evaluations of LLM factual accuracy exist in CS/NLP (TruthfulQA, FactScore), but not applied to corporate finance questions.

---

## Module 6: Financial/Market Data Merge

### What to do

Standard empirical finance data pull:

| Variable | Source | WRDS table | Frequency |
|----------|--------|-----------|-----------|
| Firm size, B/M, leverage, R&D | Compustat | `comp.funda` | Annual |
| Stock returns, volume | CRSP | `crsp.dsf` / `crsp.msf` | Daily/Monthly |
| Bid-ask spread | CRSP or TAQ | `crsp.dsf` (spread) or TAQ for intraday | Daily |
| Amihud illiquidity | Computed from CRSP | `crsp.dsf` (|ret|/volume) | Monthly |
| Analyst forecasts, dispersion | I/B/E/S | `ibes.statsumu_epsus` | Monthly |
| Institutional ownership | Thomson Reuters 13F | `tfn.s34` | Quarterly |
| Earnings announcements | Compustat/I/B/E/S | `comp.fundq` dates or `ibes.actu_epsus` | Quarterly |
| 10-K Fog Index (control) | Computed or WRDS SEC Analytics | | Annual |
| Industry (SIC/NAICS) | Compustat | `comp.funda` | Annual |

### Can Claude Code automate this?
- **Data pull**: I can write WRDS SQL queries / Python `wrds` library code. You need a WRDS account (your university likely has one).
- **Merge**: Standard `gvkey-permno-cik` linking tables. I can write this.
- **Variable construction**: Amihud ratio, bid-ask spread, etc. — standard, I can write all of it.
- **Estimated code**: ~300 lines of Python/SQL.

### Existing resources
- **WRDS Research Applications**: pre-built SAS/Python code for most of these variables
- **Open Source Asset Pricing** (Chen & Zimmermann, 2022 RFS): replication code for many standard variables
- Your university's WRDS subscription is essential. Check with your library.

---

## Module 7: Analysis & Regression

### What to do

Run the three-stage empirical design from the proposal:

| Stage | Regression | Key output |
|-------|-----------|-----------|
| 1 | `AIBlock ~ AIDisclosure + controls + FE` | Who blocks? Correlation with disclosure |
| 1 | Descriptive: 2×2 table, time trends, industry heatmap | Prevalence of AI hypocrisy |
| 2 | `LLMAccuracy ~ AIBlock + AIDisclosure + controls + firm FE` | Does blocking degrade LLM info? |
| 2 | Diff-in-diff: web-ON vs web-OFF accuracy gap | Mechanism test |
| 3 | `Spread/Dispersion ~ AIBlock × AIDisclosure + controls + firm FE` | Market consequences |
| 3 | Staggered DID (Callaway-Sant'Anna) | Causal evidence |
| 3 | IV (peer blocking rate) | Robustness |

### Can Claude Code automate this?
- **Yes, fully.** I can write:
  - `statsmodels` / `linearmodels` for panel regressions with fixed effects
  - `pyfixest` for high-dimensional FE and clustered SE (faster, R's `fixest` equivalent)
  - Callaway-Sant'Anna: Python package `csdid` or R package `did`
  - IV: `linearmodels.iv` in Python
  - Output: publication-quality LaTeX tables (`stargazer` style)

---

## Complete Automation Assessment

| Module | Can Claude Code do it? | You must do manually | Cost |
|--------|----------------------|---------------------|------|
| 1. Sample universe | Write scripts, ~80% auto | Verify ~300-500 URLs | Free |
| 2. robots.txt scraping | **100% auto** (write + run) | Nothing (review output) | Free (just bandwidth) |
| 3. AI disclosure NLP | **95% auto** | Validate ~200 sentences | Free (EDGAR is free) |
| 4. Hypocrisy construction | **100% auto** | Nothing | Free |
| 5. LLM accuracy test | Write scripts, ~70% auto | Design questions, validate scoring | **$1K-5K API costs** |
| 6. Financial data | Write queries, ~90% auto | Run on WRDS (need account) | WRDS subscription |
| 7. Regressions | **100% auto** | Interpret results | Free |

### What you CANNOT automate with Claude Code
1. **WRDS data access**: I can write the SQL/Python, but you need to run it with your credentials
2. **Ground truth validation**: Someone needs to spot-check LLM answers against actual firm data
3. **API costs**: LLM queries cost money ($1K-5K depending on scale)
4. **Intellectual judgment**: Which results tell a coherent story? What to emphasize?

---

## Recommended Execution Order

```
Month 1:  [START HERE]
  ├── Set up Python environment (I write requirements.txt)
  ├── Build Russell 3000 URL mapping (Module 1)
  └── Start monthly robots.txt scraping (Module 2a) ← DO THIS ASAP

Month 2-3:
  ├── Wayback Machine historical robots.txt (Module 2b)
  ├── Download 10-K filings from EDGAR (Module 3a)
  └── Build AI keyword classifier (Module 3b-c)

Month 3-4:
  ├── Compute AIDisclosure, merge with AIBlock → AIHypocrisy (Module 4)
  └── Email Cao et al. (2024) for their keyword dictionary

Month 4-5:
  ├── Descriptive stats and Stage 1 regressions (Module 7, partial)
  └── Design LLM question set (Module 5a)

Month 5-7:
  ├── Run LLM accuracy tests (Module 5b-c) ← most expensive step
  └── Stage 2 regressions (Module 7)

Month 7-9:
  ├── Pull WRDS financial data (Module 6)
  ├── Full merge of all modules
  └── Stage 3 regressions (Module 7)

Month 9-13:
  └── Paper writing, robustness, conference prep
```

---

## Crawler-Level Heterogeneity Test (你之前问的)

你提出的问题（封的是哪个crawler）可以做一个很好的cross-sectional test：

| Pattern | 封了什么 | 没封什么 | 可能的动机 | 对论文的含义 |
|---------|---------|---------|-----------|-------------|
| A | GPTBot, ClaudeBot | CCBot, Googlebot | 不想AI实时回答问题引用公司信息 | **信息管理动机** ← 这是你paper的核心故事 |
| B | CCBot | GPTBot, ClaudeBot, Googlebot | 不想数据被用于训练 | IP/版权保护动机，不是信息管理 |
| C | GPTBot, ClaudeBot, CCBot | Googlebot | 反所有AI，但允许搜索 | 混合动机 |
| D | 全部（`*`） | 无 | Generic anti-bot | Noise，应该被排除或控制 |

**建议**: 把Pattern A作为你最clean的subsample。论文可以说 "Our results are strongest among firms that selectively block AI retrieval crawlers (GPTBot, ClaudeBot) while allowing search indexing (Googlebot) and training crawlers (CCBot)—consistent with an information management motive rather than IP protection or generic security."

---

## Summary Answer to Your Questions

**Q: 用LLM处理年报可以吗？**
可以，但作为validation/robustness，不作为primary method。Primary用keyword dictionary（标准方法）。

**Q: 代码是Claude生成的有关系吗？**
完全没关系。你不需要说明代码是谁写的。但你必须理解代码、能解释方法。

**Q: 每个步骤能不能自动化？**
大部分可以。robots.txt scraping、NLP、regression全部可以自动化。WRDS需要你的账号，LLM测试需要API费用。

**Q: 有现成数据库吗？**
没有现成的"公司AI blocking"数据库——这就是你的贡献。但10-K文本（EDGAR）、财务数据（WRDS）都是现成的。
