# Pilot Test Protocol v2 — AI Search Information Accessibility
# Research-grade design with baseline checks, replication, and per-question source tracking

---

## PART 0: COMPANY ROSTER & CORPORATE DOMAIN REGISTRY

| Company   | Ticker | robots.txt Status     | Corporate Domains (pre-defined)                                                         |
|-----------|--------|-----------------------|------------------------------------------------------------------------------------------|
| Amazon    | AMZN   | Blocks all AI bots    | aboutamazon.com, sustainability.aboutamazon.com, ir.aboutamazon.com, press.aboutamazon.com |
| Walmart   | WMT    | Explicitly allows AI  | corporate.walmart.com, one.walmart.com, tech.walmart.com                                 |
| NVIDIA    | NVDA   | Explicitly allows AI  | nvidia.com, investor.nvidia.com, blogs.nvidia.com, nvidianews.nvidia.com                 |
| Apple     | AAPL   | No AI rules (default) | apple.com, investor.apple.com, nr.apple.com, machinelearning.apple.com                   |

**Rule**: A source counts as "Corporate" ONLY if its domain matches one of the above. Everything else is Third-party or SEC.

---

## PART 1: BASELINE CHECK (do this FIRST, before any search test)

For each company, record the following in a spreadsheet BEFORE running the search test. This separates "policy effect" from "technical accessibility."

### 1a. robots.txt Policy Baseline
For each company, visit `https://[main corporate domain]/robots.txt` and record:
- Does it mention GPTBot? (Allow / Disallow / Not mentioned)
- Does it mention ClaudeBot? (Allow / Disallow / Not mentioned)
- Does it mention OAI-SearchBot? (Allow / Disallow / Not mentioned)
- Does it mention Claude-SearchBot? (Allow / Disallow / Not mentioned)
- Does it mention Googlebot? (Allow / Disallow / Not mentioned)

### 1b. Direct Fetch Baseline
For each corporate domain, try opening in a normal browser and record:
- HTTP status (200 / 403 / 429 / redirect)
- Is the page content visible in a plain HTML view, or does it require JavaScript rendering?
- Mark: Static HTML / JS-heavy / Mixed

Record in this table:

| Company | Domain | GPTBot | ClaudeBot | OAI-SearchBot | Claude-SearchBot | HTTP Status | JS-heavy? |
|---------|--------|--------|-----------|---------------|------------------|-------------|-----------|
| Amazon  | aboutamazon.com | | | | | | |
| Amazon  | sustainability.aboutamazon.com | | | | | | |
| Walmart | corporate.walmart.com | | | | | | |
| ... | ... | | | | | | |

---

## PART 2: SEARCH TEST — PROMPT TO COPY

### Execution Rules:
1. **Each company × each run = one independent NEW conversation** (no continuation)
2. **Each company runs 3 times** (Run A, Run B, Run C) to measure variance
3. **Randomize company order** across runs. Suggested schedule:
   - Run A: Walmart → Apple → Amazon → NVIDIA
   - Run B: NVIDIA → Amazon → Walmart → Apple
   - Run C: Apple → NVIDIA → Amazon → Walmart   ← note: not → Walmart
4. **Platform**: Claude.ai with Web Search ON (toggle the 🔍 icon)
5. **Optional repeat on ChatGPT** with Search enabled (tests OAI-SearchBot)

### The Prompt (copy-paste, replace [COMPANY] and [DOMAINS]):

---

I need you to search the web and answer 5 questions about **[COMPANY]**. This is for academic research on AI information retrieval.

**CRITICAL INSTRUCTIONS — follow exactly:**

1. **Use web search for EVERY question.** Do not rely on pre-trained knowledge. Search even if you think you know the answer.
2. **For every factual claim**, cite the source URL in parentheses right after the claim.
3. **Prioritize high-quality sources**: corporate websites, SEC filings, and major news outlets (Reuters, Bloomberg, CNBC, WSJ, FT). Use lower-quality SEO sites only as supplements, never as sole evidence.
4. **After EACH question**, list the sources used for that specific question in a mini-table.
5. **After ALL questions**, provide a combined source summary.
6. **If you attempted to access any URL but could not retrieve content**, explicitly say so — this is critically important.

**For reference, [COMPANY]'s corporate website domains are:** [DOMAINS]

---

**Q1 [Financial — EDGAR-sufficient]:**
What was [COMPANY]'s total revenue, net income, and diluted EPS for its most recently reported fiscal quarter? Provide exact numbers, the quarter/year, and the filing date.

*Sources used for Q1:*
| URL | Domain | Type (Corporate / SEC / Third-party) |

---

**Q2 [AI Strategy — Website-incremental]:**
What is [COMPANY]'s current AI strategy as of early 2026? List specific AI products, investments, partnerships, or initiatives announced in the past 12 months. Include announcement dates, dollar amounts, and named projects where available.

*Sources used for Q2:*
| URL | Domain | Type (Corporate / SEC / Third-party) |

---

**Q3 [ESG/Sustainability — Website-incremental]:**
What are [COMPANY]'s most recent sustainability or ESG metrics? Provide specific numbers: Scope 1 & 2 greenhouse gas emissions (metric tons CO2e), percentage of energy from renewable sources, waste reduction metrics, water usage, and climate targets with deadlines.

*Sources used for Q3:*
| URL | Domain | Type (Corporate / SEC / Third-party) |

---

**Q4 [Recent News — Website-exclusive]:**
What are the 3 most significant announcements, product launches, or corporate news items from [COMPANY] in the **last 30 days** (approximately late February to March 20, 2026)? Provide the exact date for each item.

*Sources used for Q4:*
| URL | Domain | Type (Corporate / SEC / Third-party) |

---

**Q5 [Product Detail — Website-exclusive]:**
Describe [COMPANY]'s most recently launched product or service as of March 2026. When exactly was it announced? What are its key features, target market, and pricing?

*Sources used for Q5:*
| URL | Domain | Type (Corporate / SEC / Third-party) |

---

## COMBINED SOURCE SUMMARY

| # | Source URL | Domain | Type (Corporate / SEC / Third-party) | Used in Q# |
|---|-----------|--------|--------------------------------------|------------|

**Summary statistics:**
- Total unique sources: ___
- Corporate website sources (from [DOMAINS]): ___
- SEC/EDGAR sources: ___
- Third-party sources: ___
- URLs attempted but FAILED to retrieve: ___ (list them)
- Any explicit "I could not access this page" messages? (yes/no, details)

---

## PART 3: DATA RECORDING TEMPLATE

After all 12 conversations (4 companies × 3 runs), fill this master table:

| Company | Run | Q1 Corp% | Q2 Corp% | Q3 Corp% | Q4 Corp% | Q5 Corp% | Total Corp | Total Sources | Failed URLs | Access Denied Messages |
|---------|-----|----------|----------|----------|----------|----------|------------|---------------|-------------|----------------------|
| Amazon  | A   |          |          |          |          |          |            |               |             |                      |
| Amazon  | B   |          |          |          |          |          |            |               |             |                      |
| Amazon  | C   |          |          |          |          |          |            |               |             |                      |
| Walmart | A   |          |          |          |          |          |            |               |             |                      |
| Walmart | B   |          |          |          |          |          |            |               |             |                      |
| Walmart | C   |          |          |          |          |          |            |               |             |                      |
| NVIDIA  | A   |          |          |          |          |          |            |               |             |                      |
| NVIDIA  | B   |          |          |          |          |          |            |               |             |                      |
| NVIDIA  | C   |          |          |          |          |          |            |               |             |                      |
| Apple   | A   |          |          |          |          |          |            |               |             |                      |
| Apple   | B   |          |          |          |          |          |            |               |             |                      |
| Apple   | C   |          |          |          |          |          |            |               |             |                      |

**Corp% = (# corporate website sources) / (total sources) for that question**

---

## PART 4: WHAT TO LOOK FOR (Evaluation Criteria)

### Primary outcome: Corporate site citation share by question type

**Expected pattern if mechanism exists:**

| Question Type | Amazon (Block) | Walmart/NVIDIA/Apple (Allow) |
|---------------|----------------|------------------------------|
| Q1 (EDGAR-sufficient) | Low Corp% | Low Corp% (no difference = placebo ✓) |
| Q2-Q3 (Website-incremental) | Lower Corp% | Higher Corp% |
| Q4-Q5 (Website-exclusive) | Lowest Corp% / zero | Highest Corp% |

### Secondary outcomes:
- **Failed URL count**: Amazon should have more "could not access" messages
- **Answer freshness**: For Q4, are Amazon's "recent news" items actually recent, or stale?
- **Answer completeness**: For Q3 (ESG metrics), does Amazon's answer have fewer specific numbers?

### Interpretation guidelines (conservative):
- ✅ **Supportive**: Amazon Corp% consistently lower across Q2-Q5, with explicit "could not access" messages for aboutamazon.com
- ⚠️ **Ambiguous**: Mixed results — Amazon Corp% lower in some runs but not others, or no access failure messages
- ❌ **Against mechanism**: No difference in Corp% across companies, or Amazon Corp% is comparable to others
- **Note**: Even strong results from 4 companies × 3 runs = preliminary evidence only. A proper test needs 40-80 firms stratified by market cap and media coverage.

---

## QUICK-START (minimum viable test)

If you want to start small:
1. Do Part 1 baseline for Amazon + Walmart only
2. Run the prompt 1 time each on Claude.ai (2 conversations)
3. Compare the Corp% and failed URL counts
4. If the pattern is clear, expand to full protocol
