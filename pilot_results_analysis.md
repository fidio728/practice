# Pilot Test Results Analysis — UPDATED
## Including Corporate Domain robots.txt Audit
## Date: March 20, 2026

---

## 1. CRITICAL FINDING: Retail vs Corporate Domain Divergence

Amazon has TWO completely different robots.txt policies:

| Domain | Nature | AI Bot Policy |
|--------|--------|--------------|
| www.amazon.com | Retail/e-commerce | BLOCKS ALL (ClaudeBot, Claude-SearchBot, GPTBot, OAI-SearchBot, ChatGPT-User, Google-Extended, CCBot) |
| aboutamazon.com | Corporate news/IR | NO AI RESTRICTIONS (User-agent: * allowed) |
| sustainability.aboutamazon.com | ESG reporting | NO AI RESTRICTIONS |

This means our earlier spot-check table was MISLEADING -- it checked retail domains but our paper is about corporate information accessibility.

---

## 2. Full Corporate Domain Robots.txt Audit (10 Firms)

| Company | Corporate/IR Domain | AI Search Bots (Claude-SearchBot, OAI-SearchBot) | AI Training Bots (ClaudeBot, GPTBot) | Classification |
|---------|--------------------|-------------------------------------------------|--------------------------------------|---------------|
| Amazon | aboutamazon.com | ALLOW (no rules) | ALLOW (no rules) | OPEN |
| Walmart | corporate.walmart.com | EXPLICITLY ALLOW (named) | ALLOW (ClaudeBot named) | OPEN+ |
| NVIDIA | investor.nvidia.com / nvidianews.nvidia.com | ALLOW (no rules) | ALLOW (no rules) | OPEN |
| Apple | investor.apple.com / apple.com | ALLOW (no rules) | ALLOW (no rules) | OPEN |
| JPMorgan | jpmorganchase.com | ALLOW (no rules) | ALLOW (no rules) | OPEN |
| ExxonMobil | corporate.exxonmobil.com | ALLOW (no rules) | ALLOW (no rules) | OPEN |
| Pfizer | pfizer.com | ALLOW (no rules) | ALLOW (no rules) | OPEN |
| Tesla | ir.tesla.com | ALLOW (no rules) | ALLOW (no rules) | OPEN |
| Goldman Sachs | goldmansachs.com | ALLOW (no rules) | PARTIAL BLOCK (GPTBot + ChatGPT-User blocked on /research/, /insights/, /alumni/) | PARTIAL |
| **Meta** | **about.meta.com** | **NOT LISTED (falls under * = limited)** | **BLOCK (ClaudeBot, GPTBot, Google-Extended: Disallow /)** | **BLOCK** |

### Key distinction for Meta:
- about.meta.com: Blocks ClaudeBot, GPTBot, Google-Extended. Claude-SearchBot and OAI-SearchBot are NOT listed -- they fall under User-agent: * which has limited access.
- investor.fb.com: NO AI restrictions at all (open)
- So even Meta has a split: corporate news site blocks, investor relations site allows.

### Retail domains (for comparison):
| Company | Retail Domain | AI Bot Policy |
|---------|--------------|--------------|
| Amazon | amazon.com | BLOCKS ALL |
| NVIDIA | nvidia.com | AI bots restricted to llms.txt only |
| Tesla | tesla.com | Returns HTTP 403 (server-level block) |

---

## 3. Revised Treatment Prevalence

### On RETAIL domains:
- Clear block: Amazon, Tesla (403)
- Partial/restricted: NVIDIA (llms.txt only), Goldman Sachs (specific paths)
- No rules / open: Walmart, Apple, JPMorgan, ExxonMobil, Pfizer
- Training-only block: Meta

### On CORPORATE/IR domains (what matters for our paper):
- Clear block: **Meta only** (about.meta.com blocks training bots; search bots fall under restricted *)
- Partial block: **Goldman Sachs** (blocks GPTBot/ChatGPT-User on research/insights paths)
- Open: **8 out of 10 firms** (Amazon, Walmart, NVIDIA, Apple, JPMorgan, ExxonMobil, Pfizer, Tesla)

### Treatment rate on corporate domains: ~10-20% (1-2 out of 10)
### Treatment rate on retail domains: ~40-50%

---

## 4. Claude.ai Pilot Results (Amazon vs Walmart)

### Per-Question Corporate Citation Share

| Question | Amazon Corp% | Walmart Corp% | Difference |
|----------|-------------|---------------|------------|
| Q1 (Financial/EDGAR) | 25% | 14% | Amazon +11pp |
| Q2 (AI Strategy) | 36% | 33% | Amazon +3pp |
| Q3 (ESG) | 43% | 29% | Amazon +14pp |
| Q4 (Recent News) | 38% | 50% | Walmart +12pp |
| Q5 (Product Detail) | 40% | 40% | Same |
| **Overall** | **40%** | **25%** | **Amazon +15pp** |

- Failed URLs (Amazon): 2 (JS rendering, NOT policy blocking)
- Failed URLs (Walmart): 0
- Access denied messages: NONE for either firm

### Why no difference was observed:
Both aboutamazon.com and corporate.walmart.com are OPEN to AI search bots.
The pilot was comparing two firms with identical treatment status (both = Allow).

---

## 5. Implications for the Paper

### Problem: Treatment is rare on corporate domains
Only ~1-2 out of 10 large firms block AI on their corporate/IR sites. This means:
- The "say AI but block AI" story needs revision -- most firms do NOT block AI on corporate sites
- The blocking on retail sites (amazon.com, nvidia.com) is about COMMERCIAL protection (product data, pricing), not INFORMATION management
- The treatment variable (AIRetrievalBlock on corporate domain) may have very low prevalence in large caps

### What this means for research design:
1. MUST distinguish retail vs corporate domains when constructing the treatment variable
2. The interesting variation is on CORPORATE domains, not retail domains
3. May need to expand sample beyond mega-caps to find meaningful treatment variation
4. Meta is the best "treated" firm among large caps -- blocks training bots on about.meta.com but keeps investor.fb.com open (economically interesting: block competitors' AI but keep investor access)

### Revised economic story:
- Most firms keep corporate sites open to AI (they WANT their message accessible)
- A small subset blocks AI even on corporate sites (information management motive)
- The more common behavior is retail-site blocking (commercial protection motive)
- The paper could pivot to: WHY do some firms block AI on corporate sites when most don't? What predicts this rare choice? And does it matter?

### Or: pivot to retail domain blocking
- Retail-site blocking is more common and economically significant
- Amazon.com blocking AI crawlers affects product information, pricing, reviews
- This could affect consumer information access, market competition, platform power
- But this is more IO/consumer than corporate finance

---

## 6. Honest Assessment

### What works:
- The robots.txt data is real, measurable, and cross-sectionally varying
- The retail vs corporate domain split is a NOVEL and interesting finding itself
- Meta's selective blocking (corporate blocks, IR open) is a great case study
- The staggered rollout identification strategy still holds

### What doesn't work:
- "Do firms talk AI but block AI?" -- most firms don't block AI on corporate sites, so the "hypocrisy" story is weak for large caps
- Stage 2b (AI search quality degradation from corporate blocking) requires firms that actually block on corporate domains, which may be rare
- The Amazon pilot showed no effect because Amazon doesn't block on its corporate site

### Recommendation:
Before proceeding with the full research design, need a broader scan:
- Check corporate domain robots.txt for 50-100 firms across market cap quintiles
- If corporate-domain blocking is <5%, the paper needs significant redesign
- If it's 15-30%, especially concentrated in certain industries/sizes, the paper works

---

## 6. BATCH SCAN RESULTS: 55 Firms, Corporate Domains Only (March 20, 2026)

### Core Variables Prevalence (n=50 valid firms, 5 excluded due to 403)

| Variable | Count | Prevalence |
|----------|-------|-----------|
| **RetrievalBlockRoot = 1** | **1** | **2.0%** |
| **TrainBlockRoot = 1** | **2** | **4.0%** |
| **GooglebotBlock = 1** | **0** | **0.0%** |
| Any AI-specific rule | 7 | 14.0% |

### By Market Cap

| Group | N_valid | RetrievalBlockRoot | TrainBlockRoot | GooglebotBlock | Any AI Rule |
|-------|---------|-------------------|---------------|---------------|-------------|
| Mega  | 10 | 0 (0%) | 1 (10%) - Meta | 0 (0%) | 1 (10%) |
| Large | 14 | 0 (0%) | 0 (0%) | 0 (0%) | 3 (21%) |
| Mid   | 13 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| Small | 13 | 1 (8%) - SSTK | 1 (8%) - SSTK | 0 (0%) | 3 (23%) |

### Only 2 firms with ROOT blocks on corporate domains:

1. **Meta** (about.meta.com) — BLOCK_TRAINING only
   - Blocks: GPTBot, ClaudeBot, Google-Extended (training bots)
   - Does NOT block: OAI-SearchBot, Claude-SearchBot (search bots)
   - Economic logic: Blocks competitors' AI from training on Meta's content, but allows AI search

2. **Shutterstock** (shutterstock.com) — BLOCK_ALL
   - Blocks: OAI-SearchBot (search), GPTBot (training), CCBot (training)
   - Economic logic: IP/content company, protects image content from both training and search

### 5 firms with PATH-ONLY blocks (not root):
- Goldman Sachs: GPTBot blocked on /research/, /insights/ (protect proprietary research)
- Adobe: path blocks on specific areas
- Walmart: path blocks on /content/corporate/tools/ (generic site maintenance, not AI-targeted)
- Coursera: path blocks (protect course content on specific paths)
- ScanSource: path blocks

### Interpretation:
**RetrievalBlockRoot prevalence on corporate domains: ~2%**
This is BELOW the 5% threshold we set as minimum for the paper to work.
Only Shutterstock root-blocks AI search bots on its corporate domain.
The paper's treatment variable is extremely rare for large/mid-cap firms.
