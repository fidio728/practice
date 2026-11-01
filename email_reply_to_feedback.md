# Reply to Emanuele's Feedback

Hi Emanuele,

Thank you for the detailed feedback — all four points are very helpful. Let me address them in order.

**1) Clarification on root-blocking.** Root-blocking refers strictly to `Disallow: /` — a blanket prohibition covering the entire domain. It does not include selective blocking of specific paths. The 2% and 4% figures I reported are under this strict definition. If we expand the definition to include any AI-specific rule — including path-level blocks like Goldman Sachs blocking GPTBot on /research/ — the incidence rises to about 14% of firms. Importantly, each domain has its own robots.txt, so a firm can (and often does) have different policies across domains. Amazon, for instance, root-blocks all AI bots on amazon.com but has zero AI restrictions on aboutamazon.com. In my scan, I focused on corporate/IR domains — the ones hosting press releases, strategy documents, and investor information.

**2) Why not simply remove the information?** This is a good question and I think the answer is that robots.txt allows firms to control access at the agent level rather than at the content level. Removing information from the website would cut off all audiences — customers, Google, investors, analysts — simultaneously. Robots.txt allows the firm to keep the information visible to human visitors and indexed by Google, while selectively blocking AI crawlers. Goldman Sachs, for example, wants its research summaries on the website for client access and brand visibility, but does not want GPTBot to ingest the full content into a general-purpose AI model. That said, I should acknowledge a real limitation: robots.txt is advisory, not enforced. A non-compliant bot can simply ignore it. And much of the information may already exist in third-party sources or in historical training data collected before the block was added. So the effectiveness of robots.txt as a gatekeeping mechanism is an open empirical question — which is part of what makes the setting interesting but also challenging.

**3) Relevance: why do only a few firms block?** I agree this is the central question. There are several possible explanations, and sorting between them is likely the first empirical task:

- *Awareness lag.* GPTBot was introduced in August 2023, and most AI-specific search bots only appeared in 2024. The majority of firms in my sample have no AI-specific rules at all in their robots.txt — not because they actively chose to allow AI, but because they have not yet updated a file that may not have been reviewed in years. If this is the explanation, we should see the blocking rate increasing over time, which is testable with monthly panel data.

- *Low proprietary cost for most firms.* For a typical manufacturing or utility company, the information on the corporate website — product descriptions, press releases, sustainability reports — has little proprietary value. AI ingesting this content does not meaningfully harm the firm. Only firms with high-value web content (paid research, proprietary images, competitive pricing data) face a real cost from AI access. If this is the explanation, blocking should be concentrated among IP-intensive firms and firms with commercially valuable web content.

- *Benefits of openness outweigh costs.* For most firms, being accessible to AI-powered search is desirable — it means that when investors or customers ask ChatGPT about the firm, the answer draws on the firm's own content rather than third-party sources. Blocking AI could mean losing control over the narrative. Under this explanation, not blocking is the rational default, and blocking is a deliberate choice made only when the proprietary cost is high enough to outweigh the visibility benefit.

Your suggestion to compare blocking firms with non-blocking firms on observable dimensions is the right first step. I plan to expand the sample and test whether blocking is predicted by IP intensity (R&D/revenue, patent counts), the commercial nature of web content (e.g., whether the firm operates an e-commerce platform or sells content/data products), industry, and firm size.

**4) Server overload as an alternative explanation.** This is a valid concern. However, several patterns in the data are hard to reconcile with a pure server-load story:

- *Selective blocking.* If the goal were to reduce traffic, firms should either block all high-frequency bots or impose rate limits. Instead, we see surgical choices: Meta blocks training bots but explicitly allows search bots on the same domain. Goldman Sachs blocks GPTBot on /research/ but allows it on the rest of the site. These path-specific and bot-type-specific decisions suggest a content-protection motive, not a bandwidth concern.

- *Crawl-delay vs. Disallow.* The robots.txt protocol offers a `Crawl-delay` directive specifically for managing request frequency. Many firms use it (e.g., aboutamazon.com sets `Crawl-delay: 10`). A firm concerned about server load would use Crawl-delay to throttle the bot, not `Disallow: /` to block it entirely. Disallow says "I don't want you to see this content," not "you're visiting too often."

- *Googlebot is never blocked.* Googlebot generates far more traffic than any AI crawler — it crawls millions of pages per day. Yet zero firms in my sample block Googlebot on their corporate domain. If server load were the primary concern, Googlebot would be the first target, not GPTBot.

That said, I cannot fully rule out server-load motivations without actual traffic data, and for smaller firms with limited infrastructure, even modest bot traffic could matter. I will keep this as a control dimension — for instance, by checking whether blocking correlates with proxies for website capacity (firm size, IT spending) rather than content value.

I will expand the sample and run the characteristics comparison you suggested. Happy to discuss next steps whenever works for you.

Best regards,
Xueyang
