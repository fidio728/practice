"""Generate Request Letter to PhD Commission as a Word document."""
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from pathlib import Path

doc = Document()

# Page margins (1 inch)
for section in doc.sections:
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

# Default paragraph style
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)
style.paragraph_format.line_spacing = 1.15
style.paragraph_format.space_after = Pt(10)

def para(text, bullet=False, indent_left=0.0):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(0)
    if bullet:
        p.paragraph_format.left_indent = Inches(indent_left)
    p.add_run(text)
    return p

# Header: sender info (optional — user can edit)
header = doc.add_paragraph()
header.paragraph_format.first_line_indent = Inches(0)
header.paragraph_format.space_after = Pt(6)
header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
header.add_run(
    "Xueyang Liu\n"
    "PhD Programme in Management Sciences\n"
    "Universitat Ramon Llull (IQS School of Management)\n"
    "April 19, 2026"
)

# Salutation
para("Dear Members of the MRes and PhD Commission,")

# Body
para(
    "I am writing to formally request a change of supervisor and research group. "
    "I am currently supervised by Dr. Joren Gijsbrechts in the Business Network "
    "Dynamics (BuNeD) research group, and I would like to transfer to the Group "
    "for Research in Economics and Finance (GREF), under the supervision of "
    "Dr. Emanuele Rizzo."
)

para(
    "During my MRes and the first year of my PhD, I explored several topics in "
    "operations research using empirical, optimization, and game-theoretic methods. "
    "Through this work, I came to recognize that my research interests sit more "
    "naturally in empirical corporate finance, specifically on capital markets and "
    "the firm's information environment. I attended Dr. Rizzo's empirical corporate "
    "finance course this semester, which confirmed that both the research questions "
    "and the methodology of this field fit my background in economics and my "
    "longer-term research agenda."
)

para(
    "Since March 2026, I have been working with Dr. Rizzo on what will become the "
    "first chapter of my dissertation, with the working title \"Website Access Rules "
    "and Visibility in AI-Powered Search.\" The project examines whether firms whose "
    "websites block AI search crawlers (such as ChatGPT Search, Perplexity, and "
    "Claude) are systematically less visible to investors who rely on these tools, "
    "and whether reduced visibility translates into measurable changes in the firm's "
    "information environment, such as analyst forecast dispersion and bid-ask spreads. "
    "The main contribution is a new firm-level measure of AI search visibility, "
    "constructed from the technical configuration of corporate websites rather than "
    "from disclosure content. This measure captures a form of information friction "
    "that does not appear in standard disclosure data and that firms may inherit "
    "unintentionally from default website templates. The project speaks to research "
    "on the technology of investor information acquisition and to emerging work on "
    "how AI systems interact with web content."
)

para(
    "To support this study, I am constructing an original firm-level dataset that "
    "covers approximately 9,500 US public firms identified from Compustat. The "
    "dataset records the crawling permissions of 4,350 unique corporate websites "
    "across 62 documented AI crawlers, distinguishing search, training, and "
    "user-triggered bots. I am currently extending the cross-section into a monthly "
    "panel from 2021 to 2025 using the Wayback Machine, which will allow me to "
    "identify firms that changed their AI access policy over time and use these "
    "changes for causal identification. Data construction is expected to be complete "
    "in the coming weeks."
)

# Timeline intro
para("My timeline for the Research Proposal is as follows:")

# Timeline bullets
def bullet(text):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("• " + text)

bullet(
    "Late April – early May: Mechanism test using AI search APIs to verify that "
    "blocked firms are cited less often in AI-generated answers."
)
bullet(
    "May 4 – May 17: Regression analysis linking web-access restrictions to "
    "information environment outcomes."
)
bullet(
    "May 18 – May 24: Drafting the first chapter, and working with Dr. Rizzo "
    "to develop preliminary directions for the remaining two chapters of the "
    "dissertation."
)
bullet(
    "End of May: Submit the full draft of the first chapter to Dr. Rizzo for "
    "review."
)
bullet(
    "Early June: Present the Research Proposal draft to the GREF research group "
    "and inform the PhD Office of the presentation date."
)
bullet(
    "June 19: Submit the final Research Proposal and accompanying documents to "
    "the PhD Programme Office."
)

# Closing
para(
    "I remain grateful to Dr. Gijsbrechts for his guidance during my time in BuNeD, "
    "and I thank him for his continued support through this transition. I would also "
    "like to thank the Commission for considering my request."
)

# Sign-off
para("Sincerely,")
para("Xueyang Liu")

out_path = Path(__file__).parent / "Liu_request_letter.docx"
doc.save(out_path)
print(f"Wrote {out_path}")
