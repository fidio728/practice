"""Concatenate all per-week merged PDFs into one big PDF with bookmarks."""
from pathlib import Path
from pypdf import PdfWriter, PdfReader

MERGED_DIR = Path(r"C:\Users\xl\OneDrive - Universitat Ramón Llull\Grad-IO\merged")
OUT = Path(r"C:\Users\xl\OneDrive - Universitat Ramón Llull\Grad-IO\Grad-IO_AllWeeks.pdf")

ORDER = [
    "Week_01", "Week_02", "Week_03",
    "Week_04", "Week_04_Old",
    "Week_05", "Week_06", "Week_07", "Week_08",
    "Week_10", "Week_11", "Week_12", "Week_13", "Week_14",
    "Week_15", "Week_15b", "Week_15c",
]

writer = PdfWriter()
running = 0
for name in ORDER:
    src = MERGED_DIR / f"{name}.pdf"
    reader = PdfReader(str(src))
    n = len(reader.pages)
    for page in reader.pages:
        writer.add_page(page)
    writer.add_outline_item(name, running)
    print(f"+ {name}: {n} pages (starts at page {running + 1})")
    running += n

with open(OUT, "wb") as f:
    writer.write(f)
print(f"\nTotal: {running} pages -> {OUT}")
