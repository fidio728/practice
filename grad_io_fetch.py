"""Download all PDFs from chrisconlon/Grad-IO and merge per-week."""
from __future__ import annotations
import json, os, re, sys, time, urllib.parse, urllib.request
from pathlib import Path
from pypdf import PdfWriter, PdfReader

REPO = "chrisconlon/Grad-IO"
BRANCH = "master"
OUT_ROOT = Path(r"C:\Users\xl\OneDrive - Universitat Ramón Llull\Grad-IO")
RAW_DIR = OUT_ROOT / "raw"
MERGED_DIR = OUT_ROOT / "merged"

EXCLUDE_FILES = {
    "Week 12-Persistence and Switching Costs": {"test.pdf"},
    "Week 15b-Vertical Markets": {
        "newfigure1.pdf", "profitsbyfirm.pdf",
        "rebate1.pdf", "rebate3.pdf",
        "standardrebates.pdf", "allunitsdiscount.pdf",
    },
}


def gh_json(path: str):
    url = f"https://api.github.com/repos/{REPO}/contents/{urllib.parse.quote(path)}"
    req = urllib.request.Request(url, headers={"User-Agent": "grad-io-fetch"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def natural_key(s: str) -> str:
    s = re.sub(r"\s+", "", s.lower())
    return re.sub(r"(\d+)", lambda m: m.group(1).zfill(10), s)


def normalize_week_name(folder: str) -> str:
    m = re.match(r"Week\s+(\d+)([a-z]?)", folder, re.IGNORECASE)
    if not m:
        return folder
    base = f"Week_{int(m.group(1)):02d}{m.group(2)}"
    if re.search(r"-Old\b", folder):
        base += "_Old"
    return base


def download(url: str, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst.stat().st_size
    req = urllib.request.Request(url, headers={"User-Agent": "grad-io-fetch"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        data = r.read()
        f.write(data)
    return len(data)


def merge_pdfs(pdfs: list[Path], out: Path) -> tuple[int, int]:
    writer = PdfWriter()
    total_pages = 0
    for p in pdfs:
        reader = PdfReader(str(p))
        for page in reader.pages:
            writer.add_page(page)
            total_pages += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        writer.write(f)
    return len(pdfs), total_pages


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    top = gh_json("")
    weeks = sorted(
        [d["name"] for d in top if d["type"] == "dir" and d["name"].startswith("Week")],
        key=natural_key,
    )

    summary = []
    for folder in weeks:
        wname = normalize_week_name(folder)
        items = gh_json(folder)
        pdfs = [d for d in items if d["type"] == "file" and d["name"].lower().endswith(".pdf")]
        excludes = EXCLUDE_FILES.get(folder, set())
        pdfs = [d for d in pdfs if d["name"] not in excludes]
        pdfs.sort(key=lambda d: natural_key(d["name"]))

        if not pdfs:
            print(f"[skip] {folder}: no eligible PDFs")
            continue

        local_files = []
        raw_subdir = RAW_DIR / wname
        for d in pdfs:
            dst = raw_subdir / d["name"]
            size = download(d["download_url"], dst)
            local_files.append(dst)
            print(f"  dl {wname}/{d['name']} ({size/1024:.0f} KB)")

        merged_out = MERGED_DIR / f"{wname}.pdf"
        try:
            n_files, n_pages = merge_pdfs(local_files, merged_out)
            print(f"[ok]  {wname}.pdf: {n_files} files, {n_pages} pages")
            summary.append((wname, n_files, n_pages, [p.name for p in local_files]))
        except Exception as e:
            print(f"[err] {wname}: {e}")
            summary.append((wname, len(local_files), -1, [p.name for p in local_files]))

    print("\n=== SUMMARY ===")
    for wname, nf, np_, names in summary:
        flag = "OK" if np_ >= 0 else "FAIL"
        print(f"{flag}  {wname:14s}  {nf} files / {np_} pages")
        for n in names:
            print(f"        - {n}")


if __name__ == "__main__":
    main()
