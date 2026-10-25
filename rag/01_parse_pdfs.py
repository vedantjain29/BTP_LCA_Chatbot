#!/usr/bin/env python3
"""
01_parse_pdfs.py
----------------
Converts all PDFs in data/pdfs/ into clean Markdown files.
Saves results to data/processed/

Supports:
- Docling (preferred for academic papers)
- Fallback to pdfplumber + pypdf
"""

import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Optional but recommended
try:
    from docling.document_converter import DocumentConverter
    HAS_DOCLING = True
except ImportError:
    HAS_DOCLING = False
    print("Docling not installed. Falling back to pdfplumber + pypdf.")

import pdfplumber
from pypdf import PdfReader

# ====================== CONFIG ======================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
MAX_WORKERS = 4          # adjust based on your CPU
# ====================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Basic cleaning of extracted text."""
    if not text:
        return ""
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove common header/footer noise (simple version)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue
        # Skip very short repetitive lines that look like page numbers
        if re.fullmatch(r"\d+", line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def extract_with_docling(pdf_path: Path) -> str:
    """Best quality extraction for research papers."""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    # Docling can export to Markdown
    markdown = result.document.export_to_markdown()
    return markdown


def extract_with_fallback(pdf_path: Path) -> str:
    """Fallback using pdfplumber + pypdf."""
    text_parts = []

    # Try pdfplumber first
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        if text_parts:
            return clean_text("\n\n".join(text_parts))
    except Exception as e:
        print(f"  pdfplumber failed on {pdf_path.name}: {e}")

    # Fallback to pypdf
    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return clean_text("\n\n".join(text_parts))
    except Exception as e:
        print(f"  pypdf also failed on {pdf_path.name}: {e}")
        return ""


def process_one_pdf(pdf_path: Path) -> tuple[str, bool]:
    """Process a single PDF and save as Markdown."""
    print(f"Processing: {pdf_path.name}")

    try:
        if HAS_DOCLING:
            content = extract_with_docling(pdf_path)
        else:
            content = extract_with_fallback(pdf_path)

        if not content or len(content) < 200:
            print(f"  → Very little text extracted from {pdf_path.name}")
            return pdf_path.name, False

        # Output filename: keep original name but change to .md
        out_name = pdf_path.stem + ".md"
        out_path = OUTPUT_DIR / out_name

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# Source: {pdf_path.name}\n\n")
            f.write(content)

        print(f"  → Saved: {out_path.name} ({len(content):,} characters)")
        return pdf_path.name, True

    except Exception as e:
        print(f"  → FAILED: {pdf_path.name}")
        traceback.print_exc()
        return pdf_path.name, False


def main():
    pdf_files = list(PDF_DIR.glob("*.pdf")) + list(PDF_DIR.glob("*.PDF"))

    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        print("Please put your downloaded papers inside data/pdfs/")
        return

    print(f"Found {len(pdf_files)} PDF files.\n")

    success = 0
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one_pdf, pdf): pdf for pdf in pdf_files}
        for future in as_completed(futures):
            name, ok = future.result()
            if ok:
                success += 1
            else:
                failed.append(name)

    print("\n" + "="*60)
    print(f"Finished. Successfully processed: {success}/{len(pdf_files)}")
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  - {f}")
    print(f"\nClean Markdown files are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()