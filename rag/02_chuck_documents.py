#!/usr/bin/env python3
"""
02_chunk_documents.py
---------------------
Takes clean Markdown files from data/processed/
and creates high-quality chunks with metadata.

Output: data/processed/chunks.jsonl
"""

import re
import json
from pathlib import Path
from typing import List, Dict

# ====================== CONFIG ======================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "chunks.jsonl"

# Chunking settings
MAX_CHUNK_CHARS = 1800        # roughly 400-500 tokens
MIN_CHUNK_CHARS = 200
CHUNK_OVERLAP = 200
# ====================================================


def split_by_headings(markdown: str) -> List[Dict]:
    """
    Split markdown by headings (##, ###, etc.).
    Returns list of {heading, content}
    """
    # Pattern for markdown headings
    heading_pattern = re.compile(r'^(#{1,4})\s+(.+)$', re.MULTILINE)

    sections = []
    matches = list(heading_pattern.finditer(markdown))

    if not matches:
        # No headings found → treat whole document as one section
        return [{"heading": "Full Document", "content": markdown.strip()}]

    # Content before first heading
    first_start = matches[0].start()
    if first_start > 0:
        pre_content = markdown[:first_start].strip()
        if pre_content:
            sections.append({"heading": "Preamble", "content": pre_content})

    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        content = markdown[start:end].strip()

        if content:
            sections.append({
                "heading": heading_text,
                "content": content
            })

    return sections


def further_split_long_section(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """
    If a section is still too long, split by paragraphs with overlap.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= max_chars:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start new chunk (with simple overlap of last part)
            if len(para) > max_chars:
                # Very long paragraph → force split
                for i in range(0, len(para), max_chars - CHUNK_OVERLAP):
                    chunks.append(para[i:i + max_chars].strip())
                current_chunk = ""
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def create_chunks_from_markdown(md_path: Path) -> List[Dict]:
    """Process one markdown file into chunks."""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove the "# Source: ..." line we added earlier
    content = re.sub(r'^# Source:.*\n+', '', content, count=1)

    source_name = md_path.stem  # filename without .md
    sections = split_by_headings(content)

    chunks = []
    chunk_id = 0

    for section in sections:
        heading = section["heading"]
        section_text = section["content"]

        sub_chunks = further_split_long_section(section_text)

        for sub in sub_chunks:
            if len(sub) < MIN_CHUNK_CHARS:
                continue

            chunk_id += 1
            chunks.append({
                "chunk_id": f"{source_name}__{chunk_id:03d}",
                "source": source_name,
                "heading": heading,
                "text": sub,
                "char_count": len(sub)
            })

    return chunks


def main():
    md_files = list(PROCESSED_DIR.glob("*.md"))

    if not md_files:
        print(f"No Markdown files found in {PROCESSED_DIR}")
        print("Please run 01_parse_pdfs.py first.")
        return

    print(f"Found {len(md_files)} Markdown files.\n")

    all_chunks = []

    for md_file in md_files:
        print(f"Chunking: {md_file.name}")
        try:
            chunks = create_chunks_from_markdown(md_file)
            all_chunks.extend(chunks)
            print(f"  → Created {len(chunks)} chunks")
        except Exception as e:
            print(f"  → Failed: {e}")

    # Save as JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()