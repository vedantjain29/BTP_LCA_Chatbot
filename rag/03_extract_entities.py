#!/usr/bin/env python3
"""
03_extract_entities.py
----------------------
Reads chunks from data/processed/chunks.jsonl
Extracts LCA-relevant entities and light relations using a local LLM (Ollama)
Saves enriched chunks to data/processed/chunks_with_entities.jsonl
"""

import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any
import ollama

# ====================== CONFIG ======================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks_with_entities.jsonl"

# Ollama model for entity extraction (use a capable but not too heavy model)
ENTITY_MODEL = "qwen3:8b"          # or "llama3.1:8b" / "qwen2.5:7b" etc.
MAX_CHUNKS_TO_PROCESS = None       # Set to a number (e.g. 50) for testing, None = all
SLEEP_BETWEEN_CALLS = 0.3          # be gentle on local GPU/CPU
# ====================================================


SYSTEM_PROMPT = """You are an expert in Life Cycle Assessment (LCA) and Industrial Ecology.

Extract structured information from the given text chunk of a research paper.

Return ONLY valid JSON with this exact structure:

{
  "entities": {
    "methods": [],
    "databases": [],
    "impact_categories": [],
    "technologies": [],
    "locations": [],
    "standards": [],
    "other_key_concepts": []
  },
  "relations": [
    {"subject": "...", "relation": "...", "object": "..."}
  ],
  "summary": "One short sentence summarizing the main point of this chunk"
}

Guidelines:
- methods: e.g. "attributional LCA", "consequential LCA", "hybrid LCA", "prospective LCA", "S-LCA"
- databases: e.g. "ecoinvent", "GaBi", "ELCD", "US LCI"
- impact_categories: e.g. "GWP", "climate change", "acidification", "eutrophication", "human toxicity"
- technologies: e.g. "solar PV", "wind power", "hydropower", "biogas", "coal power"
- locations: countries, states, or regions mentioned (especially India-related)
- standards: e.g. "ISO 14040", "ISO 14044"
- Keep lists short and only include clearly mentioned items
- Relations should be simple and factual (max 3-5 per chunk)
- If nothing relevant is found, return empty lists
"""


def call_ollama(text: str) -> Dict[str, Any]:
    """Call local Ollama model to extract entities."""
    user_prompt = f"Extract entities and relations from this text:\n\n{text[:3500]}"

    try:
        response = ollama.chat(
            model=ENTITY_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.1,
                "num_predict": 800
            },
            format="json"
        )

        raw = response["message"]["content"]
        data = json.loads(raw)
        return data

    except Exception as e:
        print(f"  [Error] {e}")
        return {
            "entities": {
                "methods": [],
                "databases": [],
                "impact_categories": [],
                "technologies": [],
                "locations": [],
                "standards": [],
                "other_key_concepts": []
            },
            "relations": [],
            "summary": ""
        }


def main():
    if not CHUNKS_FILE.exists():
        print(f"Chunks file not found: {CHUNKS_FILE}")
        print("Please run 02_chunk_documents.py first.")
        return

    # Load chunks
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    total = len(chunks)
    if MAX_CHUNKS_TO_PROCESS:
        chunks = chunks[:MAX_CHUNKS_TO_PROCESS]
        print(f"Processing only first {len(chunks)} chunks (testing mode)")
    else:
        print(f"Processing all {total} chunks...")

    print(f"Using model: {ENTITY_MODEL}\n")

    enriched = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}/{len(chunks)}] {chunk['chunk_id']} | {chunk['heading'][:50]}")

        extraction = call_ollama(chunk["text"])

        # Merge original chunk with extraction
        new_chunk = chunk.copy()
        new_chunk["entities"] = extraction.get("entities", {})
        new_chunk["relations"] = extraction.get("relations", [])
        new_chunk["summary"] = extraction.get("summary", "")

        enriched.append(new_chunk)

        # Small sleep to avoid overloading local model
        time.sleep(SLEEP_BETWEEN_CALLS)

    # Save enriched chunks
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in enriched:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"Done! Enriched {len(enriched)} chunks")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()