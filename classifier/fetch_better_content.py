import csv
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fetch_abstract(url: str) -> str:
    """Try to extract abstract or main content from common academic sites"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        # Common abstract selectors
        selectors = [
            {"name": "div", "class_": "abstract"},
            {"name": "div", "id": "abstract"},
            {"name": "section", "class_": "abstract"},
            {"name": "div", "class_": "Abstract"},
            {"name": "div", "class_": "article-abstract"},
            {"name": "div", "class_": "abstract-content"},
            {"name": "p", "class_": "abstract"},
            {"name": "div", "class_": "hlFld-Abstract"},
            {"name": "div", "id": "Abs1"},
            {"name": "section", "id": "abstract"},
            {"name": "div", "class_": "c-article-section__content"},  # Springer
            {"name": "div", "class_": "abstractSection"},             # ScienceDirect sometimes
        ]

        for sel in selectors:
            element = soup.find(sel["name"], class_=sel.get("class_"), id=sel.get("id"))
            if element:
                text = clean_text(element.get_text())
                if len(text) > 100:
                    return text[:2500]  # limit length

        # Fallback: try meta description
        meta = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if meta and meta.get("content"):
            return clean_text(meta["content"])[:1500]

        # Last fallback: first few paragraphs
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs[:8]])
        return clean_text(text)[:2000]

    except Exception as e:
        return f"[Fetch failed: {str(e)[:80]}]"

def improve_dataset(input_csv: str, output_csv: str):
    rows = []

    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        original_rows = list(reader)

    print(f"Improving content for {len(original_rows)} papers...\n")

    for i, row in enumerate(original_rows, 1):
        title = row.get("title", "")
        url = row.get("url", "")
        old_content = row.get("content", "")

        print(f"[{i}/{len(original_rows)}] {title[:65]}...")

        abstract = fetch_abstract(url)

        # Combine old notes + new abstract
        if abstract and not abstract.startswith("[Fetch failed"):
            new_content = f"Abstract: {abstract}\n\nPrevious notes: {old_content}"
        else:
            new_content = old_content  # keep old if fetch failed

        rows.append({
            "url": url,
            "title": title,
            "content": new_content
        })

        time.sleep(1.2)  # be polite to servers

    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "title", "content"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Improved dataset saved to → {output_csv}")

if __name__ == "__main__":
    improve_dataset(
        input_csv="../data/urls_with_content.csv",
        output_csv="../data/urls_with_better_content.csv"
    )