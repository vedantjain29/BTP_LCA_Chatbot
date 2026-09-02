import csv
from pathlib import Path

def prepare_dataset(input_file: str, output_file: str):
    rows = []

    with open(input_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("Title", "").strip()
            url = row.get("URL", "").strip()
            notes = row.get("Notes", "").strip()
            subdomain = row.get("Subdomain", "").strip()
            year = row.get("Year", "").strip()
            source = row.get("Source/Venue", "").strip()

            # Build a useful "content" field from available information
            content_parts = []
            if notes:
                content_parts.append(f"Notes: {notes}")
            if subdomain:
                content_parts.append(f"Subdomain: {subdomain}")
            if source:
                content_parts.append(f"Source: {source}")
            if year:
                content_parts.append(f"Year: {year}")

            content = " | ".join(content_parts)

            rows.append({
                "url": url,
                "title": title,
                "content": content
            })

    # Write the prepared file
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "title", "content"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Prepared {len(rows)} entries")
    print(f"Saved to → {output_file}")

if __name__ == "__main__":
    prepare_dataset(
        input_file="../data/LCA_papers_dataset.csv",
        output_file="../data/urls_with_content.csv"
    )