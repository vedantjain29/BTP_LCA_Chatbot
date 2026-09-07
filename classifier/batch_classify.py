import csv
import time
from pathlib import Path
from lca_classifier import LCAClassifier

def process_url_list(input_csv: str, output_csv: str, model: str = "llama3.2"):
    clf = LCAClassifier(model=model)
    results = []

    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} rows to classify with improved content + tags...\n")

    for i, row in enumerate(rows, 1):
        title = row.get("title", "")
        content = row.get("content", "")
        url = row.get("url", "")

        print(f"[{i}/{len(rows)}] {title[:70]}...")

        result = clf.classify(title=title, content=content, url=url)

        results.append({
            "url": result.url,
            "title": title,
            "is_useful": result.is_useful,
            "confidence": result.confidence,
            "category": result.category,
            "tags": " | ".join(result.tags),
            "country": result.country,
            "industry": result.industry,
            "priority_rank": result.priority_rank,
            "priority_label": result.priority_label,
            "reason": result.reason
        })

        time.sleep(0.6)

    # Sort so the highest-priority (renewable electricity) papers are first
    results.sort(key=lambda r: r["priority_rank"])

    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "url", "title", "is_useful", "confidence", "category",
                "tags", "country", "industry", "priority_rank",
                "priority_label", "reason"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    useful = sum(1 for r in results if r["is_useful"])
    high_priority = sum(1 for r in results if r["priority_rank"] == 1)
    print(f"\nFinished!")
    print(f"Useful resources        : {useful} / {len(results)}")
    print(f"High priority (renewable electricity): {high_priority} / {len(results)}")
    print(f"Results saved to        : {output_csv}")

if __name__ == "__main__":
    process_url_list(
        input_csv="../data/urls_with_better_content.csv",
        output_csv="../data/classification_results_v3.csv",
        model="llama3.2"
    )