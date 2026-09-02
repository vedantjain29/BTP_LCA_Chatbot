import ollama
import json
from dataclasses import dataclass
from typing import List

@dataclass
class ClassificationResult:
    url: str
    is_useful: bool
    confidence: str
    tags: List[str]
    reason: str
    category: str
    raw_response: str

class LCAClassifier:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.system_prompt = """You are an expert in Life Cycle Assessment (LCA), Industrial Ecology, and methodological research.

Your task is to evaluate whether a research paper is useful for building **theoretical and methodological understanding** of Life Cycle Analysis.

A paper is USEFUL if it contributes to:
- Theoretical foundations of LCA
- Methodological advances (goal & scope, LCI, LCIA, interpretation, allocation, uncertainty, consequential vs attributional, prospective LCA, etc.)
- Reviews that synthesize methods or frameworks
- Standards, databases methodology (ecoinvent, etc.)
- Data quality, uncertainty, sensitivity methods

A paper is NOT useful if it is mainly:
- A pure applied case study with almost no methodological discussion
- Only about general LLMs / RAG / Hallucination with almost no LCA theory
- Software tool documentation
- Very specific product/system LCA without broader insights

Respond ONLY in valid JSON with this exact structure:
{
  "is_useful": true or false,
  "confidence": "high" or "medium" or "low",
  "category": "theoretical_lca" | "methodological_review" | "case_study" | "dataset" | "tool" | "irrelevant",
  "tags": ["tag1", "tag2", "tag3"],
  "reason": "clear and concise explanation"
}

Good tags examples: 
["LCA methodology", "uncertainty", "prospective LCA", "LCI methods", "ecoinvent", "consequential LCA", "allocation", "S-LCA", "hybrid LCA", "data quality", "LLM+LCA", "review"]
"""

    def classify(self, title: str, content: str, url: str = "") -> ClassificationResult:
        content = content[:5500] if content else ""

        user_prompt = f"""Title: {title}

Content:
{content}

URL: {url}

Evaluate this paper carefully."""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.15,
                    "num_predict": 500
                },
                format="json"
            )

            raw = response["message"]["content"]
            data = json.loads(raw)

            tags = data.get("tags", [])
            if not isinstance(tags, list):
                tags = []

            return ClassificationResult(
                url=url,
                is_useful=bool(data.get("is_useful", False)),
                confidence=data.get("confidence", "low"),
                tags=tags,
                reason=data.get("reason", ""),
                category=data.get("category", "irrelevant"),
                raw_response=raw
            )

        except Exception as e:
            return ClassificationResult(
                url=url,
                is_useful=False,
                confidence="low",
                tags=[],
                reason=f"Error: {str(e)}",
                category="error",
                raw_response=str(e)
            )