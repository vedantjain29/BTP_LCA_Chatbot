import ollama
import json
from dataclasses import dataclass, field
from typing import List

# Keywords used to deterministically rank priority based on the
# LLM-extracted `industry` field. Edit this list to retune priority rules
# without needing to re-run the classifier.
RENEWABLE_KEYWORDS = [
    "solar", "pv", "photovoltaic", "wind", "hydro",
    "geothermal", "biomass", "renewable"
]

def compute_priority(industry: str) -> tuple[int, str]:
    """
    Deterministically rank a paper's priority based on its industry field.
    Returns (priority_rank, priority_label).
      1 / "high"   -> renewable electricity
      2 / "medium" -> electricity, non-renewable / general / unspecified
      3 / "low"    -> everything else (including NA / unrelated industries)
    """
    if not industry or industry.strip().upper() == "NA":
        return 3, "low"

    industry_lower = industry.lower()

    if "electric" in industry_lower:
        if any(keyword in industry_lower for keyword in RENEWABLE_KEYWORDS):
            return 1, "high"
        return 2, "medium"

    return 3, "low"


@dataclass
class ClassificationResult:
    url: str
    is_useful: bool
    confidence: str
    tags: List[str]
    reason: str
    category: str
    country: str
    industry: str
    priority_rank: int
    priority_label: str
    raw_response: str


class LCAClassifier:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.system_prompt = """You are an expert in Life Cycle Assessment (LCA), Industrial Ecology, and methodological research.

Your task is to evaluate whether a research paper is useful for building **theoretical and methodological understanding** of Life Cycle Analysis, and to extract structured metadata about its geographic and industrial context.

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

In addition to usefulness, extract two more fields:

1. "country": The specific country the paper's study/data/case is grounded in (e.g. "India", "Germany", "United States"). Use "NA" if the paper is generic, purely theoretical/methodological, multi-country without a clear primary focus, or doesn't specify a country.

2. "industry": The industry/sector the paper relates to, written as a hierarchy using " > " to separate levels, most general first. Examples:
   - "Electricity > Solar/PV"
   - "Electricity > Wind"
   - "Electricity > Hydro"
   - "Electricity > Grid-mix/Non-renewable"
   - "Electricity > General/Unspecified"   (mentions electricity but not renewable vs non-renewable)
   - "Chemical Manufacturing"
   - "Transportation"
   - "Agriculture"
   - "Construction/Buildings"
   - "Waste Management"
   - "General/Cross-sector"   (methodology papers not tied to one industry)
   - "NA"   (cannot be determined / not applicable)
   Be as specific as the paper allows, but do not guess a subtype if it isn't clearly stated — use "Electricity > General/Unspecified" rather than inventing a renewable/non-renewable split that isn't supported by the text.

Respond ONLY in valid JSON with this exact structure:
{
  "is_useful": true or false,
  "confidence": "high" or "medium" or "low",
  "category": "theoretical_lca" | "methodological_review" | "case_study" | "dataset" | "tool" | "irrelevant",
  "tags": ["tag1", "tag2", "tag3"],
  "reason": "clear and concise explanation",
  "country": "country name or NA",
  "industry": "hierarchical industry string or NA"
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

            country = data.get("country", "NA") or "NA"
            industry = data.get("industry", "NA") or "NA"

            priority_rank, priority_label = compute_priority(industry)

            return ClassificationResult(
                url=url,
                is_useful=bool(data.get("is_useful", False)),
                confidence=data.get("confidence", "low"),
                tags=tags,
                reason=data.get("reason", ""),
                category=data.get("category", "irrelevant"),
                country=country,
                industry=industry,
                priority_rank=priority_rank,
                priority_label=priority_label,
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
                country="NA",
                industry="NA",
                priority_rank=3,
                priority_label="low",
                raw_response=str(e)
            )