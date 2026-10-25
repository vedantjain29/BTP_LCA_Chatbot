#!/usr/bin/env python3
"""
retriever.py
------------
Hybrid Retriever for the LCA Chatbot

- Dense + Sparse (BM25-style) search via Qdrant
- Metadata filtering
- Light GraphRAG (entity expansion)
"""

from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
import ollama

# ====================== CONFIG ======================
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_db"
COLLECTION_NAME = "lca_papers"

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"

# Retrieval settings
TOP_K = 8
DENSE_WEIGHT = 0.6          # relative importance of dense vs sparse
# ====================================================


class LCARetriever:
    def __init__(self):
        self.client = QdrantClient(path=str(QDRANT_PATH))
        self.dense_model = TextEmbedding(model_name=DENSE_MODEL)
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    def _get_dense_vector(self, text: str) -> List[float]:
        emb = list(self.dense_model.embed([text]))[0]
        return emb.tolist()

    def _get_sparse_vector(self, text: str) -> models.SparseVector:
        emb = list(self.sparse_model.embed([text]))[0]
        return models.SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist()
        )

    def hybrid_search(
        self,
        query: str,
        top_k: int = TOP_K,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Perform hybrid search (dense + sparse) with optional metadata filters.
        """
        dense_vec = self._get_dense_vector(query)
        sparse_vec = self._get_sparse_vector(query)

        # Build Qdrant filter if provided
        qdrant_filter = None
        if filters:
            must_conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value)
                        )
                    )
                else:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
            qdrant_filter = models.Filter(must=must_conditions)

        # Prefetch both dense and sparse
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=top_k * 2,
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=top_k * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),  # Reciprocal Rank Fusion
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

        # Format results
        retrieved = []
        for point in results.points:
            payload = point.payload
            retrieved.append({
                "score": point.score,
                "chunk_id": payload.get("chunk_id"),
                "source": payload.get("source"),
                "heading": payload.get("heading"),
                "text": payload.get("text"),
                "summary": payload.get("summary"),
                "entities": payload.get("entities", {}),
                "relations": payload.get("relations", []),
            })

        return retrieved

    def light_graphrag_expand(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Simple light GraphRAG:
        - Collect important entities from top results
        - Optionally expand by finding other chunks that share key entities
        """
        # Collect key entities from the retrieved chunks
        key_entities = set()
        for r in results[:4]:  # only top few
            ents = r.get("entities", {})
            for category in ["technologies", "methods", "locations", "databases"]:
                for item in ents.get(category, []):
                    key_entities.add(item.lower())

        if not key_entities:
            return results

        # For now we just attach the collected entities
        # (You can later add a second search using these entities)
        for r in results:
            r["expanded_entities"] = list(key_entities)

        return results

    def retrieve(self, query: str, filters: Optional[Dict] = None, top_k: int = TOP_K) -> List[Dict]:
        """
        Main retrieval method.
        """
        results = self.hybrid_search(query, top_k=top_k, filters=filters)
        results = self.light_graphrag_expand(results, query)
        return results


# ====================== Quick Test ======================
if __name__ == "__main__":
    retriever = LCARetriever()

    test_query = "What is the global warming potential of solar PV in India?"
    print(f"Query: {test_query}\n")

    results = retriever.retrieve(test_query, top_k=5)

    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['source']}] {r['heading']}")
        print(f"   Score: {r['score']:.4f}")
        print(f"   Summary: {r.get('summary', '')[:120]}...")
        print(f"   Text preview: {r['text'][:150]}...\n")

'''
# For searching

from retriever import LCARetriever

retriever = LCARetriever()

# Simple search
results = retriever.retrieve("life cycle assessment of wind energy in India")

# With metadata filter (example)
results = retriever.retrieve(
    "solar PV carbon footprint",
    filters={"source": "some_paper_name"}   # or other payload fields
)
'''