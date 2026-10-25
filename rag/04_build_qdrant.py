#!/usr/bin/env python3
"""
04_build_qdrant.py
------------------
Loads enriched chunks (with entities) and creates a hybrid Qdrant collection.

Features:
- Dense embeddings (semantic)
- Sparse embeddings (keyword / BM25-style) using fastembed
- Rich metadata + entities for filtering and light GraphRAG
"""

import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, PointStruct
from fastembed import TextEmbedding, SparseTextEmbedding
import ollama

# ====================== CONFIG ======================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks_with_entities.jsonl"
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant_db"
COLLECTION_NAME = "lca_papers"

# Embedding models
DENSE_MODEL = "BAAI/bge-small-en-v1.5"          # good balance (or use bge-base / bge-m3)
SPARSE_MODEL = "Qdrant/bm25"                    # sparse / keyword

BATCH_SIZE = 32
# ====================================================


def load_chunks() -> List[Dict]:
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def get_dense_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate dense embeddings using fastembed."""
    model = TextEmbedding(model_name=DENSE_MODEL)
    embeddings = list(model.embed(texts))
    return [emb.tolist() for emb in embeddings]


def get_sparse_embeddings(texts: List[str]):
    """Generate sparse embeddings (BM25 style)."""
    model = SparseTextEmbedding(model_name=SPARSE_MODEL)
    embeddings = list(model.embed(texts))
    return embeddings


def main():
    if not CHUNKS_FILE.exists():
        print(f"File not found: {CHUNKS_FILE}")
        print("Please run 03_extract_entities.py first.")
        return

    print("Loading chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks\n")

    # Initialize Qdrant (embedded / local mode)
    client = QdrantClient(path=str(QDRANT_PATH))

    # Delete collection if it already exists (for clean rebuild)
    if client.collection_exists(COLLECTION_NAME):
        print(f"Deleting existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)

    # Create collection with both dense + sparse vectors
    print("Creating hybrid collection...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=384,                    # bge-small-en-v1.5 dimension
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        }
    )

    print("Generating embeddings and uploading to Qdrant...\n")

    # Process in batches
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Uploading"):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        # Dense embeddings
        dense_embs = get_dense_embeddings(texts)

        # Sparse embeddings
        sparse_embs = get_sparse_embeddings(texts)

        points = []
        for idx, chunk in enumerate(batch):
            # Prepare payload (metadata + entities for filtering & light GraphRAG)
            payload = {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "heading": chunk.get("heading", ""),
                "text": chunk["text"],
                "summary": chunk.get("summary", ""),
                "entities": chunk.get("entities", {}),
                "relations": chunk.get("relations", []),
                "char_count": chunk.get("char_count", 0),
            }

            # Sparse vector format required by Qdrant
            sparse_vector = models.SparseVector(
                indices=sparse_embs[idx].indices.tolist(),
                values=sparse_embs[idx].values.tolist()
            )

            point = PointStruct(
                id=i + idx,                          # simple integer ID
                vector={
                    "dense": dense_embs[idx],
                    "sparse": sparse_vector
                },
                payload=payload
            )
            points.append(point)

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    print("\n" + "=" * 60)
    print("Qdrant collection built successfully!")
    print(f"Collection name : {COLLECTION_NAME}")
    print(f"Location        : {QDRANT_PATH}")
    print(f"Total points    : {len(chunks)}")
    print("=" * 60)

    # Quick verification
    info = client.get_collection(COLLECTION_NAME)
    print(f"\nCollection info: {info.points_count} points stored.")


if __name__ == "__main__":
    main()