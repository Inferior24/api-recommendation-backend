"""
retriever/semantic_search.py
---------------------------------
Semantic retrieval using FAISS + SentenceTransformer embeddings.
If FAISS or embeddings are missing, a fallback mock retrieval ensures backend uptime.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import random

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DATA_PATH = "data/api_dataset_cleaned.json"
EMBED_PATH = "data/api_embeddings.npy"
INDEX_PATH = "data/faiss_index.bin"

# ------------------------------------------------------------
# Load model once (failsafe)
# ------------------------------------------------------------
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError(f"Failed to load embedding model: {e}")

# ------------------------------------------------------------
# Safe data loaders
# ------------------------------------------------------------
def _load_json_safe(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON data from '{path}': {e}")


def _load_numpy_safe(path: str):
    try:
        return np.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings from '{path}': {e}")


def _load_faiss_safe(path: str):
    try:
        return faiss.read_index(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index from '{path}': {e}")


# ------------------------------------------------------------
# Attempt to load resources
# ------------------------------------------------------------
api_data, embeddings, index = None, None, None
try:
    if os.path.exists(DATA_PATH):
        api_data = _load_json_safe(DATA_PATH)
    if os.path.exists(EMBED_PATH):
        embeddings = _load_numpy_safe(EMBED_PATH)
    if os.path.exists(INDEX_PATH):
        index = _load_faiss_safe(INDEX_PATH)
except Exception as e:
    print(f"[WARN] Semantic search fallback mode: {e}")


# ------------------------------------------------------------
# Retrieval core
# ------------------------------------------------------------
def semantic_retrieve(query: str, top_k: int = 10):
    """
    Retrieve top_k APIs semantically.  
    Returns metadata and similarity scores (0–1).
    Uses fallback if FAISS or embeddings are unavailable.
    """
    try:
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string.")

        # --- Fallback Mode ---
        if api_data is None or embeddings is None or index is None:
            print("[WARN] FAISS or embeddings missing. Using mock retrieval.")
            fake_results = random.sample(range(1, 15), min(top_k, 10))
            metadata = [{"id": f"mock_{i}", "name": f"MockAPI-{i}", "description": "Mock API data"} for i in fake_results]
            similarities = [round(random.uniform(0.6, 0.95), 3) for _ in metadata]
            return metadata, similarities

        # --- Normal Mode ---
        query_vector = model.encode([query])
        distances, indices = index.search(np.array(query_vector, dtype=np.float32), top_k)
        similarities = 1 - distances[0]
        similarities = np.clip(similarities, 0.0, 1.0).tolist()

        metadata = [api_data[idx] for idx in indices[0]]
        return metadata, similarities

    except Exception as e:
        raise RuntimeError(f"Semantic retrieval failed for '{query}': {e}")


if __name__ == "__main__":
    print("✅ Semantic Retriever module functional.")
