"""
Retriever module — loads FAISS index + metadata and exposes vector search.

Fully path-stable:
- Always resolves paths relative to the project root.
"""

import os
import json
import faiss
import numpy as np


# ----------------------------------------------------------------------
# Resolve paths relative to project root
# ----------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Recommendation_Part/

EMBED_PATH = os.path.join(BASE_DIR, "data", "api_embeddings.npy")
META_PATH = os.path.join(BASE_DIR, "data", "api_metadata.json")
FAISS_PATH = os.path.join(BASE_DIR, "data", "faiss_index.bin")


# ----------------------------------------------------------------------
# Load metadata
# ----------------------------------------------------------------------

def load_metadata():
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata not found: {META_PATH}")

    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Load FAISS index
# ----------------------------------------------------------------------

def load_faiss():
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(f"FAISS index not found: {FAISS_PATH}")

    index = faiss.read_index(FAISS_PATH)
    return index


# ----------------------------------------------------------------------
# Load embedding matrix (for ask/evaluate)
# ----------------------------------------------------------------------

def load_embedding_matrix():
    if not os.path.exists(EMBED_PATH):
        raise FileNotFoundError(f"Embedding matrix missing: {EMBED_PATH}")

    return np.load(EMBED_PATH)


# ----------------------------------------------------------------------
# Query FAISS index
# ----------------------------------------------------------------------

def query_index(query_vector: np.ndarray, top_k: int = 10):
    index = load_faiss()
    metadata = load_metadata()

    if query_vector.ndim == 1:
        query_vector = np.expand_dims(query_vector, axis=0)

    scores, ids = index.search(query_vector, top_k)

    out = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        entry = metadata[idx]
        entry_out = {
            "id": entry["id"],
            "api_name": entry["api_name"],
            "metadata": entry,
            "score": float(score),
        }
        out.append(entry_out)

    return out
