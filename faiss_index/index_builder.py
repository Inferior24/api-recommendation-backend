"""
Builds a FAISS index from embedding vectors + metadata.

Fixes:
- Consistent project-root-relative paths
- Automatic directory resolution
"""

import os
import json
import numpy as np
import faiss


# ----------------------------------------------------------------------
# Correct project-relative paths
# ----------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBED_PATH = os.path.join(BASE_DIR, "data", "api_embeddings.npy")
META_PATH = os.path.join(BASE_DIR, "data", "api_metadata.json")
FAISS_PATH = os.path.join(BASE_DIR, "data", "faiss_index.bin")


def load_embeddings(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")

    print(f"📦 Loading embeddings: {path}")
    return np.load(path)


def load_metadata(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")

    print(f"📦 Loading metadata: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_faiss_index(vectors: np.ndarray):
    dim = vectors.shape[1]
    print(f"\n🧠 Building FAISS index | Dimension = {dim}")

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dim)   # Inner product = cosine when normalized
    index.add(vectors)

    return index


def save_faiss_index(index, path: str):
    faiss.write_index(index, path)
    print(f"✅ FAISS index built and saved to: {path}")


def demo_query(index, metadata):
    query = "AI text generation API"
    print(f"\n🔍 Query: {query}\n")

    # Use embedding model again? No. We'll reuse metadata "cleaned_text"
    # This file only builds the index. It doesn't perform inference.
    # So we fake a query vector by embedding length match.

    # Quick patch: random normalized vector (for sanity testing)
    dim = index.d
    q = np.random.rand(dim).astype("float32")
    q /= np.linalg.norm(q)

    scores, ids = index.search(np.array([q]), 5)
    ids = ids[0]
    scores = scores[0]

    for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
        name = metadata[idx]["api_name"]
        print(f"{rank}. {name}  |  Score: {round(float(score), 4)}")


def main():
    vectors = load_embeddings(EMBED_PATH)
    metadata = load_metadata(META_PATH)

    index = build_faiss_index(vectors)

    save_faiss_index(index, FAISS_PATH)

    print(f"✅ Total vectors indexed: {index.ntotal}")

    # Optional sanity check
    demo_query(index, metadata)


if __name__ == "__main__":
    main()
