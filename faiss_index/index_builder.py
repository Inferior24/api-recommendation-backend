# index_builder.py

import faiss
import numpy as np
import json
from pathlib import Path

def build_faiss_index(embedding_path="data/api_embeddings.npy",
                      metadata_path="data/api_metadata.json",
                      index_path="data/faiss_index.bin"):
    Path("data").mkdir(exist_ok=True)

    # Load embeddings and metadata
    embeddings = np.load(embedding_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    dim = embeddings.shape[1]
    print(f"\n🧠 Building FAISS index | Dimension = {dim}")

    # Initialize index
    index = faiss.IndexFlatIP(dim)   # Inner Product for cosine similarity
    index.add(embeddings)

    # Save index
    faiss.write_index(index, index_path)
    print(f"✅ FAISS index built and saved to: {index_path}")
    print(f"✅ Total vectors indexed: {index.ntotal}")

def test_search(query, model_name="all-MiniLM-L6-v2",
                index_path="data/faiss_index.bin", metadata_path="data/api_metadata.json", top_k=5):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    query_vec = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(query_vec, top_k)

    print(f"\n🔍 Query: {query}\n")
    for i, idx in enumerate(ids[0]):
        print(f"{i+1}. {metadata[idx]['api_name']}  |  Score: {round(float(scores[0][i]), 4)}")

if __name__ == "__main__":
    build_faiss_index()
    # Optional: quick sanity check
    test_search("AI text generation API")
