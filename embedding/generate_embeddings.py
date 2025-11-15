"""
Generate embeddings for normalized metadata and save FAISS input files.
"""

import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------------------
# Resolve project-root-relative paths correctly
# ----------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "api_dataset_normalized.json")
EMBED_PATH = os.path.join(BASE_DIR, "data", "api_embeddings.npy")
META_PATH = os.path.join(BASE_DIR, "data", "api_metadata.json")

# ----------------------------------------------------------------------
# Load dataset
# ----------------------------------------------------------------------

def load_dataset(path):
    print(f"📦 Loading normalized dataset: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ----------------------------------------------------------------------
# Main embedding function
# ----------------------------------------------------------------------

def main():
    dataset = load_dataset(DATASET_PATH)

    texts = [item["cleaned_text"] for item in dataset]
    print(f"📦 Loaded {len(texts)} cleaned texts for embedding.")

    print("🧠 Loading SentenceTransformer model...")
    model = SentenceTransformer("all-mpnet-base-v2")

    print("⚙️ Generating embeddings...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    print("\n✅ Embedding generation complete.")

    # Save vectors
    np.save(EMBED_PATH, embeddings)
    print(f"✅ Saved vectors to: {EMBED_PATH}")

    # Save metadata for FAISS mapping
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Saved metadata to: {META_PATH}")
    print(f"✅ Embedding matrix shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
