# generate_embeddings.py

import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pathlib import Path

def generate_embeddings(input_path="data/api_dataset_cleaned.json",
                        output_vectors="data/api_embeddings.npy",
                        output_meta="data/api_metadata.json"):
    Path("data").mkdir(exist_ok=True)

    # Load cleaned dataset
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [d["cleaned_text"] for d in data]
    print(f"\n📦 Loaded {len(texts)} cleaned texts for embedding.")

    # Load model
    print("🧠 Loading SentenceTransformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings
    print("⚙️ Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # Save vectors and metadata
    np.save(output_vectors, embeddings)
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"\n✅ Embedding generation complete.")
    print(f"✅ Saved vectors to: {output_vectors}")
    print(f"✅ Saved metadata to: {output_meta}")
    print(f"✅ Embedding matrix shape: {embeddings.shape}")

if __name__ == "__main__":
    generate_embeddings()
