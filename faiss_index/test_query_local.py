# faiss_index/test_query_local.py
import yaml, os, json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

cfg_path = os.path.join(os.path.dirname(__file__), "..", "backend", "config.yaml")
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

EMBEDDINGS_IN = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", cfg["data"]["embeddings_npy"]))
METADATA_IN = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", cfg["data"]["metadata_json"]))
FAISS_INDEX_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", cfg["data"]["faiss_index"]))
MODEL_NAME = cfg["model"]["sentence_transformer"]

emb = np.load(EMBEDDINGS_IN)
meta = json.load(open(METADATA_IN, "r", encoding="utf-8"))
index = faiss.read_index(FAISS_INDEX_PATH)
model = SentenceTransformer(MODEL_NAME)

def query(q, k=5):
    q_emb = model.encode([q], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    for rank, (i, s) in enumerate(zip(I[0], D[0]), start=1):
        print(f"{rank}. {meta[i].get('api_name','UNKNOWN')} (id={meta[i].get('id')}) score={float(s):.4f}")

if __name__ == "__main__":
    print("Test query: AI text generation API")
    query("AI text generation API", k=10)
