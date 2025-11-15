import os
import json
import threading
from typing import List, Tuple
import numpy as np
import faiss
import yaml
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Absolute project root resolution
# ---------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CFG_PATH = os.path.join(_PROJECT_ROOT, "backend", "config.yaml")

with open(_CFG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Paths (absolute)
EMBEDDINGS_NPY = os.path.join(_PROJECT_ROOT, cfg["data"]["embeddings_npy"])
METADATA_JSON = os.path.join(_PROJECT_ROOT, cfg["data"]["metadata_json"])
FAISS_INDEX_PATH = os.path.join(_PROJECT_ROOT, cfg["data"]["faiss_index"])

MODEL_NAME = cfg["model"]["sentence_transformer"]
TOP_K_DEFAULT = cfg["retriever"]["top_k_default"]

_lock = threading.Lock()
_index = None
_metadata = None
_model = None


def _load_index():
    global _index, _metadata, _model
    with _lock:
        if _index is not None:
            return

        # Load metadata
        if not os.path.exists(METADATA_JSON):
            raise FileNotFoundError(f"Metadata missing: {METADATA_JSON}")
        with open(METADATA_JSON, "r", encoding="utf-8") as f:
            _metadata = json.load(f)

        # Load FAISS index
        if not os.path.exists(FAISS_INDEX_PATH):
            raise FileNotFoundError(f"FAISS index missing: {FAISS_INDEX_PATH}")
        _index = faiss.read_index(FAISS_INDEX_PATH)

        # Load SentenceTransformer model
        _model = SentenceTransformer(MODEL_NAME)


def _standardize_metadata(m: dict, idx: int) -> dict:
    out = dict(m)
    out.setdefault("id", out.get("id") or f"doc_{idx}")

    # doc_quality
    try:
        out["doc_quality"] = float(out.get("doc_quality", 0))
    except:
        out["doc_quality"] = 0.0

    # popularity
    try:
        out["popularity"] = float(out.get("popularity", 0))
    except:
        out["popularity"] = 0.0

    out.setdefault("last_updated", out.get("last_updated", ""))
    out.setdefault("cleaned_text", out.get("cleaned_text") or out.get("description", "")[:512])

    return out


def semantic_retrieve(query: str, top_k: int = None) -> Tuple[List[dict], List[float]]:
    if top_k is None:
        top_k = TOP_K_DEFAULT

    _load_index()

    if not query:
        return [], []

    # Encode
    q_emb = _model.encode([query], convert_to_numpy=True).astype("float32")

    # Normalize vector
    faiss.normalize_L2(q_emb)

    # Search
    D, I = _index.search(q_emb, top_k)

    ids = I[0].tolist()
    sim_scores = [float(max(0, min(1, s))) for s in D[0].tolist()]

    metadata_list = []
    for idx in ids:
        if 0 <= idx < len(_metadata):
            metadata_list.append(_standardize_metadata(_metadata[idx], idx))

    return metadata_list, sim_scores[:len(metadata_list)]
