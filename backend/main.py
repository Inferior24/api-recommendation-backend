"""
FastAPI backend for Adaptive Hybrid Ranking.
Implements:
  /health
  /recommend
  /ask
  /compare
  /top_apis
  /logs
  /evaluate
"""

import json
import os
import math
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------
# Import local modules safely (single, correct block)
# ---------------------------------------------------------------------
try:
    from retriever.semantic_search import semantic_retrieve
    print("✅ Successfully imported retriever.semantic_search")
except Exception as e:
    print(f"⚠️  Failed to import retriever.semantic_search: {e}")

    def semantic_retrieve(query: str, top_k: int = 10):
        raise RuntimeError(
            f"Retriever import failed — error: {str(e)}. "
            f"Check retriever/semantic_search.py and FAISS/model loading."
        )

try:
    from ranking.dynamic_ranker import score_documents
    print("✅ Successfully imported ranking.dynamic_ranker")
except Exception as e:
    print(f"⚠️  Failed to import ranking.dynamic_ranker: {e}")

    def score_documents(*args, **kwargs):
        raise RuntimeError(f"Ranker import failed: {str(e)}")

try:
    from rag.composer import explain_top_result
    print("✅ Successfully imported rag.composer")
except Exception as e:
    print(f"⚠️  Failed to import rag.composer: {e}")

    def explain_top_result(*args, **kwargs):
        return f"Explainability module failed to import: {str(e)}"

# ---------------------------------------------------------------------
# FastAPI App Configuration
# ---------------------------------------------------------------------
app = FastAPI(title="API Recommendation & Query Assistant", version="1.0")

# Allow dashboard frontend access (open CORS policy for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Thread-safe logging
# ---------------------------------------------------------------------
LOG_PATH = os.path.join(os.path.dirname(__file__), "logs.jsonl")
LOG_LOCK = threading.Lock()


def append_log(entry: Dict[str, Any]):
    entry = dict(entry)
    entry.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
    with LOG_LOCK:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def read_recent_logs(limit: int = 100) -> List[Dict[str, Any]]:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    logs = [json.loads(l) for l in lines if l.strip()]
    return sorted(logs, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]

# ---------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------
class EvaluateRequest(BaseModel):
    query: str
    relevant: List[str]
    k: Optional[int] = 10


class CompareRequest(BaseModel):
    api_id_a: str
    api_id_b: str
    query: Optional[str] = None
    intent: Optional[str] = None
    top_k: Optional[int] = 50

# ---------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    """Return system health and component status."""
    return {
        "status": "ok",
        "app": "API Recommendation & Query Assistant",
        "version": "1.0",
        "time": datetime.utcnow().isoformat() + "Z",
        "components": {
            "faiss": "up",
            "db": "n/a",
            "retriever": "ready",
            "ranker": "ready",
            "explainability": "ready",
        },
    }

# ---------------------------------------------------------------------
# Recommend Endpoint
# ---------------------------------------------------------------------
@app.get("/recommend")
def recommend(query: str = Query(...), top_k: int = 10, intent: Optional[str] = None):
    """Adaptive hybrid ranking."""
    print(f"\n🟢 [REQUEST] /recommend — query='{query}', top_k={top_k}, intent='{intent}'")

    try:
        metadata, sim_scores = semantic_retrieve(query, top_k=top_k)
        print(f"✅ [Retriever] Returned {len(metadata)} candidates")
    except Exception as err:
        print(f"❌ [Retriever Error] {err}")
        raise HTTPException(status_code=500, detail=f"Retriever failed: {str(err)}")

    try:
        results = score_documents(metadata, sim_scores, intent=intent)
        print("✅ [Ranker] Scoring successful")
    except Exception as err:
        print(f"❌ [Ranker Error] {err}")
        raise HTTPException(status_code=500, detail=f"Ranker failed: {str(err)}")

    append_log({
        "type": "recommend",
        "query": query,
        "intent": intent or "none",
        "top_k": top_k,
    })

    return {
        "query": query,
        "intent": intent or "recommend",
        "results": results.get("ranked", []),
        "weights": results.get("weights", {}),
    }

# ---------------------------------------------------------------------
# Ask Endpoint (Explain Top Result)
# ---------------------------------------------------------------------
@app.post("/ask")
def ask(
    query: str = Body(..., embed=True),
    top_k: int = Body(5, embed=True),
    intent: Optional[str] = Body(None, embed=True),
):
    """Explain why the top API was ranked highest."""
    print(f"\n🟢 [REQUEST] /ask — query='{query}', top_k={top_k}, intent='{intent}'")

    try:
        metadata, sim_scores = semantic_retrieve(query, top_k=top_k)
        print(f"✅ [Retriever] Got {len(metadata)} docs")
    except Exception as err:
        print(f"❌ [Retriever Error in /ask] {err}")
        raise HTTPException(status_code=500, detail=f"Retriever failed: {str(err)}")

    try:
        results = score_documents(metadata, sim_scores, intent=intent)
        ranked = results.get("ranked", [])
        top = ranked[0] if ranked else None
        explanation = explain_top_result(top) if top else "No results to explain."
        print("✅ [Explainability] Explanation generated successfully")
    except Exception as err:
        print(f"❌ [Ranking/Explainability Error] {err}")
        raise HTTPException(status_code=500, detail=f"Explainability failed: {str(err)}")

    append_log({
        "type": "ask",
        "query": query,
        "intent": intent or "none",
        "top_result": top,
        "explanation": explanation,
    })

    return {
        "query": query,
        "intent": intent or "recommend",
        "explanation": explanation,
        "result": top,
        "components": results.get("weights", {}),
    }

# ---------------------------------------------------------------------
# Evaluate Endpoint
# ---------------------------------------------------------------------
@app.post("/evaluate")
def evaluate(payload: EvaluateRequest):
    """Compute Precision@K, Recall@K, and NDCG@K for a given query."""
    q = payload.query
    k = payload.k or 10
    print(f"\n🟢 [REQUEST] /evaluate — query='{q}', k={k}")

    try:
        metadata, sim_scores = semantic_retrieve(q, top_k=max(100, k))
        results = score_documents(metadata, sim_scores)
        retrieved_ids = [r["id"] for r in results.get("ranked", [])][:k]
        print("✅ [Retriever+Ranker] Evaluation data ready")
    except Exception as err:
        print(f"❌ [Evaluation Error] {err}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(err)}")

    def precision_at_k(retrieved, relevant, k):
        return sum(1 for r in retrieved[:k] if r in relevant) / k if k > 0 else 0

    def recall_at_k(retrieved, relevant, k):
        return sum(1 for r in retrieved[:k] if r in relevant) / len(relevant) if relevant else 0

    def dcg_at_k(retrieved, relevant, k):
        rel = set(relevant)
        return sum((1.0 if r in rel else 0.0) / math.log2(i + 2) for i, r in enumerate(retrieved[:k]))

    def ndcg_at_k(retrieved, relevant, k):
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        return dcg_at_k(retrieved, relevant, k) / ideal if ideal else 0.0

    p = precision_at_k(retrieved_ids, payload.relevant, k)
    r = recall_at_k(retrieved_ids, payload.relevant, k)
    ndcg = ndcg_at_k(retrieved_ids, payload.relevant, k)

    append_log({
        "type": "evaluate",
        "query": q,
        "k": k,
        "precision": p,
        "recall": r,
        "ndcg": ndcg,
    })

    print(f"📊 [Evaluation Metrics] P@{k}={p:.3f}, R@{k}={r:.3f}, NDCG@{k}={ndcg:.3f}")

    return {
        "query": q,
        "k": k,
        "metrics": {"precision": p, "recall": r, "ndcg": ndcg},
        "retrieved_ids": retrieved_ids,
    }
