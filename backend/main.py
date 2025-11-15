"""
FastAPI backend for API Recommendation System.
Implements:
  /health
  /recommend
  /ask
  /evaluate
  /logs
"""

import json
import os
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------
# Backend utilities
# ------------------------------------------------------------
from backend.logger import get_logger
from backend.schemas import (
    RecommendRequest,
    AskRequest,
    EvaluateRequest
)
from backend.pipeline import (
    run_recommend_pipeline,
    run_ask_pipeline,
    run_evaluate_pipeline
)

# ------------------------------------------------------------
# Initialize logger, app
# ------------------------------------------------------------
logger = get_logger()

app = FastAPI(
    title="API Recommendation & Query Assistant",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------------------------------------------------
# Local log file (thread-safe)
# ------------------------------------------------------------
LOG_PATH = os.path.join(os.path.dirname(__file__), "logs.jsonl")
LOG_LOCK = threading.Lock()


def append_log(entry: Dict[str, Any]):
    entry = dict(entry)
    entry.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")

    with LOG_LOCK:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Logged event: {entry.get('type', 'unknown')} request_id={entry.get('request_id', '-')}")


def read_recent_logs(limit: int = 100) -> List[Dict[str, Any]]:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    logs = [json.loads(l) for l in lines if l.strip()]
    logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return logs[:limit]


# ------------------------------------------------------------
# /health
# ------------------------------------------------------------
@app.get("/health")
def health():
    """Simple system health endpoint."""
    return {
        "status": "ok",
        "app": "API Recommendation & Query Assistant",
        "version": "1.0",
        "time": datetime.utcnow().isoformat() + "Z",
        "components": {
            "faiss": "up",
            "retriever": "ready",
            "ranker": "ready",
            "explainability": "ready"
        }
    }


# ------------------------------------------------------------
# /recommend
# ------------------------------------------------------------
@app.get("/recommend")
def recommend(
    query: str = Query(...),
    top_k: int = 10,
    intent: Optional[str] = None,
    request_id: Optional[str] = None
):
    """
    Thin wrapper — delegates to backend.pipeline.run_recommend_pipeline()
    """
    logger.info(
        f"[REQUEST] /recommend query='{query}' top_k={top_k} intent='{intent}'"
    )

    req = RecommendRequest(
        request_id=request_id,
        query=query,
        intent=intent,
        top_k=top_k
    )

    try:
        resp = run_recommend_pipeline(req)
    except Exception as e:
        logger.error(f"[ERROR] /recommend pipeline failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    append_log({
        "type": "recommend",
        "request_id": resp.request_id,
        "query": query,
        "intent": intent or "none",
        "top_k": top_k
    })

    return JSONResponse(status_code=200, content=resp.dict())


# ------------------------------------------------------------
# /ask
# ------------------------------------------------------------
@app.post("/ask")
def ask(payload: AskRequest = Body(...)):
    """
    Thin wrapper — delegates to backend.pipeline.run_ask_pipeline()
    """
    logger.info(
        f"[REQUEST] /ask query='{payload.query}' top_k={payload.top_k} intent='{payload.intent}'"
    )

    try:
        resp = run_ask_pipeline(payload)
    except Exception as e:
        logger.error(f"[ERROR] /ask pipeline failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    append_log({
        "type": "ask",
        "request_id": resp.request_id,
        "query": payload.query,
        "intent": payload.intent or "none",
        "top_k": payload.top_k
    })

    return JSONResponse(status_code=200, content=resp.dict())


# ------------------------------------------------------------
# /evaluate
# ------------------------------------------------------------
@app.post("/evaluate")
def evaluate(payload: EvaluateRequest):
    """
    Thin wrapper — delegates to backend.pipeline.run_evaluate_pipeline()
    """
    logger.info(
        f"[REQUEST] /evaluate query='{payload.query}' top_k='{payload.top_k}'"
    )

    try:
        resp = run_evaluate_pipeline(payload)
    except Exception as e:
        logger.error(f"[ERROR] /evaluate pipeline failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    append_log({
        "type": "evaluate",
        "request_id": resp.request_id,
        "query": payload.query,
        "top_k": payload.top_k
    })

    return JSONResponse(status_code=200, content=resp.dict())


# ------------------------------------------------------------
# /logs
# ------------------------------------------------------------
@app.get("/logs")
def logs(limit: int = 100):
    """Fetch recent backend logs."""
    logger.info(f"[REQUEST] /logs limit={limit}")
    logs = read_recent_logs(limit=limit)
    return {"logs": logs}
