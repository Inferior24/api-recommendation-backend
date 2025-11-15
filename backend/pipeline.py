# backend/pipeline.py
"""
Unified backend pipeline that orchestrates:
  semantic_retrieve -> score_documents -> explain_top_result -> assemble response

Design goals:
- Defensive: handle missing/odd return types from retriever
- Stable: always return Pydantic response models
- JSON-serializable: convert DocumentResult -> dict for payloads
- Non-invasive: do NOT change internal logic of retriever/ranker/composer
"""

from __future__ import annotations
import uuid
import time
from typing import List, Dict, Any, Tuple, Optional

from backend.logger import get_logger
from backend.schemas import (
    RecommendRequest,
    RecommendResponse,
    AskRequest,
    AskResponse,
    EvaluateRequest,
    EvaluateResponse,
    DocumentResult,
)
from backend.config import load_config

# Existing working modules (do NOT change their internals)
from retriever.semantic_search import semantic_retrieve  # expected -> (metadata_list, sim_scores)
from ranking.dynamic_ranker import score_documents     # expected -> {"weights":..., "ranked":[...]}
from rag.composer import explain_top_result            # explanation generator (string or dict)

logger = get_logger()
config = load_config()


def _ensure_request_id(req_id: Optional[str]) -> str:
    """Return given request_id or generate a new UUID string."""
    return req_id if req_id else str(uuid.uuid4())


def _normalize_retriever_output(
    raw: Any
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Defensive adapter for semantic_retrieve outputs.

    Accepts:
      - (metadata_list, sim_scores)
      - metadata_list only (list) -> will synthesize sim_scores as zeros
      - empty list / None -> returns ([], [])

    Returns:
      (metadata_list, sim_scores) where both are lists and lengths match (sim_scores may be shorter).
    """
    if raw is None:
        return [], []

    # If retriever returns a 2-tuple-like
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        metadata_list, sim_scores = raw
        if metadata_list is None:
            metadata_list = []
        if sim_scores is None:
            sim_scores = []
        # Ensure types
        if not isinstance(metadata_list, list):
            metadata_list = list(metadata_list)
        if not isinstance(sim_scores, list):
            sim_scores = list(sim_scores)
        return metadata_list, sim_scores

    # If retriever returned only a metadata list
    if isinstance(raw, list):
        return raw, []

    # Unknown return -> fallback
    return [], []


def _docresult_from_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert ranking output document dict to a JSON-serializable dict that matches DocumentResult.
    We build a DocumentResult pydantic model then return its .dict() for consistent field names/types.
    """
    # Provide defaults in case fields missing
    doc = {
        "id": d.get("id") or d.get("doc_id") or d.get("api_id") or "unknown",
        "similarity": float(d.get("similarity") or 0.0),
        "doc_quality": float(d.get("doc_quality") or 0.0),
        "recency": float(d.get("recency") or 0.0),
        "popularity": float(d.get("popularity") or 0.0),
        "hybrid_score": float(d.get("hybrid_score") or 0.0),
        "metadata": d.get("metadata") or {},
    }
    # Use DocumentResult to validate/normalize
    dr = DocumentResult(**doc)
    return dr.dict()


# ---------------------------
# Recommend pipeline
# ---------------------------
def run_recommend_pipeline(req: RecommendRequest) -> RecommendResponse:
    request_id = _ensure_request_id(req.request_id)
    t0 = time.time()
    logger.info(f"[{request_id}] Starting recommend pipeline (q='{req.query}', top_k={req.top_k})")

    # 1) Retrieval (defensive)
    try:
        raw = semantic_retrieve(req.query, top_k=req.top_k)
        metadata_list, sim_scores = _normalize_retriever_output(raw)
        logger.info(f"[{request_id}] Retriever returned {len(metadata_list)} items")
    except Exception as e:
        logger.exception(f"[{request_id}] Retriever failed: {e}")
        return RecommendResponse(
            request_id=request_id,
            status="error",
            error={"code": "RETRIEVER_ERROR", "message": str(e)},
            payload={
                "query": req.query,
                "intent": req.intent,
                "results": [],
                "timing": {
                    "retrieval_ms": int((time.time() - t0) * 1000),
                    "ranking_ms": 0,
                    "compose_ms": 0,
                },
            },
        )

    retrieval_ms = int((time.time() - t0) * 1000)

    # 2) Ranking
    t1 = time.time()
    try:
        rank_out = score_documents(metadata_list, sim_scores, intent=req.intent)
    except Exception as e:
        logger.exception(f"[{request_id}] Ranker failed: {e}")
        return RecommendResponse(
            request_id=request_id,
            status="error",
            error={"code": "RANKER_ERROR", "message": str(e)},
            payload={
                "query": req.query,
                "intent": req.intent,
                "results": [],
                "timing": {
                    "retrieval_ms": retrieval_ms,
                    "ranking_ms": int((time.time() - t1) * 1000),
                    "compose_ms": 0,
                },
            },
        )

    ranking_ms = int((time.time() - t1) * 1000)

    # rank_out expected: {"weights": {...}, "ranked": [ {id, similarity, ... , metadata}, ... ]}
    ranked_docs = rank_out.get("ranked", []) if isinstance(rank_out, dict) else []
    weights = rank_out.get("weights", {}) if isinstance(rank_out, dict) else {}

    # 3) Normalize results to DocumentResult dicts for JSON
    results = []
    for d in ranked_docs:
        try:
            results.append(_docresult_from_dict(d))
        except Exception:
            # If something odd happens, fall back to a minimal shape
            fld = {
                "id": d.get("id", "unknown"),
                "similarity": float(d.get("similarity") or 0.0),
                "doc_quality": float(d.get("doc_quality") or 0.0),
                "recency": float(d.get("recency") or 0.0),
                "popularity": float(d.get("popularity") or 0.0),
                "hybrid_score": float(d.get("hybrid_score") or 0.0),
                "metadata": d.get("metadata", {}),
            }
            results.append(fld)

    # 4) Build response
    resp = RecommendResponse(
        request_id=request_id,
        status="ok",
        error=None,
        payload={
            "query": req.query,
            "intent": req.intent,
            "results": results,
            "weights": weights,
            "timing": {
                "retrieval_ms": retrieval_ms,
                "ranking_ms": ranking_ms,
                "compose_ms": 0,
            },
        },
    )
    logger.info(f"[{request_id}] Recommend pipeline done — {len(results)} results (retrieval={retrieval_ms}ms ranking={ranking_ms}ms)")
    return resp


# ---------------------------
# Ask pipeline (explain top result)
# ---------------------------
def run_ask_pipeline(req: AskRequest) -> AskResponse:
    request_id = _ensure_request_id(req.request_id)
    t0 = time.time()
    logger.info(f"[{request_id}] Starting ask pipeline (q='{req.query}', top_k={req.top_k})")

    # 1) Retrieve
    try:
        raw = semantic_retrieve(req.query, top_k=req.top_k)
        metadata_list, sim_scores = _normalize_retriever_output(raw)
        logger.info(f"[{request_id}] Retriever returned {len(metadata_list)} items")
    except Exception as e:
        logger.exception(f"[{request_id}] Retriever failed: {e}")
        return AskResponse(
            request_id=request_id,
            status="error",
            error={"code": "RETRIEVER_ERROR", "message": str(e)},
            payload={
                "answer": "",
                "source": None,
                "explanation": None,
                "timing": {
                    "retrieval_ms": int((time.time() - t0) * 1000),
                    "ranking_ms": 0,
                    "compose_ms": 0,
                },
            },
        )

    retrieval_ms = int((time.time() - t0) * 1000)

    # 2) Rank
    t1 = time.time()
    try:
        rank_out = score_documents(metadata_list, sim_scores, intent=req.intent)
    except Exception as e:
        logger.exception(f"[{request_id}] Ranker failed: {e}")
        return AskResponse(
            request_id=request_id,
            status="error",
            error={"code": "RANKER_ERROR", "message": str(e)},
            payload={
                "answer": "",
                "source": None,
                "explanation": None,
                "timing": {"retrieval_ms": retrieval_ms, "ranking_ms": int((time.time() - t1) * 1000), "compose_ms": 0},
            },
        )
    ranking_ms = int((time.time() - t1) * 1000)

    ranked_docs = rank_out.get("ranked", []) if isinstance(rank_out, dict) else []

    if not ranked_docs:
        return AskResponse(
            request_id=request_id,
            status="ok",
            error=None,
            payload={
                "answer": "No relevant results found.",
                "source": None,
                "explanation": None,
                "timing": {"retrieval_ms": retrieval_ms, "ranking_ms": ranking_ms, "compose_ms": 0},
            },
        )

    # 3) Explain top result
    top = ranked_docs[0]
    try:
        t2 = time.time()
        explanation = explain_top_result(top)
        compose_ms = int((time.time() - t2) * 1000)
    except Exception as e:
        logger.exception(f"[{request_id}] Explainability failed: {e}")
        explanation = str(e)
        compose_ms = 0

    # Build top document structure
    try:
        top_doc = _docresult_from_dict(top)
    except Exception:
        top_doc = {
            "id": top.get("id", "unknown"),
            "similarity": float(top.get("similarity") or 0.0),
            "doc_quality": float(top.get("doc_quality") or 0.0),
            "recency": float(top.get("recency") or 0.0),
            "popularity": float(top.get("popularity") or 0.0),
            "hybrid_score": float(top.get("hybrid_score") or 0.0),
            "metadata": top.get("metadata", {}),
        }

    payload = {
        "answer": explanation if isinstance(explanation, str) else explanation,
        "source": {"document": top_doc, "excerpt": (explanation.get("excerpt") if isinstance(explanation, dict) else "")},
        "explanation": explanation,
        "timing": {"retrieval_ms": retrieval_ms, "ranking_ms": ranking_ms, "compose_ms": compose_ms},
    }

    resp = AskResponse(request_id=request_id, status="ok", error=None, payload=payload)
    logger.info(f"[{request_id}] Ask pipeline done — top_doc={top_doc.get('id')}")
    return resp


# ---------------------------
# Evaluate pipeline (basic)
# ---------------------------
def run_evaluate_pipeline(req: EvaluateRequest) -> EvaluateResponse:
    request_id = _ensure_request_id(req.request_id)
    t0 = time.time()
    logger.info(f"[{request_id}] Starting evaluate pipeline (q='{req.query}', top_k={req.top_k})")

    # 1) Retrieve
    try:
        raw = semantic_retrieve(req.query, top_k=req.top_k)
        metadata_list, sim_scores = _normalize_retriever_output(raw)
    except Exception as e:
        logger.exception(f"[{request_id}] Retriever failed: {e}")
        return EvaluateResponse(request_id=request_id, status="error", error={"code": "RETRIEVER_ERROR", "message": str(e)}, payload=None)

    retrieval_ms = int((time.time() - t0) * 1000)

    # 2) Rank
    t1 = time.time()
    try:
        rank_out = score_documents(metadata_list, sim_scores, intent=None)
    except Exception as e:
        logger.exception(f"[{request_id}] Ranker failed: {e}")
        return EvaluateResponse(request_id=request_id, status="error", error={"code": "RANKER_ERROR", "message": str(e)}, payload=None)
    ranking_ms = int((time.time() - t1) * 1000)

    ranked_docs = rank_out.get("ranked", []) if isinstance(rank_out, dict) else []

    results = []
    for d in ranked_docs:
        try:
            results.append(_docresult_from_dict(d))
        except Exception:
            results.append(
                {
                    "id": d.get("id", "unknown"),
                    "similarity": float(d.get("similarity") or 0.0),
                    "doc_quality": float(d.get("doc_quality") or 0.0),
                    "recency": float(d.get("recency") or 0.0),
                    "popularity": float(d.get("popularity") or 0.0),
                    "hybrid_score": float(d.get("hybrid_score") or 0.0),
                    "metadata": d.get("metadata", {}),
                }
            )

    # Placeholder metrics (can be replaced with real calculations)
    metrics = {"precision@k": 0.0, "recall@k": 0.0, "ndcg@k": 0.0}

    resp = EvaluateResponse(
        request_id=request_id,
        status="ok",
        error=None,
        payload={
            "query": req.query,
            "ground_truth": req.ground_truth,
            "metrics": metrics,
            "results": results,
            "timing": {"retrieval_ms": retrieval_ms, "ranking_ms": ranking_ms, "eval_ms": 0},
        },
    )
    logger.info(f"[{request_id}] Evaluate pipeline done — results={len(results)}")
    return resp
