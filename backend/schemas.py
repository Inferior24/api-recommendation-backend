from typing import Optional, List, Dict, Any
from pydantic import BaseModel


# ------------------------
# Document result schema
# ------------------------
class DocumentResult(BaseModel):
    id: str
    similarity: float
    doc_quality: float
    recency: float
    popularity: float
    hybrid_score: float
    metadata: Dict[str, Any]


# ------------------------
# Recommend endpoint
# ------------------------
class RecommendRequest(BaseModel):
    request_id: Optional[str] = None
    query: str
    intent: Optional[str] = None
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = None


class RecommendResponse(BaseModel):
    request_id: str
    status: str
    error: Optional[Dict[str, Any]]
    payload: Dict[str, Any]


# ------------------------
# Ask endpoint
# ------------------------
class AskRequest(BaseModel):
    request_id: Optional[str] = None
    query: str
    intent: Optional[str] = None
    top_k: int = 5


class AskResponse(BaseModel):
    request_id: str
    status: str
    error: Optional[Dict[str, Any]]
    payload: Dict[str, Any]


# ------------------------
# Evaluate endpoint
# ------------------------
class EvaluateRequest(BaseModel):
    request_id: Optional[str] = None
    query: str
    ground_truth: str
    top_k: int = 10


class EvaluateResponse(BaseModel):
    request_id: str
    status: str
    error: Optional[Dict[str, Any]]
    payload: Optional[Dict[str, Any]]
