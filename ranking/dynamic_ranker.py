# ranking/dynamic_ranker.py
"""
Adaptive Hybrid Ranker (Explainable MCDM-like Ranking)
Safe, explicit exception handling (no bare `except`).
"""

from typing import List, Dict, Any
from datetime import datetime
import math

__all__ = ["score_documents", "compute_dynamic_weights"]

_INTENT_WEIGHTS = {
    "recommend": {"similarity": 4.0, "doc_quality": 3.0, "recency": 1.0, "popularity": 2.0},
    "latest": {"similarity": 2.0, "doc_quality": 1.0, "recency": 5.0, "popularity": 1.0},
    "popular": {"similarity": 2.5, "doc_quality": 1.5, "recency": 1.0, "popularity": 5.0},
    "reliable": {"similarity": 3.0, "doc_quality": 5.0, "recency": 1.0, "popularity": 1.0},
    "default": {"similarity": 4.0, "doc_quality": 3.0, "recency": 1.0, "popularity": 2.0},
}

_DOC_QUALITY_CANDIDATES = ["doc_quality", "quality", "score", "rating", "stars"]
_POPULARITY_CANDIDATES = ["popularity", "usage_count", "uses", "downloads", "stars", "forks"]
_RECENCY_CANDIDATES = ["last_updated", "updated_at", "modified", "last_modified", "updated"]


def _normalize_weights(raw_weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(raw_weights.values()) or 1.0
    return {k: float(v) / s for k, v in raw_weights.items()}


def _get_field_safe(item: Dict[str, Any], candidates: List[str]):
    for k in candidates:
        if k in item and item[k] is not None:
            return item[k]
    return None


def _parse_date_to_epoch(d: Any) -> float:
    if d is None:
        return 0.0
    if isinstance(d, (int, float)):
        return float(d)
    if isinstance(d, str):
        s = d.strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.timestamp()
            except Exception:
                continue
        try:
            dt = datetime.fromisoformat(s)
            return dt.timestamp()
        except Exception:
            return 0.0
    return 0.0


def _minmax_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(hi, lo):
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def compute_dynamic_weights(intent: str) -> Dict[str, float]:
    intent_l = (intent or "").strip().lower()
    base = _INTENT_WEIGHTS.get(intent_l)
    if base is None:
        if "latest" in intent_l:
            base = _INTENT_WEIGHTS["latest"]
        elif "popular" in intent_l or "trend" in intent_l:
            base = _INTENT_WEIGHTS["popular"]
        elif "reliab" in intent_l or "quality" in intent_l or "reliable" in intent_l:
            base = _INTENT_WEIGHTS["reliable"]
        elif "recommend" in intent_l or not intent_l:
            base = _INTENT_WEIGHTS["recommend"]
        else:
            base = _INTENT_WEIGHTS["default"]
    return _normalize_weights(base)


def score_documents(
    metadata_list: List[Dict[str, Any]],
    sim_scores: List[float],
    intent: str = None,
) -> Dict[str, Any]:
    try:
        n = max(len(metadata_list), len(sim_scores or []))
        if n == 0:
            return {"weights": compute_dynamic_weights(intent), "ranked": []}

        raw_sim = list(sim_scores) + [0.0] * max(0, n - len(sim_scores))
        raw_quality = []
        raw_pop = []
        raw_recency = []

        for i in range(n):
            md = metadata_list[i] if i < len(metadata_list) else {}
            q = _get_field_safe(md, _DOC_QUALITY_CANDIDATES)
            try:
                qf = float(q) if q is not None else None
            except Exception:
                qf = None
            if qf is None:
                desc = md.get("description") or md.get("summary") or ""
                qf = min(5.0, max(0.0, len(str(desc)) / 200.0))
            raw_quality.append(float(qf))

            p = _get_field_safe(md, _POPULARITY_CANDIDATES)
            try:
                pf = float(p) if p is not None else 0.0
            except Exception:
                pf = 0.0
            raw_pop.append(float(pf))

            r = _get_field_safe(md, _RECENCY_CANDIDATES)
            epoch = _parse_date_to_epoch(r)
            raw_recency.append(float(epoch))

        sim_norm = []
        for s in raw_sim:
            try:
                sv = float(s)
            except Exception:
                sv = 0.0
            if sv < -1.0 or sv > 1.0:
                sim_norm.append(sv)
            else:
                sim_norm.append(max(0.0, min(1.0, (sv + 1.0) / 2.0 if sv < 0 else sv)))

        sim_norm = _minmax_normalize(sim_norm)
        q_norm = _minmax_normalize(raw_quality)
        pop_norm = _minmax_normalize(raw_pop)
        rec_norm = _minmax_normalize(raw_recency)

        weights = compute_dynamic_weights(intent)

        ranked = []
        for i in range(n):
            hybrid = (
                weights.get("similarity", 0.0) * sim_norm[i]
                + weights.get("doc_quality", 0.0) * q_norm[i]
                + weights.get("recency", 0.0) * rec_norm[i]
                + weights.get("popularity", 0.0) * pop_norm[i]
            )
            md = metadata_list[i] if i < len(metadata_list) else {}
            doc_id = md.get("id") or md.get("doc_id") or md.get("api_id") or f"doc_{i}"
            ranked.append(
                {
                    "id": doc_id,
                    "similarity": round(float(sim_norm[i]), 6),
                    "doc_quality": round(float(q_norm[i]), 6),
                    "recency": round(float(rec_norm[i]), 6),
                    "popularity": round(float(pop_norm[i]), 6),
                    "hybrid_score": round(float(hybrid), 6),
                    "metadata": md,
                }
            )

        ranked_sorted = sorted(ranked, key=lambda x: x["hybrid_score"], reverse=True)
        return {"weights": weights, "ranked": ranked_sorted}

    except Exception as e:
        # Provide a clear error message for upstream handlers
        raise RuntimeError(f"score_documents failed: {e}")
