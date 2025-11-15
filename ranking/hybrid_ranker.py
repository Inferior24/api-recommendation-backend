# ranking/hybrid_ranker.py
"""
Safe wrapper / helper for hybrid ranking functions.
Make sure exception blocks bind `as e`.
"""

from typing import List, Dict, Any

def hybrid_rank(metadata_list: List[Dict[str, Any]], sim_scores: List[float], weights: Dict[str, float]) -> List[Dict[str, Any]]:
    try:
        # Minimal example: combine similarity and provided weights (fallback)
        results = []
        for i, md in enumerate(metadata_list):
            sim = float(sim_scores[i]) if i < len(sim_scores) else 0.0
            score = weights.get("similarity", 1.0) * sim
            results.append({"id": md.get("id", f"doc_{i}"), "score": score, "metadata": md})
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results
    except Exception as e:
        raise RuntimeError(f"hybrid_rank failed: {e}")
