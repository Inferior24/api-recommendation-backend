"""
rag/composer.py
-------------------------------------------------
Explainability Composer for API Recommendation Assistant.
Generates human-readable justification for top-ranked API.
"""

from typing import Dict, Any

def _fmt(details: Dict[str, float]) -> str:
    return "\n".join([f"• {k.capitalize()}: {v:.2f}" for k, v in details.items()]) if details else "No data."


def explain_top_result(top: Dict[str, Any]) -> str:
    if not top:
        return "No result available for explanation."
    name = top.get("metadata", {}).get("name") or top.get("id", "Unnamed API")
    parts = {
        "similarity": top.get("similarity", 0),
        "quality": top.get("doc_quality", 0),
        "recency": top.get("recency", 0),
        "popularity": top.get("popularity", 0),
    }
    total = top.get("hybrid_score", 0)
    msg = (
        f"API '{name}' ranked highest with a hybrid score of {total:.3f}.\n"
        f"It balances semantic match ({parts['similarity']:.2f}), quality ({parts['quality']:.2f}), "
        f"recency ({parts['recency']:.2f}), and popularity ({parts['popularity']:.2f}).\n\n"
        f"Component Breakdown:\n{_fmt(parts)}\n\n"
        "This reflects adaptive weighting tuned to the user's intent."
    )
    return msg
