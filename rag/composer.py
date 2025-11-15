from typing import Dict, Any

def _fmt(details: Dict[str, float]) -> str:
    return "\n".join([f"• {k.capitalize()}: {v:.2f}" for k, v in details.items()]) if details else "No data."

def explain_top_result(top: Dict[str, Any]) -> str:
    if not top:
        return "No result available for explanation."

    md = top.get("metadata", {})

    name = md.get("api_name") or md.get("name") or top.get("id", "Unknown API")

    parts = {
        "similarity": float(top.get("similarity", 0)),
        "quality": float(top.get("doc_quality", 0)),
        "recency": float(top.get("recency", 0)),
        "popularity": float(top.get("popularity", 0)),
    }

    score = float(top.get("hybrid_score", 0))

    explanation = (
        f"API '{name}' ranked highest with a hybrid score of {score:.3f}.\n"
        f"It balances semantic match ({parts['similarity']:.2f}), "
        f"quality ({parts['quality']:.2f}), "
        f"recency ({parts['recency']:.2f}), "
        f"popularity ({parts['popularity']:.2f}).\n\n"
        f"Component Breakdown:\n{_fmt(parts)}\n\n"
        "This reflects adaptive weighting tuned to the user's intent."
    )

    return explanation
