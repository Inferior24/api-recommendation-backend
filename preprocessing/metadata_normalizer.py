import json
import os
from datetime import datetime
from typing import Any, Dict, List


# ---------------------------------------------------------
# Helper: safe float cast
# ---------------------------------------------------------
def _to_float(v, default=0.0):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


# ---------------------------------------------------------
# Helper: parse ISO timestamps → epoch seconds
# ---------------------------------------------------------
def _parse_date(d):
    if not d:
        return 0.0
    if isinstance(d, (int, float)):
        return float(d)

    s = str(d).strip()
    formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y",
    ]

    for fmt in formats:
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


# ---------------------------------------------------------
# Helper: normalize popularity
# Uses existing:
#   popularity_score (0–1)
#   _raw_pop (integer-ish)
#   popularity (0–1)
# ---------------------------------------------------------
def _normalize_popularity(item: Dict[str, Any]):
    pop_score = _to_float(item.get("popularity_score"), None)
    pop_norm = _to_float(item.get("popularity"), None)
    raw_pop = _to_float(item.get("_raw_pop"), None)

    # If normalized score is already present → trust it
    if pop_norm is not None and 0 <= pop_norm <= 1:
        return pop_norm

    # If pop_score exists, treat it as normalized
    if pop_score is not None and 0 <= pop_score <= 1:
        return pop_score

    # If raw_pop exists → minmax cannot be done here, fallback sigmoid
    if raw_pop is not None:
        return 1 / (1 + (1 / (raw_pop + 1)))

    # fallback
    return 0.0


# ---------------------------------------------------------
# Normalize ONE item
# ---------------------------------------------------------
def normalize_item(item: Dict[str, Any], idx: int):
    cleaned = {}

    cleaned["id"] = f"doc_{idx}"

    cleaned["api_name"] = item.get("api_name", "").strip()
    cleaned["description"] = item.get("description", "").strip()
    cleaned["endpoints"] = item.get("endpoints", []) or []
    cleaned["documentation_url"] = item.get("documentation_url", "")

    cleaned["version"] = str(item.get("version", "")).strip()

    cleaned["cleaned_text"] = item.get("cleaned_text", "").strip()

    # numerical fields
    cleaned["doc_quality"] = _to_float(item.get("doc_quality"), 0.0)
    cleaned["popularity"] = _normalize_popularity(item)

    # recency → epoch
    cleaned["recency_epoch"] = _parse_date(item.get("last_updated"))

    return cleaned


# ---------------------------------------------------------
# Normalize entire dataset
# ---------------------------------------------------------
def normalize_dataset(input_path: str, output_path: str):
    print(f"🔄 Loading dataset: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    normalized = []

    for idx, item in enumerate(data):
        norm = normalize_item(item, idx)
        normalized.append(norm)

    print(f"💾 Saving normalized dataset: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)

    print("✅ Normalization complete.")
    return normalized


if __name__ == "__main__":
    # Example run
    in_path = "data/api_dataset_cleaned.json"
    out_path = "data/api_dataset_normalized.json"
    normalize_dataset(in_path, out_path)
