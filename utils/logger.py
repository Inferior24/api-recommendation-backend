import json
import os
from datetime import datetime

# Global path to log file
LOG_PATH = os.path.join("logs", "query_logs.json")

def log_query(query: str, response_time: float, results: list):
    """
    Append query logs to JSON file.
    Each record stores query text, latency, and top recommendation results.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "response_time": response_time,
        "results": [
            {
                "api_name": r.get("api_name"),
                "hybrid_score": r.get("hybrid_score"),
                "similarity": r.get("similarity"),
                "recency": r.get("recency"),
                "doc_quality": r.get("doc_quality"),
                "popularity": r.get("popularity"),
            }
            for r in results
        ],
    }

    try:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data.append(log_entry)
                f.seek(0)
                json.dump(data, f, indent=2)
        else:
            with open(LOG_PATH, "w", encoding="utf-8") as f:
                json.dump([log_entry], f, indent=2)
    except Exception as e:
        print(f"[LOGGER ERROR] {e}")
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump([log_entry], f, indent=2)
