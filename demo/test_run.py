# demo/test_evaluation.py
"""
Simple demo that POSTs to /evaluate and prints results.
Run the FastAPI server first:
    uvicorn backend.main:app --host 127.0.0.1 --port 8000
Then run:
    python demo/test_evaluation.py
"""

import requests
import pprint
import sys
import time

BASE = "http://127.0.0.1:8000"

def run_demo():
    payload = {
        "query": "REST API for user authentication token generation",  # sample query — adjust to your domain
        # provide a toy relevant list of ids that exist in your dataset. Replace with actual ids for real metrics.
        "relevant": ["auth_api_1", "auth_api_2"],
        "k": 10
    }
    print("Posting evaluation request to /evaluate ...")
    r = requests.post(BASE + "/evaluate", json=payload, timeout=30)
    if r.status_code != 200:
        print("Failed:", r.status_code, r.text)
        sys.exit(1)
    print("Response:")
    pprint.pprint(r.json())

    # run adaptive ranking /recommend with intent
    print("\nCalling /recommend with intent='latest' ...")
    r2 = requests.get(BASE + "/recommend", params={"query": payload["query"], "top_k": 5, "intent": "latest"}, timeout=30)
    if r2.status_code != 200:
        print("Failed recommend:", r2.status_code, r2.text); sys.exit(1)
    rec = r2.json()
    print("Recommend response (top entries):")
    pprint.pprint(rec if isinstance(rec, dict) and "ranked" in rec else rec)

if __name__ == "__main__":
    run_demo()
