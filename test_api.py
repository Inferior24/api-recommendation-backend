import requests
import json

base = "http://127.0.0.1:8000"

def pp(x):
    print(json.dumps(x, indent=2))

print("\n=== /recommend ===")
r = requests.post(f"{base}/recommend", json={"query": "AI text generation API"})
pp(r.json())

print("\n=== /ask ===")
r = requests.post(f"{base}/ask", json={"query": "AI text generation API"})
pp(r.json())

print("\n=== /evaluate ===")
r = requests.post(f"{base}/evaluate", json={"query": "AI text generation API", "ground_truth_id": "doc_0"})
pp(r.json())
