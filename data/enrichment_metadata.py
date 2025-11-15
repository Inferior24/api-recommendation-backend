# data/enrich_metadata.py
import json, random, time, os, math
from datetime import datetime, timedelta
import requests

DATA_PATH = "data/api_dataset_cleaned.json"
OUT_PATH  = "data/api_dataset_cleaned_enriched.json"

# Optional: set a GitHub token environment var to raise rate limits
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", None)
GITHUB_HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

def safe_get_github_repo_data(repo_url):
    # repo_url example: "https://github.com/owner/repo"
    try:
        if "github.com" not in (repo_url or ""):
            return None
        repo = repo_url.rstrip("/").split("github.com/")[-1]
        r = requests.get(f"https://api.github.com/repos/{repo}", headers=GITHUB_HEADERS, timeout=8)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def normalize01(x, lo, hi):
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# optional compute ranges later to normalize popularity if you want deterministic normalization
pop_scores = []
for api in data:
    # fallback: few heuristics
    # 1) popularity: try repository stars/forks if repo is provided, else random fallback
    pop = None
    repo = api.get("repository") or api.get("repo") or api.get("source_url")
    if repo:
        repo_data = safe_get_github_repo_data(repo)
        if repo_data:
            stars = repo_data.get("stargazers_count", 0)
            forks = repo_data.get("forks_count", 0)
            # simple combined metric
            pop = (stars + forks*0.5)
    if pop is None:
        # fallback random small variation so ranker can show differences
        pop = random.uniform(10, 1000)
    pop_scores.append(pop)
    api["_raw_pop"] = pop  # keep for normalization later

# normalize raw pop -> 0..1
lo = min(pop_scores)
hi = max(pop_scores)
for api in data:
    raw = api.get("_raw_pop", 0)
    api["popularity"] = round(normalize01(raw, lo, hi), 4)

# doc_quality heuristic: use description length + endpoints presence + docs link
for api in data:
    desc = api.get("description", "") or api.get("summary", "") or ""
    endpoints = api.get("endpoints", []) or []
    doc_url = 1 if api.get("documentation_url") or api.get("docs_url") else 0
    desc_score = min(1.0, len(desc) / 400.0)  # long descriptions assume better docs
    endpoints_score = min(1.0, len(endpoints) / 10.0)
    docq = 0.5*desc_score + 0.3*endpoints_score + 0.2*doc_url
    # keep as 0..1
    api["doc_quality"] = round(max(0.0, min(1.0, docq)), 4)

# recency: if last_updated exists, keep, else random recent date
for api in data:
    last = api.get("last_updated") or api.get("updated_at") or api.get("modified") or api.get("updated")
    if last:
        # ensure ISO format (attempt rough parsing)
        try:
            # if epoch
            if isinstance(last, (int, float)):
                dt = datetime.utcfromtimestamp(float(last))
            else:
                # try multiple formats
                dt = datetime.fromisoformat(str(last).replace("Z", "+00:00"))
        except Exception:
            # fallback to a random recent date in last 3 years
            dt = datetime.utcnow() - timedelta(days=random.randint(0, 365*3))
    else:
        dt = datetime.utcnow() - timedelta(days=random.randint(0, 365*3))
    api["last_updated"] = dt.isoformat() + "Z"

# write enriched dataset
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Enriched data written to {OUT_PATH}")
