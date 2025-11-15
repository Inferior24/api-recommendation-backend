# System Architecture — API Recommendation System

Date: YYYY-MM-DD
Version: 1.0

## High-level components
1. Retriever
   - Location: retriever/semantic_search.py
   - Responsibilities: load FAISS, produce top-N candidate docs with embeddings and similarity scores.
   - Notes: already implemented; we will standardize fallback outputs.

2. Ranker
   - Location: ranking/dynamic_ranker.py
   - Responsibilities: normalize numeric fields, compute hybrid_score using configured intent weights, output ranked list.
   - Notes: do NOT change ranking formulas.

3. Composer (Explainability)
   - Location: rag/composer.py
   - Responsibilities: generate human-readable explanation, weight breakdown for final chosen result.

4. Pipeline (New)
   - Location: backend/pipeline.py
   - Responsibilities: orchestrate flow:
       a) semantic_retrieve(query, top_k, filters) -> candidate_docs
       b) score_documents(candidate_docs, intent) -> ranked_docs
       c) explain_top_result(ranked_docs[0]) -> explanation
       d) assemble final JSON using schemas.py and logger
   - Important: pipeline will not implement ranking logic — it calls existing ranker functions.

5. Backend API (FastAPI)
   - Location: backend/main.py
   - Responsibilities: expose endpoints; minimal orchestration and validation; delegate to pipeline.run_pipeline()
   - Behavior: endpoints must use `backend/schemas.py` for Pydantic models and `backend/config.yaml` for static paths.

## Data & config
- FAISS index, embeddings, dataset paths must move to `backend/config.yaml`.
- All modules read config via a centralized loader (we will add a small helper in pipeline.py or import from config loader).

## Logging & errors
- Add `backend/logger.py` to centralize JSON-line logs.
- Replace prints in endpoints with logger.info/error.
- Standardize `request_id` flow: if request does not provide one, pipeline should generate a UUID and echo it in response.

## Sequence diagram (text)
Client -> /recommend -> backend/main.py -> pipeline.run_pipeline()
pipeline.run_pipeline():
  -> retriever.semantic_retrieve(query, top_k, filters)
  -> ranking.dynamic_ranker.score_documents(candidates, intent)
  -> rag.composer.explain_top_result(best_doc)
  -> assemble envelope and return

## Testing plan (manual)
- Start server: `uvicorn backend.main:app --reload --port 8000`
- POST /recommend with test payload (see docs/api_contract.md)
- Confirm response schema, top_k respected, fields: id, similarity, doc_quality, recency, popularity, hybrid_score, metadata
- If retriever fails, confirm `status: error` and empty results
- See `tests/e2e_manual_checklist.md` for full manual steps (to be created later)

## Constraints & non-goals
- Do not change retriever/ranker/composer algorithms.
- Scope is backend only (no UI).
