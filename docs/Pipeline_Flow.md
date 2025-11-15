# Pipeline Flow

Stepwise flow from user input to recommendation output:

1. **User Submission:** Receives natural language queries.
2. **Retriever Module:** Converts query to embeddings; fetches candidate APIs.
3. **Ranking Engine:** Evaluates, scores, and sorts results.
4. **Recommendation:** Renders best-matched API with usage instructions.
5. **Feedback Loop (optional):** User ratings can be logged for continuous improvement.

See [API Endpoints](API_Endpoints.md) for interaction formats.