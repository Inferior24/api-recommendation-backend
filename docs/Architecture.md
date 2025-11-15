# System Architecture

This section explains the high-level architecture.

```mermaid
graph TD;
    User[User Query]
    Retriever["Retriever Module<br/>(Semantic Search)"]
    Ranker["Ranking Engine<br/>(Score & Rank Candidates)"]
    Backend["Backend API Server"]
    Response["Recommendation Response"]

    User --> Backend
    Backend --> Retriever
    Retriever --> Ranker
    Ranker --> Backend
    Backend --> Response
```

- **Retriever:** Uses semantic embeddings for API/document search.
- **Ranker:** Orders results based on task, relevance, and evaluation metrics.
- **Backend:** Exposes RESTful API endpoints for interaction.

See [Pipeline Flow](Pipeline_Flow.md) for end-to-end processing.