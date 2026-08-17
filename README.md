# DPI AI Prototype

This repository contains a minimal prototype for a Digital Public Infrastructure (DPI) AI assistant.

What it includes:

- backend: FastAPI app implementing a simple RAG-like retrieval using TF-IDF (no LLM required for prototype). Endpoints:
  - POST /v1/auth/login : stub authentication returning a JWT
  - POST /v1/assistant/query : protected endpoint to submit queries and receive answers with sources and confidence
  - POST /v1/assistant/escalate : protected endpoint to escalate to human (stores escalation records)
- frontend: a minimal HTML page to try queries
- sample data: a few example policy documents to retrieve from

Next steps you can ask me to do:
- Add an actual LLM integration (open-source or managed) for answer generation
- Add a vector DB (Weaviate, Milvus) instead of TF-IDF
- Add unit tests, CI, Docker Compose, or Kubernetes manifests

Branch: prototype/dpi-ai

