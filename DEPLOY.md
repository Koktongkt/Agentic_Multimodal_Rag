Setup and run (Updated for Chroma + Ollama)

Backend (Python / FastAPI):

1. Create a virtualenv and activate it:
   python -m venv .venv
   .\.venv\Scripts\activate

2. Install requirements:
   pip install -r backend\requirements.txt

3. Requirements and services:
   - Ollama must be running locally for embeddings (https://ollama.ai). By default the code calls http://localhost:11434/embed with model mxbai-embed-large. Set OLLAMA_URL or EMBED_MODEL environment variables to change.
   - ChromaDB persistence directory will be created at backend/../chroma_db (relative to project).

4. Ingest markdown docs:
   - Place your .md files inside the docs/ folder (agentic_rag/docs)
   - Trigger ingestion (first time or after changes):
       POST http://localhost:8000/ingest?force=true
     or from command line use curl:
       curl -X POST "http://localhost:8000/ingest?force=true"

5. Start the backend:
   uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

Frontend (React / Vite):

1. cd frontend
2. npm install
3. npm run dev

Notes:
- The backend exposes /chat which accepts POST {message: string} and returns JSON.
- RAG now uses Ollama embeddings and ChromaDB; ensure Ollama is running and chromadb is installed.
- Web search uses duckduckgo_search package.

Development tips:
- Adjust route logic in backend/app.py (route_query) to tune when web search vs RAG is used.
- If embeddings fail, check OLLAMA_URL and that the embed model is available.

