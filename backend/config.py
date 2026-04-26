import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
BACKEND_ROOT = Path(__file__).parent
DOCS_DIR = PROJECT_ROOT / "docs"
DATA_DIR = BACKEND_ROOT / "data"
CHROMA_DB_PATH = DATA_DIR / "chroma_db"

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DB_PATH.mkdir(exist_ok=True)

# Ollama Configuration
OLLAMA_BASE_URL =  "http://localhost:11434"

# LLM Model Configuration
LLM_MODEL = "gemma4:e4b"

# Embeddings Configuration
EMBEDDING_MODEL = "mxbai-embed-large"

# RAG Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))

# FastAPI Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"

# Web Search Configuration
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", 3))

# LLM Generation Configuration
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
