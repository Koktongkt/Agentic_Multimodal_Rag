import os
import logging
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown

from .config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """
    Ingest documents using MarkItDown first, then fallback to LangChain loaders.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " "]
        )

        self.md = MarkItDown()


    # Primary loader: MarkItDown
    def load_with_markitdown(self, file_path: str) -> str:
        try:
            result = self.md.convert(file_path)
            text = result.text_content
            if text and text.strip():
                return text
        except Exception as e:
            logger.warning("MarkItDown failed for %s: %s", file_path, e)

        return None

    # Fallback PDF loader
    def load_pdf_fallback(self, file_path: str) -> str:
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            return "\n".join([d.page_content for d in docs])
        except Exception as e:
            logger.error("PyPDFLoader failed for %s: %s", file_path, e)
            return ""

    # Fallback DOCX loader
    def load_docx_fallback(self, file_path: str) -> str:
        try:
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            return "\n".join([d.page_content for d in docs])
        except Exception as e:
            logger.error("Docx2txtLoader failed for %s: %s", file_path, e)
            return ""

    # Generic file loader
    def load_file(self, file_path: str) -> Dict[str, Any]:
        ext = Path(file_path).suffix.lower()
        filename = Path(file_path).name

        content = None

        # 1. Try MarkItDown FIRST
        content = self.load_with_markitdown(file_path)

        # 2. Fallbacks
        if not content:
            if ext == ".pdf":
                content = self.load_pdf_fallback(file_path)
                ftype = "pdf"
            elif ext == ".docx":
                content = self.load_docx_fallback(file_path)
                ftype = "docx"
            elif ext in (".md", ".markdown"):
                content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
                ftype = "markdown"
            else:
                content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
                ftype = "text"
        else:
            ftype = "markitdown"

        return {
            "filename": filename,
            "content": content,
            "path": str(file_path),
            "type": ftype
        }

    # Load all documents
    def load_all_documents(self) -> List[Dict]:
        documents = []

        if not DOCS_DIR.exists():
            logger.warning("Documents directory not found: %s", DOCS_DIR)
            return documents

        patterns = ["*.pdf", "*.md", "*.markdown", "*.txt", "*.docx"]

        for pat in patterns:
            for file_path in DOCS_DIR.glob(pat):
                try:
                    doc = self.load_file(str(file_path))
                    documents.append(doc)
                    logger.info("Loaded: %s (%s)", file_path.name, doc["type"])
                except Exception as e:
                    logger.error("Failed loading %s: %s", file_path.name, e)

        logger.info("Loaded %d documents", len(documents))
        return documents

    # Chunking
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        chunks = []

        for doc in documents:
            split_texts = self.splitter.split_text(doc["content"])

            for i, chunk in enumerate(split_texts):
                chunks.append({
                    "filename": doc["filename"],
                    "chunk_id": i,
                    "content": chunk,
                    "source": doc["path"],
                    "type": doc.get("type", "text")
                })

        logger.info("Created %d chunks", len(chunks))
        return chunks