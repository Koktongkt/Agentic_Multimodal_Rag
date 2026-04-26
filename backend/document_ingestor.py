import logging
import os
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from markitdown import MarkItDown

from config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP

# Optional imports for additional formats
try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

logger = logging.getLogger(__name__)

# Reuse PDF loading pipeline from existing module when available
try:
    from document_loader import DocumentLoader as PDFLoader
except Exception:
    PDFLoader = None


class DocumentIngestor:
    """Ingests PDFs, text, markdown and docx files and splits them into chunks.

    Behavior mirrors document_loader.py for PDFs (uses the same robust pipeline when
    PDFLoader is available) and adds simple loaders for .txt, .md/.markdown and .docx.
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
        # If PDFLoader is available, instantiate it for PDF handling
        self.pdf_loader = PDFLoader() if PDFLoader is not None else None

    def load_text_file(self, file_path: str) -> str:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # fallback to latin-1 if utf-8 fails
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()

    def load_markdown_file(self, file_path: str) -> str:
        # Prefer MarkItDown rendering for markdown if available
        try:
            result = self.md.convert(file_path)
            text = result.text_content
            if text and len(text.strip()) > 0:
                return text
        except Exception:
            logger.debug("MarkItDown failed for %s, falling back to plain read", file_path)

        return self.load_text_file(file_path)

    def load_docx_file(self, file_path: str) -> str:
        if DocxDocument is None:
            raise RuntimeError("python-docx is not installed; cannot load .docx files")

        doc = DocxDocument(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n\n".join(paragraphs)

    def load_pdf_file(self, file_path: str) -> str:
        if self.pdf_loader is not None:
            # Delegate to the robust loader
            return self.pdf_loader.load_pdf_from_file(file_path)
        else:
            raise RuntimeError("PDF support requires document_loader.DocumentLoader to be importable")

    def load_file(self, file_path: str) -> dict:
        """Detect file type and load content. Returns a dict with filename, content and path."""
        ext = Path(file_path).suffix.lower()
        filename = Path(file_path).name

        try:
            if ext == ".pdf":
                content = self.load_pdf_file(file_path)
                ftype = "pdf"
            elif ext in (".md", ".markdown"):
                content = self.load_markdown_file(file_path)
                ftype = "markdown"
            elif ext == ".txt":
                content = self.load_text_file(file_path)
                ftype = "text"
            elif ext == ".docx":
                content = self.load_docx_file(file_path)
                ftype = "docx"
            else:
                # Unknown extension: attempt plain text read
                content = self.load_text_file(file_path)
                ftype = "text"

            return {"filename": filename, "content": content, "path": str(file_path), "type": ftype}

        except Exception as e:
            logger.error("Failed to load %s: %s", filename, e)
            raise

    def load_all_documents(self) -> List[dict]:
        """Load all supported documents from DOCUMENTS_DIR."""
        documents = []

        if not DOCUMENTS_DIR.exists():
            logger.warning(f"Documents directory not found: {DOCUMENTS_DIR}")
            return documents

        # Supported patterns
        patterns = ["*.pdf", "*.md", "*.markdown", "*.txt", "*.docx"]
        for pat in patterns:
            for file_path in DOCUMENTS_DIR.glob(pat):
                try:
                    doc = self.load_file(str(file_path))
                    documents.append(doc)
                    logger.info("Loaded document: %s", file_path.name)
                except Exception as e:
                    logger.error("Failed to load %s: %s", file_path.name, e)
                    continue

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def chunk_documents(self, documents: List[dict]) -> List[dict]:
        """Split documents into chunks and attach metadata about source type."""
        chunks = []

        for doc in documents:
            split_texts = self.splitter.split_text(doc["content"])
            for i, chunk_text in enumerate(split_texts):
                chunks.append({
                    "filename": doc["filename"],
                    "chunk_id": i,
                    "content": chunk_text,
                    "source": doc["path"],
                    "type": doc.get("type", "text")
                })

        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
