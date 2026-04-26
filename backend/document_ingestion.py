# Backwards-compatible shim.
# Keep this file so older imports referencing `document_ingestion` continue to work.

from .document_ingestor import DocumentIngestor

__all__ = ["DocumentIngestor"]
