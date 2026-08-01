from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

from docuweave.models import Block, BlockType, Document, DocumentMetadata
from docuweave.parser import parse_pdf
from docuweave.hierarchy import build_hierarchy
from docuweave.chunking import build_chunks
from docuweave.exporter import export_document
from docuweave.integrations import (
    to_langchain_documents,
    to_llamaindex_nodes,
    to_haystack_documents,
)
from docuweave.vector_exporters import (
    build_vector_records,
    export_chroma,
    export_faiss_jsonl,
    export_milvus,
    export_pinecone,
    export_qdrant,
    export_weaviate,
)


# ─────────────────────────────────────────────────────────────
# Hierarchy Confidence Score
# ─────────────────────────────────────────────────────────────

def _compute_confidence(blocks: List[Block], sections: List) -> float:
    """
    Return a [0, 1] score indicating how well DocuWeave detected hierarchy.

    1.0 = rich, multi-level structure with clearly detected headings.
    0.0 = all text fell into a single flat fallback section.
    """
    if not blocks:
        return 0.0

    # Fallback: single "Document" section with no hierarchy
    if len(sections) == 1 and getattr(sections[0], "id", "") == "section_document":
        return 0.0

    heading_count = sum(
        1 for b in blocks if b.type in (BlockType.HEADING, BlockType.TITLE)
    )
    heading_ratio = heading_count / len(blocks)

    # Reasonable docs have 5-20% headings; reward multi-level sections
    depth_bonus = 0.2 if any(
        getattr(s, "subsections", []) for s in sections
    ) else 0.0

    # Clamp to [0, 1]
    score = min(1.0, heading_ratio * 5 + depth_bonus)
    return round(score, 3)


# ─────────────────────────────────────────────────────────────
# DocuWeaveDocument
# ─────────────────────────────────────────────────────────────

class DocuWeaveDocument:
    """
    Parsed representation of a single PDF.

    Typical usage::

        from docuweave import parse

        doc    = parse("paper.pdf")
        chunks = doc.to_chunks(max_tokens=512)

        # LangChain
        lc_docs = doc.to_langchain()

        # ChromaDB
        import chromadb
        col = chromadb.Client().get_or_create_collection("papers")
        doc.to_chroma(col)          # Chroma auto-embeds

        # Qdrant
        from qdrant_client import QdrantClient
        qc   = QdrantClient(":memory:")
        vecs = embed(chunks)
        doc.to_qdrant(qc, "papers", embeddings=vecs)

    Attributes:
        file_path          -- path to source PDF
        blocks             -- flat list of layout-aware Block objects
        sections           -- hierarchical Section tree
        hierarchy_confidence -- [0, 1] quality score for hierarchy detection
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        try:
            self.blocks = parse_pdf(file_path)
        except Exception as exc:
            raise ValueError(
                f"DocuWeave could not parse '{file_path}'. "
                "Check that the file exists, is a valid PDF, and is not password-protected. "
                f"Underlying error: {exc}"
            ) from exc
        self.sections = build_hierarchy(self.blocks)
        self.hierarchy_confidence: float = _compute_confidence(self.blocks, self.sections)

        self.document = Document(
            metadata=DocumentMetadata(
                source=file_path,
                total_pages=max((b.page for b in self.blocks), default=0),
            ),
            blocks=self.blocks,
            sections=self.sections,
        )

        self._chunks: Optional[List[Dict[str, Any]]] = None
        self._chunk_config: Optional[tuple] = None

    # ── Chunks ───────────────────────────────────────────────

    def to_chunks(
        self,
        max_tokens: int = 512,
        model_name: str = "gpt-4",
    ) -> List[Dict[str, Any]]:
        """Return layout-aware chunks with full metadata."""
        key = (max_tokens, model_name)
        if self._chunks is None or self._chunk_config != key:
            self._chunks = build_chunks(
                self.sections, max_tokens=max_tokens, model_name=model_name
            )
            self._chunk_config = key
        return self._chunks

    def iter_chunks(
        self,
        max_tokens: int = 512,
        model_name: str = "gpt-4",
    ) -> Iterator[Dict[str, Any]]:
        """Yield chunks one at a time — useful for large documents."""
        for chunk in self.to_chunks(max_tokens=max_tokens, model_name=model_name):
            yield chunk

    # ── JSON ─────────────────────────────────────────────────

    def to_json(self) -> Dict[str, Any]:
        payload = export_document(self.document, chunks=self._chunks or [])
        payload["hierarchy_confidence"] = self.hierarchy_confidence
        return payload

    def save_json(self, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2, ensure_ascii=False)

    # ── Accessors ─────────────────────────────────────────────

    def get_sections(self):
        return self.sections

    def get_blocks(self):
        return self.blocks

    # ── Framework integrations ────────────────────────────────

    def to_langchain(self, max_tokens: int = 512, model_name: str = "gpt-4") -> List[Any]:
        """Return LangChain Document objects."""
        return to_langchain_documents(self.to_chunks(max_tokens, model_name))

    def to_llamaindex(self, max_tokens: int = 512, model_name: str = "gpt-4") -> List[Any]:
        """Return LlamaIndex TextNode objects."""
        return to_llamaindex_nodes(self.to_chunks(max_tokens, model_name))

    def to_haystack(self, max_tokens: int = 512, model_name: str = "gpt-4") -> List[Any]:
        """Return Haystack Document objects."""
        return to_haystack_documents(self.to_chunks(max_tokens, model_name))

    # ── Vector DB exporters ───────────────────────────────────

    def to_vector_records(
        self,
        embeddings: Optional[List[List[float]]] = None,
        max_tokens: int = 512,
        model_name: str = "gpt-4",
    ) -> List[Dict[str, Any]]:
        return build_vector_records(self.to_chunks(max_tokens, model_name), embeddings=embeddings)

    def to_pinecone(
        self,
        embeddings: Optional[List[List[float]]] = None,
        max_tokens: int = 512,
        model_name: str = "gpt-4",
    ) -> List[Dict[str, Any]]:
        """Return Pinecone upsert-ready dicts."""
        return export_pinecone(self.to_chunks(max_tokens, model_name), embeddings=embeddings)

    # keep old name as alias
    export_pinecone = to_pinecone  # type: ignore[assignment]

    def to_weaviate(
        self,
        embeddings: Optional[List[List[float]]] = None,
        max_tokens: int = 512,
        model_name: str = "gpt-4",
    ) -> List[Dict[str, Any]]:
        """Return Weaviate v4 import-ready dicts."""
        return export_weaviate(self.to_chunks(max_tokens, model_name), embeddings=embeddings)

    export_weaviate = to_weaviate  # type: ignore[assignment]

    def to_chroma(
        self,
        collection: Any,
        embeddings: Optional[List[List[float]]] = None,
        max_tokens: int = 512,
        model_name: str = "gpt-4",
    ) -> None:
        """Upsert chunks directly into a ChromaDB collection."""
        export_chroma(self.to_chunks(max_tokens, model_name), collection, embeddings=embeddings)

    def to_qdrant(
        self,
        client: Any,
        collection_name: str,
        embeddings: List[List[float]],
        max_tokens: int = 512,
        model_name: str = "gpt-4",
    ) -> None:
        """Upsert chunks into a Qdrant collection (embeddings required)."""
        export_qdrant(self.to_chunks(max_tokens, model_name), client, collection_name, embeddings)

    def to_milvus(
        self,
        collection: Any,
        embeddings: List[List[float]],
        max_tokens: int = 512,
        model_name: str = "gpt-4",
    ) -> None:
        """Insert chunks into a Milvus collection (embeddings required)."""
        export_milvus(self.to_chunks(max_tokens, model_name), collection, embeddings)

    def to_faiss_jsonl(
        self,
        output_path: str,
        embeddings: Optional[List[List[float]]] = None,
        max_tokens: int = 512,
        model_name: str = "gpt-4",
    ) -> str:
        """Write JSONL file for FAISS workflows."""
        return export_faiss_jsonl(
            self.to_chunks(max_tokens, model_name), output_path, embeddings=embeddings
        )

    export_faiss_jsonl = to_faiss_jsonl  # type: ignore[assignment]

    @property
    def num_pages(self) -> int:
        """Total number of pages in the source PDF."""
        return self.document.metadata.total_pages or 0

    def __len__(self) -> int:
        """Number of chunks (after to_chunks() has been called; 0 if not yet called)."""
        return len(self._chunks) if self._chunks else 0

    def __repr__(self) -> str:
        n_chunks = len(self._chunks) if self._chunks else "?"
        return (
            f"DocuWeaveDocument(file='{Path(self.file_path).name}', "
            f"pages={self.num_pages}, sections={len(self.sections)}, "
            f"chunks={n_chunks}, confidence={self.hierarchy_confidence:.2f})"
        )


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def parse(file_path: str) -> DocuWeaveDocument:
    """Parse a single PDF into a DocuWeaveDocument."""
    return DocuWeaveDocument(file_path)


def parse_directory(
    directory: str,
    pattern: str = "**/*.pdf",
    on_error: str = "warn",
    min_confidence: float = 0.0,
    progress: bool = True,
) -> List[DocuWeaveDocument]:
    """
    Parse every PDF matching *pattern* under *directory*.

    Args:
        directory:       Root directory to scan.
        pattern:         Glob pattern (default ``**/*.pdf``).
        on_error:        ``"warn"`` (default) — skip and warn on failure;
                         ``"raise"`` — re-raise; ``"skip"`` — silently skip.
        min_confidence:  Discard documents whose ``hierarchy_confidence`` is
                         below this threshold (0.0 = keep all). Useful to
                         filter out scanned / image-only PDFs automatically.
        progress:        Print a progress line to stdout.

    Returns:
        List of :class:`DocuWeaveDocument` objects, one per successfully
        parsed PDF that meets the *min_confidence* threshold.

    Example::

        from docuweave import parse_directory

        # Parse all PDFs; skip any with poor hierarchy detection
        docs = parse_directory("./papers/", min_confidence=0.1)
        print(f"Got {len(docs)} usable documents")
        for doc in docs:
            chunks = doc.to_chunks(max_tokens=512)
    """
    root  = Path(directory)
    paths = sorted(root.glob(pattern))
    total = len(paths)

    if total == 0:
        warnings.warn(f"No PDFs found matching '{pattern}' in '{directory}'", UserWarning)
        return []

    results: List[DocuWeaveDocument] = []
    skipped_confidence = 0

    for idx, path in enumerate(paths, 1):
        if progress:
            print(f"\r[{idx}/{total}] {path.name[:55]:<55}", end="", flush=True)
        try:
            doc = DocuWeaveDocument(str(path))
            if doc.hierarchy_confidence < min_confidence:
                skipped_confidence += 1
                continue
            results.append(doc)
        except Exception as exc:
            if on_error == "raise":
                raise
            msg = f"DocuWeave: failed to parse '{path}': {exc}"
            if on_error == "warn":
                warnings.warn(msg, UserWarning)

    if progress:
        suffix = f" ({skipped_confidence} below confidence threshold)" if skipped_confidence else ""
        print(f"\r✓ Parsed {len(results)}/{total} PDFs from '{directory}'{suffix}")

    return results