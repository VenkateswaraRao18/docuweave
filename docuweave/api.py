from __future__ import annotations

import json
from typing import List, Dict, Any, Optional

from docuweave.models import Document, DocumentMetadata
from docuweave.parser import parse_pdf
from docuweave.hierarchy import build_hierarchy
from docuweave.chunking import build_chunks
from docuweave.exporter import export_document
from docuweave.integrations import to_langchain_documents, to_llamaindex_nodes
from docuweave.vector_exporters import (
    build_vector_records,
    export_faiss_jsonl,
    export_pinecone,
    export_weaviate,
)


class DocuWeaveDocument:
    def __init__(self, file_path: str):
        self.file_path = file_path

        # Step 1: Parse flat blocks
        self.blocks = parse_pdf(file_path)

        # Step 2: Build hierarchy
        self.sections = build_hierarchy(self.blocks)

        # Internal document model
        self.document = Document(
            metadata=DocumentMetadata(
                source=file_path,
                total_pages=max((b.page for b in self.blocks), default=0),
            ),
            blocks=self.blocks,
            sections=self.sections,
        )

        self._chunks: Optional[List[Dict[str, Any]]] = None
        self._chunk_config: Optional[tuple[int, str]] = None

    # --------------------------------------------------------
    # Chunk Generation
    # --------------------------------------------------------

    def to_chunks(
        self,
        max_tokens: int = 800,
        model_name: str = "gpt-4",
    ) -> List[Dict[str, Any]]:
        requested_config = (max_tokens, model_name)
        if self._chunks is None or self._chunk_config != requested_config:
            self._chunks = build_chunks(
                self.sections,
                max_tokens=max_tokens,
                model_name=model_name,
            )
            self._chunk_config = requested_config
        return self._chunks

    # --------------------------------------------------------
    # JSON Export
    # --------------------------------------------------------

    def to_json(self) -> Dict[str, Any]:
        return export_document(
            self.document,
            chunks=self._chunks or [],
        )

    def save_json(self, output_path: str) -> None:
        """
        Save structured document JSON to file.
        """
        data = self.to_json()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # Accessors
    # --------------------------------------------------------

    def get_sections(self):
        return self.sections

    def get_blocks(self):
        return self.blocks

    # --------------------------------------------------------
    # Integrations
    # --------------------------------------------------------

    def to_langchain(self, max_tokens: int = 800, model_name: str = "gpt-4"):
        chunks = self.to_chunks(max_tokens=max_tokens, model_name=model_name)
        return to_langchain_documents(chunks)

    def to_llamaindex(self, max_tokens: int = 800, model_name: str = "gpt-4"):
        chunks = self.to_chunks(max_tokens=max_tokens, model_name=model_name)
        return to_llamaindex_nodes(chunks)

    # --------------------------------------------------------
    # Vector DB Exporters
    # --------------------------------------------------------

    def to_vector_records(
        self,
        embeddings: Optional[List[List[float]]] = None,
        max_tokens: int = 800,
        model_name: str = "gpt-4",
    ) -> List[Dict[str, Any]]:
        chunks = self.to_chunks(max_tokens=max_tokens, model_name=model_name)
        return build_vector_records(chunks, embeddings=embeddings)

    def export_pinecone(
        self,
        embeddings: Optional[List[List[float]]] = None,
        max_tokens: int = 800,
        model_name: str = "gpt-4",
    ) -> List[Dict[str, Any]]:
        chunks = self.to_chunks(max_tokens=max_tokens, model_name=model_name)
        return export_pinecone(chunks, embeddings=embeddings)

    def export_weaviate(
        self,
        embeddings: Optional[List[List[float]]] = None,
        max_tokens: int = 800,
        model_name: str = "gpt-4",
    ) -> List[Dict[str, Any]]:
        chunks = self.to_chunks(max_tokens=max_tokens, model_name=model_name)
        return export_weaviate(chunks, embeddings=embeddings)

    def export_faiss_jsonl(
        self,
        output_path: str,
        embeddings: Optional[List[List[float]]] = None,
        max_tokens: int = 800,
        model_name: str = "gpt-4",
    ) -> str:
        chunks = self.to_chunks(max_tokens=max_tokens, model_name=model_name)
        return export_faiss_jsonl(chunks, output_path=output_path, embeddings=embeddings)


def parse(file_path: str) -> DocuWeaveDocument:
    return DocuWeaveDocument(file_path)