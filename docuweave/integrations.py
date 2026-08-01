from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


def _chunk_to_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": chunk.get("id"),
        "tokens": chunk.get("tokens"),
        "section_title": chunk.get("section_title"),
        "section_path": chunk.get("section_path"),
        "section_level": chunk.get("section_level"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "previous_chunk_id": chunk.get("previous_chunk_id"),
        "next_chunk_id": chunk.get("next_chunk_id"),
    }


# ─────────────────────────────────────────────────────────────
# LangChain — Document converter
# ─────────────────────────────────────────────────────────────

def to_langchain_documents(chunks: List[Dict[str, Any]]) -> List[Any]:
    """Convert DocuWeave chunks into LangChain Document objects."""
    try:
        from langchain_core.documents import Document as LC
    except ImportError:
        return [{"page_content": c.get("text", ""), "metadata": _chunk_to_metadata(c)} for c in chunks]
    return [LC(page_content=c.get("text", ""), metadata=_chunk_to_metadata(c)) for c in chunks]


# ─────────────────────────────────────────────────────────────
# LangChain — BaseLoader  (drop-in for any LangChain loader)
# ─────────────────────────────────────────────────────────────

class DocuWeaveLoader:
    """
    LangChain-compatible document loader for DocuWeave.

    Use it anywhere LangChain expects a ``BaseLoader``::

        from docuweave.integrations import DocuWeaveLoader

        loader = DocuWeaveLoader("paper.pdf", max_tokens=512)
        docs   = loader.load()                     # List[LangChain Document]

        # Stream one chunk at a time
        for doc in loader.lazy_load():
            print(doc.page_content[:80])

    Works with LangChain's ``RecursiveCharacterTextSplitter``,
    ``VectorstoreIndexCreator``, and every pipeline that accepts loaders.
    """

    def __init__(self, file_path: str, max_tokens: int = 512, model_name: str = "gpt-4"):
        self.file_path  = file_path
        self.max_tokens = max_tokens
        self.model_name = model_name

    # Try to inherit from BaseLoader if langchain_core is present.
    # We patch the MRO at class-definition time so the class is always
    # importable even without langchain installed.
    try:
        from langchain_core.document_loaders import BaseLoader as _LC_Base  # type: ignore
        _lc_base = _LC_Base
    except ImportError:
        _lc_base = object  # type: ignore

    def load(self) -> List[Any]:
        from docuweave.api import parse
        doc = parse(self.file_path)
        chunks = doc.to_chunks(max_tokens=self.max_tokens, model_name=self.model_name)
        return to_langchain_documents(chunks)

    def lazy_load(self) -> Iterator[Any]:
        for doc in self.load():
            yield doc


# ─────────────────────────────────────────────────────────────
# LlamaIndex — TextNode converter
# ─────────────────────────────────────────────────────────────

def to_llamaindex_nodes(chunks: List[Dict[str, Any]]) -> List[Any]:
    """Convert DocuWeave chunks into LlamaIndex TextNode objects."""
    try:
        from llama_index.core.schema import TextNode
    except ImportError:
        return [{"text": c.get("text", ""), "metadata": _chunk_to_metadata(c)} for c in chunks]
    return [
        TextNode(text=c.get("text", ""), metadata=_chunk_to_metadata(c), id_=c.get("id"))
        for c in chunks
    ]


# ─────────────────────────────────────────────────────────────
# LlamaIndex — Reader  (drop-in for any LlamaIndex reader)
# ─────────────────────────────────────────────────────────────

class DocuWeaveReader:
    """
    LlamaIndex-compatible file reader for DocuWeave.

    Use it with LlamaIndex's ``SimpleDirectoryReader`` or directly::

        from docuweave.integrations import DocuWeaveReader

        reader = DocuWeaveReader(max_tokens=512)
        nodes  = reader.load_data("paper.pdf")

        # Or pass to SimpleDirectoryReader
        from llama_index.core import SimpleDirectoryReader
        docs = SimpleDirectoryReader(
            input_files=["paper.pdf"],
            file_extractor={".pdf": DocuWeaveReader()},
        ).load_data()
    """

    def __init__(self, max_tokens: int = 512, model_name: str = "gpt-4"):
        self.max_tokens = max_tokens
        self.model_name = model_name

    def load_data(self, file: str, **kwargs: Any) -> List[Any]:
        from docuweave.api import parse
        doc = parse(str(file))
        chunks = doc.to_chunks(max_tokens=self.max_tokens, model_name=self.model_name)
        return to_llamaindex_nodes(chunks)


# ─────────────────────────────────────────────────────────────
# Haystack — Document converter
# ─────────────────────────────────────────────────────────────

def to_haystack_documents(chunks: List[Dict[str, Any]]) -> List[Any]:
    """
    Convert DocuWeave chunks into Haystack ``Document`` objects.

    Example::

        from docuweave import parse
        from docuweave.integrations import to_haystack_documents
        from haystack import Pipeline
        from haystack.document_stores.in_memory import InMemoryDocumentStore

        store = InMemoryDocumentStore()
        doc   = parse("paper.pdf")
        haystack_docs = to_haystack_documents(doc.to_chunks())
        store.write_documents(haystack_docs)
    """
    try:
        from haystack.dataclasses import Document as HaystackDoc
    except ImportError:
        return [{"content": c.get("text", ""), "meta": _chunk_to_metadata(c)} for c in chunks]

    return [
        HaystackDoc(content=c.get("text", ""), meta=_chunk_to_metadata(c))
        for c in chunks
    ]
