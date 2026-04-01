from __future__ import annotations

from typing import Any, Dict, List


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


def to_langchain_documents(chunks: List[Dict[str, Any]]) -> List[Any]:
    """
    Convert DocuWeave chunks into LangChain Document objects.
    Falls back to dict payloads if langchain is not installed.
    """
    try:
        from langchain_core.documents import Document as LangChainDocument
    except ImportError:
        return [
            {
                "page_content": c.get("text", ""),
                "metadata": _chunk_to_metadata(c),
            }
            for c in chunks
        ]

    return [
        LangChainDocument(
            page_content=c.get("text", ""),
            metadata=_chunk_to_metadata(c),
        )
        for c in chunks
    ]


def to_llamaindex_nodes(chunks: List[Dict[str, Any]]) -> List[Any]:
    """
    Convert DocuWeave chunks into LlamaIndex TextNode objects.
    Falls back to dict payloads if llama-index is not installed.
    """
    try:
        from llama_index.core.schema import TextNode
    except ImportError:
        return [
            {
                "text": c.get("text", ""),
                "metadata": _chunk_to_metadata(c),
            }
            for c in chunks
        ]

    return [
        TextNode(
            text=c.get("text", ""),
            metadata=_chunk_to_metadata(c),
            id_=c.get("id"),
        )
        for c in chunks
    ]
