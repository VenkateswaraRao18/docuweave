from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _record(chunk: Dict[str, Any], embedding: Optional[List[float]]) -> Dict[str, Any]:
    return {
        "id": chunk.get("id"),
        "text": chunk.get("text"),
        "values": embedding,
        "metadata": {
            "tokens": chunk.get("tokens"),
            "section_title": chunk.get("section_title"),
            "section_path": chunk.get("section_path"),
            "section_level": chunk.get("section_level"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "previous_chunk_id": chunk.get("previous_chunk_id"),
            "next_chunk_id": chunk.get("next_chunk_id"),
        },
    }


def build_vector_records(
    chunks: List[Dict[str, Any]],
    embeddings: Optional[List[List[float]]] = None,
) -> List[Dict[str, Any]]:
    if embeddings is not None and len(embeddings) != len(chunks):
        raise ValueError("embeddings length must match chunks length")
    return [
        _record(chunk, embeddings[idx] if embeddings is not None else None)
        for idx, chunk in enumerate(chunks)
    ]


def export_pinecone(
    chunks: List[Dict[str, Any]],
    embeddings: Optional[List[List[float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build Pinecone upsert-ready records.
    """
    records = build_vector_records(chunks, embeddings=embeddings)
    return [{"id": r["id"], "values": r["values"], "metadata": r["metadata"]} for r in records]


def export_weaviate(
    chunks: List[Dict[str, Any]],
    embeddings: Optional[List[List[float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Build Weaviate import-ready records.
    """
    records = build_vector_records(chunks, embeddings=embeddings)
    payload = []
    for r in records:
        payload.append(
            {
                "id": r["id"],
                "properties": {
                    "text": r["text"],
                    **r["metadata"],
                },
                "vector": r["values"],
            }
        )
    return payload


def export_faiss_jsonl(
    chunks: List[Dict[str, Any]],
    output_path: str,
    embeddings: Optional[List[List[float]]] = None,
) -> str:
    """
    Export records to JSONL for FAISS and local vector workflows.
    """
    records = build_vector_records(chunks, embeddings=embeddings)
    target = Path(output_path)
    with target.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(target)
