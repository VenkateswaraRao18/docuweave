from __future__ import annotations

from typing import List, Dict, Any

# from models import Document, Section
from docuweave.models import Block,Document,Section


# ============================================================
# Recursive Section Export
# ============================================================

def _export_block(block: Block) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": block.id,
        "type": block.type,
        "page": block.page,
    }
    if block.text is not None:
        out["text"] = block.text
    if block.items is not None:
        out["items"] = block.items
    if block.font_size is not None:
        out["font_size"] = block.font_size
    return out


def export_section(section: Section) -> Dict[str, Any]:
    return {
        "id": section.id,
        "title": section.title,
        "level": section.level,
        "blocks": [_export_block(b) for b in section.blocks],
        "subsections": [export_section(s) for s in section.subsections],
    }


# ============================================================
# Full Document Export
# ============================================================

def export_document(
    document: Document,
    chunks: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:

    return {
        "metadata": document.metadata.model_dump(),
        "sections": [
            export_section(section)
            for section in document.sections
        ],
        "chunks": chunks or [],
    }