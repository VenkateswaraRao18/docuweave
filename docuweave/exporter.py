from __future__ import annotations

from typing import List, Dict, Any

# from models import Document, Section
from docuweave.models import Block,Document,Section


# ============================================================
# Recursive Section Export
# ============================================================

def export_section(section: Section) -> Dict[str, Any]:
    return {
        "id": section.id,
        "title": section.title,
        "level": section.level,
        "blocks": [
            {
                "id": block.id,
                "type": block.type,
                "text": block.text,
                "page": block.page,
            }
            for block in section.blocks
        ],
        "subsections": [
            export_section(subsection)
            for subsection in section.subsections
        ],
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