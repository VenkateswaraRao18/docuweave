from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ============================================================
# Block Types
# ============================================================

class BlockType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    HEADER = "header"
    FOOTER = "footer"


# ============================================================
# Layout Block (Atomic Unit from Parser)
# ============================================================

class Block(BaseModel):
    """
    Represents a single layout-aware text unit extracted from a document.
    This is the smallest meaningful piece of structured content.
    """

    id: str
    type: BlockType
    text: str

    page: int

    # Bounding box: [x0, y0, x1, y1]
    bbox: Optional[List[float]] = None

    font_size: Optional[float] = None
    font_name: Optional[str] = None

    class Config:
        frozen = False
        use_enum_values = True


# ============================================================
# Hierarchical Section Model
# ============================================================

class Section(BaseModel):
    """
    Represents a hierarchical section in the document.
    Built from flat blocks using heading detection.
    """

    id: str
    title: Optional[str]
    level: int

    blocks: List[Block] = Field(default_factory=list)
    subsections: List[Section] = Field(default_factory=list)

    def add_block(self, block: Block) -> None:
        self.blocks.append(block)

    def add_subsection(self, section: Section) -> None:
        self.subsections.append(section)

    def all_blocks_recursive(self) -> List[Block]:
        """
        Returns all blocks in this section including subsections.
        Useful for chunking.
        """
        collected = list(self.blocks)
        for subsection in self.subsections:
            collected.extend(subsection.all_blocks_recursive())
        return collected


Section.model_rebuild()


# ============================================================
# Document Metadata
# ============================================================

class DocumentMetadata(BaseModel):
    source: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[str] = None
    total_pages: Optional[int] = None


# ============================================================
# Top-Level Document Model
# ============================================================

class Document(BaseModel):
    """
    Root container for the parsed document.
    Holds both flat blocks and hierarchical sections.
    """

    metadata: DocumentMetadata
    blocks: List[Block]

    # Built after hierarchy detection
    sections: List[Section] = Field(default_factory=list)

    # -------------------------------
    # Convenience Query Methods
    # -------------------------------

    def get_sections(self) -> List[Section]:
        return self.sections

    def get_all_blocks(self) -> List[Block]:
        return self.blocks

    def find_blocks_by_type(self, block_type: BlockType) -> List[Block]:
        return [b for b in self.blocks if b.type == block_type]

    def to_dict(self) -> dict:
        """
        Export full structured JSON.
        """
        return self.model_dump()

    def to_json(self, **kwargs) -> str:
        """
        Export structured JSON string.
        """
        return self.model_dump_json(**kwargs)