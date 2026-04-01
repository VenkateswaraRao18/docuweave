from __future__ import annotations

from typing import List, Dict

# from models import Block, BlockType, Section
from docuweave.models import Block,BlockType,Section


# ============================================================
# Strict Layout Hierarchy Builder (Depth-Limited)
# ============================================================

def build_hierarchy(blocks: List[Block]) -> List[Section]:
    """
    Convert flat blocks into hierarchical Sections.
    Strict layout-based hierarchy with max 3 levels.
    """

    if not blocks:
        return []

    # --------------------------------------------------------
    # Collect heading font sizes only
    # --------------------------------------------------------
    heading_blocks = [
        b for b in blocks
        if b.type in (BlockType.TITLE, BlockType.HEADING)
        and b.font_size is not None
    ]

    unique_sizes = sorted(
        {b.font_size for b in heading_blocks},
        reverse=True
    )

    # Limit to top 3 sizes only
    unique_sizes = unique_sizes[:3]

    # Map font size → hierarchy level (max depth 2)
    font_to_level: Dict[float, int] = {
        size: level for level, size in enumerate(unique_sizes)
    }

    max_level = 2

    root_sections: List[Section] = []
    section_stack: List[Section] = []
    intro_blocks: List[Block] = []

    # --------------------------------------------------------
    # Build hierarchy
    # --------------------------------------------------------
    for block in blocks:

        # ----------------------------
        # If heading/title → new section
        # ----------------------------
        if block.type in (BlockType.TITLE, BlockType.HEADING):
            if block.font_size is None:
                continue

            level = font_to_level.get(block.font_size, max_level)

            # Clamp level
            level = min(level, max_level)

            new_section = Section(
                id=block.id,
                title=block.text,
                level=level,
            )

            # Adjust stack
            while section_stack and section_stack[-1].level >= level:
                section_stack.pop()

            if section_stack:
                section_stack[-1].add_subsection(new_section)
            else:
                root_sections.append(new_section)

            section_stack.append(new_section)

        # ----------------------------
        # Otherwise attach paragraph
        # ----------------------------
        else:
            if section_stack:
                section_stack[-1].add_block(block)
            else:
                intro_blocks.append(block)

    if intro_blocks:
        intro_section = Section(
            id="section_intro",
            title="Introduction",
            level=0,
            blocks=intro_blocks,
        )
        root_sections.insert(0, intro_section)

    if not root_sections and blocks:
        fallback_section = Section(
            id="section_document",
            title="Document",
            level=0,
            blocks=[b for b in blocks if b.type not in (BlockType.TITLE, BlockType.HEADING)],
        )
        return [fallback_section]

    return root_sections