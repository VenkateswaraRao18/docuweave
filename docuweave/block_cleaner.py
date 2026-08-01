from __future__ import annotations

import re
import uuid
from typing import List

from .models import Block, BlockType


# ─────────────────────────────────────────────────────────────
# Bullet continuation merging
# ─────────────────────────────────────────────────────────────

def merge_bullet_continuations(blocks: List[Block]) -> List[Block]:
    """
    Merge a paragraph that immediately follows a list_item into that item.
    Handles the common PDF artefact where a wrapped bullet line is emitted
    as a separate PARAGRAPH block.
    """
    cleaned: List[Block] = []
    for block in blocks:
        prev = cleaned[-1] if cleaned else None
        if (
            prev is not None
            and prev.type == "list_item"
            and block.type == "paragraph"
            and block.text  # guard against None
        ):
            prev.text = (prev.text or "") + " " + block.text
        else:
            cleaned.append(block)
    return cleaned


# ─────────────────────────────────────────────────────────────
# Bullet normalisation
# ─────────────────────────────────────────────────────────────

_BULLET_CHARS = r"[•▪‣◦\*]"


def normalize_bullets(blocks: List[Block]) -> List[Block]:
    """
    Strip duplicate/leading bullet characters and split compound bullets
    (a single span containing multiple embedded bullet chars) into
    individual list_item blocks — each with a fresh unique ID.
    """
    new_blocks: List[Block] = []
    for block in blocks:
        if block.type != "list_item":
            new_blocks.append(block)
            continue

        text = (block.text or "").strip()

        # Remove any leading bullet prefix
        text = re.sub(rf"^(?:{_BULLET_CHARS}\s*)+", "", text)

        # Split on embedded bullet chars (compound bullets)
        parts = re.split(rf"\s*{_BULLET_CHARS}\s*", text)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) <= 1:
            block.text = "• " + (parts[0] if parts else text)
            new_blocks.append(block)
        else:
            # Each sub-bullet gets a UNIQUE ID — never share the parent ID
            for part in parts:
                new_blocks.append(Block(
                    id=str(uuid.uuid4()),
                    type="list_item",
                    text="• " + part,
                    page=block.page,
                    font_size=block.font_size,
                ))
    return new_blocks


# ─────────────────────────────────────────────────────────────
# List grouping
# ─────────────────────────────────────────────────────────────

def group_lists(blocks: List[Block]) -> List[Block]:
    """
    Collapse runs of consecutive list_item blocks into a single LIST block.
    """
    result: List[Block] = []
    i = 0
    while i < len(blocks):
        block = blocks[i]
        if block.type == "list_item":
            items: List[str] = []
            start_page = block.page
            while i < len(blocks) and blocks[i].type == "list_item":
                raw = (blocks[i].text or "").strip()
                raw = re.sub(r"^[•▪‣◦*]+\s*", "", raw)
                items.append(raw)
                i += 1
            result.append(Block(
                id=str(uuid.uuid4()),   # unique ID, not positional
                type=BlockType.LIST,
                items=items,
                page=start_page,
            ))
        else:
            result.append(block)
            i += 1
    return result


# ─────────────────────────────────────────────────────────────
# Pipe-separator heading detection
# ─────────────────────────────────────────────────────────────

_PIPE_HEADING_RE = re.compile(r"^[A-Z][^|]{2,60}\|[^|]{2,60}$")


def detect_project_titles(blocks: List[Block]) -> List[Block]:
    """
    Promote certain paragraph blocks to headings when they match
    the "Title || Subtitle" pattern common in resumes / project lists.

    Deliberately conservative — only matches "X | Y" where both sides
    are short, capitalised, and contain no embedded pipes (avoids hitting
    markdown tables, code fences, or data rows).
    """
    for b in blocks:
        text = b.text or ""
        if b.type == "paragraph" and _PIPE_HEADING_RE.match(text):
            b.type = "heading"
    return blocks


# ─────────────────────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────────────────────

def clean_blocks(blocks: List[Block]) -> List[Block]:
    blocks = merge_bullet_continuations(blocks)
    blocks = normalize_bullets(blocks)
    blocks = detect_project_titles(blocks)
    blocks = group_lists(blocks)
    return blocks
