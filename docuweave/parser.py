from __future__ import annotations

import fitz
import uuid
import re
from typing import List

# from models import Block, BlockType
from docuweave.models import Block,BlockType


# ============================================================
# Utility Functions
# ============================================================

def is_mostly_upper(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False

    uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return uppercase_ratio > 0.7


def is_noise_line(text: str) -> bool:
    stripped = text.strip()

    if len(stripped) < 3:
        return True

    if "...." in stripped:
        return True

    letters = [c for c in stripped if c.isalpha()]
    if len(letters) == 0:
        return True

    return False


def looks_like_sentence(text: str) -> bool:
    """
    Detect if text looks like normal sentence content.
    """
    if text.endswith((".", ",", ";", ":", "-")):
        return True

    # If contains many lowercase words, likely paragraph
    words = text.split()
    lowercase_words = [w for w in words if w.islower()]
    if len(lowercase_words) > 3:
        return True

    return False


def contains_too_many_symbols(text: str) -> bool:
    symbols = re.findall(r"[^\w\s]", text)
    return len(symbols) > 5


# ============================================================
# Main Parser
# ============================================================

def parse_pdf(file_path: str) -> List[Block]:

    doc = fitz.open(file_path)
    blocks: List[Block] = []
    font_sizes: List[float] = []

    # -------------------------------
    # PASS 1: Collect font sizes
    # -------------------------------
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_dict = page.get_text("dict")

        for block in page_dict.get("blocks", []):
            if "lines" not in block:
                continue

            for line in block["lines"]:
                for span in line.get("spans", []):
                    size = span.get("size")
                    if size:
                        font_sizes.append(size)

    if not font_sizes:
        doc.close()
        return []

    unique_sizes = sorted(set(font_sizes), reverse=True)
    title_size = unique_sizes[0]
    top_sizes = unique_sizes[:4]

    # -------------------------------
    # PASS 2: Build Blocks
    # -------------------------------
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_dict = page.get_text("dict")

        for block in page_dict.get("blocks", []):
            if "lines" not in block:
                continue

            for line in block["lines"]:
                line_text = ""
                font_size = None
                font_name = None

                for span in line.get("spans", []):
                    line_text += span.get("text", "")

                    if font_size is None:
                        font_size = span.get("size")

                    if font_name is None:
                        font_name = span.get("font")

                line_text = line_text.strip()

                if not line_text:
                    continue

                # ---------------------------
                # Classification
                # ---------------------------
                if font_size == title_size:
                    block_type = BlockType.TITLE

                elif (
                    font_size in top_sizes
                    and font_size >= 11
                    and not is_noise_line(line_text)
                    and not looks_like_sentence(line_text)
                    and not contains_too_many_symbols(line_text)
                    and len(line_text) < 100
                ):
                    block_type = BlockType.HEADING

                else:
                    block_type = BlockType.PARAGRAPH

                parsed_block = Block(
                    id=str(uuid.uuid4()),
                    type=block_type,
                    text=line_text,
                    page=page_index + 1,
                    bbox=line.get("bbox"),
                    font_size=font_size,
                    font_name=font_name,
                )

                blocks.append(parsed_block)

    doc.close()
    return blocks