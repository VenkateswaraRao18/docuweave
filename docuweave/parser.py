from __future__ import annotations
from .block_cleaner import clean_blocks

import uuid
import re
import fitz
from typing import List

from .models import Block


# -----------------------------
# Bullet Detection
# -----------------------------

BULLET_CHARS = ["•", "-", "*", "▪", "‣", "◦"]

BULLET_PATTERN = re.compile(r"^[\-\*\•\▪\‣\◦]\s+")


def is_bullet(text: str) -> bool:
    text = text.strip()
    return bool(BULLET_PATTERN.match(text))


def clean_text(text: str) -> str:
    return text.strip().replace("\n", " ")


# -----------------------------
# Heading Scoring
# -----------------------------

def heading_score(text: str, font_size: float, median_font: float) -> int:

    score = 0

    if font_size > median_font:
        score += 2

    if text.isupper():
        score += 1

    if len(text) < 60:
        score += 1

    if text.endswith(":"):
        score += 1

    return score


# -----------------------------
# Merge PDF Lines → Paragraphs
# -----------------------------

def merge_lines(lines):

    merged = []

    i = 0

    while i < len(lines):

        current = lines[i]
        text = current["text"]

        j = i + 1

        while j < len(lines):

            next_line = lines[j]

            same_font = abs(current["font_size"] - next_line["font_size"]) < 0.5
            small_gap = abs(next_line["y"] - current["y"]) < 15
            same_indent = abs(next_line["x"] - current["x"]) < 5

            sentence_end = text.endswith((".", "?", "!", ":"))

            if same_font and same_indent and small_gap and not sentence_end:
                text += " " + next_line["text"]
                j += 1
            else:
                break

        merged.append(
            {
                "text": text,
                "font_size": current["font_size"],
                "page": current["page"],
            }
        )

        i = j

    return merged


# -----------------------------
# Fix Detached Bullets
# -----------------------------

def fix_detached_bullets(lines):

    fixed = []

    i = 0

    while i < len(lines):

        text = lines[i]["text"].strip()

        if text in BULLET_CHARS and i + 1 < len(lines):

            next_line = lines[i + 1]

            next_line["text"] = text + " " + next_line["text"]

            i += 1

            continue

        fixed.append(lines[i])

        i += 1

    return fixed


# -----------------------------
# Main PDF Parser
# -----------------------------

def parse_pdf(path: str) -> List[Block]:

    doc = fitz.open(path)

    lines = []

    for page_index, page in enumerate(doc):

        blocks = page.get_text("dict")["blocks"]

        for block in blocks:

            if "lines" not in block:
                continue

            for line in block["lines"]:

                spans = line["spans"]

                text = "".join(span["text"] for span in spans).strip()

                if not text:
                    continue

                font_size = spans[0]["size"]
                x = spans[0]["origin"][0]
                y = spans[0]["origin"][1]

                lines.append(
                    {
                        "text": clean_text(text),
                        "font_size": font_size,
                        "page": page_index + 1,
                        "x": x,
                        "y": y,
                    }
                )

    if not lines:
        return []

    # -----------------------------
    # Fix bullet-only lines
    # -----------------------------

    lines = fix_detached_bullets(lines)

    # -----------------------------
    # Compute median font
    # -----------------------------

    font_sizes = [l["font_size"] for l in lines]
    median_font = sorted(font_sizes)[len(font_sizes) // 2]

    # -----------------------------
    # Merge lines into paragraphs
    # -----------------------------

    merged_lines = merge_lines(lines)

    blocks: List[Block] = []

    for line in merged_lines:

        text = line["text"]

        if is_bullet(text):
            block_type = "list_item"

        else:

            score = heading_score(text, line["font_size"], median_font)

            if score >= 3:
                block_type = "heading"
            else:
                block_type = "paragraph"

        blocks.append(
            Block(
                id=str(uuid.uuid4()),
                type=block_type,
                text=text,
                page=line["page"],
                font_size=line["font_size"],
            )
        )

    
    blocks = clean_blocks(blocks)
    return blocks