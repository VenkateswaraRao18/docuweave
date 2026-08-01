from __future__ import annotations

import uuid
import re
import warnings
import fitz
from typing import List, Tuple

from .models import Block
from .block_cleaner import clean_blocks


# -----------------------------
# Constants
# -----------------------------

BULLET_CHARS = {"•", "-", "*", "▪", "‣", "◦"}
BULLET_PATTERN = re.compile(r"^[\-\*\•\▪\‣\◦]\s+")

# PyMuPDF font flag bitmask: bit 4 = bold
_BOLD_FLAG = 16

# Minimum average characters per page to consider a PDF text-based.
# Below this threshold we warn the user (scanned/image PDF).
_MIN_CHARS_PER_PAGE = 50


# -----------------------------
# Scanned PDF Detection
# -----------------------------

def _check_scanned(doc: fitz.Document) -> None:
    total_pages = len(doc)
    if total_pages == 0:
        return
    total_chars = sum(len(doc[i].get_text()) for i in range(min(total_pages, 10)))
    avg = total_chars / min(total_pages, 10)
    if avg < _MIN_CHARS_PER_PAGE:
        warnings.warn(
            f"DocuWeave: '{doc.name}' appears to be a scanned or image-based PDF "
            f"(avg {avg:.0f} chars/page). Hierarchy detection will be poor. "
            "Consider running OCR (e.g. pytesseract, pymupdf4llm) first.",
            UserWarning,
            stacklevel=4,
        )


# -----------------------------
# Helpers
# -----------------------------

def _is_bullet(text: str) -> bool:
    return bool(BULLET_PATTERN.match(text.strip()))


def _clean_text(text: str) -> str:
    return text.strip().replace("\n", " ")


def _is_bold(flags: int) -> bool:
    return bool(flags & _BOLD_FLAG)


# -----------------------------
# Heading Scoring
# (higher score = more likely a heading)
# -----------------------------

def _heading_score(
    text: str,
    font_size: float,
    median_font: float,
    is_bold: bool,
) -> int:
    score = 0
    if font_size > median_font * 1.05:   # at least 5% larger than median
        score += 2
    if is_bold:
        score += 2                        # bold is a strong signal
    if text.isupper() and len(text) > 2:
        score += 1
    if len(text) < 80:                   # headings are short
        score += 1
    if text.endswith(":"):
        score += 1
    if re.match(r"^\d+[\.\)]\s+\w", text):   # "1. Introduction" style
        score += 1
    return score


# -----------------------------
# Merge PDF Lines → Paragraphs
# -----------------------------

def _merge_lines(lines: List[dict]) -> List[dict]:
    merged = []
    i = 0
    while i < len(lines):
        current = lines[i]
        text = current["text"]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            same_font = abs(current["font_size"] - nxt["font_size"]) < 0.5
            same_bold = current["bold"] == nxt["bold"]
            small_gap = abs(nxt["y"] - current["y"]) < 15
            same_indent = abs(nxt["x"] - current["x"]) < 5
            sentence_end = text.endswith((".", "?", "!", ":"))
            if same_font and same_bold and same_indent and small_gap and not sentence_end:
                text += " " + nxt["text"]
                j += 1
            else:
                break
        merged.append({
            "text": text,
            "font_size": current["font_size"],
            "bold": current["bold"],
            "font_name": current["font_name"],
            "page": current["page"],
            "bbox": current["bbox"],
        })
        i = j
    return merged


# -----------------------------
# Fix Detached Bullets
# -----------------------------

def _fix_detached_bullets(lines: List[dict]) -> List[dict]:
    fixed = []
    i = 0
    while i < len(lines):
        text = lines[i]["text"].strip()
        if text in BULLET_CHARS and i + 1 < len(lines):
            lines[i + 1]["text"] = text + " " + lines[i + 1]["text"]
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
    _check_scanned(doc)

    raw: List[dict] = []

    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                first = spans[0]
                raw.append({
                    "text": _clean_text(text),
                    "font_size": first["size"],
                    "bold": _is_bold(first.get("flags", 0)),
                    "font_name": first.get("font", ""),
                    "page": page_num,
                    "x": first["origin"][0],
                    "y": first["origin"][1],
                    "bbox": list(line["bbox"]),
                })

    if not raw:
        return []

    raw = _fix_detached_bullets(raw)

    font_sizes = [l["font_size"] for l in raw]
    median_font = sorted(font_sizes)[len(font_sizes) // 2]

    merged = _merge_lines(raw)

    blocks: List[Block] = []
    for line in merged:
        text = line["text"]
        if _is_bullet(text):
            block_type = "list_item"
        else:
            score = _heading_score(text, line["font_size"], median_font, line["bold"])
            block_type = "heading" if score >= 3 else "paragraph"

        blocks.append(Block(
            id=str(uuid.uuid4()),
            type=block_type,
            text=text,
            page=line["page"],
            font_size=line["font_size"],
            font_name=line["font_name"],
            bbox=line["bbox"],
        ))

    return clean_blocks(blocks)