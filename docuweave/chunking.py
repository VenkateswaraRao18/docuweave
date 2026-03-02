from __future__ import annotations

import uuid
from typing import List, Optional

import tiktoken

# from models import Section, Block
from docuweave.models import Block,Section


# ============================================================
# Token Estimator
# ============================================================

class TokenCounter:
    def __init__(self, model_name: str = "gpt-4"):
        try:
            self.encoder = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.encoder.encode(text))


# ============================================================
# Chunk Model (Internal Only)
# ============================================================

class Chunk:
    def __init__(
        self,
        text: str,
        section_title: Optional[str],
        section_level: int,
        page_start: Optional[int],
        page_end: Optional[int],
    ):
        self.id = str(uuid.uuid4())
        self.text = text
        self.section_title = section_title
        self.section_level = section_level
        self.page_start = page_start
        self.page_end = page_end
        self.tokens: Optional[int] = None

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "tokens": self.tokens,
            "section_title": self.section_title,
            "section_level": self.section_level,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }


# ============================================================
# Section-Aware Chunk Builder
# ============================================================

def build_chunks(
    sections: List[Section],
    max_tokens: int = 800,
    model_name: str = "gpt-4",
) -> List[dict]:

    token_counter = TokenCounter(model_name)
    chunks: List[Chunk] = []

    def process_section(section: Section):
        buffer_text = ""
        buffer_tokens = 0
        page_start = None
        page_end = None

        # Combine all blocks in section
        for block in section.blocks:

            text = block.text.strip()
            if not text:
                continue

            token_count = token_counter.count(text)

            # Initialize page tracking
            if page_start is None:
                page_start = block.page
            page_end = block.page

            # If adding exceeds limit → flush
            if buffer_tokens + token_count > max_tokens and buffer_text:
                chunk = Chunk(
                    text=buffer_text.strip(),
                    section_title=section.title,
                    section_level=section.level,
                    page_start=page_start,
                    page_end=page_end,
                )
                chunk.tokens = token_counter.count(buffer_text)
                chunks.append(chunk)

                # Reset buffer
                buffer_text = text + "\n"
                buffer_tokens = token_count
                page_start = block.page
            else:
                buffer_text += text + "\n"
                buffer_tokens += token_count

        # Flush remaining
        if buffer_text:
            chunk = Chunk(
                text=buffer_text.strip(),
                section_title=section.title,
                section_level=section.level,
                page_start=page_start,
                page_end=page_end,
            )
            chunk.tokens = token_counter.count(buffer_text)
            chunks.append(chunk)

        # Process subsections recursively
        for subsection in section.subsections:
            process_section(subsection)

    for section in sections:
        process_section(section)

    return [chunk.to_dict() for chunk in chunks]