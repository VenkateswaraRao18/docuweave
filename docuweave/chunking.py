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
        section_path: str,
        section_level: int,
        page_start: Optional[int],
        page_end: Optional[int],
    ):
        self.id = str(uuid.uuid4())
        self.text = text
        self.section_title = section_title
        self.section_path = section_path
        self.section_level = section_level
        self.page_start = page_start
        self.page_end = page_end
        self.tokens: Optional[int] = None
        self.previous_chunk_id: Optional[str] = None
        self.next_chunk_id: Optional[str] = None

    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "tokens": self.tokens,
            "section_title": self.section_title,
            "section_path": self.section_path,
            "section_level": self.section_level,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "previous_chunk_id": self.previous_chunk_id,
            "next_chunk_id": self.next_chunk_id,
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

    def process_section(section: Section, parent_path: Optional[str] = None):
        buffer_text = ""
        buffer_tokens = 0
        page_start = None
        page_end = None
        last_page_for_buffer = None
        section_label = (section.title or "Untitled").strip()
        section_path = section_label if not parent_path else f"{parent_path} > {section_label}"

        # Combine all blocks in section
        for block in section.blocks:

            text = (block.text or "").strip()
            if not text:
                continue

            token_count = token_counter.count(text)

            # Initialize page tracking
            if page_start is None:
                page_start = block.page

            # If adding exceeds limit → flush
            if buffer_tokens + token_count > max_tokens and buffer_text:
                chunk = Chunk(
                    text=buffer_text.strip(),
                    section_title=section.title,
                    section_path=section_path,
                    section_level=section.level,
                    page_start=page_start,
                    page_end=last_page_for_buffer,
                )
                chunk.tokens = token_counter.count(buffer_text)
                chunks.append(chunk)

                # Reset buffer
                buffer_text = text + "\n"
                buffer_tokens = token_count
                page_start = block.page
                last_page_for_buffer = block.page
            else:
                buffer_text += text + "\n"
                buffer_tokens += token_count
                last_page_for_buffer = block.page

        # Flush remaining
        if buffer_text:
            chunk = Chunk(
                text=buffer_text.strip(),
                section_title=section.title,
                section_path=section_path,
                section_level=section.level,
                page_start=page_start,
                page_end=last_page_for_buffer,
            )
            chunk.tokens = token_counter.count(buffer_text)
            chunks.append(chunk)

        # Process subsections recursively
        for subsection in section.subsections:
            process_section(subsection, section_path)

    for section in sections:
        process_section(section)

    for index, chunk in enumerate(chunks):
        if index > 0:
            chunk.previous_chunk_id = chunks[index - 1].id
        if index < len(chunks) - 1:
            chunk.next_chunk_id = chunks[index + 1].id

    return [chunk.to_dict() for chunk in chunks]