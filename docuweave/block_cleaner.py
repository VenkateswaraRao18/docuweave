from typing import List
from .models import Block


def is_project_title(text: str) -> bool:

    if "||" in text and len(text) < 140:
        return True

    if text.endswith("|"):
        return True

    return False


def merge_bullet_continuations(blocks: List[Block]) -> List[Block]:

    cleaned = []

    for block in blocks:

        if (
            cleaned
            and cleaned[-1].type == "list_item"
            and block.type == "paragraph"
        ):
            cleaned[-1].text += " " + block.text
        else:
            cleaned.append(block)

    return cleaned


def detect_project_titles(blocks: List[Block]) -> List[Block]:

    for b in blocks:

        if b.type == "paragraph" and is_project_title(b.text):
            b.type = "heading"

    return blocks


import re
from .models import Block, BlockType


def group_lists(blocks):

    result = []
    i = 0

    while i < len(blocks):

        block = blocks[i]

        if block.type == "list_item":

            items = []

            while i < len(blocks) and blocks[i].type == "list_item":

                text = blocks[i].text.strip()

                # remove any bullet prefix
                text = re.sub(r"^[•▪‣◦*]+\s*", "", text)

                items.append(text)

                i += 1

            result.append(
                Block(
                    id="list_" + str(len(result)),
                    type=BlockType.LIST,
                    items=items,
                    page=block.page,
                )
            )

        else:
            result.append(block)
            i += 1

    return result

import re
from .models import Block


BULLET_CHARS = r"[•▪‣◦\*]"


def normalize_bullets(blocks):

    new_blocks = []

    for block in blocks:

        if block.type != "list_item":
            new_blocks.append(block)
            continue

        text = block.text.strip()

        # remove duplicate bullet prefixes
        text = re.sub(rf"^(?:{BULLET_CHARS}\s*)+", "", text)

        # split only on real bullet characters
        parts = re.split(rf"\s*{BULLET_CHARS}\s*", text)

        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) == 1:

            block.text = "• " + parts[0]
            new_blocks.append(block)

        else:

            for part in parts:
                new_blocks.append(
                    Block(
                        id=block.id,
                        type="list_item",
                        text="• " + part,
                        page=block.page,
                        font_size=block.font_size,
                    )
                )

    return new_blocks


def clean_blocks(blocks):

    blocks = merge_bullet_continuations(blocks)

    blocks = normalize_bullets(blocks)

    blocks = detect_project_titles(blocks)

    blocks = group_lists(blocks)

    return blocks

