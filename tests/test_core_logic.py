import unittest

from docuweave.chunking import build_chunks
from docuweave.hierarchy import build_hierarchy
from docuweave.models import Block, BlockType, Section


class TestHierarchy(unittest.TestCase):
    def test_intro_blocks_are_not_dropped(self):
        blocks = [
            Block(id="b1", type=BlockType.PARAGRAPH, text="Preface text", page=1, font_size=10),
            Block(id="h1", type=BlockType.HEADING, text="Chapter 1", page=1, font_size=18),
            Block(id="b2", type=BlockType.PARAGRAPH, text="Body", page=1, font_size=10),
        ]

        sections = build_hierarchy(blocks)
        self.assertGreaterEqual(len(sections), 1)
        self.assertEqual(sections[0].title, "Introduction")
        self.assertEqual(len(sections[0].blocks), 1)
        self.assertEqual(sections[0].blocks[0].text, "Preface text")


class TestChunking(unittest.TestCase):
    def test_chunk_relationship_and_section_path(self):
        section = Section(
            id="s1",
            title="Intro",
            level=0,
            blocks=[
                Block(id="b1", type=BlockType.PARAGRAPH, text="Hello world", page=1, font_size=10),
                Block(id="b2", type=BlockType.PARAGRAPH, text="Second sentence", page=2, font_size=10),
            ],
        )

        chunks = build_chunks([section], max_tokens=1, model_name="gpt-4")
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["section_path"], "Intro")
        self.assertIsNone(chunks[0]["previous_chunk_id"])
        self.assertEqual(chunks[0]["next_chunk_id"], chunks[1]["id"])
        self.assertEqual(chunks[1]["previous_chunk_id"], chunks[0]["id"])


if __name__ == "__main__":
    unittest.main()
