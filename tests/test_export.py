"""Tests for the JSON exporter."""
import unittest

from docuweave.exporter import export_document, export_section
from docuweave.models import Block, BlockType, Document, DocumentMetadata, Section


def _make_doc() -> tuple:
    blocks = [
        Block(id="h1", type=BlockType.HEADING,   text="Introduction", page=1, font_size=16),
        Block(id="b1", type=BlockType.PARAGRAPH,  text="Hello.",        page=1, font_size=10),
        Block(id="l1", type=BlockType.LIST,       items=["A", "B"],    page=2),
    ]
    section = Section(id="s1", title="Introduction", level=0, blocks=blocks)
    doc = Document(
        metadata=DocumentMetadata(source="test.pdf", total_pages=3),
        blocks=blocks,
        sections=[section],
    )
    return doc, section


class TestExportSection(unittest.TestCase):

    def setUp(self):
        _, self.section = _make_doc()

    def test_basic_keys(self):
        out = export_section(self.section)
        self.assertIn("id", out)
        self.assertIn("title", out)
        self.assertIn("level", out)
        self.assertIn("blocks", out)
        self.assertIn("subsections", out)

    def test_no_null_items_on_non_list_blocks(self):
        out = export_section(self.section)
        for block in out["blocks"]:
            if block.get("type") != "list":
                self.assertNotIn("items", block,
                    "Non-list blocks must not have an 'items' key")

    def test_list_block_has_items(self):
        out = export_section(self.section)
        list_blocks = [b for b in out["blocks"] if b.get("type") == "list"]
        self.assertTrue(list_blocks, "Expected at least one list block")
        self.assertEqual(list_blocks[0]["items"], ["A", "B"])

    def test_nested_subsections(self):
        parent = Section(id="p", title="Parent", level=0)
        child  = Section(id="c", title="Child",  level=1,
                         blocks=[Block(id="b", type=BlockType.PARAGRAPH, text="x", page=1)])
        parent.subsections.append(child)
        out = export_section(parent)
        self.assertEqual(len(out["subsections"]), 1)
        self.assertEqual(out["subsections"][0]["title"], "Child")


class TestExportDocument(unittest.TestCase):

    def setUp(self):
        self.doc, _ = _make_doc()
        self.chunks = [
            {"id": "c1", "text": "Hello.", "tokens": 1,
             "section_path": "Introduction", "section_title": "Introduction",
             "section_level": 0, "page_start": 1, "page_end": 1,
             "previous_chunk_id": None, "next_chunk_id": None}
        ]

    def test_top_level_keys(self):
        out = export_document(self.doc, self.chunks)
        self.assertIn("metadata", out)
        self.assertIn("sections", out)
        self.assertIn("chunks", out)

    def test_chunks_included(self):
        out = export_document(self.doc, self.chunks)
        self.assertEqual(len(out["chunks"]), 1)

    def test_empty_chunks_default(self):
        out = export_document(self.doc)
        self.assertEqual(out["chunks"], [])

    def test_metadata_source(self):
        out = export_document(self.doc)
        self.assertEqual(out["metadata"]["source"], "test.pdf")


if __name__ == "__main__":
    unittest.main()
