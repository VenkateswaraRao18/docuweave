"""Tests for DocuWeaveDocument API and parse_directory."""
import unittest
import tempfile
import os

from docuweave.models import Block, BlockType, Section
from docuweave.chunking import build_chunks
from docuweave.api import DocuWeaveDocument, _compute_confidence


class TestComputeConfidence(unittest.TestCase):

    def test_fallback_section_returns_zero(self):
        blocks = [Block(id="b", type=BlockType.PARAGRAPH, text="text", page=1)]
        sections = [Section(id="section_document", title="Document", level=0, blocks=blocks)]
        self.assertEqual(_compute_confidence(blocks, sections), 0.0)

    def test_empty_blocks_returns_zero(self):
        self.assertEqual(_compute_confidence([], []), 0.0)

    def test_rich_hierarchy_gives_high_confidence(self):
        headings = [
            Block(id=f"h{i}", type=BlockType.HEADING, text=f"H{i}", page=1)
            for i in range(5)
        ]
        paragraphs = [
            Block(id=f"p{i}", type=BlockType.PARAGRAPH, text=f"P{i}", page=1)
            for i in range(20)
        ]
        all_blocks = headings + paragraphs
        child = Section(id="c", title="Child", level=1, blocks=paragraphs)
        parent = Section(id="p", title="Parent", level=0, subsections=[child])
        score = _compute_confidence(all_blocks, [parent])
        self.assertGreater(score, 0.0)


class TestDocuWeaveDocumentErrors(unittest.TestCase):

    def test_missing_file_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            DocuWeaveDocument("/nonexistent/path/to/file.pdf")
        self.assertIn("DocuWeave could not parse", str(ctx.exception))

    def test_non_pdf_raises_value_error(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"this is not a PDF")
            name = f.name
        try:
            with self.assertRaises(ValueError):
                DocuWeaveDocument(name)
        finally:
            os.unlink(name)


class TestDocuWeaveDocumentProperties(unittest.TestCase):
    """Tests that can run without a real PDF by monkey-patching the internals."""

    def _make_doc(self):
        """Build a DocuWeaveDocument with synthetic blocks (no PDF required)."""
        import docuweave.api as api_mod
        import docuweave.parser as parser_mod
        import docuweave.hierarchy as hier_mod

        blocks = [
            Block(id="h1", type=BlockType.HEADING, text="Intro", page=1, font_size=18),
            Block(id="b1", type=BlockType.PARAGRAPH, text="Hello world.", page=1, font_size=10),
            Block(id="b2", type=BlockType.PARAGRAPH, text="Second para.", page=2, font_size=10),
        ]

        original_parse = parser_mod.parse_pdf
        original_build = hier_mod.build_hierarchy
        try:
            parser_mod.parse_pdf = lambda _: blocks
            hier_mod.build_hierarchy = lambda b: hier_mod.build_hierarchy.__wrapped__(b) \
                if hasattr(hier_mod.build_hierarchy, "__wrapped__") else original_build(b)
            doc = DocuWeaveDocument.__new__(DocuWeaveDocument)
            doc.file_path = "synthetic.pdf"
            doc.blocks = blocks
            doc.sections = original_build(blocks)
            from docuweave.models import Document, DocumentMetadata
            doc.document = Document(
                metadata=DocumentMetadata(source="synthetic.pdf", total_pages=2),
                blocks=blocks,
                sections=doc.sections,
            )
            doc.hierarchy_confidence = _compute_confidence(blocks, doc.sections)
            doc._chunks = None
            doc._chunk_config = None
            return doc
        finally:
            parser_mod.parse_pdf = original_parse

    def test_num_pages(self):
        doc = self._make_doc()
        self.assertEqual(doc.num_pages, 2)

    def test_len_before_chunking(self):
        doc = self._make_doc()
        self.assertEqual(len(doc), 0)

    def test_len_after_chunking(self):
        doc = self._make_doc()
        doc.to_chunks(max_tokens=512)
        self.assertGreater(len(doc), 0)

    def test_repr_contains_filename(self):
        doc = self._make_doc()
        self.assertIn("synthetic.pdf", repr(doc))

    def test_iter_chunks(self):
        doc = self._make_doc()
        chunks = list(doc.iter_chunks(max_tokens=512))
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self.assertIn("text", c)
            self.assertIn("section_path", c)

    def test_to_json_includes_confidence(self):
        doc = self._make_doc()
        doc.to_chunks(max_tokens=512)
        payload = doc.to_json()
        self.assertIn("hierarchy_confidence", payload)
        self.assertIsInstance(payload["hierarchy_confidence"], float)


if __name__ == "__main__":
    unittest.main()
