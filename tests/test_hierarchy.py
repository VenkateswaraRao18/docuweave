"""Tests for hierarchy builder edge cases."""
import unittest

from docuweave.hierarchy import build_hierarchy
from docuweave.models import Block, BlockType


def _h(bid: str, text: str, fs: float, page: int = 1) -> Block:
    return Block(id=bid, type=BlockType.HEADING, text=text, page=page, font_size=fs)


def _p(bid: str, text: str = "body", page: int = 1) -> Block:
    return Block(id=bid, type=BlockType.PARAGRAPH, text=text, page=page, font_size=10.0)


class TestBuildHierarchyEdgeCases(unittest.TestCase):

    def test_empty_returns_empty(self):
        self.assertEqual(build_hierarchy([]), [])

    def test_only_paragraphs_gives_single_intro_section(self):
        # No headings → all blocks become intro_blocks → one "Introduction" section
        sections = build_hierarchy([_p("b1"), _p("b2")])
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].id, "section_intro")
        self.assertEqual(len(sections[0].blocks), 2)

    def test_preface_before_first_heading_goes_to_intro(self):
        blocks = [_p("p0"), _h("h1", "Chapter", 18), _p("p1")]
        sections = build_hierarchy(blocks)
        self.assertEqual(sections[0].title, "Introduction")
        self.assertTrue(any(b.id == "p0" for b in sections[0].blocks))

    def test_three_levels_nested_correctly(self):
        blocks = [
            _h("h1", "Part",        24),
            _h("h2", "Chapter",     18),
            _h("h3", "Section",     14),
            _p("p1"),
        ]
        sections = build_hierarchy(blocks)
        self.assertEqual(sections[0].title, "Part")
        ch = sections[0].subsections[0]
        self.assertEqual(ch.title, "Chapter")
        sec = ch.subsections[0]
        self.assertEqual(sec.title, "Section")
        self.assertEqual(len(sec.blocks), 1)

    def test_heading_without_font_size_is_skipped(self):
        blocks = [
            Block(id="h1", type=BlockType.HEADING, text="No size", page=1, font_size=None),
            _p("p1"),
        ]
        # Should not crash; falls back gracefully
        sections = build_hierarchy(blocks)
        self.assertIsInstance(sections, list)

    def test_heading_text_none_is_skipped(self):
        blocks = [
            Block(id="h1", type=BlockType.HEADING, text=None, page=1, font_size=18),
            _p("p1"),
        ]
        sections = build_hierarchy(blocks)
        self.assertIsInstance(sections, list)

    def test_many_headings_same_size_all_become_root(self):
        blocks = [
            _h("h1", "A", 18), _p("p1"),
            _h("h2", "B", 18), _p("p2"),
            _h("h3", "C", 18), _p("p3"),
        ]
        sections = build_hierarchy(blocks)
        self.assertEqual(len(sections), 3)
        for s in sections:
            self.assertEqual(len(s.blocks), 1)


if __name__ == "__main__":
    unittest.main()
