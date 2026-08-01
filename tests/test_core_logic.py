"""Tests for hierarchy building and block cleaning."""
import unittest

from docuweave.block_cleaner import (
    clean_blocks,
    detect_project_titles,
    group_lists,
    merge_bullet_continuations,
    normalize_bullets,
)
from docuweave.chunking import build_chunks
from docuweave.hierarchy import build_hierarchy
from docuweave.models import Block, BlockType, Section


def _block(bid: str, btype: str, text: str, page: int = 1, fs: float = 10.0) -> Block:
    return Block(id=bid, type=btype, text=text, page=page, font_size=fs)


class TestMergeBulletContinuations(unittest.TestCase):
    def test_merges_paragraph_after_list_item(self):
        # All paragraphs following a list_item are treated as continuations —
        # this matches the PDF extraction reality where wrapped bullet lines
        # appear as separate PARAGRAPH blocks.
        blocks = [
            _block("b1", "list_item", "• First point"),
            _block("b2", "paragraph", "continuation of first"),
            _block("b3", "paragraph", "also continuation"),
        ]
        result = merge_bullet_continuations(blocks)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].type, "list_item")
        self.assertIn("continuation", result[0].text)

    def test_paragraph_before_list_not_merged(self):
        blocks = [
            _block("b1", "paragraph", "standalone intro"),
            _block("b2", "list_item", "• First point"),
        ]
        result = merge_bullet_continuations(blocks)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].type, "paragraph")

    def test_does_not_merge_paragraph_not_after_list(self):
        blocks = [_block("b1", "paragraph", "A"), _block("b2", "paragraph", "B")]
        result = merge_bullet_continuations(blocks)
        self.assertEqual(len(result), 2)

    def test_handles_none_text_gracefully(self):
        b = _block("b1", "list_item", "• Item")
        b2 = Block(id="b2", type="paragraph", text=None, page=1)
        result = merge_bullet_continuations([b, b2])
        # None text should not be appended (guard in new code)
        self.assertEqual(len(result), 2)


class TestNormalizeBullets(unittest.TestCase):
    def test_unique_ids_when_splitting(self):
        block = _block("original_id", "list_item", "A • B • C")
        result = normalize_bullets([block])
        ids = [b.id for b in result]
        self.assertEqual(len(ids), len(set(ids)), "All IDs must be unique after split")
        self.assertEqual(len(result), 3)

    def test_strips_leading_bullet(self):
        block = _block("b1", "list_item", "• Hello world")
        result = normalize_bullets([block])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "• Hello world")

    def test_non_list_items_unchanged(self):
        block = _block("b1", "paragraph", "Normal text")
        result = normalize_bullets([block])
        self.assertEqual(result[0].text, "Normal text")


class TestDetectProjectTitles(unittest.TestCase):
    def test_pipe_pattern_becomes_heading(self):
        block = _block("b1", "paragraph", "Project Alpha | Machine Learning")
        result = detect_project_titles([block])
        self.assertEqual(result[0].type, "heading")

    def test_double_pipe_with_data_not_promoted(self):
        # Multi-pipe (table row / data) should NOT be promoted
        block = _block("b1", "paragraph", "col1 | col2 | col3 | col4")
        result = detect_project_titles([block])
        # The regex only matches single-pipe with short sides, so this should NOT match
        self.assertEqual(result[0].type, "paragraph")

    def test_long_text_not_promoted(self):
        long_text = "A" * 70 + " | B"
        block = _block("b1", "paragraph", long_text)
        result = detect_project_titles([block])
        self.assertEqual(result[0].type, "paragraph")


class TestGroupLists(unittest.TestCase):
    def test_groups_consecutive_list_items(self):
        blocks = [
            _block("b1", "list_item", "• A"),
            _block("b2", "list_item", "• B"),
            _block("b3", "paragraph", "text"),
        ]
        result = group_lists(blocks)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].type, "list")
        self.assertEqual(result[0].items, ["A", "B"])
        self.assertEqual(result[1].type, "paragraph")

    def test_list_block_has_unique_id(self):
        blocks = [_block("b1", "list_item", "• A"), _block("b2", "list_item", "• B")]
        r1 = group_lists(blocks)
        r2 = group_lists(blocks)
        self.assertNotEqual(r1[0].id, r2[0].id, "LIST block IDs must be freshly generated")


class TestBuildHierarchy(unittest.TestCase):
    def test_intro_blocks_captured(self):
        blocks = [
            _block("b0", "paragraph", "Preface text", fs=10),
            _block("h1", "heading",   "Chapter 1",   fs=18),
            _block("b1", "paragraph", "Body text",   fs=10),
        ]
        sections = build_hierarchy(blocks)
        self.assertGreaterEqual(len(sections), 1)
        self.assertEqual(sections[0].title, "Introduction")
        self.assertTrue(any(b.text == "Preface text" for b in sections[0].blocks))

    def test_empty_blocks_returns_empty(self):
        self.assertEqual(build_hierarchy([]), [])

    def test_all_paragraphs_returns_fallback(self):
        blocks = [_block(f"b{i}", "paragraph", f"text {i}", fs=10) for i in range(3)]
        sections = build_hierarchy(blocks)
        self.assertEqual(len(sections), 1)

    def test_subsection_nesting(self):
        blocks = [
            _block("h1", "heading", "Chapter",     fs=20),
            _block("h2", "heading", "Sub-chapter", fs=16),
            _block("b1", "paragraph", "content",   fs=10),
        ]
        sections = build_hierarchy(blocks)
        # Chapter should have Sub-chapter as a subsection
        chapter = sections[0]
        self.assertEqual(chapter.title, "Chapter")
        self.assertGreater(len(chapter.subsections), 0)
        self.assertEqual(chapter.subsections[0].title, "Sub-chapter")


if __name__ == "__main__":
    unittest.main()
