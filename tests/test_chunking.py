"""Tests for chunking logic and _merge_small_chunks."""
import unittest

from docuweave.chunking import build_chunks, _merge_small_chunks, TokenCounter
from docuweave.models import Block, BlockType, Section


def _section(title: str, texts: list[str], level: int = 0) -> Section:
    blocks = [
        Block(id=f"b{i}", type=BlockType.PARAGRAPH, text=t, page=1, font_size=10)
        for i, t in enumerate(texts)
    ]
    return Section(id=f"s_{title}", title=title, level=level, blocks=blocks)


class TestBuildChunks(unittest.TestCase):

    def test_section_path_single_section(self):
        sec = _section("Intro", ["Hello world."])
        chunks = build_chunks([sec], max_tokens=500)
        self.assertEqual(chunks[0]["section_path"], "Intro")

    def test_section_path_nested(self):
        parent = _section("Methods", [], level=0)
        child  = _section("Data Collection", ["We collected data."], level=1)
        parent.subsections.append(child)
        chunks = build_chunks([parent], max_tokens=500)
        self.assertEqual(chunks[0]["section_path"], "Methods > Data Collection")

    def test_nav_links_correct(self):
        sec = _section("S", ["A " * 200, "B " * 200], level=0)  # force 2 chunks
        chunks = build_chunks([sec], max_tokens=50)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIsNone(chunks[0]["previous_chunk_id"])
        self.assertEqual(chunks[0]["next_chunk_id"], chunks[1]["id"])
        self.assertEqual(chunks[1]["previous_chunk_id"], chunks[0]["id"])
        self.assertIsNone(chunks[-1]["next_chunk_id"])

    def test_empty_sections_return_empty(self):
        chunks = build_chunks([], max_tokens=500)
        self.assertEqual(chunks, [])

    def test_untitled_section_falls_back(self):
        sec = Section(id="s", title=None, level=0,
                      blocks=[Block(id="b", type=BlockType.PARAGRAPH, text="text", page=1)])
        chunks = build_chunks([sec], max_tokens=500)
        self.assertIn("Untitled", chunks[0]["section_path"])


class TestMergeSmallChunks(unittest.TestCase):

    def _chunks(self, data: list[tuple]) -> list[dict]:
        """data: list of (section_path, text, tokens)"""
        result = []
        for i, (path, text, toks) in enumerate(data):
            result.append({
                "id": f"c{i}",
                "text": text,
                "tokens": toks,
                "section_path": path,
                "section_title": path,
                "section_level": 0,
                "page_start": 1,
                "page_end": 1,
                "previous_chunk_id": None,
                "next_chunk_id": None,
            })
        return result

    def test_merges_same_section(self):
        chunks = self._chunks([("A", "hello", 10), ("A", "world", 10)])
        tc = TokenCounter()
        merged = _merge_small_chunks(chunks, max_tokens=100, token_counter=tc)
        self.assertEqual(len(merged), 1)
        self.assertIn("hello", merged[0]["text"])
        self.assertIn("world", merged[0]["text"])

    def test_does_not_merge_different_sections(self):
        chunks = self._chunks([("A", "hello", 10), ("B", "world", 10)])
        tc = TokenCounter()
        merged = _merge_small_chunks(chunks, max_tokens=100, token_counter=tc)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["section_path"], "A")
        self.assertEqual(merged[1]["section_path"], "B")

    def test_does_not_merge_when_over_limit(self):
        chunks = self._chunks([("A", "big " * 100, 400), ("A", "also big " * 100, 400)])
        tc = TokenCounter()
        merged = _merge_small_chunks(chunks, max_tokens=512, token_counter=tc)
        self.assertEqual(len(merged), 2)

    def test_nav_links_rewritten_after_merge(self):
        chunks = self._chunks([("A", "a", 5), ("A", "b", 5), ("A", "c", 5)])
        tc = TokenCounter()
        merged = _merge_small_chunks(chunks, max_tokens=100, token_counter=tc)
        self.assertEqual(len(merged), 1)
        self.assertIsNone(merged[0]["previous_chunk_id"])
        self.assertIsNone(merged[0]["next_chunk_id"])

    def test_empty_input(self):
        tc = TokenCounter()
        self.assertEqual(_merge_small_chunks([], 512, tc), [])


if __name__ == "__main__":
    unittest.main()
