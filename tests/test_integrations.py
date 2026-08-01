"""Tests for framework integrations and vector exporters."""
import unittest

from docuweave.integrations import (
    to_haystack_documents,
    to_langchain_documents,
    to_llamaindex_nodes,
)
from docuweave.vector_exporters import (
    build_vector_records,
    export_chroma,
    export_pinecone,
    export_weaviate,
)


CHUNKS = [
    {
        "id": "c1", "text": "hello world", "tokens": 2,
        "section_title": "Intro", "section_path": "Intro",
        "section_level": 0, "page_start": 1, "page_end": 1,
        "previous_chunk_id": None, "next_chunk_id": "c2",
    },
    {
        "id": "c2", "text": "next chunk", "tokens": 2,
        "section_title": "Intro", "section_path": "Intro",
        "section_level": 0, "page_start": 1, "page_end": 2,
        "previous_chunk_id": "c1", "next_chunk_id": None,
    },
]
EMBEDDINGS = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


class TestLangChainIntegration(unittest.TestCase):
    def test_returns_two_docs(self):
        docs = to_langchain_documents(CHUNKS)
        self.assertEqual(len(docs), 2)

    def test_page_content_matches_text(self):
        docs = to_langchain_documents(CHUNKS)
        first = docs[0]
        text = first.page_content if hasattr(first, "page_content") else first["page_content"]
        self.assertEqual(text, "hello world")

    def test_metadata_has_section_path(self):
        docs = to_langchain_documents(CHUNKS)
        meta = docs[0].metadata if hasattr(docs[0], "metadata") else docs[0]["metadata"]
        self.assertIn("section_path", meta)


class TestLlamaIndexIntegration(unittest.TestCase):
    def test_returns_two_nodes(self):
        nodes = to_llamaindex_nodes(CHUNKS)
        self.assertEqual(len(nodes), 2)

    def test_text_field_populated(self):
        nodes = to_llamaindex_nodes(CHUNKS)
        first = nodes[0]
        text = first.text if hasattr(first, "text") else first["text"]
        self.assertEqual(text, "hello world")


class TestHaystackIntegration(unittest.TestCase):
    def test_returns_two_docs(self):
        docs = to_haystack_documents(CHUNKS)
        self.assertEqual(len(docs), 2)

    def test_content_or_dict_shape(self):
        docs = to_haystack_documents(CHUNKS)
        first = docs[0]
        content = first.content if hasattr(first, "content") else first.get("content")
        self.assertEqual(content, "hello world")


class TestVectorExporters(unittest.TestCase):
    def test_build_vector_records_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            build_vector_records(CHUNKS, embeddings=[[0.1]])

    def test_build_vector_records_with_embeddings(self):
        records = build_vector_records(CHUNKS, embeddings=EMBEDDINGS)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["values"], EMBEDDINGS[0])
        self.assertIn("section_path", records[0]["metadata"])

    def test_pinecone_shape(self):
        rows = export_pinecone(CHUNKS, embeddings=EMBEDDINGS)
        self.assertIn("id", rows[0])
        self.assertIn("values", rows[0])
        self.assertIn("metadata", rows[0])

    def test_weaviate_shape(self):
        rows = export_weaviate(CHUNKS, embeddings=EMBEDDINGS)
        self.assertIn("id", rows[0])
        self.assertIn("properties", rows[0])
        self.assertIn("vector", rows[0])
        self.assertIn("text", rows[0]["properties"])

    def test_chroma_upsert_called(self):
        """ChromaDB export should call collection.upsert with correct keys."""
        calls = {}

        class FakeCollection:
            def upsert(self, **kwargs):
                calls.update(kwargs)

        export_chroma(CHUNKS, FakeCollection(), embeddings=EMBEDDINGS)
        self.assertIn("ids", calls)
        self.assertIn("documents", calls)
        self.assertIn("metadatas", calls)
        self.assertIn("embeddings", calls)
        self.assertEqual(calls["ids"], ["c1", "c2"])
        self.assertEqual(calls["documents"][0], "hello world")

    def test_chroma_without_embeddings(self):
        """ChromaDB export without embeddings should not pass embeddings key."""
        calls = {}

        class FakeCollection:
            def upsert(self, **kwargs):
                calls.update(kwargs)

        export_chroma(CHUNKS, FakeCollection())
        self.assertNotIn("embeddings", calls)

    def test_qdrant_length_mismatch_raises(self):
        from docuweave.vector_exporters import export_qdrant

        class FakeClient:
            def upsert(self, **kwargs): pass

        with self.assertRaises((ValueError, ImportError)):
            # ValueError if qdrant_client installed, ImportError if not
            export_qdrant(CHUNKS, FakeClient(), "col", embeddings=[[0.1]])


if __name__ == "__main__":
    unittest.main()
