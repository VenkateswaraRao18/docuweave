import unittest

from docuweave.integrations import to_langchain_documents, to_llamaindex_nodes
from docuweave.vector_exporters import build_vector_records, export_pinecone, export_weaviate


class TestIntegrations(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "id": "c1",
                "text": "hello world",
                "tokens": 2,
                "section_title": "Intro",
                "section_path": "Intro",
                "section_level": 0,
                "page_start": 1,
                "page_end": 1,
                "previous_chunk_id": None,
                "next_chunk_id": "c2",
            },
            {
                "id": "c2",
                "text": "next chunk",
                "tokens": 2,
                "section_title": "Intro",
                "section_path": "Intro",
                "section_level": 0,
                "page_start": 1,
                "page_end": 2,
                "previous_chunk_id": "c1",
                "next_chunk_id": None,
            },
        ]

    def test_langchain_fallback_shape(self):
        docs = to_langchain_documents(self.chunks)
        self.assertEqual(len(docs), 2)
        if isinstance(docs[0], dict):
            self.assertIn("page_content", docs[0])
            self.assertIn("metadata", docs[0])

    def test_llamaindex_fallback_shape(self):
        nodes = to_llamaindex_nodes(self.chunks)
        self.assertEqual(len(nodes), 2)
        if isinstance(nodes[0], dict):
            self.assertIn("text", nodes[0])
            self.assertIn("metadata", nodes[0])

    def test_vector_export_shapes(self):
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        records = build_vector_records(self.chunks, embeddings=embeddings)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["values"], [0.1, 0.2])

        pinecone_rows = export_pinecone(self.chunks, embeddings=embeddings)
        self.assertIn("id", pinecone_rows[0])
        self.assertIn("values", pinecone_rows[0])
        self.assertIn("metadata", pinecone_rows[0])

        weaviate_rows = export_weaviate(self.chunks, embeddings=embeddings)
        self.assertIn("id", weaviate_rows[0])
        self.assertIn("properties", weaviate_rows[0])
        self.assertIn("vector", weaviate_rows[0])


if __name__ == "__main__":
    unittest.main()
