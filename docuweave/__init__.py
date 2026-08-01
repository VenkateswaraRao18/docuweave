from .api import parse, parse_directory, DocuWeaveDocument
from .integrations import (
    to_langchain_documents,
    to_llamaindex_nodes,
    to_haystack_documents,
    DocuWeaveLoader,
    DocuWeaveReader,
)
from .vector_exporters import (
    build_vector_records,
    export_chroma,
    export_faiss_jsonl,
    export_milvus,
    export_pinecone,
    export_qdrant,
    export_weaviate,
)

__version__ = "0.2.0"

__all__ = [
    # Core
    "parse",
    "parse_directory",
    "DocuWeaveDocument",
    # Framework loaders
    "DocuWeaveLoader",
    "DocuWeaveReader",
    # Converters
    "to_langchain_documents",
    "to_llamaindex_nodes",
    "to_haystack_documents",
    # Vector exporters
    "build_vector_records",
    "export_chroma",
    "export_faiss_jsonl",
    "export_milvus",
    "export_pinecone",
    "export_qdrant",
    "export_weaviate",
]
