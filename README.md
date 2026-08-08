# DocuWeave

[![PyPI version](https://img.shields.io/pypi/v/docuweave)](https://pypi.org/project/docuweave/)
[![Python](https://img.shields.io/pypi/pyversions/docuweave)](https://pypi.org/project/docuweave/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-mrjvenky18%2Fdocuweave--bench-yellow)](https://huggingface.co/datasets/mrjvenky18/docuweave-bench)

**Layout-aware PDF chunker for production RAG pipelines.**

DocuWeave reads font sizes and bold signals from PDFs to reconstruct the heading hierarchy, then cuts chunks at section boundaries — never across them. Each chunk knows which section it came from, what page it lives on, and what surrounds it.

---

## Benchmark

Evaluated on **417 PDFs · 6,100 QA pairs** across five domains (research, technical, legal, financial, medical) using `bge-base-en-v1.5` embeddings and FAISS retrieval. Full dataset on [HuggingFace](https://huggingface.co/datasets/mrjvenky18/docuweave-bench).

| Chunker | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| **DocuWeave** | **0.286** | **0.448** | **0.515** | **0.591** | **0.384** | **0.434** |
| Naive (fixed-size) | 0.198 | 0.350 | 0.435 | 0.565 | 0.300 | 0.396 |
| Recursive (LangChain default) | 0.179 | 0.331 | 0.414 | 0.533 | 0.279 | 0.366 |
| LangChain (full doc) | 0.148 | 0.271 | 0.340 | 0.438 | 0.229 | 0.294 |
| Semantic | 0.118 | 0.202 | 0.252 | 0.334 | 0.175 | 0.274 |
| PDFPlumber | 0.113 | 0.193 | 0.239 | 0.312 | 0.166 | 0.265 |

DocuWeave ranks **#1 on every metric**. All differences are statistically significant (Wilcoxon signed-rank, p < 0.001).

vs. LangChain Recursive (the most common RAG default): **+60% R@1**  
vs. LangChain full-doc loader: **+94% R@1**  
vs. Semantic chunking: **+142% R@1**

---

## Why not just split by characters?

Character-based splitters cut at fixed budgets — a chunk can start mid-sentence in one section and end mid-sentence in another. When you ask "What is the token expiry time for API authentication?", the answer might be split across two chunks with different section contexts, and neither chunk's embedding points clearly at the answer.

DocuWeave cuts at section boundaries. The entire answer lives in one self-contained chunk, its embedding is fully anchored to that topic, and retrieval finds it.

A chunk from DocuWeave:

```json
{
  "id": "c_0014",
  "text": "All API requests must include a valid OAuth 2.0 bearer token...",
  "tokens": 487,
  "section_title": "3.2 Authentication",
  "section_path": "3 API Reference > 3.2 Authentication",
  "section_level": 1,
  "page_start": 4,
  "page_end": 5,
  "previous_chunk_id": "c_0013",
  "next_chunk_id": "c_0015"
}
```

---

## Installation

```bash
pip install docuweave
```

With optional integrations:

```bash
pip install "docuweave[langchain]"     # LangChain BaseLoader
pip install "docuweave[llamaindex]"    # LlamaIndex BaseReader
pip install "docuweave[haystack]"      # Haystack Document
pip install "docuweave[qdrant]"        # qdrant-client
pip install "docuweave[milvus]"        # pymilvus
pip install "docuweave[all]"           # everything
```

Requires Python 3.9+.

---

## Quick start

```python
from docuweave import parse

doc = parse("paper.pdf")

# how confident DocuWeave is about the heading structure (0.0–1.0)
print(doc.hierarchy_confidence)

chunks = doc.to_chunks(max_tokens=512)
doc.save_json("paper.json")
```

---

## Processing a folder

```python
from docuweave import parse_directory

docs = parse_directory(
    "pdfs/",
    pattern="**/*.pdf",
    min_confidence=0.3,   # skip scanned/image-only PDFs
    on_error="skip",
    progress=True,
)

for doc in docs:
    chunks = doc.to_chunks(max_tokens=512)
```

---

## LangChain

```python
from docuweave.integrations import DocuWeaveLoader

loader = DocuWeaveLoader("paper.pdf", max_tokens=512)

# load all at once
docs = loader.load()

# or stream one chunk at a time
for doc in loader.lazy_load():
    print(doc.page_content[:100])
    print(doc.metadata["section_path"])
```

Or convert existing chunks:

```python
from docuweave.integrations import to_langchain_documents
lc_docs = to_langchain_documents(chunks)
```

---

## LlamaIndex

```python
from docuweave.integrations import DocuWeaveReader

reader = DocuWeaveReader(max_tokens=512)
nodes = reader.load_data("paper.pdf")
```

---

## Haystack

```python
from docuweave.integrations import to_haystack_documents
haystack_docs = to_haystack_documents(chunks)
```

---

## Vector DB exports

### ChromaDB

```python
import chromadb
from docuweave import parse
from docuweave.vector_exporters import export_chroma

doc = parse("paper.pdf")
chunks = doc.to_chunks(max_tokens=512)

client = chromadb.Client()
collection = client.get_or_create_collection("papers")
export_chroma(chunks, collection)
```

### Qdrant

```python
from qdrant_client import QdrantClient
from docuweave.vector_exporters import export_qdrant

client = QdrantClient(url="http://localhost:6333")
embeddings = your_model.encode([c["text"] for c in chunks]).tolist()
export_qdrant(chunks, client, collection_name="papers", embeddings=embeddings)
```

### Milvus

```python
from pymilvus import Collection
from docuweave.vector_exporters import export_milvus

collection = Collection("papers")
embeddings = your_model.encode([c["text"] for c in chunks]).tolist()
export_milvus(chunks, collection, embeddings=embeddings)
```

### Pinecone / Weaviate / FAISS

```python
doc.to_pinecone(embeddings=embeddings)
doc.to_weaviate(embeddings=embeddings)
doc.to_faiss_jsonl("faiss_records.jsonl")
```

---

## CLI

```bash
# single file
docuweave paper.pdf -o paper.json --max-tokens 512

# batch mode
docuweave --directory pdfs/ --output-dir out/ --min-confidence 0.3

# skip low-quality PDFs silently
docuweave --directory pdfs/ --output-dir out/ --min-confidence 0.3 --on-error skip

# vector export
docuweave paper.pdf --export chroma
docuweave paper.pdf --export qdrant
docuweave paper.pdf --export faiss-jsonl -o records.jsonl
```

---

## `hierarchy_confidence`

Every parsed document has a score between 0.0 and 1.0:

- **≥ 0.6** — clear multi-level heading structure; chunking will be accurate
- **0.3–0.6** — partial structure; DocuWeave still outperforms flat splitting
- **< 0.3** — likely scanned, image-heavy, or uniform-font document

Use `min_confidence` in `parse_directory()` to filter these out automatically.

---

## Other useful properties

```python
doc = parse("paper.pdf")

doc.num_pages                       # int
len(doc)                            # chunk count (after to_chunks())
repr(doc)                           # DocuWeaveDocument(file='paper.pdf', pages=12, sections=8, chunks=24, confidence=0.71)
doc.iter_chunks(max_tokens=512)     # lazy iterator
doc.to_json()                       # full dict including hierarchy_confidence
```

---

## How it works

1. **Block extraction** — PyMuPDF extracts text spans with font size, bold flag, and bounding box.
2. **Heading scoring** — each block gets a score based on font size vs. page median, bold, uppercase, length, and numbered-heading patterns. Blocks scoring ≥ 3 become headings.
3. **Hierarchy construction** — headings are stacked into a tree by font size. Paragraphs following a heading belong to that section.
4. **Block cleaning** — bullet continuations are merged, list items are grouped, headers/footers are removed.
5. **Chunking** — sections are sliced into token-bounded chunks (default 512 tokens). Small chunks merge *within* the same section only — never across section boundaries.
6. **Export** — chunks carry `section_path`, page span, and linked-list pointers for context expansion.

---

## Known limitations

- Scanned PDFs (image-only) return no text. Check `hierarchy_confidence`.
- Multi-column layouts occasionally mis-order blocks (PyMuPDF limitation).
- Tables are treated as text blocks, not structured data.
- DOCX and HTML not yet supported.

---

## Dataset

The benchmark dataset (417 PDFs, 6,100 QA pairs) is publicly available:

**HuggingFace:** [mrjvenky18/docuweave-bench](https://huggingface.co/datasets/mrjvenky18/docuweave-bench)

---

## Citation

```bibtex
@article{jannegorla2026docuweave,
  title   = {DocuWeave: Layout-Aware {PDF} Chunking for Retrieval-Augmented Generation},
  author  = {Jannegorla, Venkateswara Rao},
  journal = {arXiv preprint},
  year    = {2026},
}
```

---

## License

MIT — see [LICENSE](LICENSE).

**Author:** Venkateswara Rao Jannegorla · [GitHub](https://github.com/VenkateswaraRao18) · venkyjannegorla@gmail.com
