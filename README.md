# DocuWeave

[![PyPI version](https://img.shields.io/pypi/v/docuweave)](https://pypi.org/project/docuweave/)
[![Python](https://img.shields.io/pypi/pyversions/docuweave)](https://pypi.org/project/docuweave/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Typed](https://img.shields.io/badge/typing-PEP%20561-informational)](https://peps.python.org/pep-0561/)

PDF chunker that preserves document structure for RAG pipelines.

Instead of splitting text by fixed character counts, DocuWeave reads font sizes and bold signals from the PDF to reconstruct the heading hierarchy, then cuts chunks at section boundaries. Each chunk knows which section it came from, what page it lives on, and what surrounds it.

---

## Why not just split by characters?

Character-based splitters treat every paragraph the same. A chunk tagged `"section_path": "3.2 Experimental Setup"` retrieves more precisely than one that happens to contain those words somewhere in the middle. When a retrieval miss happens, you also know *where* in the document to look.

Our benchmark on 390 PDFs across five domains (academic papers, legal, medical, technical, financial) with 3,927 question-answer pairs:

| Chunker | R@1 | R@3 | R@5 |
|---|---|---|---|
| **DocuWeave** | **best** | **best** | **best** |
| LangChain Recursive (full doc) | -23.4%* | -8.4%** | — |
| Recursive (per-page) | -19.1%* | -6.2%** | — |
| Naive (fixed-size) | -28.7%* | -11.3%** | — |
| PdfPlumber | -21.3%* | -9.1%** | — |

\* p<0.01 · \*\* p<0.05 (paired t-test, bge-base-en-v1.5 embeddings)

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

# check how confident DocuWeave is about the heading structure
print(doc.hierarchy_confidence)   # 0.0–1.0; below ~0.3 means scanned/image PDF

chunks = doc.to_chunks(max_tokens=512)
doc.save_json("paper.json")
```

Each chunk looks like this:

```json
{
  "id": "c_0014",
  "text": "We train on 80% of the dataset and hold out...",
  "tokens": 487,
  "section_title": "Experimental Setup",
  "section_path": "3 Methods > 3.2 Experimental Setup",
  "section_level": 1,
  "page_start": 4,
  "page_end": 5,
  "previous_chunk_id": "c_0013",
  "next_chunk_id": "c_0015"
}
```

---

## Processing a folder

```python
from docuweave import parse_directory

docs = parse_directory(
    "pdfs/",
    pattern="**/*.pdf",
    min_confidence=0.3,   # skip scanned/image-only PDFs
    on_error="skip",      # or "raise"
    progress=True,
)

for doc in docs:
    chunks = doc.to_chunks(max_tokens=512)
    # do something with chunks
```

---

## LangChain

DocuWeave ships a proper `BaseDocumentLoader` subclass, not just a converter:

```python
from docuweave.integrations import DocuWeaveLoader

loader = DocuWeaveLoader("paper.pdf", max_tokens=512)
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

With your own embeddings:

```python
embeddings = your_model.encode([c["text"] for c in chunks]).tolist()
export_chroma(chunks, collection, embeddings=embeddings)
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

Single file:

```bash
docuweave paper.pdf -o paper.json --max-tokens 512
```

Batch mode:

```bash
docuweave --directory pdfs/ --output-dir out/ --min-confidence 0.3
```

Skip low-quality PDFs silently:

```bash
docuweave --directory pdfs/ --output-dir out/ --min-confidence 0.3 --on-error skip --no-progress
```

Vector export from CLI (prints Python snippet to wire up your client):

```bash
docuweave paper.pdf --export chroma
docuweave paper.pdf --export qdrant
docuweave paper.pdf --export faiss-jsonl -o records.jsonl
```

---

## `hierarchy_confidence`

Every parsed document has a `hierarchy_confidence` score between 0.0 and 1.0. It measures how much usable heading structure DocuWeave found:

- **≥ 0.6** — clear multi-level heading structure; chunking will be accurate
- **0.3–0.6** — partial structure; DocuWeave still does better than flat splitting
- **< 0.3** — likely scanned, image-heavy, or a single-column document with no heading signals

Use `min_confidence` in `parse_directory()` to filter these out automatically.

---

## Other useful properties

```python
doc = parse("paper.pdf")

doc.num_pages              # int
len(doc)                   # chunk count (after to_chunks())
repr(doc)                  # DocuWeaveDocument(file='paper.pdf', pages=12, sections=8, chunks=24, confidence=0.71)
doc.iter_chunks(max_tokens=512)   # iterator, same as to_chunks() but lazy
doc.to_json()              # full dict including hierarchy_confidence
```

---

## How it works

1. **Parse** — PyMuPDF extracts text blocks with font size, bold flag, and bounding box per span.
2. **Score headings** — each text block gets a score based on font size relative to the page median, bold, uppercase, length, and common heading patterns. Blocks above threshold become `HEADING` nodes.
3. **Build hierarchy** — headings are organized into a tree by font size. Paragraphs following a heading belong to that section.
4. **Clean blocks** — bullet continuations are merged, list items are grouped, noise from headers/footers is removed.
5. **Chunk** — sections are sliced into token-bounded chunks. Small chunks are merged *within* the same section only (never across section boundaries).
6. **Export** — chunks carry `section_path`, page span, and linked-list pointers so downstream retrieval can do context expansion.

---

## Known limitations

- Scanned PDFs (image-only) return no text. Check `hierarchy_confidence` and filter with `min_confidence`.
- Multi-column layouts occasionally mis-order text blocks (PyMuPDF limitation).
- Tables are treated as text blocks, not structured data.
- DOCX and HTML are not supported yet.

---

## Running tests

```bash
pip install -e ".[dev]"
python -m unittest discover tests/ -v
```

---

## Contributing

Bug reports are most useful with a minimal PDF that reproduces the issue. Open a GitHub issue and attach the file (or a public link to it).

Pull requests are welcome. The areas with the most room to improve are heading detection on noisy PDFs and table extraction.

---

## Citation

If you use DocuWeave in research, please cite:

```bibtex
@software{jannegorla2025docuweave,
  author  = {Jannegorla, Venkateswara Rao},
  title   = {{DocuWeave}: Layout-Aware PDF Chunking for RAG Pipelines},
  year    = {2025},
  url     = {https://github.com/venkateswararao18/docuweave},
}
```

---

## License

MIT — see [LICENSE](LICENSE).

**Author:** Venkateswara Rao Jannegorla · [GitHub](https://github.com/VenkateswaraRao18) · venkyjannegorla@gmail.com
