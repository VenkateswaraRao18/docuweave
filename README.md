# DocuWeave

**Layout-aware document parser that converts PDFs into structured, hierarchical, LLM-ready JSON.**

DocuWeave is designed specifically for Retrieval-Augmented Generation (RAG) pipelines.  
Unlike simple PDF text extractors, it preserves layout structure, builds semantic hierarchy, and produces token-aware chunks optimized for embeddings.

---

## 🚀 Why DocuWeave?

Most PDF loaders:
- Return raw text blobs
- Ignore layout
- Break section boundaries
- Produce poor RAG chunks

DocuWeave provides:

✅ Deterministic layout-aware parsing  
✅ Automatic section hierarchy detection  
✅ Token-aware smart chunking  
✅ Embedding-ready JSON output  
✅ Clean Python API  
✅ Lightweight & dependency-minimal  

---

## 📦 Installation

```bash
pip install docuweave
```

Requires Python 3.9+

---

## ⚡ Quick Start

```python
from docuweave import parse

doc = parse("sample.pdf")

# Generate RAG-ready chunks
chunks = doc.to_chunks(max_tokens=500)

# Save structured JSON file
doc.save_json("output.json")
```

---

## 🧠 What Makes It Different?

DocuWeave follows a deterministic pipeline:

```
PDF → Layout Blocks → Hierarchy → Token-Aware Chunks → Structured JSON
```

It preserves:

- Section structure
- Heading levels
- Page numbers
- Layout metadata
- Token counts per chunk

---

## 📄 Output JSON Structure

```json
{
  "metadata": {
    "source": "sample.pdf",
    "total_pages": 120
  },
  "sections": [
    {
      "id": "...",
      "title": "Chapter 1",
      "level": 1,
      "blocks": [...],
      "subsections": [...]
    }
  ],
  "chunks": [
    {
      "id": "...",
      "text": "...",
      "tokens": 487,
      "section_title": "Chapter 1",
      "section_level": 1,
      "page_start": 3,
      "page_end": 5
    }
  ]
}
```

---

## 🔥 Designed for RAG Pipelines

DocuWeave optimizes for:

- Vector database ingestion
- Embedding generation
- Section-aware retrieval
- Metadata filtering
- Explainable chunk origins

Works well with:

- OpenAI embeddings
- HuggingFace models
- Pinecone
- Weaviate
- FAISS
- Chroma

---

## 🏗 Architecture Overview

DocuWeave follows a clean modular design:

- `parser` → Layout extraction using PyMuPDF  
- `hierarchy` → Font-size-based section tree builder  
- `chunking` → Token-aware section-based chunk generator  
- `exporter` → Structured JSON export  
- `api` → Clean public interface  

Deterministic first.  
AI enrichment can be added later.

---

## 🛠 Advanced Usage

Custom token limit:

```python
doc.to_chunks(max_tokens=800)
```

Access hierarchy:

```python
sections = doc.get_sections()
```

Access flat blocks:

```python
blocks = doc.get_blocks()
```

---

## 🧪 Development Setup

Clone repository:

```bash
git clone https://github.com/venkateswararao18/docuweave.git
cd docuweave
pip install -e .
```

Run tests manually:

```bash
python tests/test_api.py
```

---

## 📌 Roadmap

Planned features:

- DOCX support
- HTML support
- Table extraction improvements
- Section path identifiers
- CLI tool
- Optional AI-enhanced semantic mode
- Improved heading detection robustness

---

## 📄 License

MIT License

---

## 👤 Author

venkateswararao jannegorla 
GitHub: https://github.com/venkateswararao18

---

## ⭐ Contributing

Pull requests welcome.  
If you find a bug or improvement idea, open an issue.

---

DocuWeave — Structured Documents for LLMs.