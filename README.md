# DocuWeave

**RAG document compiler for PDFs: layout-aware parsing -> section hierarchy -> token-aware chunks -> vector-ready outputs.**

DocuWeave helps you convert raw PDFs into structured context that performs better in retrieval pipelines.

## Why DocuWeave

Most basic PDF loaders return flat text and lose structure. DocuWeave preserves document shape so retrieval can be more accurate and explainable.

- Layout-aware block parsing using PyMuPDF
- Automatic hierarchy construction from heading signals
- Token-aware chunking for embedding workflows
- Rich chunk metadata (`section_path`, page span, chunk links)
- Export paths for Pinecone, Weaviate, and FAISS-style JSONL
- LangChain and LlamaIndex-friendly adapters

## Installation

```bash
pip install docuweave
```

Requires Python 3.9+.

Install optional integration dependencies:

```bash
pip install "docuweave[integrations]"
```

## Quick Start (Python)

```python
from docuweave import parse

doc = parse("sample.pdf")

chunks = doc.to_chunks(max_tokens=500)
doc.save_json("output.json")

print(len(doc.get_sections()), len(chunks))
```

## Quick Start (CLI)

```bash
docuweave sample.pdf -o output.json --max-tokens 500
```

Vector export modes:

```bash
docuweave sample.pdf --export pinecone -o pinecone_records.json
docuweave sample.pdf --export weaviate -o weaviate_records.json
docuweave sample.pdf --export faiss-jsonl -o faiss_records.jsonl
```

## Output Shape

Chunks include retrieval-friendly metadata:

```json
{
  "id": "...",
  "text": "...",
  "tokens": 487,
  "section_title": "Chapter 1",
  "section_path": "Chapter 1 > Background",
  "section_level": 1,
  "page_start": 3,
  "page_end": 5,
  "previous_chunk_id": "...",
  "next_chunk_id": "..."
}
```

## Integrations

Use adapters for common orchestration stacks:

```python
langchain_docs = doc.to_langchain(max_tokens=500)
llama_nodes = doc.to_llamaindex(max_tokens=500)
```

Use vector payload exporters:

```python
pinecone_records = doc.export_pinecone()
weaviate_records = doc.export_weaviate()
doc.export_faiss_jsonl("faiss_records.jsonl")
```

## Architecture

- `docuweave/parser.py` -> layout block extraction and cleanup
- `docuweave/hierarchy.py` -> section tree construction
- `docuweave/chunking.py` -> token-aware chunk generation
- `docuweave/integrations.py` -> LangChain/LlamaIndex adapters
- `docuweave/vector_exporters.py` -> vector DB payload builders
- `docuweave/api.py` -> public API facade

## Development

```bash
git clone https://github.com/venkateswararao18/docuweave.git
cd docuweave
pip install -e .
```

Run logic-focused tests:

```bash
python -m unittest tests/test_core_logic.py tests/test_integrations.py -v
```

## Roadmap

- DOCX and HTML support
- More robust heading detection across noisy PDFs
- Table extraction improvements
- Optional semantic chunking mode
- Benchmark suite and quality report

## Contributing

Pull requests are welcome. Open an issue with a PDF sample when reporting parsing bugs.

## License

MIT