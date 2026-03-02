from parser import parse_pdf
from hierarchy import build_hierarchy
from chunking import build_chunks
from models import Document, DocumentMetadata
from exporter import export_document

blocks = parse_pdf("sample.pdf")
sections = build_hierarchy(blocks)
chunks = build_chunks(sections, max_tokens=500)

doc = Document(
    metadata=DocumentMetadata(
        source="sample.pdf",
        total_pages=max(b.page for b in blocks),
    ),
    blocks=blocks,
    sections=sections,
)

exported = export_document(doc, chunks=chunks)

print("Keys:", exported.keys())
print("Total sections:", len(exported["sections"]))
print("Total chunks:", len(exported["chunks"]))