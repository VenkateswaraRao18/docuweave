from docuweave.parser import parse_pdf
from docuweave.hierarchy import build_hierarchy
from docuweave.chunking import build_chunks

blocks = parse_pdf("sample.pdf")
sections = build_hierarchy(blocks)

chunks = build_chunks(sections, max_tokens=500)

print(f"\nTotal chunks: {len(chunks)}\n")

for c in chunks[:3]:
    print("Chunk ID:", c["id"])
    print("Section:", c["section_title"])
    print("Tokens:", c["tokens"])
    print("Pages:", c["page_start"], "-", c["page_end"])
    print("Preview:", c["text"][:200])
    print("-" * 60)