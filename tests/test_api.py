from api import parse

doc = parse("sample.pdf")

print("Total sections:", len(doc.get_sections()))

chunks = doc.to_chunks(max_tokens=500)
print("Total chunks:", len(chunks))

exported = doc.to_json()
print("Export keys:", exported.keys())