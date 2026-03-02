from parser import parse_pdf

blocks = parse_pdf("sample.pdf")

for b in blocks[:20]:
    print(b.type, "|", b.text, "|", b.font_size)