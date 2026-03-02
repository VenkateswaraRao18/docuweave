from parser import parse_pdf
from hierarchy import build_hierarchy


def print_section(section, indent=0):
    prefix = "  " * indent
    print(f"{prefix}- {section.title} (Level {section.level})")
    print(f"{prefix}  Blocks: {len(section.blocks)}")

    for subsection in section.subsections:
        print_section(subsection, indent + 1)


blocks = parse_pdf("sample.pdf")
sections = build_hierarchy(blocks)

print("\nDocument Structure:\n")

for sec in sections:
    print_section(sec)