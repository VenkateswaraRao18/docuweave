from docuweave.parser import parse_pdf


def main():

    pdf_path = "thebook.pdf"   # change if needed

    blocks = parse_pdf(pdf_path)

    print("\nParsed Blocks:\n")

    for b in blocks[:80]:

        # Handle list blocks
        if b.type == "list":
            print(f"\nLIST ({len(b.items)} items):")

            for item in b.items:
                print("   •", item[:120])

            continue

        # Normal blocks
        text = (b.text or "").replace("\n", " ")

        print(
            f"{b.type:10} | "
            f"font={round(b.font_size,1) if b.font_size else None} | "
            f"page={b.page} | "
            f"{text[:120]}"
        )


if __name__ == "__main__":
    main()