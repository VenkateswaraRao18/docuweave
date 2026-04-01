from __future__ import annotations

import argparse
import json
from pathlib import Path

from docuweave.api import parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docuweave",
        description="Parse PDFs into structured, RAG-ready JSON.",
    )
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument(
        "-o",
        "--output",
        default="docuweave_output.json",
        help="Path for JSON output",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Maximum tokens per chunk",
    )
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="Tokenizer model name used for token counting",
    )
    parser.add_argument(
        "--export",
        choices=["json", "pinecone", "weaviate", "faiss-jsonl"],
        default="json",
        help="Output format for downstream vector workflows",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_pdf)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    document = parse(str(input_path))
    document.to_chunks(max_tokens=args.max_tokens, model_name=args.model)
    payload = document.to_json()

    output_path = Path(args.output)
    if args.export == "json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    elif args.export == "pinecone":
        records = document.export_pinecone(max_tokens=args.max_tokens, model_name=args.model)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    elif args.export == "weaviate":
        records = document.export_weaviate(max_tokens=args.max_tokens, model_name=args.model)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    else:
        document.export_faiss_jsonl(
            output_path=str(output_path),
            max_tokens=args.max_tokens,
            model_name=args.model,
        )

    print(f"Saved {args.export} output to {output_path}")
    print(f"Sections: {len(payload.get('sections', []))}")
    print(f"Chunks: {len(payload.get('chunks', []))}")


if __name__ == "__main__":
    main()
