from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from docuweave.api import parse, parse_directory


EXPORT_CHOICES = ["json", "pinecone", "weaviate", "chroma", "qdrant", "faiss-jsonl"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="docuweave",
        description="Parse PDFs into structured, RAG-ready JSON with layout-aware chunking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file → JSON
  docuweave paper.pdf -o paper_chunks.json

  # Whole directory → one JSON per PDF
  docuweave --directory ./papers/ --output-dir ./chunks/

  # Export to Pinecone format
  docuweave paper.pdf --export pinecone -o pinecone_records.json

  # Skip low-confidence (scanned) PDFs when processing a directory
  docuweave --directory ./docs/ --min-confidence 0.2
        """,
    )

    # ── Input ──────────────────────────────────────────────────
    input_grp = p.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("input_pdf", nargs="?", help="Path to a single PDF file")
    input_grp.add_argument(
        "--directory", "-d",
        metavar="DIR",
        help="Directory to scan for PDFs (recursive)",
    )

    # ── Output ─────────────────────────────────────────────────
    p.add_argument("-o", "--output", default=None,
                   help="Output file path (single-file mode). Default: <stem>_chunks.json")
    p.add_argument("--output-dir", metavar="DIR", default=".",
                   help="Output directory for batch mode (default: current directory)")

    # ── Chunking ───────────────────────────────────────────────
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Max tokens per chunk (default: 512)")
    p.add_argument("--model", default="gpt-4",
                   help="Tokenizer model for token counting (default: gpt-4)")

    # ── Export format ──────────────────────────────────────────
    p.add_argument("--export", choices=EXPORT_CHOICES, default="json",
                   help="Output format (default: json)")

    # ── Quality filter ─────────────────────────────────────────
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="Skip PDFs with hierarchy_confidence below this (0–1, default: 0)")

    # ── Misc ───────────────────────────────────────────────────
    p.add_argument("--no-progress", action="store_true",
                   help="Suppress progress output in batch mode")
    p.add_argument("--on-error", choices=["warn", "skip", "raise"], default="warn",
                   help="Error handling for batch mode (default: warn)")

    return p


def _write(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✓  {path}")


def _process_one(doc, args, output_path: Path) -> None:
    chunks = doc.to_chunks(max_tokens=args.max_tokens, model_name=args.model)

    if args.export == "json":
        _write(doc.to_json(), output_path)

    elif args.export == "pinecone":
        _write(doc.to_pinecone(max_tokens=args.max_tokens, model_name=args.model), output_path)

    elif args.export == "weaviate":
        _write(doc.to_weaviate(max_tokens=args.max_tokens, model_name=args.model), output_path)

    elif args.export == "faiss-jsonl":
        from docuweave.vector_exporters import export_faiss_jsonl
        export_faiss_jsonl(chunks, str(output_path))
        print(f"✓  {output_path}")

    elif args.export in ("chroma", "qdrant"):
        # These require a live client — explain what the user needs to do
        print(
            f"[docuweave] '{args.export}' export requires a running {args.export} client.\n"
            "Use the Python API instead:\n\n"
            "  from docuweave import parse\n"
            f"  doc = parse('{doc.file_path}')\n"
            f"  doc.to_{args.export}(client, ...)\n"
        )
        sys.exit(1)

    payload = doc.to_json()
    confidence = payload.get("hierarchy_confidence", "?")
    print(f"   sections={len(payload.get('sections', []))}  "
          f"chunks={len(payload.get('chunks', []))}  "
          f"confidence={confidence}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Single file ────────────────────────────────────────────
    if args.input_pdf:
        path = Path(args.input_pdf)
        if not path.exists():
            raise SystemExit(f"File not found: {path}")

        doc = parse(str(path))

        if doc.hierarchy_confidence < args.min_confidence:
            print(
                f"[skip] '{path.name}' confidence={doc.hierarchy_confidence:.2f} "
                f"< --min-confidence={args.min_confidence}"
            )
            return

        out = Path(args.output) if args.output else path.with_name(path.stem + "_chunks.json")
        _process_one(doc, args, out)

    # ── Batch directory ────────────────────────────────────────
    else:
        docs = parse_directory(
            args.directory,
            on_error=args.on_error,
            min_confidence=args.min_confidence,
            progress=not args.no_progress,
        )
        out_dir = Path(args.output_dir)
        for doc in docs:
            stem = Path(doc.file_path).stem
            ext  = ".jsonl" if args.export == "faiss-jsonl" else ".json"
            out  = out_dir / (stem + "_chunks" + ext)
            try:
                _process_one(doc, args, out)
            except Exception as exc:
                if args.on_error == "raise":
                    raise
                print(f"[error] {Path(doc.file_path).name}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
