"""
pipeline.py
===========
Unified RAG pipeline — runs all steps in sequence from a single command.

Steps
-----
  1. Web extraction   web_processor.py
                      Fetches URLs from CSVs → web_extractor/outputs/*.json

  2. Web chunking     web_chunker/evaluation.py
                      Chunks web JSON files   → out/*_chunks.jsonl

  3. PDF chunking     pdf_chunker/evaluation.py
                      Chunks PDF .txt files   → out/*_chunks.jsonl

  4. Retrieval eval   embedding_retrieval/evaluation.py
                      Evaluates BGE-small-en  → outputs/comparison_summary.csv

Usage
-----
Run all steps (default):
    python src/rag/pipeline.py

Skip individual steps if already done:
    python src/rag/pipeline.py --skip-web-extract --skip-web-chunk

Custom directories:
    python src/rag/pipeline.py \\
        --web-json-dir  src/rag/web_extractor/outputs \\
        --pdf-txt-dir   src/rag/pdf_extractor/outputs \\
        --chunks-dir    ./out \\
        --target-tokens 600
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

RAG_DIR = Path(__file__).parent
PROJECT_ROOT = RAG_DIR.parent.parent


def _header(step: int, total: int, title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  Step {step}/{total}: {title}")
    print(f"{'=' * 70}")


def _run(cmd: list[str], step_name: str) -> None:
    """Run a subprocess command; exit the pipeline on failure."""
    print(f"  $ {' '.join(cmd)}\n")
    start = time.time()
    result = subprocess.run(cmd, check=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n[FAILED] {step_name} exited with code {result.returncode}.")
        sys.exit(result.returncode)

    print(f"\n  Completed in {elapsed:.1f}s")


# ── Results summary ───────────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _col_width(rows: list[list], headers: list[str]) -> list[int]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    return widths


def _print_table(title: str, headers: list[str], rows: list[list]) -> None:
    widths = _col_width(rows, headers)
    sep = "  " + "-+-".join("-" * w for w in widths)
    fmt = "  " + " | ".join(f"{{:<{w}}}" for w in widths)
    print(f"\n  {title}")
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))
    print(sep)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    web_json_dir: str,
    pdf_txt_dir: str,
    chunks_dir: str,
    target_tokens: int,
    skip_web_extract: bool,
    skip_web_chunk: bool,
    skip_pdf_chunk: bool,
    skip_eval: bool,
) -> None:

    python = sys.executable
    total_steps = 4

    print("\n" + "=" * 70)
    print("  RAG PIPELINE")
    print("=" * 70)
    print(f"  Web JSON dir  : {web_json_dir}")
    print(f"  PDF txt dir   : {pdf_txt_dir}")
    print(f"  Chunks dir    : {chunks_dir}")
    print(f"  Target tokens : {target_tokens}")
    skipped = [
        s for s, flag in [
            ("web-extract", skip_web_extract),
            ("web-chunk",   skip_web_chunk),
            ("pdf-chunk",   skip_pdf_chunk),
            ("eval",        skip_eval),
        ] if flag
    ]
    if skipped:
        print(f"  Skipping      : {', '.join(skipped)}")

    # ── Step 1: Web extraction ────────────────────────────────────────────────
    _header(1, total_steps, "Web Extraction")
    if skip_web_extract:
        print("  [SKIPPED]")
    else:
        _run(
            [python, str(RAG_DIR / "web_processor.py")],
            "Web extraction",
        )

    # ── Step 2: Web chunking ──────────────────────────────────────────────────
    _header(2, total_steps, "Web Chunking")
    if skip_web_chunk:
        print("  [SKIPPED]")
    else:
        _run(
            [
                python,
                str(RAG_DIR / "web_chunker" / "evaluation.py"),
                "--json_dir",      web_json_dir,
                "--out_dir",       chunks_dir,
                "--target_tokens", str(target_tokens),
            ],
            "Web chunking",
        )

    # ── Step 3: PDF chunking ──────────────────────────────────────────────────
    _header(3, total_steps, "PDF Chunking")
    if skip_pdf_chunk:
        print("  [SKIPPED]")
    else:
        _run(
            [
                python,
                str(RAG_DIR / "pdf_chunker" / "evaluation.py"),
                "--txt_dir", pdf_txt_dir,
                "--out_dir", chunks_dir,
                "--target_tokens", str(target_tokens),
            ],
            "PDF chunking",
        )

    # ── Step 4: Retrieval evaluation ──────────────────────────────────────────
    _header(4, total_steps, "Retrieval Evaluation")
    if skip_eval:
        print("  [SKIPPED]")
    else:
        _run(
            [
                python,
                str(RAG_DIR / "embedding_retrieval" / "evaluation.py"),
                "--chunker_output_dir", chunks_dir,
            ],
            "Retrieval evaluation",
        )

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Chunks        : {chunks_dir}/")
    print(f"  Results       : {PROJECT_ROOT}/outputs/")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full RAG pipeline (web extract → chunk → evaluate).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Directory args
    parser.add_argument(
        "--web-json-dir",
        default=str(RAG_DIR / "web_extractor" / "outputs"),
        help="Directory where web_processor.py saves *.json files",
    )
    parser.add_argument(
        "--pdf-txt-dir",
        default=str(RAG_DIR / "pdf_extractor" / "outputs"),
        help="Directory containing extracted PDF *.txt files",
    )
    parser.add_argument(
        "--chunks-dir",
        default=str(PROJECT_ROOT / "out"),
        help="Output directory for all *_chunks.jsonl files",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=600,
        help="Target chunk size in tokens",
    )

    # Skip flags
    parser.add_argument(
        "--skip-web-extract",
        action="store_true",
        help="Skip Step 1 (web extraction) — reuse existing *.json files",
    )
    parser.add_argument(
        "--skip-web-chunk",
        action="store_true",
        help="Skip Step 2 (web chunking) — reuse existing web *_chunks.jsonl files",
    )
    parser.add_argument(
        "--skip-pdf-chunk",
        action="store_true",
        help="Skip Step 3 (PDF chunking) — reuse existing PDF *_chunks.jsonl files",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip Step 4 (retrieval evaluation)",
    )

    args = parser.parse_args()

    run_pipeline(
        web_json_dir=args.web_json_dir,
        pdf_txt_dir=args.pdf_txt_dir,
        chunks_dir=args.chunks_dir,
        target_tokens=args.target_tokens,
        skip_web_extract=args.skip_web_extract,
        skip_web_chunk=args.skip_web_chunk,
        skip_pdf_chunk=args.skip_pdf_chunk,
        skip_eval=args.skip_eval,
    )


if __name__ == "__main__":
    main()