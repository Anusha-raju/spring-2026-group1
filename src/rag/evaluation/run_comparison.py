#!/usr/bin/env python3
"""
Embedding & Retrieval Comparison Pipeline
==========================================
Compares multiple embedding techniques (dense + sparse) across different k values
using ground-truth QA pairs with precision, recall, MRR, and F1 metrics.

Usage:
    python src/rag/evaluation/run_comparison.py
"""

import csv
import json
import logging
import os
import sys
import time

# Ensure parent rag/ directory is importable (for web_processor, pdf_parsing, etc.)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, RAG_DIR)
sys.path.insert(0, SCRIPT_DIR)

from tabulate import tabulate

from embedding_models import get_all_models
from retrieval_evaluator import RetrievalEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

K_VALUES = [1, 3, 5, 10]
GROUND_TRUTH_PATH = os.path.join(SCRIPT_DIR, "ground_truth.json")

# Output paths
PROJECT_ROOT = os.path.abspath(os.path.join(RAG_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DETAIL_CSV = os.path.join(OUTPUT_DIR, "comparison_results.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "comparison_summary.csv")


# ── Chunk loading ────────────────────────────────────────────────────────────

def load_chunks_from_jsonl(path: str):
    """Load chunks from the existing JSONL output file."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # The file uses Python dict repr, not JSON — use ast.literal_eval
            import ast
            record = ast.literal_eval(line)
            chunks.append(record.get("text", ""))
    return chunks


def load_chunks_from_markdown(path: str):
    """Load chunks by splitting a markdown file on heading boundaries."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on markdown headings
    import re
    sections = re.split(r"\n(?=## )", content)
    chunks = [s.strip() for s in sections if s.strip() and len(s.strip()) > 50]
    return chunks


def gather_all_chunks():
    """Collect chunks from all available sources."""
    all_chunks = []

    # Web chunks from JSONL
    jsonl_path = os.path.join(OUTPUT_DIR, "chunks_output.jsonl")
    if os.path.exists(jsonl_path):
        web_chunks = load_chunks_from_jsonl(jsonl_path)
        logger.info(f"Loaded {len(web_chunks)} web chunks from {jsonl_path}")
        all_chunks.extend(web_chunks)
    else:
        logger.warning(f"Web chunks not found at {jsonl_path}")

    # PDF chunks from markdown
    hybrid_dir = os.path.join(PROJECT_ROOT, "hybrid_output")
    if os.path.isdir(hybrid_dir):
        for fname in sorted(os.listdir(hybrid_dir)):
            if fname.endswith(".md"):
                md_path = os.path.join(hybrid_dir, fname)
                pdf_chunks = load_chunks_from_markdown(md_path)
                logger.info(f"Loaded {len(pdf_chunks)} PDF chunks from {md_path}")
                all_chunks.extend(pdf_chunks)
    else:
        logger.warning(f"Hybrid output dir not found at {hybrid_dir}")

    return all_chunks


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("  EMBEDDING & RETRIEVAL COMPARISON PIPELINE")
    print("=" * 80)

    # 1. Load chunks
    print("\n[1/4] Loading chunks...")
    chunks = gather_all_chunks()
    if not chunks:
        print("ERROR: No chunks found. Run the main RAG pipeline first.")
        sys.exit(1)
    print(f"  Total chunks: {len(chunks)}")

    # 2. Load ground truth
    print("\n[2/4] Loading ground truth queries...")
    with open(GROUND_TRUTH_PATH, "r") as f:
        ground_truth = json.load(f)
    print(f"  Total queries: {len(ground_truth)}")

    # 3. Initialize models
    print("\n[3/4] Initializing embedding models...")
    include_openai = bool(os.getenv("OPENAI_API_KEY"))
    if not include_openai:
        print("  (OPENAI_API_KEY not set — skipping OpenAI embeddings)")
    models = get_all_models(include_openai=include_openai)
    print(f"  Models: {[m.name for m in models]}")

    # 4. Evaluate
    print(f"\n[4/4] Running evaluation (k = {K_VALUES})...")
    evaluator = RetrievalEvaluator(chunks, K_VALUES)

    summary_rows = []
    detail_rows = []

    for model in models:
        print(f"\n{'─' * 60}")
        print(f"  Evaluating: {model.name}")
        print(f"{'─' * 60}")

        start = time.time()

        # Aggregated results
        averaged = evaluator.evaluate_model(model, ground_truth)
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s")

        for k, metrics in averaged.items():
            summary_rows.append({
                "model": model.name,
                "k": k,
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "mrr": round(metrics["mrr"], 4),
                "f1": round(metrics["f1"], 4),
                "time_sec": round(elapsed, 1),
            })

        # Detailed per-query results
        detail_rows.extend(evaluator.evaluate_model_detailed(model, ground_truth))

    # ── Save CSVs ────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Summary CSV
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "k", "precision", "recall", "mrr", "f1", "time_sec"]
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary saved to: {SUMMARY_CSV}")

    # Detail CSV
    with open(DETAIL_CSV, "w", newline="") as f:
        fieldnames = [
            "model", "query", "k", "precision", "recall", "mrr", "f1",
            "num_relevant_total", "top_score", "retrieved_preview",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"Details saved to: {DETAIL_CSV}")

    # ── Console summary table ────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  RESULTS SUMMARY")
    print(f"{'=' * 80}\n")

    # Build a pivot-style table: rows = models, columns = k values × metrics
    model_names = list(dict.fromkeys(r["model"] for r in summary_rows))

    # Table 1: Precision@k
    _print_metric_table("Precision@k", summary_rows, model_names, K_VALUES, "precision")
    # Table 2: Recall@k
    _print_metric_table("Recall@k", summary_rows, model_names, K_VALUES, "recall")
    # Table 3: MRR@k
    _print_metric_table("MRR@k", summary_rows, model_names, K_VALUES, "mrr")
    # Table 4: F1@k
    _print_metric_table("F1@k", summary_rows, model_names, K_VALUES, "f1")

    # Combined compact table
    print(f"\n{'─' * 80}")
    print("  Combined (P / R / F1 / MRR)")
    print(f"{'─' * 80}")
    headers = ["Model"] + [f"k={k}" for k in K_VALUES]
    rows = []
    for model_name in model_names:
        row = [model_name]
        for k in K_VALUES:
            entry = next(
                r for r in summary_rows if r["model"] == model_name and r["k"] == k
            )
            cell = (
                f"{entry['precision']:.2f} / {entry['recall']:.2f} / "
                f"{entry['f1']:.2f} / {entry['mrr']:.2f}"
            )
            row.append(cell)
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()


def _print_metric_table(
    title: str,
    summary_rows: list,
    model_names: list,
    k_values: list,
    metric_key: str,
):
    print(f"\n  {title}")
    headers = ["Model"] + [f"k={k}" for k in k_values]
    rows = []
    for model_name in model_names:
        row = [model_name]
        for k in k_values:
            entry = next(
                r for r in summary_rows if r["model"] == model_name and r["k"] == k
            )
            row.append(f"{entry[metric_key]:.4f}")
        rows.append(row)
    print(tabulate(rows, headers=headers, tablefmt="simple"))


if __name__ == "__main__":
    main()
