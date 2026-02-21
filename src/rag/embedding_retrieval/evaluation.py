#!/usr/bin/env python3
"""
Embedding & Retrieval Evaluation
=================================
Compares multiple embedding techniques (dense + sparse) across different k values
using ground-truth QA pairs with precision, recall, MRR, and F1 metrics.

Usage:
    python src/rag/embedding_retrieval/evaluation.py [--chunker_output_dir DIR]
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import Dict, List

# Ensure parent rag/ directory is importable (for web_processor, pdf_parsing, etc.)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, RAG_DIR)
sys.path.insert(0, SCRIPT_DIR)

from tabulate import tabulate

from embedding_models import get_all_models, BGESmallEmbedding
from retrieval_evaluator import RetrievalEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

K_VALUES = [1, 3, 5, 10, 15]
GROUND_TRUTH_PATH = os.path.join(SCRIPT_DIR, "ground_truth.json")

# Output paths
PROJECT_ROOT = os.path.abspath(os.path.join(RAG_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DETAIL_CSV = os.path.join(OUTPUT_DIR, "comparison_results.csv")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
FILTER_CSV = os.path.join(OUTPUT_DIR, "metadata_filter_comparison.csv")
PROFESSION_CSV = os.path.join(OUTPUT_DIR, "profession_filter_comparison.csv")

# Chunker output path
DEFAULT_CHUNKER_OUTPUT = os.path.join(PROJECT_ROOT, "out")


# ── Chunk loading ────────────────────────────────────────────────────────────

def load_chunks_from_chunker_output(chunker_output_dir: str):
    """
    Load chunks from pdf_chunker evaluation output.

    Args:
        chunker_output_dir: Base directory containing *_chunks.jsonl files (e.g., ./out)

    Returns:
        List of chunk texts from all *_chunks.jsonl files
    """
    if not os.path.isdir(chunker_output_dir):
        logger.warning(f"Chunker output directory not found: {chunker_output_dir}")
        return []

    chunks = []

    # Find all *_chunks.jsonl files
    import glob
    chunk_files = glob.glob(os.path.join(chunker_output_dir, "*_chunks.jsonl"))

    if not chunk_files:
        logger.warning(f"No *_chunks.jsonl files found in {chunker_output_dir}")
        return []

    for chunks_file in sorted(chunk_files):
        file_chunks = _load_chunks_jsonl(chunks_file)
        filename = os.path.basename(chunks_file)
        logger.info(f"Loaded {len(file_chunks)} chunks from {filename}")
        chunks.extend(file_chunks)

    return chunks


def _load_chunks_jsonl(jsonl_path: str) -> List[Dict]:
    """Load full chunk records from a JSONL file (preserves all metadata)."""
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                import ast
                records.append(ast.literal_eval(line))
    return records


def gather_all_chunks(chunker_output_dir: str = None) -> List[Dict]:
    """
    Load full chunk records (with metadata) from *_chunks.jsonl files.
    Falls back to wrapping plain text in {"text": ...} dicts for legacy sources.
    """
    if chunker_output_dir and os.path.isdir(chunker_output_dir):
        records = load_chunks_from_chunker_output(chunker_output_dir)
        if records:
            logger.info(f"Loaded {len(records)} chunk records from chunker output")
            return records

    # Fallback: legacy JSONL without metadata
    logger.info("Loading from legacy sources...")
    all_records = []

    jsonl_path = os.path.join(OUTPUT_DIR, "chunks_output.jsonl")
    if os.path.exists(jsonl_path):
        import ast
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        rec = ast.literal_eval(line)
                    # Wrap plain text as record if needed
                    if isinstance(rec, str):
                        rec = {"text": rec, "topics": [], "is_tagged": False}
                    all_records.append(rec)
        logger.info(f"Loaded {len(all_records)} records from {jsonl_path}")

    return all_records


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate embedding models on chunked data"
    )
    parser.add_argument(
        "--chunker_output_dir",
        default=DEFAULT_CHUNKER_OUTPUT,
        help="Directory containing *_chunks.jsonl files from pdf_chunker evaluation (default: ./out)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  EMBEDDING & RETRIEVAL COMPARISON PIPELINE")
    print("=" * 80)

    # 1. Load chunks
    print("\n[1/4] Loading chunks...")
    print(f"  Chunker output directory: {args.chunker_output_dir}")

    chunks = gather_all_chunks(args.chunker_output_dir)
    if not chunks:
        print("ERROR: No chunks found. Run the pdf_chunker evaluation first.")
        sys.exit(1)
    tagged = sum(1 for r in chunks if isinstance(r, dict) and r.get("is_tagged"))
    print(f"  Total chunks: {len(chunks)} ({tagged} tagged with topics, {len(chunks)-tagged} untagged)")

    # 2. Load ground truth
    print("\n[2/4] Loading ground truth queries...")
    with open(GROUND_TRUTH_PATH, "r") as f:
        ground_truth = json.load(f)
    print(f"  Total queries: {len(ground_truth)}")

    # 3. Initialize models
    print("\n[3/4] Initializing embedding models...")
    models = get_all_models(include_openai=False)
    print(f"  Models: {', '.join(m.name for m in models)}")

    # 4. Evaluate
    print(f"\n[4/4] Running evaluation (k = {K_VALUES})...")
    evaluator = RetrievalEvaluator(chunks, K_VALUES)

    summary_rows = []
    detail_rows = []
    filter_rows = []
    profession_rows = []

    for model in models:
        print(f"\n{'─' * 60}")
        print(f"  Evaluating: {model.name}")
        print(f"{'─' * 60}")

        # ── Baseline: search all chunks ───────────────────────────────
        start = time.time()
        averaged = evaluator.evaluate_model(model, ground_truth)
        elapsed = time.time() - start
        print(f"  Baseline done in {elapsed:.1f}s")

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
            filter_rows.append({
                "model": model.name,
                "k": k,
                "mode": "baseline",
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "mrr": round(metrics["mrr"], 4),
                "f1": round(metrics["f1"], 4),
            })

        # ── Filtered: search only topic-tagged chunks ─────────────────
        start = time.time()
        filtered = evaluator.evaluate_model_filtered(model, ground_truth)
        elapsed_f = time.time() - start
        print(f"  Filtered done in {elapsed_f:.1f}s")

        for k, metrics in filtered.items():
            filter_rows.append({
                "model": model.name,
                "k": k,
                "mode": "topic_filtered",
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "mrr": round(metrics["mrr"], 4),
                "f1": round(metrics["f1"], 4),
            })

        # ── Profession-filtered: search only profession-specific chunks ───
        start = time.time()
        prof_filtered = evaluator.evaluate_model_profession_filtered(model, ground_truth)
        elapsed_p = time.time() - start
        print(f"  Profession-filtered done in {elapsed_p:.1f}s")

        for k, metrics in prof_filtered.items():
            profession_rows.append({
                "model": model.name,
                "k": k,
                "mode": "profession_filtered",
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "mrr": round(metrics["mrr"], 4),
                "f1": round(metrics["f1"], 4),
            })

        # Detailed per-query results (baseline)
        detail_rows.extend(evaluator.evaluate_model_detailed(model, ground_truth))

    # ── Save CSVs ────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Summary CSV (baseline)
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
            "model", "query", "topic", "k", "mode", "precision", "recall",
            "mrr", "f1", "num_relevant_total", "top_score", "retrieved_preview",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"Details saved to: {DETAIL_CSV}")

    # Metadata filter comparison CSV
    with open(FILTER_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "k", "mode", "precision", "recall", "mrr", "f1"]
        )
        writer.writeheader()
        writer.writerows(filter_rows)
    print(f"Filter comparison saved to: {FILTER_CSV}")

    # Profession filter comparison CSV
    with open(PROFESSION_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "k", "mode", "precision", "recall", "mrr", "f1"]
        )
        writer.writeheader()
        writer.writerows(profession_rows)
    print(f"Profession filter comparison saved to: {PROFESSION_CSV}")

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

    # Metadata filter comparison: baseline vs topic_filtered
    print(f"\n{'─' * 80}")
    print("  Metadata Filter Comparison — Baseline vs Topic-Filtered (F1)")
    print(f"{'─' * 80}")
    headers = ["Model", "k"] + ["Baseline F1", "Filtered F1", "Delta"]
    frows = []
    for model_name in model_names:
        for k in K_VALUES:
            base = next((r for r in filter_rows if r["model"] == model_name and r["k"] == k and r["mode"] == "baseline"), None)
            filt = next((r for r in filter_rows if r["model"] == model_name and r["k"] == k and r["mode"] == "topic_filtered"), None)
            if base and filt:
                delta = round(filt["f1"] - base["f1"], 4)
                sign = "+" if delta >= 0 else ""
                frows.append([model_name, k, base["f1"], filt["f1"], f"{sign}{delta}"])
    print(tabulate(frows, headers=headers, tablefmt="simple"))

    # Profession filter comparison: baseline vs profession_filtered
    print(f"\n{'─' * 80}")
    print("  Profession Filter Comparison — Baseline vs Profession-Filtered (F1)")
    print(f"{'─' * 80}")
    headers = ["Model", "k"] + ["Baseline F1", "Profession F1", "Delta"]
    prows = []
    for model_name in model_names:
        for k in K_VALUES:
            base = next((r for r in filter_rows if r["model"] == model_name and r["k"] == k and r["mode"] == "baseline"), None)
            prof = next((r for r in profession_rows if r["model"] == model_name and r["k"] == k and r["mode"] == "profession_filtered"), None)
            if base and prof:
                delta = round(prof["f1"] - base["f1"], 4)
                sign = "+" if delta >= 0 else ""
                prows.append([model_name, k, base["f1"], prof["f1"], f"{sign}{delta}"])
    print(tabulate(prows, headers=headers, tablefmt="simple"))

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