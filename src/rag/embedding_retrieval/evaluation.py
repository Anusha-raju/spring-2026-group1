#!/usr/bin/env python3
"""
Compares multiple embedding techniques (dense + sparse) across different k values
using ground-truth QA pairs with precision, recall, MRR, and F1 metrics.

USE_PINECONE=false (default): runs all models with numpy (full comparison mode).
USE_PINECONE=true:            uses MPNet at k=5 via Pinecone (production mode).
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import Dict, List

from dotenv import load_dotenv
load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, RAG_DIR)
sys.path.insert(0, SCRIPT_DIR)

from tabulate import tabulate

from embedding_models import get_all_models, MPNetEmbedding, BGESmallEmbedding
from retrieval_evaluator import RetrievalEvaluator

USE_PINECONE = os.getenv("USE_PINECONE", "false").strip().lower() == "true"
PRODUCTION_K = 5  # best k, used in Pinecone mode

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

K_VALUES = [1, 3, 5, 7, 10, 15]
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
        chunker_output_dir: Base directory containing chunks file

    Returns:
        List of chunk texts from all chunk files
    """
    if not os.path.isdir(chunker_output_dir):
        logger.warning(f"Chunker output directory not found: {chunker_output_dir}")
        return []

    chunks = []

    # Find all chunk files
    import glob
    chunk_files = glob.glob(os.path.join(chunker_output_dir, "*_chunks.jsonl"))

    if not chunk_files:
        logger.warning(f"No chunk files found in {chunker_output_dir}")
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
    Load full chunk records (with metadata) from chunk jsonl files.
    Falls back to wrapping plain text in {"text": ...} dicts for legacy sources.
    """
    if chunker_output_dir and os.path.isdir(chunker_output_dir):
        records = load_chunks_from_chunker_output(chunker_output_dir)
        if records:
            logger.info(f"Loaded {len(records)} chunk records from chunker output")
            return records
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
                    if isinstance(rec, str):
                        rec = {"text": rec, "topics": [], "is_tagged": False}
                    all_records.append(rec)
        logger.info(f"Loaded {len(all_records)} records from {jsonl_path}")

    return all_records


def evaluate_with_pinecone(chunks: List[Dict], ground_truth: List[Dict]) -> None:
    """
    Production evaluation path: MPNet at k=5 via Pinecone.
    Computes metrics against ground truth using Pinecone as the retriever.
    """
    from pinecone_store import build_store_from_env
    from retrieval_evaluator import is_chunk_relevant

    model = MPNetEmbedding()
    store = build_store_from_env(model, index_name="bridge-rag")

    # Build chunk_id → index map so Pinecone results can be scored
    chunk_id_to_idx = {r["chunk_id"]: i for i, r in enumerate(chunks)}
    chunk_texts = [r["text"] for r in chunks]

    print(f"\n  Upserting {len(chunks)} chunks to Pinecone (skipped if already present)...")
    store.upsert_chunks(chunks)

    print(f"\n  Evaluating MPNet k={PRODUCTION_K} via Pinecone...")
    summary_rows = []

    for query_gt in ground_truth:
        query = query_gt["query"]
        keywords = query_gt["relevant_keywords"]
        min_matches = query_gt.get("min_keyword_matches", 2)
        relevant_indices = [
            i for i, t in enumerate(chunk_texts)
            if is_chunk_relevant(t, keywords, min_matches)
        ]

        results = store.query(query, k=PRODUCTION_K)
        retrieved_indices = [
            chunk_id_to_idx[r["chunk_id"]]
            for r in results
            if r["chunk_id"] in chunk_id_to_idx
        ]

        relevant_set = set(relevant_indices)
        retrieved_set = set(retrieved_indices[:PRODUCTION_K])
        hits = relevant_set & retrieved_set

        precision = len(hits) / PRODUCTION_K if PRODUCTION_K > 0 else 0.0
        recall = len(hits) / len(relevant_set) if relevant_set else 0.0
        mrr = 0.0
        for rank, idx in enumerate(retrieved_indices[:PRODUCTION_K], start=1):
            if idx in relevant_set:
                mrr = 1.0 / rank
                break
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        summary_rows.append({
            "model": "MPNet-base-v2 (Pinecone)",
            "k": PRODUCTION_K,
            "query": query,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mrr": round(mrr, 4),
            "f1": round(f1, 4),
        })

    # Print table
    print(tabulate(
        [[r["query"][:50], r["precision"], r["recall"], r["mrr"], r["f1"]] for r in summary_rows],
        headers=["Query", "Precision", "Recall", "MRR", "F1"],
        floatfmt=".4f",
    ))

    # Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pinecone_csv = os.path.join(OUTPUT_DIR, "pinecone_summary.csv")
    with open(pinecone_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "k", "query", "precision", "recall", "mrr", "f1"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n  Results saved to: {pinecone_csv}")

    stats = store.stats()
    print(f"  Pinecone index total vectors: {stats.get('total_vector_count', 'N/A')}")


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
    if USE_PINECONE:
        print("  MODE: Pinecone (MPNet, k=5)")
    else:
        print("  MODE: Numpy (all models, all k)")
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

    # Branch: Pinecone mode (MPNet k=5 only) or numpy mode (all models)
    if USE_PINECONE:
        evaluate_with_pinecone(chunks, ground_truth)
        return

    # 3. Initialize models (numpy mode only)
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

        # Baseline (search all chunks)
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

        # Filtered (search only topic-tagged chunks)
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

        # Profession-filtered (search only profession-specific chunks)
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
    os.makedirs(OUTPUT_DIR, exist_ok=True) # save csvs

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

if __name__ == "__main__":
    main()