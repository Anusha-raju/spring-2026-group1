"""
One-time script to upsert all chunks into Pinecone and verify the index.
Requires PINECONE_API_KEY in .env or environment.
"""

import json
import os
import sys
import glob

from dotenv import load_dotenv
load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from embedding_models import MPNetEmbedding
from pinecone_store import build_store_from_env

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
CHUNKS_DIR = os.path.join(PROJECT_ROOT, "out")


def load_all_chunks(chunks_dir: str):
    records = []
    for path in sorted(glob.glob(os.path.join(chunks_dir, "*_chunks.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def main():
    print("=" * 60)
    print("  BRIDGE — Pinecone Upsert")
    print("=" * 60)

    # 1. Load chunks
    print(f"\n[1/3] Loading chunks from {CHUNKS_DIR} ...")
    chunks = load_all_chunks(CHUNKS_DIR)
    if not chunks:
        print("ERROR: No chunks found. Run the pipeline first.")
        sys.exit(1)
    print(f"  Loaded {len(chunks)} chunks")

    # 2. Connect to Pinecone and upsert
    print("\n[2/3] Connecting to Pinecone and upserting ...")
    model = MPNetEmbedding()
    store = build_store_from_env(model, index_name="bridge-rag")
    store.upsert_chunks(chunks)

    # 3. Verify with a test query
    print("\n[3/3] Verifying with a test query ...")
    test_query = "How do I administer naloxone for an overdose?"

    print(f"\n  Query: '{test_query}'")
    print("\n  --- Full corpus (top 3) ---")
    results = store.query(test_query, k=3)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] score={r['score']:.4f} | {r['doc_id']} | {r['text'][:120].replace(chr(10), ' ')}...")

    print("\n  --- Nurse-filtered (top 3) ---")
    results = store.query_by_profession(test_query, profession="Nurse", k=3)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] score={r['score']:.4f} | {r['doc_id']} | {r['text'][:120].replace(chr(10), ' ')}...")

    print("\n  --- PA-filtered (top 3) ---")
    results = store.query_by_profession(test_query, profession="PA", k=3)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] score={r['score']:.4f} | {r['doc_id']} | {r['text'][:120].replace(chr(10), ' ')}...")

    print("\n  Index stats:")
    stats = store.stats()
    print(f"  Total vectors: {stats.get('total_vector_count', 'N/A')}")
    print(f"  Dimension:     {stats.get('dimension', 'N/A')}")

    print("\nDone. Pinecone index is ready for production use.")


if __name__ == "__main__":
    main()
