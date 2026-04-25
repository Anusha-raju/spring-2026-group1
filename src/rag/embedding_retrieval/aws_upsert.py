import os, glob, json
from pathlib import Path
import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import torch

# ====== SET THESE ======
CHUNKS_DIR = str(Path(__file__).resolve().parents[3] / "out")
DB_HOST = "XXXXX"  # your RDS endpoint, e.g. mydb.abc123xyz.us-east-1.rds.amazonaws.com
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "XXXXX"  # your RDS username
DB_PASSWORD = "XXXXX"  # your RDS password
SSLMODE = "require"   # use "verify-full" only if cert file is configured
TABLE = "bridge_chunks"

# 1) Load all chunk records from the pipeline output directory
if not os.path.isdir(CHUNKS_DIR):
    raise RuntimeError(f"Chunks directory not found: {CHUNKS_DIR}. Run the pipeline first.")
paths = sorted(glob.glob(os.path.join(CHUNKS_DIR, "*_chunks.jsonl")))
if not paths:
    raise RuntimeError(f"No *_chunks.jsonl files found in {CHUNKS_DIR}")

records = []
for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

print(f"Loaded {len(records)} chunks from {len(paths)} files")

# 2) Build embeddings with bge-m3 on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
model = SentenceTransformer("BAAI/bge-m3", device=device)
model.max_seq_length = 8192
texts = [r.get("text", "") for r in records]

emb = model.encode(texts, show_progress_bar=True, batch_size=4)
emb = np.asarray(emb, dtype=np.float32)
emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
dim = emb.shape[1]
print("Embedding dim:", dim)

# 3) Connect to Postgres + create schema
conn = psycopg2.connect(
    host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, sslmode=SSLMODE
)
register_vector(conn)
conn.autocommit = False

with conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            chunk_id   TEXT PRIMARY KEY,
            doc_id     TEXT,
            source     TEXT,
            text       TEXT,
            topics     TEXT[],
            categories TEXT[],
            is_tagged  BOOLEAN,
            embedding  VECTOR({dim})
        );
    """)
    cur.execute(f"""
        CREATE INDEX IF NOT EXISTS {TABLE}_embedding_hnsw
        ON {TABLE}
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    """)
conn.commit()

# 4) Upsert in batches
rows = []
for r, v in zip(records, emb):
    rows.append((
        r["chunk_id"],
        r.get("doc_id", ""),
        r.get("source", ""),
        r.get("text", ""),
        r.get("topics", []),
        r.get("categories", []),
        bool(r.get("is_tagged", False)),
        v.tolist(),
    ))

sql = f"""
INSERT INTO {TABLE}
(chunk_id, doc_id, source, text, topics, categories, is_tagged, embedding)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (chunk_id) DO UPDATE SET
  doc_id=EXCLUDED.doc_id,
  source=EXCLUDED.source,
  text=EXCLUDED.text,
  topics=EXCLUDED.topics,
  categories=EXCLUDED.categories,
  is_tagged=EXCLUDED.is_tagged,
  embedding=EXCLUDED.embedding;
"""

BATCH = 100
try:
    with conn.cursor() as cur:
        for i in range(0, len(rows), BATCH):
            psycopg2.extras.execute_batch(cur, sql, rows[i:i+BATCH], page_size=BATCH)
            print(f"Upserted {min(i+BATCH, len(rows))}/{len(rows)}")
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE};")
        total = cur.fetchone()[0]
    print("Done. Total rows in table:", total)
finally:
    conn.close()