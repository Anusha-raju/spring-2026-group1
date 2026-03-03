"""
Handles upserting chunk embeddings to a Pinecone index and querying with
optional metadata filters (profession, topic).  This is the production
retriever — the numpy in-memory search in retrieval_evaluator.py is used
for offline evaluation only.
"""

from __future__ import annotations

import logging
import os
import unicodedata
from typing import Any, Dict, List, Optional

import numpy as np

from embedding_models import EmbeddingModel

logger = logging.getLogger(__name__)

# Pinecone free tier supports up to 1536 dimensions; instructor-xl outputs 768.
# Adjust UPSERT_BATCH_SIZE if you hit rate limits.
UPSERT_BATCH_SIZE = 100


def _ascii_id(chunk_id: str) -> str:
    """Return an ASCII-safe version of chunk_id for Pinecone vector IDs.

    Pinecone requires vector IDs to be ASCII-only. This normalises unicode
    (e.g. em-dashes, accented letters) via NFKD decomposition, then drops
    any remaining non-ASCII bytes.  The resulting ID stays unique because
    the numeric suffix (::sentence_pack::N) is always ASCII.
    """
    normalised = unicodedata.normalize("NFKD", chunk_id)
    return normalised.encode("ascii", "ignore").decode("ascii")

class PineconeVectorStore:
    """
    Wraps a Pinecone index for BRIDGE chunk storage and retrieval.

    Each vector is stored with metadata:
        - chunk_id   : unique string ID
        - doc_id     : source filename
        - source     : "pdf" or "website"
        - text       : full chunk text (stored for retrieval without a separate DB)
        - topics     : list of opioid topic tags
        - categories : list of profession labels
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        embedding_model: EmbeddingModel,
        dimension: Optional[int] = None,
    ):
        """
        Args:
            api_key:         Pinecone API key (or set PINECONE_API_KEY env var)
            index_name:      Name of the Pinecone index to use / create
            embedding_model: EmbeddingModel instance used to encode chunks and queries
            dimension:       Embedding dimension. Auto-detected if None.
        """
        from pinecone import Pinecone, ServerlessSpec

        self._model = embedding_model
        self._index_name = index_name

        pc = Pinecone(api_key=api_key)

        # Auto-detect dimension by encoding a dummy string
        if dimension is None:
            sample = self._model.encode(["dimension probe"])
            dimension = int(sample.shape[1]) if sample.ndim == 2 else int(sample.shape[0])
            logger.info(f"Auto-detected embedding dimension: {dimension}")

        # Create index if it doesn't exist
        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            logger.info(f"Creating Pinecone index '{index_name}' (dim={dimension}, cosine)...")
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info("Index created.")
        else:
            logger.info(f"Using existing Pinecone index '{index_name}'.")

        self._index = pc.Index(index_name)

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunk_records: List[Dict[str, Any]]) -> None:
        """
        Encode all chunks and upsert them into Pinecone with full metadata.

        Args:
            chunk_records: List of chunk dicts as produced by the chunker pipeline.
                           Each dict must have at least: chunk_id, text.
        """
        texts = [r["text"] for r in chunk_records]
        logger.info(f"Encoding {len(texts)} chunks with {self._model.name}...")
        embeddings = self._model.encode(texts)

        # Normalise to unit vectors for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        embeddings = embeddings / norms

        vectors = []
        for record, emb in zip(chunk_records, embeddings):
            vectors.append({
                "id": _ascii_id(record["chunk_id"]),
                "values": emb.tolist(),
                "metadata": {
                    "text":       record.get("text", ""),
                    "doc_id":     record.get("doc_id", ""),
                    "source":     record.get("source", ""),
                    "topics":     record.get("topics", []),
                    "categories": record.get("categories", []),
                    "is_tagged":  record.get("is_tagged", False),
                },
            })

        # Upsert in batches to stay within Pinecone request size limits
        total = len(vectors)
        for i in range(0, total, UPSERT_BATCH_SIZE):
            batch = vectors[i : i + UPSERT_BATCH_SIZE]
            self._index.upsert(vectors=batch)
            logger.info(f"  Upserted {min(i + UPSERT_BATCH_SIZE, total)}/{total} vectors")

        logger.info(f"Upsert complete. {total} chunks in index '{self._index_name}'.")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Encode a query and retrieve the top-k most similar chunks.

        Args:
            query_text: The user's natural-language query.
            k:          Number of results to return.
            filter:     Optional Pinecone metadata filter dict.
                        Example: {"categories": {"$in": ["Nurse"]}}

        Returns:
            List of dicts with keys: chunk_id, score, text, doc_id, source,
            topics, categories.
        """
        query_emb = self._model.encode_query(query_text)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-10)

        response = self._index.query(
            vector=query_emb.tolist(),
            top_k=k,
            include_metadata=True,
            filter=filter,
        )

        results = []
        for match in response.matches:
            meta = match.metadata or {}
            results.append({
                "chunk_id":   match.id,
                "score":      match.score,
                "text":       meta.get("text", ""),
                "doc_id":     meta.get("doc_id", ""),
                "source":     meta.get("source", ""),
                "topics":     meta.get("topics", []),
                "categories": meta.get("categories", []),
            })
        return results

    def query_by_profession(
        self, query_text: str, profession: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks filtered to a specific BRIDGE profession.

        Args:
            query_text: The user's query.
            profession: One of "Nurse", "PA", "Public_Health", "Social_Work",
                        "Physical_Therapist", "Health_Administrator".
            k:          Number of results to return.
        """
        return self.query(
            query_text,
            k=k,
            filter={"categories": {"$in": [profession]}},
        )

    def query_by_topic(
        self, query_text: str, topic: str, k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks filtered to a specific opioid topic tag.

        Args: topic
        """
        return self.query(
            query_text,
            k=k,
            filter={"topics": {"$in": [topic]}},
        )


    def delete_all(self) -> None:
        """Delete all vectors from the index (useful for re-indexing)."""
        self._index.delete(delete_all=True)
        logger.info(f"Deleted all vectors from index '{self._index_name}'.")

    def stats(self) -> Dict[str, Any]:
        """Return index stats (total vector count, dimension, etc.)."""
        return self._index.describe_index_stats()


def build_store_from_env(
    embedding_model: EmbeddingModel,
    index_name: str = "bridge-rag",
) -> PineconeVectorStore:
    """
    Convenience factory that reads PINECONE_API_KEY from environment.

    Args:
        embedding_model: The model to use for encoding.
        index_name:      Pinecone index name (default: "bridge-rag").
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable is not set.")
    return PineconeVectorStore(
        api_key=api_key,
        index_name=index_name,
        embedding_model=embedding_model,
    )
