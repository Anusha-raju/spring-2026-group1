import logging
from typing import Optional, List
from db import DBConnection
from config import EMBEDDING_MODEL
import ollama

logger = logging.getLogger(__name__)

def get_embedding(text: str) -> List[float]:
    try:
        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=text
        )
        return response["embedding"]
    except Exception as e:
        logger.exception("Embedding generation failed")
        raise RuntimeError(f"Failed to generate embedding: {e}") from e

def format_vector(vector: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vector) + "]"

def retrieve_similar_chunks(
    db: DBConnection,
    question: str,
    top_k: int = 7,
    source_filter: Optional[str] = None,
) -> list[dict]:
    query_embedding = get_embedding(question)
    embedding_str = format_vector(query_embedding)

    if source_filter:
        sql = """
            SELECT
                id,
                source,
                chunk_index,
                content,
                metadata,
                embedding <=> %s::vector AS distance
            FROM documents
            WHERE source = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = (embedding_str, source_filter, embedding_str, top_k)
    else:
        sql = """
            SELECT
                id,
                source,
                chunk_index,
                content,
                metadata,
                embedding <=> %s::vector AS distance
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = (embedding_str, embedding_str, top_k)

    rows = db.fetch_all(sql, params)
    return rows


def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""

    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source") or "unknown"
        chunk_index = chunk.get("chunk_index")
        label = f"Source: {source}"
        if chunk_index is not None:
            label += f", Chunk: {chunk_index}"

        formatted.append(
            f"[Retrieved Chunk {i}]\n"
            f"{label}\n"
            f"{chunk['content']}"
        )

    return "\n\n".join(formatted)