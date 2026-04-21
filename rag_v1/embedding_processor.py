import os
from typing import List, Dict, Optional
from langchain_core.documents import Document
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """Generate embeddings and store in vector database"""

    def __init__(
        self,
        embedding_provider: str = "sentence_transformers",
        vector_db: str = "chromadb",
        collection_name: str = "opioid_documents",
        db_path: str = "./pgvector",
        pg_dsn: str | None = None
    ):
        """
        Initialize embedding processor

        Args:
            embedding_provider: "sentence_transformers"
            vector_db: "pgvector"
            collection_name: Name of the collection/index
            db_path: Path for local database (ChromaDB only)
            pg_dsn: Postgres DSN for pgvector (e.g., RDS)
        """
        self.embedding_provider = embedding_provider
        self.vector_db = vector_db
        self.collection_name = collection_name

        # Initialize embedding model
        self._init_embedding_model()

        # Initialize vector database
        self._init_vector_db(db_path=db_path, pg_dsn=pg_dsn)

    def _init_embedding_model(self):
        """Initialize the embedding model"""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        logger.info("Initialized Sentence Transformers: all-MiniLM-L6-v2")

    def _init_vector_db(self, db_path: str, pg_dsn: str | None = None):
        """Initialize the vector database"""
        if self.vector_db == "pgvector":
            try:
                import psycopg2
                from pgvector.psycopg2 import register_vector
            except ImportError as e:
                raise ImportError(
                    "pgvector backend requires psycopg2-binary and pgvector"
                ) from e

            dsn = pg_dsn or os.getenv("PGVECTOR_DSN")
            if not dsn:
                raise ValueError(
                    "PGVECTOR_DSN is not set and no pg_dsn was provided")

            self.pg_conn = psycopg2.connect(dsn)
            self.pg_conn.autocommit = True
            register_vector(self.pg_conn)

            table = self.collection_name
            with self.pg_conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT PRIMARY KEY,
                        embedding VECTOR({self.embedding_dim}),
                        document TEXT,
                        metadata JSONB
                    );
                    """
                )
                # IVFFLAT index is optional; requires ANALYZE for best results
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {table}_embedding_idx
                    ON {table} USING ivfflat (embedding vector_cosine_ops);
                    """
                )
            logger.info(f"Initialized pgvector table: {table}")
            return

        raise ValueError(f"Unsupported vector_db: {self.vector_db}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        """
        try:
            return self.embedding_model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def batch_generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings in batches
        """
        embeddings = []
        total_batches = (len(texts) - 1) // batch_size + 1

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(
                f"Generating embeddings for batch {batch_num}/{total_batches}")

            try:
                batch_embeddings = self.embedding_model.encode(
                    batch).tolist()

                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                raise

        return embeddings

    def store_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ):
        """
        Generate embeddings and store documents in vector database

        Args:
            documents: List of Document objects
            batch_size: Number of documents to process per batch
        """
        logger.info(f"Processing {len(documents)} documents...")

        # Extract data from documents
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [doc.metadata['chunk_id'] for doc in documents]

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.batch_generate_embeddings(texts, batch_size)

        # Store in vector database
        logger.info("Storing in vector database...")

        if self.vector_db == "pgvector":
            self._store_pgvector(texts, embeddings, metadatas, ids, batch_size)
        else:
            raise ValueError(f"Unsupported vector_db: {self.vector_db}")

        logger.info(f"✓ Successfully stored {len(documents)} documents")

    def search(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents

        Args:
            query: Search query text
            n_results: Number of results to return
            filters: Metadata filters (e.g., {"scenario_type": "opioid_overdose_response"})

        Returns:
            List of result dictionaries
        """
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        if self.vector_db == "pgvector":
            return self._search_pgvector(query_embedding, n_results, filters)
        raise ValueError(f"Unsupported vector_db: {self.vector_db}")

    def build_and_filter(self, filters: dict):
        return {"$and": [{k: v} for k, v in filters.items()]}

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection

        Returns:
            Dictionary with collection statistics
        """
        if self.vector_db == "pgvector":
            with self.pg_conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.collection_name}")
                count = cur.fetchone()[0]
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "embedding_dimension": self.embedding_dim
            }
        raise ValueError(f"Unsupported vector_db: {self.vector_db}")

    def delete_collection(self):
        """Delete the entire collection (use with caution!)"""
        if self.vector_db == "pgvector":
            with self.pg_conn.cursor() as cur:
                cur.execute(f"DROP TABLE IF EXISTS {self.collection_name}")
            logger.info(f"Deleted pgvector table: {self.collection_name}")
            return
        raise ValueError(f"Unsupported vector_db: {self.vector_db}")

    def _store_pgvector(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str],
        batch_size: int
    ):
        table = self.collection_name
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            with self.pg_conn.cursor() as cur:
                for j in range(i, batch_end):
                    cur.execute(
                        f"""
                        INSERT INTO {table} (id, embedding, document, metadata)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            document = EXCLUDED.document,
                            metadata = EXCLUDED.metadata
                        """,
                        (ids[j], embeddings[j], texts[j], metadatas[j])
                    )
            logger.info(
                f"  Stored batch {(i // batch_size) + 1}/{(len(texts) - 1) // batch_size + 1}")

    def _search_pgvector(
        self,
        query_embedding: List[float],
        n_results: int,
        filters: Optional[Dict]
    ) -> List[Dict]:
        table = self.collection_name

        where_sql = ""
        filter_params: List = []
        if filters:
            clauses = []
            for k, v in filters.items():
                clauses.append(f"metadata->>%s = %s")
                filter_params.extend([k, str(v)])
            where_sql = "WHERE " + " AND ".join(clauses)

        with self.pg_conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, document, metadata, embedding <=> %s AS distance
                FROM {table}
                {where_sql}
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                [query_embedding] + filter_params +
                [query_embedding, n_results]
            )
            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "text": r[1],
                "metadata": r[2],
                "distance": float(r[3]),
            }
            for r in rows
        ]
