import os
from typing import List, Dict, Optional
from langchain_core.documents import Document
import logging
from sentence_transformers import SentenceTransformer
import chromadb

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """Generate embeddings and store in vector database"""

    def __init__(
        self,
        embedding_provider: str = "sentence_transformers",
        vector_db: str = "chromadb",
        collection_name: str = "opioid_documents",
        db_path: str = "./chroma_db"
    ):
        """
        Initialize embedding processor

        Args:
            embedding_provider: "sentence_transformers"
            vector_db: "chromadb" or "pinecone"
            collection_name: Name of the collection/index
            db_path: Path for local database (ChromaDB only)
        """
        self.embedding_provider = embedding_provider
        self.vector_db = vector_db
        self.collection_name = collection_name

        # Initialize embedding model
        self._init_embedding_model()

        # Initialize vector database
        self._init_vector_db(db_path)

    def _init_embedding_model(self):
        """Initialize the embedding model"""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        logger.info("Initialized Sentence Transformers: all-MiniLM-L6-v2")

    def _init_vector_db(self, db_path: str):
        """Initialize the vector database"""
        self.client = chromadb.PersistentClient(path=db_path)

        # Create or get collection
        self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={
                                                               "description": "Opioid healthcare documents with embeddings"})
        logger.info(
            f"Initialized ChromaDB collection: {self.collection_name}")

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

        self._store_chromadb(texts, embeddings, metadatas, ids, batch_size)

        logger.info(f"✓ Successfully stored {len(documents)} documents")

    def _store_chromadb(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: List[str],
        batch_size: int
    ):
        """Store documents in ChromaDB"""
        total_batches = (len(texts) - 1) // batch_size + 1

        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            batch_num = i // batch_size + 1

            self.collection.upsert(
                embeddings=embeddings[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
            logger.info(f"  Stored batch {batch_num}/{total_batches}")

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
        return self._search_chromadb(query_embedding, n_results, filters)

    def build_and_filter(self, filters: dict):
        return {"$and": [{k: v} for k, v in filters.items()]}

    def _search_chromadb(
        self,
        query_embedding: List[float],
        n_results: int,
        filters: Optional[Dict]
    ) -> List[Dict]:
        """Search in ChromaDB"""
        parsed_filters = None
        if filters is not None:
            parsed_filters = self.build_and_filter(filters)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=parsed_filters
        )

        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })

        return formatted_results

    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection

        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "embedding_dimension": self.embedding_dim
        }

    def delete_collection(self):
        """Delete the entire collection (use with caution!)"""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Deleted ChromaDB collection: {self.collection_name}")
