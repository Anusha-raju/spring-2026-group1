import os
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Base class for all embedding models."""

    name: str = "base"

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into embeddings."""

    @abstractmethod
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query."""

    def similarity_search(
        self,
        query: str,
        corpus_embeddings: np.ndarray,
        k: int,
    ) -> List[Tuple[int, float]]:
        """Return top-k (index, score) pairs. Higher score = more relevant."""
        query_emb = self.encode_query(query)
        # Cosine similarity
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        corpus_norms = corpus_embeddings / (
            np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-10
        )
        scores = corpus_norms @ query_norm
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def filtered_similarity_search(
        self,
        query: str,
        corpus_embeddings: np.ndarray,
        filtered_indices: List[int],
        k: int,
    ) -> List[Tuple[int, float]]:
        """
        Search only within filtered_indices (dense default).
        Returns (global_index, score) pairs.
        """
        if not filtered_indices:
            return []
        k = min(k, len(filtered_indices))
        arr = np.array(filtered_indices)
        filtered_embs = corpus_embeddings[arr]
        query_emb = self.encode_query(query)
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        norms = filtered_embs / (np.linalg.norm(filtered_embs, axis=1, keepdims=True) + 1e-10)
        scores = norms @ query_norm
        top_local = np.argsort(scores)[::-1][:k]
        return [(filtered_indices[int(i)], float(scores[i])) for i in top_local]


# Dense: SentenceTransformer models

class SentenceTransformerEmbedding(EmbeddingModel):
    """Wrapper around any SentenceTransformer model."""

    def __init__(self, model_name: str, display_name: str):
        from sentence_transformers import SentenceTransformer
        self.name = display_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(texts, show_progress_bar=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self._model.encode([query])[0]


class MiniLMEmbedding(SentenceTransformerEmbedding):
    def __init__(self):
        super().__init__("all-MiniLM-L6-v2", "MiniLM-L6-v2")


class MPNetEmbedding(SentenceTransformerEmbedding):
    def __init__(self):
        super().__init__("all-mpnet-base-v2", "MPNet-base-v2")


class BGESmallEmbedding(SentenceTransformerEmbedding):
    def __init__(self):
        super().__init__("BAAI/bge-small-en-v1.5", "BGE-small-en")


class BGEM3Embedding(EmbeddingModel):
    """BAAI/bge-m3 — 8192-token context window, forced to CPU to avoid MPS OOM."""

    name = "BGE-M3"

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("BAAI/bge-m3", device="cpu")
        self._model.max_seq_length = 8192

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._model.encode(texts, show_progress_bar=True, batch_size=4)

    def encode_query(self, query: str) -> np.ndarray:
        return self._model.encode([query], device="cpu")[0]


class InstructorXLEmbedding(EmbeddingModel):
    """
    HKUNLP Instructor-XL — instruction-tuned embedding model.
    Uses task-specific instruction prefixes for corpus and query encoding.
    Forced to CPU to avoid MPS out-of-memory on Apple Silicon.
    """

    name = "Instructor-XL"

    CORPUS_INSTRUCTION = "Represent the medical document for retrieval:"
    QUERY_INSTRUCTION = "Represent the medical question for retrieving relevant documents:"

    def __init__(self):
        from InstructorEmbedding import INSTRUCTOR
        self._model = INSTRUCTOR("hkunlp/instructor-xl", device="cpu")

    def encode(self, texts: List[str]) -> np.ndarray:
        pairs = [[self.CORPUS_INSTRUCTION, t] for t in texts]
        return self._model.encode(pairs, batch_size=16, show_progress_bar=True)

    def encode_query(self, query: str) -> np.ndarray:
        return self._model.encode([[self.QUERY_INSTRUCTION, query]])[0]


class OllamaEmbedding(EmbeddingModel):
    """Ollama local embedding model (default: nomic-embed-text)."""

    def __init__(self, model_name: str = "nomic-embed-text", base_url: str | None = None):
        self._model_name = model_name
        self._base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.name = f"Ollama-{model_name}"
        self._max_chars = int(os.getenv("OLLAMA_EMBED_MAX_CHARS", "12000"))

    def _embed_one(self, text: str):
        import requests

        # Remove problematic null bytes and normalise whitespace.
        clean = text.replace("\x00", " ").strip()
        if not clean:
            clean = " "

        lengths = [
            None,
            self._max_chars,
            max(8000, self._max_chars // 2),
            4000,
            2000,
            1000,
        ]
        last_error = None
        for max_len in lengths:
            prompt = clean if max_len is None else clean[:max_len]
            resp = requests.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model_name, "prompt": prompt},
                timeout=120,
            )
            if resp.status_code < 400:
                return resp.json()["embedding"]
            last_error = f"{resp.status_code} {resp.text[:200]}"

        raise RuntimeError(
            f"Ollama embedding failed after retries for model '{self._model_name}'. "
            f"Last error: {last_error}"
        )

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        import requests

        # Prefer batch endpoint when available.
        try:
            resp = requests.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model_name, "input": texts},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            vectors = data.get("embeddings")
            if vectors:
                return np.asarray(vectors, dtype=np.float32)
        except Exception:
            pass

        # Fallback to per-text endpoint for compatibility.
        vectors = []
        for text in texts:
            vectors.append(self._embed_one(text))
        return np.asarray(vectors, dtype=np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self._embed_batch(texts)

    def encode_query(self, query: str) -> np.ndarray:
        return self._embed_batch([query])[0]


# Dense: OpenAI

class OpenAIEmbedding(EmbeddingModel):
    """OpenAI text-embedding-3-small via the openai SDK."""

    name = "OpenAI-3-small"

    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self._client = OpenAI(api_key=api_key)
        self._model = "text-embedding-3-small"

    def _embed(self, texts: List[str]) -> np.ndarray:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return np.array([d.embedding for d in response.data])

    def encode(self, texts: List[str]) -> np.ndarray:
        # OpenAI has a batch limit; process in chunks of 2048
        all_embs = []
        for i in range(0, len(texts), 2048):
            batch = texts[i : i + 2048]
            all_embs.append(self._embed(batch))
        return np.vstack(all_embs)

    def encode_query(self, query: str) -> np.ndarray:
        return self._embed([query])[0]


# Sparse: TF-IDF

class TFIDFEmbedding(EmbeddingModel):
    """TF-IDF vectoriser with cosine similarity search."""

    name = "TF-IDF"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._fitted = False

    def encode(self, texts: List[str]) -> np.ndarray:
        matrix = self._vectorizer.fit_transform(texts)
        self._fitted = True
        return matrix  # sparse matrix, handled in similarity_search

    def encode_query(self, query: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Must call encode() on corpus before encode_query()")
        return self._vectorizer.transform([query])

    def similarity_search(
        self, query: str, corpus_embeddings, k: int
    ) -> List[Tuple[int, float]]:
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = self.encode_query(query)
        scores = cosine_similarity(query_vec, corpus_embeddings).flatten()
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def filtered_similarity_search(
        self, query: str, corpus_embeddings, filtered_indices: List[int], k: int
    ) -> List[Tuple[int, float]]:
        from sklearn.metrics.pairwise import cosine_similarity
        if not filtered_indices:
            return []
        k = min(k, len(filtered_indices))
        query_vec = self.encode_query(query)
        filtered_embs = corpus_embeddings[filtered_indices]
        scores = cosine_similarity(query_vec, filtered_embs).flatten()
        top_local = np.argsort(scores)[::-1][:k]
        return [(filtered_indices[int(i)], float(scores[i])) for i in top_local]


# Sparse: BM25

class BM25Retriever(EmbeddingModel):
    """BM25 ranking (no vector embeddings needed)."""

    name = "BM25"

    def __init__(self):
        self._bm25 = None
        self._tokenized_corpus = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    def encode(self, texts: List[str]) -> np.ndarray:
        from rank_bm25 import BM25Okapi
        self._tokenized_corpus = [self._tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        return np.array([])

    def encode_query(self, query: str) -> np.ndarray:
        return np.array([])

    def similarity_search(
        self, query: str, corpus_embeddings, k: int
    ) -> List[Tuple[int, float]]:
        if self._bm25 is None:
            raise RuntimeError("Must call encode() on corpus before searching")
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def filtered_similarity_search(
        self, query: str, corpus_embeddings, filtered_indices: List[int], k: int
    ) -> List[Tuple[int, float]]:
        if self._bm25 is None:
            raise RuntimeError("Must call encode() on corpus before searching")
        if not filtered_indices:
            return []
        k = min(k, len(filtered_indices))
        tokens = self._tokenize(query)
        all_scores = self._bm25.get_scores(tokens)
        pairs = [(idx, float(all_scores[idx])) for idx in filtered_indices]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:k]

def get_all_models(include_openai: bool = True) -> List[EmbeddingModel]:
    """Instantiate all available embedding models."""
    models: List[EmbeddingModel] = [
        MiniLMEmbedding(),
        MPNetEmbedding(),
        BGESmallEmbedding(),
        # BGEM3Embedding(),
        # InstructorXLEmbedding(),
        TFIDFEmbedding(),
        BM25Retriever(),
    ]
    if include_openai:
        try:
            models.insert(3, OpenAIEmbedding())
        except (ValueError, ImportError) as e:
            logger.warning(f"Skipping OpenAI embeddings: {e}")
    # try:
    #     models.append(InstructorXLEmbedding())
    # except (ImportError, Exception) as e:
    #     logger.warning(f"Skipping Instructor-XL: {e}")
    return models