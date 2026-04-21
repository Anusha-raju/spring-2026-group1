from dataclass import Chunk
from typing import List, Tuple
import numpy as np
import faiss 



class VectorIndex:
    def __init__(self, embed_dim: int):
        self.index = faiss.IndexFlatIP(embed_dim)  # cosine via L2-normalize
        self.chunks: List[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_emb: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        query_emb = query_emb.astype(np.float32)
        faiss.normalize_L2(query_emb)
        scores, idxs = self.index.search(query_emb, top_k)
        out: List[Tuple[Chunk, float]] = []
        for i, s in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            out.append((self.chunks[int(i)], float(s)))
        return out