"""Retrieval evaluation with Precision, Recall, MRR, and F1 metrics."""

import json
import logging
import os
import re
from typing import Dict, List, Tuple

import numpy as np

from embedding_models import EmbeddingModel

logger = logging.getLogger(__name__)


def is_chunk_relevant(chunk_text: str, keywords: List[str], min_matches: int) -> bool:
    """Check if a chunk is relevant based on keyword matching."""
    text_lower = chunk_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches >= min_matches


class RetrievalEvaluator:
    """Evaluate retrieval quality across embedding models and k values."""

    def __init__(self, chunks: List[str], k_values: List[int]):
        self.chunks = chunks
        self.k_values = k_values

    def load_ground_truth(self, path: str) -> List[Dict]:
        with open(path, "r") as f:
            return json.load(f)

    def _label_relevant_chunks(self, query_gt: Dict) -> List[int]:
        """Return indices of all chunks relevant to a ground-truth query."""
        keywords = query_gt["relevant_keywords"]
        min_matches = query_gt.get("min_keyword_matches", 2)
        return [
            i
            for i, chunk in enumerate(self.chunks)
            if is_chunk_relevant(chunk, keywords, min_matches)
        ]

    def evaluate_single_query(
        self,
        retrieved_indices: List[int],
        relevant_indices: List[int],
        k: int,
    ) -> Dict[str, float]:
        """Compute metrics for a single query at a given k."""
        if not relevant_indices:
            return {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "f1": 0.0}

        relevant_set = set(relevant_indices)
        retrieved_set = set(retrieved_indices[:k])
        hits = relevant_set & retrieved_set

        precision = len(hits) / k if k > 0 else 0.0
        recall = len(hits) / len(relevant_set) if relevant_set else 0.0

        # MRR: reciprocal rank of first relevant result
        mrr = 0.0
        for rank, idx in enumerate(retrieved_indices[:k], start=1):
            if idx in relevant_set:
                mrr = 1.0 / rank
                break

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {"precision": precision, "recall": recall, "mrr": mrr, "f1": f1}

    def evaluate_model(
        self,
        model: EmbeddingModel,
        ground_truth: List[Dict],
    ) -> Dict[int, Dict[str, float]]:
        """
        Evaluate a single embedding model across all k values.

        Returns:
            {k: {metric_name: averaged_value}}
        """
        logger.info(f"Encoding corpus with {model.name}...")
        corpus_embeddings = model.encode(self.chunks)

        results_by_k: Dict[int, List[Dict[str, float]]] = {
            k: [] for k in self.k_values
        }

        for query_gt in ground_truth:
            query = query_gt["query"]
            relevant_indices = self._label_relevant_chunks(query_gt)

            if not relevant_indices:
                logger.warning(
                    f"No relevant chunks found for query: {query[:60]}..."
                )

            max_k = max(self.k_values)
            retrieved = model.similarity_search(query, corpus_embeddings, max_k)
            retrieved_indices = [idx for idx, _ in retrieved]

            for k in self.k_values:
                metrics = self.evaluate_single_query(
                    retrieved_indices, relevant_indices, k
                )
                results_by_k[k].append(metrics)

        # Average metrics across all queries for each k
        averaged: Dict[int, Dict[str, float]] = {}
        for k, metric_list in results_by_k.items():
            if not metric_list:
                averaged[k] = {"precision": 0, "recall": 0, "mrr": 0, "f1": 0}
                continue
            averaged[k] = {
                metric: np.mean([m[metric] for m in metric_list])
                for metric in ["precision", "recall", "mrr", "f1"]
            }
        return averaged

    def evaluate_model_detailed(
        self,
        model: EmbeddingModel,
        ground_truth: List[Dict],
    ) -> List[Dict]:
        """
        Return per-query, per-k detailed results for CSV export.

        Returns list of dicts with keys:
            model, query, k, precision, recall, mrr, f1, retrieved_texts
        """
        logger.info(f"Encoding corpus with {model.name} (detailed)...")
        corpus_embeddings = model.encode(self.chunks)
        rows = []

        for query_gt in ground_truth:
            query = query_gt["query"]
            relevant_indices = self._label_relevant_chunks(query_gt)

            max_k = max(self.k_values)
            retrieved = model.similarity_search(query, corpus_embeddings, max_k)
            retrieved_indices = [idx for idx, _ in retrieved]
            retrieved_scores = [score for _, score in retrieved]

            for k in self.k_values:
                metrics = self.evaluate_single_query(
                    retrieved_indices, relevant_indices, k
                )
                # Include top-k retrieved chunk previews
                top_k_previews = [
                    self.chunks[idx][:120].replace("\n", " ")
                    for idx in retrieved_indices[:k]
                ]
                rows.append(
                    {
                        "model": model.name,
                        "query": query,
                        "k": k,
                        "precision": round(metrics["precision"], 4),
                        "recall": round(metrics["recall"], 4),
                        "mrr": round(metrics["mrr"], 4),
                        "f1": round(metrics["f1"], 4),
                        "num_relevant_total": len(relevant_indices),
                        "top_score": round(retrieved_scores[0], 4) if retrieved_scores else 0,
                        "retrieved_preview": " ||| ".join(top_k_previews),
                    }
                )
        return rows
