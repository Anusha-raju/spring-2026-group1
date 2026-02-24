"""Retrieval evaluation with Precision, Recall, MRR, and F1 metrics."""

import json
import logging
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

    def __init__(self, chunk_records: List[Dict], k_values: List[int]):
        """
        Args:
            chunk_records: List of chunk dicts
            k_values: List of k values to evaluate at
        """
        self.chunk_records = chunk_records
        self.chunks = [r["text"] if isinstance(r, dict) else r for r in chunk_records]
        self.k_values = k_values
        self._corpus_cache: Dict = {}

    def _get_corpus_embeddings(self, model: "EmbeddingModel"):
        """Encode corpus once per model and reuse across all evaluate_* calls."""
        if model.name not in self._corpus_cache:
            logger.info(f"Encoding corpus with {model.name} (once, cached)...")
            self._corpus_cache[model.name] = model.encode(self.chunks)
        return self._corpus_cache[model.name]

    def _label_relevant_chunks(self, query_gt: Dict) -> List[int]:
        """Return indices of all chunks relevant to a ground-truth query."""
        keywords = query_gt["relevant_keywords"]
        min_matches = query_gt.get("min_keyword_matches", 2)
        return [
            i for i, chunk in enumerate(self.chunks)
            if is_chunk_relevant(chunk, keywords, min_matches)
        ]

    def _get_filtered_indices(self, topic: str) -> List[int]:
        """Return indices of chunks tagged with the given topic."""
        return [
            i for i, r in enumerate(self.chunk_records)
            if isinstance(r, dict) and topic in r.get("topics", [])
        ]

    def _get_profession_filtered_indices(self, profession: str) -> List[int]:
        """Return indices of chunks whose categories include the given profession."""
        return [
            i for i, r in enumerate(self.chunk_records)
            if isinstance(r, dict) and profession in r.get("categories", [])
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

    def _average_metrics(self, results_by_k: Dict[int, List[Dict]]) -> Dict[int, Dict[str, float]]:
        averaged = {}
        for k, metric_list in results_by_k.items():
            if not metric_list:
                averaged[k] = {"precision": 0, "recall": 0, "mrr": 0, "f1": 0}
                continue
            averaged[k] = {
                metric: np.mean([m[metric] for m in metric_list])
                for metric in ["precision", "recall", "mrr", "f1"]
            }
        return averaged

    def evaluate_model(
        self,
        model: EmbeddingModel,
        ground_truth: List[Dict],
    ) -> Dict[int, Dict[str, float]]:
        """Baseline evaluation — searches all chunks."""
        corpus_embeddings = self._get_corpus_embeddings(model)

        results_by_k: Dict[int, List[Dict]] = {k: [] for k in self.k_values}

        for query_gt in ground_truth:
            query = query_gt["query"]
            relevant_indices = self._label_relevant_chunks(query_gt)

            if not relevant_indices:
                logger.warning(f"No relevant chunks for query: {query[:60]}...")

            max_k = max(self.k_values)
            retrieved = model.similarity_search(query, corpus_embeddings, max_k)
            retrieved_indices = [idx for idx, _ in retrieved]

            for k in self.k_values:
                results_by_k[k].append(
                    self.evaluate_single_query(retrieved_indices, relevant_indices, k)
                )

        return self._average_metrics(results_by_k)

    def evaluate_model_filtered(
        self,
        model: EmbeddingModel,
        ground_truth: List[Dict],
    ) -> Dict[int, Dict[str, float]]:
        """
        Filtered evaluation — each query's 'topic' field is used to narrow
        the search space to only chunks tagged with that topic before retrieval.
        Queries without a 'topic' field are skipped.
        """
        corpus_embeddings = self._get_corpus_embeddings(model)

        results_by_k: Dict[int, List[Dict]] = {k: [] for k in self.k_values}
        skipped = 0

        for query_gt in ground_truth:
            query = query_gt["query"]
            topic = query_gt.get("topic")

            if not topic:
                skipped += 1
                continue

            filtered_indices = self._get_filtered_indices(topic)
            relevant_indices = self._label_relevant_chunks(query_gt)

            if not filtered_indices:
                logger.warning(f"No chunks tagged '{topic}' for: {query[:60]}")
                for k in self.k_values:
                    results_by_k[k].append({"precision": 0.0, "recall": 0.0, "mrr": 0.0, "f1": 0.0})
                continue

            max_k = max(self.k_values)
            retrieved = model.filtered_similarity_search(
                query, corpus_embeddings, filtered_indices, max_k
            )
            retrieved_indices = [idx for idx, _ in retrieved]

            for k in self.k_values:
                results_by_k[k].append(
                    self.evaluate_single_query(retrieved_indices, relevant_indices, k)
                )

        if skipped:
            logger.warning(f"Skipped {skipped} queries with no 'topic' field.")

        return self._average_metrics(results_by_k)

    def evaluate_model_profession_filtered(
        self,
        model: EmbeddingModel,
        ground_truth: List[Dict],
    ) -> Dict[int, Dict[str, float]]:
        """
        Profession-filtered evaluation — each query's 'profession' field is used
        to restrict the search space to only chunks tagged with that profession
        in their 'categories' metadata.  This mirrors the multi-agent IPE setup
        where each agent (Nurse, PA, Public_Health, Social_Work) only searches
        its own corpus.  Queries without a 'profession' field are skipped.
        """
        corpus_embeddings = self._get_corpus_embeddings(model)

        results_by_k: Dict[int, List[Dict]] = {k: [] for k in self.k_values}
        skipped = 0

        for query_gt in ground_truth:
            query = query_gt["query"]
            profession = query_gt.get("profession")

            if not profession:
                skipped += 1
                continue

            filtered_indices = self._get_profession_filtered_indices(profession)
            relevant_indices = self._label_relevant_chunks(query_gt)

            if not filtered_indices:
                logger.warning(f"No chunks for profession '{profession}': {query[:60]}")
                for k in self.k_values:
                    results_by_k[k].append({"precision": 0.0, "recall": 0.0, "mrr": 0.0, "f1": 0.0})
                continue

            max_k = max(self.k_values)
            retrieved = model.filtered_similarity_search(
                query, corpus_embeddings, filtered_indices, max_k
            )
            retrieved_indices = [idx for idx, _ in retrieved]

            for k in self.k_values:
                results_by_k[k].append(
                    self.evaluate_single_query(retrieved_indices, relevant_indices, k)
                )

        if skipped:
            logger.warning(f"Skipped {skipped} queries with no 'profession' field.")

        return self._average_metrics(results_by_k)

    def evaluate_model_detailed(
        self,
        model: EmbeddingModel,
        ground_truth: List[Dict],
    ) -> List[Dict]:
        """Per-query, per-k detailed results for CSV export (baseline)."""
        corpus_embeddings = self._get_corpus_embeddings(model)
        rows = []

        for query_gt in ground_truth:
            query = query_gt["query"]
            relevant_indices = self._label_relevant_chunks(query_gt)

            max_k = max(self.k_values)
            retrieved = model.similarity_search(query, corpus_embeddings, max_k)
            retrieved_indices = [idx for idx, _ in retrieved]
            retrieved_scores = [score for _, score in retrieved]

            for k in self.k_values:
                metrics = self.evaluate_single_query(retrieved_indices, relevant_indices, k)
                top_k_previews = [
                    self.chunks[idx][:120].replace("\n", " ")
                    for idx in retrieved_indices[:k]
                ]
                rows.append({
                    "model": model.name,
                    "query": query,
                    "topic": query_gt.get("topic", ""),
                    "k": k,
                    "mode": "baseline",
                    "precision": round(metrics["precision"], 4),
                    "recall": round(metrics["recall"], 4),
                    "mrr": round(metrics["mrr"], 4),
                    "f1": round(metrics["f1"], 4),
                    "num_relevant_total": len(relevant_indices),
                    "top_score": round(retrieved_scores[0], 4) if retrieved_scores else 0,
                    "retrieved_preview": " ||| ".join(top_k_previews),
                })
        return rows