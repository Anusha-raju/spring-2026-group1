## Overview

The `pdf_chunker` module is an experimental framework for optimizing document segmentation in RAG (Retrieval-Augmented Generation) pipelines. It provides multiple segmentation strategies and a comprehensive evaluation framework to measure retrieval accuracy (Hit@K) across different document structures.

## Core Chunking Strategies

This module implements five distinct chunking algorithms, each optimized for different retrieval patterns:

- **Fixed-Size**: Simple word-count based segmentation for uniform chunk distribution.
- **Fixed-Overlap**: Sliding window approach to preserve context across chunk boundaries.
- **Recursive**: A structural chunker that respects document hierarchy, prioritizing breaks at headings and paragraph boundaries.
- **Sentence-Pack**: Token-aware bundling of complete sentences to ensure semantic integrity.
- **Semantic**: A dynamic strategy that uses cosine similarity dips between sentence embeddings to detect topic shifts and create semantically coherent boundaries.

## Evaluation Framework

The module includes a robust benchmarking framework to identify the optimal chunking configuration:

### 1. `evaluation.py` (Accuracy Assessment)
Measures **Hit@K** by verifying if ground-truth answers (keywords or regex) are present within the top-k retrieved chunks.
- Supports multi-label tagging of chunks using the `OPIOID_TOPICS` ontology.
- Generates per-question and summary CSV reports.

### 2. `evaluation_grid.py` (Parameter Sweep)
Executes a systematic grid search across:
- **Chunking Method**: Comparison of all 5 strategies.
- **Token Targets**: Variable chunk sizes (e.g., 3k, 5k, 7k, 10k tokens).
- **Search Depth (K)**: Retrieval performance at different top-k values.
Resulting metrics include hit rate, average query latency, and first-hit rank.

## Metadata & Topic Tagging

Chunks are automatically enriched with domain-specific metadata:
- **Opioid Topics**: Multi-label tagging based on expert keyword matching (e.g., *Naloxone*, *Overdose*, *Treatment*).
- **Role Categories**: Mapping of documents to specific healthcare professions (Nurse, PA, etc.) via `pdf_knowledge.csv`.

## Technical Stack

- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss) (IndexFlatIP) for fast similarity retrieval.
- **Embeddings**: [SentenceTransformers](https://sbert.net/) (default: `all-MiniLM-L6-v2`).
- **Processing**: [NLTK](https://www.nltk.org/) for sentence tokenization.

## Usage

### Running Evaluation
To benchmark chunking performance against a set of queries:
```bash
python evaluation.py --txt_dir path/to/extracted_text --out_dir ./results
```

### Grid Search Comparison
To run a full parameter sweep across multiple configurations:
```bash
python evaluation_grid.py --txt_dir path/to/text --ks "3,5,7" --target_tokens_list "3000,5000,7000"
```
