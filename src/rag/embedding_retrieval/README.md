## Overview

The `embedding_retrieval` module is designed to identify the most effective retrieval strategies for clinical and healthcare-related documents. It supports a dual-mode operation:
1.  **Offline Evaluation**: Benchmarking seven retrievers (SentenceTransformers dense models, BM25, TF-IDF) against a 15-query ground-truth set using standard information retrieval metrics.
2.  **Production Retrieval**: High-performance, metadata-filtered search using Pinecone vector database.

## Key Components

### 1. `embedding_models.py` (Model Zoo)
Provides a unified interface for multiple embedding and ranking techniques:
- **Dense Models**: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `BAAI/bge-small-en`, `hkunlp/instructor-xl`, and `BAAI/bge-m3`.
- **Sparse Models**: `TF-IDF` and `BM25Okapi`.
- **Hybrid Support**: Abstract base classes allow for easy extension to new models.

### 2. `pinecone_store.py` (Vector Database Interface)
The production-ready retriever that manages:
- **Metadata Filtering**: RESTful filtering by `profession` (e.g., Nurse, PA) or `topic`.
- **ASCII Normalisation**: Handles unique chunk IDs for Pinecone compatibility.
- **Batch Upsert**: Efficiently indexes large document corpuses.

### 3. `retrieval_evaluator.py` (Metric Engine)
Calculates performance metrics for retrieval quality:
- **Precision@K**: The proportion of retrieved chunks that are relevant.
- **Recall@K**: The proportion of relevant chunks that were successfully retrieved.
- **MRR (Mean Reciprocal Rank)**: Measures where the first relevant chunk appears in the results.
- **F1 Score**: Harmonic mean of Precision and Recall.

### 4. `evaluation.py` (Orchestrator)
The main execution script to run full comparison pipelines. It compares model performance across different `k` values and filtering modes (Baseline, Topic-Filtered, and Profession-Filtered).

## Getting Started

### Prerequisites
1.  **Environment Variables**: Create a `.env` file with:
    ```env
    PINECONE_API_KEY=your_key
    OPENAI_API_KEY=your_key (optional)
    USE_PINECONE=false (set to true for production mode)
    ```
2.  **Ground Truth**: Ensure `ground_truth.json` is populated with query-keyword pairs for evaluation. The current set contains 15 queries spanning all 10 topics of the opioid taxonomy (overdose, emergency, naloxone, withdrawal, dosage, treatment, prevention, mental health, legal, patient education).

### Running Evaluation
To run the full benchmarking pipeline and generate comparison CSVs:
```bash
python evaluation.py --chunker_output_dir ../../out
```

### Production Indexing
To index all processed chunks into Pinecone:
```bash
python pinecone_upsert.py
```

## Metrics & Reporting

The evaluation scripts generate several reports in the `outputs/` directory:
- `comparison_summary.csv`: Aggregated MRR, Precision, Recall, and F1 per model.
- `comparison_results.csv`: Detailed per-query retrieval performance across all models.
- `metadata_filter_comparison.csv`: F1 delta from topic filtering vs. baseline per model.
- `profession_filter_comparison.csv`: Retrieval metrics under profession-filtered mode per agent.
