## Overview

The `rag` directory contains the core orchestration logic for the project's Retrieval-Augmented Generation (RAG) system. It manages the end-to-end lifecycle of clinical and professional healthcare data—from raw source extraction (PDF and Web) to structural chunking and vector-based retrieval benchmarking.

## Pipeline Architecture

The system is designed as a unified 4-stage pipeline, orchestrated by `pipeline.py`:

1.  **Extraction**: Ingests document data from local PDFs (`pdf_extractor`) and remote URLs (`web_extractor`).
2.  **Web Chunking**: Segments web-scraped content into normalized JSONL records (`web_chunker`).
3.  **PDF Chunking**: Applies multiple segmentation strategies (Fixed, Fixed-with-Overlap, Recursive, Sentence-Pack, Semantic) to extracted PDF text (`pdf_chunker`).
4.  **Evaluation**: Benchmarks retrieval accuracy (Hit@K, Precision, Recall, MRR) across different embedding models and search depths (`embedding_retrieval`).

## Knowledge Management

The pipeline is driven by centralized knowledge maps that associate source documentation with specific healthcare personas:

- **`pdf_knowledge.csv`**: Maps extracted PDF files to target roles such as Nurse, Physical Therapist, and Health Administrator.
- **`website_knowledge.csv`**: Maps source URLs to the appropriate professional contexts, ensuring role-specific retrieval accuracy.

## Core Modules

- **`pdf_extractor` / `web_extractor`**: Multi-engine extractors supporting Docling, PDFPlumber, and Trafilatura.
- **`pdf_chunker` / `web_chunker`**: Experimental frameworks for testing five segmentation strategies: fixed, fixed-with-overlap, recursive, sentence-pack, and semantic.
- **`embedding_retrieval`**: Integration with Pinecone for production retrieval and Numpy-based benchmarking for offline model comparison.

## Usage

### Running the Full Pipeline
To execute the entire workflow from extraction to evaluation:
```bash
python pipeline.py
```

### Advanced Execution
The pipeline supports modular execution to reuse existing data:
```bash
# Skip extraction and chunking, only run retrieval evaluation
python pipeline.py --skip-web-extract --skip-web-chunk --skip-pdf-chunk

# Configure custom token targets and directories
python pipeline.py --target-tokens 5000 --chunks-dir ./custom_out
```
