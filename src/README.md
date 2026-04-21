## Overview

The `src` directory contains the core source code for the project, organized into two primary systems: a robust Retrieval-Augmented Generation (RAG) pipeline and a dynamic prompt engineering framework. Together, these modules enable the ingestion of healthcare documentation and the generation of role-specific clinical prompts.

## Core Systems

### 1. Data Ingestion & Retrieval (`rag`)
The `rag` module is the project's data engine. It handles:
- **Extraction**: Transforming raw PDFs and web pages into normalized text.
- **Segmentation**: Using structural and semantic chunking strategies to optimize context window usage.
- **Vector Storage**: Indexing document chunks into Pinecone for high-performance retrieval.
- **Evaluation**: Benchmarking retrieval accuracy using Hit@K, MRR, and other standard metrics.

### 2. Prompt Engineering (`dynamic_prompts`)
The `dynamic_prompts` module focuses on the LLM interaction layer. It provides:
- **Role-Based Templates**: Pre-defined personas for various healthcare professionals (Nurse, Physician Assistant, Public Health, etc.).
- **Context Injection**: Combining RAG-retrieved data with expert instructions to create grounded prompts.
- **Bulk Evaluation**: Tools to generate and compare prompts across different open-source models (Llama 3, Mistral, Gemma).

## Directory Structure

```text
src/
├── rag/                # End-to-end RAG pipeline (Extract → Chunk → Retrieve)
│   ├── pdf_extractor/  # Raw PDF-to-text conversion
│   ├── web_extractor/  # Web scraping and normalization
│   ├── pdf_chunker/    # Structural and semantic PDF segmentation
│   ├── web_chunker/    # Web content segmentation
│   └── embedding_retrieval/  # Vector storage and retrieval benchmarking
└── dynamic_prompts/    # Persona-aligned prompt generation and model evaluation
```
