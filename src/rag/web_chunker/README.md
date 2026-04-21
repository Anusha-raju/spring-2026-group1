## Overview

The `web_chunker` module is for normalizing and segmenting web-scraped content into a format compatible with the project's RAG retrieval pipeline. It ensures that data originating from website sources is processed with the same semantic consistency as the PDF-based corpus.

## Core Functionality

The primary processing pipeline (`evaluation.py`) performs the following operations:

- **Multi-Source Loading**: Ingests web content from both raw JSON extracts and cleaned TXT files.
- **Semantic Consistency**: Leverages the `sentence_pack` chunking strategy inherited from the `pdf_chunker` module to maintain uniform chunk sizing across all data sources.
- **Domain Tagging**: Applies the centralized `OPIOID_TOPICS` ontology to perform multi-label keyword matching for clinical relevance.
- **Unified Output**: Generates `.jsonl` chunk files, allowing web data to be seamlessly indexed alongside PDF data in the vector store.

## Metadata & Role Mapping

To support professional-specific retrieval, the module integrates dynamic metadata mapping:
- **`website_knowledge.csv`**: Maps source URLs to specific professional roles (Social Work, PA, etc.).
- **URL Normalization**: Handles complex web paths and macOS-specific filename formats to ensure accurate metadata attachment during the chunking process.

## Integration & Dependencies

This module is designed as an extension of the core RAG infrastructure:
- **Shared Utilities**: Directly imports data classes and token estimation logic from the `rag/pdf_chunker` directory.
- **NLP Engine**: Utilizes [NLTK](https://www.nltk.org/) for robust sentence-level tokenization.

## Usage

### Processing Web Extracts
The pipeline supports two modes of execution depending on the input data format:

**Text Mode (Preferred):**
Processes cleaned text files produced by previous extraction steps.
```bash
python evaluation.py --txt_dir path/to/web_txt_outputs --out_dir ./out
```

**JSON Mode:**
Processes raw JSON extracts containing metadata and site structure.
```bash
python evaluation.py --json_dir path/to/web_json_outputs --out_dir ./out
```
