## Overview

The `web_extractor` module is a scraping and text transformation engine designed to ingest online healthcare resources into the RAG pipeline. It utilizes a structured extraction methodology to ensure that web content is converted into a high-fidelity, normalized format suitable for semantic search and LLM context injection.

## 4-Stage Extraction Pipeline

The core logic (`web_processor.py`) implements a sequential pipeline to maximize content quality and structural integrity:

1.  **Resilient Fetch**: Implements an HTTP layer with automatic retries for transient errors (429, 5xx), exponential backoff, and size-based guards to prevent memory overflow.
2.  **Structural Extraction**: Uses [trafilatura](https://trafilatura.readthedocs.io/) to isolate main body content from boilerplate (navbars, ads, footers) and convert it into a structured XML representation.
3.  **Semantic Formatting**: Parses the intermediate XML into a typed dictionary of elements, maintaining the distinction between headings, paragraphs, lists, and tables.
4.  **Normalization**: Renders the structural data into plain text while collapsing excess whitespace and ensuring uniform encoding.

## Key Features

- **Deduplication & Mapping**: Integrates with `extract_urls.py` to process unique URLs from `website_knowledge.csv` and automatically associate them with professional role categories (e.g., Nurse, Physician Assistant).
- **Table & List Support**: Specifically preserves the integrity of structured data commonly found in clinical guidelines.
- **Pipeline Compatibility**: Can output results as either consolidated JSON records or individual `.txt` files, mirroring the output structure of the `pdf_extractor` module for seamless downstream integration.

## Technical Stack

- **Extraction Engine**: Trafilatura
- **Networking**: Requests with custom HTTPAdapters for retry logic.
- **Parsing**: Python standard library `xml.etree.ElementTree`.

## Usage

### Bulk Extraction to Text
To crawl all URLs in the knowledge database and save them as plain text files:
```bash
python extract_to_txt.py --output-dir path/to/output_txt
```

### Extraction to JSON Records
To generate detailed JSON payloads containing full metadata (URL, Title, Date, Categories):
```bash
python web_processor.py
```
