## Overview

The `pdf_extractor` module is a multi-engine extraction framework designed to transform complex PDF documents into high-quality, normalized text for downstream RAG tasks. It supports multiple extraction backends and a robust post-processing pipeline to remove document artifacts.

## Extraction Backends

The module evaluates several specialized PDF extraction engines, selectable via the `ExtractPDF` interface:

- **Docling**: High-fidelity structured text extraction using IBM's document conversion technology.
- **PDFPlumber**: Layout-aware extraction, particularly effective for table data and precise positioning.
- **DeepSeek**: Specialized extraction logic for structured document understanding.
- **XML Extractor**: Structural preservation via XML-based intermediate representation.
- **PyPDF**: Lightweight, high-velocity extraction for standard document layouts.

## Cleaning & Normalization Pipeline

After extraction, the `clean_text.py` module applies a series of heuristics to ensure the resulting text is noise-free:

1.  **Artifact Removal**: Strips image placeholders and system-inserted page markers (e.g., `=== Page N ===`).
2.  **Header/Footer Deduplication**: Identifies and removes repetitive running headers and footers that occur across numerous pages.
3.  **Whitespace Optimization**: Normalizes line breaks and removes excessive vertical spacing to improve token efficiency.

## Modular Architecture

The module uses a plugin-based architecture located in the `pdf_extractors/` directory:
- `docling.py`: Integration with the Docling conversion engine.
- `pdfplumber.py`: Detailed coordinate-based extraction.
- `pdf_to_structured_text_xml.py`: Logic for hierarchical mapping.

## Usage

### Running the Extraction Pipeline
To process a single file or an entire directory using the default Docling engine:
```bash
python pdf_extractor_pipeline.py path/to/input -o path/to/output.txt
```

### Text Post-Processing
To apply the cleaning heuristics to existing extraction outputs:
```bash
python clean_text.py
```
