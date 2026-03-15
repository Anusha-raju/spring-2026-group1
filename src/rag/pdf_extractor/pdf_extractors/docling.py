from pathlib import Path
import logging
from docling.document_converter import DocumentConverter
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    converter = DocumentConverter()
except Exception as e:
    logger.exception("Failed to initialize Docling")
    raise RuntimeError("Docling initialization failed") from e

def extract_markdown_from_pdf(converter: DocumentConverter, pdf_path: Path) -> str:
    """Extract markdown content from a PDF using Docling."""
    try:
        result = converter.convert(str(pdf_path))
        return result.document.export_to_markdown()
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF '{pdf_path}': {e}") from e


def process_file(pdf_path: Path, output_path: Path | None = None) -> None:
    text = extract_markdown_from_pdf(converter, pdf_path)

    if output_path is None:
        output_path = pdf_path.with_name(f"{pdf_path.stem}_extracted.txt")
    output_path.write_text(text, encoding="utf-8")
    logger.info("Successfully extracted text to: %s", output_path)