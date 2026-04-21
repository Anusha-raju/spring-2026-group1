import argparse
import logging
import sys
from pathlib import Path
from pdf_extractors.docling import process_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract text from PDFs using Docling.")
    parser.add_argument("path", help="Path to a PDF file or directory")
    parser.add_argument("-o", "--output", help="Optional output path for a single PDF", default=None)
    args = parser.parse_args()

    input_path = Path(args.path)

    if not input_path.exists():
        logger.error("Path not found: %s", input_path)
        return 1

    try:
        if input_path.is_file():
            logger.info("Processing PDF: %s", input_path)

            process_file(
                input_path,
                Path(args.output) if args.output else None,
            )
        elif input_path.is_dir():
            pdf_files = sorted(input_path.glob("*.pdf"))
            if not pdf_files:
                logger.warning("No PDF files found in directory: %s", input_path)
                return 0

            for pdf_file in pdf_files:
                process_file(pdf_file)
        else:
            logger.error("Invalid path type: %s", input_path)
            return 1

    except Exception:
        logger.exception("PDF extraction failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())