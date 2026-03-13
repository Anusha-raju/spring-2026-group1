"""
Fetches URLs from website_knowledge.csv using WebExtractor and saves
extracted text as .txt files — same format as pdf_extractor/outputs/.

Usage:
    python3 src/rag/web_extractor/extract_to_txt.py
    python3 src/rag/web_extractor/extract_to_txt.py --output-dir src/rag/web_extractor/outputs_txt
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from web_processor import WebExtractor, WebExtractorError

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_WEBSITE_KNOWLEDGE_CSV = os.path.join(os.path.dirname(__file__), "..", "website_knowledge.csv")
_DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs_txt")


def _url_to_filename(url: str) -> str:
    """Convert URL to a readable filename, e.g. www.cdc.gov/overdose → www_cdc_gov_overdose.txt"""
    name = re.sub(r"https?://", "", url)
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:120] + ".txt"


def load_urls(csv_path: str):
    seen = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("Web URL", "").strip()
            if url and url not in seen:
                seen[url] = None
    return list(seen.keys())


def main():
    parser = argparse.ArgumentParser(description="Extract web pages to .txt files")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR,
                        help="Directory to save .txt files")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip URLs already extracted (default: True)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = load_urls(_WEBSITE_KNOWLEDGE_CSV)
    logger.info("Found %d unique URLs in website_knowledge.csv", len(urls))

    passed, failed, skipped = [], [], []

    with WebExtractor() as extractor:
        for url in urls:
            filename = _url_to_filename(url)
            out_path = output_dir / filename

            if args.skip_existing and out_path.exists():
                logger.info("Skipping (exists): %s", filename)
                skipped.append(url)
                continue

            try:
                page = extractor.extract_from_url(url)
                out_path.write_text(page.text, encoding="utf-8")
                logger.info("Saved: %s (%d chars)", filename, len(page.text))
                passed.append(url)
            except WebExtractorError as exc:
                logger.error("Failed %s: %s", url, exc)
                failed.append((url, str(exc)))

    logger.info(
        "\n── Summary ──────────────────────────────\n"
        "  Total   : %d\n"
        "  Saved   : %d\n"
        "  Skipped : %d\n"
        "  Failed  : %d\n"
        "─────────────────────────────────────────",
        len(urls), len(passed), len(skipped), len(failed),
    )
    if failed:
        logger.warning("Failed URLs:")
        for url, reason in failed:
            logger.warning("  %s — %s", url, reason)


if __name__ == "__main__":
    main()
