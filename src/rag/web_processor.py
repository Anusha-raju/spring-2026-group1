"""
web content extractor for the RAG pipeline.
Extracts clean structured text from URLs via a 4-stage pipeline::
    fetch_html  →  html_to_xml  →  xml_content_formattor  →  format_from_xml
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import trafilatura
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from extract_urls import extract_all_urls

__all__ = [
    "WebExtractor",
    "ExtractedPage",
    "WebExtractorError",
    "FetchError",
    "ExtractionError",
    "ParseError",
]

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT: Tuple[int, int] = (10, 30)   # (connect_sec, read_sec)
_DEFAULT_RETRIES: int = 3
_DEFAULT_BACKOFF: float = 0.5                   # wait = backoff * 2^(attempt-1)
_MAX_CONTENT_BYTES: int = 10 * 1024 * 1024      # 10 MB hard cap
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_ACCEPTED_CONTENT_TYPES = ("text/html", "text/plain", "application/xhtml+xml")
_USER_AGENT = "Mozilla/5.0 (compatible; RAGBot/1.0)"


class WebExtractorError(Exception):
    """Base class for all WebExtractor failures."""


class FetchError(WebExtractorError):
    """
    Raised when the HTTP layer fails to deliver a usable response.

    Attributes:
        url:         The URL that was requested.
        status_code: HTTP status code, or ``None`` for network-level errors.
    """

    def __init__(
        self, url: str, reason: str, status_code: Optional[int] = None
    ) -> None:
        self.url = url
        self.status_code = status_code
        prefix = f"HTTP {status_code} — " if status_code else ""
        super().__init__(f"{prefix}Failed to fetch '{url}': {reason}")


class ExtractionError(WebExtractorError):
    """
    Raised when trafilatura cannot extract any usable content from the HTML.

    Attributes:
        url: The URL that was processed.
    """

    def __init__(self, url: str, reason: str) -> None:
        self.url = url
        super().__init__(f"Extraction failed for '{url}': {reason}")


class ParseError(WebExtractorError):
    """Raised when the trafilatura XML output cannot be parsed."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"XML parse error: {reason}")


# Data model

@dataclass(frozen=True)
class ExtractedPage:
    """
    Immutable, typed result of a successful web extraction.

    Attributes:
        url:           Source URL.
        title:         Page title from the document metadata.
        sitename:      Site name extracted by trafilatura.
        date:          Publication date (ISO string) if found, else empty string.
        text:          Clean, normalised plain-text body.
        element_count: Number of structural elements parsed (paragraphs, headings, …).
    """

    url: str
    title: str
    sitename: str
    date: str
    text: str
    element_count: int


# helper functions

def _normalize_text(text: str) -> str:
    """Collapse excess blank lines and horizontal whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _build_session(retries: int, backoff_factor: float) -> requests.Session:
    """
    Return a ``requests.Session`` pre-configured with retry logic.

    Retries are attempted only for idempotent GET/HEAD requests and only
    for HTTP status codes in ``_RETRYABLE_STATUS`` (e.g. 429, 5xx).
    Client errors (4xx, except 429) are **not** retried.
    """
    session = requests.Session()
    retry_policy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=_RETRYABLE_STATUS,
        allowed_methods={"GET", "HEAD"},
        raise_on_status=False,  
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry_policy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": _USER_AGENT})
    return session


# WebExtractor

class WebExtractor:
    """
    Runs a 4-stage pipeline to turn a URL into clean plain text:

    1. :meth:`fetch_html`          — HTTP GET with retry, timeout, and size guards
    2. :meth:`html_to_xml`         — trafilatura HTML → XML
    3. :meth:`xml_content_formattor` — XML → structured dict
    4. :meth:`format_from_xml`     — structured dict → normalised plain text

    The high-level entry point is :meth:`extract_from_url`, which runs the
    full pipeline and returns an :class:`ExtractedPage`.

    Args:
        timeout:        ``(connect_timeout, read_timeout)`` in seconds.
        retries:        Maximum retry attempts for transient HTTP errors.
        backoff_factor: Multiplier for exponential back-off between retries.
                        Actual wait = ``backoff_factor * 2 ** (attempt - 1)`` s.
    """

    def __init__(
        self,
        timeout: Tuple[int, int] = _DEFAULT_TIMEOUT,
        retries: int = _DEFAULT_RETRIES,
        backoff_factor: float = _DEFAULT_BACKOFF,
    ) -> None:
        self._timeout = timeout
        self._session = _build_session(retries, backoff_factor)

    def close(self) -> None:
        """Release the underlying HTTP session and connection pool."""
        self._session.close()

    def __enter__(self) -> "WebExtractor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # Stage 1: Fetch

    def fetch_html(self, url: str) -> str:
        """
        Perform an HTTP GET and return the response body as a string.

        Validates HTTP status, Content-Type, and response size before
        returning.  Transient errors (429, 5xx) are retried automatically
        by the underlying session adapter.

        Args:
            url: Fully-qualified ``http://`` or ``https://`` URL.

        Returns:
            Raw HTML string decoded with the charset declared by the server
            (falls back to UTF-8).
        """
        logger.debug("Fetching %s", url)
        try:
            response = self._session.get(url, timeout=self._timeout)
        except requests.Timeout:
            raise FetchError(url, f"Timed out after {self._timeout}s")
        except requests.ConnectionError as exc:
            raise FetchError(url, f"Connection error: {exc}")
        except requests.RequestException as exc:
            raise FetchError(url, f"Request error: {exc}")

        if not response.ok:
            raise FetchError(
                url,
                response.reason or "Non-2xx response",
                status_code=response.status_code,
            )

        content_type = response.headers.get("Content-Type", "")
        if not any(ct in content_type for ct in _ACCEPTED_CONTENT_TYPES):
            raise FetchError(
                url,
                f"Unsupported Content-Type '{content_type}'",
                status_code=response.status_code,
            )

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > _MAX_CONTENT_BYTES:
            raise FetchError(
                url,
                f"Response too large: {content_length} bytes "
                f"(limit {_MAX_CONTENT_BYTES // (1024 * 1024)} MB)",
            )

        logger.info("Fetched %s — %d bytes, status %d", url, len(response.content), response.status_code)
        return response.text

    # Stage 2: HTML → XML

    def html_to_xml(self, html: str, url: str = "") -> str:
        """
        Convert raw HTML to a trafilatura XML string.

        Args:
            html: Raw HTML content.
            url:  Source URL used only in error messages.

        Returns:
            Trafilatura XML as a UTF-8 string.
        """
        xml_str = trafilatura.extract(
            html,
            output_format="xml",
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        if not xml_str:
            raise ExtractionError(url or "<html>", "trafilatura returned no content")
        return xml_str

    # Stage 3: XML → structured dict

    def xml_content_formattor(self, xml_str: str) -> Dict[str, Any]:
        """
        Parse a trafilatura XML string into a structured intermediate dict.
        Args:
            xml_str: XML string produced by :meth:`html_to_xml`.

        Returns:
            Dict with keys ``title``, ``date``, ``sitename``, and ``elements``.
            Each element in ``elements`` is one of::

                {"type": "heading",   "text": str}
                {"type": "paragraph", "text": str}
                {"type": "list",      "items": List[str]}
                {"type": "table",     "text": str}
        """
        try:
            root = ET.fromstring(xml_str.encode("utf-8"))
        except ET.ParseError as exc:
            raise ParseError(str(exc)) from exc

        result: Dict[str, Any] = {
            "title":    root.get("title", ""),
            "date":     root.get("date", ""),
            "sitename": root.get("sitename", ""),
            "elements": [],
        }

        container = root.find("./main") or root

        for elem in container:
            tag = elem.tag
            text = (elem.text or "").strip()

            if tag == "p" and text:
                result["elements"].append({"type": "paragraph", "text": text})

            elif tag == "head" and text:
                result["elements"].append({"type": "heading", "text": text})

            elif tag == "list":
                items = [
                    (item.text or "").strip()
                    for item in elem.findall("item")
                    if (item.text or "").strip()
                ]
                if items:
                    result["elements"].append({"type": "list", "items": items})

            elif tag == "table":
                cells = [t.strip() for t in elem.itertext() if t.strip()]
                if cells:
                    result["elements"].append(
                        {"type": "table", "text": " | ".join(cells)}
                    )

        return result

    # ── Stage 4: structured dict → plain text ─────────────────────────────────

    def format_from_xml(self, result: Dict[str, Any], url: str = "") -> str:
        """
        Render the dict produced by :meth:`xml_content_formattor` into
        normalised plain text.

        Args:
            result: Output of :meth:`xml_content_formattor`.
            url:    Source URL used only in error messages.

        Returns:
            Normalised plain-text string.
        """
        parts: List[str] = []

        if result.get("title"):
            parts.append(result["title"])

        for elem in result.get("elements", []):
            etype = elem["type"]
            if etype in ("paragraph", "heading", "table"):
                parts.append(elem["text"])
            elif etype == "list":
                parts.append("\n".join(f"- {item}" for item in elem["items"]))

        text = _normalize_text("\n\n".join(parts))
        if not text:
            raise ExtractionError(url or "<result>", "No text content after formatting")
        return text


    def extract(self, html: str, url: str = "") -> ExtractedPage:
        """
        Run the extraction pipeline on pre-fetched HTML.

        Args:
            html: Raw HTML string.
            url:  Source URL for metadata and error context (optional).

        Returns:
            :class:`ExtractedPage` with title, sitename, date, text, and element count.
        """
        xml_str = self.html_to_xml(html, url=url)
        result = self.xml_content_formattor(xml_str)
        text = self.format_from_xml(result, url=url)
        return ExtractedPage(
            url=url,
            title=result["title"],
            sitename=result["sitename"],
            date=result["date"],
            text=text,
            element_count=len(result["elements"]),
        )

    def extract_from_url(self, url: str) -> ExtractedPage:
        """
        Fetch *url* and run the full extraction pipeline.

        Args: url

        Returns:
            :class:`ExtractedPage` with title, sitename, date, text, and element count.
        """
        html = self.fetch_html(url)
        return self.extract(html, url=url)


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    records = extract_all_urls()

    if not records:
        logger.warning("No URLs found across all CSVs.")
        sys.exit(0)

    logger.info("Processing %d unique URLs.", len(records))

    output_dir = Path("src/rag/web_extractor/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    passed: List[str] = []
    failed: List[tuple] = []

    with WebExtractor() as extractor:
        for record in records:
            url = record["url"]
            categories = record["categories"]
            try:
                page = extractor.extract_from_url(url)
                safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", url.split("//")[-1])[:80]
                output_file = output_dir / f"{safe_name}.json"
                payload = {
                    "url":        page.url,
                    "title":      page.title,
                    "sitename":   page.sitename,
                    "date":       page.date,
                    "categories": categories,
                    "source":     "website",
                    "text":       page.text,
                }
                output_file.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info(
                    "Saved '%s' → %s [%s] (%d elements)",
                    page.title or url,
                    output_file,
                    ", ".join(categories),
                    page.element_count,
                )
                passed.append(url)
            except WebExtractorError as exc:
                logger.error("Skipping %s: %s", url, exc)
                failed.append((url, str(exc)))

    logger.info(
        "\n── Summary ──────────────────────────────\n"
        "  Total   : %d\n"
        "  Passed  : %d\n"
        "  Failed  : %d\n"
        "─────────────────────────────────────────",
        len(records), len(passed), len(failed),
    )
    if failed:
        logger.warning("Failed URLs:")
        for url, reason in failed:
            logger.warning("  %s — %s", url, reason)