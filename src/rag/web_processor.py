import re
import trafilatura
from pathlib import Path


def normalize_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


class WebExtractor:
    def extract(self, html: str) -> str:
        """Extract clean plain text from raw HTML."""
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        if not text:
            raise ValueError("Trafilatura extraction failed (no text returned).")
        return normalize_text(text)

    def extract_from_url(self, url: str) -> str:
        """Fetch a URL and extract clean plain text."""
        html = trafilatura.fetch_url(url)
        if not html:
            raise ValueError(f"Failed to fetch URL: {url}")
        return self.extract(html)


if __name__ == "__main__":
    urls = [
        # Add URLs to extract here
    ]

    output_dir = Path("src/rag/web_extractor/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = WebExtractor()

    for url in urls:
        try:
            text = extractor.extract_from_url(url)
            # Use domain+path as filename
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', url.split("//")[-1])[:80]
            output_file = output_dir / f"{safe_name}.txt"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved: {output_file}")
        except ValueError as e:
            print(f"Error extracting {url}: {e}")
