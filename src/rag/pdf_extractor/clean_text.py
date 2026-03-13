#!/usr/bin/env python3
import glob
import os
import re

INPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def remove_image_placeholders(text: str) -> str:
    return re.sub(r"<!--\s*image\s*-->", "", text, flags=re.IGNORECASE)


def remove_page_markers(text: str) -> str:
    """Remove === Page N === markers inserted by the PDF extractor."""
    return re.sub(r"={3,}\s*Page\s*\d+\s*={3,}\n?", "", text, flags=re.IGNORECASE)


def remove_running_headers_footers(text: str) -> str:
    """Remove lines that repeat 5+ times — running headers/footers from PDF pagination."""
    from collections import Counter
    lines = text.splitlines()
    counts = Counter(l.strip() for l in lines if l.strip())
    # Only remove short-to-medium repeated lines (headers/footers, not section content)
    noise = {line for line, count in counts.items() if count >= 20 and len(line) <= 20}
    cleaned = [l for l in lines if l.strip() not in noise]
    return "\n".join(cleaned)


def clean_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        original = f.read()

    text = remove_image_placeholders(original)
    text = remove_page_markers(text)
    text = remove_running_headers_footers(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"

    chars_removed = len(original) - len(text)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    return {"file": os.path.basename(path), "chars_removed": chars_removed}


def main():
    txt_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    if not txt_files:
        print(f"No .txt files found in {INPUT_DIR}")
        return

    print(f"Cleaning {len(txt_files)} files in {INPUT_DIR}\n")
    total_removed = 0

    for path in txt_files:
        result = clean_file(path)
        status = f"  -{result['chars_removed']:>6} chars" if result["chars_removed"] > 0 else "  (no change)"
        print(f"  {result['file'][:60]:<60} {status}")
        total_removed += result["chars_removed"]

    print(f"\nDone. Total characters removed: {total_removed:,}")


if __name__ == "__main__":
    main()