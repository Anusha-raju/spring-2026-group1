#!/usr/bin/env python3
"""
Clean parsed PDF text files in-place.

Issues addressed:
  1. <!-- image --> placeholders
  2. /gid00048/-style OCR-encoded characters
  3. Doubled-letter OCR artifacts (e.g. "N NA AR RC CO OT" → "NARCOTIC")
  4. Markdown table noise (lines that are mostly pipes/dashes)
  5. Excessive blank lines (collapse to max 2)
"""

import glob
import os
import re

INPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def remove_image_placeholders(text: str) -> str:
    return re.sub(r"<!--\s*image\s*-->", "", text, flags=re.IGNORECASE)


def remove_gid_chars(text: str) -> str:
    """Remove gidXXXXX OCR-encoded character sequences (with or without surrounding slashes)."""
    # Remove whole lines that are predominantly gid tokens
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        # If >50% of non-space content is gid tokens, drop the line
        gid_chars = len(re.findall(r"gid\d+", line))
        if gid_chars > 0:
            total_tokens = len(line.split())
            if total_tokens == 0 or gid_chars / total_tokens > 0.4:
                continue
        # Otherwise strip individual gid tokens
        line = re.sub(r"/?gid\d+/?", "", line).strip()
        if line:
            cleaned.append(line)
        else:
            cleaned.append("")
    return "\n".join(cleaned)


def fix_doubled_letters(text: str) -> str:
    """
    Fix OCR doubled-letter artifacts.
    Handles both space-separated ('N NA AR RC CO') and contiguous ('NAARRCCOO') forms.
    """
    def fix_contiguous(line: str) -> str:
        """Fix contiguous doubled letters: NAARRCCOOTTIIC → NARCOTIC."""
        # Only attempt if line is all caps and has repeated adjacent chars
        if not re.match(r"^[A-Z\s]+$", line):
            return line
        # Collapse any char repeated 2+ times consecutively to single
        return re.sub(r"(.)\1+", r"\1", line)

    def fix_spaced(line: str) -> str:
        """Fix space-separated doubled tokens: N NA AR RC CO OT → NARCOTIC."""
        tokens = line.split()
        if len(tokens) < 3:
            return line
        doubled = [t for t in tokens if re.match(r"^[A-Z]{1,2}$", t)]
        if len(doubled) / max(len(tokens), 1) < 0.6:
            return line
        pairs = [t for t in tokens if re.match(r"^[A-Z]{2}$", t)]
        return "".join(pairs) if pairs else "".join(doubled)

    lines = text.splitlines()
    result = []
    for line in lines:
        line = fix_spaced(line)
        line = fix_contiguous(line)
        result.append(line)
    return "\n".join(result)


def remove_table_noise(text: str) -> str:
    """Remove lines that are mostly markdown table separators, page markers, or empty table rows."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        # Remove === Page N === markers
        if re.match(r"^={3,}\s*Page\s*\d+\s*={3,}$", stripped, re.IGNORECASE):
            continue
        # Remove [Table N] markers
        if re.match(r"^\[Table\s*\d+\]$", stripped, re.IGNORECASE):
            continue
        # Remove lines that are only pipes and spaces (empty table rows)
        if re.match(r"^[\|\s]+$", stripped):
            continue
        non_table = re.sub(r"[\|\-\+\s\.]", "", stripped)
        # If less than 20% of chars are actual content, it's table noise
        if len(non_table) / max(len(stripped), 1) < 0.2:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def collapse_blank_lines(text: str) -> str:
    """Collapse 3+ consecutive blank lines to 2."""
    return re.sub(r"\n{3,}", "\n\n", text)


def clean_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        original = f.read()

    text = original
    text = remove_image_placeholders(text)
    text = remove_gid_chars(text)
    text = fix_doubled_letters(text)
    text = remove_table_noise(text)
    text = collapse_blank_lines(text)
    text = text.strip() + "\n"

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