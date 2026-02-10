from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

from web_processor import (
    estimate_tokens,
    split_by_paragraphs,
    merge_small_remainders,
    apply_overlap,
)


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
LIST_RE = re.compile(r"^\s*(?:[-*•‣–—]|\d+[\.)])\s+")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$")


def parse_markdown_sections(md_text: str) -> List[Tuple[str, str]]:
    """
    Parse markdown into sections of (heading_path, text).
    """
    lines = md_text.splitlines()
    heading_stack: List[Tuple[int, str]] = []
    buffer: List[str] = []
    table_buf: List[str] = []
    sections: List[Tuple[str, str]] = []

    def current_heading_path() -> str:
        return " > ".join(h for _, h in heading_stack) if heading_stack else "Document"

    def flush():
        nonlocal buffer
        nonlocal table_buf
        if table_buf:
            buffer.append("\n".join(table_buf))
            table_buf = []
        if not buffer:
            return
        text = "\n".join(buffer).strip()
        buffer = []
        if not text:
            return
        sections.append((current_heading_path(), text))

    for raw in lines:
        line = raw.rstrip()

        # Table handling: collect contiguous table lines as a single block
        if "|" in line:
            if HEADING_RE.match(line):
                flush()
            else:
                table_buf.append(line)
                continue

        m = HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading_text))
            continue

        if not line.strip():
            flush()
            continue

        if LIST_RE.match(line):
            buffer.append(f"• {LIST_RE.sub('', line).strip()}")
        else:
            buffer.append(line.strip())

    flush()
    return sections


def chunk_sections(
    sections: List[Tuple[str, str]],
    max_tokens: int,
    min_tokens: int,
    overlap_tokens: int,
    merge_overflow: int,
) -> List[Dict]:
    chunks: List[Dict] = []
    pending_small: Dict | None = None

    def maybe_merge_small(chunk: Dict):
        nonlocal pending_small, chunks
        if chunk["tokens"] >= min_tokens:
            if pending_small:
                merged_text = pending_small["text"] + "\n" + chunk["text"]
                merged_tokens = estimate_tokens(merged_text)
                if merged_tokens <= max_tokens + merge_overflow:
                    chunk = {
                        "heading_path": pending_small["heading_path"],
                        "text": merged_text,
                        "tokens": merged_tokens,
                    }
                    pending_small = None
                else:
                    chunks.append(pending_small)
                    pending_small = None
            chunks.append(chunk)
            return

        if pending_small is None:
            pending_small = chunk
        else:
            merged_text = pending_small["text"] + "\n" + chunk["text"]
            merged_tokens = estimate_tokens(merged_text)
            if merged_tokens <= max_tokens + merge_overflow:
                pending_small = {
                    "heading_path": pending_small["heading_path"],
                    "text": merged_text,
                    "tokens": merged_tokens,
                }
            else:
                chunks.append(pending_small)
                pending_small = chunk

    def is_table_block(text: str) -> bool:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return False
        if not all("|" in ln for ln in lines[:2]):
            return False
        return bool(TABLE_SEP_RE.match(lines[1]))

    def split_table_block(text: str) -> List[str]:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return [text]
        header = lines[0]
        sep = lines[1]
        rows = lines[2:]
        if not rows:
            return [text]

        table_chunks: List[str] = []
        buf: List[str] = []

        def flush():
            if not buf:
                return
            table_chunks.append("\n".join([header, sep] + buf))

        for row in rows:
            buf.append(row)
            if estimate_tokens("\n".join([header, sep] + buf)) > max_tokens + merge_overflow:
                buf.pop()
                if buf:
                    flush()
                buf = [row]

            if estimate_tokens("\n".join([header, sep] + buf)) > max_tokens + merge_overflow:
                table_chunks.append("\n".join([header, sep, row]))
                buf = []

        flush()
        return table_chunks

    for heading_path, text in sections:
        tokens = estimate_tokens(text)
        if is_table_block(text):
            if pending_small:
                chunks.append(pending_small)
                pending_small = None
            table_parts = split_table_block(text)
            for i, part in enumerate(table_parts, 1):
                chunks.append(
                    {
                        "heading_path": f"{heading_path} (table {i})",
                        "text": part,
                        "tokens": estimate_tokens(part),
                    }
                )
            continue
        if tokens > max_tokens:
            parts = split_by_paragraphs(
                text,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                merge_overflow=merge_overflow,
            )
            parts = merge_small_remainders(
                parts,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                merge_overflow=merge_overflow,
            )
            parts = apply_overlap(
                parts,
                overlap_tokens=overlap_tokens,
                max_tokens=max_tokens,
                merge_overflow=merge_overflow,
            )
            for i, part in enumerate(parts, 1):
                maybe_merge_small(
                    {
                        "heading_path": f"{heading_path} (part {i})",
                        "text": part,
                        "tokens": estimate_tokens(part),
                    }
                )
        else:
            maybe_merge_small(
                {
                    "heading_path": heading_path,
                    "text": text,
                    "tokens": tokens,
                }
            )
    if pending_small:
        chunks.append(pending_small)
    return chunks


def chunk_markdown_file(
    md_path: str | Path,
    out_path: str | Path,
    max_tokens: int = 400,
    min_tokens: int = 100,
    overlap_tokens: int = 40,
    merge_overflow: int = 20,
) -> int:
    md_path = Path(md_path)
    if not md_path.exists():
        print(f"File not found: {md_path}")
        return 2

    md_text = md_path.read_text(encoding="utf-8")
    sections = parse_markdown_sections(md_text)
    chunks = chunk_sections(
        sections,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        overlap_tokens=overlap_tokens,
        merge_overflow=merge_overflow,
    )

    out_path = Path(out_path)
    os.makedirs(out_path.parent, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(f"{c}\n")

    print(f"Wrote {len(chunks)} chunks to {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk a Markdown file.")
    parser.add_argument("md_path", help="Path to .md file")
    parser.add_argument("--out", default="output/md_chunks.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--min-tokens", type=int, default=100)
    parser.add_argument("--overlap-tokens", type=int, default=40)
    parser.add_argument("--merge-overflow", type=int, default=20)
    args = parser.parse_args()

    md_path = Path(args.md_path)
    if not md_path.exists():
        print(f"File not found: {md_path}")
        return 2

    return chunk_markdown_file(
        md_path=md_path,
        out_path=args.out,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        overlap_tokens=args.overlap_tokens,
        merge_overflow=args.merge_overflow,
    )


if __name__ == "__main__":
    raise SystemExit(main())
