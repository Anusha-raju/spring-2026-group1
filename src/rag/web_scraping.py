from typing import List, Dict, Optional
from lxml import etree
import trafilatura
import requests
import re
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?;:])\s+')


def extract_xml(html: str) -> str:
    xml = trafilatura.extract(
        html,
        output_format="xml",
        include_comments=False,
        include_tables=True,
        favor_precision=True
    )
    if not xml:
        raise ValueError("Trafilatura extraction failed (no XML returned).")
    return xml


def estimate_tokens(text: str) -> int:
    return len(_enc.encode(text))


def last_n_tokens(text: str, n: int) -> str:
    """Return the last n tokens (decoded) from text."""
    if n <= 0:
        return ""
    toks = _enc.encode(text)
    if not toks:
        return ""
    return _enc.decode(toks[-n:])


def split_large_text(text: str, max_tokens: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    def split_into_sentences(paragraph: str) -> List[str]:
        paragraph = paragraph.strip()
        if not paragraph:
            return []
        return [s.strip() for s in _SENT_SPLIT_RE.split(paragraph) if s.strip()]

    chunks: List[str] = []
    buffer: List[str] = []

    def flush_buffer(buf: List[str]):
        joined = " ".join(buf).strip()
        if joined:
            chunks.append(joined)

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    for p in paragraphs:
        buffer.append(p)
        if estimate_tokens(" ".join(buffer)) > max_tokens:
            buffer.pop()
            if buffer:
                flush_buffer(buffer)
                buffer = []

            if estimate_tokens(p) > max_tokens:
                sent_buf: List[str] = []
                for s in split_into_sentences(p):
                    sent_buf.append(s)
                    if estimate_tokens(" ".join(sent_buf)) > max_tokens:
                        sent_buf.pop()
                        if sent_buf:
                            flush_buffer(sent_buf)
                        sent_buf = [s]

                        if estimate_tokens(s) > max_tokens:
                            words = s.split()
                            hard_buf: List[str] = []
                            for w in words:
                                hard_buf.append(w)
                                if estimate_tokens(" ".join(hard_buf)) > max_tokens:
                                    hard_buf.pop()
                                    if hard_buf:
                                        flush_buffer(hard_buf)
                                    hard_buf = [w]
                            if hard_buf:
                                flush_buffer(hard_buf)
                            sent_buf = []

                if sent_buf:
                    flush_buffer(sent_buf)
                buffer = []
            else:
                buffer = [p]

    if buffer:
        flush_buffer(buffer)
    cleaned: List[str] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if estimate_tokens(c) <= max_tokens:
            cleaned.append(c)
        else:
            words = c.split()
            hard_buf: List[str] = []
            for w in words:
                hard_buf.append(w)
                if estimate_tokens(" ".join(hard_buf)) > max_tokens:
                    hard_buf.pop()
                    if hard_buf:
                        cleaned.append(" ".join(hard_buf))
                    hard_buf = [w]
            if hard_buf:
                cleaned.append(" ".join(hard_buf))
    return cleaned


def merge_small_remainders(parts: List[str], min_tokens: int, max_tokens: int, merge_overflow: int = 30) -> List[str]:
    merged: List[str] = []
    soft_max = max_tokens + merge_overflow

    for part in parts:
        part = (part or "").strip()
        if not part:
            continue

        if not merged:
            merged.append(part)
            continue

        ptoks = estimate_tokens(part)
        if ptoks < min_tokens:
            candidate = merged[-1] + " " + part
            if estimate_tokens(candidate) <= soft_max:
                merged[-1] = candidate
            else:
                merged.append(part)
        else:
            merged.append(part)

    return merged


def apply_overlap(parts: List[str], overlap_tokens: int, max_tokens: int, merge_overflow: int = 30) -> List[str]:
    if overlap_tokens <= 0 or len(parts) <= 1:
        return parts

    soft_max = max_tokens + merge_overflow
    out = [parts[0]]

    for i in range(1, len(parts)):
        prev = out[-1]
        cur = parts[i]

        overlap = last_n_tokens(prev, overlap_tokens).strip()
        if overlap:
            candidate = f"{overlap} {cur}".strip()
        else:
            candidate = cur.strip()
        if estimate_tokens(candidate) > soft_max and overlap_tokens > 0:
            lo, hi = 0, overlap_tokens
            best = ""
            while lo <= hi:
                mid = (lo + hi) // 2
                ov = last_n_tokens(prev, mid).strip()
                cand = f"{ov} {cur}".strip() if ov else cur.strip()
                if estimate_tokens(cand) <= soft_max:
                    best = cand
                    lo = mid + 1
                else:
                    hi = mid - 1
            candidate = best if best else cur.strip()

        out.append(candidate)

    return out


def chunk_by_heading(
    xml_text: str,
    max_tokens: int = 500,
    min_tokens: int = 100,
    overlap_tokens: int = 50,
    merge_overflow: int = 30
) -> List[Dict]:
    root = etree.fromstring(xml_text.encode("utf-8"))

    chunks: List[Dict] = []
    heading_stack: List[str] = []
    buffer: List[str] = []

    def flush():
        nonlocal buffer, chunks, heading_stack
        if not buffer:
            return

        text = " ".join(buffer).strip()
        buffer = []
        if not text:
            return

        tokens = estimate_tokens(text)
        if tokens < min_tokens:
            return

        heading_path = " > ".join(
            heading_stack) if heading_stack else "Document"

        if tokens > max_tokens:
            parts = split_large_text(text, max_tokens=max_tokens)
            parts = merge_small_remainders(
                parts, min_tokens=min_tokens, max_tokens=max_tokens, merge_overflow=merge_overflow)
            parts = apply_overlap(parts, overlap_tokens=overlap_tokens,
                                  max_tokens=max_tokens, merge_overflow=merge_overflow)

            for i, part in enumerate(parts, 1):
                chunks.append({
                    "heading_path": f"{heading_path} (part {i})",
                    "text": part,
                    "tokens": estimate_tokens(part)
                })
        else:
            chunks.append({
                "heading_path": heading_path,
                "text": text,
                "tokens": tokens
            })

    for elem in root.iter():
        tag = etree.QName(elem).localname

        if tag == "head" and elem.text and elem.text.strip():
            flush()
            heading_text = elem.text.strip()
            heading_stack = heading_stack[:1] if heading_stack else []
            heading_stack.append(heading_text)

        elif tag in {"p", "item"} and elem.text and elem.text.strip():
            buffer.append(elem.text.strip())

    flush()
    return chunks


def extract_and_chunk(
    html: str,
    url: Optional[str] = None,
    max_tokens: int = 150,
    min_tokens: int = 80,
    overlap_tokens: int = 50,
    merge_overflow: int = 30
) -> List[Dict]:
    xml = extract_xml(html)
    chunks = chunk_by_heading(
        xml,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        overlap_tokens=overlap_tokens,
        merge_overflow=merge_overflow
    )

    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"chunk_{i}"
        chunk["source_url"] = url

    return chunks


if __name__ == "__main__":
    url = "https://www.chcs.org/resource/a-federally-qualified-health-center-and-certified-community-behavioral-health-clinic-partnership-in-rural-missouri/"

    html = trafilatura.fetch_url(url)
    if not html:
        html = requests.get(url, timeout=15).text

    MAX_TOKENS = 400
    MIN_TOKENS = 100
    OVERLAP_TOKENS = 40
    MERGE_OVERFLOW = 20

    chunks = extract_and_chunk(
        html,
        url=url,
        max_tokens=MAX_TOKENS,
        min_tokens=MIN_TOKENS,
        overlap_tokens=OVERLAP_TOKENS,
        merge_overflow=MERGE_OVERFLOW
    )

    print(f"\nTotal chunks: {len(chunks)}\n")

    for c in chunks:
        print("=" * 90)
        print("HEADING:", c["heading_path"])
        print("TOKENS:", c["tokens"])
        print(c["text"])
