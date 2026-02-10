from typing import List, Dict, Optional
from lxml import etree
import trafilatura
import re
import tiktoken
import os

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


def split_by_paragraphs(
    text: str,
    max_tokens: int,
    min_tokens: int,
    merge_overflow: int = 30
) -> List[str]:
    """
    Split text strictly on paragraph boundaries, only falling back to
    finer splits when a single paragraph exceeds max_tokens.
    For large paragraphs, split on sentence boundaries and try to avoid
    tiny remainders by choosing the nearest sentence end under max_tokens.
    """
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: List[str] = []
    buffer: List[str] = []

    def flush(buf: List[str]):
        joined = " ".join(buf).strip()
        if joined:
            chunks.append(joined)

    for p in paragraphs:
        buffer.append(p)
        if estimate_tokens(" ".join(buffer)) > max_tokens:
            buffer.pop()
            if buffer:
                flush(buffer)
                buffer = []

            # Single paragraph too large; fall back to finer splitting
            if estimate_tokens(p) > max_tokens:
                chunks.extend(split_large_text_near_max(
                    p,
                    max_tokens=max_tokens,
                    min_tokens=min_tokens,
                    merge_overflow=merge_overflow
                ))
                buffer = []
            else:
                buffer = [p]

    if buffer:
        flush(buffer)

    return [c for c in chunks if c.strip()]


def split_large_text_near_max(
    text: str,
    max_tokens: int,
    min_tokens: int,
    merge_overflow: int = 30
) -> List[str]:
    """
    Split large text by sentence boundaries, selecting the nearest sentence
    end under max_tokens. If the last chunk is too small, rebalance by
    moving sentences from the previous chunk when possible.
    """
    text = (text or "").strip()
    if not text:
        return []

    def split_into_sentences(paragraph: str) -> List[str]:
        paragraph = paragraph.strip()
        if not paragraph:
            return []
        return [s.strip() for s in _SENT_SPLIT_RE.split(paragraph) if s.strip()]

    def split_into_clauses(sentence: str) -> List[str]:
        sentence = sentence.strip()
        if not sentence:
            return []
        return [sentence]

    bullet_re = re.compile(r'^\s*(?:[-*•‣–—]|\\d+[\\.)])\\s+')
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    has_bullets = any(bullet_re.match(ln) for ln in lines)

    if has_bullets:
        units = lines
    else:
        units = split_into_sentences(text)

    if not units:
        return []

    soft_max = max_tokens + merge_overflow
    chunks: List[List[str]] = []
    cur: List[str] = []

    for s in units:
        cur.append(s)
        cur_tokens = estimate_tokens(" ".join(cur))
        if cur_tokens > soft_max:
            cur.pop()
            if cur:
                chunks.append(cur)
            cur = [s]
            cur_tokens = estimate_tokens(" ".join(cur))

        if cur_tokens > soft_max:
            # If a single unit (sentence or bullet) is too long, keep it intact.
            s_tokens = estimate_tokens(s)
            chunks.append([s])
            cur = []

    if cur:
        chunks.append(cur)

    if len(chunks) >= 2:
        last = chunks[-1]
        prev = chunks[-2]
        last_tokens = estimate_tokens(" ".join(last))
        if last_tokens < min_tokens and prev:
            while prev and last_tokens < min_tokens:
                moved = prev.pop()
                last.insert(0, moved)
                last_tokens = estimate_tokens(" ".join(last))
            if not prev:
                chunks[-2] = []

        chunks = [c for c in chunks if c]

    return [" ".join(c).strip() for c in chunks if " ".join(c).strip()]


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
    bullet_re = re.compile(r'^\s*(?:[-*•‣–—]|\d+[\.)])\s+')
    out = [parts[0]]

    def sentence_overlap(text: str, budget: int) -> str:
        sentences = [s.strip()
                     for s in _SENT_SPLIT_RE.split(text) if s.strip()]
        if not sentences:
            return ""
        picked: List[str] = []
        total = 0
        for s in reversed(sentences):
            st = estimate_tokens(s)
            if total + st > budget:
                break
            picked.append(s)
            total += st
        if not picked:
            return ""
        return " ".join(reversed(picked)).strip()

    def bullet_overlap(text: str, budget: int) -> str:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        bullet_lines = [ln for ln in lines if bullet_re.match(ln)]
        if not bullet_lines:
            return ""
        picked: List[str] = []
        total = 0
        for ln in reversed(bullet_lines):
            lt = estimate_tokens(ln)
            if total + lt > budget:
                break
            picked.append(ln)
            total += lt
        if not picked:
            return ""
        return "\n".join(reversed(picked)).strip()

    for i in range(1, len(parts)):
        prev = out[-1]
        cur = parts[i]

        overlap = bullet_overlap(
            prev, overlap_tokens) or sentence_overlap(prev, overlap_tokens)
        if not overlap:
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
    heading_stack: List[tuple] = []
    buffer: List[str] = []

    def flush():
        nonlocal buffer, chunks, heading_stack
        if not buffer:
            return

        text = "\n".join(buffer).strip()
        buffer = []
        if not text:
            return

        tokens = estimate_tokens(text)
        if tokens < min_tokens:
            return

        heading_path = " > ".join(
            h for _, h in heading_stack) if heading_stack else "Document"

        if tokens > max_tokens:
            parts = split_by_paragraphs(
                text,
                max_tokens=max_tokens,
                min_tokens=min_tokens,
                merge_overflow=merge_overflow
            )
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
            rend = (elem.get("rend") or "").lower()
            level = None
            if rend.startswith("h") and rend[1:].isdigit():
                level = int(rend[1:])
            if level is None:
                heading_stack = [(1, heading_text)]
            else:
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading_text))

        elif tag in {"p", "item"} and elem.text and elem.text.strip():
            line = elem.text.strip()
            if tag == "item":
                line = f"• {line}"
            buffer.append(line)

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
    xml_file = "outputs/xml_file.xml"
    dir_path = os.path.dirname(xml_file)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(xml_file, "a", encoding="utf-8") as f:
        f.write(f"{xml}\n")
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
