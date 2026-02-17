from __future__ import annotations
import re
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from dataclass import Chunk
from utils import detect_heading, split_paragraphs, estimate_tokens, _cos_sim

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def chunk_fixed(text: str, doc_id: str, target_tokens: int = 600) -> List[Chunk]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    target_words = max(50, int(target_tokens * 0.75))
    chunks: List[Chunk] = []
    start, cid = 0, 0
    while start < len(words):
        end = min(len(words), start + target_words)
        chunk_text = " ".join(words[start:end]).strip()
        chunks.append(Chunk("fixed", doc_id, f"{doc_id}::fixed::{cid}", chunk_text))
        cid += 1
        start = end
    return chunks

def chunk_fixed_overlap(text: str, doc_id: str, target_tokens: int = 600, overlap_tokens: int = 120) -> List[Chunk]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    target_words = max(50, int(target_tokens * 0.75))
    overlap_words = max(0, int(overlap_tokens * 0.75))
    step = max(10, target_words - overlap_words)

    chunks: List[Chunk] = []
    start, cid = 0, 0
    while start < len(words):
        end = min(len(words), start + target_words)
        chunk_text = " ".join(words[start:end]).strip()
        chunks.append(Chunk("fixed_overlap", doc_id, f"{doc_id}::fixed_overlap::{cid}", chunk_text))
        cid += 1
        if end == len(words):
            break
        start += step
    return chunks

def chunk_recursive(text: str, doc_id: str, target_tokens: int = 600) -> List[Chunk]:
    lines = text.splitlines()
    blocks: List[str] = []
    buf: List[str] = []

    for line in lines:
        if detect_heading(line):
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
            blocks.append(line.strip())
        else:
            buf.append(line)
    if buf:
        blocks.append("\n".join(buf).strip())

    pieces: List[str] = []
    for b in blocks:
        if detect_heading(b):
            pieces.append(b)
        else:
            pieces.extend(split_paragraphs(b))

    chunks: List[Chunk] = []
    cid = 0
    current: List[str] = []
    cur_tokens = 0

    def flush():
        nonlocal cid, current, cur_tokens
        if current:
            chunk_text = "\n\n".join([c for c in current if c.strip()]).strip()
            if chunk_text:
                chunks.append(Chunk("recursive", doc_id, f"{doc_id}::recursive::{cid}", chunk_text))
                cid += 1
        current, cur_tokens = [], 0

    for p in pieces:
        p = p.strip()
        if not p:
            continue
        ptoks = estimate_tokens(p)
        if ptoks <= target_tokens:
            if cur_tokens + ptoks > target_tokens and current:
                flush()
            current.append(p)
            cur_tokens += ptoks
        else:
            sents = sent_tokenize(p)
            if not sents:
                for ch in chunk_fixed(p, doc_id, target_tokens):
                    ch.method = "recursive"
                    ch.chunk_id = ch.chunk_id.replace("fixed", "recursive")
                    chunks.append(ch)
                continue
            for s in sents:
                stoks = estimate_tokens(s)
                if stoks > target_tokens:
                    for ch in chunk_fixed(s, doc_id, target_tokens):
                        ch.method = "recursive"
                        ch.chunk_id = ch.chunk_id.replace("fixed", "recursive")
                        chunks.append(ch)
                    continue
                if cur_tokens + stoks > target_tokens and current:
                    flush()
                current.append(s)
                cur_tokens += stoks

    flush()
    return chunks

def chunk_sentence_pack(text: str, doc_id: str, target_tokens: int = 600) -> List[Chunk]:
    sents = sent_tokenize(text)
    if not sents:
        return chunk_fixed(text, doc_id, target_tokens)

    chunks: List[Chunk] = []
    cid = 0
    current: List[str] = []
    cur_tokens = 0

    def flush():
        nonlocal cid, current, cur_tokens
        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(Chunk("sentence_pack", doc_id, f"{doc_id}::sentence_pack::{cid}", chunk_text))
                cid += 1
        current, cur_tokens = [], 0

    for s in sents:
        s = s.strip()
        if not s:
            continue
        stoks = estimate_tokens(s)
        if stoks > target_tokens:
            flush()
            for ch in chunk_fixed(s, doc_id, target_tokens):
                ch.method = "sentence_pack"
                ch.chunk_id = ch.chunk_id.replace("fixed", "sentence_pack")
                chunks.append(ch)
            continue

        if cur_tokens + stoks > target_tokens and current:
            flush()
        current.append(s)
        cur_tokens += stoks

    flush()
    return chunks



def chunk_semantic(
    text: str,
    doc_id: str,
    model: SentenceTransformer,
    target_tokens: int = 600,
    min_tokens: int = 200,
    topic_shift_threshold: float = 0.72,
    window: int = 2,
) -> List[Chunk]:
    """Semantic chunking: sentence-by-sentence, break when topic shifts."""
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if not sents:
        return chunk_fixed(text, doc_id, target_tokens)

    sent_embs = model.encode(sents, batch_size=64, show_progress_bar=False, normalize_embeddings=False)
    sent_embs = np.asarray(sent_embs, dtype=np.float32)

    chunks: List[Chunk] = []
    cid = 0

    cur_sents: List[str] = []
    cur_embs: List[np.ndarray] = []
    cur_tokens = 0

    def centroid(embs: List[np.ndarray]) -> np.ndarray:
        if not embs:
            return np.zeros((sent_embs.shape[1],), dtype=np.float32)
        return np.mean(np.stack(embs, axis=0), axis=0)

    def last_window_avg(embs: List[np.ndarray], w: int) -> np.ndarray:
        if not embs:
            return np.zeros((sent_embs.shape[1],), dtype=np.float32)
        use = embs[-w:] if len(embs) >= w else embs
        return np.mean(np.stack(use, axis=0), axis=0)

    def flush():
        nonlocal cid, cur_sents, cur_embs, cur_tokens
        if cur_sents:
            chunk_text = " ".join(cur_sents).strip()
            if chunk_text:
                chunks.append(Chunk("semantic", doc_id, f"{doc_id}::semantic::{cid}", chunk_text))
                cid += 1
        cur_sents, cur_embs, cur_tokens = [], [], 0

    for s, e in zip(sents, sent_embs):
        stoks = estimate_tokens(s)

        if stoks > target_tokens:
            flush()
            for ch in chunk_fixed(s, doc_id, target_tokens):
                ch.method = "semantic"
                ch.chunk_id = ch.chunk_id.replace("fixed", "semantic")
                chunks.append(ch)
            continue

        if cur_tokens + stoks > target_tokens and cur_sents:
            flush()

        if not cur_sents:
            cur_sents.append(s)
            cur_embs.append(e)
            cur_tokens += stoks
            continue

        cen = centroid(cur_embs)
        lw = last_window_avg(cur_embs, window)
        sim_to_cen = _cos_sim(e, cen)
        sim_to_lw = _cos_sim(e, lw)
        sim = (sim_to_cen + sim_to_lw) / 2.0

        allow_break = cur_tokens >= min_tokens
        if allow_break and sim < topic_shift_threshold:
            flush()
            cur_sents.append(s)
            cur_embs.append(e)
            cur_tokens += stoks
        else:
            cur_sents.append(s)
            cur_embs.append(e)
            cur_tokens += stoks

    flush()
    chunks = [c for c in chunks if estimate_tokens(c.text) >= 30]
    return chunks