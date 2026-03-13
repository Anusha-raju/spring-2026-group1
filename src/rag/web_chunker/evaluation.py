"""
Chunks web pages extracted by web_processor.py and saves the results as
``*_chunks.jsonl`` — the same format produced by pdf_chunker, so both
sources feed into the same embedding retrieval pipeline.
"""

from __future__ import annotations

import argparse
import csv as _csv
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

_PDF_CHUNKER_DIR = os.path.join(os.path.dirname(__file__), "..", "pdf_chunker")
sys.path.insert(0, os.path.abspath(_PDF_CHUNKER_DIR))

import nltk
from nltk.tokenize import sent_tokenize
from chunkers import chunk_sentence_pack
from dataclass import Chunk
from utils import estimate_tokens, normalize_text

# ----------------------------
# Web knowledge lookup (from website_knowledge.csv)
# Builds three maps keyed by: url, and normalized filename stem
# ----------------------------

_WEB_KNOWLEDGE_CSV = os.path.join(os.path.dirname(__file__), "..", "website_knowledge.csv")

def _parse_web_roles(role_str: str) -> List[str]:
    return [r.strip() for r in role_str.split(",") if r.strip()]

def _normalize_stem(fname: str) -> str:
    """Normalize a filename (with or without .txt) to a lowercase lookup key."""
    stem = os.path.splitext(fname)[0]
    # treat both '/' and ':' as equivalent separators (macOS uses ':' for '/')
    return stem.lower().replace("/", ":").replace("\\", ":")

def _load_web_knowledge() -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, List[str]]]:
    """
    Returns:
        url_category_map  : url  → List[role]
        file_url_map      : normalized stem → url
        file_category_map : normalized stem → List[role]
    """
    url_cats: Dict[str, List[str]] = {}
    file_url: Dict[str, str] = {}
    file_cats: Dict[str, List[str]] = {}
    if not os.path.exists(_WEB_KNOWLEDGE_CSV):
        return url_cats, file_url, file_cats
    with open(_WEB_KNOWLEDGE_CSV, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            url = row.get("Web URL", "").strip()
            role_str = row.get("Role", "").strip()
            fname = row.get("Parsed File", "").strip()
            if not role_str:
                continue
            roles = _parse_web_roles(role_str)
            if url:
                url_cats[url] = roles
            if fname:
                key = _normalize_stem(fname)
                file_url[key] = url
                file_cats[key] = roles
    return url_cats, file_url, file_cats

_WEB_CATEGORY_MAP, _FILE_TO_URL, _FILE_CATEGORY_MAP = _load_web_knowledge()

def _get_web_categories(url: str) -> List[str]:
    if url in _WEB_CATEGORY_MAP:
        return _WEB_CATEGORY_MAP[url]
    for csv_url, cats in _WEB_CATEGORY_MAP.items():
        if url and csv_url and (url in csv_url or csv_url in url):
            return cats
    return []


OPIOID_TOPICS: Dict[str, List[str]] = {
    "overdose": [
        "overdose", "unresponsive", "unconscious", "blue lips", "cyanosis",
        "stopped breathing", "not breathing", "limp", "won't wake",
    ],
    "emergency": [
        "call 911", "emergency", "immediate action", "acute", "life-threatening",
        "emergency room", "er visit", "urgent care",
    ],
    "naloxone": [
        "naloxone", "narcan", "intranasal", "nasal spray", "intramuscular",
        "opioid reversal", "antagonist",
    ],
    "withdrawal": [
        "withdrawal", "detox", "detoxification", "cravings", "taper", "tapering",
        "physical dependence", "abstinence", "discontinuation",
    ],
    "dosage": [
        "dosage", "dose", "mg", "milligram", "prescribe", "titrate", "titration",
        "twice daily", "once daily", "frequency",
    ],
    "treatment": [
        "buprenorphine", "methadone", "suboxone", "mat", "medication assisted",
        "treatment program", "opioid use disorder", "oud", "subutex",
    ],
    "prevention": [
        "harm reduction", "prevention", "safe use", "risk reduction",
        "safe storage", "disposal", "lock box", "take back",
    ],
    "mental_health": [
        "mental health", "depression", "anxiety", "ptsd", "co-occurring",
        "dual diagnosis", "counseling", "therapy", "psychiatric",
    ],
    "legal": [
        "law", "legal", "regulation", "prescription", "controlled substance",
        "dea", "schedule", "patient rights", "privacy", "hipaa",
    ],
    "patient_education": [
        "patient education", "inform patient", "tell patient", "family",
        "caregiver", "warning signs", "side effects", "what to expect",
    ],
}


def tag_chunk(text: str) -> Dict[str, Any]:
    """Multi-label opioid topic tagging. Unmatched chunks get topics=[]."""
    text_lower = text.lower()
    matched = [
        topic for topic, keywords in OPIOID_TOPICS.items()
        if any(kw in text_lower for kw in keywords)
    ]
    return {"topics": matched, "is_tagged": len(matched) > 0}

# Main processing pipeline
def load_web_pages(json_dir: str) -> List[Dict[str, Any]]:
    """
    Load all ``*.json`` files from *json_dir*.

    Returns:
        List of page dicts with keys: url, title, sitename, date,
        categories, source, text.

    Raises:
        FileNotFoundError: If no ``.json`` files are found in *json_dir*.
    """
    paths = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No .json files found in '{json_dir}'")

    pages: List[Dict[str, Any]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            page = json.load(f)
        page["_filename"] = os.path.basename(p)   # keep filename as doc_id
        pages.append(page)

    return pages


def load_web_pages_txt(txt_dir: str) -> List[Dict[str, Any]]:
    """
    Load all ``*.txt`` files from *txt_dir* (produced by extract_to_txt.py).

    Looks up URL and categories from the ``Parsed File`` column in
    website_knowledge.csv via _FILE_TO_URL and _FILE_CATEGORY_MAP.

    Returns:
        List of page dicts with keys: url, title, categories, source, text, _filename.
    """
    paths = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in '{txt_dir}'")

    pages: List[Dict[str, Any]] = []
    for p in paths:
        basename = os.path.basename(p)
        key = _normalize_stem(basename)
        url = _FILE_TO_URL.get(key, "")
        categories = _FILE_CATEGORY_MAP.get(key, [])
        stem = os.path.splitext(basename)[0]
        title = stem.replace("-", " ").replace("_", " ")

        text = open(p, "r", encoding="utf-8").read()
        pages.append({
            "_filename":  basename,
            "url":        url,
            "title":      title,
            "categories": categories,
            "source":     "website",
            "text":       text,
        })

    return pages

def chunk_page(
    page: Dict[str, Any],
    target_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Chunk a single web page and return a list of chunk records.

    Each record carries the page-level metadata (url, title, categories,
    source) so every chunk is fully self-describing.
    """
    doc_id = page["_filename"]
    text = normalize_text(page.get("text", ""))

    if not text:
        return []

    chunks: List[Chunk] = chunk_sentence_pack(text, doc_id, target_tokens)
    chunks = [c for c in chunks if estimate_tokens(c.text) >= 30]

    records: List[Dict[str, Any]] = []
    for chunk in chunks:
        tags = tag_chunk(chunk.text)
        records.append({
            "chunk_id":   chunk.chunk_id,
            "doc_id":     doc_id,
            "source":     page.get("source", "website"),
            "url":        page.get("url", ""),
            "title":      page.get("title", ""),
            "categories": page.get("categories") or _get_web_categories(page.get("url", "")),
            "text":       chunk.text,
            "token_count": estimate_tokens(chunk.text),
            "topics":     tags["topics"],
            "is_tagged":  tags["is_tagged"],
        })

    return records


""" Save chunks in the same format as pdf_chunker for downstream embedding/retrieval evaluation."""
def save_chunks(
    doc_id: str,
    chunk_records: List[Dict[str, Any]],
    out_dir: str,
) -> None:
    """Save chunk records to ``<out_dir>/<doc_id stem>_chunks.jsonl``."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(doc_id)[0]
    out_path = os.path.join(out_dir, f"{base}_chunks.jsonl")

    tagged = sum(1 for r in chunk_records if r["is_tagged"])
    with open(out_path, "w", encoding="utf-8") as f:
        for record in chunk_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"  Saved {len(chunk_records)} chunks → {out_path} "
        f"({tagged} tagged, {len(chunk_records) - tagged} untagged)"
    )


def run(json_dir: str, out_dir: str, target_tokens: int = 7000, txt_dir: str = None) -> None:
    if txt_dir:
        pages = load_web_pages_txt(txt_dir)
        print(f"Loaded {len(pages)} web pages from '{txt_dir}' (txt mode)")
    else:
        pages = load_web_pages(json_dir)
        print(f"Loaded {len(pages)} web pages from '{json_dir}'")

    total_chunks = 0
    for page in pages:
        doc_id = page["_filename"]
        print(f"\n  Processing: {doc_id}")
        chunk_records = chunk_page(page, target_tokens)
        if not chunk_records:
            print(f"    Skipped — no text content.")
            continue
        save_chunks(doc_id, chunk_records, out_dir)
        total_chunks += len(chunk_records)

    print(f"\nDone. {total_chunks} total chunks saved to '{out_dir}'")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk web pages from web_extractor outputs into JSONL."
    )
    parser.add_argument(
        "--json_dir",
        default=None,
        help="Directory containing *.json files from web_processor.py",
    )
    parser.add_argument(
        "--txt_dir",
        default="src/rag/web_extractor/outputs_txt",
        help="Directory containing *.txt files from extract_to_txt.py (preferred)",
    )
    parser.add_argument(
        "--out_dir",
        default="./out",
        help="Output directory for *_chunks.jsonl files",
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=7000,
        help="Target chunk size in tokens (default: 7000)",
    )
    args = parser.parse_args()

    try:
        sent_tokenize("test sentence.")
    except LookupError:
        nltk.download("punkt")

    run(
        json_dir=args.json_dir or "src/rag/web_extractor/outputs",
        out_dir=args.out_dir,
        target_tokens=args.target_tokens,
        txt_dir=args.txt_dir,
    )


if __name__ == "__main__":
    main()