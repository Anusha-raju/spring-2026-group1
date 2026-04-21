import os, re, json, math, glob, argparse
from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from utils import estimate_tokens, normalize_text, embed_texts
from dataclass import Chunk
from chunkers import chunk_fixed, chunk_fixed_overlap, chunk_recursive, chunk_sentence_pack, chunk_semantic
from vectorindex import VectorIndex
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def contains_answer(chunk_text: str, qspec: Dict[str, Any]) -> bool:
    """
    question.json answer criteria (pick one per question):
      - "answer": exact snippet (case-insensitive substring)
      - "keywords": list of keywords, ALL must appear (case-insensitive)
      - "regex": regex pattern (case-insensitive)
    """
    t = chunk_text.lower()

    if isinstance(qspec.get("answer"), str) and qspec["answer"].strip():
        return qspec["answer"].strip().lower() in t

    if isinstance(qspec.get("keywords"), list) and qspec["keywords"]:
        kws = [str(k).strip().lower() for k in qspec["keywords"] if str(k).strip()]
        return bool(kws) and all(kw in t for kw in kws)

    if isinstance(qspec.get("regex"), str) and qspec["regex"].strip():
        return re.search(qspec["regex"], chunk_text, flags=re.IGNORECASE | re.MULTILINE) is not None

    # fallback alias
    if isinstance(qspec.get("preferred_answer"), str) and qspec["preferred_answer"].strip():
        return qspec["preferred_answer"].strip().lower() in t

    return False


# ----------------------------
# I/O
# ----------------------------

def load_txt_corpus(txt_dir: str) -> Dict[str, str]:
    paths = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {txt_dir}")
    corpus: Dict[str, str] = {}
    for p in paths:
        doc_id = os.path.basename(p)
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            corpus[doc_id] = normalize_text(f.read())
    return corpus

def load_questions(path: str) -> List[Dict[str, Any]]:
    """
    Accepts either:
      - a list of question objects
      - or {"questions": [ ... ]}

    Expected schema per question:
      - "query": str
      - "relevant_keywords": [str, ...]
      - "min_keyword_matches": int (default 1)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["questions"] if isinstance(data, dict) and "questions" in data else data
    if not isinstance(items, list):
        raise ValueError("question.json must be a list or a dict with 'questions' list.")

    out = []
    for i, q in enumerate(items):
        if not isinstance(q, dict) or "query" not in q:
            raise ValueError(f"Invalid question at index {i}: must contain 'query'.")

        q2 = dict(q)
        q2.setdefault("id", f"Q{i+1}")
        q2.setdefault("relevant_keywords", [])
        q2.setdefault("min_keyword_matches", 1)

        if not isinstance(q2["relevant_keywords"], list):
            raise ValueError(f"Question {q2['id']} relevant_keywords must be a list.")
        if not isinstance(q2["min_keyword_matches"], int):
            raise ValueError(f"Question {q2['id']} min_keyword_matches must be an int.")

        out.append(q2)

    return out

def keyword_threshold_hit(chunk_text: str, qspec: Dict[str, Any]) -> Tuple[bool, int, List[str]]:
    """
    Hit if >= min_keyword_matches keywords found in chunk.
    Robust to punctuation/hyphens by normalizing both keyword and text.
    """
    # normalize: lowercase, replace non-alphanum with spaces
    def norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    text = norm(chunk_text)
    kws_raw = qspec.get("relevant_keywords", [])
    kws = [norm(str(k)) for k in kws_raw if str(k).strip()]
    min_matches = int(qspec.get("min_keyword_matches", 1))

    matched = []
    for kw in kws:
        if not kw:
            continue
        # word boundary-ish match (on normalized text where words are separated by spaces)
        # This prevents "link" matching "blink" after normalization.
        if re.search(rf"(^| )({re.escape(kw)})( |$)", text):
            matched.append(kw)

    return (len(matched) >= min_matches, len(matched), matched)



# ----------------------------
# Build chunks per method
# ----------------------------

def build_chunks_for_method(
    method: str,
    corpus: Dict[str, str],
    target_tokens: int,
    model: SentenceTransformer,
) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc_id, text in corpus.items():
        if method == "fixed":
            all_chunks.extend(chunk_fixed(text, doc_id, target_tokens))
        elif method == "fixed_overlap":
            all_chunks.extend(chunk_fixed_overlap(text, doc_id, target_tokens, overlap_tokens=120))
        elif method == "recursive":
            all_chunks.extend(chunk_recursive(text, doc_id, target_tokens))
        elif method == "sentence_pack":
            all_chunks.extend(chunk_sentence_pack(text, doc_id, target_tokens))
        elif method == "semantic":
            all_chunks.extend(
                chunk_semantic(
                    text,
                    doc_id,
                    model=model,
                    target_tokens=target_tokens,
                    min_tokens=200,
                    topic_shift_threshold=0.72,
                    window=2,
                )
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    # Filter super tiny fragments
    return [c for c in all_chunks if estimate_tokens(c.text) >= 30]


# ----------------------------
# Experiment runner
# ----------------------------

def run_experiment(
    txt_dir: str,
    question_json: str,
    out_dir: str,
    k: int = 3,
    target_tokens: int = 600,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    corpus = load_txt_corpus(txt_dir)
    questions = load_questions(question_json)

    model = SentenceTransformer(embed_model_name)
    embed_dim = model.get_sentence_embedding_dimension()

    methods = ["fixed", "fixed_overlap", "recursive", "sentence_pack", "semantic"]

    # Build 5 indexes
    indexes: Dict[str, VectorIndex] = {}
    for m in methods:
        print(f"\n=== Building index for: {m} ===")
        chunks = build_chunks_for_method(m, corpus, target_tokens, model=model)
        print(f"Chunks: {len(chunks)}")
        embs = embed_texts(model, [c.text for c in chunks])
        idx = VectorIndex(embed_dim)
        idx.add(embs, chunks)
        indexes[m] = idx

    # Run queries + score Hit@k
    all_rows: List[Dict[str, Any]] = []

    for q in questions:
        qid = q["id"]
        qtext = q["query"]

        q_emb = embed_texts(model, [qtext], batch_size=1)[0]

        rows = []
        for m in methods:
            results = indexes[m].search(q_emb, top_k=k)

            hit = 0
            first_rank = ""
            evidence = ""
            match_count = 0
            matched_keywords = []

            for rank, (chunk, _score) in enumerate(results, start=1):
                ok, cnt, matched = keyword_threshold_hit(chunk.text, q)
                if ok:
                    hit = 1
                    first_rank = rank
                    evidence = chunk.chunk_id
                    match_count = cnt
                    matched_keywords = matched
                    break


            top1_preview = (results[0][0].text[:200].replace("\n", " ") + "...") if results else ""

            rows.append({
                "question_id": qid,
                "question": qtext,
                "method": m,
                "k": k,
                "target_tokens": target_tokens,
                "hit_at_k": hit,
                "first_hit_rank": first_rank,
                "evidence_chunk_id": evidence,
                "top1_preview": top1_preview,
                "matched_keyword_count": match_count,
                "matched_keywords": "|".join(matched_keywords),

            })

        dfq = pd.DataFrame(rows)
        dfq.to_csv(os.path.join(out_dir, f"{qid}_hit@{k}.csv"), index=False)
        all_rows.extend(rows)
        print(f"Wrote: {os.path.join(out_dir, f'{qid}_hit@{k}.csv')}")

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(out_dir, f"all_results_hit@{k}.csv"), index=False)

    summary = (
        df_all.groupby("method", as_index=False)
        .agg(hits=("hit_at_k", "sum"), questions=("question_id", "nunique"))
    )

    hits_only = df_all[df_all["hit_at_k"] == 1].copy()
    if not hits_only.empty:
        hits_only["first_hit_rank"] = pd.to_numeric(hits_only["first_hit_rank"], errors="coerce")
        rank_summary = hits_only.groupby("method", as_index=False).agg(avg_first_hit_rank=("first_hit_rank", "mean"))
        summary = summary.merge(rank_summary, on="method", how="left")
    else:
        summary["avg_first_hit_rank"] = np.nan

    summary = summary.sort_values(["hits", "avg_first_hit_rank"], ascending=[False, True])
    summary.to_csv(os.path.join(out_dir, f"summary_hit@{k}.csv"), index=False)

    print("\nDone. Outputs:")
    print(f"- {os.path.join(out_dir, f'all_results_hit@{k}.csv')}")
    print(f"- {os.path.join(out_dir, f'summary_hit@{k}.csv')}")
    print(f"- One CSV per question: {os.path.join(out_dir, 'Q*_hit@k.csv')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_dir", required=True, help="Folder containing *.txt files.")
    parser.add_argument("--question_json", required=True, help="Path to question.json.")
    parser.add_argument("--out_dir", default="./out", help="Output folder for CSV tables.")
    parser.add_argument("--k", type=int, default=3, help="Top-k retrieval (you want 3).")
    parser.add_argument("--target_tokens", type=int, default=600, help="Target chunk size (approx tokens).")
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL, help="SentenceTransformer model.")
    args = parser.parse_args()

    # Ensure NLTK punkt is present
    try:
        _ = sent_tokenize("Test. Another test.")
    except LookupError:
        nltk.download("punkt")

    run_experiment(
        txt_dir=args.txt_dir,
        question_json=args.question_json,
        out_dir=args.out_dir,
        k=args.k,
        target_tokens=args.target_tokens,
        embed_model_name=args.embed_model,
    )

if __name__ == "__main__":
    main()