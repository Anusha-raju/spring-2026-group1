import os, re, json, glob, argparse, time
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

from utils import estimate_tokens, normalize_text, embed_texts
from dataclass import Chunk
from chunkers import (
    chunk_fixed,
    chunk_fixed_overlap,
    chunk_recursive,
    chunk_sentence_pack,
    chunk_semantic,
)
from vectorindex import VectorIndex

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



def keyword_threshold_hit(chunk_text: str, qspec: Dict[str, Any]) -> Tuple[bool, int, List[str]]:
    """Hit if >= min_keyword_matches keywords found in chunk.

    Robust to punctuation/hyphens by normalizing both keyword and text.

    Expected schema per question:
      - "relevant_keywords": [str, ...]
      - "min_keyword_matches": int (default 1)
    """
    def norm(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    text = norm(chunk_text)
    kws_raw = qspec.get("relevant_keywords", [])
    kws = [norm(str(k)) for k in kws_raw if str(k).strip()]
    min_matches = int(qspec.get("min_keyword_matches", 1))

    matched: List[str] = []
    for kw in kws:
        if not kw:
            continue
        if re.search(rf"(^| )({re.escape(kw)})( |$)", text):
            matched.append(kw)

    return (len(matched) >= min_matches, len(matched), matched)



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

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data["questions"] if isinstance(data, dict) and "questions" in data else data
    if not isinstance(items, list):
        raise ValueError("question.json must be a list or a dict with 'questions' list.")

    out: List[Dict[str, Any]] = []
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
            overlap_tokens = max(50, int(round(target_tokens * 0.2)))
            all_chunks.extend(chunk_fixed_overlap(text, doc_id, target_tokens, overlap_tokens=overlap_tokens))
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
                    min_tokens=max(60, int(round(target_tokens * 0.33))),
                    topic_shift_threshold=0.72,
                    window=2,
                )
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    return [c for c in all_chunks if estimate_tokens(c.text) >= 30]



def build_indexes(
    methods: List[str],
    corpus: Dict[str, str],
    target_tokens: int,
    model: SentenceTransformer,
) -> Tuple[Dict[str, VectorIndex], Dict[str, Dict[str, Any]]]:
    """Build a FAISS index per method for a given target_tokens.

    Returns:
      - indexes: method -> VectorIndex
      - meta: method -> {num_chunks, build_time_s, avg_chunk_tokens, p95_chunk_tokens}
    """
    indexes: Dict[str, VectorIndex] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    embed_dim = model.get_sentence_embedding_dimension()

    for m in methods:
        print(f"\n=== Building index for: {m} (target_tokens={target_tokens}) ===")
        t0 = time.time()
        chunks = build_chunks_for_method(m, corpus, target_tokens, model=model)
        build_chunks_time = time.time() - t0

        print(f"Chunks: {len(chunks)}  (chunk_build_time_s={build_chunks_time:.2f})")

        # embeddings + faiss build
        t1 = time.time()
        embs = embed_texts(model, [c.text for c in chunks])
        idx = VectorIndex(embed_dim)
        idx.add(embs, chunks)
        build_index_time = time.time() - t1

        tok_lens = [estimate_tokens(c.text) for c in chunks] if chunks else [0]
        meta[m] = {
            "num_chunks": len(chunks),
            "chunk_build_time_s": build_chunks_time,
            "index_build_time_s": build_index_time,
            "avg_chunk_tokens": float(np.mean(tok_lens)) if tok_lens else 0.0,
            "p95_chunk_tokens": float(np.percentile(tok_lens, 95)) if tok_lens else 0.0,
        }
        indexes[m] = idx

    return indexes, meta


def run_grid_experiment(
    txt_dir: str,
    question_json: str,
    out_dir: str,
    ks: List[int],
    target_tokens_list: List[int],
    embed_model_name: str = DEFAULT_EMBED_MODEL,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    corpus = load_txt_corpus(txt_dir)
    questions = load_questions(question_json)

    model = SentenceTransformer(embed_model_name)
    methods = ["fixed", "fixed_overlap", "recursive", "sentence_pack", "semantic"]

    all_rows: List[Dict[str, Any]] = []

    for target_tokens in target_tokens_list:
        indexes, meta = build_indexes(methods, corpus, target_tokens=target_tokens, model=model)

        for k in ks:
            print(f"\n--- Evaluating: target_tokens={target_tokens}  top_k={k} ---")

            stats = {m: {"hits": 0, "first_ranks": [], "lat_ms": []} for m in methods}

            for q in questions:
                qid = q["id"]
                qtext = q["query"]
                q_emb = embed_texts(model, [qtext], batch_size=1)[0]

                for m in methods:
                    t0 = time.time()
                    results = indexes[m].search(q_emb, top_k=k)
                    stats[m]["lat_ms"].append((time.time() - t0) * 1000.0)

                    hit = 0
                    first_rank: Optional[int] = None
                    match_count = 0
                    matched_keywords: List[str] = []
                    evidence = ""

                    for rank, (chunk, _score) in enumerate(results, start=1):
                        ok, cnt, matched = keyword_threshold_hit(chunk.text, q)
                        if ok:
                            hit = 1
                            first_rank = rank
                            evidence = chunk.chunk_id
                            match_count = cnt
                            matched_keywords = matched
                            break

                    stats[m]["hits"] += hit
                    if first_rank is not None:
                        stats[m]["first_ranks"].append(first_rank)

                    top1_preview = (results[0][0].text[:200].replace("\n", " ") + "...") if results else ""

                    all_rows.append({
                        "question_id": qid,
                        "question": qtext,
                        "method": m,
                        "k": k,
                        "target_tokens": target_tokens,
                        "hit_at_k": hit,
                        "first_hit_rank": first_rank if first_rank is not None else "",
                        "evidence_chunk_id": evidence,
                        "top1_preview": top1_preview,
                        "matched_keyword_count": match_count,
                        "matched_keywords": "|".join(matched_keywords),
                    })

            config_rows = []
            n_q = max(1, len(questions))
            for m in methods:
                hit_rate = stats[m]["hits"] / n_q
                avg_first = float(np.mean(stats[m]["first_ranks"])) if stats[m]["first_ranks"] else np.nan
                avg_lat = float(np.mean(stats[m]["lat_ms"])) if stats[m]["lat_ms"] else np.nan

                row = {
                    "method": m,
                    "target_tokens": target_tokens,
                    "k": k,
                    "hit_rate": hit_rate,
                    "avg_first_hit_rank": avg_first,
                    "avg_query_latency_ms": avg_lat,
                    **meta[m],
                }
                config_rows.append(row)

            df_cfg = pd.DataFrame(config_rows).sort_values(["hit_rate", "avg_first_hit_rank"], ascending=[False, True])
            cfg_path = os.path.join(out_dir, f"summary_tokens{target_tokens}_k{k}.csv")
            df_cfg.to_csv(cfg_path, index=False)
            print(f"Wrote: {cfg_path}")

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(out_dir, "all_results_long.csv"), index=False)

    summary_long = (
        df_all.assign(first_hit_rank_num=pd.to_numeric(df_all["first_hit_rank"], errors="coerce"))
        .groupby(["method", "target_tokens", "k"], as_index=False)
        .agg(
            hit_rate=("hit_at_k", "mean"),
            avg_first_hit_rank=("first_hit_rank_num", "mean"),
        )
    )


    cfg_files = sorted(glob.glob(os.path.join(out_dir, "summary_tokens*_k*.csv")))
    cfg_df = pd.concat([pd.read_csv(p) for p in cfg_files], ignore_index=True) if cfg_files else pd.DataFrame()

    if not cfg_df.empty:
        summary_long = cfg_df

    summary_long = summary_long.sort_values(["target_tokens", "method", "k"])
    summary_long_path = os.path.join(out_dir, "summary_long.csv")
    summary_long.to_csv(summary_long_path, index=False)
    print(f"\nWrote: {summary_long_path}")

    wide = summary_long.pivot_table(
        index=["method", "target_tokens"],
        columns="k",
        values=["hit_rate", "avg_first_hit_rank"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}@{k}" for metric, k in wide.columns]
    wide = wide.reset_index()

    wide_csv = os.path.join(out_dir, "summary_wide.csv")
    wide.to_csv(wide_csv, index=False)

    wide_md = os.path.join(out_dir, "summary_wide.md")
    with open(wide_md, "w", encoding="utf-8") as f:
        f.write(wide.to_markdown(index=False))

    print("Wrote:")
    print(f"- {wide_csv}")
    print(f"- {wide_md}")
    print(f"- {os.path.join(out_dir, 'all_results_long.csv')}")


def parse_int_list(s: str) -> List[int]:
    parts = re.split(r"[ ,]+", s.strip())
    return [int(p) for p in parts if p.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_dir", required=True, help="Folder containing *.txt files.")
    parser.add_argument("--question_json", required=True, help="Path to question.json.")
    parser.add_argument("--out_dir", default="./out", help="Output folder for CSV tables.")
    parser.add_argument("--ks", default="3,5,7,9", help="Top-k values, e.g. '3,5,7,9'")
    parser.add_argument("--target_tokens_list", default="3000,5000,7000,10000", help="Chunk sizes (approx tokens).")
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL, help="SentenceTransformer model.")
    args = parser.parse_args()

    try:
        _ = sent_tokenize("Test. Another test.")
    except LookupError:
        nltk.download("punkt")

    ks = parse_int_list(args.ks)
    target_tokens_list = parse_int_list(args.target_tokens_list)

    run_grid_experiment(
        txt_dir=args.txt_dir,
        question_json=args.question_json,
        out_dir=args.out_dir,
        ks=ks,
        target_tokens_list=target_tokens_list,
        embed_model_name=args.embed_model,
    )


if __name__ == "__main__":
    main()
