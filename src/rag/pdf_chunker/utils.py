import re, math
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

def estimate_tokens(text: str) -> int:
    words = re.findall(r"\S+", text)
    return int(math.ceil(len(words) / 0.75)) if words else 0

def normalize_text(raw: str) -> str:
    t = raw.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)  
    t = re.sub(r"\n{3,}", "\n\n", t)    
    t = "\n".join(line.rstrip() for line in t.splitlines())
    return t.strip()

def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def detect_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith("#"):
        return True
    if re.match(r"^\d+(\.\d+)*\s+\S+", s):
        return True
    if s.isupper() and 4 <= len(s) <= 80:
        return True
    return False

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=False)
    return np.asarray(emb, dtype=np.float32)