from dataclasses import dataclass

@dataclass
class Chunk:
    method: str
    doc_id: str
    chunk_id: str
    text: str
