# app/retrieval.py
from typing import List, Tuple
import numpy as np
from pathlib import Path
import pickle

from sentence_transformers import SentenceTransformer
from app.embeddings import load_index_and_chunks

# reuse the same sentence-transformer model used for building the index
_EMB_MODEL_NAME = "all-MiniLM-L6-v2"
_emb_model = SentenceTransformer(_EMB_MODEL_NAME)

def embed_query(text: str) -> np.ndarray:
    vec = _emb_model.encode([text], convert_to_numpy=True)
    if vec.dtype != np.float32:
        vec = vec.astype("float32")
    return vec[0]

def retrieve_top_k(prefix: str, query: str, top_k: int = 5) -> Tuple[List[str], List[int]]:
    """
    Load the index and chunks for `prefix`, compute query embedding,
    run FAISS search and return (selected_chunks, indices).
    """
    index, chunks = load_index_and_chunks(prefix)
    qvec = embed_query(query).reshape(1, -1).astype("float32")
    D, I = index.search(qvec, top_k)
    indices = [int(i) for i in I[0] if i != -1]
    selected = [chunks[i] for i in indices]
    return selected, indices

def assemble_context(chunks: List[str], max_chars: int = 4000) -> str:
    """
    Concatenate chunks to a single context string, trimming if necessary
    to approx max_chars (simple char-based trim).
    """
    out = []
    cur_len = 0
    for c in chunks:
        if cur_len + len(c) + 3 > max_chars:
            break
        out.append(c)
        cur_len += len(c) + 3
    return "\n\n---\n\n".join(out)