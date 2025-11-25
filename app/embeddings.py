# app/embeddings.py
from pathlib import Path
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, good for retrieval
EMBED_DIM = 384                   # embedding dim for this model

model = SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks: list) -> np.ndarray:
    """
    Return a (N, D) float32 numpy array of embeddings for list of chunk strings.
    """
    embs = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    if embs.dtype != np.float32:
        embs = embs.astype("float32")
    return embs


def build_faiss_index(emb_matrix: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build and return a FAISS IndexFlatL2 for given embedding matrix.
    (IndexFlatL2 is simple and fine for small-medium datasets.)
    """
    d = emb_matrix.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(emb_matrix)
    return index


def persist_index(index: faiss.IndexFlatL2, chunks: list, output_prefix: str):
    """
    Save faiss index and chunks to outputs/ using a prefix string.
    Writes:
      - outputs/<output_prefix>_faiss.idx  (faiss.write_index)
      - outputs/<output_prefix>_chunks.pkl (pickle list of chunks)
    """
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = out_dir / f"{output_prefix}_faiss.idx"
    chunks_path = out_dir / f"{output_prefix}_chunks.pkl"

    faiss.write_index(index, str(faiss_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    return str(faiss_path), str(chunks_path)


def load_index_and_chunks(prefix: str):
    """
    Load index and chunks from outputs/<prefix>_*
    """
    out_dir = Path("outputs")
    faiss_path = out_dir / f"{prefix}_faiss.idx"
    chunks_path = out_dir / f"{prefix}_chunks.pkl"

    if not faiss_path.exists() or not chunks_path.exists():
        raise FileNotFoundError("Index or chunks file not found for prefix: " + prefix)

    index = faiss.read_index(str(faiss_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks