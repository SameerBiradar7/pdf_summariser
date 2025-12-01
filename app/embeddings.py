# app/embeddings.py
from pathlib import Path
import os
import numpy as np
import pickle
import faiss
import torch

# LOCAL_MODEL_PATH must match where you downloaded all-MiniLM-L6-v2
LOCAL_MODEL_PATH = os.path.expanduser("~/pdf_summariser_models/all-MiniLM-L6-v2")
EMBED_DIM = 384

# Lazy model singleton
_MODEL = None

def get_model():
    """
    Lazy-load SentenceTransformer from local folder.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(LOCAL_MODEL_PATH)
        # attempt to use MPS on mac M1/M2
        try:
            if torch.backends.mps.is_available():
                _MODEL.to(torch.device("mps"))
        except Exception:
            pass
    return _MODEL


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (in-place copy safe)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-10)
    return x / norms


def embed_chunks(chunks: list, batch_size: int = 64, show_progress: bool = False) -> np.ndarray:
    """
    Embed list of text chunks into a (N, D) float32 numpy array.
    Returns normalized vectors (float32) suitable for IndexFlatIP (cosine).
    """
    if not chunks:
        return np.zeros((0, EMBED_DIM), dtype="float32")

    model = get_model()
    embs_list = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        b_emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=show_progress)
        embs_list.append(b_emb)
    embs = np.vstack(embs_list).astype("float32")
    embs = _normalize_rows(embs)
    return embs


def build_faiss_index(emb_matrix: np.ndarray, use_ivf: bool = False, nlist: int = 128):
    """
    Build FAISS index. Default: IndexFlatIP on normalized vectors (fast cosine via inner product).
    If use_ivf=True, build an IVF index (faster for large corpora) -- remember to call index.train().
    Returns the FAISS index instance.
    """
    d = int(emb_matrix.shape[1])
    if use_ivf:
        quant = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_L2)
        # training required: index.train(emb_matrix)
        index.train(emb_matrix)
        index.add(emb_matrix)
    else:
        index = faiss.IndexFlatIP(d)  # inner product on normalized vectors ~ cosine
        index.add(emb_matrix)
    return index


def persist_index(index: faiss.IndexFlatIP, chunks: list, output_prefix: str, embeddings: np.ndarray = None):
    """
    Save faiss index, chunks list and (optionally) embeddings to outputs/<prefix>_*
    Returns paths (faiss_path, chunks_path, embeddings_path_or_None)
    """
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = out_dir / f"{output_prefix}_faiss.idx"
    chunks_path = out_dir / f"{output_prefix}_chunks.pkl"
    emb_path = out_dir / f"{output_prefix}_embeddings.npy"

    faiss.write_index(index, str(faiss_path))

    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    if embeddings is not None:
        np.save(str(emb_path), embeddings)
        return str(faiss_path), str(chunks_path), str(emb_path)

    return str(faiss_path), str(chunks_path), None


def load_index_and_chunks(prefix: str):
    """
    Load FAISS index and chunks given a prefix.
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


def load_embeddings_if_exists(prefix: str):
    """
    Return embeddings np.ndarray if outputs/<prefix>_embeddings.npy exists, else None.
    """
    emb_path = Path("outputs") / f"{prefix}_embeddings.npy"
    if emb_path.exists():
        return np.load(str(emb_path), mmap_mode=None)
    return None