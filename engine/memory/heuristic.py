from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import pickle, pathlib, os

_EMB = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
IDX = pathlib.Path("configs/heuristic.faiss")
IDX.parent.mkdir(parents=True, exist_ok=True)

_vect: FAISS | None = None  # in‑process cache


# -----------------------------------------------------------------------
def _load() -> FAISS | None:
    """Return FAISS index if file exists, else None."""
    if IDX.exists():
        return pickle.loads(IDX.read_bytes())
    return None


# -----------------------------------------------------------------------
def get_index() -> FAISS | None:
    global _vect
    if _vect is None:
        _vect = _load()
    return _vect


# -----------------------------------------------------------------------
def add(text: str, meta: dict) -> None:
    """
    Add a fragment; create index on first insert.
    Persist to disk every time (cheap—few KB).
    """
    global _vect
    if _vect is None:
        _vect = FAISS.from_texts([text], _EMB, metadatas=[meta])
    else:
        _vect.add_texts([text], metadatas=[meta])

    with IDX.open("wb") as f:
        pickle.dump(_vect, f)


# -----------------------------------------------------------------------
def search(q: str, k: int = 4):
    vect = get_index()
    if vect is None:
        return []  # empty cache ⇒ miss
    return vect.similarity_search(q, k)
