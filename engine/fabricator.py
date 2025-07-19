"""
Internal helpers that *actually* create the heavy singletons.
Nothing here is imported at top‑level; only via getters in __init__.py.
"""

from engine.llm_client import llm as _llm
from engine.memory import DecayMemory
from engine.indexer import VectorIndexManager, DOCS_DIR, INDEX_PATH

# Singletons kept in module‑level cache
_mem = None
_index = None


def llm():
    return _llm  # already light; safe to reuse


def mem():
    global _mem
    if _mem is None:
        _mem = DecayMemory()
    return _mem


def index():
    """
    Return FAISS index object.
    Loads (or builds) on first call, then reuses.
    """
    global _index
    if _index is None:
        mgr = VectorIndexManager(path=str(INDEX_PATH))
        if not INDEX_PATH.exists():
            mgr.build(folder=DOCS_DIR)
        mgr.load()
        _index = mgr.vect
    return _index
