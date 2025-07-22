# engine.fabricator

from engine.llm_client import llm as _llm
from engine.indexer import VectorIndexManager, DOCS_DIR, INDEX_PATH

_index_singleton = None


def llm():
    return _llm  # already lazy inside llm_client


def index():
    global _index_singleton
    if _index_singleton is None:
        mgr = VectorIndexManager(str(INDEX_PATH))
        if not INDEX_PATH.exists():
            mgr.build(DOCS_DIR)
        mgr.load()
        _index_singleton = mgr.vect
    return _index_singleton
