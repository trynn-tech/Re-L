"""
engine – lightweight RAG + dialectic + code‑proof framework

* Heavy singletons (LLM, FAISS index, DecayMemory) live in
  `engine.fabricator` and are created lazily.
* Importing `engine` is therefore cheap (<20 ms).

Public helpers
──────────────
    from engine import get_llm, get_memory, get_index
"""

from importlib import import_module
from types import ModuleType

__version__ = "0.4.1"
__all__ = ["get_llm", "get_memory", "get_index", "__version__"]

# ------------------------------------------------------------------
# Lazy accessors
# ------------------------------------------------------------------


def _lazy(attr: str):
    """
    Import `engine.fabricator` on first access and return the requested
    singleton (`llm`, `mem`, or `index`).
    """
    module: ModuleType = import_module("engine.fabricator")
    return getattr(module, attr)()


def get_llm():
    """Return the global llama‑cpp client."""
    return _lazy("llm")


def get_memory():
    """Return the global DecayMemory instance."""
    return _lazy("mem")


def get_index():
    """Return (and auto‑load) the FAISS index."""
    return _lazy("index")
