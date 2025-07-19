"""
engine.__main__
===============

This file lets you run the entire framework with:

    python -m engine [options]

Options
-------
    --reindex        Rebuild the document FAISS index, then exit.
    --mem            Dump the current DecayMemory JSON, then exit.
    -h, --help       Show help message.
"""

import argparse, sys
from engine import get_index, get_memory, get_llm  # lazy singletons
from engine.indexer import VectorIndexManager, DOCS_DIR, INDEX_PATH
from engine.agent import hegelian_qa
from engine.gauges import analyse_turn
from engine.identity import refresh_if_changed


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m engine", description="Run RAG + dialectic REPL"
    )
    p.add_argument(
        "--reindex", action="store_true", help="rebuild document index and exit"
    )
    p.add_argument("--mem", action="store_true", help="print decay memory and exit")
    return p


def run_repl():
    index_mgr = VectorIndexManager(path=str(INDEX_PATH))
    if not INDEX_PATH.exists():
        index_mgr.build(folder=DOCS_DIR)
    index_mgr.load()  # sets global FAISS via engine.__init__
    mem = get_memory()

    while True:
        try:
            query = input("🜂 ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        refresh_if_changed()  # hot‑reload identity.yaml
        meta = analyse_turn(query, "user")
        mem.remember(f"user_turn_{len(mem)}", meta)

        if query in ("/mem", "/memory"):
            print(mem)
            continue
        if query == "/reindex":
            index_mgr.build(folder=DOCS_DIR)
            continue
        if not query:
            continue

        answer = hegelian_qa(query)
        print("\n", answer, "\n")


if __name__ == "__main__":
    args = build_argparser().parse_args()

    if args.reindex:
        VectorIndexManager(path=str(INDEX_PATH)).build(folder=DOCS_DIR)
        sys.exit(0)

    if args.mem:
        print(get_memory())
        sys.exit(0)

    run_repl()
