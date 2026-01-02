"""
engine.__main__
===============

This file lets you run the entire framework with:

    python -m engine [options]

Options
-------
    --reindex        Rebuild the document FAISS index, then exit.
    -h, --help       Show help message.
"""
# engine/__main__.py refactor

import argparse, sys, threading
from engine.indexer import VectorIndexManager, DOCS_DIR, INDEX_PATH
from engine.agent import hegelian_qa
from engine.orchestrator import run_objective  # New import
from engine.identity import refresh_if_changed
from engine.loss_plotter import plot_learning_curve # Import the plotter

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m engine")
    p.add_argument("--reindex", action="store_true")
    p.add_argument("--auto", type=str, help="Run an autonomous objective and exit")
    p.add_argument("--limit", type=int, default=5, help="Max iterations for auto-mode")
    return p

def run_repl():
    index_mgr = VectorIndexManager(path=str(INDEX_PATH))

    # 1. Start the Background Watcher
    watcher_thread = threading.Thread(target=index_mgr.watch, daemon=True)
    watcher_thread.start()
    print("📡 Background Watcher active: listening to the Vault...")

    # 2. Initial Build if missing
    if not INDEX_PATH.exists():
        print("🛠️ Index not found. Building initial index...")
        # REMOVED / "vault" HERE:
        index_mgr.build(folder=DOCS_DIR) 

    index_mgr.load()

    while True:
        try:
            query = input("=^-.-^= ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        refresh_if_changed()

        # Handle explicit reindex command
        if query == "/reindex":
            print("🔄 Reindexing Vault...")
            # REMOVED / "vault" HERE:
            index_mgr.build(folder=DOCS_DIR) 
            continue

        if not query:
            continue

        answer = hegelian_qa(query)

if __name__ == "__main__":
    args = build_argparser().parse_args()

    if args.reindex:
        VectorIndexManager(path=str(INDEX_PATH)).build(folder=DOCS_DIR)
        sys.exit(0)

    if args.auto:
        print(f"🚀 ENTERING AUTONOMOUS MODE: {args.auto}")
        print(f"🛡️ SAFETY LIMIT: {args.limit} iterations.")
        try:
            run_objective(args.auto, max_iterations=args.limit)
            print("\n✅ Objective cycle complete. Generating analysis...")
        except Exception as e:
            print(f"🚨 EMERGENCY HALT: {e}")
        finally:
            # RUN THE PLOTTER HERE: 
            # This ensures even on failure, you see the curve leading up to the crash.
            try:
                plot_learning_curve()
            except Exception as plot_err:
                print(f"⚠️ Plotter failed: {plot_err}")
        sys.exit(0)

    run_repl()
