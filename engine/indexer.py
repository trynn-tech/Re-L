import pathlib
import pickle
import time
from typing import Union
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from engine.config import INDEX_PATH, EMBED_MODEL, DOCS_DIR
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS

def get_embedding():
    """Return the standard embedding model used for queries."""
    # This uses the config variable EMBED_MODEL we imported earlier
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

class VectorIndexManager(FileSystemEventHandler):
    def __init__(self, path=INDEX_PATH, model=EMBED_MODEL):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.emb = SentenceTransformerEmbeddings(model_name=model)
        self.vect = None
        # This is where the watcher looks
        self.vault_dir = DOCS_DIR 
        self.vault_dir.mkdir(parents=True, exist_ok=True)

    def load(self):
        if self.path.exists():
            with open(self.path, "rb") as f:
                self.vect = pickle.load(f)
            print(f"🧠 Index loaded from {self.path.name}")
        else:
            print("❌ No index found to load.")

    def build(self, folder=None, chunk=500, overlap=50) -> None:
        target_dir = folder if folder else self.vault_dir
        print(f"🔨 Forging index from Vault: {target_dir}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
        docs = []
        for ext in ["*.pdf", "*.txt", "*.md"]:
            for p in target_dir.glob(ext):
                try:
                    loader = UnstructuredFileLoader(str(p))
                    docs.extend(loader.load_and_split(splitter))
                except Exception as e:
                    print(f"⚠️  Contamination in {p.name}: {e}")

        if not docs:
            print(f"📭 Vault is empty at {target_dir}. Standing by...")
            return

        self.vect = FAISS.from_documents(docs, self.emb)
        with open(self.path, "wb") as f:
            pickle.dump(self.vect, f)
        print(f"✅ Cognitive Layer Updated: {len(docs)} signal chunks saved.")

    def on_created(self, event):
        if not event.is_directory:
            print(f"📡 New signal detected: {pathlib.Path(event.src_path).name}")
            time.sleep(1)
            self.build()

    def watch(self):
        observer = Observer()
        observer.schedule(self, str(self.vault_dir), recursive=False)
        print(f"👁️  Re-L is now observing: {self.vault_dir}")
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()



