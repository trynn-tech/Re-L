# forge/indexer.py
import pathlib, pickle, json
from typing import Union
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from engine.config import EMBED_MODEL, DOCS_DIR, INDEX_PATH

EMB = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)


def get_embedding():
    return EMB


class VectorIndexManager:
    def __init__(self, path="faiss_index.pkl", model="all-MiniLM-L6-v2"):
        self.path = pathlib.Path(path)
        self.emb = SentenceTransformerEmbeddings(model_name=model)
        self.vect = None

    def build(
        self,
        folder: Union[str, pathlib.Path] = DOCS_DIR,  # type‑flexible default
        chunk: int = 400,
        overlap: int = 60,
    ) -> None:
        """Create a FAISS index from every file in *folder*."""
        folder = pathlib.Path(folder)  # normalise early

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk, chunk_overlap=overlap
        )

        docs = []
        for p in folder.glob("*"):
            try:
                loader = UnstructuredFileLoader(str(p))
                docs.extend(loader.load_and_split(splitter))
            except Exception as e:
                print(f"⚠️  Skipping {p.name}: {e}")

        self.vect = FAISS.from_documents(docs, self.emb)
        with open(self.path, "wb") as f:
            pickle.dump(self.vect, f)
        print(f"Indexed {len(docs)} chunks → {self.path}")

    def load(self):
        if not self.vect:
            if not self.path.exists():
                self.build(folder=DOCS_DIR)  # auto‑build if missing
            with open(self.path, "rb") as f:
                self.vect = pickle.load(f)

    def search(self, query, k=4):
        self.load()
        return self.vect.similarity_search(query, k=k)
