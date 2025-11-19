//--------- engine/agent.py ----------//

import numpy as np, hashlib, sys, logging, time
import requests, html
from langchain.schema import Document
from engine.indexer import get_embedding
from engine.llm_client import invoke, stream, llm
from engine.memory import recall, store, reinforce  # re‑exported in memory/__init__
from engine.fabricator import index as get_index  # FAISS singleton
from engine.config import SIM_HIGH, SIM_LOW

# TODO
from engine.identity import get as id_get


logging.basicConfig(level=logging.DEBUG, format="%(message)s")
dbg = logging.debug


EMB = get_embedding()
index = None  # cached FAISS fallback


def wiki_snippet(q):
    params = {"action": "opensearch", "search": q, "limit": 1, "format": "json"}
    data = requests.get(
        "https://en.wikipedia.org/w/api.php", params=params, timeout=5
    ).json()
    return html.unescape(data[2][0]) if data and data[2] else ""


def guarded_retrieve(
    query: str, k: int = 4, high: float = SIM_HIGH, low: float = SIM_LOW
):
    """
    Wrapper that first calls api.recall() (STM → Heur → LTM).
    Falls back to local FAISS only when recall() returns nothing
    so your existing vector index is still used as a safety‑net.
    """
    # ── Tiered memory first ───────────────────────────────
    hits = recall(query, k=k)  # list[str] (may be empty)
    if hits:
        docs = [Document(page_content=t) for t in hits]
        mode = "reuse" if len(docs) == 1 else "dialectic"
        return docs, mode  # compatible with old code

    # ── Old FAISS logic (unchanged) ───────────────────────
    global index
    if index is None:
        index = get_index()

    pairs = index.similarity_search_with_score(query, k=k)
    pairs = [
        (d, s)
        for d, s in pairs
        if not d.page_content.startswith(
            "The thesis and antithesis both"
        )  # crude filter
    ]
    if not pairs:
        return [], "novel"

    # sort is already highest‑first in LC >=0.2, but do it explicitly
    pairs.sort(key=lambda x: x[1], reverse=True)

    docs = [d for d, _ in pairs]
    sims = [s for _, s in pairs]
    best_sim = sims[0]

    # OPTIONAL dynamic_k
    if dynamic := True:
        threshold = 0.9 * best_sim
        docs = [d for d, s in pairs if s >= threshold]
        sims = [s for s in sims if s >= threshold]

    mode = "reuse" if best_sim >= high else "dialectic" if best_sim >= low else "novel"

    docs = [
        d
        for d in docs
        if not d.page_content.startswith("The thesis and antithesis both")
    ]

    return docs, mode


# ---------------- lint / proof stubs -----------------
def passes_lint(code: str) -> bool:
    # TODO integrate ruff/flake8; stub = always True
    return True


def passes_quicktest(code: str, spec: str) -> bool:
    return True


def prove_invariant(code: str, spec: str) -> bool:
    # integrate z3 / hypothesis later
    return True


def code_proof_cycle(spec: str) -> str:
    # 1. Retrieval
    docs, mode = guarded_retrieve(spec, k=2, high=0.90, low=0.40)
    if mode.startswith("reuse"):
        return docs[0].page_content

    # 2. Draft N candidates
    prompt = f"Write Python that satisfies:\n{spec}"
    drafts = [invoke(prompt, temperature=0.9) for _ in range(4)]

    # 3. Static and unit smoke
    good = [d for d in drafts if passes_lint(d) and passes_quicktest(d, spec)]

    # 4. Property proof (optional but nice)
    for code in good:
        if prove_invariant(code, spec):
            # 5. Store proof vector
            vect_proof.add_texts([code], metadatas=[{"spec": spec, "id": id_hash}])
            return code

    return "No candidate satisfied proof."


# --- Hegelian dialectic helpers ---------------------------------------------
def _semantic_similarity(a: str, b: str) -> float:
    """
    Calculate semantic similarity using embeddings.
    Returns cosine similarity (1.0 = identical meaning, -1.0 = opposite meaning).
    """
    dbg("Calculating semantic similarity using embeddings")
    # Get embeddings for the two texts
    embeddings_a = EMB.embed_query(a)
    embeddings_b = EMB.embed_query(b)

    # Calculate cosine similarity
    # Formula: dot_product(A, B) / (norm(A) * norm(B))
    dot_product = np.dot(embeddings_a, embeddings_b)
    norm_a = np.linalg.norm(embeddings_a)
    norm_b = np.linalg.norm(embeddings_b)

    if norm_a == 0 or norm_b == 0:  # Avoid division by zero
        return 0.0  # Or some other defined value for empty/zero vectors

    return dot_product / (norm_a * norm_b)


def _split_thesis_antithesis(docs: list[Document]) -> tuple[str, str]:
    """
    Decide which two chunks are most opposed (lowest semantic similarity) and
    treat first as thesis, second as antithesis.  Fallback = first vs last.
    """
    if len(docs) < 2:
        dbg("Less than 2 docs, returning first as thesis, empty antithesis.")
        # If only one document, it can be the thesis, no antithesis
        return docs[0].page_content, "" if docs else ""  # Handle empty docs list too

    worst_sim = (
        1.1  # Initialize with a value higher than max possible cosine similarity (1.0)
    )
    thesis_doc_content = docs[0].page_content
    antithesis_doc_content = docs[-1].page_content

    # Iterate through all unique pairs to find the most opposed
    for i in range(len(docs)):
        for j in range(
            i + 1, len(docs)
        ):  # Start j from i+1 to avoid self-comparison and redundant pairs
            d1 = docs[i]
            d2 = docs[j]

            sim = _semantic_similarity(d1.page_content, d2.page_content)
            # dbg(f"Compared doc {i} and {j}. Similarity: {sim:.4f}") # Re-enable for debugging sim values

            if (
                sim < worst_sim
            ):  # We are looking for the *lowest* similarity score (most opposed)
                worst_sim = sim
                thesis_doc_content = d1.page_content
                antithesis_doc_content = d2.page_content
                # dbg(f"New worst sim found: {worst_sim:.4f} for docs {i} and {j}") # Re-enable for debugging selection

    dbg(f"Found most opposed pair with similarity: {worst_sim:.4f}")
    return thesis_doc_content, antithesis_doc_content


# ─────────────────── Hegelian QA (refactored) ────────────────────
def hegelian_qa(query: str, k: int = 4) -> str:

    # TODO: turn this into a proper router rather than hijacking this function
    # --- 0. Special command:  /code <spec> ------------------------------
    if query.startswith("/code"):
        spec = query[len("/code") :].strip() or "Write a hello‑world fn."
        return code_proof_cycle(spec)

    dbg(f"query: {query!r}")
    tone_preface = "Maintain a neutral, professional, and concise tone.\n"

    # ---------------- Retrieve context ----------------
    docs, mode = guarded_retrieve(query, k=k)

    if mode == "novel":
        snippet = wiki_snippet(query)
        if snippet:
            dbg(f"Wiki snippet: {snippet}")
            docs.append(Document(page_content=snippet))

    if not docs:
        return "I have no relevant context for that question."

    dbg(f"docs exist")
    thesis, antithesis = _split_thesis_antithesis(docs)

    # If retrieval produced only one fragment, fabricate a counter‑view
    if antithesis.strip().upper() in {"", "N/A"}:
        antithesis = invoke(
            "Write a concise counter‑argument (≤120 words) to:\n"
            f"```{thesis[:800]}```",
            temperature=0.7,
        ).strip()

    used_keys = [
        d.metadata["id"] for d in docs if hasattr(d, "metadata") and "id" in d.metadata
    ]
    for k in used_keys:
        reinforce(k, delta=0.5)

    def _trim(text, max_tokens=200):
        words = text.split()
        return " ".join(words[:max_tokens]) + (" …" if len(words) > max_tokens else "")

    thesis = _trim(thesis)
    antithesis = _trim(antithesis)

    # ---------------- Synthesis instruction (ENHANCED) ----------------
    synthesis_instruction = (
        "You are a code‑side assistant.\n"
        "• In ≤150 words summarise the thesis.\n"
        "• In ≤150 words summarise the antithesis.\n"
        "• Craft a synthesis **focused on ONE concrete step the user can take "
        "today** (≤120 words).\n"
        "End with a single‑sentence takeaway beginning “Therefore …”."
    )

    # Guiding instruction for Hegel's dialectic nuance
    hegel_nuance_instruction = (
        "Be aware that the common 'thesis-antithesis-synthesis' model can be misleading. "
        "Hegel's logic distinguishes a one-sided position from a recognition of its inadequacy "
        "revealed in internal contradictions, leading to a higher reconciliation, not simply "
        "contraries. Focus on deriving new triads and structural roles rather than mere opposition."
    )

    prompt = (
        f"### System\n{tone_preface}"
        "You are a Hegelian analyst. "
        f"{hegel_nuance_instruction}\n\n"
        f"### Query\n{query}\n\n"
        f"### Thesis\n{thesis}\n\n"
        f"### Antithesis\n{antithesis or 'N/A'}\n\n"
        f"### Task\n{synthesis_instruction}\n"
        f"### Begin Synthesis\n"
    )

    dbg(f"Prompt: {prompt.strip()}")
    dbg(f"Token Input Total: {llm.get_num_tokens(prompt)}")
    answer = _invoke_collect(prompt, temperature=0.45)
    store(answer, valence=+0.1)  # logs the synthesis into STM/LTM path
    return answer


# ──────────── Helper: robust streaming + idle valve ────────────
def _invoke_collect(
    prompt: str,
    temperature: float = 0.45,
    stop: list[str] | None = None,
    idle_ms: int = 300000,  # break if no token for 15 s
) -> str:

    prompt = prompt.rstrip()

    buf: list[str] = []
    last_tok_time = time.time()

    try:
        for tok in stream(prompt, temperature=temperature, stop=stop or []):
            now = time.time()
            if (now - last_tok_time) * 1000 > idle_ms:
                dbg(f"⏳ Idle for {idle_ms} ms — aborting stream.")
                break

            buf.append(tok)
            print(tok, end="", flush=True)
            last_tok_time = now  # reset timer
    except KeyboardInterrupt:
        dbg("⏹️  Interrupted by user (Ctrl‑C)")
        return "".join(buf) + " …[interrupted]"
    except Exception as e:
        dbg(f"⚠️  LLM stream error: {e}")
        raise

    return "".join(buf)

//--------- engine/config.py ----------//
# forge/config.py
from pathlib import Path
import os
from dotenv import load_dotenv

# ──────────────── secrets ────────────────────────────────────────────
# Load .env once and expose REDIS_URL to any module that imports config.
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=False)

_pw = os.getenv("REDIS_PASSWORD")
if not _pw:
    raise RuntimeError(
        "REDIS_PASSWORD missing.  Create a .env file or export the "
        "variable before running."
    )

REDIS_URL = f"redis://:{_pw}@localhost:6379/0"

# --- Paths -------------------------------------------------------
DOCS_DIR = ROOT / "docs"
IDENTITY_PATH = ROOT / "configs" / "identity.yaml"
INDEX_PATH = ROOT / "configs" / "faiss_index.pkl"
PROOF_PATH = ROOT / "configs" / "proof_index.pkl"
MEM_PATH = ROOT / "configs" / "decay_mem.json"

# --- Embedding / model names -------------------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "~/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"


# --- Retrieval thresholds ---------------------------------------
SIM_HIGH = 0.85
SIM_LOW = 0.10


//--------- engine/fabricator.py ---------//
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


//--------- engine/gauges.py ---------//
# forge/gauges.py
from typing import Literal
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from engine.indexer import get_embedding

EMB = get_embedding()
# ───────────── Sentiment & Identity Gauges ──────────────────────
POS_SEEDS = ["great", "excellent", "love", "happy", "wonderful"]
NEG_SEEDS = ["hate", "terrible", "bad", "angry", "sad"]


def _embed(text: str) -> np.ndarray:
    return np.asarray(EMB.embed_query(text), dtype=np.float32)


# Pre-compute seed vectors once
_pos_vecs = np.stack([_embed(w) for w in POS_SEEDS])
_neg_vecs = np.stack([_embed(w) for w in NEG_SEEDS])
X_seed = np.vstack([_pos_vecs, _neg_vecs])
y_seed = np.array([1] * len(_pos_vecs) + [0] * len(_neg_vecs))

# Simple logistic-reg pipeline
_sent_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
_sent_clf.fit(X_seed, y_seed)


def analyse_turn(text: str, speaker: Literal["user", "assistant"]) -> dict:
    """Return {'speaker':…, 'tone':…, 'valence': float}."""

    vec = _embed(text)
    prob_pos = _sent_clf.predict_proba(vec.reshape(1, -1))[0, 1]
    valence = (prob_pos * 2) - 1  # map 0..1 ➜ -1..+1
    tone = "positive" if valence > 0.3 else "negative" if valence < -0.3 else "neutral"

    return {"speaker": speaker, "tone": tone, "valence": round(valence, 3)}


//--------- engine/identity.py ----------//
import yaml, pathlib, time, functools
from engine.config import IDENTITY_PATH as _ID_PATH

_last_mtime = 0


@functools.lru_cache(1)
def _load() -> dict:
    return yaml.safe_load(_ID_PATH.read_text()) if _ID_PATH.exists() else {}


def get(key, default=None):
    return _load().get(key, default)


def refresh_if_changed():
    global _last_mtime
    try:
        m = _ID_PATH.stat().st_mtime
        if m != _last_mtime:
            _load.cache_clear()
            _last_mtime = m
    except FileNotFoundError:
        pass


//--------- engine/indexer.py ----------//
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

    def __init__(self, path=INDEX_PATH, model="all-MiniLM-L6-v2"):
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


//--------- engine/llm_client.py ----------//
# forge/llm_client.py
from langchain_community.llms import LlamaCpp
from engine.config import LLM_MODEL
from pathlib import Path

MODEL_PATH = Path(LLM_MODEL).expanduser()

llm = LlamaCpp(
    model_path=str(MODEL_PATH),
    n_ctx=4096,
    max_tokens=512,
    n_threads=12,
    n_batch=512,  # decodes 512 tokens per KV‑cache update
    temperature=0.65,
    top_k=40,
    top_p=0.9,
)


def invoke(prompt: str, **kw) -> str:
    kw.setdefault("max_tokens", 512)
    return llm(prompt, **kw).strip()


def stream(prompt: str, **kw):
    print("kw is")
    print(kw)
    kw.setdefault("max_tokens", 512)  # ← ensure non‑zero generation
    return llm.stream(prompt, **kw)


//--------- .env.example ----------//
REDIS_PASSWORD=myStrongPass

//--------- .gitignore ----------//
.gitignore
