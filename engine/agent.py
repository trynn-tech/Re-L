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
