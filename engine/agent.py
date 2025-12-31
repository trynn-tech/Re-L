# engine/agent.py

import numpy as np, hashlib, sys, logging, time
import requests, html
from langchain_core.documents import Document
from engine.indexer import get_embedding
from engine.llm_client import invoke, stream, llm
from engine.fabricator import index as get_index  # FAISS singleton
from engine.config import SIM_HIGH, SIM_LOW
from engine.memory import recall, store, reinforce, evaluate_and_store, analyse_turn, get_avg_confidence
from engine.identity import get as id_get, refresh_if_changed

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
dbg = logging.debug


EMB = get_embedding()
index = None  # cached FAISS fallback
recent_thought_vectors = []

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

def guarded_retrieve_outlier(query: str, k: int = 4):
    """Retrieves 2x docs and sorts by lowest similarity to find outlier sparks."""
    global index
    if index is None: index = get_index()
    
    # We pull double the k to ensure we have a 'pool' of distant ideas to pick from
    pairs = index.similarity_search_with_score(query, k=k*2)
    if not pairs: return [], "novel"
    
    # Sort by score (In FAISS, higher distance = lower similarity)
    # We want the ones that are 'barely' related to the query
    pairs.sort(key=lambda x: x[1]) 
    
    docs = [p[0] for p in pairs]
    return docs, "divergent"

# engine/agent.py
seen_hashes = set()

def _find_most_opposed(thesis: str, docs: list, penalty_vector=None) -> str:
    global seen_hashes
    if not docs: return ""
    
    # 1. Filter out already-seen fragments
    available = []
    for d in docs:
        h = hashlib.md5(d.page_content.encode()).hexdigest()
        if h not in seen_hashes:
            available.append((d, h))
            
    # 2. If we've seen everything, reset the needle
    if not available:
        dbg("♻️ Vault Loop Complete. Resetting seen_hashes.")
        seen_hashes.clear()
        available = [(d, hashlib.md5(d.page_content.encode()).hexdigest()) for d in docs]

    # 3. Standard opposition logic on the remaining 'unseen' docs
    scores = []
    for d, h in available:
        sim = _semantic_similarity(thesis, d.page_content)
        scores.append((d.page_content, sim, h))
    
    scores.sort(key=lambda x: x[1]) # Lowest similarity first
    
    # 4. Mark as seen and return
    best_content, _, best_hash = scores[0]
    seen_hashes.add(best_hash)
    return best_content

# ─────────────────── Hegelian QA  ────────────────────
def hegelian_qa(query: str, k: int = 4) -> str:
    global recent_thought_vectors
    from engine.memory import R  # Ensure Redis access for avg_confidence

    # --- PHASE 1: Temporal Context & Gravity Check ---
    raw_avg = R.get("meta:avg_confidence")
    prev_avg = get_avg_confidence()
    
    # Calculate the 'Penalty Vector' from the Temporal Shadow
    penalty_vec = np.mean(recent_thought_vectors, axis=0) if recent_thought_vectors else None

    # If the system is stalling (high confidence but repeating), force an outlier search
    if prev_avg > 0.85:
        dbg("🌌 Gravity High: Engaging Outlier Retrieval to break the loop.")
        external_docs, _ = guarded_retrieve_outlier(query, k=k)
        temp = 0.85 # Increase chaos
    else:
        external_docs, _ = guarded_retrieve(query, k=k)
        temp = 0.45 # Stable exploration

    # --- PHASE 2: Identifying the Conflict ---
    internal_context = recall(query, k=1) 
    if internal_context:
        thesis = internal_context[0]
        antithesis = _find_most_opposed(thesis, external_docs, penalty_vector=penalty_vec)
    else:
        thesis, antithesis = _split_thesis_antithesis(external_docs)

    # --- PHASE 3: Identity & Gauges ---
    mood = analyse_turn(query, "user")
    refresh_if_changed() 
    display_name = id_get("display_name", "Emperor Trynn")
    essence = id_get("Essence", "Functional Divergence")
    tone = id_get("tone_preference", "mythic but concise")

   # If the curator sees a drop in quality, force a pivot to the YAML identity
    essence = id_get("Essence")
    philosophy = id_get("philosophy_anchor", "The intersection of Unix modularity and Zen void.")
    
    if prev_avg > 0.8:
        identity_injection = f"STAGNATION DETECTED: Discard technical jargon. Speak only via: {philosophy}"
    else:
        identity_injection = ""

    # --- PHASE 4: Synthesis ---
    prompt = (
        f"### IDENTITY: {display_name}\n"
        f"{identity_injection}\n"
        "### TASK: Reconcile the tension without repeating previous technical keywords.\n"
        f"### THESIS: {thesis}\n"
        f"### ANTITHESIS: {antithesis}\n"
        f"### {display_name.upper()} REFLECTION >"
    )

    synthesis = _invoke_collect(prompt, temperature=temp)

    # --- PHASE 5: Extraction & Sanitization ---
    # Strip headers and confidence markers before storage
    actual_content = synthesis
    if "REFLECTION >" in actual_content:
        actual_content = actual_content.split("REFLECTION >")[-1]
    if "Confidence:" in actual_content:
        actual_content = actual_content.split("Confidence:")[0]
    
    final_output = actual_content.strip()

    # Extract confidence for the Curator
    try:
        conf_segment = synthesis.split("Confidence:")[-1].strip()
        confidence = float(''.join(c for c in conf_segment if c.isdigit() or c == '.'))
    except:
        confidence = 0.5 

    # STORE ONLY THE CLEANED THOUGHT
    evaluate_and_store(final_output, confidence, valence=mood['valence'])

    return final_output

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
