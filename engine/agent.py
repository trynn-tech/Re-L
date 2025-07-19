import numpy as np, hashlib, sys, logging, time
from langchain.schema import Document
from engine import get_index
from engine.indexer import get_embedding
from engine.llm_client import invoke, stream, llm
from engine.memory import mem
from engine.identity import get as id_get
from engine.config import SIM_HIGH, SIM_LOW

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
dbg = logging.debug


def decay():  # thin wrappers to keep old calls working
    mem.decay()


def remember(k, v):
    mem.remember(k, v)


def tiered_retrieve(*args, **kw):  # still a stub
    return [], "novel", None


EMB = get_embedding()
index = None  # will be lazily set via get_index() later                         # module‑level


def faiss_guarded_retrieve(query, k=4, high=0.85, low=0.10):
    global index
    if index is None:
        index = get_index()  # FAISS object
    vec = EMB.embed_query(query)
    sims, docs = index.similarity_search_with_score(vec, k=k)
    best_sim = sims[0]

    if best_sim >= high:
        return docs[:1], "reuse"
    elif best_sim >= low:
        return docs[:k], "dialectic"
    else:
        return [], "novel"


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
    docs, mode, src = tiered_retrieve(spec, k=2, high=0.90, low=0.40)
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

    global index
    if index is None:
        index = get_index()  # FAISS object

    decay()
    dbg(f"query: {query!r}")

    SENTINEL = "<|END|>"

    # ---------------- Sentiment-aware preface ----------------
    recent = [v["v"] for k, v in mem.items() if k.startswith("user_turn_")][-3:]
    avg_val = np.mean([r["valence"] for r in recent]) if recent else 0
    if avg_val < -0.3:
        tone_preface = (
            "You sense the user is stressed; respond with extra empathy "
            "while remaining concise and clear.\n"
        )
    elif avg_val > 0.3:
        tone_preface = "Match the user's upbeat tone while remaining concise.\n"
    elif (
        avg_val == 0 and recent
    ):  # If no strong sentiment, but recent turns exist, try to match neutral
        tone_preface = "Maintain a neutral, professional, and concise tone.\n"
    elif not recent:  # No recent turns, default neutral
        tone_preface = "Maintain a neutral, professional, and concise tone.\n"
    else:
        tone_preface = ""

    # ---------------- Retrieve context ----------------
    docs = index.similarity_search(query, k=k)
    if not docs:
        return "I have no relevant context for that question."

    dbg(f"docs exist")
    thesis, antithesis = _split_thesis_antithesis(docs)

    def _trim(text, max_tokens=200):
        words = text.split()
        return " ".join(words[:max_tokens]) + (" …" if len(words) > max_tokens else "")

    thesis = _trim(thesis)
    antithesis = _trim(antithesis)

    # ---------------- Synthesis instruction (ENHANCED) ----------------
    synthesis_instruction = (
        "Analyse the thesis and antithesis as follows:\n"
        "• Summarise each core claim and its key evidence concisely, without repetition.\n"
        "• Identify and explain any logical gaps within each, or contradictions between them.\n"
        "• Pinpoint the most significant overlap or conflict that drives the dialectic.\n"
        "• **Craft a Synthesis that actively reconciles or transcends both positions.** "
        "   - **Prioritize the most impactful and actionable insights.**\n"
        "   - **Derive a novel conceptualization that emerges from the tension.**\n"
        "   - **Avoid merely combining or listing points; aim for a genuine conceptual leap.**\n"
        "• Finish with a single-sentence takeaway beginning “Therefore …” that encapsulates the new understanding."
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
    remember("last_answer", answer)
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
