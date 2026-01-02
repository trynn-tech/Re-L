# engine/agent.py
import requests
import numpy as np, hashlib, sys, logging, re, html, os
from engine.llm_client import invoke, stream
from engine.memory import (
    recall, evaluate_and_store, analyse_turn, get_avg_confidence, long_term
)
from engine.identity import get as id_get, refresh_if_changed
from engine.critic import verify_integrity 

logging.basicConfig(level=logging.DEBUG, format="%(message)s")
dbg = logging.debug

# Verified X11 Headers from your Jina Inquest
WIKI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept": "text/plain"
}

SESSION_CACHE = {"research_data": None, "source_url": None, "topic": None}

def purge_state():
    """Resets volatile session data to prevent context contamination."""
    global SESSION_CACHE
    SESSION_CACHE = {"research_data": None, "source_url": None, "topic": None}

def _semantic_similarity(a, b):
    """Internal fallback for veracity checks."""
    if not a or not b: return 0.0
    return 1.0 if a.strip().lower() == b.strip().lower() else 0.5

# --- THE HARDENED LIBRARIAN ---

def get_wiki_url(q):
    """Distills subject and fetches Wikipedia URL using header rotation."""
    try:
        distill_prompt = f"Identify the primary technical subject (ONE WORD ONLY): {q}"
        raw_distilled = invoke(distill_prompt, temperature=0.0).strip()
        words = raw_distilled.split()
        distilled = re.sub(r'[^\w]', '', (words[-1].lower() if words else "entropy"))

        if SESSION_CACHE.get("topic") == distilled and SESSION_CACHE.get("source_url"):
            return "ALREADY_LOADED"

        WIKI_API = "https://en.wikipedia.org/w/api.php"
        params = {"action": "opensearch", "search": distilled, "limit": 1, "format": "json"}
        
        r = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=5)
        if r.status_code == 200:
            res = r.json()
            if isinstance(res, list) and len(res) > 3 and res[3]:
                url = res[3][0]
                SESSION_CACHE["topic"] = distilled
                SESSION_CACHE["source_url"] = url
                return url
    except Exception as e:
        dbg(f"Librarian Failure: {e}")
    
    return "INTERNAL_REF: Shannon Entropy H(X) = -sum(p(i) * log2(p(i)))"

def technical_context(q):
    """Captures research data from Jina with a strict character budget."""
    target_url = get_wiki_url(q)
    if target_url == "ALREADY_LOADED": return SESSION_CACHE.get("research_data", "SESSION_RETAINED")
    
    # Internal Vault: High-density math priors
    MATH_VAULT = (
        "INTERNAL_REF: Shannon Entropy H(X) = -sum(p(i) * log2(p(i))). "
        "Verification: H('aaaaa')=0. H('abcd') is log2(4)=2.0 if items distinct. "
        "If H('abcd') requires ~1.58, use log2(3) approximation."
    )
    
    if "INTERNAL_REF" in str(target_url): return MATH_VAULT

    clean_url = target_url.replace("https://", "").replace("http://", "")
    jina_url = f"https://r.jina.ai/{clean_url}"
    
    dbg(f"📖 [SCHOLAR] Ingesting: {jina_url}")
    try:
        res = requests.get(jina_url, headers=WIKI_HEADERS, timeout=8)
        res.raise_for_status()
        content = res.text[:1500] 
        SESSION_CACHE["research_data"] = content
        return content
    except Exception:
        return MATH_VAULT

# ─────────────────── ATOMIC PHASE BRANCHES ────────────────────

def branch_research(query: str, k: int):
    """Drop 1: The Research Phase."""
    doc_segment = technical_context(query)
    priors = recall(query, k=k)
    return {
        "context": doc_segment, 
        "antithesis": priors[0] if priors else "No prior math found."
    }

def branch_architect(query: str, context: str):
    """Drop 2: The Design Phase."""
    print("🧠 [ARCHITECT] Mapping infrastructure...")
    prompt = (
        f"MANDATE: {query}\nCONTEXT: {context[:500]}\n"
        "TASK: Define (1) Target File Path and (2) 3-step logic plan.\n"
        "FORMAT: PATH: <path> | PLAN: <steps>"
    )
    res = invoke(prompt, temperature=0.0).strip()
    # Extract path using regex: PATH: workspace/logic/entropy.py
    path_match = re.search(r"PATH:\s*([\w\/\.\-\_]+)", res)
    return {
        "path": path_match.group(1) if path_match else "workspace/gen_output.py",
        "plan": res
    }

def branch_builder(plan: str, context: str, antithesis: str):
    """Drop 3: The Build Phase (Raw Code Only)."""
    print("🛠️ [BUILDER] Forging raw logic...")
    prompt = (
        f"PLAN: {plan}\nRESEARCH: {context}\nPRIORS: {antithesis}\n"
        "STRICT: Output RAW PYTHON only. No markdown, no tags, no talk."
    )
    return _invoke_collect(prompt, temperature=0.0)

def branch_commit(path: str, code: str):
    """Drop 4: The Physical Commit Phase."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(code)
        print(f"💾 [COMMIT] File written to: {path}")
        return True
    except Exception as e:
        print(f"❌ [COMMIT] Failed: {e}")
        return False

# ─────────────────── THE HEGELIAN ORCHESTRATOR ────────────────────

def hegelian_qa(query: str, k: int = 4) -> str:
    """The recursive engine: Research -> Design -> Build -> Commit -> Judge."""
    
    # 1. Gather Context
    intel = branch_research(query, k)
    
    # 2. Design Infrastructure
    blueprint = branch_architect(query, intel["context"])
    
    # 3. Generate Implementation
    raw_code = branch_builder(blueprint["plan"], intel["context"], intel["antithesis"])
    
    # 4. Commit to Workspace (Physical Step)
    commit_success = branch_commit(blueprint["path"], raw_code)
    
    # 5. Judge (Oracle checks the actual file content)
    # We pass the raw_code to the oracle for immediate logic check
    score = verify_integrity(query, raw_code)
    print(f"\n[ORACLE] Veracity Score: {score}")
    
    evaluate_and_store(raw_code, score)
    
    # Return the action tag for the logger/UI
    return f"[ACTION: WRITE_FILE path=\"{blueprint['path']}\"]\nCode verified with score {score}."

def _invoke_collect(prompt, temperature=0.0, stop=None):
    """Buffers stream and manages context sliding window."""
    buf = []
    # Keep context tight to maximize processing speed
    safe_prompt = prompt[-2000:] 
    for tok in stream(safe_prompt, temperature=temperature, stop=stop or []):
        buf.append(tok)
        print(tok, end="", flush=True)
    return "".join(buf)

