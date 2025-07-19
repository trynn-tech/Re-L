# forge/config.py
from pathlib import Path

# --- Paths -------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT.parent / "docs"
IDENTITY_PATH = ROOT.parent / "configs" / "identity.yaml"
INDEX_PATH = ROOT / "faiss_index.pkl"
PROOF_PATH = ROOT / "proof_index.pkl"
MEM_PATH = ROOT / "decay_mem.json"

# --- Embedding / model names -------------------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "~/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"


# --- Retrieval thresholds ---------------------------------------
SIM_HIGH = 0.85
SIM_LOW = 0.10
