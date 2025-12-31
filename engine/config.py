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

REDIS_URL = f"redis://:{_pw}@re-l-redis:6379/0"

# --- Paths -------------------------------------------------------
VAULT_DIR = ROOT / "data" / "vault"
INDEX_DIR = ROOT / "data" / "indices"
DOCS_DIR = ROOT / "docs"/ "vault"
IDENTITY_PATH = ROOT / "configs" / "identity.yaml"
INDEX_PATH = ROOT / "configs" / "faiss_index.pkl"
PROOF_PATH = ROOT / "configs" / "proof_index.pkl"
MEM_PATH = ROOT / "configs" / "decay_mem.json"

# --- Embedding / model names -------------------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = ROOT / "models" / "mistral-7b-instruct-v0.1.Q8_0.gguf"


# --- Retrieval thresholds ---------------------------------------
SIM_HIGH = 0.85
SIM_LOW = 0.10

# --- Model Bounds ---------------------------------------
GRAVITY_THRESHOLD = 0.85  # Confidence level that triggers divergence
MAX_THOUGHT_HISTORY = 3   # How many thought vectors to keep for repulsion
STAGNATION_RECOVERY_TEMP = 0.85
STABLE_EXPLORATION_TEMP = 0.45
STOP_SEQUENCES = ["###", "IDENTITY:", "THESIS:", "REFLECTION >", "Confidence:"]
