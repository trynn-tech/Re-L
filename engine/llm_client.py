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
