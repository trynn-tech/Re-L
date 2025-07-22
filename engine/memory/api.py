"""
Routing logic =  ① STM  → ② Heuristics → ③ LTM
Decay / boost handled in STM; promotion when weight >= STM_CUTOFF.
"""

from . import stm, ltm, heuristic
from datetime import datetime

STM_CUTOFF = 1.5


def recall(query: str, k: int = 4):
    # 1) fast approximate
    local = heuristic.search(query, k)
    if local:
        for d in local:
            stm.set_frag(d.metadata["id"], d.metadata)  # refresh TTL
        return [d.page_content for d in local]

    # 2) fall back to LTM db
    return [m["v"] for m in ltm.topk(k)]


def store(text: str, *, valence: float = 0.0):
    blob = {"v": text, "weight": 1.0, "valence": valence}
    key = stm.remember(text, valence=valence)  # write STM
    heuristic.add(text, {"id": key})  # semantic route
    return key


def reinforce(key: str, delta: float = 0.3):
    frag = stm.get_frag(key)
    if not frag:
        return
    frag["weight"] += delta
    stm.set_frag(key, frag)
    # promotion?
    if frag["weight"] >= STM_CUTOFF:
        ltm.upsert(key, frag)
