# engine/memory/curator.py

"""
Routing logic =  ① STM  → ② Heuristics → ③ LTM
Decay / boost handled in STM; promotion when weight >= STM_CUTOFF.
"""
import redis, json, time
from engine.config import REDIS_URL
import engine.memory.short_term as stm
import engine.memory.long_term as ltm
import engine.memory.heuristic as heuristic
# And for our Redis 'R' and avg_confidence helper:
from engine.memory.short_term import R

STM_CUTOFF = 1.5

R = redis.from_url(REDIS_URL, decode_responses=True)

def get_avg_confidence() -> float:
    """Helper for the Agent to sense the 'Idea Gravity' level."""
    raw = R.get("meta:avg_confidence")

    return float(raw) if raw else 0.5

def recall(query: str, k: int = 1):
    internal_state = []
    local = heuristic.search(query, k)
    if local:
        internal_state.extend([d.page_content for d in local])
    if len(internal_state) < k:
        internal_state.extend([m["v"] for m in ltm.topk(k - len(internal_state))])
    return internal_state

def evaluate_and_store(synthesis: str, confidence: float, valence: float = 0.0):
    """The Crystallization Gate with Repetition Penalty."""
    # 1. Similarity Check (The 'Anti-Stagnation' Guard)
    last_thought = recall("", k=1)
    if last_thought:
        # We use the internal helper to see how similar this is to our last LTM
        from engine.agent import _semantic_similarity 
        sim = _semantic_similarity(synthesis, last_thought[0])
        if sim > 0.85:
            confidence *= (1.0 - sim) # If sim is 0.9, confidence is crushed by 90%
            print(f"[CURATOR] ⚠️ High similarity ({sim:.2f}). Penalty applied to breakthrough score.")

    # 2. Delta logic remains the same...
    raw_avg = R.get("meta:avg_confidence")
    prev_avg = float(raw_avg) if raw_avg else 0.5
    delta = confidence - prev_avg

    if delta > 0.15:
        # Crystallize to LTM
        key = stm.remember(synthesis, valence=valence)
        ltm.upsert(key, stm.get_frag(key)) 
        print(f"[CURATOR] 🚀 Evolution! Delta +{delta:.2f}. New structure crystallized.")
    else:
        # Standard STM
        stm.remember(synthesis, valence=valence)
        print(f"[CURATOR] ☁️ Iteration. Delta {delta:.2f}. Volatile store.")

    # Update moving average
    R.set("meta:avg_confidence", (prev_avg * 0.7) + (confidence * 0.3))

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
