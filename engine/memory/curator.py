# engine/memory/curator.py

"""
Routing logic = ① STM → ② Heuristics → ③ LTM
Decay / boost handled in STM; promotion when weight >= STM_CUTOFF.
Fixed for clean float serialization to prevent Redis/Numpy conversion errors.
"""
import redis
import json
import time
import re
from engine.config import REDIS_URL
import engine.memory.short_term as stm
import engine.memory.long_term as ltm
import engine.memory.heuristic as heuristic
# And for our Redis 'R'
from engine.memory.short_term import R

STM_CUTOFF = 1.5

# Ensure decode_responses=True to handle string parsing
R = redis.from_url(REDIS_URL, decode_responses=True)

def safe_float(value, default=0.5) -> float:
    """
    Surgical extraction of floats from strings. 
    Handles '0.5', 'np.float64(0.46)', and None types.
    """
    if value is None:
        return default
    try:
        # Try direct conversion first
        return float(value)
    except (ValueError, TypeError):
        # Regex fallback: Find digits, optional dot, more digits
        match = re.search(r"(\d+\.\d+|\d+)", str(value))
        if match:
            return float(match.group(1))
        return default

def get_avg_confidence() -> float:
    """Helper for the Agent to sense the 'Idea Gravity' level."""
    raw = R.get("meta:avg_confidence")
    return safe_float(raw)

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
    # Ensure inputs are native Python floats to prevent Redis 'np.float' strings
    confidence = float(confidence)
    valence = float(valence)

    # 1. Similarity Check (The 'Anti-Stagnation' Guard)
    last_thought = recall("", k=1)
    if last_thought:
        from engine.agent import _semantic_similarity 
        sim = _semantic_similarity(synthesis, last_thought[0])
        if sim > 0.85:
            confidence *= (1.0 - sim)
            print(f"[CURATOR] ⚠️ High similarity ({sim:.2f}). Penalty applied.")

    # 2. Delta logic with linted float conversion
    prev_avg = get_avg_confidence()
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

    # Update moving average - Explicitly cast result to float for Redis
    new_avg = float((prev_avg * 0.7) + (confidence * 0.3))
    R.set("meta:avg_confidence", new_avg)

def store(text: str, *, valence: float = 0.0):
    valence = float(valence)
    blob = {"v": text, "weight": 1.0, "valence": valence}
    key = stm.remember(text, valence=valence)  # write STM
    heuristic.add(text, {"id": key})  # semantic route
    return key

def reinforce(key: str, delta: float = 0.3):
    frag = stm.get_frag(key)
    if not frag:
        return
    # Ensure we are doing math on a clean float
    frag["weight"] = float(frag.get("weight", 0.0)) + float(delta)
    stm.set_frag(key, frag)
    # promotion?
    if frag["weight"] >= STM_CUTOFF:
        ltm.upsert(key, frag)
