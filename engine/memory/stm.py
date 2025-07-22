"""
STM  – fast key‑value store on Redis
Weight‑aware TTL acts as the forget gate.
"""

import os, json, time, redis
from datetime import datetime
from engine.config import REDIS_URL


REDIS_PW = os.getenv("REDIS_PASSWORD")
R = redis.from_url(REDIS_URL, decode_responses=True)


BASE_TTL = 7 * 24 * 3600  # one week in seconds
TTL_CAP = 5  # weight above which TTL no longer grows


# ──────────────────────────────────────────────────────────────────────────
def _ttl_for_weight(w: float) -> int:
    """
    Map current weight to TTL.
    • Linear up to TTL_CAP, then clipped.
    """
    return int(BASE_TTL * min(w, TTL_CAP))


def set_frag(key: str, blob: dict) -> None:
    """Create / overwrite key with TTL derived from blob['weight']."""
    ttl = _ttl_for_weight(blob.get("weight", 1.0))
    R.setex(key, ttl, json.dumps(blob))


def get_frag(key: str) -> dict | None:
    raw = R.get(key)
    return json.loads(raw) if raw else None


def remember(text: str, *, valence: float = 0.0) -> str:
    """Insert new fragment – initial weight=1.0, full TTL."""
    key = f"m:{int(time.time()*1000)}"
    blob = {
        "v": text,
        "weight": 1.0,
        "valence": valence,
        "last_used": datetime.utcnow().isoformat(),
    }
    set_frag(key, blob)
    return key


def find_recent(n: int = 20) -> list[dict]:
    """
    Redis KEYS order is undefined; this helper is best‑effort.
    """
    keys = list(R.scan_iter(match="m:*", count=n * 2))[:n]
    return [json.loads(R.get(k)) for k in keys]
