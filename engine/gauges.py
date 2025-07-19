# forge/gauges.py
from typing import Literal
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from engine.indexer import get_embedding

EMB = get_embedding()
# ───────────── Sentiment & Identity Gauges ──────────────────────
POS_SEEDS = ["great", "excellent", "love", "happy", "wonderful"]
NEG_SEEDS = ["hate", "terrible", "bad", "angry", "sad"]


def _embed(text: str) -> np.ndarray:
    return np.asarray(EMB.embed_query(text), dtype=np.float32)


# Pre-compute seed vectors once
_pos_vecs = np.stack([_embed(w) for w in POS_SEEDS])
_neg_vecs = np.stack([_embed(w) for w in NEG_SEEDS])
X_seed = np.vstack([_pos_vecs, _neg_vecs])
y_seed = np.array([1] * len(_pos_vecs) + [0] * len(_neg_vecs))

# Simple logistic-reg pipeline
_sent_clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
_sent_clf.fit(X_seed, y_seed)


def analyse_turn(text: str, speaker: Literal["user", "assistant"]) -> dict:
    """Return {'speaker':…, 'tone':…, 'valence': float}."""

    vec = _embed(text)
    prob_pos = _sent_clf.predict_proba(vec.reshape(1, -1))[0, 1]
    valence = (prob_pos * 2) - 1  # map 0..1 ➜ -1..+1
    tone = "positive" if valence > 0.3 else "negative" if valence < -0.3 else "neutral"

    return {"speaker": speaker, "tone": tone, "valence": round(valence, 3)}
