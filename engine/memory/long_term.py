# engine/memory/long_term.py

import sqlite3, json, contextlib, pathlib, time

DB = pathlib.Path("configs/ltm.sqlite")
# SCHEMA now includes confidence (the Critic's score)
SCHEMA = """CREATE TABLE IF NOT EXISTS memo (
    id        TEXT PRIMARY KEY,
    data      TEXT,
    weight    REAL,
    valence   REAL,
    confidence REAL,
    updated   REAL
);"""

def _cx():
    cx = sqlite3.connect(DB)
    cx.execute(SCHEMA)
    return cx

@contextlib.contextmanager
def cx():
    conn = _cx()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def upsert(id: str, blob: dict):
    """
    Saves the thought and the Critic's evaluation.
    This 'confidence' score will drive future temperature shifts.
    """
    with cx() as c:
        c.execute(
            "REPLACE INTO memo VALUES(?,?,?,?,?,?)",
            (
                id, 
                json.dumps(blob), 
                blob.get("weight", 1.0), 
                blob.get("valence", 0.0),
                blob.get("confidence", 0.5),
                time.time()
            ),
        )

def get_forbidden_centroid(limit: int = 5):
    """
    Retrieves the most recent failures (Critique < 0.3).
    Used in Agent logic to push the search away from failed paths.
    """
    with cx() as c:
        rows = c.execute(
            "SELECT data FROM memo WHERE confidence < 0.3 ORDER BY updated DESC LIMIT ?", 
            (limit,)
        ).fetchall()
        return [json.loads(r[0]).get('v') for r in rows]

def get_global_avg_confidence():
    """Calculates the current 'State of the Empire'."""
    with cx() as c:
        row = c.execute("SELECT AVG(confidence) FROM memo").fetchone()
        return row[0] if row[0] is not None else 0.5

def topk(limit: int = 100):
    with cx() as c:
        return [
            json.loads(r[0])
            for r in c.execute(
                "SELECT data FROM memo ORDER BY weight DESC, confidence DESC LIMIT ?", (limit,)
            )
        ]
